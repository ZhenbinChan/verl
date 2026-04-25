# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tree Reward Manager for TreeRL integration.

This reward manager is designed for TreeRL's EPTree sampling mode. It computes:
1. Outcome reward (answer correctness) — used by TreeManager for tree backpropagation
2. Optional external step-level process rewards (format, fol, random, etc.)
   — same plugin mechanism as StepRewardManager

The tree-derived step rewards (GA + LA) / sqrt(n) are computed externally by
TreeManager and stored in non_tensor_batch['treerl_step_reward'].
External PRM scores are stored as '{type}_step_reward' in reward_extra_info.
"""

import asyncio
import inspect
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score
from verl.utils.step_splitter import (
    default_split_fn,
    get_step_token_positions,
    split_by_xml_step_tags,
    split_response_into_steps,
)


def _compute_step_reward_random(step_text: str, prompt_text: str, step_history: list[str], **kwargs) -> float:
    """Random baseline process reward."""
    return float(random.randint(0, 1))


@register("tree")
class TreeRewardManager(RewardManagerBase):
    """Tree Reward Manager for TreeRL.

    Computes:
    - outcome_reward: scalar correctness score (used by TreeManager backprop)
    - process_reward (optional): per-step scores via configurable step_reward_types,
      same plugin mechanism as StepRewardManager

    The tree-topology process supervision (GA+LA)/sqrt(n) is handled externally
    by TreeManager and stored in non_tensor_batch['treerl_step_reward'].
    External PRM scores complement the tree-derived rewards.
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
        split_fn: Optional[Callable[[str], list[str]]] = None,
        step_reward_type: Optional[str | list[str]] = None,
        step_reward_fns: Optional[dict] = None,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        # Pluggable step splitter
        self.split_fn = split_fn or default_split_fn

        # API configuration for LLM-based step rewards (FOL, self_eval, etc.)
        # Priority: config.reward.api_config > env vars > defaults
        self.api_config = {
            "model": os.environ.get("SELF_EVAL_MODEL", os.environ.get("FOL_MODEL")),
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": os.environ.get("OPENAI_BASE_URL"),
            "temperature": 0.6,
            "max_tokens": 1024,
        }
        cfg_override = config.get("reward", {}).get("api_config",
                       config.get("reward", {}).get("fol_api_config", {}))
        if cfg_override:
            self.api_config.update({k: v for k, v in cfg_override.items() if v is not None})

        # FOL configuration: correct_loop max retries, timeout, pipeline settings
        reward_cfg_fol = config.get("reward", {})
        algo_cfg_fol = config.get("algorithm", {})
        max_tries = reward_cfg_fol.get("fol_max_tries", algo_cfg_fol.get("fol_max_tries", None))
        if max_tries is not None:
            self.api_config["max_tries"] = int(max_tries)
        z3_timeout = reward_cfg_fol.get("fol_timeout", algo_cfg_fol.get("fol_timeout", None))
        if z3_timeout is not None:
            self.api_config["timeout"] = int(z3_timeout)
        api_timeout = reward_cfg_fol.get("api_timeout", algo_cfg_fol.get("api_timeout", None))
        if api_timeout is not None:
            self.api_config["api_timeout"] = int(api_timeout)
        cumulative = reward_cfg_fol.get("fol_verify_with_cumulative_steps", algo_cfg_fol.get("fol_verify_with_cumulative_steps", False))
        self.api_config["cumulative"] = bool(cumulative)
        print(f"FOL config 'fol_verify_with_cumulative_steps' is set to: {self.api_config['cumulative']}")

        # FOL pipeline / translation mode
        fol_preprocess = reward_cfg_fol.get("fol_preprocess", algo_cfg_fol.get("fol_preprocess", None))
        if fol_preprocess is not None:
            self.api_config["fol_preprocess"] = str(fol_preprocess)
        fol_translation = reward_cfg_fol.get("fol_translation", algo_cfg_fol.get("fol_translation", None))
        if fol_translation is not None:
            self.api_config["fol_translation"] = str(fol_translation)
        fol_judge_use_outlines = reward_cfg_fol.get(
            "fol_judge_use_outlines",
            algo_cfg_fol.get("fol_judge_use_outlines", False),
        )
        self.api_config["fol_judge_use_outlines"] = bool(fol_judge_use_outlines)

        # Step reward type: explicit parameter > reward config > algorithm config > None
        if step_reward_type is not None:
            if isinstance(step_reward_type, str):
                self.step_reward_types = [step_reward_type]
            else:
                self.step_reward_types = list(step_reward_type)
        else:
            reward_cfg = config.get("reward", {})
            algo_cfg = config.get("algorithm", {})

            srt = reward_cfg.get("step_reward_type", None)
            if srt is None:
                srt = algo_cfg.get("step_reward_type", None)

            if srt is None:
                # No external PRM configured — tree reward only
                self.step_reward_types = []
            elif isinstance(srt, str):
                self.step_reward_types = [srt]
            else:
                self.step_reward_types = list(srt)

        # Initialize pluggable reward functions registry
        self.step_reward_fns = {
            "random": _compute_step_reward_random,
        }

        # Lazy-load built-in extra reward types
        if any(rt in ["fol", "fol_old", "format"] for rt in self.step_reward_types):
            try:
                from verl.utils.reward_score.fol import compute_step_reward_format_fol, compute_step_reward_fol
                if "format" not in self.step_reward_fns:
                    self.step_reward_fns["format"] = compute_step_reward_format_fol
                if "fol" not in self.step_reward_fns:
                    self.step_reward_fns["fol"] = compute_step_reward_fol
            except ImportError as e:
                logger.warning("Failed to lazily load built-in FOL reward functions: %s", e)
            try:
                from verl.utils.reward_score.fol_old import compute_step_reward_fol as compute_step_reward_fol_old
                if "fol_old" not in self.step_reward_fns:
                    self.step_reward_fns["fol_old"] = compute_step_reward_fol_old
            except ImportError:
                pass

        if "self_eval" in self.step_reward_types:
            from verl.utils.reward_score.self_eval import compute_step_reward_self_eval
            if "self_eval" not in self.step_reward_fns:
                self.step_reward_fns["self_eval"] = compute_step_reward_self_eval

        # Override with any user-provided step_reward_fns
        if step_reward_fns:
            self.step_reward_fns.update(step_reward_fns)

        # Resolve use_xml_steps: reward config > algorithm config > False
        reward_cfg = config.get("reward", {})
        algo_cfg = config.get("algorithm", {})
        use_xml_cfg = reward_cfg.get("use_xml_steps", None)
        if use_xml_cfg is None:
            use_xml_cfg = algo_cfg.get("use_xml_steps", None)
        self.use_xml = bool(use_xml_cfg) if use_xml_cfg is not None else False

        # --- Anti-reward-hacking penalty config ---
        reward_cfg = config.get("reward", {})
        algo_cfg = config.get("algorithm", {})
        self.penalty_max_steps = int(
            reward_cfg.get("penalty_max_steps", algo_cfg.get("penalty_max_steps", 0))
        )
        self.penalty_on_truncated = bool(
            reward_cfg.get("penalty_on_truncated", algo_cfg.get("penalty_on_truncated", False))
        )
        self.penalty_on_multi_boxed = bool(
            reward_cfg.get("penalty_on_multi_boxed", algo_cfg.get("penalty_on_multi_boxed", False))
        )
        self.penalty_on_bad_format = bool(
            reward_cfg.get("penalty_on_bad_format", algo_cfg.get("penalty_on_bad_format", False))
        )
        self.penalty_score = float(
            reward_cfg.get("penalty_score", algo_cfg.get("penalty_score", 0.0))
        )

        max_workers = reward_cfg_fol.get(
            "step_reward_max_workers",
            algo_cfg_fol.get("step_reward_max_workers", os.environ.get("VERL_STEP_REWARD_MAX_WORKERS")),
        )
        if max_workers is None:
            uses_llm_step_reward = any(rt in {"fol", "self_eval"} for rt in self.step_reward_types)
            max_workers = 4 if uses_llm_step_reward else min(16, os.cpu_count() or 4)
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))

    def _get_step_token_positions(self, response_text: str, valid_response_ids, valid_response_length: int):
        """Map character-level step boundaries to token positions.

        Delegates to the shared ``get_step_token_positions`` utility.

        Returns:
            List of (step_text, token_end_pos) where token_end_pos is the
            index of the last token in this step (within response_ids).
        """
        return get_step_token_positions(
            response_text=response_text,
            valid_response_length=valid_response_length,
            tokenizer=self.tokenizer,
            use_xml=self.use_xml,
            split_fn=self.split_fn,
        )

    def _init_reward_extra_info(self, score) -> dict:
        """Initialize a stable reward_extra_info schema for this manager."""
        reward_extra_info = {
            "acc": score,
            "num_steps": 0,
            "process_reward_penalized": False,
            "penalty_reason": "",
        }
        for reward_type in self.step_reward_types:
            reward_extra_info[f"{reward_type}_step_reward"] = []
        if "fol" in self.step_reward_types:
            reward_extra_info.update(
                {
                    "fol_debug": [],
                    "fol_judge_prompt_tokens": 0,
                    "fol_judge_completion_tokens": 0,
                    "fol_judge_total_tokens": 0,
                    "fol_judge_calls": 0,
                    "fol_judge_completion_tokens_per_call": 0.0,
                }
            )
        return reward_extra_info

    async def run_single(self, data: DataProto) -> dict:
        """Compute outcome + optional process rewards for a single data item.

        TreeRL step rewards are computed externally by TreeManager;
        this method computes outcome score and optional external PRM scores.
        """
        assert len(data) == 1, "TreeRewardManager only supports single data item"
        data_item = data[0]

        # Extract response
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]

        # Extract metadata
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        # Decode response
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # 1. Compute outcome reward
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        if isinstance(result, dict):
            score = result["score"]
        else:
            score = result

        reward_extra_info = self._init_reward_extra_info(score)
        if isinstance(result, dict):
            reward_extra_info.update(result)

        # 2. Compute step-level process rewards (external PRM)
        if self.step_reward_types:
            step_positions = self._get_step_token_positions(
                response_str, valid_response_ids, valid_response_length
            )
            reward_extra_info["num_steps"] = len(step_positions)

            # --- Anti-reward-hacking precheck ---
            # If a response already matches a configured penalty condition, skip
            # expensive external-PRM calls and directly assign the penalty score.
            penalize = False
            penalty_reason = []

            num_steps = len(step_positions)

            if self.penalty_max_steps > 0 and num_steps > self.penalty_max_steps:
                penalize = True
                penalty_reason.append(f"num_steps={num_steps}>{self.penalty_max_steps}")

            if self.penalty_on_truncated and valid_response_length >= response_length:
                penalize = True
                penalty_reason.append("truncated")

            if self.penalty_on_multi_boxed:
                import re as _re

                boxed_count = len(_re.findall(r'\\boxed\{', response_str))
                if boxed_count > 1:
                    penalize = True
                    penalty_reason.append(f"multi_boxed={boxed_count}")

            if self.penalty_on_bad_format:
                step_open = response_str.count("<step>")
                step_close = response_str.count("</step>")
                has_conclusion_outside_step = False
                last_step_close = response_str.rfind("</step>")
                last_conclusion = response_str.rfind("<conclusion>")
                if last_conclusion > last_step_close and last_step_close != -1:
                    has_conclusion_outside_step = True
                if step_open != step_close or has_conclusion_outside_step:
                    penalize = True
                    penalty_reason.append(
                        f"bad_format(open={step_open},close={step_close},conclusion_outside={has_conclusion_outside_step})"
                    )

            if penalize:
                penalty_val = self.penalty_score
                penalty_rewards = [(int(pos), penalty_val) for _, pos in step_positions]
                for reward_type in self.step_reward_types:
                    reward_extra_info[f"{reward_type}_step_reward"] = penalty_rewards
                reward_extra_info["process_reward_penalized"] = True
                reward_extra_info["penalty_reason"] = "|".join(penalty_reason)
                return {"reward_score": score, "reward_extra_info": reward_extra_info}

            # Extract prompt text for reward functions that need it
            raw_prompt = data_item.non_tensor_batch.get("raw_prompt", [])
            if len(raw_prompt) > 0:
                prompt_text = raw_prompt[-1]["content"] if isinstance(raw_prompt[-1], dict) else str(raw_prompt[-1])
            else:
                prompt_text = ""

            for reward_type in self.step_reward_types:
                reward_fn = self.step_reward_fns.get(reward_type)
                if reward_fn is None:
                    raise ValueError(f"Unknown step reward type: {reward_type}")

                fol_shared_state = None
                if reward_type == "fol":
                    from verl.utils.reward_score.fol import prepare_fol_shared_state

                    loop = asyncio.get_event_loop()
                    fol_shared_state = await loop.run_in_executor(
                        self._executor,
                        lambda: prepare_fol_shared_state(
                            prompt_text,
                            api_config=self.api_config,
                            extra_info=extra_info,
                        ),
                    )

                # Pre-build all step histories so calls can run in parallel
                # (same pattern as StepRewardManager). For LLM-based reward
                # functions (self_eval / fol) this is the difference between
                # N sequential blocking calls and ceil(N/16) concurrent ones.
                call_args = []
                for i, (step_text, token_end_pos) in enumerate(step_positions):
                    history = [s for s, _ in step_positions[: i + 1]]
                    call_args.append((step_text, prompt_text, history, token_end_pos))

                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(
                        self._executor,
                        lambda args=args: reward_fn(
                            args[0], args[1], args[2],
                            api_config=self.api_config, extra_info=extra_info,
                            **({"fol_shared_state": fol_shared_state} if reward_type == "fol" else {}),
                        ),
                    )
                    for args in call_args
                ]
                scores = await asyncio.gather(*futures)

                step_rewards = [
                    (int(args[3]), float(score))
                    for args, score in zip(call_args, scores)
                ]

                key = f"{reward_type}_step_reward"
                reward_extra_info[key] = step_rewards

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
