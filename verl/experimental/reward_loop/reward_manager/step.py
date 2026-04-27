"""
Step Reward Manager for Step-GDPO and non-parametric process rewards.
TODO: support parametric process reward models

Computes both outcome reward (answer correctness) and step-level process rewards
(format / random / fol). Process rewards are passed as per-step (position, score)
lists via reward_extra_info so the advantage estimator can reconstruct token-level
tensors and apply big-pool normalization.
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


@register("step")
class StepRewardManager(RewardManagerBase):
    """
    Step Reward Manager for Step-GDPO and non-parametric process rewards.
    TODO: support parametric process reward models

    Computes:
    - outcome_reward: scalar correctness score (placed at last valid response token)
    - process_reward: per-step scores via configurable step_reward_types

    The step-level process rewards are serialized as lists of (position, score) tuples
    in reward_extra_info, keyed by "{type}_step_reward". The advantage estimator
    (step_gdpo) reads these to build token-level tensors and perform big-pool normalization.
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
        # Priority: reward config > algorithm config > defaults
        reward_cfg = config.get("reward", {})
        algo_cfg = config.get("algorithm", {})
        max_tries = reward_cfg.get("fol_max_tries", algo_cfg.get("fol_max_tries", None))
        if max_tries is not None:
            self.api_config["max_tries"] = int(max_tries)
        old_max_tries = reward_cfg.get("fol_old_max_tries", algo_cfg.get("fol_old_max_tries", None))
        if old_max_tries is not None:
            self.api_config["old_max_tries"] = int(old_max_tries)
        z3_timeout = reward_cfg.get("fol_timeout", algo_cfg.get("fol_timeout", None))
        if z3_timeout is not None:
            self.api_config["timeout"] = int(z3_timeout)
        api_timeout = reward_cfg.get("api_timeout", algo_cfg.get("api_timeout", None))
        if api_timeout is not None:
            self.api_config["api_timeout"] = int(api_timeout)
        cumulative = reward_cfg.get("fol_verify_with_cumulative_steps", algo_cfg.get("fol_verify_with_cumulative_steps", False))
        self.api_config["cumulative"] = bool(cumulative)
        print(f"FOL config 'fol_verify_with_cumulative_steps' is set to: {self.api_config['cumulative']}")

        # FOL pipeline / translation mode
        fol_preprocess = reward_cfg.get("fol_preprocess", algo_cfg.get("fol_preprocess", None))
        if fol_preprocess is not None:
            self.api_config["fol_preprocess"] = str(fol_preprocess)
        fol_translation = reward_cfg.get("fol_translation", algo_cfg.get("fol_translation", None))
        if fol_translation is not None:
            self.api_config["fol_translation"] = str(fol_translation)
        fol_judge_use_outlines = reward_cfg.get(
            "fol_judge_use_outlines",
            algo_cfg.get("fol_judge_use_outlines", False),
        )
        self.api_config["fol_judge_use_outlines"] = bool(fol_judge_use_outlines)

        # Step reward type: explicit parameter > reward config > algorithm config > default "random"
        if step_reward_type is not None: # explicit parameter only exists @ unit tests
            if isinstance(step_reward_type, str):
                self.step_reward_types = [step_reward_type]
            else:
                self.step_reward_types = list(step_reward_type)
        else: # all training scripts follow this branch
            reward_cfg = config.get("reward", {})
            algo_cfg = config.get("algorithm", {})
            
            srt = reward_cfg.get("step_reward_type", None)
            if srt is None:
                srt = algo_cfg.get("step_reward_type", None)
            if srt is None:
                raise ValueError("step_reward_type is not specified")
                
            if isinstance(srt, str):
                self.step_reward_types = [srt]
            else:
                self.step_reward_types = list(srt)
                
        # Initialize pluggable reward functions registry
        self.step_reward_fns = {
            "random": _compute_step_reward_random
        }
        
        # Built-in extra reward types if requested (Lazy loading)
        if any(rt in ["fol", "fol_old", "format"] for rt in self.step_reward_types):
            try:
                from verl.utils.reward_score.fol import compute_step_reward_format_fol, compute_step_reward_fol
                if "format" not in self.step_reward_fns:
                    self.step_reward_fns["format"] = compute_step_reward_format_fol
                if "fol" not in self.step_reward_fns:
                    self.step_reward_fns["fol"] = compute_step_reward_fol
            except ImportError as e:
                logging.getLogger(__name__).warning("Failed to lazily load built-in FOL reward functions: %s", e)
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
        # When the model exploits process reward by inflating steps, breaking format,
        # or hitting max_response_length, zero out all process rewards for that response.
        # Configurable via algorithm or reward config.
        self.penalty_max_steps = int(
            reward_cfg.get("penalty_max_steps", algo_cfg.get("penalty_max_steps", 0))
        )  # 0 = disabled; e.g. 12 to penalize responses with >12 steps
        self.penalty_on_truncated = bool(
            reward_cfg.get("penalty_on_truncated", algo_cfg.get("penalty_on_truncated", False))
        )  # penalize responses that hit max_response_length
        self.penalty_on_multi_boxed = bool(
            reward_cfg.get("penalty_on_multi_boxed", algo_cfg.get("penalty_on_multi_boxed", False))
        )  # penalize responses with multiple \boxed{}
        self.penalty_on_bad_format = bool(
            reward_cfg.get("penalty_on_bad_format", algo_cfg.get("penalty_on_bad_format", False))
        )  # penalize mismatched <step>/<conclusion> tags
        self.penalty_score = float(
            reward_cfg.get("penalty_score", algo_cfg.get("penalty_score", 0.0))
        )  # score to assign when penalized (default 0.0, can be negative)

        # LLM-backed step rewards are token-budget-bound long before 64 threads
        # are useful. Keep the default conservative, and let scripts opt in to
        # more parallelism via config/env when the upstream API budget allows it.
        max_workers = reward_cfg.get(
            "step_reward_max_workers",
            algo_cfg.get("step_reward_max_workers", os.environ.get("VERL_STEP_REWARD_MAX_WORKERS")),
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
            response_ids=valid_response_ids,
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
                    "fol_verifier_steps": 0,
                    "fol_entailed_steps": 0,
                    "fol_not_entailed_steps": 0,
                    "fol_invalid_translation_steps": 0,
                    "fol_invalid_expression_steps": 0,
                    "fol_expression_repair_steps": 0,
                    "fol_leakage_steps": 0,
                    "fol_student_duplicate_steps": 0,
                    "fol_declaration_failed_steps": 0,
                    "fol_format_failed_steps": 0,
                }
            )
        return reward_extra_info

    async def run_single(self, data: DataProto) -> dict:
        """Compute outcome + process rewards for a single data item."""
        assert len(data) == 1, "StepRewardManager only supports single data item"
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

        # 2. Compute step-level process rewards
        # 2.1 Splitting | get token positions (end pos) of each step for assigning rewards
        step_positions = self._get_step_token_positions(response_str, valid_response_ids, valid_response_length)
        reward_extra_info["num_steps"] = len(step_positions)

        # --- Anti-reward-hacking precheck ---
        # If a response already matches a configured penalty condition, skip
        # expensive process-reward calls and directly assign the penalty score.
        penalize = False
        penalty_reason = []

        num_steps = len(step_positions)

        # 1. Too many steps
        if self.penalty_max_steps > 0 and num_steps > self.penalty_max_steps:
            penalize = True
            penalty_reason.append(f"num_steps={num_steps}>{self.penalty_max_steps}")

        # 2. Response truncated (hit max_response_length)
        if self.penalty_on_truncated and valid_response_length >= response_length:
            penalize = True
            penalty_reason.append("truncated")

        # 3. Multiple \boxed{} in response
        if self.penalty_on_multi_boxed:
            import re as _re

            boxed_count = len(_re.findall(r'\\boxed\{', response_str))
            if boxed_count > 1:
                penalize = True
                penalty_reason.append(f"multi_boxed={boxed_count}")

        # 4. Bad format: mismatched <step>/<conclusion> tags
        if self.penalty_on_bad_format:
            step_open = response_str.count("<step>")
            step_close = response_str.count("</step>")
            has_conclusion_outside_step = False
            # Simple heuristic: <conclusion> after the last </step>
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

        # 2.2 Extract prompt text for reward functions that need it
        raw_prompt = data_item.non_tensor_batch.get("raw_prompt", [])
        if raw_prompt is not None and len(raw_prompt) > 0:
            # raw_prompt is a list of message dicts; take the last user message content
            prompt_text = raw_prompt[-1]["content"] if isinstance(raw_prompt[-1], dict) else str(raw_prompt[-1])
        else:
            prompt_text = ""

        # 2.3 Compute process rewards for each step_reward_type (parallel)
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
                        **({"return_debug": True} if reward_type == "fol" else {}),
                        **({"fol_shared_state": fol_shared_state} if reward_type == "fol" else {}),
                    ),
                )
                for args in call_args
            ]
            scores = await asyncio.gather(*futures)

            step_rewards = []
            fol_debug = []
            fol_judge_prompt_tokens = 0
            fol_judge_completion_tokens = 0
            fol_judge_total_tokens = 0
            fol_judge_calls = 0
            fol_verifier_steps = 0
            fol_entailed_steps = 0
            fol_not_entailed_steps = 0
            fol_invalid_translation_steps = 0
            fol_invalid_expression_steps = 0
            fol_expression_repair_steps = 0
            fol_leakage_steps = 0
            fol_student_duplicate_steps = 0
            fol_declaration_failed_steps = 0
            fol_format_failed_steps = 0
            for args, score_item in zip(call_args, scores):
                if reward_type == "fol" and isinstance(score_item, dict):
                    score_value = float(score_item.get("score", 0.0))
                    step_debug = score_item.get("debug", {})
                    fol_debug.append(step_debug)
                    fol_verifier_steps += 1
                    if isinstance(step_debug, dict):
                        judge_usage = step_debug.get("judge_usage", {})
                        if isinstance(judge_usage, dict):
                            fol_judge_prompt_tokens += int(judge_usage.get("prompt_tokens", 0) or 0)
                            fol_judge_completion_tokens += int(judge_usage.get("completion_tokens", 0) or 0)
                            fol_judge_total_tokens += int(judge_usage.get("total_tokens", 0) or 0)
                            fol_judge_calls += int(judge_usage.get("calls", 0) or 0)
                        z3_output = str(step_debug.get("z3_output", "") or "")
                        if "SUCCESS_ENTAILED" in z3_output:
                            fol_entailed_steps += 1
                        if "FAILED_NOT_ENTAILED" in z3_output:
                            fol_not_entailed_steps += 1
                        if step_debug.get("translation_failed_closed") or "FAILED_INVALID_TRANSLATION" in z3_output:
                            fol_invalid_translation_steps += 1
                        if (
                            step_debug.get("invalid_expression_syntax")
                            or step_debug.get("invalid_expression_syntax_initial")
                            or "FAILED_INVALID_EXPRESSION" in z3_output
                        ):
                            fol_invalid_expression_steps += 1
                        if int(step_debug.get("expression_correction_attempts", 0) or 0) > 0:
                            fol_expression_repair_steps += 1
                        if step_debug.get("conclusion_leakage_detected") or "FAILED_LEAKED_CONCLUSION" in z3_output:
                            fol_leakage_steps += 1
                        if step_debug.get("student_premise_conclusion_duplicate"):
                            fol_student_duplicate_steps += 1
                        if step_debug.get("declaration_failed_closed"):
                            fol_declaration_failed_steps += 1
                        if step_debug.get("format_failed_closed"):
                            fol_format_failed_steps += 1
                else:
                    score_value = float(score_item)
                step_rewards.append((int(args[3]), score_value))

            key = f"{reward_type}_step_reward"
            reward_extra_info[key] = step_rewards
            if reward_type == "fol" and fol_debug:
                reward_extra_info["fol_debug"] = fol_debug
                reward_extra_info["fol_judge_prompt_tokens"] = fol_judge_prompt_tokens
                reward_extra_info["fol_judge_completion_tokens"] = fol_judge_completion_tokens
                reward_extra_info["fol_judge_total_tokens"] = fol_judge_total_tokens
                reward_extra_info["fol_judge_calls"] = fol_judge_calls
                reward_extra_info["fol_judge_completion_tokens_per_call"] = (
                    float(fol_judge_completion_tokens) / fol_judge_calls if fol_judge_calls > 0 else 0.0
                )
                reward_extra_info["fol_verifier_steps"] = fol_verifier_steps
                reward_extra_info["fol_entailed_steps"] = fol_entailed_steps
                reward_extra_info["fol_not_entailed_steps"] = fol_not_entailed_steps
                reward_extra_info["fol_invalid_translation_steps"] = fol_invalid_translation_steps
                reward_extra_info["fol_invalid_expression_steps"] = fol_invalid_expression_steps
                reward_extra_info["fol_expression_repair_steps"] = fol_expression_repair_steps
                reward_extra_info["fol_leakage_steps"] = fol_leakage_steps
                reward_extra_info["fol_student_duplicate_steps"] = fol_student_duplicate_steps
                reward_extra_info["fol_declaration_failed_steps"] = fol_declaration_failed_steps
                reward_extra_info["fol_format_failed_steps"] = fol_format_failed_steps

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
