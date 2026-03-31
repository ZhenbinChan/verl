"""
Step Reward Manager for Step-GDPO and non-parametric process rewards.
TODO: support parametric process reward models

Computes both outcome reward (answer correctness) and step-level process rewards
(format / random / fol). Process rewards are passed as per-step (position, score)
lists via reward_extra_info so the advantage estimator can reconstruct token-level
tensors and apply big-pool normalization.
"""

import inspect
import logging
import os
import random
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
        # Process Reward
        # COMMONLY USED FUNCTIONS GO HERE
        if any(rt in ["fol", "format"] for rt in self.step_reward_types):
            try:
                from verl.utils.reward_score.fol import compute_step_reward_format_fol, compute_step_reward_fol
                if "format" not in self.step_reward_fns:
                    self.step_reward_fns["format"] = compute_step_reward_format_fol
                if "fol" not in self.step_reward_fns:
                    self.step_reward_fns["fol"] = compute_step_reward_fol
            except ImportError as e:
                logging.getLogger(__name__).warning("Failed to lazily load built-in FOL reward functions: %s", e)

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

        reward_extra_info = {}
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        # 2. Compute step-level process rewards
        # 2.1 Splitting | get token positions (end pos) of each step for assigning rewards
        step_positions = self._get_step_token_positions(response_str, valid_response_ids, valid_response_length)

        # 2.2 Extract prompt text for reward functions that need it
        raw_prompt = data_item.non_tensor_batch.get("raw_prompt", [])
        if raw_prompt is not None and len(raw_prompt) > 0:
            # raw_prompt is a list of message dicts; take the last user message content
            prompt_text = raw_prompt[-1]["content"] if isinstance(raw_prompt[-1], dict) else str(raw_prompt[-1])
        else:
            prompt_text = ""

        # 2.3 Compute process rewards for each step_reward_type
        for reward_type in self.step_reward_types:
            # NOTE: Process reward fn
            reward_fn = self.step_reward_fns.get(reward_type)
            if reward_fn is None:
                raise ValueError(f"Unknown step reward type: {reward_type}")
            step_rewards = []  # list of (token_position, score)
            step_history = []
            for step_text, token_end_pos in step_positions:
                step_history.append(step_text)
                step_score = reward_fn(
                    step_text,
                    prompt_text,
                    step_history,
                    api_config=self.api_config,
                    extra_info=extra_info,
                )
                step_rewards.append((int(token_end_pos), float(step_score)))

            # Store as serializable list of (pos, score) tuples
            key = f"{reward_type}_step_reward"
            reward_extra_info[key] = step_rewards

        # Store number of steps for debugging
        reward_extra_info["num_steps"] = len(step_positions)

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
