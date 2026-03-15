"""
Step Reward Manager for Step-GDPO.

Computes both outcome reward (answer correctness) and step-level process rewards
(format / random / fol). Process rewards are passed as per-step (position, score)
lists via reward_extra_info so the advantage estimator can reconstruct token-level
tensors and apply big-pool normalization.
"""

import inspect
import random
import re
from typing import Callable, Optional

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


def default_split_fn(response_text: str) -> list[str]:
    """Default step splitter: split by double newline."""
    if not response_text:
        return [""]
    return response_text.split("\n\n")


def _compute_step_reward_random(step_text: str, prompt_text: str, step_history: list[str]) -> float:
    """Random baseline process reward."""
    return float(random.randint(0, 1))


def _compute_step_reward_format(step_text: str, prompt_text: str, step_history: list[str]) -> float:
    """Format-check process reward (e.g., must contain action tags)."""
    score = 0.0
    if re.search(r"<Action>.*</Action>", step_text, re.DOTALL):
        score += 1.0
    return score


# Registry of step reward scoring functions
STEP_REWARD_FN_REGISTRY = {
    "random": _compute_step_reward_random,
    "format": _compute_step_reward_format,
    "fol": _compute_step_reward_random,  # TODO: FOL not implemented yet, fallback to random
}


@register("step")
class StepRewardManager(RewardManagerBase):
    """Step Reward Manager for Step-GDPO.

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
    ):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        # Pluggable step splitter
        self.split_fn = split_fn or default_split_fn

        # Read step reward config
        reward_cfg = config.reward if hasattr(config, "reward") else config
        self.step_reward_types = []
        step_reward_type = reward_cfg.get("step_reward_type", "random") if hasattr(reward_cfg, "get") else "random"
        if isinstance(step_reward_type, str):
            self.step_reward_types = [step_reward_type]
        else:
            self.step_reward_types = list(step_reward_type)

    def _split_response_into_steps(self, response_text: str) -> list[tuple[str, int, int]]:
        """Split response text into steps and return (step_text, char_start, char_end).

        Returns:
            List of (step_text, char_start, char_end) tuples.
        """
        segments = self.split_fn(response_text)
        steps = []
        cursor = 0
        delimiter = "\n\n"
        for idx, seg in enumerate(segments):
            start = cursor
            end = cursor + len(seg)
            steps.append((seg, start, end))
            cursor = end + len(delimiter)  # skip delimiter
        return steps

    def _get_step_token_positions(self, response_text: str, valid_response_ids, valid_response_length: int):
        """Map character-level step boundaries to token positions.

        Returns:
            List of (step_text, token_end_pos) where token_end_pos is the
            index of the last token in this step (within response_ids).
        """
        steps = self._split_response_into_steps(response_text)
        result = []
        for step_text, char_start, char_end in steps:
            # Encode the text up to the end of this step to find token position
            text_up_to_end = response_text[:char_end]
            tokens_up_to_end = self.tokenizer.encode(text_up_to_end, add_special_tokens=False)
            token_end_pos = min(len(tokens_up_to_end) - 1, valid_response_length - 1)
            token_end_pos = max(0, token_end_pos)
            result.append((step_text, token_end_pos))
        return result

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

        # 2.2 Compute process rewards for each step_reward_type
        for reward_type in self.step_reward_types:
            reward_fn = STEP_REWARD_FN_REGISTRY.get(reward_type, _compute_step_reward_random)
            step_rewards = []  # list of (token_position, score)
            step_history = []
            for step_text, token_end_pos in step_positions:
                step_history.append(step_text)
                step_score = reward_fn(step_text, "", step_history)
                step_rewards.append((int(token_end_pos), float(step_score)))

            # Store as serializable list of (pos, score) tuples
            key = f"{reward_type}_step_reward"
            reward_extra_info[key] = step_rewards

        # Store number of steps for debugging
        reward_extra_info["num_steps"] = len(step_positions)

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
