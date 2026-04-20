"""MCTSRewardManager — reward manager for the parallel_mcts sampling strategy.

Two reward signals are produced:

1. **PRM reward_tensor** (main training signal, token-level):
   - If the batch already contains ``reward_fn_scores`` (placed there by
     :class:`~verl.trainer.ppo.sampling.parallel_mcts.ParallelMCTSStrategy`),
     those scores are used directly as the token-level reward.
   - Fallback (validation / non-MCTS batches): the full response is checked
     against the format rule and a single scalar is written at the last valid
     token position, mirroring :class:`NaiveRewardManager`.

2. **ORM verifiable reward** (logging / tracking):
   - ``compute_score(response_str, ground_truth)`` → 0/1 binary.
   - Stored in ``reward_extra_info["verifiable_reward"]`` and ``outcome_reward``
     so that ``ray_trainer.py`` can log ``reward/mean_fn_reward``.

The constructor accepts ``**kwargs`` to absorb all YAML ``reward_kwargs``
fields that other managers understand (e.g. ``print_entropy_tree``,
``entropy_tree_graphviz_dir``), preventing ``TypeError`` on unexpected kwargs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


class MCTSRewardManager:
    """Reward manager for the ``parallel_mcts`` sampling strategy."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        reward_style: str = "format",   # "format" | "fol"
        **kwargs,                         # absorb all other yaml reward_kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_style = reward_style

        # Lazy-load the PRM function so we don't pay the import cost unless needed
        self._step_prm_fn: Optional[Callable[[str], float]] = None

    @property
    def step_prm_fn(self) -> Callable[[str], float]:
        if self._step_prm_fn is None:
            from verl.trainer.ppo.sampling.mcts_prm import get_prm_fn
            self._step_prm_fn = get_prm_fn(self.reward_style)
        return self._step_prm_fn

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_response(self, data_item, prompt_length: int):
        """Decode valid prompt and response strings from a DataProtoItem."""
        prompt_ids = data_item.batch["prompts"]
        valid_prompt_len = int(data_item.batch["attention_mask"][:prompt_length].sum())
        valid_prompt_ids = prompt_ids[-valid_prompt_len:]

        response_ids = data_item.batch["responses"]
        valid_response_len = int(data_item.batch["attention_mask"][prompt_length:].sum())
        valid_response_ids = response_ids[:valid_response_len]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        return prompt_str, response_str, valid_response_len

    def _fallback_format_score(self, response_str: str) -> float:
        """Whole-response format check: all <step> blocks must be well-formed."""
        import re
        steps = re.findall(r"<step>(.*?)</step>", response_str, re.DOTALL)
        if not steps:
            return 0.0
        try:
            return 1.0 if all(self.step_prm_fn(s) == 1.0 for s in steps) else 0.0
        except NotImplementedError:
            return 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute reward tensors for a batch.

        Returns
        -------
        reward_tensor : torch.Tensor  (when return_dict=False)
        dict          : {reward_tensor, reward_extra_info, outcome_reward,
                         prompt, response, ground_truth}  (when return_dict=True)
        """
        # ---- Choose PRM tensor source --------------------------------
        has_precomputed = "reward_fn_scores" in data.batch

        if has_precomputed:
            # MCTS training path: use scores already computed by the strategy
            reward_tensor = data.batch["reward_fn_scores"].clone().float()
        else:
            # Fallback path: build a zero tensor, fill at end of response
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # ---- Per-sample ORM + logging --------------------------------
        reward_extra_info: dict = defaultdict(list)
        prompts_list: list = []
        responses_list: list = []
        ground_truths_list: list = []
        orm_scores: list = []
        response_lens: list = []

        already_printed: dict = {}

        for i in range(len(data)):
            item = data[i]
            prompt_length = item.batch["prompts"].shape[-1]
            prompt_str, response_str, valid_resp_len = self._decode_response(item, prompt_length)
            response_lens.append(valid_resp_len)

            # --- Fallback PRM fill (only when no precomputed scores) ---
            if not has_precomputed and valid_resp_len > 0:
                fmt_score = self._fallback_format_score(response_str)
                reward_tensor[i, valid_resp_len - 1] = fmt_score

            # --- ORM verifiable reward ---------------------------------
            ground_truth = item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            if ground_truth is None:
                # try the flat 'answer' key used by some datasets
                ground_truth = item.non_tensor_batch.get("answer", None)

            data_source = item.non_tensor_batch.get(self.reward_fn_key, "unknown")

            orm_score = 0.0
            if ground_truth is not None:
                try:
                    raw = self.compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                    )
                    orm_score = float(raw["score"] if isinstance(raw, dict) else raw)
                except Exception:
                    orm_score = 0.0

            orm_scores.append(orm_score)
            reward_extra_info["verifiable_reward"].append(orm_score)
            reward_extra_info["prm_score"].append(float(reward_tensor[i].sum()))

            prompts_list.append(prompt_str)
            responses_list.append(response_str)
            ground_truths_list.append(str(ground_truth) if ground_truth is not None else "")

            # --- Optional console print --------------------------------
            data_source_key = str(data_source)
            already_printed.setdefault(data_source_key, 0)
            if already_printed[data_source_key] < self.num_examine:
                already_printed[data_source_key] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[orm_score]", orm_score)
                print("[prm_sum]", reward_tensor[i].sum().item())

        if return_dict:
            # outcome_reward_tensor: [B, L_resp], ORM 0/1 at last valid token
            outcome_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            for i, (resp_len, score) in enumerate(zip(response_lens, orm_scores)):
                if resp_len > 0:
                    outcome_reward_tensor[i, resp_len - 1] = score

            return {
                "reward_tensor": reward_tensor,
                "outcome_reward_tensor": outcome_reward_tensor,
                "reward_extra_info": reward_extra_info,
                "outcome_reward": orm_scores,
                "prompt": prompts_list,
                "response": responses_list,
                "ground_truth": ground_truths_list,
            }
        return reward_tensor
