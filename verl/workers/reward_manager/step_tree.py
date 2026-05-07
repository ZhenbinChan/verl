"""StepTreeRewardManager — reward manager for the step_treerl sampling strategy.

Two reward signals are produced:

1. **PRM reward_tensor** (main training signal, token-level):
   - If the batch already contains ``reward_fn_scores`` (placed there by
     :class:`~verl.trainer.ppo.sampling.step_treerl.StepTreeRLStrategy`),
     those scores are used directly as the token-level reward.
   - Fallback (validation / non-step-tree batches): each ``<step>`` block is
     scored via the format/FOL PRM function, written at the step's last token.

2. **ORM verifiable reward** (logging / tracking):
   - ``compute_score(response_str, ground_truth)`` → 0/1 binary.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Callable, Dict, Optional

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


class StepTreeRewardManager:
    """Reward manager for the ``step_treerl`` sampling strategy.

    Supports two reward styles:
    - 'format': checks <step>/<premise>/<conclusion> tag structure
    - 'fol': FOL/Z3 verification (requires fol_metadata_path)
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        reward_style: str = "format",
        fol_metadata_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_style = reward_style

        self._step_prm_fn: Optional[Callable[[str], float]] = None

        # FOL verifier initialization
        self.fol_verifier = None
        self.fol_metadata_map: Dict[str, "FOLMetadata"] = {}

        if reward_style == "fol" and fol_metadata_path:
            self._init_fol_verifier(fol_metadata_path)

    def _init_fol_verifier(self, metadata_path: str) -> None:
        if not os.path.exists(metadata_path):
            print(f"[FOL Warning] FOL metadata path not found: {metadata_path}")
            return

        try:
            from verl.utils.fol_verifier import FOLVerifier, FOLMetadata
            import json

            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                if item.get("fol_metadata"):
                    sample_id = (
                        item.get("sample_id")
                        or item.get("extra_info", {}).get("index")
                        or item.get("extra_info", {}).get("id")
                    )
                    if sample_id is not None:
                        self.fol_metadata_map[str(sample_id)] = FOLMetadata.from_dict(
                            item["fol_metadata"]
                        )

            self.fol_verifier = FOLVerifier()
            print(f"[FOL] StepTree RewardManager loaded {len(self.fol_metadata_map)} FOL metadata entries")

        except Exception as e:
            print(f"[FOL Warning] Failed to initialize FOL verifier: {e}")

    @property
    def step_prm_fn(self) -> Callable[[str], float]:
        if self._step_prm_fn is None:
            from verl.trainer.ppo.sampling.mcts_prm import get_prm_fn

            if self.reward_style == "fol" and self.fol_verifier:
                self._step_prm_fn = get_prm_fn(
                    self.reward_style,
                    verifier=self.fol_verifier,
                    metadata_map=self.fol_metadata_map,
                )
            else:
                self._step_prm_fn = get_prm_fn(self.reward_style)
        return self._step_prm_fn

    def _decode_response(self, data_item, prompt_length: int):
        prompt_ids = data_item.batch["prompts"]
        valid_prompt_len = int(data_item.batch["attention_mask"][:prompt_length].sum())
        valid_prompt_ids = prompt_ids[-valid_prompt_len:]

        response_ids = data_item.batch["responses"]
        valid_response_len = int(data_item.batch["attention_mask"][prompt_length:].sum())
        valid_response_ids = response_ids[:valid_response_len]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        return prompt_str, response_str, valid_response_len

    def _fallback_step_scores(self, response_str: str) -> list:
        """Score each <step> block individually using the PRM function."""
        import re

        steps = re.findall(r"<step>(.*?)</step>", response_str, re.DOTALL)
        if not steps:
            return []
        scores = []
        for s in steps:
            try:
                scores.append(self.step_prm_fn(s))
            except NotImplementedError:
                scores.append(0.0)
            except Exception:
                scores.append(0.0)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        has_precomputed = "reward_fn_scores" in data.batch

        if has_precomputed:
            reward_tensor = data.batch["reward_fn_scores"].clone().float()
        else:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

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

            # Fallback: fill step-level PRM scores at each </step> boundary
            if not has_precomputed and valid_resp_len > 0:
                step_scores = self._fallback_step_scores(response_str)
                if step_scores:
                    # Locate </step> boundaries in tokenized response
                    response_ids = item.batch["responses"][:valid_resp_len]
                    full_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    import re
                    boundaries = []
                    for m in re.finditer(r"</step>", full_text):
                        # Approximate token position by re-encoding up to match end
                        prefix = full_text[:m.end()]
                        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                        boundaries.append(len(prefix_ids) - 1)

                    for j, (boundary_pos, score) in enumerate(zip(boundaries, step_scores)):
                        if boundary_pos < valid_resp_len:
                            reward_tensor[i, boundary_pos] = score

            # ORM verifiable reward
            ground_truth = item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            if ground_truth is None:
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
            # Add "acc" as alias for verifiable_reward so it becomes val-core in wandb
            reward_extra_info["acc"].append(orm_score)
            reward_extra_info["prm_score"].append(float(reward_tensor[i].sum()))

            prompts_list.append(prompt_str)
            responses_list.append(response_str)
            ground_truths_list.append(str(ground_truth) if ground_truth is not None else "")

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
