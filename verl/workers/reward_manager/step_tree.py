"""StepTreeRewardManager — reward manager for the step_treerl sampling strategy.

Three outputs are produced:

1. **Training reward_tensor** (token-level):
   - If the batch already contains ``reward_fn_scores`` (placed there by
     :class:`~verl.trainer.ppo.sampling.step_treerl.StepTreeRLStrategy`),
     those scores are used directly as the token-level reward. Depending on
     the configured training mode these are dense segment rewards or a sparse
     leaf-outcome reward.
   - Fallback (validation / non-step-tree batches): each ``<step>`` block is
     scored via the format/FOL PRM function, written at the step's last token.
   - ``self_eval`` requires precomputed scores from StepTreeRL because actor
     generation is not available inside the reward manager.

2. **Process-reward statistics** (logging / tracking):
   - Precomputed ``process_reward_scores`` are kept separate from the training
     reward and summed per selected trajectory.
   - Validation falls back to scoring each complete ``<step>...</step>`` block.

3. **Outcome accuracy** (logging / tracking):
   - ``compute_score(response_str, ground_truth)`` → 0/1 binary.
   - Stored as ``acc`` so validation logs it under ``val-core/<dataset>/acc``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Optional

import torch

from verl import DataProto
from verl.trainer.ppo.sampling.mcts_prm import classify_rollout_format, rollout_format_infos_to_metric_columns
from verl.utils.reward_score import _default_compute_score
from verl.utils.process_reward import (
    build_process_reward_runtime,
    get_item_sample_id,
    normalize_process_reward_cfg,
)


class StepTreeRewardManager:
    """Reward manager for the ``step_treerl`` sampling strategy.

    Supports two reward styles:
    - 'format': checks <step>/<premise>/<conclusion> tag structure
    - 'fol': FOL/Z3 verification (requires trainer.process_reward.fol.*)
    - 'self_eval': actor-generated step correctness (precomputed by StepTreeRL)
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        process_reward_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        if process_reward_cfg is None:
            raise ValueError("StepTreeRewardManager requires process_reward_cfg.")
        self.process_reward_cfg = normalize_process_reward_cfg(process_reward_cfg)
        self.process_reward_runtime = build_process_reward_runtime(self.process_reward_cfg)
        self.process_reward_type = self.process_reward_runtime.reward_type
        if self.process_reward_type == "none":
            raise ValueError(
                "StepTreeRewardManager requires trainer.process_reward.type to be 'format', 'fol', or 'self_eval'."
            )
        self._step_prm_fn = self.process_reward_runtime.step_prm_fn
        self.fol_verifier = self.process_reward_runtime.fol_verifier
        self.fol_metadata_map = self.process_reward_runtime.fol_metadata_map

    @property
    def step_prm_fn(self) -> Callable[[str], float]:
        return self._step_prm_fn

    def _decode_response(self, data_item, prompt_length: int):
        """从 batch 中抽取一个样本的 prompt 和 response 文本。"""
        prompt_ids = data_item.batch["prompts"]
        response_ids = data_item.batch["responses"]
        attention_mask = data_item.batch["attention_mask"]
        response_width = response_ids.shape[-1]

        if prompt_length != prompt_ids.shape[-1]:
            raise ValueError(
                f"Prompt width mismatch: prompt_length={prompt_length}, prompts.shape[-1]={prompt_ids.shape[-1]}."
            )
        if response_width <= 0:
            raise ValueError("StepTreeRewardManager received a response tensor with zero width.")

        expected_total_width = prompt_length + response_width
        if attention_mask.shape[-1] != expected_total_width:
            raise ValueError(
                "Invalid prompt/response batch layout: "
                f"attention_mask width {attention_mask.shape[-1]} does not equal "
                f"prompt width {prompt_length} + response width {response_width}."
            )

        prompt_mask = attention_mask[:prompt_length].bool()
        attention_response_mask = attention_mask[-response_width:].bool()
        response_mask = (
            data_item.batch["response_mask"]
            if "response_mask" in data_item.batch
            else None
        )
        if response_mask is None:
            response_mask = attention_response_mask
        else:
            if response_mask.shape[-1] != response_width:
                raise ValueError(
                    f"Response mask width {response_mask.shape[-1]} does not equal response width {response_width}."
                )
            response_mask = response_mask.bool()
            if not torch.equal(response_mask, attention_response_mask):
                raise ValueError("response_mask does not match the response segment of attention_mask.")

        valid_prompt_ids = prompt_ids[prompt_mask]
        valid_response_ids = response_ids[response_mask]
        valid_response_len = int(response_mask.sum().item())
        if valid_response_len <= 0:
            raise ValueError("StepTreeRewardManager received an empty valid response.")

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        return prompt_str, response_str, valid_response_len

    def _fallback_step_scores(self, response_str: str, sample_id: Optional[str] = None) -> list:
        """Score each <step> block individually using the PRM function."""
        """当 batch 中没有预计算的 reward_fn_scores 时（比如验证阶段），回退到用正则提取 <step>...</step> 块，逐块调用 PRM 函数评分。"""
        import re

        steps = re.findall(r"(<step>.*?</step>)", response_str, re.DOTALL)
        if not steps:
            return []
        if self.process_reward_type == "self_eval":
            raise ValueError("self_eval process reward requires precomputed reward_fn_scores from StepTreeRLStrategy.")
        scores = []
        for s in steps:
            if self.process_reward_type == "fol":
                if sample_id is None:
                    raise ValueError("FOL process reward fallback requires sample_id.")
                scores.append(self.step_prm_fn(s, sample_id=sample_id))
            else:
                scores.append(self.step_prm_fn(s))
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        has_precomputed = "reward_fn_scores" in data.batch

        if has_precomputed:
            reward_tensor = data.batch["reward_fn_scores"].clone().float()
        else:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        process_reward_tensor = (
            data.batch["process_reward_scores"]
            if "process_reward_scores" in data.batch
            else None
        )
        if process_reward_tensor is None:
            # Backward compatibility for validation and older precomputed
            # batches where the training reward was also the process reward.
            process_reward_tensor = reward_tensor
        else:
            process_reward_tensor = process_reward_tensor.clone().float()
            if process_reward_tensor.shape != reward_tensor.shape:
                raise ValueError(
                    "process_reward_scores shape "
                    f"{tuple(process_reward_tensor.shape)} does not match reward tensor shape "
                    f"{tuple(reward_tensor.shape)}."
                )

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
                sample_id = get_item_sample_id(item.non_tensor_batch)
                step_scores = self._fallback_step_scores(response_str, sample_id=sample_id)
                if step_scores:
                    # Locate </step> boundaries in tokenized response
                    response_width = item.batch["responses"].shape[-1]
                    if "response_mask" in item.batch:
                        response_mask = item.batch["response_mask"].bool()
                    else:
                        response_mask = item.batch["attention_mask"][-response_width:].bool()
                    response_ids = item.batch["responses"][response_mask]
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
            reward_extra_info["acc"].append(orm_score)
            reward_extra_info["prm_score"].append(float(process_reward_tensor[i].sum()))
            format_info = classify_rollout_format(response_str)
            for key, values in rollout_format_infos_to_metric_columns([format_info]).items():
                reward_extra_info[key].extend(values)

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
                print("[prm_sum]", process_reward_tensor[i].sum().item())

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
