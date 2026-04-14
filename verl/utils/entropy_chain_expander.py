from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from verl import DataProto

_MCTS_UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mcts_utils"))
if _MCTS_UTILS_DIR not in sys.path:
    sys.path.insert(0, _MCTS_UTILS_DIR)

from tree_node import TreeNode  # noqa: E402


def _repeat_lst(lst: Optional[List], times: int) -> Optional[List]:
    if lst is None:
        return None
    return [item for item in lst for _ in range(times)]


def _pad_and_stack(
    seqs: List[torch.Tensor], pad_token_id: int, device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(seq.size(0) for seq in seqs)
    dtype = seqs[0].dtype
    batch = torch.full((len(seqs), max_len), pad_token_id, dtype=dtype, device=device)
    attn = torch.zeros((len(seqs), max_len), dtype=dtype, device=device)
    for i, seq in enumerate(seqs):
        length = seq.size(0)
        batch[i, :length] = seq
        attn[i, :length] = 1
    pos_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(len(seqs), max_len)
    return batch, attn, pos_ids


@dataclass
class ExpansionPlan:
    expansion_batch: Optional[DataProto]
    task_mapping: Dict[int, Tuple[int, int, TreeNode, int]]

    @property
    def batch_size(self) -> int:
        if self.expansion_batch is None or self.expansion_batch.batch is None:
            return 0
        return int(self.expansion_batch.batch.batch_size[0])


def _serialize_tree_lists(tree_lists) -> List[List[dict]]:
    """Convert TreeNode lists to plain-dict lists for Ray-safe serialization.

    TreeNode objects live in ``mcts_utils/tree_node.py`` which is not a
    proper Python package.  Passing them through Ray (pickle) causes
    ``ModuleNotFoundError: No module named 'tree_node'`` on remote workers.
    """
    result: List[List[dict]] = []
    for tree_list in tree_lists:
        serialized_tree: List[dict] = []
        for node in tree_list:
            serialized_tree.append({
                "node_idx": node.node_idx,
                "token_id_list": list(node.token_id_list),
                "score": node.score,
                "binary_score": node.binary_score,
                "finish_reason": node.finish_reason,
                "is_end": node.is_end,
                "parent_node_split_idx": node.parent_node_split_idx,
                "child_node_indices": [child.node_idx for child in node.child_nodes],
            })
        result.append(serialized_tree)
    return result


class EntropyChainExpander:
    """Entropy-guided chain expansion for verl rollout batches.

    This class keeps the core expansion logic aligned with
    `mcts_utils/entropy_chain_local_manager.py`, while replacing the standalone
    local-vLLM calls with verl's `generate_sequences` and `compute_log_prob`.
    """

    def __init__(
        self,
        tokenizer,
        pad_token_id: int,
        N: int = 4,
        L: int = 3,
        T: int = 1,
        max_token_num: int = 4096,
        evaluation_strategy: str = "token-entropy",
        enforce_uniform_per_prompt: bool = True,
    ):
        if evaluation_strategy != "token-entropy":
            raise ValueError(
                "EntropyChainExpander currently supports evaluation_strategy='token-entropy' only."
            )

        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.N = N
        self.L = L
        self.T = T
        self.max_token_num = max_token_num
        self.evaluation_strategy = evaluation_strategy
        self.enforce_uniform_per_prompt = enforce_uniform_per_prompt

        self.tree_lists: List[List[TreeNode]] = []
        self.init_inputs: List[List[int]] = []
        self.expanded_init_inputs: List[List[int]] = []
        self.tree_to_prompt_idx: List[int] = []
        self.prompt_batch_size: int = 0
        self._prompt_non_tensors: Dict[str, list] = {}

    def _decode_fn(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def _detect_finish_reason(self, token_ids: List[int]) -> str:
        if len(token_ids) > 0 and token_ids[-1] == self.tokenizer.eos_token_id:
            return "stop"
        return "length"

    @staticmethod
    def _compute_response_mask(data: DataProto) -> torch.Tensor:
        responses = data.batch["responses"]
        response_length = responses.size(1)
        attention_mask = data.batch["attention_mask"] if "attention_mask" in data.batch.keys() else None
        if attention_mask is None:
            return torch.ones_like(responses, dtype=torch.float32)
        return attention_mask[:, -response_length:]

    def _extract_valid_ids(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> List[int]:
        """Extract token IDs where mask is 1, handling both left-padded and right-padded sequences."""
        mask_bool = token_mask.bool()
        if not mask_bool.any():
            return []
        return token_ids[mask_bool].tolist()

    def _build_expansion_batch(self, input_sequences: List[torch.Tensor]) -> DataProto:
        input_ids, attention_mask, position_ids = _pad_and_stack(input_sequences, pad_token_id=self.pad_token_id)
        # Provide clean (unpadded) prompt IDs so vLLM skips _pre_process_inputs
        # and doesn't include trailing PAD tokens as part of the prompt.
        raw_prompt_ids = np.array([seq.tolist() for seq in input_sequences], dtype=object)
        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": input_ids,
            },
            non_tensors={"raw_prompt_ids": raw_prompt_ids},
            meta_info={},
        )

    def initialize(
        self,
        gen_batch: DataProto,
        gen_batch_output: DataProto,
        compute_log_prob_fn: Callable[[DataProto], DataProto],
    ) -> None:
        self.tree_lists = []
        self.init_inputs = []
        self.expanded_init_inputs = []
        self.tree_to_prompt_idx = []

        prompt_input_ids = gen_batch.batch["input_ids"]
        if "attention_mask" in gen_batch.batch.keys():
            prompt_attention_mask = gen_batch.batch["attention_mask"]
        else:
            prompt_attention_mask = torch.ones_like(prompt_input_ids)

        self.prompt_batch_size = int(prompt_input_ids.size(0))
        for i in range(self.prompt_batch_size):
            self.init_inputs.append(self._extract_valid_ids(prompt_input_ids[i], prompt_attention_mask[i]))

        self._prompt_non_tensors = {}
        if gen_batch.non_tensor_batch is not None:
            for key, vals in gen_batch.non_tensor_batch.items():
                self._prompt_non_tensors[key] = list(vals)

        responses = gen_batch_output.batch["responses"]
        response_mask = self._compute_response_mask(gen_batch_output)

        output_batch_size = int(responses.size(0))
        if self.prompt_batch_size <= 0:
            raise ValueError("Prompt batch size must be positive when initializing entropy chain.")
        if output_batch_size % self.prompt_batch_size != 0:
            raise ValueError(
                f"Output batch size {output_batch_size} is not divisible by prompt batch size {self.prompt_batch_size}."
            )

        rollout_n = output_batch_size // self.prompt_batch_size
        self.expanded_init_inputs = _repeat_lst(self.init_inputs, rollout_n)
        self.tree_to_prompt_idx = _repeat_lst(list(range(self.prompt_batch_size)), rollout_n)

        log_prob_output = compute_log_prob_fn(gen_batch_output)
        old_log_probs = log_prob_output.batch["old_log_probs"]

        for idx in range(output_batch_size):
            token_id_list = self._extract_valid_ids(responses[idx], response_mask[idx])
            log_prob_list = (
                old_log_probs[idx, : len(token_id_list)].tolist() if len(token_id_list) > 0 else []
            )
            finish_reason = self._detect_finish_reason(token_id_list)
            root_node = TreeNode(
                tree_idx=idx,
                node_idx=0,
                decode_fn=self._decode_fn,
                token_id_list=token_id_list,
                log_prob_list=log_prob_list,
                is_end=True,
                finish_reason=finish_reason,
                max_length=self.max_token_num,
                evaluation_strategy=self.evaluation_strategy,
            )
            self.tree_lists.append([root_node])

    def _build_expansion_plan(self) -> Optional[ExpansionPlan]:
        expansion_tasks = []
        for tree_idx, tree_list in enumerate(self.tree_lists):
            tree_entropy_tokens = []
            for node_idx, node in enumerate(tree_list):
                if not all(node.mask):
                    entropy_tokens = node.get_max_entropy_tokens(top_n=self.N) # 当前 Node 里面熵最高的 N 个 tokens
                    for token_idx in entropy_tokens:
                        entropy_value = -node.token_score[token_idx]
                        tree_entropy_tokens.append((entropy_value, tree_idx, node_idx, node, token_idx))

            tree_entropy_tokens.sort(reverse=True)
            expansion_tasks.extend(
                [
                    (tree_idx, node_idx, node, token_idx)
                    for _, tree_idx, node_idx, node, token_idx in tree_entropy_tokens[: self.N]
                ]
            )

        if not expansion_tasks:
            return None

        model_inputs: List[torch.Tensor] = []
        task_mapping: Dict[int, Tuple[int, int, TreeNode, int]] = {}
        repeated_tasks = _repeat_lst(expansion_tasks, self.T)
        for tree_idx, node_idx, node, split_idx in repeated_tasks:
            node.mask[split_idx] = True
            prefix_ids = node.get_prefix_ids(split_idx)
            full_input = self.expanded_init_inputs[tree_idx] + prefix_ids
            model_inputs.append(torch.tensor(full_input, dtype=torch.long))
            task_mapping[len(model_inputs) - 1] = (tree_idx, node_idx, node, split_idx)

        if not model_inputs:
            return None

        return ExpansionPlan(
            expansion_batch=self._build_expansion_batch(model_inputs),
            task_mapping=task_mapping,
        )

    def expand_one_round(
        self,
        generate_fn: Callable[[DataProto], DataProto],
        compute_log_prob_fn: Callable[[DataProto], DataProto],
    ) -> bool:
        plan = self._build_expansion_plan()
        if plan is None or plan.batch_size == 0:
            return False

        expansion_output = generate_fn(plan.expansion_batch)
        log_prob_output = compute_log_prob_fn(expansion_output)

        output_batch_size = int(expansion_output.batch.batch_size[0])
        if output_batch_size % plan.batch_size != 0:
            raise ValueError(
                f"Expanded output batch size {output_batch_size} is not divisible by expansion plan size {plan.batch_size}."
            )
        sample_multiplier = output_batch_size // plan.batch_size

        responses = expansion_output.batch["responses"][::sample_multiplier]
        attention_mask = expansion_output.batch["attention_mask"][::sample_multiplier]
        old_log_probs = log_prob_output.batch["old_log_probs"][::sample_multiplier]

        if int(responses.size(0)) != plan.batch_size:
            raise ValueError(
                f"Compacted expansion output size {int(responses.size(0))} does not match expected {plan.batch_size}."
            )

        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]

        for i in range(plan.batch_size):
            tree_idx, node_idx, parent_node, split_idx = plan.task_mapping[i]
            token_id_list = self._extract_valid_ids(responses[i], response_mask[i])
            if len(token_id_list) == 0:
                continue

            log_prob_list = old_log_probs[i, : len(token_id_list)].tolist()
            finish_reason = self._detect_finish_reason(token_id_list)

            new_node = TreeNode(
                tree_idx=tree_idx,
                node_idx=len(self.tree_lists[tree_idx]),
                token_id_list=token_id_list,
                decode_fn=self._decode_fn,
                log_prob_list=log_prob_list,
                is_end=True,
                parent_node=parent_node,
                parent_node_idx=node_idx,
                parent_node_split_idx=split_idx,
                finish_reason=finish_reason,
                max_length=self.max_token_num,
                evaluation_strategy=self.evaluation_strategy,
            )
            parent_node.add_child(new_node, split_idx)
            self.tree_lists[tree_idx].append(new_node)

        return True

    def build_expanded_batch(self) -> DataProto:
        if self.prompt_batch_size <= 0:
            raise ValueError("EntropyChainExpander is not initialized.")

        # Each entry: (prompt_ids, response_ids, tree_idx, node_idx, prompt_idx)
        grouped_samples: List[List[Tuple[List[int], List[int], int, int, int]]] = [
            [] for _ in range(self.prompt_batch_size)
        ]
        for tree_idx, tree_list in enumerate(self.tree_lists):
            prompt_idx = self.tree_to_prompt_idx[tree_idx]
            prompt_ids = self.expanded_init_inputs[tree_idx]
            for node_idx, node in enumerate(tree_list):
                if not node.is_end:
                    continue
                response_ids = node.aggregate_token_ids + node.token_id_list
                grouped_samples[prompt_idx].append(
                    (prompt_ids, response_ids, tree_idx, node_idx, prompt_idx)
                )

        per_prompt_counts = [len(samples) for samples in grouped_samples]
        if any(count == 0 for count in per_prompt_counts):
            raise ValueError(f"Some prompts have no entropy-chain samples: {per_prompt_counts}")

        target_count = min(per_prompt_counts)
        if self.enforce_uniform_per_prompt and len(set(per_prompt_counts)) > 1:
            print(
                "[EntropyChainExpander] Uneven sample counts per prompt "
                f"{per_prompt_counts}; truncating each prompt to {target_count}."
            )

        ordered_samples: List[Tuple[List[int], List[int], int, int, int]] = []
        for prompt_idx in range(self.prompt_batch_size):
            samples = grouped_samples[prompt_idx]
            if self.enforce_uniform_per_prompt:
                samples = samples[:target_count]
            ordered_samples.extend(samples)

        if not ordered_samples:
            raise ValueError("No expanded samples were collected from entropy-chain trees.")

        max_prompt_len = max(len(p) for p, _, _, _, _ in ordered_samples)
        max_response_len = max(len(r) for _, r, _, _, _ in ordered_samples)
        if max_response_len <= 0:
            raise ValueError("All expanded responses are empty.")

        total_len = max_prompt_len + max_response_len
        batch_size = len(ordered_samples)

        input_ids = torch.full((batch_size, total_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, total_len), dtype=torch.long)
        prompts_padded = torch.full((batch_size, max_prompt_len), self.pad_token_id, dtype=torch.long)
        responses_padded = torch.full((batch_size, max_response_len), self.pad_token_id, dtype=torch.long)

        sample_tree_idxs = np.empty(batch_size, dtype=np.int64)
        sample_node_idxs = np.empty(batch_size, dtype=np.int64)
        sample_prompt_idxs = np.empty(batch_size, dtype=np.int64)

        for i, (prompt_ids, response_ids, t_idx, n_idx, p_idx) in enumerate(ordered_samples):
            p_len = len(prompt_ids)
            r_len = len(response_ids)
            p_offset = max_prompt_len - p_len

            input_ids[i, p_offset:max_prompt_len] = torch.tensor(prompt_ids, dtype=torch.long)
            attention_mask[i, p_offset:max_prompt_len] = 1
            input_ids[i, max_prompt_len:max_prompt_len + r_len] = torch.tensor(response_ids, dtype=torch.long)
            attention_mask[i, max_prompt_len:max_prompt_len + r_len] = 1

            prompts_padded[i, p_offset:] = torch.tensor(prompt_ids, dtype=torch.long)
            responses_padded[i, :r_len] = torch.tensor(response_ids, dtype=torch.long)

            sample_tree_idxs[i] = t_idx
            sample_node_idxs[i] = n_idx
            sample_prompt_idxs[i] = p_idx

        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        non_tensors: Dict[str, np.ndarray] = {
            "entropy_tree_idx": sample_tree_idxs,
            "entropy_node_idx": sample_node_idxs,
            "entropy_prompt_idx": sample_prompt_idxs,
        }
        for key, per_prompt_vals in self._prompt_non_tensors.items():
            non_tensors[key] = np.array(
                [per_prompt_vals[int(sample_prompt_idxs[i])] for i in range(batch_size)],
                dtype=object,
            )

        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompts_padded,
                "responses": responses_padded,
            },
            non_tensors=non_tensors,
            meta_info={
                "entropy_tree_metadata": {
                    "tree_lists": _serialize_tree_lists(self.tree_lists),
                    "tree_to_prompt_idx": list(self.tree_to_prompt_idx),
                },
            },
        )
