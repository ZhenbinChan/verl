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
TreeRL: Tree structure utilities for EPTree-based tree search in RL training.

Implements the EPTree algorithm (arXiv:2506.11902) for (cross-)entropy-guided tree search.
The core idea: iteratively expand search trees by forking new branches from the
top-N most uncertain steps, then use the tree structure to compute process
supervision signals via leave-one-out normalization + backprop + step normalization.

Step reward pipeline (aligned with TreeRL reference code):
    evaluate_leaves → leaf_normalize → backpropagate → normalize_all_steps
    → reweight_steps (optional) → compute_step_rewards

Reference: Algorithm 1 in "TreeRL: LLM Reinforcement Learning with On-Policy Tree Search"
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from verl.utils.fol_utils.common import check_step_format_fol
from verl.utils.step_splitter import default_split_fn, get_split_fn

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_repeated_patterns(s: str, pattern_length: int = 50, threshold: int = 20) -> dict:
    """Detect degenerate repetition via N-gram frequency. (2nd strategy of repetition penalty)

    Ref: THUNLP/TreeRL parallel_mcts.py:184, remote_reward.py:19
    """
    if len(s) < pattern_length:
        return {}
    ngrams = [s[i:i + pattern_length] for i in range(len(s) - pattern_length + 1)]
    ngram_counts = Counter(ngrams)
    return {gram: count for gram, count in ngram_counts.items() if count > threshold}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A node in the search tree. Each node corresponds to one "step" of reasoning.

    In the EPTree framework, a "step" is a segment of tokens separated by
    the split_fn (e.g., split by "\\n\\n"). Each node stores the token IDs and
    their log probabilities for this segment.
    """

    node_id: int
    # Token data for this segment
    token_ids: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)  # log π(yt|x, y<t)

    # Tree structure
    parent: Optional[TreeNode] = None
    children: List[TreeNode] = field(default_factory=list)

    # Step text (decoded)
    step_text: str = ""

    # Token position range within the full response [start, end)
    token_start: int = 0
    token_end: int = 0

    # Value and reward (populated during backpropagation)
    value: float = 0.0          # V(sn) = A(sn) / |L(sn)|
    reward: float = 0.0         # step reward: V(sn) - V(parent)
    correctness: Optional[float] = None  # 0/1 for leaf nodes, None for internal
    accumulated_value: float = 0.0      # A(sn) = sum of normalized leaf scores
    terminal_in_subtree: int = 0        # |L(sn)| for backprop counting
    selected_terminal_in_subtree: int = 0  # for optional reweight (compute_weighted_update)

    # External PRM scores per step, e.g. {"format": 0.8, "fol": 0.6}
    # Populated by _map_ext_prm_to_nodes (original chain) and
    # evaluate_branch_ext_prm (forked nodes).
    ext_prm_scores: dict = field(default_factory=dict)

    # Metadata
    tree_idx: int = 0           # which tree this node belongs to
    is_forked: bool = False     # whether this node was created by forking
    finish_reason: Optional[str] = None  # "stop" (EOS) / "length" (token/step limit) / "repetition" (loop detected)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def entropy_scores(self) -> List[float]:
        """(Cross) Entropy = -log_prob for each token (cross-entropy as in the paper)."""
        return [-lp for lp in self.log_probs]

    @property
    def max_entropy(self) -> float:
        """Maximum entropy among all tokens in this node."""
        if not self.log_probs:
            return 0.0
        return max(-lp for lp in self.log_probs)

    @property
    def max_entropy_token_idx(self) -> int:
        """Index of the token with highest entropy within this node."""
        if not self.log_probs:
            return 0
        return int(np.argmax([-lp for lp in self.log_probs]))

    def descendant_leaves(self) -> List[TreeNode]:
        """Get all leaf nodes reachable from this node."""
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.descendant_leaves())
        return leaves

    def path_from_root(self) -> List[TreeNode]:
        """Get the path from root to this node (inclusive)."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def full_token_ids(self) -> List[int]:
        """Get all token IDs from root to this node."""
        path = self.path_from_root()
        ids = []
        for node in path:
            ids.extend(node.token_ids)
        return ids


@dataclass
class SearchTree:
    """A single search tree, initialized from one rollout response."""

    tree_idx: int
    root: TreeNode
    all_nodes: List[TreeNode] = field(default_factory=list)

    @property
    def all_leaves(self) -> List[TreeNode]:
        return [n for n in self.all_nodes if n.is_leaf]

    @property
    def num_leaves(self) -> int:
        return len(self.all_leaves)


# ---------------------------------------------------------------------------
# TreeManager: Coordinates the full EPTree pipeline
# ---------------------------------------------------------------------------

class TreeManager:
    """Manages the EPTree search process within the RL training loop.

    This class coordinates:
    1. Initializing trees from rollout responses
    2. Selecting forking points based on step entropy (per-tree Top-N)
    3. Preparing branch inputs for continuation generation
    4. Committing branch outputs back to the tree
    5. Evaluating leaves (correctness scoring)
    6. Computing step rewards via the TreeRL reference pipeline:
       leaf_normalize → backpropagate → normalize_all_steps → reweight (opt) → step_rewards
       Paper formula: R(sn) = [GA(sn) + LA(sn)] / sqrt(|L(sn)|)
       Reference code default: R(sn) = V(sn) - V(parent) (LA only)
       Configurable via tree_step_reward_mode: "la" / "ga_la" / "ga" / "value_only"
    7. Flattening all leaf paths into a standard DataProto batch
    """

    def __init__(
        self,
        config,
        tokenizer,
        split_fn: Optional[Callable] = None,
        use_xml: bool = False,
        ext_prm_max_workers: Optional[int] = None,
        penalty_on_bad_format: bool = False,
        penalty_score: float = 0.0,
    ):
        """
        Args:
            config: Trainer config (OmegaConf), needs tree_rounds, tree_top_n, tree_branches, etc.
            tokenizer: HuggingFace tokenizer for encoding/decoding.
            split_fn: Step splitter function. If None, derived from use_xml.
            use_xml: If True, attempt XML ``<step>`` tag splitting before falling
                back to the delimiter splitter.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.use_xml = use_xml
        self.penalty_on_bad_format = bool(penalty_on_bad_format)
        self.penalty_score = float(penalty_score)

        # Derive split_fn from use_xml when not explicitly provided
        if split_fn is not None:
            self.split_fn = split_fn
        else:
            self.split_fn = get_split_fn(use_xml=use_xml)

        # EPTree parameters from config
        self.tree_rounds = config.get("tree_rounds", 1)       # L
        self.tree_top_n = config.get("tree_top_n", 2)         # N
        self.tree_branches = config.get("tree_branches", 2)   # T
        self.mask_tail_ratio = config.get("tree_mask_tail_ratio", 0.1)  # mask末尾tokens
        self.use_weighted_value = config.get("tree_use_weighted_value", False)
        self.weighted_value_style = config.get("tree_weighted_value_style", "sqrt")
        self.overall_norm_style = config.get("tree_overall_norm_style", "token")
        self.step_reward_mode = config.get("tree_step_reward_mode", "la")

        # Anti-degeneration parameters (ref: THUNLP/TreeRL)
        # Inner repetition penalty applies a heuristic penalty to nodes whose step text contains repeated patterns,
        # which are often indicative of degenerate loops. (1st strategy for repetition penalty in TreeRL)
        self.inner_repetition_penalty = config.get("tree_inner_repetition_penalty", True)
        # max depth (3rd strategy)
        self.max_steps_per_path = config.get("tree_max_steps_per_path", 40)
        self.repetition_pattern_length = config.get("tree_repetition_pattern_length", 50)
        self.repetition_threshold = config.get("tree_repetition_threshold", 20)

        self.trees: List[SearchTree] = []
        self._node_counter = 0
        # Store prompt info for branch construction
        self._prompt_ids_list: List[List[int]] = []
        self._prompt_lengths: List[int] = []
        self._meta_info = {}
        self._non_tensor_batch_template = {}
        if ext_prm_max_workers is None:
            ext_prm_max_workers = min(32, os.cpu_count() or 4)
        self.ext_prm_max_workers = max(1, int(ext_prm_max_workers))
        self.ext_prm_profile: dict[str, float] = {}
        self.pipeline_profile: dict[str, float] = {}
        
        # TODO: 1. Diverse Sampling as a new anti-degeneration strategy (ref: THUNLP/TreeRL entropy_chain_local_manager.py:255)

    def _new_node_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def _should_penalize_step_format(self, step_text: str) -> bool:
        """Whether a per-step external PRM score should use the tree format penalty."""
        return bool(
            self.use_xml
            and self.penalty_on_bad_format
            and not check_step_format_fol(step_text or "")
        )

    # ------------------------------------------------------------------
    # Step 1: Initialize Trees
    # ------------------------------------------------------------------

    def initialize_trees(self, rollout_output: DataProto) -> List[SearchTree]:
        """Build M initial chain-trees from rollout responses.

        Each response is split into steps using split_fn, and each step becomes
        a TreeNode. Token-level log_probs from rollout are used to compute entropy.

        Args:
            rollout_output: DataProto from generate_sequences, containing:
                - batch["input_ids"]: (M, total_seq_len) or batch["prompts"] + batch["responses"]
                - batch["attention_mask"]: (M, total_seq_len)
                - batch["log_probs"] or batch["rollout_log_probs"]: (M, response_len)
                - non_tensor_batch: reward_model, data_source, etc.
        """
        self.trees = []
        self._node_counter = 0

        batch = rollout_output.batch
        M = batch.batch_size[0] if hasattr(batch, 'batch_size') else batch["responses"].shape[0]

        prompts = batch["prompts"]          # (M, prompt_len)
        responses = batch["responses"]      # (M, response_len)
        attention_mask = batch["attention_mask"]  # (M, total_len)

        # Get log probs from rollout
        log_probs_key = "rollout_log_probs" if "rollout_log_probs" in batch.keys() else "log_probs"
        if log_probs_key in batch.keys():
            all_log_probs = batch[log_probs_key]  # (M, response_len)
        else:
            # Fallback: no log probs available, use zeros (entropy will be 0)
            all_log_probs = torch.zeros_like(responses, dtype=torch.float32)

        prompt_len = prompts.shape[1]
        response_len = responses.shape[1]

        # Store prompt info + meta for branch construction
        self._prompt_ids_list = []
        self._prompt_lengths = []
        self._meta_info = dict(rollout_output.meta_info) if rollout_output.meta_info else {}
        self._non_tensor_batch_template = {}
        for key in rollout_output.non_tensor_batch.keys():
            self._non_tensor_batch_template[key] = rollout_output.non_tensor_batch[key]

        for i in range(M):
            # Compute valid response length
            resp_mask = attention_mask[i, prompt_len:]
            valid_resp_len = int(resp_mask.sum().item())
            valid_resp_ids = responses[i, :valid_resp_len].tolist()
            valid_log_probs = all_log_probs[i, :valid_resp_len].tolist()

            # Store prompt ids
            prompt_ids = prompts[i].tolist()
            # Remove padding from prompt (trailing pad tokens)
            prompt_mask = attention_mask[i, :prompt_len]
            valid_prompt_len = int(prompt_mask.sum().item())
            prompt_ids = prompt_ids[prompt_len - valid_prompt_len:]  # left-padded
            self._prompt_ids_list.append(prompt_ids)
            self._prompt_lengths.append(len(prompt_ids))

            # Decode full response
            response_text = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)

            # Split into steps.
            # XML path: character-level split then char→token mapping (same
            #   approach as StepRewardManager — keeps BPE drift but stays
            #   aligned with reward computation).
            # Delimiter path: token-level search — no decode→re-encode drift.
            from verl.utils.step_splitter import (split_by_xml_step_tags,
                                                  split_tokens_by_delimiter)

            xml_steps = split_by_xml_step_tags(response_text) if self.use_xml else []
            if xml_steps:
                # XML: map char boundaries → token positions via re-encode
                step_ranges: list[tuple[int, int, str]] = []
                _prev = 0
                for step_text, _cs, char_end in xml_steps:
                    text_up_to_end = response_text[:char_end]
                    toks_up = self.tokenizer.encode(text_up_to_end, add_special_tokens=False)
                    tok_end = min(len(toks_up), valid_resp_len)
                    step_ranges.append((_prev, tok_end, step_text))
                    _prev = tok_end
            else:
                # Delimiter: split directly in token space (no BPE drift)
                step_ranges = split_tokens_by_delimiter(
                    valid_resp_ids, self.tokenizer
                )

            # Build chain of TreeNodes
            root = None
            prev_node = None
            tree_nodes = []
            truncated_by_max_steps = False

            for step_token_start, step_token_end, step_text in step_ranges:
                if step_token_start >= step_token_end:
                    continue  # skip empty segments

                # max_steps_per_path truncation
                if len(tree_nodes) >= self.max_steps_per_path:
                    truncated_by_max_steps = True
                    break

                # Get tokens and log_probs for this step
                step_ids = valid_resp_ids[step_token_start:step_token_end]
                step_lps = valid_log_probs[step_token_start:step_token_end]

                node = TreeNode(
                    node_id=self._new_node_id(),
                    token_ids=step_ids,
                    log_probs=step_lps,
                    parent=prev_node,
                    step_text=step_text,
                    token_start=step_token_start,
                    token_end=step_token_end,
                    tree_idx=i,
                )

                if prev_node is not None:
                    prev_node.children.append(node)
                else:
                    root = node

                tree_nodes.append(node)
                prev_node = node

            if root is None:
                # Fallback: single node with all tokens
                root = TreeNode(
                    node_id=self._new_node_id(),
                    token_ids=valid_resp_ids,
                    log_probs=valid_log_probs,
                    step_text=response_text,
                    token_start=0,
                    token_end=valid_resp_len,
                    tree_idx=i,
                )
                tree_nodes = [root]

            # Determine finish_reason for the leaf node
            leaf_node = tree_nodes[-1]
            eos_token_id = self.tokenizer.eos_token_id
            if truncated_by_max_steps:
                leaf_node.finish_reason = "length"
            elif find_repeated_patterns(
                response_text,
                pattern_length=self.repetition_pattern_length,
                threshold=self.repetition_threshold,
            ):
                leaf_node.finish_reason = "repetition"
            elif eos_token_id is not None and valid_resp_ids and valid_resp_ids[-1] == eos_token_id:
                leaf_node.finish_reason = "stop"
            else:
                # response_length exhausted without EOS
                leaf_node.finish_reason = "length"

            tree = SearchTree(tree_idx=i, root=root, all_nodes=tree_nodes)
            self.trees.append(tree)

        # Map external PRM scores from template onto original-chain nodes
        self._map_ext_prm_to_nodes()

        return self.trees

    def _map_ext_prm_to_nodes(self) -> None:
        """Map external PRM step rewards from the template onto TreeNode.ext_prm_scores.

        The template stores per-rollout ``[(pos, score), ...]`` where ``pos`` is
        a token position in the original response.  Each ``(pos, score)`` is
        assigned to the node whose ``[token_start, token_end)`` range contains
        ``pos``.  Only original-chain nodes (``is_forked=False``) are matched;
        forked nodes must be evaluated separately via ``evaluate_branch_ext_prm``.
        """
        for tree in self.trees:
            tidx = tree.tree_idx
            # Collect all external PRM keys
            for key in self._non_tensor_batch_template:
                if not key.endswith("_step_reward") or key == "treerl_step_reward":
                    continue

                # e.g. key="format_step_reward" → prm_name="format"
                prm_name = key[: -len("_step_reward")]

                vals = self._non_tensor_batch_template[key]
                raw_scores = None
                if isinstance(vals, np.ndarray) and tidx < len(vals):
                    raw_scores = vals[tidx]
                elif isinstance(vals, list) and tidx < len(vals):
                    raw_scores = vals[tidx]

                if raw_scores is None or not isinstance(raw_scores, (list, tuple)):
                    continue

                # Build a lookup: original-chain nodes sorted by token_start
                orig_nodes = [n for n in tree.all_nodes if not n.is_forked]
                orig_nodes.sort(key=lambda n: n.token_start)

                for pos, score in raw_scores:
                    pos = int(pos)
                    # Find the node containing this position
                    for node in orig_nodes:
                        if node.token_start <= pos < node.token_end:
                            if prm_name == "fol" and self._should_penalize_step_format(node.step_text):
                                score = self.penalty_score
                            node.ext_prm_scores[prm_name] = float(score)
                            break

    # ------------------------------------------------------------------
    # Step 2: Select Forking Points
    # ------------------------------------------------------------------

    def select_forking_points(self, top_n: Optional[int] = None) -> List[Tuple[SearchTree, TreeNode, int]]:
        """Select the Top-N highest entropy steps *per tree* as forking points.

        Each node (step) is scored by the max token entropy it contains.
        Selection is done independently per tree so that every tree gets
        a chance to be expanded (aligned with the original TreeRL impl).

        Returns list of (tree, node, token_idx_of_max_entropy) tuples.
        The token_idx is informational only; downstream fork uses the full node.
        """
        top_n = top_n or self.tree_top_n

        selected = []

        for tree in self.trees:
            # Collect per-node (step) candidates for this tree
            # Use max token entropy within each node as the node's score
            candidates = []  # (max_entropy, node, t_idx_of_max)
            seen_nodes = set()

            for leaf in tree.all_leaves:
                path = leaf.path_from_root()
                num_path_nodes = len(path)
                # Mask the last mask_tail_ratio fraction of steps
                mask_threshold = max(1, int(num_path_nodes * (1 - self.mask_tail_ratio)))

                for step_idx, path_node in enumerate(path):
                    # if step_idx == 0:
                    #     continue  # skip root node
                    if step_idx >= mask_threshold:
                        continue  # mask tail steps
                    if path_node.node_id in seen_nodes:
                        continue  # shared ancestor, already considered
                    if not path_node.log_probs:
                        continue
                    seen_nodes.add(path_node.node_id)

                    max_ent = -1.0
                    max_t_idx = 0
                    for t_idx, lp in enumerate(path_node.log_probs):
                        ent = -lp
                        if ent > max_ent:
                            max_ent = ent
                            max_t_idx = t_idx
                    candidates.append((max_ent, path_node, max_t_idx))

            # Sort by entropy descending and take per-tree top-N
            candidates.sort(key=lambda x: x[0], reverse=True)
            for entropy, node, t_idx in candidates[:top_n]:
                selected.append((tree, node, t_idx))

        return selected

    # ------------------------------------------------------------------
    # Step 3: Prepare Branch Inputs
    # ------------------------------------------------------------------

    def prepare_branch_inputs(
        self,
        forking_points: List[Tuple[SearchTree, TreeNode, int]],
    ) -> Tuple[DataProto, List[dict]]:
        """Construct input batch for branch continuation generation.

        For each forking point, builds input_ids = prompt + response_prefix_up_to_fork.

        Args:
            forking_points: List of (tree, node, token_offset) from select_forking_points.

        Returns:
            branch_batch: DataProto with input_ids/attention_mask for generation.
            fork_info: List of dicts with metadata for commit_branches.
        """
        input_ids_list = []
        attention_mask_list = []
        fork_info_list = []

        for tree, node, t_idx in forking_points:
            # FIXME: changed from token-level truncation to step-wise fork.
            #   Old code: prefix_response_ids.extend(path_node.token_ids[:t_idx + 1])
            #   If this causes errors, revert to the old line above.
            # Step-wise: include the full selected node in prefix, branch from its end.
            path = node.path_from_root()
            prefix_response_ids = []
            for path_node in path:
                prefix_response_ids.extend(path_node.token_ids)

            # Full input = prompt + prefix_response
            prompt_ids = self._prompt_ids_list[tree.tree_idx]
            full_input = prompt_ids + prefix_response_ids

            input_ids_list.append(full_input)
            attention_mask_list.append([1] * len(full_input))

            fork_info_list.append({
                "tree_idx": tree.tree_idx,
                "fork_node_id": node.node_id,
                "fork_token_idx": t_idx,
                "prefix_response_len": len(prefix_response_ids),
                "prompt_len": len(prompt_ids),
            })

        if not input_ids_list:
            return None, []

        # Pad to same length
        max_len = max(len(ids) for ids in input_ids_list)
        pad_token_id = self.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - len(ids)
            # Left-pad (consistent with verl convention)
            padded_input_ids.append([pad_token_id] * pad_len + ids)
            padded_attention_mask.append([0] * pad_len + mask)

        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long)

        # Build DataProto
        batch_dict = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
        }

        # Copy ALL non_tensor_batch fields from the template (not just hardcoded keys)
        non_tensor_batch = {}
        for key in self._non_tensor_batch_template:
            if key.endswith("_step_reward"):
                continue  # step rewards are tree-specific, not carried over
            vals = self._non_tensor_batch_template[key]
            # Replicate the value from the corresponding tree
            replicated = []
            for info in fork_info_list:
                tidx = info["tree_idx"]
                if isinstance(vals, np.ndarray) and tidx < len(vals):
                    replicated.append(vals[tidx])
                elif isinstance(vals, list) and tidx < len(vals):
                    replicated.append(vals[tidx])
                else:
                    replicated.append(vals[0] if len(vals) > 0 else None)
            non_tensor_batch[key] = np.array(replicated, dtype=object)

        from verl import DataProto
        branch_batch = DataProto.from_single_dict(batch_dict)
        branch_batch.non_tensor_batch = non_tensor_batch
        branch_batch.meta_info = {
            "eos_token_id": self._meta_info.get("eos_token_id", self.tokenizer.eos_token_id),
            "pad_token_id": self._meta_info.get("pad_token_id", self.tokenizer.pad_token_id),
            "recompute_log_prob": False,
            "do_sample": True,
            # Signal the Agent Loop to use input_ids directly as the prompt
            # instead of re-tokenizing from raw_prompt (continuation mode)
            "continuation_mode": True,
        }

        return branch_batch, fork_info_list

    # ------------------------------------------------------------------
    # Step 4: Commit Branches
    # ------------------------------------------------------------------

    def commit_branches(
        self,
        branch_output: DataProto,
        fork_info_list: List[dict],
    ) -> None:
        """Attach generated branch responses back to the trees.

        Each branch output becomes a new leaf path from the forking point.
        The branch response is split into steps and new TreeNodes are created.

        Args:
            branch_output: DataProto from generate_sequences with branch continuations.
            fork_info_list: Metadata from prepare_branch_inputs (one per forking point).
        """
        responses = branch_output.batch["responses"]           # (num_branches, resp_len)
        attention_mask = branch_output.batch["attention_mask"]  # (num_branches, total_len)

        # Determine how many branches per fork point
        num_forks = len(fork_info_list)
        total_branches = responses.shape[0]
        branches_per_fork = total_branches // num_forks if num_forks > 0 else 0

        # log probs from branch generation
        log_probs_key = "rollout_log_probs" if "rollout_log_probs" in branch_output.batch.keys() else "log_probs"
        if log_probs_key in branch_output.batch.keys():
            all_log_probs = branch_output.batch[log_probs_key]
        else:
            all_log_probs = torch.zeros_like(responses, dtype=torch.float32)

        for branch_idx in range(total_branches):
            fork_idx = branch_idx // branches_per_fork if branches_per_fork > 0 else 0
            fork_idx = min(fork_idx, num_forks - 1)
            info = fork_info_list[fork_idx]

            tree = self.trees[info["tree_idx"]]
            fork_node_id = info["fork_node_id"]
            fork_token_idx = info["fork_token_idx"]

            # Find the fork node
            fork_node = None
            for n in tree.all_nodes:
                if n.node_id == fork_node_id:
                    fork_node = n
                    break
            if fork_node is None:
                continue

            # Get valid response tokens
            resp_len = responses.shape[1]
            prompt_and_prefix_len = attention_mask.shape[1] - resp_len
            resp_mask = attention_mask[branch_idx, prompt_and_prefix_len:]
            valid_resp_len = int(resp_mask.sum().item())
            valid_resp_ids = responses[branch_idx, :valid_resp_len].tolist()
            valid_lps = all_log_probs[branch_idx, :valid_resp_len].tolist()

            if not valid_resp_ids:
                continue

            # Split branch response into steps.
            # XML path: char→token mapping; Delimiter path: token-level search.
            response_text = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)
            from verl.utils.step_splitter import (split_by_xml_step_tags,
                                                  split_tokens_by_delimiter)

            xml_steps = split_by_xml_step_tags(response_text) if self.use_xml else []
            if xml_steps:
                step_ranges: list[tuple[int, int, str]] = []
                _prev = 0
                for step_text, _cs, char_end in xml_steps:
                    text_up_to_end = response_text[:char_end]
                    toks_up = self.tokenizer.encode(text_up_to_end, add_special_tokens=False)
                    tok_end = min(len(toks_up), valid_resp_len)
                    step_ranges.append((_prev, tok_end, step_text))
                    _prev = tok_end
            else:
                step_ranges = split_tokens_by_delimiter(
                    valid_resp_ids, self.tokenizer
                )

            # Create a new branch starting from the fork point.
            prev_node = fork_node
            fork_depth = len(fork_node.path_from_root())
            branch_step_count = 0
            truncated_by_max_steps = False

            for step_token_start, step_token_end, step_text in step_ranges:
                if step_token_start >= step_token_end:
                    continue

                # max_steps_per_path truncation (fork depth + new steps)
                if fork_depth + branch_step_count >= self.max_steps_per_path:
                    truncated_by_max_steps = True
                    break

                step_ids = valid_resp_ids[step_token_start:step_token_end]
                step_lps = valid_lps[step_token_start:step_token_end]

                new_node = TreeNode(
                    node_id=self._new_node_id(),
                    token_ids=step_ids,
                    log_probs=step_lps,
                    parent=prev_node,
                    step_text=step_text,
                    token_start=step_token_start,
                    token_end=step_token_end,
                    tree_idx=info["tree_idx"],
                    is_forked=True,
                )

                prev_node.children.append(new_node)
                tree.all_nodes.append(new_node)
                prev_node = new_node
                branch_step_count += 1

            # Determine finish_reason for the branch leaf
            if prev_node != fork_node:  # at least one node was added
                eos_token_id = self.tokenizer.eos_token_id
                if truncated_by_max_steps:
                    prev_node.finish_reason = "length"
                elif find_repeated_patterns(
                    response_text,
                    pattern_length=self.repetition_pattern_length,
                    threshold=self.repetition_threshold,
                ):
                    prev_node.finish_reason = "repetition"
                elif eos_token_id is not None and valid_resp_ids and valid_resp_ids[-1] == eos_token_id:
                    prev_node.finish_reason = "stop"
                else:
                    prev_node.finish_reason = "length"

    def evaluate_branch_ext_prm(
        self,
        ext_prm_fns: Optional[dict] = None,
    ) -> None:
        """Evaluate external PRM scores for forked (branch) nodes.

        Original-chain nodes already have ext_prm_scores from
        ``_map_ext_prm_to_nodes``.  This method fills in scores for
        forked nodes so that every node on every leaf path has PRM data.

        Args:
            ext_prm_fns: Dict mapping prm_name → reward_fn.
                Each reward_fn has signature:
                ``(step_text: str, prompt_text: str, step_history: list[str], **kwargs) -> float``
                Same as StepRewardManager reward functions.
                If None, forked nodes get no external PRM scores (backward
                compatible — they simply won't contribute to the bigpool).
        """
        if not ext_prm_fns:
            return

        # Collect all PRM names from the original-chain nodes
        prm_names: set = set()
        for tree in self.trees:
            for node in tree.all_nodes:
                if not node.is_forked:
                    prm_names.update(node.ext_prm_scores.keys())

        if not prm_names:
            return

        # Only evaluate PRM types that we have functions for
        eval_prms = [p for p in prm_names if p in ext_prm_fns]
        if not eval_prms:
            return

        self.ext_prm_profile = {
            "tasks_total": 0,
            "fol_tasks": 0,
            "fol_judge_calls": 0,
            "fol_judge_prompt_tokens": 0,
            "fol_judge_completion_tokens": 0,
            "fol_judge_total_tokens": 0,
            "fol_translation_s_sum": 0.0,
            "fol_translation_s_max": 0.0,
            "fol_correct_loop_s_sum": 0.0,
            "fol_correct_loop_s_max": 0.0,
            "fol_z3_run_s_sum": 0.0,
            "fol_z3_run_s_max": 0.0,
            "fol_correction_llm_s_sum": 0.0,
            "fol_correction_llm_s_max": 0.0,
            "fol_correction_z3_s_sum": 0.0,
            "fol_correction_z3_s_max": 0.0,
            "fol_verify_step_s_sum": 0.0,
            "fol_verify_step_s_max": 0.0,
            "fol_prepare_trees": 0,
            "fol_prepare_unique": 0,
            "fol_prepare_failed": 0,
            "fol_prepare_s_sum": 0.0,
            "fol_prepare_s_max": 0.0,
        }
        tasks = []
        tree_inputs = []

        for tree in self.trees:
            tidx = tree.tree_idx
            # Get prompt text for this tree
            prompt_text = ""
            if self._prompt_ids_list and tidx < len(self._prompt_ids_list):
                prompt_text = self.tokenizer.decode(
                    self._prompt_ids_list[tidx], skip_special_tokens=True
                )

            # Get extra_info for this tree (e.g., fol_context/question/options)
            extra_info = {}
            if "extra_info" in self._non_tensor_batch_template:
                ei_vals = self._non_tensor_batch_template["extra_info"]
                if isinstance(ei_vals, np.ndarray) and tidx < len(ei_vals):
                    extra_info = ei_vals[tidx] or {}
                elif isinstance(ei_vals, list) and tidx < len(ei_vals):
                    extra_info = ei_vals[tidx] or {}

            tree_inputs.append((tree, tidx, prompt_text, extra_info))

        fol_shared_state_by_tree_idx = {}
        if "fol" in eval_prms and tree_inputs:
            from verl.utils.reward_score.fol import prepare_fol_shared_state

            fol_api_config = getattr(ext_prm_fns.get("fol"), "keywords", {}).get("api_config")

            def _fol_prepare_key(prompt_text: str, extra_info: dict) -> tuple[str, str]:
                return (
                    prompt_text,
                    json.dumps(extra_info or {}, sort_keys=True, default=str),
                )

            unique_prepare_inputs = {}
            tree_prepare_keys = {}
            for _, tidx, prompt_text, extra_info in tree_inputs:
                prepare_key = _fol_prepare_key(prompt_text, extra_info)
                tree_prepare_keys[tidx] = prepare_key
                unique_prepare_inputs.setdefault(prepare_key, (prompt_text, extra_info))

            self.ext_prm_profile["fol_prepare_trees"] = len(tree_inputs)
            self.ext_prm_profile["fol_prepare_unique"] = len(unique_prepare_inputs)

            def _prepare_one(item):
                prepare_key, (prompt_text, extra_info) = item
                start = time.perf_counter()
                shared_state = prepare_fol_shared_state(
                    prompt_text,
                    extra_info=extra_info,
                    api_config=fol_api_config,
                )
                return prepare_key, shared_state, time.perf_counter() - start

            prepared_states = {}

            def _record_prepare_result(prepare_key, shared_state, elapsed):
                prepared_states[prepare_key] = shared_state
                self.ext_prm_profile["fol_prepare_s_sum"] += elapsed
                self.ext_prm_profile["fol_prepare_s_max"] = max(
                    self.ext_prm_profile["fol_prepare_s_max"],
                    elapsed,
                )
                if shared_state is None:
                    self.ext_prm_profile["fol_prepare_failed"] += 1

            prepare_items = list(unique_prepare_inputs.items())
            if self.ext_prm_max_workers <= 1 or len(prepare_items) <= 1:
                for item in prepare_items:
                    _record_prepare_result(*_prepare_one(item))
            else:
                with ThreadPoolExecutor(max_workers=min(self.ext_prm_max_workers, len(prepare_items))) as executor:
                    futures = [executor.submit(_prepare_one, item) for item in prepare_items]
                    for future in as_completed(futures):
                        _record_prepare_result(*future.result())

            for _, tidx, _, _ in tree_inputs:
                fol_shared_state_by_tree_idx[tidx] = prepared_states.get(tree_prepare_keys[tidx])

        for tree, tidx, prompt_text, extra_info in tree_inputs:
            fol_shared_state = fol_shared_state_by_tree_idx.get(tidx)
            for node in tree.all_nodes:
                if not node.is_forked:
                    continue
                for prm_name in eval_prms:
                    if prm_name in node.ext_prm_scores:
                        continue
                    path = node.path_from_root()
                    step_history = [n.step_text for n in path[:-1]]
                    tasks.append(
                        (
                            node,
                            prm_name,
                            prompt_text,
                            step_history,
                            extra_info,
                            fol_shared_state,
                        )
                    )

        if not tasks:
            return
        self.ext_prm_profile["tasks_total"] = len(tasks)

        def _run_task(task):
            node, prm_name, prompt_text, step_history, extra_info, fol_shared_state = task
            reward_fn = ext_prm_fns[prm_name]
            if prm_name == "fol" and self._should_penalize_step_format(node.step_text):
                debug = None
                debug = {
                    "format_failed_closed": True,
                    "judge_usage": {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
                return node, prm_name, self.penalty_score, debug
            kwargs = {"extra_info": extra_info}
            if prm_name == "fol" and fol_shared_state is not None:
                kwargs["fol_shared_state"] = fol_shared_state
            score = reward_fn(node.step_text, prompt_text, step_history, **kwargs)
            debug = None
            if isinstance(score, dict):
                debug = score.get("debug", {})
                score_value = float(score.get("score", 0.0))
            else:
                score_value = float(score)
            return node, prm_name, score_value, debug

        def _record_profile(prm_name, debug):
            if prm_name != "fol" or not isinstance(debug, dict):
                return
            self.ext_prm_profile["fol_tasks"] += 1
            judge_usage = debug.get("judge_usage", {})
            if isinstance(judge_usage, dict):
                self.ext_prm_profile["fol_judge_calls"] += int(judge_usage.get("calls", 0) or 0)
                self.ext_prm_profile["fol_judge_prompt_tokens"] += int(judge_usage.get("prompt_tokens", 0) or 0)
                self.ext_prm_profile["fol_judge_completion_tokens"] += int(
                    judge_usage.get("completion_tokens", 0) or 0
                )
                self.ext_prm_profile["fol_judge_total_tokens"] += int(judge_usage.get("total_tokens", 0) or 0)

            timing_map = {
                "translation_s": "fol_translation_s",
                "correct_loop_s": "fol_correct_loop_s",
                "z3_run_s": "fol_z3_run_s",
                "correction_llm_s": "fol_correction_llm_s",
                "correction_z3_s": "fol_correction_z3_s",
                "verify_step_s": "fol_verify_step_s",
            }
            for debug_key, metric_prefix in timing_map.items():
                value = float(debug.get(debug_key, 0.0) or 0.0)
                self.ext_prm_profile[f"{metric_prefix}_sum"] += value
                self.ext_prm_profile[f"{metric_prefix}_max"] = max(
                    self.ext_prm_profile[f"{metric_prefix}_max"],
                    value,
                )

        if self.ext_prm_max_workers <= 1 or len(tasks) <= 1:
            for task in tasks:
                node, prm_name, score, debug = _run_task(task)
                node.ext_prm_scores[prm_name] = score
                _record_profile(prm_name, debug)
            return

        with ThreadPoolExecutor(max_workers=min(self.ext_prm_max_workers, len(tasks))) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for future in as_completed(futures):
                node, prm_name, score, debug = future.result()
                node.ext_prm_scores[prm_name] = score
                _record_profile(prm_name, debug)

    # ------------------------------------------------------------------
    # Step 5: Evaluate Leaves
    # ------------------------------------------------------------------

    def evaluate_leaves(self, compute_score_fn: Callable) -> None:
        """Evaluate all leaf nodes for correctness.

        Args:
            compute_score_fn: Function(data_source, solution_str, ground_truth, extra_info) -> float
                Should return 1.0 for correct, 0.0 for incorrect.
        """
        for tree in self.trees:
            tree_idx = tree.tree_idx
            for leaf in tree.all_leaves:
                # Build full response text
                response_ids = leaf.full_token_ids()
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                # Get ground truth from template
                data_source = None
                ground_truth = None
                extra_info = {}
                if "data_source" in self._non_tensor_batch_template:
                    vals = self._non_tensor_batch_template["data_source"]
                    if tree_idx < len(vals):
                        data_source = vals[tree_idx]
                if "reward_model" in self._non_tensor_batch_template:
                    vals = self._non_tensor_batch_template["reward_model"]
                    if tree_idx < len(vals):
                        rm = vals[tree_idx]
                        if isinstance(rm, dict):
                            ground_truth = rm.get("ground_truth")
                if "extra_info" in self._non_tensor_batch_template:
                    vals = self._non_tensor_batch_template["extra_info"]
                    if tree_idx < len(vals):
                        extra_info = vals[tree_idx] or {}

                try:
                    score = compute_score_fn(
                        data_source=data_source,
                        solution_str=response_text,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
                    if isinstance(score, dict):
                        score = score.get("score", 0.0)
                    leaf.correctness = float(score)
                except Exception as e:
                    import traceback
                    print(f"[TreeRL] WARNING: evaluate_leaves failed for tree {tree_idx}, "
                          f"node {leaf.node_id}: {e}\n{traceback.format_exc()}")
                    leaf.correctness = 0.0

    # ------------------------------------------------------------------
    # Step 6a: Leave-one-out normalization
    # ------------------------------------------------------------------
    # Ref: github/THUNLP/TreeRL tree_node.py:377 (leaf_normalize)

    def leaf_normalize(self) -> None:
        """Per-tree leave-one-out normalization on leaves.

        For each tree (prompt), for each leaf i with raw score R(l_i):
            R_hat(l_i) = R(l_i) - (1/(K-1)) * sum_{j!=i} R(l_j)
        where K is the number of leaves in that tree.

        Result stored in leaf.accumulated_value.

        Ref: THUNLP/TreeRL tree_node.py:377 — LOO is per-prompt, not global.
        """
        for tree in self.trees:
            leaves = tree.all_leaves

            if len(leaves) <= 1:
                for leaf in leaves:
                    leaf.accumulated_value = 0.0
                continue

            scores = [leaf.correctness if leaf.correctness is not None else 0.0
                      for leaf in leaves]
            total = sum(scores)
            K = len(scores)

            for i, leaf in enumerate(leaves):
                mean_others = (total - scores[i]) / (K - 1)
                leaf.accumulated_value = scores[i] - mean_others

            # inner_repetition_penalty: override degenerate leaves after normalization
            # Ref: THUNLP/TreeRL tree_node.py:392-395
            if self.inner_repetition_penalty:
                for leaf in leaves:
                    if leaf.finish_reason != "stop":
                        leaf.accumulated_value = -1.0

    # ------------------------------------------------------------------
    # Step 6b: Backpropagate
    # ------------------------------------------------------------------
    # Ref: github/THUNLP/TreeRL tree_node.py:401 (leaf_backpropagate)

    def backpropagate(self) -> None:
        """Bottom-up accumulation: propagate each leaf's normalized score to ancestors.

        After leaf_normalize, each leaf has accumulated_value = R_hat(l_i).
        Walk up from each leaf to root:
            ancestor.accumulated_value += leaf.accumulated_value
            ancestor.terminal_in_subtree += 1
        """
        for tree in self.trees:
            for leaf in tree.all_leaves:
                leaf.terminal_in_subtree = 1
                parent = leaf.parent
                while parent is not None:
                    parent.accumulated_value += leaf.accumulated_value
                    parent.terminal_in_subtree += 1
                    parent = parent.parent

    # ------------------------------------------------------------------
    # Step 6c: Per-tree step normalization
    # ------------------------------------------------------------------
    # Ref: github/THUNLP/TreeRL tree_node.py:421 (normalize_all_steps)

    def normalize_all_steps(self) -> None:
        """Per-tree step normalization: subtract token-weighted (or step-weighted) mean.

        Token mode (default):
            μ = Σ A(s_n)·|T(s_n)| / Σ |L(s_n)|·|T(s_n)|
        Step mode:
            μ = Σ A(s_n) / Σ |L(s_n)|
        Then: A(s_n) -= μ · |L(s_n)|

        This is baseline subtraction — makes the expected (token-weighted) advantage
        zero within each tree.

        Ref: THUNLP/TreeRL tree_node.py:421 — normalize_all_steps is per-tree.
        """
        if self.overall_norm_style == "none":
            return

        for tree in self.trees:
            # Collect all nodes with terminal_in_subtree > 0
            all_steps = [node for node in tree.all_nodes
                         if node.terminal_in_subtree > 0]

            if not all_steps:
                continue

            if self.overall_norm_style == "token":
                num = sum(node.accumulated_value * len(node.token_ids)
                          for node in all_steps)
                den = sum(node.terminal_in_subtree * len(node.token_ids)
                          for node in all_steps)
            else:  # "step"
                num = sum(node.accumulated_value for node in all_steps)
                den = sum(node.terminal_in_subtree for node in all_steps)

            mean = num / den if den != 0 else 0.0

            for node in all_steps:
                node.accumulated_value -= mean * node.terminal_in_subtree

    # ------------------------------------------------------------------
    # Step 6d: Reweight (optional)
    # ------------------------------------------------------------------
    # Ref: github/THUNLP/TreeRL tree_node.py:545-579
    #      (selected_backpropagate + compute_weighted_update)

    def reweight_steps(self) -> None:
        """Optional: divide accumulated_value by sqrt/uniform of selected terminal count.

        Only active when tree_use_weighted_value=True.
        Styles: "sqrt" -> /sqrt(n), "uniform" -> /n, "original" -> no-op.
        """
        if not self.use_weighted_value:
            return

        # Phase 1: count selected terminals per node (selected_backpropagate)
        # In our case all leaves are selected, so selected == terminal.
        for tree in self.trees:
            for leaf in tree.all_leaves:
                node = leaf
                while node is not None:
                    node.selected_terminal_in_subtree += 1
                    node = node.parent

        # Phase 2: apply reweight recursively (compute_weighted_update)
        def _reweight(node):
            if node.selected_terminal_in_subtree == 0:
                return
            if self.weighted_value_style == "sqrt":
                node.accumulated_value /= math.sqrt(node.selected_terminal_in_subtree)
            elif self.weighted_value_style == "uniform":
                node.accumulated_value /= node.selected_terminal_in_subtree
            # "original" = no-op
            for child in node.children:
                _reweight(child)

        for tree in self.trees:
            _reweight(tree.root)

    # ------------------------------------------------------------------
    # Step 7: Compute Step Rewards
    # ------------------------------------------------------------------
    # Ref: github/THUNLP/TreeRL parallel_mcts.py:1603 (path_from_root_to_node)

    def compute_step_rewards(self) -> None:
        """Compute per-step reward from accumulated_value.

        Phase 1: V(s_n) = A(s_n) / |L(s_n)|
        Phase 2: step reward per mode:
            "la"         (ref code default): V(child) - V(parent)
            "ga_la"      (paper formula):    [V - V(root)] + [V - V(parent)]
            "ga":                            V - V(root)
            "value_only":                    V directly
        """
        mode = self.step_reward_mode

        for tree in self.trees:
            # Phase 1: compute V(s_n) for all nodes
            for node in tree.all_nodes:
                if node.terminal_in_subtree > 0:
                    node.value = node.accumulated_value / node.terminal_in_subtree
                else:
                    node.value = 0.0

            root_value = tree.root.value

            # Phase 2: compute step reward
            for node in tree.all_nodes:
                parent_value = node.parent.value if node.parent is not None else root_value
                la = node.value - parent_value       # Local Advantage
                ga = node.value - root_value         # Global Advantage

                if mode == "ga_la":
                    node.reward = ga + la
                elif mode == "la":
                    node.reward = la
                elif mode == "ga":
                    node.reward = ga
                elif mode == "value_only":
                    node.reward = node.value
                else:
                    node.reward = la  # fallback to ref code default

    # ------------------------------------------------------------------
    # Step 8: Build Flat Batch
    # ------------------------------------------------------------------

    def build_flat_batch(self, original_output: DataProto) -> DataProto:
        """Flatten all leaf paths into a standard DataProto batch.

        Each leaf path (root -> ... -> leaf) becomes one training sample.
        Step rewards are stored as List[(token_pos, score)] in non_tensor_batch,
        compatible with the step_gdpo advantage estimator format.

        External PRM scores are read from ``TreeNode.ext_prm_scores`` (per-node
        storage populated by ``_map_ext_prm_to_nodes`` + ``evaluate_branch_ext_prm``).
        Each node contributes its own score — no cross-path duplication of stale data.

        Args:
            original_output: The original rollout DataProto (for shape/format reference).

        Returns:
            flat_batch: DataProto with all leaf paths and TreeRL step rewards.
        """
        all_paths = []
        for tree in self.trees:
            for leaf in tree.all_leaves:
                path = leaf.path_from_root()
                all_paths.append((tree, path))

        if not all_paths:
            return original_output

        # Collect data for each path
        all_response_ids = []
        all_response_log_probs = []
        all_step_rewards = []  # List[(pos, score)] per path

        for tree, path in all_paths:
            # Concatenate all token_ids and log_probs along the path
            resp_ids = []
            resp_lps = []
            step_rewards = []

            token_offset = 0
            for node in path:
                resp_ids.extend(node.token_ids)
                resp_lps.extend(node.log_probs)

                # Step reward at the last token of this node
                if node.token_ids:
                    step_end_pos = token_offset + len(node.token_ids) - 1
                    step_rewards.append((step_end_pos, node.reward))

                token_offset += len(node.token_ids)

            all_response_ids.append(resp_ids)
            all_response_log_probs.append(resp_lps)
            all_step_rewards.append(step_rewards)

        # Determine max response length and pad
        max_resp_len = max(len(ids) for ids in all_response_ids)
        # Use original response length if larger
        if "responses" in original_output.batch.keys():
            max_resp_len = max(max_resp_len, original_output.batch["responses"].shape[1])

        pad_token_id = self.tokenizer.pad_token_id or 0
        num_paths = len(all_paths)

        # Build padded tensors
        responses = torch.full((num_paths, max_resp_len), pad_token_id, dtype=torch.long)
        response_masks = torch.zeros(num_paths, max_resp_len, dtype=torch.float32)

        for i, resp_ids in enumerate(all_response_ids):
            length = min(len(resp_ids), max_resp_len)
            responses[i, :length] = torch.tensor(resp_ids[:length], dtype=torch.long)
            response_masks[i, :length] = 1.0

        # Build prompts tensor (replicate from original trees)
        if "prompts" in original_output.batch.keys():
            orig_prompts = original_output.batch["prompts"]
            prompt_len = orig_prompts.shape[1]
            prompts = torch.zeros(num_paths, prompt_len, dtype=torch.long)
            for i, (tree, _) in enumerate(all_paths):
                if tree.tree_idx < orig_prompts.shape[0]:
                    prompts[i] = orig_prompts[tree.tree_idx]
        else:
            prompts = None

        # Build attention mask
        total_len = (prompt_len + max_resp_len) if prompts is not None else max_resp_len
        attention_mask = torch.zeros(num_paths, total_len, dtype=torch.long)
        if prompts is not None:
            for i, (tree, _) in enumerate(all_paths):
                if tree.tree_idx < original_output.batch["attention_mask"].shape[0]:
                    orig_attn = original_output.batch["attention_mask"][tree.tree_idx, :prompt_len]
                    attention_mask[i, :prompt_len] = orig_attn
            attention_mask[:, prompt_len:prompt_len + max_resp_len] = response_masks.long()
        else:
            attention_mask[:, :max_resp_len] = response_masks.long()

        # Build rm_scores from leaf correctness (outcome reward).
        # Places the score at the last valid response token for each path,
        # matching the format that RewardLoopManager.compute_rm_score() produces.
        rm_scores = torch.zeros(num_paths, max_resp_len, dtype=torch.float32)
        for i, (tree, path) in enumerate(all_paths):
            leaf = path[-1]
            score = leaf.correctness if leaf.correctness is not None else 0.0
            valid_len = len(all_response_ids[i])
            if valid_len > 0:
                pos = min(valid_len - 1, max_resp_len - 1)
                rm_scores[i, pos] = float(score)

        # Build input_ids = cat(prompts, responses) and position_ids from attention_mask,
        # required by left_right_2_no_padding in the training loop.
        if prompts is not None:
            input_ids = torch.cat([prompts, responses], dim=1)
        else:
            input_ids = responses

        from verl.utils.model import compute_position_id_with_mask
        position_ids = compute_position_id_with_mask(attention_mask)

        # Build batch dict
        batch_dict = {
            "responses": responses,
            "attention_mask": attention_mask,
            "response_mask": response_masks,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "rm_scores": rm_scores,
        }
        if prompts is not None:
            batch_dict["prompts"] = prompts

        # Build non_tensor_batch — replicate ALL keys from the template
        non_tensor_batch = {}

        for key in self._non_tensor_batch_template:
            if key.endswith("_step_reward"):
                continue  # external PRM step rewards handled below
            vals = self._non_tensor_batch_template[key]
            replicated = []
            for tree, _ in all_paths:
                tidx = tree.tree_idx
                if isinstance(vals, np.ndarray) and tidx < len(vals):
                    replicated.append(vals[tidx])
                elif isinstance(vals, list) and tidx < len(vals):
                    replicated.append(vals[tidx])
                else:
                    replicated.append(None)
            non_tensor_batch[key] = np.array(replicated, dtype=object)

        # TreeRL step rewards (computed by this TreeManager)
        non_tensor_batch["treerl_step_reward"] = np.array(all_step_rewards, dtype=object)
        non_tensor_batch["num_steps"] = np.array(
            [len(sr) for sr in all_step_rewards], dtype=np.int32
        )

        # External PRM step rewards — read from TreeNode.ext_prm_scores.
        # Each node on the path contributes its score (if it has one) at the
        # last token position of that node, identical to how treerl_step_reward
        # is built.  This covers both original-chain nodes (from
        # _map_ext_prm_to_nodes) and forked nodes (from evaluate_branch_ext_prm).
        #
        # Collect all expected PRM names, including the degenerate case where
        # every per-node external PRM list is empty for this batch.
        all_prm_names: set = set()
        for key in self._non_tensor_batch_template:
            if key.endswith("_step_reward") and key != "treerl_step_reward":
                all_prm_names.add(key[: -len("_step_reward")])
        for tree in self.trees:
            for node in tree.all_nodes:
                all_prm_names.update(node.ext_prm_scores.keys())

        for prm_name in all_prm_names:
            key = f"{prm_name}_step_reward"
            nid_key = f"{prm_name}_step_node_ids"
            per_path_scores = []
            per_path_node_ids = []
            for path_idx, (tree, path) in enumerate(all_paths):
                scores = []
                node_ids = []
                token_offset = 0
                for node in path:
                    if node.token_ids and prm_name in node.ext_prm_scores:
                        step_end_pos = token_offset + len(node.token_ids) - 1
                        scores.append((step_end_pos, node.ext_prm_scores[prm_name]))
                        node_ids.append(node.node_id)
                    token_offset += len(node.token_ids)
                per_path_scores.append(scores)
                per_path_node_ids.append(node_ids)
            non_tensor_batch[key] = np.array(per_path_scores, dtype=object)
            # Parallel node_id list for dedup in _tree_dedup_bigpool_normalize.
            # node_id is globally unique within a TreeManager instance.
            non_tensor_batch[nid_key] = np.array(per_path_node_ids, dtype=object)

        # Build DataProto
        from verl import DataProto
        flat_batch = DataProto.from_single_dict(batch_dict)
        flat_batch.non_tensor_batch = non_tensor_batch
        flat_batch.meta_info = dict(self._meta_info)

        return flat_batch

    # ------------------------------------------------------------------
    # Logging & Visualization
    # ------------------------------------------------------------------

    def log_config_summary(self, M: int) -> str:
        """Return a summary string of EPTree config and estimated total paths.

        Args:
            M: Number of initial rollouts (rollout.n).
        """
        N = self.tree_top_n
        T = self.tree_branches
        L = self.tree_rounds
        # Each tree: 1 original + N*T new branches per round
        # (simplified estimate for L=1; deeper rounds compound)
        leaves_per_tree = 1 + N * T * L
        total_paths = M * leaves_per_tree

        lines = [
            "=" * 60,
            " [TreeRL] EPTree Configuration Summary",
            "=" * 60,
            f"  M (initial rollouts)     = {M}",
            f"  N (top-N fork points)    = {N}",
            f"  T (branches per fork)    = {T}",
            f"  L (expansion rounds)     = {L}",
            f"  mask_tail_ratio          = {self.mask_tail_ratio}",
            f"  max_steps_per_path       = {self.max_steps_per_path}",
            f"  inner_repetition_penalty = {self.inner_repetition_penalty}",
            f"  repetition_pattern_len   = {self.repetition_pattern_length}",
            f"  repetition_threshold     = {self.repetition_threshold}",
            "-" * 60,
            f"  Expected leaves/tree     = {leaves_per_tree}",
            f"  Expected total paths     = {total_paths}",
            "=" * 60,
        ]
        summary = "\n".join(lines)
        print(summary)
        return summary

    def format_tree_ascii(self, tree_idx: int = 0, max_text_len: int = 40) -> str:
        """Render a single tree as ASCII art.

        Args:
            tree_idx: Which tree to visualize (0-indexed).
            max_text_len: Max characters of step_text to show per node.

        Returns:
            ASCII string representation.
        """
        if tree_idx >= len(self.trees):
            return f"[TreeRL] Tree {tree_idx} not found (total: {len(self.trees)})"

        tree = self.trees[tree_idx]

        def _fmt_node(node: TreeNode) -> str:
            text_preview = node.step_text[:max_text_len].replace("\n", "\\n")
            if len(node.step_text) > max_text_len:
                text_preview += "..."
            corr_str = ""
            if node.correctness is not None:
                corr_str = f" ✓" if node.correctness > 0.5 else f" ✗"
            forked_str = " [forked]" if node.is_forked else ""
            return (
                f"node_{node.node_id} "
                f"(V={node.value:.2f}, R={node.reward:.2f}{corr_str}{forked_str}) "
                f'"{text_preview}"'
            )

        lines = [f"[TreeRL] Tree {tree_idx} (leaves={tree.num_leaves}):"]

        def _walk(node: TreeNode, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + _fmt_node(node))
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                _walk(child, child_prefix, i == len(node.children) - 1)

        _walk(tree.root)
        result = "\n".join(lines)
        print(result)
        return result

    def log_sample_trajectory(self, tree_idx: int = 0, leaf_idx: int = 0) -> str:
        """Print one full decoded leaf path for inspection.

        Args:
            tree_idx: Which tree.
            leaf_idx: Which leaf within that tree (0-indexed).

        Returns:
            The trajectory string.
        """
        if tree_idx >= len(self.trees):
            return f"[TreeRL] Tree {tree_idx} not found"

        tree = self.trees[tree_idx]
        leaves = tree.all_leaves
        if leaf_idx >= len(leaves):
            return f"[TreeRL] Leaf {leaf_idx} not found in tree {tree_idx} (total: {len(leaves)})"

        leaf = leaves[leaf_idx]
        path = leaf.path_from_root()

        lines = [
            "=" * 60,
            f" [TreeRL] Sample Trajectory: Tree {tree_idx}, Leaf {leaf_idx}",
            f" Correctness: {leaf.correctness}",
            f" Path length: {len(path)} nodes",
            "=" * 60,
        ]
        for i, node in enumerate(path):
            marker = "→" if not node.is_forked else "⑂"
            corr_str = ""
            if node.correctness is not None:
                corr_str = " ✓" if node.correctness > 0.5 else " ✗"
            lines.append(
                f"  {marker} Step {i} (V={node.value:.3f}, R={node.reward:.3f}{corr_str}):"
            )
            step_text = node.step_text if node.step_text else "(no text)"
            for text_line in step_text.split("\n")[:5]:  # limit to 5 lines
                lines.append(f"    {text_line}")
            if len(node.step_text.split("\n")) > 5:
                lines.append(f"    ... ({len(node.step_text.split(chr(10)))} lines total)")
        lines.append("=" * 60)

        result = "\n".join(lines)
        print(result)
        return result

    # ------------------------------------------------------------------
    # Convenience: Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        rollout_output: DataProto,
        generate_fn: Callable,
        compute_score_fn: Callable,
        ext_prm_fns: Optional[dict] = None,
    ) -> DataProto:
        """Run the complete EPTree pipeline.

        Args:
            rollout_output: Initial rollout DataProto (M responses).
            generate_fn: Function to generate continuations (e.g., async_rollout_manager.generate_sequences).
            compute_score_fn: Function to evaluate response correctness.
            ext_prm_fns: Optional dict mapping prm_name → reward_fn for
                evaluating external PRM on forked nodes.  Each fn has signature
                ``(step_text, prompt_text, step_history, **kw) -> float``.
                If None, forked nodes get no external PRM scores.

        Returns:
            flat_batch: DataProto with all leaf paths and TreeRL step rewards.
        """
        self.pipeline_profile = {
            "initialize_s": 0.0,
            "branch_generation_s": 0.0,
            "ext_prm_eval_s": 0.0,
            "evaluate_leaves_s": 0.0,
            "normalize_backprop_s": 0.0,
            "build_flat_batch_s": 0.0,
        }

        # 1. Initialize trees (also maps external PRM scores to original-chain nodes)
        start = time.perf_counter()
        self.initialize_trees(rollout_output)
        self.pipeline_profile["initialize_s"] += time.perf_counter() - start

        # 2. Iterative expansion
        for round_idx in range(self.tree_rounds):
            start = time.perf_counter()
            forking_points = self.select_forking_points()
            if not forking_points:
                self.pipeline_profile["branch_generation_s"] += time.perf_counter() - start
                break

            branch_batch, fork_info = self.prepare_branch_inputs(forking_points)
            if branch_batch is None:
                self.pipeline_profile["branch_generation_s"] += time.perf_counter() - start
                break

            # Repeat for T branches per fork
            if self.tree_branches > 1:
                branch_batch = branch_batch.repeat(
                    repeat_times=self.tree_branches, interleave=True
                )
                # Must match interleave order: [A,A,B,B] not [A,B,A,B]
                fork_info = [info for info in fork_info for _ in range(self.tree_branches)]

            branch_output = generate_fn(branch_batch)
            self.commit_branches(branch_output, fork_info)
            self.pipeline_profile["branch_generation_s"] += time.perf_counter() - start

        # 2.5. Evaluate external PRM on forked nodes (if evaluator provided)
        start = time.perf_counter()
        self.evaluate_branch_ext_prm(ext_prm_fns)
        self.pipeline_profile["ext_prm_eval_s"] += time.perf_counter() - start

        # 3. Evaluate + normalize + backpropagate + step-norm + reweight + step rewards
        #    Ref: github/THUNLP/TreeRL tree_node.py build_into_tree_format (line 352-368)
        start = time.perf_counter()
        self.evaluate_leaves(compute_score_fn)
        self.pipeline_profile["evaluate_leaves_s"] += time.perf_counter() - start

        # Log finish_reason distribution before normalization
        reasons = [leaf.finish_reason for tree in self.trees for leaf in tree.all_leaves]
        reason_counts = Counter(reasons)
        print(f"[TreeRL] Leaf finish reasons: {dict(reason_counts)}")

        start = time.perf_counter()
        self.leaf_normalize()
        self.backpropagate()
        self.normalize_all_steps()
        self.reweight_steps()           # no-op if tree_use_weighted_value=False
        self.compute_step_rewards()
        self.pipeline_profile["normalize_backprop_s"] += time.perf_counter() - start

        # 4. Build flat batch
        start = time.perf_counter()
        flat_batch = self.build_flat_batch(rollout_output)
        self.pipeline_profile["build_flat_batch_s"] += time.perf_counter() - start
        return flat_batch
