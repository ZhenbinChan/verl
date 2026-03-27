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
top-N most uncertain tokens, then use the tree structure to compute process
supervision signals (GA + LA) / sqrt(n) for RL training.

Reference: Algorithm 1 in "TreeRL: LLM Reinforcement Learning with On-Policy Tree Search"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch


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
    value: float = 0.0          # V(sn) = correct_leaves / total_leaves
    reward: float = 0.0         # R(sn) = (GA + LA) / sqrt(|L(sn)|)
    correctness: Optional[float] = None  # 0/1 for leaf nodes, None for internal

    # Metadata
    tree_idx: int = 0           # which tree this node belongs to
    is_forked: bool = False     # whether this node was created by forking

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
# Default split function (matches StepRewardManager's default)
# ---------------------------------------------------------------------------

def default_split_fn(response_text: str) -> list[str]:
    """Default step splitter: split by double newline (same as StepRewardManager)."""
    if not response_text:
        return [""]
    return response_text.split("\n\n")


# ---------------------------------------------------------------------------
# TreeManager: Coordinates the full EPTree pipeline
# ---------------------------------------------------------------------------

class TreeManager:
    """Manages the EPTree search process within the RL training loop.

    This class coordinates:
    1. Initializing trees from rollout responses
    2. Selecting forking points based on token entropy
    3. Preparing branch inputs for continuation generation
    4. Committing branch outputs back to the tree
    5. Evaluating leaves and backpropagating values
    6. Computing TreeRL step rewards: R(sn) = (GA + LA) / sqrt(|L(sn)|)
    7. Flattening all leaf paths into a standard DataProto batch
    """

    def __init__(self, config, tokenizer, split_fn: Optional[Callable] = None):
        """
        Args:
            config: Trainer config (OmegaConf), needs tree_rounds, tree_top_n, tree_branches, etc.
            tokenizer: HuggingFace tokenizer for encoding/decoding.
            split_fn: Step splitter function. Defaults to split by "\\n\\n".
        """
        self.config = config
        self.tokenizer = tokenizer
        self.split_fn = split_fn or default_split_fn

        # EPTree parameters from config
        self.tree_rounds = config.get("tree_rounds", 1)       # L
        self.tree_top_n = config.get("tree_top_n", 2)         # N
        self.tree_branches = config.get("tree_branches", 2)   # T
        self.mask_tail_ratio = config.get("tree_mask_tail_ratio", 0.1)  # mask末尾tokens

        self.trees: List[SearchTree] = []
        self._node_counter = 0
        # Store prompt info for branch construction
        self._prompt_ids_list: List[List[int]] = []
        self._prompt_lengths: List[int] = []
        self._meta_info = {}
        self._non_tensor_batch_template = {}

    def _new_node_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

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

            # Split into steps
            steps = self.split_fn(response_text)

            # Build chain of TreeNodes
            root = None
            prev_node = None
            token_cursor = 0
            tree_nodes = []

            for step_idx, step_text in enumerate(steps):
                if not step_text and step_idx > 0:
                    continue  # skip empty segments

                # Find token span for this step
                text_up_to_here = self.split_fn.__name__  # dummy
                # Encode step to find how many tokens it covers
                prefix_text = ("\n\n".join(steps[:step_idx + 1])
                               if step_idx > 0 else steps[0])
                prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                step_token_end = min(len(prefix_tokens), valid_resp_len)
                step_token_start = token_cursor

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
                token_cursor = step_token_end

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

            tree = SearchTree(tree_idx=i, root=root, all_nodes=tree_nodes)
            self.trees.append(tree)

        return self.trees

    # ------------------------------------------------------------------
    # Step 2: Select Forking Points
    # ------------------------------------------------------------------

    def select_forking_points(self, top_n: Optional[int] = None) -> List[Tuple[SearchTree, TreeNode, int]]:
        """Select the Top-N highest entropy tokens across all trees as forking points.

        Returns list of (tree, node, token_offset_within_node) tuples.
        Masks tokens near the end of sequences (last mask_tail_ratio fraction).
        """
        top_n = top_n or self.tree_top_n

        # Collect all (entropy, tree, node, token_idx) candidates
        candidates = []
        for tree in self.trees:
            for node in tree.all_nodes:
                if not node.is_leaf:
                    continue  # only fork from current leaf paths
                # Walk the path and collect entropy scores
                path = node.path_from_root()
                total_tokens = sum(len(n.token_ids) for n in path)
                mask_threshold = int(total_tokens * (1 - self.mask_tail_ratio))

                token_offset = 0
                for path_node in path:
                    for t_idx, lp in enumerate(path_node.log_probs):
                        global_pos = token_offset + t_idx
                        if global_pos >= mask_threshold:
                            continue  # mask tail tokens
                        if global_pos == 0:
                            continue  # skip first token
                        entropy = -lp
                        candidates.append((entropy, tree, path_node, t_idx))
                    token_offset += len(path_node.token_ids)

        if not candidates:
            return []

        # Sort by entropy descending and take top-N
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate: don't fork the same (node, token_idx) twice
        seen = set()
        selected = []
        for entropy, tree, node, t_idx in candidates:
            key = (node.node_id, t_idx)
            if key in seen:
                continue
            seen.add(key)
            selected.append((tree, node, t_idx))
            if len(selected) >= top_n:
                break

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
            # Build prefix: all tokens from root up to (node, t_idx)
            path = node.path_from_root()
            prefix_response_ids = []
            for path_node in path:
                if path_node.node_id == node.node_id:
                    # Include tokens up to and including t_idx
                    prefix_response_ids.extend(path_node.token_ids[:t_idx + 1])
                    break
                else:
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

        # Copy non_tensor_batch fields from the first tree's data
        non_tensor_batch = {}
        for key in ["data_source", "reward_model", "extra_info", "uid"]:
            if key in self._non_tensor_batch_template:
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

            # Decode and split into steps
            response_text = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)
            steps = self.split_fn(response_text)

            # Create a new branch starting from the fork point.
            # The fork_node's tokens up to fork_token_idx are shared;
            # we create a "split node" for the remainder + branch tokens.
            # For simplicity, we create a single new child chain under fork_node.

            prev_node = fork_node
            token_cursor = 0

            for step_idx, step_text in enumerate(steps):
                if not step_text and step_idx > 0:
                    continue

                prefix_text = ("\n\n".join(steps[:step_idx + 1])
                               if step_idx > 0 else steps[0])
                prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                step_token_end = min(len(prefix_tokens), valid_resp_len)
                step_token_start = token_cursor

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
                token_cursor = step_token_end

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
                except Exception:
                    leaf.correctness = 0.0

    # ------------------------------------------------------------------
    # Step 6: Backpropagate
    # ------------------------------------------------------------------

    def backpropagate(self) -> None:
        """Bottom-up value propagation: V(sn) = correct_leaves(sn) / total_leaves(sn).

        Paper Eq: V(sn) = (1/|L(sn)|) * sum_{l in L(sn)} 1(l is correct)
        """
        for tree in self.trees:
            # Process nodes bottom-up (reverse topological order)
            for node in reversed(tree.all_nodes):
                leaves = node.descendant_leaves()
                if not leaves:
                    node.value = 0.0
                    continue
                correct_count = sum(
                    1 for l in leaves
                    if l.correctness is not None and l.correctness > 0.5
                )
                node.value = correct_count / len(leaves)

    # ------------------------------------------------------------------
    # Step 7: Compute Step Rewards
    # ------------------------------------------------------------------

    def compute_step_rewards(self) -> None:
        """Compute TreeRL process reward for each node.

        Paper formula (Algorithm 1):
            R(sn) = [GA(sn) + LA(sn)] / sqrt(|L(sn)|)

        where:
            GA(sn) = V(sn) - V(root)       (Global Advantage)
            LA(sn) = V(sn) - V(parent(sn)) (Local Advantage)
            |L(sn)| = number of descendant leaves (reweight factor)
        """
        for tree in self.trees:
            root_value = tree.root.value

            for node in tree.all_nodes:
                parent_value = node.parent.value if node.parent is not None else root_value
                ga = node.value - root_value        # Global Advantage
                la = node.value - parent_value      # Local Advantage

                # Reweight factor: 1 / sqrt(|L(sn)|)
                num_descendant_leaves = len(node.descendant_leaves())
                reweight = 1.0 / math.sqrt(max(num_descendant_leaves, 1))

                node.reward = (ga + la) * reweight

    # ------------------------------------------------------------------
    # Step 8: Build Flat Batch
    # ------------------------------------------------------------------

    def build_flat_batch(self, original_output: DataProto) -> DataProto:
        """Flatten all leaf paths into a standard DataProto batch.

        Each leaf path (root -> ... -> leaf) becomes one training sample.
        Step rewards are stored as List[(token_pos, score)] in non_tensor_batch,
        compatible with the step_gdpo advantage estimator format.

        Args:
            original_output: The original rollout DataProto (for shape/format reference).

        Returns:
            flat_batch: DataProto with all leaf paths, ready for training.
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
        all_tree_indices = []

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
            all_tree_indices.append(tree.tree_idx)

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

        # Build batch dict
        batch_dict = {
            "responses": responses,
            "attention_mask": attention_mask,
            "response_mask": response_masks,
        }
        if prompts is not None:
            batch_dict["prompts"] = prompts

        # Build non_tensor_batch
        non_tensor_batch = {}

        # Replicate reward_model, data_source, extra_info, uid from original trees
        for key in ["data_source", "reward_model", "extra_info"]:
            if key in self._non_tensor_batch_template:
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

        # Assign uid: all paths from same tree share the same uid
        if "uid" in self._non_tensor_batch_template:
            orig_uids = self._non_tensor_batch_template["uid"]
            uids = []
            for tree, _ in all_paths:
                tidx = tree.tree_idx
                if isinstance(orig_uids, np.ndarray) and tidx < len(orig_uids):
                    uids.append(orig_uids[tidx])
                else:
                    uids.append(f"tree_{tidx}")
            non_tensor_batch["uid"] = np.array(uids, dtype=object)

        # TreeRL step rewards
        non_tensor_batch["treerl_step_reward"] = np.array(all_step_rewards, dtype=object)
        non_tensor_batch["num_steps"] = np.array(
            [len(sr) for sr in all_step_rewards], dtype=np.int32
        )

        # Propagate external PRM step reward keys (e.g., format_step_reward)
        for key in self._non_tensor_batch_template:
            if key.endswith("_step_reward") and key != "treerl_step_reward":
                vals = self._non_tensor_batch_template[key]
                replicated = []
                for tree, _ in all_paths:
                    tidx = tree.tree_idx
                    if isinstance(vals, np.ndarray) and tidx < len(vals):
                        replicated.append(vals[tidx])
                    elif isinstance(vals, list) and tidx < len(vals):
                        replicated.append(vals[tidx])
                    else:
                        replicated.append([])
                non_tensor_batch[key] = np.array(replicated, dtype=object)

        # Build DataProto
        from verl import DataProto
        flat_batch = DataProto.from_single_dict(batch_dict)
        flat_batch.non_tensor_batch = non_tensor_batch
        flat_batch.meta_info = dict(self._meta_info)

        return flat_batch

    # ------------------------------------------------------------------
    # Convenience: Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        rollout_output: DataProto,
        generate_fn: Callable,
        compute_score_fn: Callable,
    ) -> DataProto:
        """Run the complete EPTree pipeline.

        Args:
            rollout_output: Initial rollout DataProto (M responses).
            generate_fn: Function to generate continuations (e.g., async_rollout_manager.generate_sequences).
            compute_score_fn: Function to evaluate response correctness.

        Returns:
            flat_batch: DataProto with all leaf paths and TreeRL step rewards.
        """
        # 1. Initialize trees
        self.initialize_trees(rollout_output)

        # 2. Iterative expansion
        for round_idx in range(self.tree_rounds):
            forking_points = self.select_forking_points()
            if not forking_points:
                break

            branch_batch, fork_info = self.prepare_branch_inputs(forking_points)
            if branch_batch is None:
                break

            # Repeat for T branches per fork
            if self.tree_branches > 1:
                branch_batch = branch_batch.repeat(
                    repeat_times=self.tree_branches, interleave=True
                )
                fork_info = fork_info * self.tree_branches  # replicate info

            branch_output = generate_fn(branch_batch)
            self.commit_branches(branch_output, fork_info)

        # 3. Evaluate + backpropagate + compute rewards
        self.evaluate_leaves(compute_score_fn)
        self.backpropagate()
        self.compute_step_rewards()

        # 4. Build flat batch
        return self.build_flat_batch(rollout_output)
