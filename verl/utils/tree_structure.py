"""
MCTS-style search tree utilities for branching on high-entropy response steps.

This module provides a lightweight Node/SearchTree abstraction plus a TreeManager
that can be used from trainers to track per-prompt search trees. It also offers a
helper to build new branch inputs from the highest-entropy step of each response.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Set

import hashlib
import math
import random
import re

import numpy as np
import torch

from verl.utils.reward_score.logi import compute_score

from verl.protocol import DataProto, unpad_dataproto


@dataclass
class Node:
    """A tree node corresponding to a response step.

    Attributes:
        node_id: Globally unique id for lookup/branching.
        prompt_id: Id of the owning prompt/tree.
        step_idx: Position of the step within the response sequence.
        entropy: Average entropy at this step.
        reward: Reward assigned to the step (default 0, can be updated later).
        text: Decoded text up to this step for inspection/debugging.
        prompt_tensor: Prompt tokens (to rebuild prefixes via traversal).
        response_tensor: Response tokens for this path.
        step_span: Token span (start, end) for this step within response_tensor.
        parent: Parent node. None for root.
        children: Child nodes branched from this step.
        visits: Number of times this node has been visited/updated (random step reward only).
        value_sum: Accumulated value used for backpropagation (random step reward only).
        state_value: Deterministic value computed from outcome-based backpropagation.
    """

    node_id: str
    prompt_id: str
    step_idx: int
    entropy: float
    reward: float = 0.0
    text: str = ""
    prompt_tensor: Optional[torch.Tensor] = None
    response_tensor: Optional[torch.Tensor] = None
    step_span: tuple[int, int] = (0, 0)
    parent: Optional["Node"] = None
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    state_value: float = 0.0
    global_state_value: float = 0.0
    q_value: float = 0.0
    
    def __hash__(self):
        return hash(self.node_id)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def value(self) -> float:
        return self.value_sum / max(self.visits, 1)

    def add_child(self, child: "Node") -> "Node":
        child.parent = self
        self.children.append(child)
        return child

    def backpropagate(self, reward: float) -> None:
        """Propagate reward to ancestors (simple average-style update)."""
        node: Optional[Node] = self
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node.reward = reward
            node = node.parent

    # outcome-based deterministic value (used after correctness backprop)
    def backpropagate_value(self, value: float) -> None:
        node: Optional[Node] = self
        while node is not None:
            node.state_value = value
            node = node.parent


class SearchTree:
    """A per-sample search tree that stores branching history."""

    def __init__(self, prompt_id: str, prompt_text: str | None = None, ground_truth: Optional[str] = None):
        self.prompt_id = prompt_id
        self.prompt_text = prompt_text or ""
        self.ground_truth = ground_truth
        self.root = Node(
            node_id=f"root-{prompt_id}",
            prompt_id=prompt_id,
            step_idx=-1,
            entropy=0.0,
            text=self.prompt_text,
            prompt_tensor=None,
            response_tensor=None,
            step_span=(0, 0),
            parent=None,
        )

    def add_step(
        self,
        step_idx: int,
        entropy: float,
        reward: float,
        text: str = "",
        parent: Optional[Node] = None,
        backpropagate_reward: bool = True,
        node_id: Optional[str] = None,
        prompt_id: Optional[str] = None,
        prompt_tensor: Optional[torch.Tensor] = None,
        response_tensor: Optional[torch.Tensor] = None,
        step_span: tuple[int, int] = (0, 0),
    ) -> Node:
        parent = parent or self.root

        node = Node(
            node_id=node_id,
            prompt_id=prompt_id or self.prompt_id,
            step_idx=step_idx,
            entropy=entropy,
            reward=reward,
            text=text,
            prompt_tensor=prompt_tensor,
            response_tensor=response_tensor,
            step_span=step_span,
            parent=parent,
        )
        parent.add_child(node)
        if backpropagate_reward:
            node.backpropagate(reward)
        return node


@dataclass
class BranchPlan:
    """Container for a set of branch inputs derived from entropy analysis."""

    branch_batch: Optional[DataProto]
    nodes: List[Node]

    @property
    def batch_size(self) -> int:
        if self.branch_batch is None or self.branch_batch.batch is None:
            return 0
        return int(self.branch_batch.batch.batch_size[0])


def _compute_response_mask(data: DataProto) -> torch.Tensor:
    """Compute a mask for the response portion of a batch.

    Falls back to all-ones mask if attention_mask is missing.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"] if "attention_mask" in data.batch.keys() else None
    if attention_mask is None:
        return torch.ones_like(responses, dtype=torch.float32)
    return attention_mask[:, -response_length:]


def _pad_and_stack(seqs: Sequence[torch.Tensor], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class TreeManager:
    """Manage search trees indexed by simple integer tree_id (0, 1, 2, ...).
    
    Trees are stored in a list matching the order of gen_batch_output generation.
    Each tree_id directly corresponds to the position in gen_batch_output.
    No complex prompt_id hashing needed.
    """

    def __init__(
        self,
        tokenizer=None,
        pad_token_id: int | None = None,
        default_reward: float = 1.0,
        branch_level: str = "step",
        step_reward_type: str = "random",
        defer_step_backprop: bool = False,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id if pad_token_id is not None else (tokenizer.pad_token_id if tokenizer is not None else 0)
        self.default_reward = default_reward
        self.branch_level = branch_level if branch_level in {"step", "token"} else "step"
        self.step_reward_type = (step_reward_type or "random").lower()
        if self.step_reward_type not in {"random", "fol", "format", "treerl"}:
            self.step_reward_type = "random"
        # 控制是否推迟 step reward 回传：True 时仅在叶子/终止时回传
        self.defer_step_backprop = defer_step_backprop
        # Store trees in a list indexed by tree_id (0, 1, 2, ...)
        # tree_id corresponds to position in gen_batch_output
        self.trees: List[SearchTree] = []
        # Snapshot of initial leaves per tree (before any branching)
        self.tree_leaves: Dict[int, Set[Node]] = {}
        # a stub scorer that can be overridden later
        self.step_scorer = self._random_step_scorer
        # node registries for entropy-based branching
        self.node_entropy = {}
        self.node_map: Dict[str, Node] = {}
        self._node_counter = 0

    def _random_step_scorer(self, step_text: str) -> float:
        """Placeholder step scoring: randomly returns 0 or 1.

        Replace this with a real scoring function as needed.
        """
        return float(random.randint(0, 1))

    def __repr__(self) -> str:
        summaries = [f"tree_{i}: nodes={self._count_nodes(tree.root)}" for i, tree in enumerate(self.trees)]
        inner = "; ".join(summaries) if summaries else "empty"
        return f"TreeManager({inner})"

    # -------------------------------
    # node helpers

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"node_{self._node_counter}"

    def _register_node(self, node: Node) -> None:
        if not node.node_id:
            node.node_id = self._next_node_id()
        self.node_map[node.node_id] = node
        self.node_entropy[node.node_id] = float(node.entropy)

    # -------------------------------
    # step reward helpers

    def _compute_step_reward(self, step_text: str, prompt_text: Optional[str], step_content: List[str]) -> float:
        """根据 step_reward_type 计算 step 级奖励。

        - random: 默认随机 0/1（与 naive manager 对齐）
        - fol: 使用 nl2fol 翻译与执行（需要 prompt 中有 <Context>/<Question>/<Options>）
        - format: 轻量格式检查，占位实现（非空给 1.0，否则 0.0），后续可替换更严格规则
        """

        if self.step_reward_type == "fol":
            try:
                from verl.utils.nl2fol import fol_prepocessing, translate_and_execute_fol

                if prompt_text:
                    context_match = re.search(r"<Context>(.*?)</Context>", prompt_text, re.DOTALL)
                    context = context_match.group(1).strip() if context_match else None
                    question_match = re.search(r"<Question>(.*?)</Question>", prompt_text, re.DOTALL)
                    question = question_match.group(1).strip() if question_match else None
                    options_match = re.search(r"<Options>(.*?)</Options>", prompt_text, re.DOTALL)
                    options = options_match.group(1).strip() if options_match else None
                    declaration = fol_prepocessing(context, question, options)
                    sentences = "\n\n".join(step_content)
                    return float(translate_and_execute_fol(declaration=declaration, sentences=sentences))
            except Exception:
                # FOL 失败回退到随机
                return self._random_step_scorer(step_text)
        elif self.step_reward_type == "format":
            # 检查 step 文本是否符合格式：是否被 <Action></Action> 包围
            return 1.0 if re.match(r"<Action>(.*?)</Action>", step_text, re.DOTALL) else 0.0
        elif self.step_reward_type == "treerl":
            """
            TreeRL 的奖励需要在采样结束、每个结点的 Q value 都被计算出来后才能确定，它的计算方法如下：
            1. Gobal Reward = V(s_n) - V(root): 也就是当前结点的 Q value 减去 根结点的 Q value 的值；
            2. Local Reward = V(s_n) - V(parent(s_n)): 也就是当前结点的 Q value 减去 父结点的 Q value 的值；
            3. 最终的 Reward = (Global Reward + Local Reward) / sqrt(s_n 的叶子节点个数)
            所以这里暂时返回默认奖励。
            """
            return float(self.default_reward) if self.default_reward is not None else 0.0
        else:
            # 默认 random（含异常回退）
            return self._random_step_scorer(step_text)

    def register_batch(
        self,
        gen_batch: DataProto,
        index_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Legacy method - now handled in initialize_trees().
        
        Kept for backward compatibility but does nothing.
        All initialization is now done in initialize_trees() directly.
        """
        pass

    # -------------------------------
    # Tree building and branching utilities

    def _compute_step_entropies(
        self,
        entropy_vec: torch.Tensor,
        response_mask_vec: torch.Tensor,
        step_token_spans: List[tuple[int, int]],
    ) -> List[float]:
        ent = entropy_vec * response_mask_vec
        step_entropies: list[float] = []
        for s, e in step_token_spans:
            if e > ent.size(0):
                e = ent.size(0)
            if e <= s:
                step_entropies.append(0.0)
            else:
                step_entropies.append(float(ent[s:e].mean().item()))
        return step_entropies

    def _build_nodes_from_response(
        self,
        *,
        tree: SearchTree,
        parent: Optional[Node],
        prompt_tensor: torch.Tensor,
        response_tensor: torch.Tensor,
        entropy_vec: torch.Tensor,
        response_mask_vec: torch.Tensor,
        prompt_text: Optional[str],
        ground_truth: Optional[str],
        index: Optional[Any] = None,
    ) -> tuple[list[Node], list[float], list[tuple[int, int]]]:
        if self.branch_level == "token":
            step_token_spans = [(pos, pos + 1) for pos in range(response_tensor.size(0))]
            segments = [self.tokenizer.decode(response_tensor[pos : pos + 1], skip_special_tokens=True) if self.tokenizer is not None else "" for pos in range(response_tensor.size(0))]
        else:
            resp_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)# Decode
            segments = resp_text.split("\n\n") if resp_text else [""] # 分 step
            step_token_spans = []
            cursor = parent.step_span[1]# 如果是续写的结点，那么就从父亲结点的位置开始计数
            for seg in segments:
                seg_tokens = self.tokenizer.encode(seg, add_special_tokens=False) if self.tokenizer is not None else [] # encode
                start, end = cursor, cursor + len(seg_tokens) # 计算长度
                step_token_spans.append((start, end))# 记录 step 的 token span
                cursor = end# 指针移动到下一个 step 的开始位置

        step_entropies = self._compute_step_entropies(entropy_vec, response_mask_vec, step_token_spans)# List

        parent = parent or tree.root
        node_chain: list[Node] = []# 记录当前 response 对应的节点链
        step_rewards: list[float] = []# 记录每个 step 的奖励
        step_content: list[str] = []# 记录到当前 step 为止的内容
        for (s, e), seg_text, seg_ent in zip(step_token_spans, segments, step_entropies):
            step_content.append(seg_text)
            # Process Reward
            seg_reward = 0.0 if self.branch_level == "token" else self._compute_step_reward(step_text=seg_text, prompt_text=prompt_text, step_content=step_content)
            node = tree.add_step(
                step_idx=e - 1 if e > 0 else -1,
                entropy=seg_ent,
                reward=seg_reward,
                text=seg_text,
                parent=parent,
                backpropagate_reward=not self.defer_step_backprop,
                node_id=self._next_node_id(),
                prompt_id=tree.prompt_id,
                prompt_tensor=prompt_tensor,
                response_tensor=response_tensor,
                step_span=(s, e),
            )
            self._register_node(node)
            parent = node
            node_chain.append(node)
            step_rewards.append(seg_reward)

        if self.defer_step_backprop and node_chain:
            node_chain[-1].backpropagate(node_chain[-1].reward)

        # 不再需要 _record_response，因为 build_response_batch 直接从树遍历


        return node_chain, step_rewards, step_token_spans

    def _strip_trailing_pad(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Remove trailing pad tokens (by pad_token_id) from a 1D tensor."""
        pad_id = self.pad_token_id

        # TODO: 开头的 pad token 似乎无法移除，后面优化
        # exclude_ids = torch.tensor([pad_id, 151643]) 
        # mask = ~torch.isin(tensor, exclude_ids.to(tensor.device)) 
        # non_pad = mask.nonzero(as_tuple=True)
        non_pad = (tensor != pad_id).nonzero(as_tuple=True)
        if len(non_pad) == 0 or non_pad[0].numel() == 0:
            print("[TreeManager._strip_trailing_pad] Warning: tensor is all pad tokens, returning first token to avoid empty tensor!")
            # all pads; keep a single token to avoid empty tensors downstream
            return tensor[:1]
        last_idx = int(non_pad[0][-1].item()) + 1
        return tensor[:last_idx]

    def _reconstruct_prefix_tokens(self, node: Node) -> Optional[torch.Tensor]:
        """Rebuild prompt + response prefix for a node without duplicating spans.

        Previously we concatenated each ancestor's prefix, which multiplied lengths.
        Now we take the prompt once and slice the target node's response up to its step end,
        trimming trailing pad tokens to avoid inflated lengths.
        """
        # Walk ancestors to fetch the earliest prompt tensor and the node's response tensor
        prompt_tensor: Optional[torch.Tensor] = None
        response_tensor: Optional[torch.Tensor] = None
        target_end: Optional[int] = None

        cur: Optional[Node] = node
        while cur is not None:
            if prompt_tensor is None and cur.prompt_tensor is not None:
                prompt_tensor = cur.prompt_tensor
            if response_tensor is None and cur.response_tensor is not None:
                response_tensor = cur.response_tensor
            if target_end is None and cur.step_span:
                target_end = cur.step_span[1]
            cur = cur.parent


        prompt_clean = self._strip_trailing_pad(prompt_tensor)
        end = max(0, min(target_end, response_tensor.size(0)))
        resp_slice = response_tensor[:end]
        resp_clean = self._strip_trailing_pad(resp_slice)

        return torch.cat([prompt_clean, resp_clean], dim=-1)

    def initialize_trees(
        self,
        *,
        gen_batch: DataProto,
        gen_batch_output: DataProto,
        compute_log_prob_fn: Callable[[DataProto], DataProto],
    ) -> None:
        """Initialize all tree structures directly from generation output.
        
        Creates trees in a simple list indexed by tree_id (0, 1, 2, ...) that
        corresponds directly to positions in gen_batch_output.
        
        No complex prompt_id hashing needed - just list index!
        """
        if gen_batch_output is None or gen_batch_output.batch is None:
            return

        # ===== Phase 1: Clear state from previous batch =====
        self.node_entropy.clear()
        self.node_map.clear()
        self._node_counter = 0
        self.trees.clear()  # Clear trees list
        self.tree_leaves.clear()

        # ===== Phase 2: Extract data from gen_batch =====
        prompts_source = gen_batch.batch.get("input_ids")
        if prompts_source is None:
            prompts_source = gen_batch.batch.get("prompts")
        
        if prompts_source is None:
            return
        
        # Extract ground truth from original batch if available
        ground_truths_list = []
        if gen_batch.non_tensor_batch is not None and "answer" in gen_batch.non_tensor_batch:
            ground_truths_list = gen_batch.non_tensor_batch.get("answer", [])
        else:
            print("[Warning]None groundtruth found in gen_batch.non_tensor_batch for key 'answer'")

        prompt_batch_size = prompts_source.size(0)

        # ===== Phase 3: Compute log probs and entropies for all outputs =====
        log_prob_output = compute_log_prob_fn(gen_batch_output)
        entropies = log_prob_output.batch.get("entropys")
        response_mask = _compute_response_mask(gen_batch_output)
        responses = gen_batch_output.batch.get("responses")

        if any(x is None for x in [entropies, response_mask, responses]):
            raise ValueError("Missing required tensors in gen_batch_output for tree initialization: entropys, attention_mask, or responses")

        # ===== Phase 4: Build trees for each generated output =====
        # Each gen_batch_output[i] gets its own tree at self.trees[i]
        for output_idx in range(responses.size(0)):
            # Determine which prompt this output came from
            prompt_idx = output_idx // prompt_batch_size
            prompt_tensor = prompts_source[prompt_idx]
            response_tensor = responses[output_idx]

            # Get prompt text
            prompt_text = self.tokenizer.decode(prompt_tensor, skip_special_tokens=True)
            
            # Get ground truth for this prompt
            ground_truth = ground_truths_list[prompt_idx]
            
            # Create tree with simple string id
            tree_id = f"tree_{output_idx}"
            tree = SearchTree(prompt_id=tree_id, prompt_text=prompt_text)# 初始化搜索树
            tree.ground_truth = ground_truth
            self._register_node(tree.root)
            
            # Add tree to list at index = output_idx
            self.trees.append(tree)

            # Extract entropy and mask for this output
            entropy_vec = entropies[output_idx]
            mask_vec = response_mask[output_idx]

            # Build nodes from the response tensor
            nodes, _, _ = self._build_nodes_from_response(
                tree=tree,
                parent=tree.root,
                prompt_tensor=prompt_tensor,
                response_tensor=response_tensor,
                entropy_vec=entropy_vec,
                response_mask_vec=mask_vec,
                prompt_text=prompt_text,
                ground_truth=ground_truth,
                index=prompt_idx,  # Track original prompt index
            )

        for tree_idx, tree in enumerate(self.trees):
            self.tree_leaves[tree_idx] = set(self._get_all_leaves(tree.root))


            

    def get_top_k_entropy_nodes(self, k: int) -> list[Node]:
        """Select top-k entropy nodes per tree, excluding roots.
        
        Trees are accessed by simple index (0, 1, 2, ...) from the list.
        """
        if k <= 0:
            return []

        result: list[Node] = []

        def _collect_nodes(root: Node) -> list[Node]:
            stack = [root]
            nodes: list[Node] = []
            while stack:
                n = stack.pop()
                nodes.append(n)
                stack.extend(n.children)
            return nodes

        # Iterate through trees by index
        for tree in self.trees:
            nodes = _collect_nodes(tree.root)
            # Filter out root and nodes without step info
            candidates = [n for n in nodes if n.parent is not None and n.step_idx != -1 and n.children is not None]
            # Sort by entropy
            assert candidates!= [], f"No valid nodes found in tree {tree.prompt_id} for entropy selection"
            candidates.sort(key=lambda n: self.node_entropy.get(n.node_id, float("-inf")), reverse=True)
            result.extend(candidates[:k])

        return result

    def prepare_branches(
        self,
        *,
        target_nodes: Optional[List[Node]] = None,
        node_ids: Optional[List[str]] = None,
    ) -> Optional[BranchPlan]:
        branch_sequences: list[torch.Tensor] = []
        branch_nodes: list[Node] = []

        for node in target_nodes:
            prefix = self._reconstruct_prefix_tokens(node)
            if prefix is None:
                continue
            branch_sequences.append(prefix)
            branch_nodes.append(node)

        device = branch_sequences[0].device
        input_ids, attention_mask, position_ids = _pad_and_stack(branch_sequences, pad_token_id=self.pad_token_id, device=device)

        meta_info = {"branch_from_entropy": True}

        branch_batch = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": input_ids,
            },
            non_tensors={},
            meta_info=meta_info,
        )
        return BranchPlan(branch_batch=branch_batch, nodes=branch_nodes)

    def commit_branch_outputs(
        self,
        branch_output: DataProto,
        branch_plan: BranchPlan,
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        pad_size: int = 0,
    ) -> None:
        # 会生成 rollout.n 个 response，这里我们只取第一个！
        n = len(branch_output) // branch_plan.batch_size

        responses = branch_output.batch.get("responses")[::n]
        prompt_source = branch_output.batch.get("prompts")[::n]
        if prompt_source is None:
            prompt_source = branch_output.batch.get("input_ids")[::n]
        attn_mask = branch_output.batch.get("attention_mask")[::n]
        pos_ids = branch_output.batch.get("position_ids")[::n]
        log_prob_output = compute_log_prob_fn(branch_output)
     

        entropies = log_prob_output.batch.get("entropys")[::n]
        response_mask = _compute_response_mask(branch_output)[::n]
        for idx, (node, response) in enumerate(zip(branch_plan.nodes, responses)):
            
            if node is None:
                raise ValueError(f"BranchPlan node at index {idx} is None, cannot commit branch output.")
            prompt_tensor = prompt_source[idx].clone()
            prompt_text = self.tokenizer.decode(prompt_tensor, skip_special_tokens=True) if self.tokenizer is not None else None
            # tree = next((t for t in self.trees if t.prompt_id == node.prompt_id), None)
            tree = None
            # 遍历列表
            for t in self.trees:
                # 检查条件
                if t.prompt_id == node.prompt_id:
                    tree = t
                    break  # 找到了！立刻停止循环，不再往后看了
            if tree is None:
                raise ValueError(f"No matching tree found for node with prompt_id {node.prompt_id}")

            entropy_vec = entropies[idx] 
            mask_vec = response_mask[idx]

            child_nodes, _, _ = self._build_nodes_from_response(
                tree=tree,
                parent=node,
                prompt_tensor=prompt_tensor,
                response_tensor=response,
                entropy_vec=entropy_vec,
                response_mask_vec=mask_vec,
                prompt_text=prompt_text,
                ground_truth=tree.ground_truth,
                index=None,
            )
            
            

            if not self.defer_step_backprop:
                node.backpropagate(node.reward if node.reward is not None else self.default_reward)

        if self.defer_step_backprop:
            for node in branch_plan.nodes:
                if node is not None:
                    node.backpropagate(node.reward if node.reward is not None else self.default_reward)

        self._update_leaves()

    # ------------------------------------------------------------------
    # MCTS-style correctness backpropagation

    def _compute_state_value(self, root: Node) -> float:
        """Iterative post-order to avoid Python recursion depth issues.

        Assumes leaf nodes的 state_value 已在外部写入（例如 correctness 评分）。
        """

        stack: list[tuple[Node, bool]] = [(root, False)]
        visited: set[int] = set()

        while stack:
            node, processed = stack.pop()
            if id(node) in visited and not processed:
                # 避免异常环引用造成死循环
                continue

            if processed or not node.children:
                if node.children:
                    child_values = [child.state_value for child in node.children]
                    if child_values:
                        node.state_value = float(np.mean(child_values))# 更新 node 的 state value
                continue

            # push node again as processed, then children
            stack.append((node, True))
            visited.add(id(node))
            for child in node.children:
                stack.append((child, False))

        return root.state_value

    def _assign_global_state_value(self, root: Node) -> None:
        """Compute mean state_value of all non-root nodes and store on every node (iterative)."""
        nodes: list[Node] = []
        stack = [root]
        seen: set[int] = set()
        while stack:
            n = stack.pop()# 从 stack 中 pop 出一个 node
            if id(n) in seen:# 如果这个 node 已经被 visited，skipt
                continue
            seen.add(id(n))# 标记这个 node 已经被 visited
            nodes.append(n)# 加入 nodes
            for c in n.children: # 把结点的孩子压入 stack -> node 的 孩子将如下一轮，最终 stack 中只包含 叶子结点
                stack.append(c)

        non_root = [n for n in nodes if n is not root]
        if len(non_root) == 0:
            mean_val = 0.0
        else:
            mean_val = float(np.mean([n.state_value for n in non_root]))
        for n in nodes:
            n.global_state_value = mean_val# 树所有结点的平均 value

    # ------------------------------------------------------------------
    # Q/Return computation per step

    def compute_q_values(self, gamma: float = 1.00) -> None:
        """Compute discounted returns (Q values) for each path in the tree.

        Q_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        Traverses each tree and computes Q values along all paths.
        """

        for tree_idx, tree in enumerate(self.trees):
            # For each leaf, traverse path from leaf to root and compute Q values
            leaves = list(self.tree_leaves[tree_idx])
            # leaves = self._get_all_leaves(tree.root)
            # for n in self.initial_leaves[tree_idx]:
            #     if n not in leaves:
            #         leaves.append(n)

            for leaf in leaves:
                path = self._get_node_chain(leaf)
                path.reverse()  # root -> leaf
                
                # Compute Q values backward from leaf to root
                rewards = [node.reward for node in path[1:]]  # Skip root
                if not rewards:
                    continue
                
                q_vals = [0.0 for _ in rewards]
                running = 0.0
                for i in reversed(range(len(rewards))):
                    running = rewards[i] + gamma * running
                    q_vals[i] = running
                
                # Assign Q values to nodes (skip root)
                for node, q_val in zip(path[1:], q_vals):
                    node.q_value = q_val


    # ------------------------------------------------------------------
    # TreeRL reward computation

    def apply_treerl_rewards(self) -> None:
        """Compute TreeRL-style rewards after Q values are available.

        Reward(s_n) = ( (V(s_n) - V(root)) + (V(s_n) - V(parent(s_n))) ) / sqrt(#leaves under s_n)

        - V(s) is taken from node.q_value (defaults to 0.0 if missing)
        - Leaf count guards against malformed trees by enforcing at least 1 leaf
        - Updates both node.reward and the cached step_rewards inside response_records
        """

        if self.step_reward_type != "treerl":
            return None

        # Pre-compute leaf counts for every tree
        leaf_count_cache: Dict[int, int] = {}

        for tree in self.trees:
            counts = self._compute_leaf_counts(tree.root)
            leaf_count_cache.update(counts)

            # Traverse nodes to assign rewards
            stack = [tree.root]
            while stack:
                node = stack.pop()
                stack.extend(node.children)

                if node is tree.root:
                    continue

                leaf_num = max(leaf_count_cache.get(id(node), 1), 1)
                root_v = float(tree.root.state_value) if tree.root.state_value is not None else 0.0
                node_v = float(node.state_value) if node.state_value is not None else 0.0
                parent_v = float(node.parent.state_value)

                global_reward = node_v - root_v
                local_reward = node_v - parent_v
                node.reward = (global_reward + local_reward) / math.sqrt(leaf_num)# 更新 reward

    def backpropagate_correctness(self) -> None:
        """Compute outcome correctness for leaves and backpropagate averaged values."""
        if self.tokenizer is None:
            return

        # 1) assign leaf values using compute_score for each tree
        for tree in self.trees:
            ground_truth = tree.ground_truth
            self._assign_leaf_correctness_values(tree.root, ground_truth)
            
            # 2) bottom-up averaging for each tree
            self._compute_state_value(tree.root)
            self._assign_global_state_value(tree.root)# 总体平均的 state value 存在每个结点的 global_state_value 中

    def _assign_leaf_correctness_values(self, node: Node, ground_truth: Optional[str]) -> None:
        """对所有叶子节点分配基于 ground_truth 的正确性得分。"""
        if node.is_leaf and ground_truth is not None:
            # 从这个叶子节点的 text 推导响应
            response_str = node.text or ""
            try:
                score, _ = compute_score(response_str, ground_truth)
                node.state_value = float(score)
            except Exception:
                node.state_value = 0.0
        else:
            # 递归处理所有孩子
            for child in node.children:
                self._assign_leaf_correctness_values(child, ground_truth)



    def pretty_print_tree(self, tree_idx: int = 0) -> str:
        """返回并打印指定索引树的层次结构。

        Args:
            tree_idx: 第几棵树（按 self.trees 的插入顺序/遍历顺序），默认 0。

        Returns:
            一段文本，每一行代表一个节点，使用树形连线符号展示父子关系。
            也会同步打印到 stdout 便于快速查看。
        """

        if not self.trees:
            msg = "[TreeManager] 当前没有任何树可打印"
            print(msg)
            return msg

        if tree_idx < 0 or tree_idx >= len(self.trees):
            msg = f"[TreeManager] tree_idx={tree_idx} 超出范围 (共有 {len(self.trees)} 棵树)"
            print(msg)
            return msg

        tree = self.trees[tree_idx]
        root = tree.root

        lines: list[str] = []

        def _dfs(node: Node, prefix: str, is_last: bool) -> None:
            connector = "└─" if is_last else "├─"
            # 根节点单独展示，不加连接符
            if node is root:
                lines.append(f"{node.node_id} (tree_{tree_idx})")
            else:
                lines.append(f"{prefix}{connector}{node.node_id}")

            children = node.children
            for i, child in enumerate(children):
                child_last = i == len(children) - 1
                child_prefix = prefix + ("   " if is_last else "│  ")
                _dfs(child, child_prefix, child_last)

        _dfs(root, prefix="", is_last=True)

        output = "\n".join(lines)
        print(output)
        return output

    def _count_nodes(self, node: Node) -> int:
        return 1 + sum(self._count_nodes(child) for child in node.children)

    def _compute_leaf_counts(self, root: Node) -> Dict[int, int]:
        """Compute the number of leaf nodes in every subtree (id(node) -> leaf_count).

        Uses an iterative post-order traversal to avoid recursion depth limits.
        Ensures every non-leaf subtree has at least 1 leaf (guard against malformed trees).
        """

        postorder: list[Node] = []# 这是遍历的顺序
        stack: list[Node] = [root]
        while stack:
            n = stack.pop()
            postorder.append(n)
            for c in n.children:
                stack.append(c)

        leaf_counts: Dict[int, int] = {}
        for n in reversed(postorder):
            if n.is_leaf:
                leaf_counts[n.node_id] = 1
            else:
                count = sum(leaf_counts.get(c.node_id, 1) for c in n.children)
                leaf_counts[n.node_id] = max(count, 1)
        return leaf_counts

    def build_response_batch(self, batch_index_list: Optional[list[Any]] = None, gen_batch_output: Optional[DataProto] = None, use_state_values: bool = True) -> Optional[DataProto]:
        """Build response batch by traversing trees in order (0, 1, 2, ...).
        
        Since trees are stored in a simple list indexed by tree_id,
        we just iterate through them in order to get responses.
        
        Each tree[i] corresponds to gen_batch_output[i], so union will work perfectly!
        """
        if not self.trees:
            return None
        
        # Collect responses in tree list order (0, 1, 2, ...)
        response_data_list: list[dict[str, Any]] = []
        
        for tree_idx, tree in enumerate(self.trees):
            # Combine initial snapshot leaves with current leaves, dedup by node_id
            leaves = list(self.tree_leaves[tree_idx])
            # leaves = self._get_all_leaves(tree.root)
            # leaves_initial = self.initial_leaves[tree_idx]
            # for n in leaves_initial:
            #     if n not in leaves:
            #         leaves.append(n)
            ground_truth = tree.ground_truth
            print(f"{len(leaves)} leaves found in tree_idx {tree_idx} with ground_truth: {ground_truth}")
            
            if not leaves:
                # If no leaves, extract from root
                print("build_response_batch: No leaves found in tree, extracting from root")
                response_data = self._reconstruct_path_tensors([tree.root], tree.root)
                response_data["ground_truth"] = ground_truth
                response_data_list.append(response_data)
                continue

            # Get all leaf paths (including those from initial snapshot)
            for leaf in leaves:
                node_chain = self._get_node_chain(leaf)
                node_chain.reverse()  # root -> leaf
                response_data = self._reconstruct_path_tensors(node_chain, tree.root)
                response_data["ground_truth"] = ground_truth
                response_data_list.append(response_data)

        
        if not response_data_list:
            return None
        
        device = response_data_list[0]["full_sequence_ids"].device
        
        # Collect all components in tree order
        full_sequences = [r["full_sequence_ids"] for r in response_data_list]
        prompts = [r["prompt_ids"] for r in response_data_list]
        responses = [r["response_ids"] for r in response_data_list]
        step_rewards_list = [r["step_rewards"] for r in response_data_list]
        step_state_values_list = [r["step_state_values"] for r in response_data_list]
        step_q_values_list = [r["step_q_values"] for r in response_data_list]
        step_spans_list = [r["step_spans"] for r in response_data_list]
        response_lens = [r["response_len"] for r in response_data_list]
        ground_truths = [r.get("ground_truth") for r in response_data_list]
        
        # Pad sequences
        input_ids, attention_mask, position_ids = _pad_sequences(full_sequences, pad_token_id=self.pad_token_id, device=device)
        prompts_padded, _, _ = _pad_sequences(prompts, pad_token_id=self.pad_token_id, device=device)
        responses_padded, _, _ = _pad_sequences(responses, pad_token_id=self.pad_token_id, device=device)
        
        # Build token-level scores
        token_level_scores = _build_token_level_scores(
            responses=responses_padded,
            response_lens=response_lens,
            all_step_spans=step_spans_list,
            all_step_rewards=step_rewards_list,
        )
        state_value_scores = _build_token_level_scores(
            responses=responses_padded,
            response_lens=response_lens,
            all_step_spans=step_spans_list,
            all_step_rewards=step_state_values_list,
        )
        q_value_scores = _build_token_level_scores(
            responses=responses_padded,
            response_lens=response_lens,
            all_step_spans=step_spans_list,
            all_step_rewards=step_q_values_list,
        )
        
        chosen_scores = state_value_scores if use_state_values else token_level_scores
        
        # Compute verifiable rewards
        verifiable_rewards = torch.zeros(len(response_data_list), device=device, dtype=torch.float32)
        for i, (response_ids, gt) in enumerate(zip(responses, ground_truths)):
            if gt is not None:
                response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True) if self.tokenizer else ""
                try:
                    score, _ = compute_score(response_str, gt)
                    verifiable_rewards[i] = float(score)
                except Exception:
                    verifiable_rewards[i] = 0.0
        
        # Build score_ids and reward_mask
        max_steps = max((len(x) for x in step_spans_list), default=0)
        score_ids = torch.full((len(response_data_list), max_steps), -1, device=device, dtype=torch.long)
        reward_mask = torch.zeros_like(score_ids, dtype=torch.float32)
        
        for i, ends in enumerate([[(s, e) for s, e in spans] for spans in step_spans_list]):
            step_end_indices = []
            for _, end in ends:
                end_pos = max(0, min(end - 1, response_lens[i] - 1)) if response_lens[i] > 0 else 0
                step_end_indices.append(end_pos)
            
            for j, end_pos in enumerate(step_end_indices[:max_steps]):
                score_ids[i, j] = end_pos
                reward_mask[i, j] = 1.0
        
        # Build DataProto - NO index field anymore, order matches gen_batch_output
        non_tensors_dict = {}

        response_batch = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompts_padded,
                "responses": responses_padded,
                "reward_fn_scores": chosen_scores,
                "state_value_scores": state_value_scores,
                "q_value_scores": q_value_scores,
                "score_ids": score_ids,
                "reward_mask": reward_mask,
                "verifiable_rewards": verifiable_rewards.unsqueeze(-1),
            },
            non_tensors=non_tensors_dict,
            meta_info={},
        )
        
        return response_batch
        
        reward_proto = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompts_padded,
                "responses": responses_padded,
                "reward_fn_scores": chosen_scores,
                "verifiable_rewards": verifiable_rewards,
                "step_reward_scores": token_level_scores,
                "state_value_scores": state_value_scores,
                "step_q_scores": q_value_scores,
                "score_ids": score_ids,
                "reward_mask": reward_mask,
            },
            non_tensors=non_tensors_dict,
            meta_info={},
        )
        return reward_proto

    def _update_leaves(self,) -> None:
        """更新每棵树的叶子节点列表，供后续路径提取使用。"""
        for tree_idx, tree in enumerate(self.trees):
            leaves = self._get_all_leaves(tree.root)
            self.tree_leaves[tree_idx].update(leaves)
            print("[INFO]: Add {} leaves! Updated leaves for tree_idx {}: {} leaves total".format(len(leaves), tree_idx, len(self.tree_leaves[tree_idx])))
            for leaf in leaves:
                print(leaf.node_id)
            

        self.initial_leaves = [self._get_all_leaves(t.root) for t in self.trees]
    
    def _get_all_leaves(self, root: Node) -> list[Node]:
        """后序遍历获取所有叶子节点。"""
        leaves = []
        stack = [root]
        visited = set()
        
        while stack:
            node = stack[-1]
            node_id = node.node_id
            
            if node_id in visited:
                stack.pop()
                if node.is_leaf:
                    leaves.append(node)
            else:
                visited.add(node_id)
                for child in node.children:
                    stack.append(child)
        
        return leaves
    
    def _get_node_chain(self, leaf: Node) -> list[Node]:
        """从叶子回溯到根，获取完整的路径。"""
        chain = []
        node = leaf
        while node is not None:
            chain.append(node)
            node = node.parent
        return chain
    
    def _reconstruct_path_tensors(self, node_chain: list[Node], root: Node) -> dict[str, Any]:
        """
        给定从 root 到 leaf 的完整 node_chain，重新编码每个节点，计算 step_spans 和相关得分。
        
        返回字典包含：
        - full_sequence_ids: 完整序列 (prompt + response)
        - prompt_ids: 提示部分
        - response_ids: 响应部分
        - step_rewards: 每个步骤的奖励列表
        - step_state_values: 每个步骤的状态价值列表
        - step_q_values: 每个步骤的 Q 值列表
        - step_spans: 每个步骤在响应中的 token span (start, end)
        - response_len: 响应的总长度
        """
        # 步骤1：获取根的 prompt
        prompt_tensor = root.prompt_tensor
        if prompt_tensor is None:
            # 从 node_chain 中找最早有 prompt 的节点
            for node in node_chain:
                if node.prompt_tensor is not None:
                    prompt_tensor = node.prompt_tensor
                    break
        
        if prompt_tensor is None:
            prompt_tensor = torch.empty((0,), dtype=torch.long)
        
        device = prompt_tensor.device if prompt_tensor.numel() > 0 else torch.device("cpu")
        prompt_str = self.tokenizer.decode(prompt_tensor, skip_special_tokens=True) if self.tokenizer else ""
        
        # 步骤2：遍历 node_chain 重新编码每个步骤
        step_rewards = []
        step_state_values = []
        step_q_values = []
        step_spans = []
        
        response_tokens = []
        response_len = 0
        
        for node in node_chain[1:]:  # 跳过根节点
            # 获取该节点的文本
            node_text = node.text or ""
            
            # 编码文本
            if self.tokenizer:
                step_tokens = self.tokenizer.encode(node_text, add_special_tokens=False)
                step_tokens = torch.tensor(step_tokens, dtype=torch.long, device=device)
            else:
                # 如果没有 tokenizer，使用原始的 response_tensor 切片
                step_tokens = node.response_tensor[node.step_span[0]:node.step_span[1]] if node.response_tensor is not None else torch.empty((0,), dtype=torch.long, device=device)
            
            # 记录 step_span (在累积响应中的位置)
            start_pos = response_len
            end_pos = response_len + step_tokens.numel()
            step_spans.append((start_pos, end_pos))
            response_len = end_pos
            
            # 累积响应 tokens
            response_tokens.append(step_tokens)
            
            # 记录奖励
            step_rewards.append(float(node.reward) if node.reward is not None else 0.0)
            step_state_values.append(float(node.state_value) if node.state_value is not None else 0.0)
            step_q_values.append(float(node.q_value) if node.q_value is not None else 0.0)
        
        # 步骤3：拼接所有 response tokens
        if response_tokens:
            response_ids = torch.cat(response_tokens, dim=-1)
        else:
            response_ids = torch.empty((0,), dtype=torch.long, device=device)
        
        # 步骤4：构建完整序列
        full_sequence_ids = torch.cat([prompt_tensor, response_ids], dim=-1) if prompt_tensor.numel() > 0 else response_ids
        
        return {
            "full_sequence_ids": full_sequence_ids,
            "prompt_ids": prompt_tensor,
            "response_ids": response_ids,
            "step_rewards": step_rewards,
            "step_state_values": step_state_values,
            "step_q_values": step_q_values,
            "step_spans": step_spans,
            "response_len": response_len,
        }

    # ------------------------------------------------------------------
    # Helper utilities


def _pad_sequences(seqs: Sequence[torch.Tensor], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of 1D token tensors to a batch with attention/position ids."""
    max_len = max(seq.size(0) for seq in seqs)
    dtype = seqs[0].dtype
    batch = torch.full((len(seqs), max_len), pad_token_id, dtype=dtype, device=device)
    attn = torch.zeros((len(seqs), max_len), dtype=dtype, device=device)
    for i, seq in enumerate(seqs):
        l = seq.size(0)
        batch[i, :l] = seq
        attn[i, :l] = 1
    pos_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(len(seqs), max_len)
    return batch, attn, pos_ids


def _build_token_level_scores(
    responses: torch.Tensor,
    response_lens: List[int],
    all_step_spans: List[List[tuple[int, int]]],
    all_step_rewards: List[List[float]],
) -> torch.Tensor:
    """Broadcast each step reward to all tokens within that step span."""
    scores = torch.zeros_like(responses, dtype=torch.float32)
    max_len = responses.size(1)
    for i, (lens, spans, rewards) in enumerate(zip(response_lens, all_step_spans, all_step_rewards)):
        for (s, e), r in zip(spans, rewards):
            if lens <= 0:
                continue
            start = max(0, s)
            end = max(start, min(e, lens))
            if start >= end:
                continue
            end = min(end, max_len)
            scores[i, start:end] = float(r)
    return scores
