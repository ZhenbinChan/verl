"""
MCTS-style search tree utilities for branching on high-entropy response steps.

This module provides a lightweight Node/SearchTree abstraction plus a TreeManager
that can be used from trainers to track per-prompt search trees. It also offers a
helper to build new branch inputs from the highest-entropy step of each response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import hashlib
import random
import re

import numpy as np
import torch

from verl.utils.reward_score.logi import compute_score

from verl.protocol import DataProto


@dataclass
class Node:
    """A tree node corresponding to a response step.

    Attributes:
        step_idx: Position of the step within the response sequence.
        entropy: Average entropy at this step.
        reward: Reward assigned to the step (default 0, can be updated later).
        text: Decoded text up to this step for inspection/debugging.
        parent: Parent node. None for root.
        children: Child nodes branched from this step.
        visits: Number of times this node has been visited/updated (random step reward only).
        value_sum: Accumulated value used for backpropagation (random step reward only).
        state_value: Deterministic value computed from outcome-based backpropagation.
    """

    step_idx: int
    entropy: float
    reward: float = 0.0
    text: str = ""
    parent: Optional["Node"] = None
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    state_value: float = 0.0
    global_state_value: float = 0.0
    q_value: float = 0.0

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
    """A per-prompt search tree that stores branching history."""

    def __init__(self, prompt_id: str, prompt_text: str | None = None):
        self.prompt_id = prompt_id
        self.prompt_text = prompt_text or ""
        self.root = Node(step_idx=-1, entropy=0.0, text=self.prompt_text, parent=None)

    def add_step(self, step_idx: int, entropy: float, reward: float, text: str = "", parent: Optional[Node] = None) -> Node:
        parent = parent or self.root
        node = Node(step_idx=step_idx, entropy=entropy, reward=reward, text=text, parent=parent)
        parent.add_child(node)
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
    """Manage search trees for multiple prompts and build branch inputs."""

    def __init__(
        self,
        tokenizer=None,
        pad_token_id: int | None = None,
        default_reward: float = 1.0,
        branch_level: str = "step",
        step_reward_type: str = "random",
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id if pad_token_id is not None else (tokenizer.pad_token_id if tokenizer is not None else 0)
        self.default_reward = default_reward
        self.branch_level = branch_level if branch_level in {"step", "token"} else "step"
        self.step_reward_type = (step_reward_type or "random").lower()
        if self.step_reward_type not in {"random", "fol", "format"}:
            self.step_reward_type = "random"
        self.trees: Dict[str, SearchTree] = {}
        # a stub scorer that can be overridden later
        self.step_scorer = self._random_step_scorer
        # records of all responses generated for the current batch (including branches)
        self.response_records: list[ResponseRecord] = []
        # store per-prompt ground truth for correctness scoring
        self.prompt_ground_truth: Dict[str, Optional[str]] = {}

    def _random_step_scorer(self, step_text: str) -> float:
        """Placeholder step scoring: randomly returns 0 or 1.

        Replace this with a real scoring function as needed.
        """
        return float(random.randint(0, 1))

    def __repr__(self) -> str:
        summaries = [f"prompt_id={pid}, nodes={self._count_nodes(tree.root)}" for pid, tree in self.trees.items()]
        inner = "; ".join(summaries) if summaries else "empty"
        return f"TreeManager({inner})"

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

        if self.step_reward_type == "format":
            # 占位的格式奖励：非空文本记 1，空文本记 0；可后续替换为严格规则
            return 1.0 if step_text and step_text.strip() else 0.0

        # 默认 random（含异常回退）
        return self._random_step_scorer(step_text)

    def _get_prompt_id(self, prompt_tensor: torch.Tensor) -> str:
        # Stable, content-based id: md5 over tensor bytes
        data = prompt_tensor.detach().cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()

    def _extract_ground_truth(self, gen_batch: DataProto, idx: int) -> Optional[str]:
        """Try to fetch ground truth for the i-th sample from non-tensor batch."""
        gt: Optional[str] = None
        rm_info = gen_batch.non_tensor_batch.get("reward_model") if gen_batch.non_tensor_batch is not None else None
        if isinstance(rm_info, dict):
            candidate = rm_info.get("ground_truth")
            if isinstance(candidate, (list, tuple)):
                if idx < len(candidate):
                    gt = candidate[idx]
            elif isinstance(candidate, str):
                gt = candidate
        elif isinstance(rm_info, (list, tuple)):
            if idx < len(rm_info) and isinstance(rm_info[idx], dict):
                gt = rm_info[idx].get("ground_truth")
        return gt

    def ensure_tree(self, prompt_tensor: torch.Tensor, prompt_text: Optional[str] = None, ground_truth: Optional[str] = None) -> SearchTree:
        prompt_id = self._get_prompt_id(prompt_tensor)
        if prompt_id not in self.trees:
            self.trees[prompt_id] = SearchTree(prompt_id=prompt_id, prompt_text=prompt_text)
        if ground_truth is not None:
            self.prompt_ground_truth[prompt_id] = ground_truth
        return self.trees[prompt_id]

    def register_batch(self, gen_batch: DataProto) -> None:
        """Ensure a tree exists for every prompt in the batch."""
        if gen_batch is None or gen_batch.batch is None:
            return
        # reset records for a new batch
        self.response_records = []
        prompts = gen_batch.batch["input_ids"] if "input_ids" in gen_batch.batch.keys() else None
        if prompts is None and "prompts" in gen_batch.batch.keys():
            prompts = gen_batch.batch["prompts"]
        if prompts is None:
            return
        for i in range(prompts.size(0)):
            prompt_tensor = prompts[i]
            prompt_text = None
            if self.tokenizer is not None:
                prompt_text = self.tokenizer.decode(prompt_tensor, skip_special_tokens=True)
            ground_truth = self._extract_ground_truth(gen_batch, idx=i)
            self.ensure_tree(prompt_tensor, prompt_text=prompt_text, ground_truth=ground_truth)

    def prepare_branches(
        self,
        gen_batch: DataProto,
        gen_batch_output: DataProto,
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        top_k: int = 1,
    ) -> Optional[BranchPlan]:
        """Select top-k highest-entropy steps per sample and build branch inputs.

        Returns a BranchPlan containing a new DataProto for branch generation and
        the corresponding nodes so that the caller can commit outputs back.
        top_k applies per sample; total branches = batch_size * min(top_k, num_steps).
        """
        if gen_batch_output is None or gen_batch_output.batch is None:
            return None

        log_prob_output = compute_log_prob_fn(gen_batch_output)# 计算熵
        entropies = log_prob_output.batch.get("entropys") if log_prob_output is not None else None
        if entropies is None:
            return None

        response_mask = _compute_response_mask(gen_batch_output)# Response mask
        prompts_source = gen_batch.batch["input_ids"] if "input_ids" in gen_batch.batch.keys() else None
        if prompts_source is None and "prompts" in gen_batch.batch.keys():
            prompts_source = gen_batch.batch["prompts"]

        responses = gen_batch_output.batch["responses"] if "responses" in gen_batch_output.batch.keys() else None
        if prompts_source is None or responses is None:
            return None

        prompt_batch_size = prompts_source.size(0)

        branch_sequences: list[torch.Tensor] = []
        branch_nodes: list[Node] = []

        for i in range(responses.size(0)):
            prompt_idx = i % prompt_batch_size
            prompt_tensor = prompts_source[prompt_idx]
            response_tensor = responses[i]
            # Decode response and split into steps
            if self.branch_level == "token":
                # Each token is treated as an individual step
                step_token_spans = [(pos, pos + 1) for pos in range(response_tensor.size(0))]
                if self.tokenizer is not None:
                    segments = [self.tokenizer.decode(response_tensor[pos : pos + 1], skip_special_tokens=True) for pos in range(response_tensor.size(0))]
                else:
                    segments = [""] * response_tensor.size(0)
            elif self.tokenizer is None:
                # Without tokenizer, fall back to whole response as one step
                segments = [""]
                step_token_spans = [(0, response_tensor.size(0))]
            else:
                resp_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
                segments = resp_text.split("\n\n") if resp_text else [""]# 分 step
                step_token_spans = []
                cursor = 0
                for seg in segments:
                    seg_tokens = self.tokenizer.encode(seg, add_special_tokens=False)
                    start, end = cursor, cursor + len(seg_tokens)
                    step_token_spans.append((start, end))
                    cursor = end

            # Compute per-step mean entropy over token span
            ent = entropies[i] * response_mask[i]
            step_entropies: list[float] = []
            for s, e in step_token_spans:
                if e > ent.size(0):
                    e = ent.size(0)
                if e <= s:
                    step_entropies.append(0.0)
                else:
                    step_entropies.append(float(ent[s:e].mean().item()))

            # Build chain of nodes: root -> step1 -> step2 ...
            prompt_text = None
            if self.tokenizer is not None:
                prompt_text = self.tokenizer.decode(prompt_tensor, skip_special_tokens=True)
            prompt_id = self._get_prompt_id(prompt_tensor)
            ground_truth = self.prompt_ground_truth.get(prompt_id)
            tree = self.ensure_tree(prompt_tensor, prompt_text=prompt_text, ground_truth=ground_truth)

            parent = tree.root
            node_chain: list[Node] = []
            step_rewards: list[float] = []
            step_content: list[str] = []
            for (s, e), seg_text, seg_ent in zip(step_token_spans, segments, step_entropies):
                step_content.append(seg_text)
                if self.branch_level == "token":
                    seg_reward = 0.0
                else:
                    seg_reward = self._compute_step_reward(seg_text=seg_text, prompt_text=prompt_text, step_content=step_content)
                node = tree.add_step(step_idx=e - 1 if e > 0 else -1, entropy=seg_ent, reward=seg_reward, text=seg_text, parent=parent)
                parent = node
                node_chain.append(node)
                step_rewards.append(seg_reward)

            # record the full response and its step rewards for later batching
            attn_mask = gen_batch_output.batch.get("attention_mask")
            pos_ids = gen_batch_output.batch.get("position_ids")
            self._record_response(
                prompt_tensor=prompt_tensor,
                response_tensor=response_tensor,
                attention_mask=attn_mask[i] if attn_mask is not None else None,
                position_ids=pos_ids[i] if pos_ids is not None else None,
                step_rewards=step_rewards,
                step_spans=step_token_spans,
                nodes=node_chain,
                leaf_node=node_chain[-1] if node_chain else None,
                ground_truth=ground_truth,
            )

            # select top-k steps by entropy (per sample)
            k = 1 if self.branch_level == "token" else max(1, top_k)
            sorted_idx = sorted(range(len(step_entropies)), key=lambda x: step_entropies[x], reverse=True)
            for idx in sorted_idx[:k]:
                span_start, span_end = step_token_spans[idx]
                # guard against empty span
                end_pos = max(span_end, span_start)
                response_prefix = response_tensor[:end_pos]
                branch_sequence = torch.cat([prompt_tensor, response_prefix], dim=-1)
                branch_sequences.append(branch_sequence)
                branch_nodes.append(node_chain[idx])

        if len(branch_sequences) == 0:
            return None

        device = branch_sequences[0].device
        input_ids, attention_mask, position_ids = _pad_and_stack(branch_sequences, pad_token_id=self.pad_token_id, device=device)

        meta_info = dict(gen_batch.meta_info)
        meta_info["branch_from_entropy"] = True

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

    def commit_branch_outputs(self, branch_output: DataProto, branch_plan: BranchPlan) -> None:
        """Attach branch responses to nodes and backpropagate rewards."""
        if branch_output is None or branch_output.batch is None:
            return
        responses = branch_output.batch.get("responses")
        if responses is None:
            return

        prompt_source = branch_output.batch.get("prompts")
        if prompt_source is None:
            prompt_source = branch_output.batch.get("input_ids")
        attn_mask = branch_output.batch.get("attention_mask")
        pos_ids = branch_output.batch.get("position_ids")

        for idx, (node, response) in enumerate(zip(branch_plan.nodes, responses)):
            if self.tokenizer is not None:
                node.text = self.tokenizer.decode(response, skip_special_tokens=True)
            # Currently reward is constant; can be replaced with a learned scorer
            node.backpropagate(node.reward if node.reward is not None else self.default_reward)

            # also record this branch response for downstream batching
            # compute step rewards for this branch response using the same scorer
            if self.tokenizer is not None:
                resp_text = self.tokenizer.decode(response, skip_special_tokens=True)
                segments = resp_text.split("\n\n") if resp_text else [""]
                prompt_text = self.tokenizer.decode(prompt_source[idx], skip_special_tokens=True) if prompt_source is not None else None
            else:
                prompt_text = None
                segments = [""]
            step_content: list[str] = []
            step_rewards: list[float] = []
            for seg in segments:
                step_content.append(seg)
                if self.branch_level == "token":
                    step_rewards.append(0.0)
                else:
                    step_rewards.append(self._compute_step_reward(seg, prompt_text, step_content))

            # compute spans for branch response
            step_spans = []
            if self.tokenizer is not None:
                cursor = 0
                for seg in segments:
                    seg_tokens = self.tokenizer.encode(seg, add_special_tokens=False)
                    start, end = cursor, cursor + len(seg_tokens)
                    step_spans.append((start, end))
                    cursor = end
            else:
                step_spans = [(0, response.size(0))]

            prompt_tensor = prompt_source[idx].clone() if prompt_source is not None else None
            prompt_id = self._get_prompt_id(prompt_tensor) if prompt_tensor is not None else None
            ground_truth = self.prompt_ground_truth.get(prompt_id) if prompt_id is not None else None

            self._record_response(
                prompt_tensor=prompt_tensor,
                response_tensor=response,
                attention_mask=attn_mask[idx] if attn_mask is not None else None,
                position_ids=pos_ids[idx] if pos_ids is not None else None,
                step_rewards=step_rewards,
                step_spans=step_spans,
                nodes=[node],
                leaf_node=node,
                ground_truth=ground_truth,
            )

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
                        node.state_value = float(np.mean(child_values))
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
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            nodes.append(n)
            for c in n.children:
                stack.append(c)

        non_root = [n for n in nodes if n is not root]
        if len(non_root) == 0:
            mean_val = 0.0
        else:
            mean_val = float(np.mean([n.state_value for n in non_root]))
        for n in nodes:
            n.global_state_value = mean_val

    # ------------------------------------------------------------------
    # Q/Return computation per step

    def compute_q_values(self, gamma: float = 0.99) -> None:
        """Compute discounted returns (Q values) for each recorded trajectory.

        Q_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        Stored on nodes (q_value) and ResponseRecord.step_q_values.
        """
        for record in self.response_records:
            rewards = record.step_rewards or []
            if not rewards:
                record.step_q_values = []
                continue
            q_vals = [0.0 for _ in rewards]
            running = 0.0
            for i in reversed(range(len(rewards))):
                running = rewards[i] + gamma * running
                q_vals[i] = running
            record.step_q_values = q_vals

            # attach to nodes if provided
            if record.nodes:
                for n, q in zip(record.nodes, q_vals):
                    n.q_value = q
            elif record.leaf_node is not None and q_vals:
                record.leaf_node.q_value = q_vals[0]

    def backpropagate_correctness(self) -> None:
        """Compute outcome correctness for leaves and backpropagate averaged values."""
        if self.tokenizer is None:
            return

        # 1) assign leaf values using compute_score
        for record in self.response_records:
            if record.leaf_node is None:
                continue
            response_str = self.tokenizer.decode(record.response_tensor, skip_special_tokens=True)
            gt = record.ground_truth
            try:
                score, _ = compute_score(response_str, gt) if gt is not None else (0.0, None)
            except Exception:
                score = 0.0
            record.leaf_node.state_value = float(score)

        # 2) bottom-up averaging for each tree
        for tree in self.trees.values():
            self._compute_state_value(tree.root)
            self._assign_global_state_value(tree.root)

        # 3) cache per-step state values on each recorded response
        for record in self.response_records:
            if record.nodes:
                record.step_state_values = [node.state_value for node in record.nodes]
            elif record.leaf_node is not None:
                record.step_state_values = [record.leaf_node.state_value]

    # Convenience method to inspect tree state for debugging
    def summary(self) -> Dict[str, int]:
        return {prompt_id: self._count_nodes(tree.root) for prompt_id, tree in self.trees.items()}

    def _count_nodes(self, node: Node) -> int:
        return 1 + sum(self._count_nodes(child) for child in node.children)

    def _record_response(
        self,
        prompt_tensor: Optional[torch.Tensor],
        response_tensor: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        step_rewards: List[float],
        step_spans: Optional[List[tuple[int, int]]] = None,
        nodes: Optional[List[Node]] = None,
        leaf_node: Optional[Node] = None,
        ground_truth: Optional[str] = None,
    ) -> None:
        """Store a response sample for later batching.

        step_spans are token index ranges (start, end) over the response tokens (excluding prompt).
        If not provided, they are computed using the tokenizer by splitting on blank lines.
        """

        spans = step_spans
        if spans is None:
            spans = []
            if self.tokenizer is not None:
                resp_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
                segments = resp_text.split("\n\n") if resp_text else [""]
                cursor = 0
                for seg in segments:
                    seg_tokens = self.tokenizer.encode(seg, add_special_tokens=False)
                    start, end = cursor, cursor + len(seg_tokens)
                    spans.append((start, end))
                    cursor = end
            else:
                spans = [(0, response_tensor.size(0))]

        record = ResponseRecord(
            prompt_tensor=prompt_tensor,
            response_tensor=response_tensor,
            attention_mask=attention_mask,
            position_ids=position_ids,
            step_rewards=step_rewards,
            step_spans=spans,
            nodes=nodes or [],
            leaf_node=leaf_node,
            ground_truth=ground_truth,
            step_q_values=None,
        )
        self.response_records.append(record)

    def build_response_batch(self, use_state_values: bool = True) -> Optional[DataProto]:
        """Aggregate all recorded responses into a DataProto with rewards for GRPO.

        If use_state_values is True, the returned reward_fn_scores will use the
        correctness-propagated state values; otherwise it will use random step rewards.
        In both cases, random step rewards are preserved in `step_reward_scores` and
        state values in `state_value_scores`.
        """
        if not self.response_records:
            return None

        device = self.response_records[0].response_tensor.device
        prompts = []
        responses = []
        step_spans_list: list[List[tuple[int, int]]] = []
        step_rewards_list: list[List[float]] = []
        step_state_values_list: list[List[float]] = []
        step_q_values_list: list[List[float]] = []
        response_lens: list[int] = []
        step_end_indices_list: list[List[int]] = []

        for rec in self.response_records:
            p = rec.prompt_tensor if rec.prompt_tensor is not None else torch.empty((0,), device=device, dtype=rec.response_tensor.dtype)
            prompts.append(p)
            responses.append(rec.response_tensor)
            step_spans_list.append(rec.step_spans)
            step_rewards_list.append(rec.step_rewards)
            if rec.step_state_values is not None:
                step_state_values_list.append(rec.step_state_values)
            else:
                step_state_values_list.append(rec.step_rewards)
            if rec.step_q_values is not None:
                step_q_values_list.append(rec.step_q_values)
            else:
                step_q_values_list.append(rec.step_rewards)
            response_lens.append(rec.response_tensor.size(0))
            # collect step end indices (inclusive)
            step_end_indices = []
            for _, end in rec.step_spans:
                end_pos = max(0, min(end - 1, rec.response_tensor.size(0) - 1))
                step_end_indices.append(end_pos)
            step_end_indices_list.append(step_end_indices)

        # concat prompt+response for full input ids
        full_sequences = [torch.cat([p, r], dim=-1) if p.numel() > 0 else r for p, r in zip(prompts, responses)]

        input_ids, attention_mask, position_ids = _pad_sequences(full_sequences, pad_token_id=self.pad_token_id, device=device)
        prompts_padded, _, _ = _pad_sequences(prompts, pad_token_id=self.pad_token_id, device=device)
        responses_padded, _, _ = _pad_sequences(responses, pad_token_id=self.pad_token_id, device=device)

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
        verifiable_rewards = chosen_scores.sum(dim=-1)

        # build score_ids (step end positions) and reward_mask
        max_steps = max((len(x) for x in step_end_indices_list), default=0)
        score_ids = torch.full((len(self.response_records), max_steps), -1, device=device, dtype=torch.long)
        reward_mask = torch.zeros_like(score_ids, dtype=torch.float32)
        for i, ends in enumerate(step_end_indices_list):
            for j, end_pos in enumerate(ends[:max_steps]):
                score_ids[i, j] = end_pos
                reward_mask[i, j] = 1.0

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
            non_tensors={},
            meta_info={},
        )
        return reward_proto

    # ------------------------------------------------------------------
    # Response recording and batching helpers


@dataclass
class ResponseRecord:
    prompt_tensor: Optional[torch.Tensor]
    response_tensor: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    step_rewards: List[float]
    step_spans: List[tuple[int, int]]
    nodes: List[Node]
    leaf_node: Optional[Node]
    ground_truth: Optional[str]
    step_state_values: Optional[List[float]] = None
    step_q_values: Optional[List[float]] = None


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
