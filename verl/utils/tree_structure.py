"""
MCTS-style search tree utilities for branching on high-entropy response steps.

This module provides a lightweight Node/SearchTree abstraction plus a TreeManager
that can be used from trainers to track per-prompt search trees. It also offers a
helper to build new branch inputs from the highest-entropy step of each response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import hashlib
import torch
import random
import numpy as np

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
        visits: Number of times this node has been visited/updated.
        value_sum: Accumulated value used for backpropagation.
    """

    step_idx: int
    entropy: float
    reward: float = 0.0
    text: str = ""
    parent: Optional["Node"] = None
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

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

    def __init__(self, tokenizer=None, pad_token_id: int | None = None, default_reward: float = 1.0):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id if pad_token_id is not None else (tokenizer.pad_token_id if tokenizer is not None else 0)
        self.default_reward = default_reward
        self.trees: Dict[str, SearchTree] = {}
        # a stub scorer that can be overridden later
        self.step_scorer = self._random_step_scorer
        # records of all responses generated for the current batch (including branches)
        self.response_records: list[ResponseRecord] = []

    def _random_step_scorer(self, step_text: str) -> float:
        """Placeholder step scoring: randomly returns 0 or 1.

        Replace this with a real scoring function as needed.
        """
        return float(random.randint(0, 1))

    def __repr__(self) -> str:
        summaries = [f"prompt_id={pid}, nodes={self._count_nodes(tree.root)}" for pid, tree in self.trees.items()]
        inner = "; ".join(summaries) if summaries else "empty"
        return f"TreeManager({inner})"

    def _get_prompt_id(self, prompt_tensor: torch.Tensor) -> str:
        # Stable, content-based id: md5 over tensor bytes
        data = prompt_tensor.detach().cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()

    def ensure_tree(self, prompt_tensor: torch.Tensor, prompt_text: Optional[str] = None) -> SearchTree:
        prompt_id = self._get_prompt_id(prompt_tensor)
        if prompt_id not in self.trees:
            self.trees[prompt_id] = SearchTree(prompt_id=prompt_id, prompt_text=prompt_text)
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
            self.ensure_tree(prompt_tensor, prompt_text=prompt_text)

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

        log_prob_output = compute_log_prob_fn(gen_batch_output)
        entropies = log_prob_output.batch.get("entropys") if log_prob_output is not None else None
        if entropies is None:
            return None

        response_mask = _compute_response_mask(gen_batch_output)
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
            # Decode response and split into steps by blank line
            if self.tokenizer is None:
                # Without tokenizer, fall back to whole response as one step
                segments = [""]
                step_token_spans = [(0, response_tensor.size(0))]
            else:
                resp_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
                segments = resp_text.split("\n\n") if resp_text else [""]
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
            tree = self.ensure_tree(prompt_tensor, prompt_text=prompt_text)

            parent = tree.root
            node_chain: list[Node] = []
            step_rewards: list[float] = []
            for (s, e), seg_text, seg_ent in zip(step_token_spans, segments, step_entropies):
                # use stub scorer to assign reward (random 0/1 by default)
                seg_reward = self.step_scorer(seg_text)
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
            )

            # select top-k steps by entropy (per sample)
            k = max(1, top_k)
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
            else:
                segments = [""]
            step_rewards = [self.step_scorer(seg) for seg in segments]

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

            self._record_response(
                prompt_tensor=prompt_source[idx].clone() if prompt_source is not None else None,
                response_tensor=response,
                attention_mask=attn_mask[idx] if attn_mask is not None else None,
                position_ids=pos_ids[idx] if pos_ids is not None else None,
                step_rewards=step_rewards,
                step_spans=step_spans,
            )

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
        )
        self.response_records.append(record)

    def build_response_batch(self) -> Optional[DataProto]:
        """Aggregate all recorded responses into a DataProto with rewards for GRPO."""
        if not self.response_records:
            return None

        device = self.response_records[0].response_tensor.device
        prompts = []
        responses = []
        step_spans_list: list[List[tuple[int, int]]] = []
        step_rewards_list: list[List[float]] = []
        response_lens: list[int] = []

        for rec in self.response_records:
            p = rec.prompt_tensor if rec.prompt_tensor is not None else torch.empty((0,), device=device, dtype=rec.response_tensor.dtype)
            prompts.append(p)
            responses.append(rec.response_tensor)
            step_spans_list.append(rec.step_spans)
            step_rewards_list.append(rec.step_rewards)
            response_lens.append(rec.response_tensor.size(0))

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
        verifiable_rewards = token_level_scores.sum(dim=-1)

        reward_proto = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompts_padded,
                "responses": responses_padded,
                "reward_fn_scores": token_level_scores,
                "verifiable_rewards": verifiable_rewards,
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
