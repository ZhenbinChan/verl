"""StepTreeRL sampling strategy.

Performs multi-round tree expansion where nodes are selected for branching
based on per-step average entropy (instead of UCT). Each generation may
produce multiple ``<step>...</step>`` blocks, which are split into individual
MCTSNodes. Branching candidates are drawn from all non-root nodes in each
tree, not only the current leaves.

Simplified parameters:
    rollout.n     — initial generation count per prompt
    top_k         — number of high-entropy steps to select per round
    iter_rounds   — number of branching rounds
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import torch

from verl import DataProto
from verl.trainer.ppo.sampling.base import SamplingResult, SamplingStrategy
from verl.trainer.ppo.sampling.mcts_node import (
    MCTSNode,
    collect_all_nodes,
    gather_path,
)
from verl.trainer.ppo.sampling.mcts_prm import get_prm_fn, format_step_reward


@contextmanager
def _timer(name: str, dest: dict):
    import time

    start = time.perf_counter()
    yield
    dest[name] = time.perf_counter() - start


class StepTreeRLStrategy(SamplingStrategy):
    """Step-level entropy-guided tree expansion strategy (Simplified).

    Multi-round tree expansion where:
    1. Initial complete solutions are parsed into steps
    2. Multi-round: select per-tree Top-K high-entropy nodes and branch
    3. Output all leaves with GPU padding

    Config key: ``trainer.step_treerl_config``
    """

    def __init__(self, config, tokenizer):
        cfg = config.trainer.get("step_treerl_config", {})

        # Simplified parameters
        self.max_depth = cfg.get("max_depth", 40)
        self.max_token_num = cfg.get("max_token_num", 4096)
        self.branch_max_new_tokens = cfg.get("branch_max_new_tokens", self.max_token_num)

        # New simplified params
        self.top_k = cfg.get("top_k", 2)
        self.iter_rounds = cfg.get("iter_rounds", 3)

        prm_type = cfg.get("prm", "format")

        # FOL verifier initialization (lazy loading)
        self.fol_verifier = None
        self.fol_metadata_map: Dict[str, "FOLMetadata"] = {}
        self._fol_metadata_loaded = False
        self._fol_metadata_path = cfg.get("fol_metadata_path", None)

        if prm_type == "fol" and self._fol_metadata_path:
            self._prm_type = prm_type
            self.step_prm_fn = get_prm_fn("format")
        else:
            self._prm_type = prm_type
            self.step_prm_fn = get_prm_fn(prm_type)

        self.tokenizer = tokenizer
        self.pad_token_id: int = getattr(tokenizer, "pad_token_id", 0) or 0
        self.eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)

        # Used for padding entropy-computation batches to a multiple of world size
        self._n_gpus = config.trainer.get("n_gpus_per_node", 1) * config.trainer.get("nnodes", 1)

        # Max model length from vLLM config (used to skip nodes that exceed limit)
        self.max_model_len = config.actor_rollout_ref.rollout.get("max_model_len", 4096)

        # Step boundary delimiter
        self.step_end_marker = "</step>"

    # ------------------------------------------------------------------
    # FOL Metadata Loading
    # ------------------------------------------------------------------

    def _load_fol_metadata(self, gen_batch: DataProto) -> None:
        if self._fol_metadata_loaded:
            return
        if self.fol_verifier is not None:
            return
        if not self._fol_metadata_path or not os.path.exists(self._fol_metadata_path):
            print(f"[FOL Warning] FOL metadata path not found: {self._fol_metadata_path}")
            return

        try:
            from verl.utils.fol_verifier import FOLVerifier, FOLMetadata
            import json

            with open(self._fol_metadata_path, "r", encoding="utf-8") as f:
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
            self.step_prm_fn = get_prm_fn(
                "fol",
                verifier=self.fol_verifier,
                metadata_map=self.fol_metadata_map,
            )
            self._fol_metadata_loaded = True
            print(f"[FOL] Loaded {len(self.fol_metadata_map)} FOL metadata entries")

        except Exception as e:
            print(f"[FOL Warning] Failed to load FOL metadata: {e}")

    def _get_sample_id(self, tree_idx: int, gen_batch: DataProto) -> str:
        if gen_batch.non_tensor_batch is not None:
            sample_ids = gen_batch.non_tensor_batch.get("sample_id", [])
            if sample_ids and tree_idx < len(sample_ids):
                return str(sample_ids[tree_idx])
            extra_info = gen_batch.non_tensor_batch.get("extra_info", [])
            if extra_info and tree_idx < len(extra_info):
                sample_id = extra_info[tree_idx].get("index") or extra_info[tree_idx].get("id")
                if sample_id is not None:
                    return str(sample_id)
        return str(tree_idx)

    # ------------------------------------------------------------------
    # SamplingStrategy interface
    # ------------------------------------------------------------------

    def run(
        self,
        gen_batch: DataProto,
        gen_batch_output: DataProto,
        generate_fn: Callable[[DataProto], DataProto],
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        timing_raw: dict,
    ) -> SamplingResult:
        device = gen_batch.batch["input_ids"].device

        with _timer("step_treerl", timing_raw):
            if self._prm_type == "fol":
                self._load_fol_metadata(gen_batch)

            batch_size = gen_batch.batch["input_ids"].size(0)

            # 1. Parse initial solutions into step chains
            roots = self._init_roots(gen_batch, device)
            initial_nodes = self._generate_full_solutions(gen_batch, gen_batch_output, roots, batch_size)
            candidate_pool = self._build_candidate_pool(roots, initial_nodes)
            self._score_new_candidates(initial_nodes, compute_log_prob_fn, device)

            # 2. Multi-round per-tree Top-K selection + branching
            self._branch_by_entropy(
                roots=roots,
                candidate_pool=candidate_pool,
                generate_fn=generate_fn,
                compute_log_prob_fn=compute_log_prob_fn,
                device=device,
            )

            # 3. Backpropagate
            self._backpropagate_all(roots, gen_batch)

            # 4. Build output with padding
            ground_truths = self._get_ground_truths(gen_batch, batch_size)
            result = self._build_output(gen_batch, roots, device, ground_truths)

        return result

    def _get_ground_truths(self, gen_batch: DataProto, batch_size: int) -> List[Optional[str]]:
        ground_truths: List[Optional[str]] = []
        if gen_batch.non_tensor_batch is not None and "answer" in gen_batch.non_tensor_batch:
            ground_truths = list(gen_batch.non_tensor_batch["answer"])
        ground_truths.extend([None] * max(0, batch_size - len(ground_truths)))
        return ground_truths

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_roots(self, gen_batch: DataProto, device: torch.device) -> List[MCTSNode]:
        input_ids = gen_batch.batch["input_ids"]
        attention_mask = gen_batch.batch["attention_mask"]
        roots: List[MCTSNode] = []
        for i in range(input_ids.size(0)):
            real_ids = input_ids[i][attention_mask[i].bool()].tolist()
            root = MCTSNode(
                state=real_ids,
                step_tokens=[],
                step_text="",
                accumulated_text="",
                parent=None,
                depth=0,
                terminal=False,
                tree_idx=i,
                node_idx=0,
            )
            roots.append(root)
        return roots

    # ------------------------------------------------------------------
    # Parse initial generation into step chains
    # ------------------------------------------------------------------

    def _generate_full_solutions(
        self,
        gen_batch: DataProto,
        gen_batch_output: DataProto,
        roots: List[MCTSNode],
        batch_size: int,
    ) -> List[MCTSNode]:
        """Parse initial complete solutions (from rollout.n) into step chains."""
        responses = gen_batch_output.batch["responses"]
        n_rollout = responses.size(0) // batch_size
        created_nodes: List[MCTSNode] = []

        for tree_idx in range(batch_size):
            root = roots[tree_idx]
            # Collect all responses for this prompt
            tree_responses = []
            for j in range(n_rollout):
                resp = responses[tree_idx * n_rollout + j]
                real_mask = resp != self.pad_token_id
                step_tokens_raw = resp[real_mask].tolist()
                tree_responses.append(step_tokens_raw)

            # Parse each response into steps
            for step_tokens_raw in tree_responses:
                hit_eos = (
                    self.eos_token_id is not None
                    and self.eos_token_id in step_tokens_raw
                )
                if hit_eos:
                    step_tokens_content = [t for t in step_tokens_raw if t != self.eos_token_id]
                else:
                    step_tokens_content = step_tokens_raw

                full_text = self.tokenizer.decode(step_tokens_content, skip_special_tokens=True)

                # Split by </step> into individual steps
                step_blocks = self._split_by_step_end(full_text, step_tokens_content)

                current_parent = root
                for block_text, block_tokens in step_blocks:
                    is_terminal = hit_eos or (current_parent.depth + 1 > self.max_depth)

                    accumulated_text = current_parent.accumulated_text + block_text
                    new_state = current_parent.state + block_tokens

                    # Compute PRM reward for this step
                    try:
                        r = format_step_reward(block_text)
                    except Exception:
                        r = 0.0

                    child = MCTSNode(
                        state=new_state,
                        step_tokens=block_tokens,
                        step_text=block_text,
                        accumulated_text=accumulated_text,
                        parent=current_parent,
                        depth=current_parent.depth + 1,
                        terminal=is_terminal,
                        visits=0,
                        R=r,
                        value=r,
                        tree_idx=tree_idx,
                        node_idx=0,
                    )
                    current_parent.children.append(child)
                    current_parent = child
                    created_nodes.append(child)

        return created_nodes

    def _split_by_step_end(
        self, full_text: str, full_tokens: List[int],
    ) -> List[Tuple[str, List[int]]]:
        """Split a generated text block by ``</step>`` boundaries.

        Returns a list of (step_text, step_tokens) pairs, one per ``<step>...</step>`` block.
        Each block includes the ``</step>`` closing tag.
        """
        import re

        step_blocks: List[Tuple[str, List[int]]] = []
        step_pattern = re.compile(r"(<step>.*?</step>)", re.DOTALL)
        matches = step_pattern.findall(full_text)

        if matches:
            for match in matches:
                match_tokens = self.tokenizer.encode(match, add_special_tokens=False)
                step_blocks.append((match, match_tokens))
            return step_blocks

        # Fallback: treat the entire text as one step
        return [(full_text, full_tokens)]

    # ------------------------------------------------------------------
    # Multi-round entropy-based branching
    # ------------------------------------------------------------------

    def _branch_by_entropy(
        self,
        roots: List[MCTSNode],
        candidate_pool: Dict[int, List[MCTSNode]],
        generate_fn: Callable[[DataProto], DataProto],
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        device: torch.device,
    ) -> None:
        """Multi-round: select per-tree Top-K high-entropy nodes and branch."""
        for _round_idx in range(self.iter_rounds):
            candidate_groups = self._collect_branch_candidates(roots, candidate_pool)
            if not candidate_groups:
                break

            selected: List[MCTSNode] = []
            for nodes in candidate_groups.values():
                if not nodes:
                    continue
                pairs = sorted(
                    ((node.cached_entropy, node) for node in nodes if node.cached_entropy is not None),
                    key=lambda x: x[0],
                    reverse=True,
                )
                k = min(self.top_k, len(pairs))
                selected.extend(node for _, node in pairs[:k])

            if not selected:
                break

            # Mark and branch from selected steps
            for step in selected:
                step.is_branch_point = True

            new_nodes = self._continue_from_steps(selected, generate_fn, device)
            if new_nodes:
                self._add_candidates(candidate_pool, new_nodes)
                self._score_new_candidates(new_nodes, compute_log_prob_fn, device)

    def _build_candidate_pool(self, roots: List[MCTSNode], initial_nodes: List[MCTSNode]) -> Dict[int, List[MCTSNode]]:
        candidate_pool: Dict[int, List[MCTSNode]] = {root.tree_idx: [] for root in roots}
        self._add_candidates(candidate_pool, initial_nodes)
        return candidate_pool

    def _add_candidates(self, candidate_pool: Dict[int, List[MCTSNode]], nodes: List[MCTSNode]) -> None:
        for node in nodes:
            if node.parent is None:
                continue
            bucket = candidate_pool.setdefault(node.tree_idx, [])
            if node not in bucket:
                bucket.append(node)

    def _score_new_candidates(
        self,
        nodes: List[MCTSNode],
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        device: torch.device,
    ) -> None:
        pending = [node for node in nodes if node.parent is not None and node.cached_entropy is None]
        if not pending:
            return
        entropies = self._compute_step_entropies(pending, compute_log_prob_fn, device)
        for node, entropy in zip(pending, entropies):
            node.cached_entropy = entropy

    def _collect_branch_candidates(
        self,
        roots: List[MCTSNode],
        candidate_pool: Dict[int, List[MCTSNode]],
    ) -> Dict[int, List[MCTSNode]]:
        """Collect cached non-root nodes for each tree."""
        candidates: Dict[int, List[MCTSNode]] = {}
        for root in roots:
            nodes = [
                node
                for node in candidate_pool.get(root.tree_idx, [])
                if node.parent is not None and node.cached_entropy is not None
            ]
            if nodes:
                candidates[root.tree_idx] = nodes
        return candidates

    def _continue_from_steps(
        self,
        steps: List[MCTSNode],
        generate_fn: Callable[[DataProto], DataProto],
        device: torch.device,
    ) -> List[MCTSNode]:
        """Generate continuation from selected nodes, preserving existing branches."""
        if not steps:
            return []

        branch_plans: List[Tuple[MCTSNode, int]] = []
        skipped_steps = 0
        for step in steps:
            remaining_budget = self.max_model_len - len(step.state)
            branch_budget = min(self.branch_max_new_tokens, self.max_token_num, remaining_budget)
            if branch_budget <= 0:
                skipped_steps += 1
                continue
            branch_plans.append((step, branch_budget))

        if not branch_plans:
            if skipped_steps > 0:
                print(f"[StepTreeRL] Skipped {skipped_steps} selected nodes due to exhausted context budget")
            return []

        active_steps = [step for step, _ in branch_plans]
        round_budget = min(branch_budget for _, branch_budget in branch_plans)

        seqs = [torch.tensor(s.state, dtype=torch.long, device=device) for s in active_steps]
        input_ids, attention_mask, position_ids = _pad_sequences(seqs, self.pad_token_id, device)

        batch_size = len(active_steps)
        ws = max(self._n_gpus, 1)
        padded_size = ((batch_size + ws - 1) // ws) * ws
        pad_slots = padded_size - batch_size

        if pad_slots > 0:
            input_ids = torch.cat([input_ids, input_ids[:pad_slots].clone()], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask[:pad_slots].clone()], dim=0)
            position_ids = torch.cat([position_ids, position_ids[:pad_slots].clone()], dim=0)

        data = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": input_ids.clone(),
            },
            non_tensors={},
            meta_info={"max_new_tokens": round_budget},
        )

        output = generate_fn(data)
        responses = output.batch["responses"]
        responses = responses[:batch_size]  # slice back

        total_new_nodes = 0
        total_duplicates = 0
        created_nodes: List[MCTSNode] = []

        for i, step in enumerate(active_steps):
            resp = responses[i]
            real_mask = resp != self.pad_token_id
            step_tokens_raw = resp[real_mask].tolist()

            hit_eos = (
                self.eos_token_id is not None
                and self.eos_token_id in step_tokens_raw
            )
            if hit_eos:
                step_tokens_content = [t for t in step_tokens_raw if t != self.eos_token_id]
            else:
                step_tokens_content = step_tokens_raw

            full_text = self.tokenizer.decode(step_tokens_content, skip_special_tokens=True)
            step_blocks = self._split_by_step_end(full_text, step_tokens_content)

            current_parent = step
            for block_text, block_tokens in step_blocks:
                is_terminal = hit_eos or (current_parent.depth + 1 > self.max_depth)

                accumulated_text = current_parent.accumulated_text + block_text
                new_state = current_parent.state + block_tokens

                # Deduplication is limited to sibling branches from the same parent.
                if self._is_duplicate_step(block_tokens, current_parent):
                    total_duplicates += 1
                    continue

                try:
                    r = format_step_reward(block_text)
                except Exception:
                    r = 0.0

                child = MCTSNode(
                    state=new_state,
                    step_tokens=block_tokens,
                    step_text=block_text,
                    accumulated_text=accumulated_text,
                    parent=current_parent,
                    depth=current_parent.depth + 1,
                    terminal=is_terminal,
                    visits=0,
                    R=r,
                    value=r,
                    tree_idx=step.tree_idx,
                    node_idx=0,
                )
                current_parent.children.append(child)
                current_parent = child
                total_new_nodes += 1
                created_nodes.append(child)

        if skipped_steps > 0:
            print(f"[StepTreeRL] Skipped {skipped_steps} selected nodes due to exhausted context budget")
        if total_duplicates > 0:
            print(f"[StepTreeRL] Deduplicated {total_duplicates} duplicate steps, added {total_new_nodes} new nodes")
        return created_nodes

    def _is_duplicate_step(self, block_tokens: List[int], parent_node: MCTSNode) -> bool:
        """Check whether the parent already has a child with identical step tokens."""
        for sibling in parent_node.children:
            if sibling.step_tokens == block_tokens:
                return True
        return False

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate_all(
        self,
        roots: List[MCTSNode],
        gen_batch: DataProto,
    ) -> None:
        """Backpropagate correctness through all leaf nodes."""
        for root in roots:
            tree_idx = root.tree_idx
            gt = self._get_ground_truths(gen_batch, len(roots))[tree_idx]

            # Collect all leaves
            leaves = [n for n in collect_all_nodes(root) if not n.children]

            if not leaves:
                continue

            # Mark correctness for each leaf
            for leaf in leaves:
                terminal_text = leaf.accumulated_text
                if gt is not None and terminal_text:
                    try:
                        from verl.utils.reward_score.logi import compute_score
                        score, _ = compute_score(terminal_text, gt)
                        leaf.is_correct = float(score) == 1.0
                    except Exception:
                        leaf.is_correct = False
                else:
                    leaf.is_correct = False

                if leaf.is_correct:
                    leaf.main_chain = True

            # Backpropagate for each leaf
            for leaf in leaves:
                self._leaf_backpropagate_correct(leaf)

    def _leaf_backpropagate_correct(self, leaf: MCTSNode) -> None:
        """Backward propagate: update correct/total counts in ancestors."""
        node = leaf
        while node.parent is not None:
            node.parent.terminal_in_subtree += 1
            if leaf.is_correct:
                node.parent.correct_terminal_in_subtree += 1
            node = node.parent

    # ------------------------------------------------------------------
    # Output construction with GPU padding
    # ------------------------------------------------------------------

    def _build_output(
        self,
        gen_batch: DataProto,
        roots: List[MCTSNode],
        device: torch.device,
        ground_truths: List[Optional[str]],
    ) -> SamplingResult:
        """Output all leaf nodes with GPU padding."""
        all_paths: List[List[MCTSNode]] = []
        all_gt: List[Optional[str]] = []

        for i, root in enumerate(roots):
            gt = ground_truths[i] if i < len(ground_truths) else None

            # Collect all leaves (initial + branched)
            leaves = [n for n in collect_all_nodes(root) if not n.children]

            if not leaves:
                # Fallback: deepest non-root node
                all_nodes = collect_all_nodes(root)
                non_root = [n for n in all_nodes if n.parent is not None]
                leaves = [max(non_root, key=lambda n: n.depth)] if non_root else [root]

            for leaf in leaves:
                all_paths.append(gather_path(leaf))
                all_gt.append(gt)

        # Auto padding to make divisible by n_gpus
        total = len(all_paths)
        remainder = total % self._n_gpus
        if remainder != 0:
            padding_needed = self._n_gpus - remainder
            for _ in range(padding_needed):
                all_paths.append(all_paths[-1])
                all_gt.append(all_gt[-1])
            print(f"[StepTreeRL] Padded {padding_needed} samples to make {total + padding_needed} divisible by {self._n_gpus}")

        return _build_sampling_result(all_paths, all_gt, self.pad_token_id, device, len(roots))

    # ------------------------------------------------------------------
    # Helper methods (kept for compatibility)
    # ------------------------------------------------------------------

    def _compute_step_entropies(
        self,
        nodes: List[MCTSNode],
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        device: torch.device,
    ) -> List[float]:
        """Batch-compute per-step average ``-log p(token)`` for a list of nodes."""
        if not nodes:
            return []

        prompt_prefixes: List[List[int]] = [
            node.parent.state if node.parent is not None else node.state
            for node in nodes
        ]
        max_prompt_len = max(len(prefix) for prefix in prompt_prefixes)
        max_step_len = max(len(node.step_tokens) for node in nodes)
        if max_step_len == 0:
            return [0.0] * len(nodes)

        batch_size = len(nodes)
        total_len = max_prompt_len + max_step_len

        full_input_ids = torch.full((batch_size, total_len), self.pad_token_id, dtype=torch.long, device=device)
        full_attention_mask = torch.zeros((batch_size, total_len), dtype=torch.long, device=device)
        responses = torch.full((batch_size, max_step_len), self.pad_token_id, dtype=torch.long, device=device)

        for i, node in enumerate(nodes):
            prompt_tokens = prompt_prefixes[i]
            state_len = len(prompt_tokens)
            step_len = len(node.step_tokens)
            p_offset = max_prompt_len - state_len
            full_input_ids[i, p_offset:max_prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
            full_attention_mask[i, p_offset:max_prompt_len] = 1
            full_input_ids[i, max_prompt_len:max_prompt_len + step_len] = torch.tensor(
                node.step_tokens, dtype=torch.long, device=device,
            )
            full_attention_mask[i, max_prompt_len:max_prompt_len + step_len] = 1
            responses[i, :step_len] = torch.tensor(node.step_tokens, dtype=torch.long, device=device)

        position_ids_full = full_attention_mask.long().cumsum(dim=-1) - 1
        position_ids_full.masked_fill_(full_attention_mask == 0, 0)

        orig_batch_size = len(nodes)
        ws = max(self._n_gpus, 1)
        padded_size = ((orig_batch_size + ws - 1) // ws) * ws
        pad_slots = padded_size - orig_batch_size

        if pad_slots > 0:
            full_input_ids = torch.cat([full_input_ids, full_input_ids[:pad_slots].clone()], dim=0)
            full_attention_mask = torch.cat([full_attention_mask, full_attention_mask[:pad_slots].clone()], dim=0)
            position_ids_full = torch.cat([position_ids_full, position_ids_full[:pad_slots].clone()], dim=0)
            responses = torch.cat([responses, responses[:pad_slots].clone()], dim=0)

        data = DataProto.from_dict(
            tensors={
                "input_ids": full_input_ids,
                "attention_mask": full_attention_mask,
                "position_ids": position_ids_full,
                "prompts": full_input_ids[:, :max_prompt_len],
                "responses": responses,
            },
            non_tensors={},
            meta_info={
                "micro_batch_size": padded_size,
                "max_token_len": total_len,
                "use_dynamic_bsz": False,
                "temperature": 1.0,
            },
        )

        log_prob_output = compute_log_prob_fn(data)
        old_log_probs = log_prob_output.batch["old_log_probs"]
        old_log_probs = old_log_probs[:orig_batch_size]

        step_entropies: List[float] = []
        for i, node in enumerate(nodes):
            step_len = len(node.step_tokens)
            if step_len > 0:
                step_log_probs = old_log_probs[i, :step_len]
                step_entropy = float(-step_log_probs.mean().item())
            else:
                step_entropy = 0.0
            step_entropies.append(step_entropy)

        return step_entropies

    # ==================================================================
    # END OF SIMPLIFIED STEPTREERL IMPLEMENTATION
    # Old methods below are deprecated and kept for reference only
    # ==================================================================


def _build_sampling_result(
    all_paths: List[List[MCTSNode]],
    all_gt: List[Optional[str]],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
) -> SamplingResult:
    """Build padded tensors and SamplingResult from selected paths."""
    prompt_ids_list: List[torch.Tensor] = []
    full_seq_list: List[torch.Tensor] = []
    resp_ids_list: List[torch.Tensor] = []
    step_spans_list: List[List[Tuple[int, int]]] = []
    step_rewards_list: List[List[float]] = []
    step_correctness_scores_list: List[List[float]] = []
    response_lens: List[int] = []
    verifiable_rewards_list: List[float] = []

    for path, gt in zip(all_paths, all_gt):
        root_node = path[0].parent
        prompt_tokens = torch.tensor(root_node.state, dtype=torch.long, device=device)

        response_tokens: List[int] = []
        spans: List[Tuple[int, int]] = []
        rewards: List[float] = []
        correctness_scores: List[float] = []
        offset = 0
        for node in path:
            tokens = node.step_tokens
            if not tokens:
                continue
            start, end = offset, offset + len(tokens)
            spans.append((start, end))
            rewards.append(node.R)
            if node.terminal_in_subtree > 0:
                v_score = node.correct_terminal_in_subtree / node.terminal_in_subtree
            else:
                v_score = 0.0
            correctness_scores.append(v_score)
            response_tokens.extend(tokens)
            offset = end

        resp_tensor = torch.tensor(response_tokens, dtype=torch.long, device=device)
        full_tensor = torch.cat([prompt_tokens, resp_tensor], dim=0)

        prompt_ids_list.append(prompt_tokens)
        full_seq_list.append(full_tensor)
        resp_ids_list.append(resp_tensor)
        step_spans_list.append(spans)
        step_rewards_list.append(rewards)
        step_correctness_scores_list.append(correctness_scores)
        response_lens.append(len(response_tokens))
        verifiable_rewards_list.append(1.0 if path[-1].is_correct else 0.0)

    # Pad and stack
    input_ids, attention_mask, position_ids = _pad_sequences(full_seq_list, pad_token_id, device)
    prompts_padded, _, _ = _pad_sequences(prompt_ids_list, pad_token_id, device)
    responses_padded, _, _ = _pad_sequences(resp_ids_list, pad_token_id, device)

    # Token-level PRM scores
    reward_fn_scores = _build_token_level_scores(
        responses=responses_padded,
        response_lens=response_lens,
        all_step_spans=step_spans_list,
        all_step_rewards=step_rewards_list,
    )

    # score_ids and reward_mask
    n_paths = len(all_paths)
    max_steps = max((len(s) for s in step_spans_list), default=1)
    score_ids = torch.full((n_paths, max_steps), -1, device=device, dtype=torch.long)
    reward_mask = torch.zeros(n_paths, max_steps, device=device, dtype=torch.float32)
    for i, (spans, rlen) in enumerate(zip(step_spans_list, response_lens)):
        for j, (_, end) in enumerate(spans[:max_steps]):
            end_pos = max(0, min(end - 1, rlen - 1)) if rlen > 0 else 0
            score_ids[i, j] = end_pos
            reward_mask[i, j] = 1.0

    # ORM verifiable rewards
    max_resp_len = responses_padded.size(1)
    verifiable_rewards = torch.zeros(n_paths, max_resp_len, dtype=torch.float32, device=device)
    for i, rlen in enumerate(response_lens):
        if rlen > 0:
            verifiable_rewards[i, rlen - 1] = verifiable_rewards_list[i]

    # step_correctness_scores
    step_correctness_padded = torch.full((n_paths, max_steps), 0.0, device=device, dtype=torch.float32)
    for i, scores in enumerate(step_correctness_scores_list):
        for j, s in enumerate(scores[:max_steps]):
            step_correctness_padded[i, j] = s

    output = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": prompts_padded,
            "responses": responses_padded,
            "reward_fn_scores": reward_fn_scores,
            "score_ids": score_ids,
            "reward_mask": reward_mask,
            "verifiable_rewards": verifiable_rewards,
            "step_correctness_scores": step_correctness_padded,
        },
        non_tensors={},
        meta_info={},
    )

    repeat_times = n_paths // batch_size if batch_size > 0 else 1
    return SamplingResult(gen_batch_output=output, repeat_times=repeat_times)


# ------------------------------------------------------------------
# Tensor helpers (shared with parallel_mcts)
# ------------------------------------------------------------------

def _pad_sequences(
    seqs: List[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(s.size(0) for s in seqs)
    dtype = seqs[0].dtype
    batch = torch.full((len(seqs), max_len), pad_token_id, dtype=dtype, device=device)
    attn = torch.zeros((len(seqs), max_len), dtype=dtype, device=device)
    for i, seq in enumerate(seqs):
        l = seq.size(0)
        batch[i, :l] = seq
        attn[i, :l] = 1
    pos = torch.arange(max_len, device=device).unsqueeze(0).expand(len(seqs), max_len)
    return batch, attn, pos


def _build_token_level_scores(
    responses: torch.Tensor,
    response_lens: List[int],
    all_step_spans: List[List[Tuple[int, int]]],
    all_step_rewards: List[List[float]],
) -> torch.Tensor:
    """Broadcast each step's PRM reward to all tokens within that step span."""
    scores = torch.zeros_like(responses, dtype=torch.float32)
    max_len = responses.size(1)
    for i, (rlen, spans, rewards) in enumerate(zip(response_lens, all_step_spans, all_step_rewards)):
        for (s, e), r in zip(spans, rewards):
            if rlen <= 0:
                continue
            start = max(0, s)
            end = min(e, rlen, max_len)
            if start >= end:
                continue
            scores[i, start:end] = float(r)
    return scores
