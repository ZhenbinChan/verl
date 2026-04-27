from __future__ import annotations

import json
import os
import random
from collections import deque
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import torch

from verl import DataProto
from verl.trainer.ppo.sampling.base import SamplingResult, SamplingStrategy
from verl.trainer.ppo.sampling.mcts_node import (
    MCTSNode,
    backpropagate,
    collect_all_nodes,
    collect_expandable,
    gather_path,
    leaf_normalize,
    select_terminal,
    uct,
)
from verl.trainer.ppo.sampling.mcts_prm import get_prm_fn, format_step_reward


@contextmanager
def _timer(name: str, dest: dict):
    import time
    start = time.perf_counter()
    yield
    dest[name] = time.perf_counter() - start


def _weighted_sample_no_replacement(
    candidates: List[MCTSNode],
    weights: List[float],
    k: int,
) -> List[MCTSNode]:
    """Importance sampling without replacement using pre-computed weights."""
    k = min(k, len(candidates))
    if k <= 0:
        return []
    candidates = list(candidates)
    weights = list(weights)
    selected: List[MCTSNode] = []
    for _ in range(k):
        if not candidates:
            break
        try:
            chosen = random.choices(candidates, weights=weights, k=1)[0]
        except (ValueError, IndexError):
            chosen = random.choice(candidates)
        idx = candidates.index(chosen)
        selected.append(candidates.pop(idx))
        weights.pop(idx)
    return selected


class ParallelMCTSStrategy(SamplingStrategy):
    """Parallel MCTS sampling strategy integrated into the verl training loop.

    Replaces the vLLM/API node expansion from mcts_utils with generate_fn
    (actor_rollout_wg.generate_sequences).  Each expansion round packs all
    selected node prefixes into a single DataProto, calls generate_fn once,
    then attaches the resulting steps as child MCTSNodes.

    The Process Reward Model (PRM) scores each step immediately after expansion.
    Supported PRM types: 'format' (tag-structure check) and 'fol' (FOL/Z3 verification).

    Config key: trainer.parallel_mcts_config
    """

    def __init__(self, config, tokenizer):
        cfg = config.trainer.get("parallel_mcts_config", {})
        self.max_nodes = cfg.get("max_nodes", 20)
        self.max_depth = cfg.get("max_depth", 40)
        self.max_children = cfg.get("max_children", 3)
        self.concurrent_num = cfg.get("concurrent_num", 4)
        self.pass_k = cfg.get("pass_k", 4)
        self.num_traces = cfg.get("num_traces", 4)
        self.exploration_c = cfg.get("exploration_constant", 1.0)
        self.max_token_num = cfg.get("max_token_num", 512)
        self.do_backprop = cfg.get("backprop", True)
        self.random_pick = cfg.get("random_pick", False)
        self.selection_policy = cfg.get("selection_policy", "importance_sampling")
        self.gamma = cfg.get("gamma", 0.9)
        self.use_weighted_value = cfg.get("use_weighted_value", False)
        self.normalize_style = cfg.get("normalize_style", "step")
        self.average_one_generation = cfg.get("average_one_generation", False)

        prm_type = cfg.get("prm", "format")

        # FOL verifier initialization (lazy loading)
        self.fol_verifier = None
        self.fol_metadata_map: Dict[str, "FOLMetadata"] = {}
        self._fol_metadata_loaded = False
        self._fol_metadata_path = cfg.get("fol_metadata_path", None)

        # Create PRM function
        if prm_type == "fol" and self._fol_metadata_path:
            # Defer FOL verifier initialization to first run() call
            # when we have access to gen_batch data
            self._prm_type = prm_type
            self.step_prm_fn = get_prm_fn("format")  # Temporary fallback
        else:
            self._prm_type = prm_type
            self.step_prm_fn = get_prm_fn(prm_type)

        self.tokenizer = tokenizer
        self.pad_token_id: int = getattr(tokenizer, "pad_token_id", 0) or 0
        self.eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)

    # ------------------------------------------------------------------
    # FOL Metadata Loading
    # ------------------------------------------------------------------

    def _load_fol_metadata(self, gen_batch: DataProto) -> None:
        """Load FOL metadata from file and initialize FOL verifier."""
        if self._fol_metadata_loaded:
            return
        if self.fol_verifier is not None:
            return
        if not self._fol_metadata_path or not os.path.exists(self._fol_metadata_path):
            print(f"[FOL Warning] FOL metadata path not found: {self._fol_metadata_path}")
            return

        try:
            from verl.utils.fol_verifier import FOLVerifier, FOLMetadata

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
        """Get sample_id for a given tree index."""
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

        with _timer("parallel_mcts", timing_raw):
            # Load FOL metadata on first run if prm='fol'
            if self._prm_type == "fol":
                self._load_fol_metadata(gen_batch)

            roots = self._init_roots(gen_batch, device)
            batch_size = len(roots)
            node_counts: List[int] = [1] * batch_size
            depth_counts: List[Dict[int, int]] = [{0: 1} for _ in range(batch_size)]
            leaves: List[List[MCTSNode]] = [[] for _ in range(batch_size)]

            while max(node_counts) < self.max_nodes:
                selected = self._select_nodes(roots, depth_counts)
                if not selected:
                    break
                self._expand(selected, generate_fn, device, node_counts, depth_counts, leaves, gen_batch)
                if all(len(leaves[i]) >= self.pass_k for i in range(batch_size)):
                    break

            # Extract ground truth answers for ORM correctness computation
            ground_truths: List[Optional[str]] = []
            if (
                gen_batch.non_tensor_batch is not None
                and "answer" in gen_batch.non_tensor_batch
            ):
                ground_truths = list(gen_batch.non_tensor_batch["answer"])
            ground_truths.extend([None] * max(0, batch_size - len(ground_truths)))

            result = self._build_output(gen_batch, roots, leaves, device, ground_truths)

        return result

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_roots(self, gen_batch: DataProto, device: torch.device) -> List[MCTSNode]:
        input_ids = gen_batch.batch["input_ids"]        # [B, L_prompt]
        attention_mask = gen_batch.batch["attention_mask"]  # [B, L_prompt]
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
    # Node selection (UCT)
    # ------------------------------------------------------------------

    def _select_nodes(
        self,
        roots: List[MCTSNode],
        depth_counts: List[Dict[int, int]],
    ) -> List[MCTSNode]:
        selected: List[MCTSNode] = []
        for root, dc in zip(roots, depth_counts):
            candidates = collect_expandable(root, dc, self.max_children)
            if not candidates:
                continue
            k = self.concurrent_num
            if self.random_pick:
                picked = random.sample(candidates, min(k, len(candidates)))
            elif self.selection_policy == "greedy":
                picked = sorted(candidates, key=lambda n: uct(n, self.exploration_c), reverse=True)[:k]
            else:  # importance_sampling (default)
                weights = [uct(n, self.exploration_c) for n in candidates]
                picked = _weighted_sample_no_replacement(candidates, weights, k)
            selected.extend(picked)
        return selected

    # ------------------------------------------------------------------
    # Node expansion
    # ------------------------------------------------------------------

    def _expand(
        self,
        nodes: List[MCTSNode],
        generate_fn: Callable[[DataProto], DataProto],
        device: torch.device,
        node_counts: List[int],
        depth_counts: List[Dict[int, int]],
        leaves: List[List[MCTSNode]],
        gen_batch: DataProto,
    ) -> None:
        branch_batch = self._pack_dataproto(nodes, self.max_children, device)
        branch_output = generate_fn(branch_batch)
        self._unpack_and_attach(nodes, branch_output, self.max_children, node_counts, depth_counts, leaves, gen_batch)

    def _pack_dataproto(
        self,
        nodes: List[MCTSNode],
        n_expand: int,
        device: torch.device,
    ) -> DataProto:
        """Pack node prefixes into a DataProto for generate_fn.

        Each node is repeated n_expand times so the actor generates
        n_expand independent step continuations per node.
        Follows the same contract as prepare_branches() in
        verl/utils/tree_structure.py.
        """
        seqs: List[torch.Tensor] = []
        for node in nodes:
            t = torch.tensor(node.state, dtype=torch.long, device=device)
            for _ in range(n_expand):
                seqs.append(t)

        input_ids, attention_mask, position_ids = _pad_sequences(seqs, self.pad_token_id, device)

        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": input_ids.clone(),
            },
            non_tensors={},
            meta_info={"max_new_tokens": self.max_token_num},
        )

    def _unpack_and_attach(
        self,
        nodes: List[MCTSNode],
        branch_output: DataProto,
        n_expand: int,
        node_counts: List[int],
        depth_counts: List[Dict[int, int]],
        leaves: List[List[MCTSNode]],
        gen_batch: DataProto,
    ) -> None:
        """Decode generate_fn output and attach children to parent nodes."""
        responses = branch_output.batch["responses"]  # [N*n_expand (* rollout_n), resp_len]

        # generate_fn may return rollout.n samples per input; take the first of each group
        expected = len(nodes) * n_expand
        actual = responses.size(0)
        rollout_n = actual // expected if actual >= expected else 1
        if rollout_n > 1:
            responses = responses[::rollout_n]

        for i, node in enumerate(nodes):
            tree_idx = node.tree_idx
            for j in range(n_expand):
                resp = responses[i * n_expand + j]

                # Strip right-padding
                real_mask = resp != self.pad_token_id
                step_tokens_raw = resp[real_mask].tolist()

                # Detect EOS
                hit_eos = (
                    self.eos_token_id is not None
                    and self.eos_token_id in step_tokens_raw
                )
                if hit_eos:
                    step_tokens_content = [t for t in step_tokens_raw if t != self.eos_token_id]
                else:
                    step_tokens_content = step_tokens_raw

                step_text = self.tokenizer.decode(step_tokens_content, skip_special_tokens=True)

                # Truncate at </step> boundary if present
                step_end = "</step>"
                if step_end in step_text:
                    cut = step_text.index(step_end) + len(step_end)
                    step_text = step_text[:cut]
                    step_tokens_content = self.tokenizer.encode(step_text, add_special_tokens=False)
                    hit_eos = False  # properly closed step → not terminal

                is_terminal = hit_eos or (node.depth + 1 > self.max_depth)
                accumulated_text = node.accumulated_text + step_text
                new_state = node.state + step_tokens_content

                node_idx = node_counts[tree_idx]
                node_counts[tree_idx] += 1
                depth_counts[tree_idx][node.depth + 1] = (
                    depth_counts[tree_idx].get(node.depth + 1, 0) + 1
                )

                # Compute PRM reward
                try:
                    if self.fol_verifier is not None and str(tree_idx) in self.fol_metadata_map:
                        # FOL verification mode
                        sample_id = self._get_sample_id(tree_idx, gen_batch)
                        r = self.step_prm_fn(step_text, sample_id=sample_id)
                    else:
                        # Fallback to format check
                        r = format_step_reward(step_text)
                except NotImplementedError:
                    r = 0.0
                except Exception:
                    r = 0.0

                child = MCTSNode(
                    state=new_state,
                    step_tokens=step_tokens_content,
                    step_text=step_text,
                    accumulated_text=accumulated_text,
                    parent=node,
                    depth=node.depth + 1,
                    terminal=is_terminal,
                    visits=0,
                    R=r,
                    value=r,
                    tree_idx=tree_idx,
                    node_idx=node_idx,
                )
                node.children.append(child)

                if self.do_backprop:
                    backpropagate(child, self.gamma)

                if is_terminal:
                    leaves[tree_idx].append(child)

    # ------------------------------------------------------------------
    # Output construction
    # ------------------------------------------------------------------

    def _build_output(
        self,
        gen_batch: DataProto,
        roots: List[MCTSNode],
        leaves: List[List[MCTSNode]],
        device: torch.device,
        ground_truths: List[Optional[str]],
    ) -> SamplingResult:
        from verl.utils.reward_score.logi import compute_score
        from verl.trainer.ppo.sampling.mcts_node import (
            leaf_backpropagate,
            normalize_all_steps,
            selected_backpropagate,
            compute_accumulated_value,
            compute_weighted_update,
        )

        batch_size = len(roots)
        ground_truths: List[Optional[str]] = []
        if gen_batch.non_tensor_batch is not None and "answer" in gen_batch.non_tensor_batch:
            ground_truths = list(gen_batch.non_tensor_batch.get("answer", []))
        ground_truths.extend([None] * max(0, batch_size - len(ground_truths)))

        # Per-prompt: select terminal leaves → gather training paths
        all_paths: List[List[MCTSNode]] = []
        all_gt: List[Optional[str]] = []

        for i, (root, tree_leaves) in enumerate(zip(roots, leaves)):
            gt = ground_truths[i] if i < len(ground_truths) else None
            leaf_normalize(tree_leaves)

            # Fallback: if no terminal leaves, use the deepest nodes
            if not tree_leaves:
                all_nodes = collect_all_nodes(root)
                non_root = [n for n in all_nodes if n.parent is not None]
                tree_leaves = [max(non_root, key=lambda n: n.depth)] if non_root else [root]

            # --- Compute is_correct + main_chain for each leaf ---
            for leaf in tree_leaves:
                terminal_text = leaf.accumulated_text
                if gt is not None and terminal_text:
                    # Try FOL verification first
                    if self.fol_verifier is not None and str(i) in self.fol_metadata_map:
                        try:
                            sample_id = self._get_sample_id(i, gen_batch)
                            metadata = self.fol_metadata_map[str(sample_id)]
                            reward = self.fol_verifier.verify_step(metadata, terminal_text, use_llm=True)
                            leaf.is_correct = (reward == 1.0)
                        except Exception:
                            # Fallback to compute_score
                            try:
                                score, _ = compute_score(terminal_text, gt)
                                leaf.is_correct = float(score) == 1.0
                            except Exception:
                                leaf.is_correct = False
                    else:
                        # Fallback to compute_score
                        try:
                            score, _ = compute_score(terminal_text, gt)
                            leaf.is_correct = float(score) == 1.0
                        except Exception:
                            leaf.is_correct = False
                else:
                    leaf.is_correct = None
                if leaf.is_correct:
                    leaf.main_chain = True

            # --- Backpropagate leaf correctness stats up the tree ---
            for leaf in tree_leaves:
                leaf_backpropagate(leaf)

            # --- Normalize accumulated_value across steps ---
            normalize_all_steps(root, style=self.normalize_style)

            # --- Optional: average_one_generation ---
            if self.average_one_generation:
                compute_accumulated_value(root)

            selected = select_terminal(tree_leaves, self.num_traces)

            # --- Optional: weighted update ---
            if self.use_weighted_value:
                for leaf in selected:
                    selected_backpropagate(leaf)
                compute_weighted_update(root)

            for leaf in selected:
                all_paths.append(gather_path(leaf))
                all_gt.append(gt)

        # Build tensors for each path
        prompt_ids_list: List[torch.Tensor] = []
        full_seq_list: List[torch.Tensor] = []
        resp_ids_list: List[torch.Tensor] = []
        step_spans_list: List[List[Tuple[int, int]]] = []
        step_rewards_list: List[List[float]] = []
        step_correctness_scores_list: List[List[float]] = []  # V = correct/terminal per step
        response_lens: List[int] = []
        verifiable_rewards_list: List[float] = []

        for path, gt in zip(all_paths, all_gt):
            # Prompt tokens come from the root (parent of first path node)
            root_node = path[0].parent  # root is always the direct parent of path[0]
            prompt_tokens = torch.tensor(root_node.state, dtype=torch.long, device=device)

            # Reconstruct response by concatenating step tokens along the path
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
                # V_baseline = correct_terminal_in_subtree / terminal_in_subtree
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
            # ORM: 0/1 from is_correct (computed in tree processing block above)
            verifiable_rewards_list.append(1.0 if path[-1].is_correct else 0.0)

        # Pad and stack sequences
        input_ids, attention_mask, position_ids = _pad_sequences(full_seq_list, self.pad_token_id, device)
        prompts_padded, _, _ = _pad_sequences(prompt_ids_list, self.pad_token_id, device)
        responses_padded, _, _ = _pad_sequences(resp_ids_list, self.pad_token_id, device)

        # Token-level PRM scores (broadcast reward to step span)
        reward_fn_scores = _build_token_level_scores(
            responses=responses_padded,
            response_lens=response_lens,
            all_step_spans=step_spans_list,
            all_step_rewards=step_rewards_list,
        )

        # score_ids: last token position of each step; reward_mask: 1.0 at valid steps
        n_paths = len(all_paths)
        max_steps = max((len(s) for s in step_spans_list), default=1)
        score_ids = torch.full((n_paths, max_steps), -1, device=device, dtype=torch.long)
        reward_mask = torch.zeros(n_paths, max_steps, device=device, dtype=torch.float32)
        for i, (spans, rlen) in enumerate(zip(step_spans_list, response_lens)):
            for j, (_, end) in enumerate(spans[:max_steps]):
                end_pos = max(0, min(end - 1, rlen - 1)) if rlen > 0 else 0
                score_ids[i, j] = end_pos
                reward_mask[i, j] = 1.0

        # ORM verifiable rewards: [B, L_resp], ORM 0/1 at the last valid token, 0 elsewhere
        max_resp_len = responses_padded.size(1)
        verifiable_rewards = torch.zeros(n_paths, max_resp_len, dtype=torch.float32, device=device)
        for i, rlen in enumerate(response_lens):
            if rlen > 0:
                verifiable_rewards[i, rlen - 1] = verifiable_rewards_list[i]

        # step_correctness_scores: [B, max_steps], V per step = correct/terminal
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
                "verifiable_rewards": verifiable_rewards,  # ORM 0/1 per path
                "step_correctness_scores": step_correctness_padded,  # V = correct/terminal per step
            },
            non_tensors={},
            meta_info={},
        )

        repeat_times = n_paths // batch_size if batch_size > 0 else self.num_traces
        return SamplingResult(gen_batch_output=output, repeat_times=repeat_times)


# ------------------------------------------------------------------
# Module-level tensor helpers (mirror tree_structure.py)
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
