"""Information Gain sampling strategy (KL-StepTreeRL) — Simplified Version.

Performs multi-round tree expansion:
1. Generate M complete solutions (using verl's rollout.n)
2. Split each solution into steps
3. Multi-round: select Top-K high-KL steps and branch
4. Backpropagate: V = correct_leaves / total_leaves + PRM
5. Output all leaves (with padding for GPU divisibility)

Simplified parameters:
- rollout.n: initial generation count (set via actor_rollout_ref.rollout.n)
- top_k: number of high-KL steps to select per round
- iter_rounds: number of branching rounds
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import torch

from verl import DataProto
from verl.trainer.ppo.sampling.base import SamplingResult, SamplingStrategy
from verl.trainer.ppo.sampling.mcts_node import (
    MCTSNode,
    collect_all_nodes,
    gather_path,
    leaf_backpropagate_correct,
    leaf_normalize,
)
from verl.utils.ppo_batch import build_padded_prompt_response_batch
from verl.utils.process_reward import (
    build_process_reward_runtime,
    require_batch_sample_id,
    resolve_process_reward_config,
)


@contextmanager
def _timer(name: str, dest: dict):
    import time
    start = time.perf_counter()
    yield
    dest[name] = time.perf_counter() - start


class InformationGainStrategy(SamplingStrategy):
    """KL-divergence-guided tree expansion strategy (Simplified).

    Flow:
    1. gen_batch_output contains M=rollout.n complete solutions
    2. Parse into step chains
    3. Multi-round Top-K selection + branching
    4. Backpropagate: V = correct/total + PRM
    5. Output all leaves (with padding)

    Config key: ``trainer.ig_config``
    """

    def __init__(self, config, tokenizer):
        process_reward_cfg = resolve_process_reward_config(config)
        cfg = config.trainer.get("ig_config", {})

        self.max_depth = cfg.get("max_depth", 40)
        self.max_token_num = cfg.get("max_token_num", 512)
        self.prefix_text = cfg.get("prefix_text", "So far, the most likely answer is \\boxed{")

        # Simplified parameters
        self.top_k = cfg.get("top_k", 2)
        self.iter_rounds = cfg.get("iter_rounds", 3)

        self.process_reward_cfg = process_reward_cfg
        self.process_reward_runtime = build_process_reward_runtime(process_reward_cfg)
        self.process_reward_type = self.process_reward_runtime.reward_type
        self.fol_verifier = self.process_reward_runtime.fol_verifier
        self.fol_metadata_map = self.process_reward_runtime.fol_metadata_map
        self.step_prm_fn = self.process_reward_runtime.step_prm_fn
        self._sample_ids_by_tree: Dict[int, str] = {}

        self.tokenizer = tokenizer
        self.pad_token_id: int = getattr(tokenizer, "pad_token_id", 0) or 0
        self.eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)

        # GPU count for padding
        self._n_gpus = config.trainer.get("n_gpus_per_node", 1) * config.trainer.get("nnodes", 1)

        # KL computation cache
        self._option_cache: Dict[str, dict] = {}

    def _prepare_sample_ids(self, gen_batch: DataProto) -> None:
        self._sample_ids_by_tree = {}
        for tree_idx in range(gen_batch.batch["input_ids"].size(0)):
            self._sample_ids_by_tree[tree_idx] = require_batch_sample_id(
                gen_batch.non_tensor_batch,
                tree_idx,
                context="FOL process reward",
            )

    def _get_sample_id(self, tree_idx: int) -> str:
        if tree_idx not in self._sample_ids_by_tree:
            raise ValueError(
                f"FOL process reward requires a cached sample_id for tree_idx={tree_idx}."
            )
        return self._sample_ids_by_tree[tree_idx]

    def _score_step_reward(self, step_text: str, tree_idx: int) -> float:
        if self.process_reward_type == "format":
            return self.step_prm_fn(step_text)
        if self.process_reward_type == "fol":
            return self.step_prm_fn(step_text, sample_id=self._get_sample_id(tree_idx))
        raise ValueError(
            f"InformationGainStrategy requires trainer.process_reward.type to be 'format' or 'fol', "
            f"but got {self.process_reward_type!r}."
        )

    # ------------------------------------------------------------------
    # Option detection
    # ------------------------------------------------------------------

    def _detect_option_letters(self, prompt_str: str) -> List[str]:
        """Detect option letters from prompt text.

        Handles various formats:
        - "Option (A):" / "Option (B):" format
        - "A. " / "B. " line-start format
        - "A)" / "B)" format
        - "A:" / "B:" format
        - Returns all detected letters (supports 5+ options)
        """
        all_letters = []

        # Pattern 1: "Option (A):" / "Option (B):" format
        matches = re.findall(r'Option\s*\(([A-Z])\)', prompt_str)
        if matches:
            seen = set()
            letters = [m for m in matches if not (m in seen or seen.add(m))]
            all_letters.extend(letters)

        # Pattern 2: "A. " / "B. " at line start
        matches = re.findall(r'(?:^|\n)([A-Z])\.\s', prompt_str, re.MULTILINE)
        for m in matches:
            if m not in all_letters:
                all_letters.append(m)

        # Pattern 3: "A)" / "B)" format (with or without space)
        matches = re.findall(r'(?:^|\n|\s)([A-Z])\)\s', prompt_str)
        for m in matches:
            if m not in all_letters:
                all_letters.append(m)

        # Pattern 4: "A:" at line start or after newline
        matches = re.findall(r'(?:^|\n)([A-Z]):\s', prompt_str, re.MULTILINE)
        for m in matches:
            if m not in all_letters:
                all_letters.append(m)

        # Return detected letters (supports 5+ options)
        # e.g., ["A", "B", "C", "D", "E"] for 5-option questions
        if len(all_letters) >= 2:
            return all_letters

        # Fallback: try to find any uppercase letters that look like options
        matches = re.findall(r'(?:^|\n)([A-Z])\s', prompt_str, re.MULTILINE)
        letters = [m for m in matches if m in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        if len(letters) >= 2:
            return letters[:10]  # Limit to 10 options max

        # Last resort: default to A-D
        return ["A", "B", "C", "D"]

    def _get_option_token_ids(self, option_letters: List[str]) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for letter in option_letters:
            full_text = self.prefix_text + letter
            tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
            mapping[letter] = tokens[-1]
        return mapping

    def _get_ground_truth(self, tree_idx: int, gen_batch: DataProto) -> Optional[str]:
        if gen_batch.non_tensor_batch is None:
            return None
        if "answer" in gen_batch.non_tensor_batch:
            answers = gen_batch.non_tensor_batch["answer"]
            if tree_idx < len(answers):
                return str(answers[tree_idx]).strip().upper()
        reward_model = gen_batch.non_tensor_batch.get("reward_model", None)
        if reward_model is not None and "ground_truth" in reward_model:
            gt = reward_model["ground_truth"]
            if tree_idx < len(gt):
                return str(gt[tree_idx]).strip().upper()
        return None

    def _get_ground_truths(self, gen_batch: DataProto, batch_size: int) -> List[Optional[str]]:
        ground_truths: List[Optional[str]] = []
        if gen_batch.non_tensor_batch is not None and "answer" in gen_batch.non_tensor_batch:
            ground_truths = list(gen_batch.non_tensor_batch["answer"])
        ground_truths.extend([None] * max(0, batch_size - len(ground_truths)))
        return ground_truths

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

        with _timer("information_gain", timing_raw):
            if self.process_reward_type == "fol":
                self._prepare_sample_ids(gen_batch)

            batch_size = gen_batch.batch["input_ids"].size(0)

            # 1. Initialize
            roots = self._init_roots(gen_batch, device)
            self._init_option_caches(gen_batch, batch_size)

            # 2. Parse initial solutions
            self._generate_full_solutions(gen_batch, gen_batch_output, roots, batch_size)

            # 3. Multi-round Top-K selection + branching
            self._branch_by_kl(roots, generate_fn, compute_log_prob_fn, device)

            # 4. Backpropagate
            self._backpropagate_all(roots, gen_batch)

            # 5. Build output (with padding)
            ground_truths = self._get_ground_truths(gen_batch, batch_size)
            result = self._build_output(gen_batch, roots, device, ground_truths)

        return result

    # ------------------------------------------------------------------
    # Initialization
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

    def _init_option_caches(self, gen_batch: DataProto, batch_size: int) -> None:
        self._option_cache = {}
        input_ids = gen_batch.batch["input_ids"]
        attention_mask = gen_batch.batch["attention_mask"]

        for i in range(batch_size):
            prompt_ids = input_ids[i][attention_mask[i].bool()].tolist()
            prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

            option_letters = self._detect_option_letters(prompt_str)
            option_token_ids = self._get_option_token_ids(option_letters)
            ground_truth = self._get_ground_truth(i, gen_batch)

            gt_idx = -1
            if ground_truth is not None:
                gt_upper = ground_truth.strip().upper()
                if gt_upper in option_letters:
                    gt_idx = option_letters.index(gt_upper)

            self._option_cache[str(i)] = {
                "option_letters": option_letters,
                "option_token_ids": option_token_ids,
                "ground_truth": ground_truth,
                "gt_idx": gt_idx,
            }

    # ------------------------------------------------------------------
    # Step 1: Parse initial solutions
    # ------------------------------------------------------------------

    def _generate_full_solutions(
        self,
        gen_batch: DataProto,
        gen_batch_output: DataProto,
        roots: List[MCTSNode],
        batch_size: int,
    ) -> None:
        """Parse initial M=rollout.n complete solutions into step chains."""
        responses = gen_batch_output.batch["responses"]

        # Determine num_traces from gen_batch_output shape
        # responses shape: [batch_size * num_traces, seq_len]
        num_traces = responses.size(0) // batch_size

        for tree_idx, root in enumerate(roots):
            for trace_idx in range(num_traces):
                resp_idx = tree_idx * num_traces + trace_idx
                if resp_idx >= responses.size(0):
                    continue

                resp = responses[resp_idx]
                real_mask = resp != self.pad_token_id
                tokens = resp[real_mask].tolist()
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)

                self._attach_steps_to_root(root, text, tokens, tree_idx, gen_batch)

    def _attach_steps_to_root(
        self,
        root: MCTSNode,
        full_text: str,
        full_tokens: List[int],
        tree_idx: int,
        gen_batch: DataProto,
    ) -> None:
        """Split a complete solution into steps and attach to root."""
        step_blocks = self._split_by_step(full_text, full_tokens)
        current = root

        for block_text, block_tokens in step_blocks:
            r = 0.0
            if block_text.strip():
                r = self._score_step_reward(block_text, tree_idx)

            node = MCTSNode(
                state=current.state + block_tokens,
                step_tokens=block_tokens,
                step_text=block_text,
                accumulated_text=current.accumulated_text + block_text,
                parent=current,
                depth=current.depth + 1,
                terminal=False,
                tree_idx=tree_idx,
                node_idx=0,
                is_branch_point=False,
                kl_score=0.0,
                R=r,
                value=r,
            )
            current.children.append(node)
            current = node

    def _split_by_step(
        self,
        full_text: str,
        full_tokens: List[int],
    ) -> List[Tuple[str, List[int]]]:
        """Split text into steps by </step> tags, fallback to \n\n."""
        pattern = re.compile(r'(<step>.*?</step>)', re.DOTALL)
        matches = pattern.findall(full_text)
        if len(matches) >= 2:
            return [(m, self.tokenizer.encode(m, add_special_tokens=False)) for m in matches]

        parts = full_text.split('\n\n')
        if len(parts) >= 2:
            blocks = [(p, self.tokenizer.encode(p, add_special_tokens=False)) for p in parts if p.strip()]
            if len(blocks) >= 2:
                return blocks

        return [(full_text, full_tokens)]

    # ------------------------------------------------------------------
    # Step 2: Multi-round Top-K selection + branching
    # ------------------------------------------------------------------

    def _compute_step_kl_batch(
        self,
        all_nodes: List[MCTSNode],
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        device: torch.device,
    ) -> None:
        """Batch-compute KL divergence for all step nodes."""
        if not all_nodes:
            return

        prefix_ids = self.tokenizer.encode(self.prefix_text, add_special_tokens=False)
        prefix_len = len(prefix_ids)

        batch_size = len(all_nodes)
        max_state_len = max(len(n.state) for n in all_nodes)
        total_len = max_state_len + prefix_len

        input_ids = torch.full((batch_size, total_len), self.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, total_len), dtype=torch.long, device=device)
        responses = torch.full((batch_size, prefix_len), self.pad_token_id, dtype=torch.long, device=device)

        node_details: List[Tuple[int, List[str], Dict[str, int], int]] = []

        for i, node in enumerate(all_nodes):
            state_len = len(node.state)
            p_offset = max_state_len - state_len
            input_ids[i, p_offset:max_state_len] = torch.tensor(node.state, dtype=torch.long, device=device)
            attention_mask[i, p_offset:max_state_len] = 1
            input_ids[i, max_state_len:total_len] = torch.tensor(prefix_ids, dtype=torch.long, device=device)
            attention_mask[i, max_state_len:total_len] = 1
            responses[i, :prefix_len] = torch.tensor(prefix_ids, dtype=torch.long, device=device)

            cache = self._option_cache.get(str(node.tree_idx), {})
            node_details.append((
                node.tree_idx,
                cache.get("option_letters", ["A", "B", "C", "D"]),
                cache.get("option_token_ids", {}),
                cache.get("gt_idx", -1),
            ))

        # World size padding
        orig_batch_size = batch_size
        ws = max(self._n_gpus, 1)
        padded_size = ((orig_batch_size + ws - 1) // ws) * ws
        pad_slots = padded_size - orig_batch_size

        if pad_slots > 0:
            input_ids = torch.cat([input_ids, input_ids[:pad_slots].clone()], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask[:pad_slots].clone()], dim=0)
            responses = torch.cat([responses, responses[:pad_slots].clone()], dim=0)

        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        data = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": input_ids[:, :max_state_len],
                "responses": responses,
            },
            non_tensors={},
            meta_info={
                "micro_batch_size": padded_size,
                "max_token_len": total_len,
                "use_dynamic_bsz": False,
                "temperature": 1.0,
                "return_last_logits": True,
            },
        )

        output = compute_log_prob_fn(data)
        last_logits = output.batch.get("last_logits")

        if last_logits is None:
            for node in all_nodes:
                node.kl_score = 0.0
            return

        last_logits = last_logits[:orig_batch_size]

        eps = 1e-8
        for i, (tree_idx, option_letters, option_token_ids, gt_idx) in enumerate(node_details):
            if gt_idx < 0 or not option_token_ids:
                all_nodes[i].kl_score = 0.0
                continue

            num_opts = len(option_letters)
            option_logits = torch.full((num_opts,), -1e9, device=device)
            for li, letter in enumerate(option_letters):
                tid = option_token_ids.get(letter)
                if tid is not None and tid < last_logits.size(-1):
                    option_logits[li] = last_logits[i, tid]

            p = torch.softmax(option_logits, dim=-1)
            q = torch.zeros(num_opts, device=device)
            q[gt_idx] = 1.0

            kl = float(torch.sum(q * torch.log((q + eps) / (p + eps))).item())
            all_nodes[i].kl_score = kl

    def _branch_by_kl(
        self,
        roots: List[MCTSNode],
        generate_fn: Callable[[DataProto], DataProto],
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        device: torch.device,
    ) -> None:
        """Multi-round: select Top-K high-KL steps and branch."""
        for round_idx in range(self.iter_rounds):
            # Collect all expandable steps
            all_steps: List[MCTSNode] = []
            for root in roots:
                for node in collect_all_nodes(root):
                    if node.parent is not None and not node.children:
                        all_steps.append(node)

            if not all_steps:
                break

            # Compute KL for uncalculated nodes
            uncalculated = [n for n in all_steps if n.kl_score == 0.0]
            if uncalculated:
                self._compute_step_kl_batch(uncalculated, compute_log_prob_fn, device)

            # Select Top-K
            k = min(self.top_k, len(all_steps))
            selected = sorted(all_steps, key=lambda n: n.kl_score, reverse=True)[:k]

            # Filter out zero-KL or already branched
            selected = [s for s in selected if not s.is_branch_point and s.kl_score > 0.0]
            if not selected:
                break

            # Mark and branch
            for step in selected:
                step.is_branch_point = True
            self._continue_from_steps(selected, generate_fn, device)

    def _continue_from_steps(
        self,
        steps: List[MCTSNode],
        generate_fn: Callable[[DataProto], DataProto],
        device: torch.device,
    ) -> None:
        """Continue generation from selected step positions (max_children=1)."""
        if not steps:
            return

        seqs = [torch.tensor(s.state, dtype=torch.long, device=device) for s in steps]

        # Padding for DP compatibility
        orig_size = len(seqs)
        ws = max(self._n_gpus, 1)
        padded_size = ((orig_size + ws - 1) // ws) * ws
        while len(seqs) < padded_size:
            seqs.append(seqs[0].clone())

        input_ids, attention_mask, position_ids = _pad_sequences(seqs, self.pad_token_id, device)

        data = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": input_ids,
            },
            non_tensors={},
            meta_info={"max_new_tokens": self.max_token_num},
        )

        output = generate_fn(data)
        responses = output.batch["responses"]

        # Parse and attach new steps
        for i, step in enumerate(steps):
            if i >= responses.size(0):
                continue

            resp = responses[i]
            real_mask = resp != self.pad_token_id
            tokens = resp[real_mask].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)

            step_blocks = self._split_by_step(text, tokens)
            current = step

            for block_text, block_tokens in step_blocks:
                r = 0.0
                if block_text.strip():
                    r = self._score_step_reward(block_text, step.tree_idx)

                child = MCTSNode(
                    state=current.state + block_tokens,
                    step_tokens=block_tokens,
                    step_text=block_text,
                    accumulated_text=current.accumulated_text + block_text,
                    parent=current,
                    depth=current.depth + 1,
                    terminal=False,
                    tree_idx=step.tree_idx,
                    node_idx=0,
                    is_branch_point=False,
                    kl_score=0.0,
                    R=r,
                    value=r,
                )
                current.children.append(child)
                current = child

    # ------------------------------------------------------------------
    # Step 3: Backpropagate
    # ------------------------------------------------------------------

    def _backpropagate_all(
        self,
        roots: List[MCTSNode],
        gen_batch: DataProto,
    ) -> None:
        """Backpropagate correctness counts and PRM rewards."""
        from verl.trainer.ppo.sampling.mcts_node import normalize_all_steps

        for root in roots:
            # Collect leaves
            leaves = [n for n in collect_all_nodes(root) if not n.children]
            if not leaves:
                all_nodes = collect_all_nodes(root)
                non_root = [n for n in all_nodes if n.parent is not None]
                leaves = [max(non_root, key=lambda n: n.depth)] if non_root else [root]

            # Mark leaf correctness
            gt = self._get_ground_truth(root.tree_idx, gen_batch)
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
                    leaf.is_correct = None

                if leaf.is_correct:
                    leaf.main_chain = True

            # Backpropagate correctness counts
            for leaf in leaves:
                leaf_backpropagate_correct(leaf)

            # Compute V_baseline
            self._compute_v_baseline(root)

            leaf_normalize(leaves)
            normalize_all_steps(root, style="step")

    def _compute_v_baseline(self, root: MCTSNode) -> None:
        """Compute V_baseline = correct_terminal / terminal_in_subtree for each node."""
        nodes = list(collect_all_nodes(root))
        nodes.sort(key=lambda n: n.depth, reverse=True)

        for node in nodes:
            if node.terminal_in_subtree > 0:
                node.value = node.correct_terminal_in_subtree / node.terminal_in_subtree
            else:
                node.value = 0.0

    # ------------------------------------------------------------------
    # Step 4: Build output (with padding)
    # ------------------------------------------------------------------

    def _build_output(
        self,
        gen_batch: DataProto,
        roots: List[MCTSNode],
        device: torch.device,
        ground_truths: List[Optional[str]],
    ) -> SamplingResult:
        """Build SamplingResult from all leaves, with padding for GPU divisibility."""
        batch_size = len(roots)

        all_paths: List[List[MCTSNode]] = []
        all_gt: List[Optional[str]] = []

        for i, root in enumerate(roots):
            gt = ground_truths[i] if i < len(ground_truths) else None
            leaves = [n for n in collect_all_nodes(root) if not n.children]

            if not leaves:
                all_nodes = collect_all_nodes(root)
                non_root = [n for n in all_nodes if n.parent is not None]
                leaves = [max(non_root, key=lambda n: n.depth)] if non_root else [root]

            for leaf in leaves:
                all_paths.append(gather_path(leaf))
                all_gt.append(gt)

        # Auto padding for GPU divisibility
        total = len(all_paths)
        remainder = total % self._n_gpus
        if remainder != 0:
            padding_needed = self._n_gpus - remainder
            for _ in range(padding_needed):
                all_paths.append(all_paths[-1])
                all_gt.append(all_gt[-1])
            print(f"[IG] Padded {padding_needed} samples to make {total + padding_needed} divisible by {self._n_gpus}")

        return _build_sampling_result(all_paths, all_gt, self.pad_token_id, device, batch_size)


# ------------------------------------------------------------------
# Shared helpers
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


def _build_sampling_result(
    all_paths: List[List[MCTSNode]],
    all_gt: List[Optional[str]],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
) -> SamplingResult:
    """Build padded tensors and SamplingResult from selected paths."""
    prompt_ids_list: List[torch.Tensor] = []
    resp_ids_list: List[torch.Tensor] = []
    step_spans_list: List[List[Tuple[int, int]]] = []
    step_rewards_list: List[List[float]] = []
    step_correctness_scores_list: List[List[float]] = []
    response_lens: List[int] = []
    verifiable_rewards_list: List[float] = []

    for path, gt in zip(all_paths, all_gt):
        root_node = path[0].parent
        if root_node is None or root_node.parent is not None:
            raise ValueError("InformationGain paths must start at a direct child of the root node.")
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

        prompt_ids_list.append(prompt_tokens)
        resp_ids_list.append(resp_tensor)
        step_spans_list.append(spans)
        step_rewards_list.append(rewards)
        step_correctness_scores_list.append(correctness_scores)
        response_lens.append(len(response_tokens))
        verifiable_rewards_list.append(1.0 if path[-1].is_correct else 0.0)

    padded_batch = build_padded_prompt_response_batch(prompt_ids_list, resp_ids_list, pad_token_id)
    input_ids = padded_batch.input_ids
    attention_mask = padded_batch.attention_mask
    position_ids = padded_batch.position_ids
    prompts_padded = padded_batch.prompts
    responses_padded = padded_batch.responses

    reward_fn_scores = _build_token_level_scores(
        responses=responses_padded,
        response_lens=response_lens,
        all_step_spans=step_spans_list,
        all_step_rewards=step_rewards_list,
    )

    n_paths = len(all_paths)
    max_steps = max((len(s) for s in step_spans_list), default=1)
    score_ids = torch.full((n_paths, max_steps), -1, device=device, dtype=torch.long)
    reward_mask = torch.zeros(n_paths, max_steps, device=device, dtype=torch.float32)
    for i, (spans, rlen) in enumerate(zip(step_spans_list, response_lens)):
        for j, (_, end) in enumerate(spans[:max_steps]):
            end_pos = max(0, min(end - 1, rlen - 1)) if rlen > 0 else 0
            score_ids[i, j] = end_pos
            reward_mask[i, j] = 1.0

    max_resp_len = responses_padded.size(1)
    verifiable_rewards = torch.zeros(n_paths, max_resp_len, dtype=torch.float32, device=device)
    for i, rlen in enumerate(response_lens):
        if rlen > 0:
            verifiable_rewards[i, rlen - 1] = verifiable_rewards_list[i]

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
