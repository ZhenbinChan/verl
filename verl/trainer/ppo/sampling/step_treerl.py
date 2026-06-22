"""StepTreeRL sampling strategy.

Performs multi-round tree expansion where nodes are selected for branching
based on per-step average entropy (instead of UCT). Each generation may
produce multiple ``<step>...</step>`` blocks, which are split into individual
MCTSNodes. Branching candidates are drawn from all non-root nodes in each
tree, not only the current leaves.

TreeRL-style parameters:
    m / rollout.n       - initial complete rollouts per prompt
    n / top_k           - high-entropy branch points per initial rollout tree per round
    l / iter_rounds     - branching rounds
    t / branch_repeats  - continuations sampled per selected branch point
    selected_num_traces - terminal traces selected per prompt for training
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from verl import DataProto
from verl.trainer.ppo.sampling.base import SamplingResult, SamplingStrategy
from verl.trainer.ppo.sampling.mcts_node import (
    MCTSNode,
    collect_all_nodes,
    gather_path,
)
from verl.trainer.ppo.sampling.mcts_prm import boxed_answer_format_correct, classify_trajectory_format, format_step_reward
from verl.utils.process_reward import (
    StepRewardRequest,
    build_process_reward_runtime,
    get_batch_question_text,
    require_batch_sample_id,
    resolve_process_reward_config,
)


@dataclass
class StepTreeRLMetrics:
    format_steps: int = 0
    total_steps: int = 0
    process_reward_sum: float = 0.0
    process_reward_count: int = 0
    problem_count: int = 0
    leaf_correct: int = 0
    leaf_total: int = 0
    selected_traces: int = 0
    candidate_leaves: int = 0
    terminal_padding: int = 0
    trace_total: int = 0
    full_format_correct_count: int = 0
    answer_format_only_count: int = 0
    step_format_only_count: int = 0
    step_num: float = 0.0
    # LLM RM trajectory quality scores
    llm_rm_score_sum: float = 0.0
    llm_rm_score_count: int = 0


@dataclass
class GenerationSegment:
    node_type: str
    text: str
    tokens: List[int]


@contextmanager
def _timer(name: str, dest: dict):
    import time

    start = time.perf_counter()
    yield
    dest[name] = dest.get(name, 0.0) + time.perf_counter() - start


class StepTreeRLStrategy(SamplingStrategy):
    """Step-level entropy-guided tree expansion strategy (Simplified).

    Multi-round tree expansion where:
    1. Initial complete solutions are parsed into steps
    2. Multi-round: select per-tree Top-K high-entropy nodes and branch
    3. Select terminal traces and output step-level training rewards with GPU padding

    Config key: ``trainer.step_treerl_config``
    """

    def __init__(self, config, tokenizer):
        process_reward_cfg = resolve_process_reward_config(config)
        cfg = config.trainer.get("step_treerl_config", {})
        self.adv_estimator = str(config.algorithm.get("adv_estimator", "")) if "algorithm" in config else ""
        self.use_origin_advantage = self.adv_estimator in ("step_treerl_origin", "step_rl")
        self.use_branch_point_search = self.adv_estimator == "step_treerl_origin"

        # Shared generation bounds
        self.max_depth = cfg.get("max_depth", 40)
        self.max_token_num = cfg.get("max_token_num", 4096)
        self.branch_max_new_tokens = cfg.get("branch_max_new_tokens", self.max_token_num)

        # TreeRL-style params. Legacy aliases are kept for existing scripts.
        self.rollout_n = int(config.actor_rollout_ref.rollout.get("n", 1))
        self.m = int(cfg.get("m", self.rollout_n))
        self.top_k = int(cfg.get("n", cfg.get("top_k", 2)))
        self.iter_rounds = int(cfg.get("l", cfg.get("iter_rounds", 3)))
        self.branch_repeats = int(cfg.get("t", cfg.get("branch_repeats", 1)))
        self.path_selection = str(cfg.get("path_selection", "selected_terminals")).lower()
        self.selected_num_traces = cfg.get("selected_num_traces", cfg.get("num_traces", self.rollout_n))
        self.overall_norm_style = str(cfg.get("overall_norm_style", "none")).lower()
        self.use_weighted_value = bool(cfg.get("use_weighted_value", False))
        self.weighted_value_style = str(cfg.get("weighted_value_style", "original")).lower()
        length_penalty_cfg = cfg.get("length_penalty", {}) or {}
        self.length_penalty_enabled = bool(length_penalty_cfg.get("enabled", True))
        self.length_penalty_p_max = float(length_penalty_cfg.get("p_max", 0.1))
        self.length_penalty_k = float(length_penalty_cfg.get("k", 15.0))
        self.length_penalty_t0 = float(length_penalty_cfg.get("t0", 0.7))
        self._node_counters: Dict[int, int] = {}
        self._metrics = StepTreeRLMetrics()
        self._timing: Dict[str, float] = {}

        self.process_reward_cfg = process_reward_cfg
        self.process_reward_runtime = build_process_reward_runtime(process_reward_cfg)
        self.process_reward_type = self.process_reward_runtime.reward_type
        self.fol_verifier = self.process_reward_runtime.fol_verifier
        self.fol_metadata_map = self.process_reward_runtime.fol_metadata_map
        self.step_prm_fn = self.process_reward_runtime.step_prm_fn
        self.self_eval_prompt_template = self.process_reward_runtime.self_eval_prompt_template
        self.self_eval_max_new_tokens = self.process_reward_runtime.self_eval_max_new_tokens
        self.self_eval_temperature = self.process_reward_runtime.self_eval_temperature
        self.self_eval_top_p = self.process_reward_runtime.self_eval_top_p
        self.self_eval_max_batch_size = self.process_reward_runtime.self_eval_max_batch_size
        self.self_eval_fail_on_parse_error = self.process_reward_runtime.self_eval_fail_on_parse_error
        self._sample_ids_by_tree: Dict[int, str] = {}
        self._question_texts_by_tree: Dict[int, str] = {}

        # LLM RM for trajectory quality evaluation
        self.trajectory_rm_enabled = bool(cfg.get("trajectory_rm_enabled", True))
        self.trajectory_rm_url = str(cfg.get("trajectory_rm_url", "")).strip()
        self.trajectory_rm_model = str(cfg.get("trajectory_rm_model", "eval-model"))
        self.trajectory_rm_max_tokens = int(cfg.get("trajectory_rm_max_tokens", 32))
        self.trajectory_rm_temperature = float(cfg.get("trajectory_rm_temperature", 0.0))
        self.trajectory_rm_api_key = str(cfg.get("trajectory_rm_api_key", "")).strip()
        # coeff for blending LLM quality score into leaf accumulated_value
        self.trajectory_rm_coeff = float(cfg.get("trajectory_rm_coeff", 0.5))

        self.tokenizer = tokenizer
        self.pad_token_id: int = getattr(tokenizer, "pad_token_id", 0) or 0
        self.eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)

        # Used for padding entropy-computation batches to a multiple of world size
        self._n_gpus = config.trainer.get("n_gpus_per_node", 1) * config.trainer.get("nnodes", 1)

        # Max model length from vLLM config (used to skip nodes that exceed limit)
        self.max_model_len = config.actor_rollout_ref.rollout.get("max_model_len", 4096)

        # Step boundary delimiter
        self.step_end_marker = "</step>"

    def _next_node_idx(self, tree_idx: int) -> int:
        idx = self._node_counters.get(tree_idx, 0) + 1
        self._node_counters[tree_idx] = idx
        return idx

    def _prepare_process_reward_context(self, gen_batch: DataProto, *, require_sample_id: bool) -> None:
        self._sample_ids_by_tree = {}
        self._question_texts_by_tree = {}
        for tree_idx in range(gen_batch.batch["input_ids"].size(0)):
            if require_sample_id:
                self._sample_ids_by_tree[tree_idx] = require_batch_sample_id(
                    gen_batch.non_tensor_batch,
                    tree_idx,
                    context="FOL process reward",
                )
            question_text = get_batch_question_text(gen_batch.non_tensor_batch, tree_idx)
            if question_text is None:
                root_ids = gen_batch.batch["input_ids"][tree_idx]
                attention = gen_batch.batch["attention_mask"][tree_idx]
                real_ids = root_ids[attention.bool()].tolist()
                question_text = self.tokenizer.decode(real_ids, skip_special_tokens=True)
            self._question_texts_by_tree[tree_idx] = question_text

    def _prepare_fol_context(self, gen_batch: DataProto) -> None:
        self._prepare_process_reward_context(gen_batch, require_sample_id=True)

    def _ensure_question_texts(self, gen_batch: DataProto) -> None:
        """Populate _question_texts_by_tree if not already set.

        Tries non_tensor_batch first, falls back to decoding root state
        (which is the tokenized prompt prefixed with system/instruction templates).
        """
        if self._question_texts_by_tree:
            return

        non_tensor = gen_batch.non_tensor_batch
        if non_tensor is not None:
            for tree_idx in range(gen_batch.batch["input_ids"].size(0)):
                question_text = get_batch_question_text(non_tensor, tree_idx)
                if question_text is not None:
                    self._question_texts_by_tree[tree_idx] = question_text

        # Fallback: decode the raw prompt tokens (may include system/training template)
        if not self._question_texts_by_tree:
            for tree_idx in range(gen_batch.batch["input_ids"].size(0)):
                root_ids = gen_batch.batch["input_ids"][tree_idx]
                attention = gen_batch.batch["attention_mask"][tree_idx]
                real_ids = root_ids[attention.bool()].tolist()
                self._question_texts_by_tree[tree_idx] = self.tokenizer.decode(
                    real_ids, skip_special_tokens=True
                )

    def _get_sample_id(self, tree_idx: int) -> str:
        if tree_idx not in self._sample_ids_by_tree:
            raise ValueError(
                f"FOL process reward requires a cached sample_id for tree_idx={tree_idx}."
            )
        return self._sample_ids_by_tree[tree_idx]

    def _score_step_reward(self, step_text: str, tree_idx: int) -> float:
        if self.process_reward_type == "self_eval":
            raise ValueError("self_eval process reward requires actor generation and cannot be scored synchronously.")
        scores = self._score_process_rewards(
            [
                StepRewardRequest(
                    step_text=step_text,
                    tree_idx=tree_idx,
                    sample_id=self._sample_ids_by_tree.get(tree_idx),
                    question_text=self._question_texts_by_tree.get(tree_idx),
                )
            ]
        )
        return scores[0]

    def _score_process_rewards(
        self,
        requests: List[StepRewardRequest],
        generate_fn: Optional[Callable[[DataProto], DataProto]] = None,
        device: Optional[torch.device] = None,
    ) -> List[float]:
        if not requests:
            return []
        if self.process_reward_type == "self_eval":
            if generate_fn is None or device is None:
                raise ValueError("self_eval process reward requires generate_fn and device.")
            return self._score_self_eval_rewards(requests, generate_fn, device)
        if hasattr(self.process_reward_runtime, "score_steps"):
            return self.process_reward_runtime.score_steps(requests)
        if self.process_reward_type == "format":
            return [float(self.step_prm_fn(req.step_text)) for req in requests]
        if self.process_reward_type == "fol":
            scores = []
            for req in requests:
                sample_id = req.sample_id or self._get_sample_id(req.tree_idx)
                scores.append(float(self.step_prm_fn(req.step_text, sample_id=sample_id)))
            return scores
        raise ValueError(
            f"StepTreeRLStrategy requires trainer.process_reward.type to be 'format', 'fol', or 'self_eval', "
            f"but got {self.process_reward_type!r}."
        )

    def _assign_process_rewards(
        self,
        nodes: List[MCTSNode],
        generate_fn: Optional[Callable[[DataProto], DataProto]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        step_nodes = [node for node in nodes if getattr(node, "node_type", "step") == "step"]
        if not step_nodes:
            return
        requests = [
            StepRewardRequest(
                step_text=node.step_text,
                accumulated_text=node.accumulated_text,
                tree_idx=node.tree_idx,
                node_idx=node.node_idx,
                sample_id=self._sample_ids_by_tree.get(node.tree_idx),
                question_text=self._question_texts_by_tree.get(node.tree_idx),
            )
            for node in step_nodes
        ]
        scores = self._score_process_rewards(requests, generate_fn=generate_fn, device=device)
        for node, score in zip(step_nodes, scores):
            node.process_reward = float(score)
            node.R = float(score)
            node.value = float(score)

    def _format_self_eval_prompt(self, req: StepRewardRequest) -> str:
        question_text = req.question_text or ""
        reasoning_steps = req.accumulated_text or req.step_text
        return self.self_eval_prompt_template.format(
            question_text=question_text,
            reasoning_steps=reasoning_steps,
        )

    @staticmethod
    def _parse_self_eval_score(text: str) -> Optional[float]:
        matches = list(re.finditer(r"\\boxed\{\{?\s*([01])\s*\}?\}", text or "", re.DOTALL))
        if not matches:
            return None
        return float(matches[-1].group(1))

    def _score_self_eval_rewards(
        self,
        requests: List[StepRewardRequest],
        generate_fn: Callable[[DataProto], DataProto],
        device: torch.device,
    ) -> List[float]:
        if not requests:
            return []

        scores: List[Optional[float]] = [None] * len(requests)
        key_to_prompt: Dict[Tuple[str, str], str] = {}
        request_keys: List[Tuple[str, str]] = []
        for req in requests:
            question_text = req.question_text or ""
            reasoning_steps = req.accumulated_text or req.step_text
            key = (question_text, reasoning_steps)
            request_keys.append(key)
            if key in self.process_reward_runtime._score_cache:
                continue
            if key not in key_to_prompt:
                key_to_prompt[key] = self._format_self_eval_prompt(req)

        pending_items = list(key_to_prompt.items())
        if pending_items:
            max_batch_size = self.self_eval_max_batch_size or len(pending_items)
            for start in range(0, len(pending_items), max_batch_size):
                chunk = pending_items[start:start + max_batch_size]
                chunk_scores = self._generate_self_eval_scores(
                    prompts=[prompt for _, prompt in chunk],
                    generate_fn=generate_fn,
                    device=device,
                )
                for (key, _), score in zip(chunk, chunk_scores):
                    self.process_reward_runtime._score_cache[key] = float(score)

        for i, key in enumerate(request_keys):
            scores[i] = float(self.process_reward_runtime._score_cache.get(key, 0.0))
        return [float(score or 0.0) for score in scores]

    def _generate_self_eval_scores(
        self,
        prompts: List[str],
        generate_fn: Callable[[DataProto], DataProto],
        device: torch.device,
    ) -> List[float]:
        if not prompts:
            return []

        encoded_prompts = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
        context_budget = max(1, int(self.max_model_len) - int(self.self_eval_max_new_tokens))
        encoded_prompts = [ids[-context_budget:] if len(ids) > context_budget else ids for ids in encoded_prompts]
        prompt_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in encoded_prompts]
        input_ids, attention_mask, position_ids = _pad_sequences(prompt_tensors, self.pad_token_id, device)

        batch_size = len(prompts)
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
            meta_info={
                "rollout_sampling_kwargs": {
                    "max_new_tokens": int(self.self_eval_max_new_tokens),
                    "max_tokens": int(self.self_eval_max_new_tokens),
                    "n": 1,
                    "temperature": float(self.self_eval_temperature),
                    "top_p": float(self.self_eval_top_p),
                },
            },
        )

        output = generate_fn(data)
        responses = output.batch["responses"]
        if responses.size(0) < padded_size:
            raise RuntimeError(
                f"self_eval generation returned {responses.size(0)} responses for {padded_size} padded prompts."
            )
        responses = responses[:batch_size]

        parsed_scores: List[float] = []
        for response in responses:
            real_mask = response != self.pad_token_id
            response_text = self.tokenizer.decode(response[real_mask].tolist(), skip_special_tokens=True)
            parsed = self._parse_self_eval_score(response_text)
            if parsed is None:
                if self.self_eval_fail_on_parse_error:
                    raise ValueError(f"Failed to parse self_eval boxed score from response: {response_text!r}")
                parsed = 0.0
            parsed_scores.append(float(parsed))
        return parsed_scores

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
            self._node_counters = {}
            self._metrics = StepTreeRLMetrics()
            self._timing = {}
            if not self.use_origin_advantage and self.process_reward_type in {"fol", "self_eval"}:
                self._prepare_process_reward_context(
                    gen_batch,
                    require_sample_id=self.process_reward_type == "fol",
                )

            batch_size = gen_batch.batch["input_ids"].size(0)

            # 1. Parse initial solutions into step chains
            with _timer("init_parse", self._timing):
                roots = self._init_roots(gen_batch, device)
                initial_nodes = self._generate_full_solutions(gen_batch, gen_batch_output, roots, batch_size)
            if not self.use_origin_advantage:
                with _timer("process_reward", self._timing):
                    self._assign_process_rewards(initial_nodes, generate_fn=generate_fn, device=device)
            candidate_pool = self._build_candidate_pool(roots, initial_nodes)
            self._score_new_candidates(initial_nodes, compute_log_prob_fn, device, timing_name="initial_entropy_logprob")

            # 2. Multi-round per-tree Top-K selection + branching
            self._branch_by_entropy(
                roots=roots,
                candidate_pool=candidate_pool,
                generate_fn=generate_fn,
                compute_log_prob_fn=compute_log_prob_fn,
                device=device,
            )

            # 3. Backpropagate
            with _timer("backprop", self._timing):
                self._backpropagate_all(roots, gen_batch)

            # 4. Build output with padding
            ground_truths = self._get_ground_truths(gen_batch, batch_size)
            with _timer("build_output", self._timing):
                result = self._build_output(gen_batch, roots, device, ground_truths)
            result.gen_batch_output.meta_info["step_treerl_timing"] = dict(self._timing)

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

                segments = self._split_generation_segments(full_text, step_tokens_content)

                current_parent = root
                for seg_idx, segment in enumerate(segments):
                    is_last_segment = seg_idx == len(segments) - 1
                    is_answer = segment.node_type == "answer"
                    is_terminal = is_answer or (is_last_segment and hit_eos) or (current_parent.depth + 1 > self.max_depth)

                    accumulated_text = current_parent.accumulated_text + segment.text
                    new_state = current_parent.state + segment.tokens

                    child = MCTSNode(
                        state=new_state,
                        step_tokens=segment.tokens,
                        step_text=segment.text,
                        accumulated_text=accumulated_text,
                        trajectory_text=accumulated_text if is_answer else full_text,
                        parent=current_parent,
                        depth=current_parent.depth + 1,
                        terminal=is_terminal,
                        visits=0,
                        tree_idx=tree_idx,
                        node_idx=self._next_node_idx(tree_idx),
                        node_type=segment.node_type,
                        process_reward=float(boxed_answer_format_correct(segment.text)) if is_answer else 0.0,
                    )
                    current_parent.children.append(child)
                    current_parent = child
                    created_nodes.append(child)
                    if is_answer:
                        break

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

    def _split_generation_segments(
        self,
        full_text: str,
        full_tokens: List[int],
    ) -> List[GenerationSegment]:
        """Split generated text into step nodes plus an optional terminal answer node."""
        import re

        step_pattern = re.compile(r"(<step>.*?</step>)", re.DOTALL)
        matches = list(step_pattern.finditer(full_text or ""))
        if not matches:
            if (full_text or "").strip() and boxed_answer_format_correct(full_text):
                return [GenerationSegment("answer", full_text, full_tokens)]
            return [
                GenerationSegment("step", text, tokens)
                for text, tokens in self._split_by_step_end(full_text, full_tokens)
            ]

        segments = [
            GenerationSegment("step", text, tokens)
            for text, tokens in self._split_by_step_end(full_text, full_tokens)
        ]
        suffix = full_text[matches[-1].end():]
        if suffix.strip():
            suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
            segments.append(GenerationSegment("answer", suffix, suffix_tokens))
        return segments

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
        """Multi-round: select Top-N nodes per initial rollout tree and branch."""
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

            with _timer("branch_generation", self._timing):
                new_nodes = self._continue_from_steps(selected, generate_fn, device)
            if new_nodes:
                if not self.use_origin_advantage:
                    with _timer("process_reward", self._timing):
                        self._assign_process_rewards(new_nodes, generate_fn=generate_fn, device=device)
                self._add_candidates(candidate_pool, new_nodes)
                self._score_new_candidates(new_nodes, compute_log_prob_fn, device, timing_name="branch_entropy_logprob")

    def _build_candidate_pool(self, roots: List[MCTSNode], initial_nodes: List[MCTSNode]) -> Dict[int, List[MCTSNode]]:
        candidate_pool: Dict[int, List[MCTSNode]] = {root.tree_idx: [] for root in roots}
        self._add_candidates(candidate_pool, initial_nodes)
        return candidate_pool

    def _add_candidates(self, candidate_pool: Dict[int, List[MCTSNode]], nodes: List[MCTSNode]) -> None:
        for node in nodes:
            if node.parent is None or getattr(node, "node_type", "step") != "step":
                continue
            bucket = candidate_pool.setdefault(node.tree_idx, [])
            if node not in bucket:
                bucket.append(node)

    def _score_new_candidates(
        self,
        nodes: List[MCTSNode],
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        device: torch.device,
        timing_name: Optional[str] = None,
    ) -> None:
        pending = [
            node
            for node in nodes
            if node.parent is not None
            and getattr(node, "node_type", "step") == "step"
            and node.cached_entropy is None
        ]
        if not pending:
            return
        if timing_name is None:
            entropies = self._compute_step_entropies(pending, compute_log_prob_fn, device)
        else:
            with _timer(timing_name, self._timing):
                entropies = self._compute_step_entropies(pending, compute_log_prob_fn, device)
        for node, entropy in zip(pending, entropies):
            node.cached_entropy = entropy

    def _collect_branch_candidates(
        self,
        roots: List[MCTSNode],
        candidate_pool: Dict[int, List[MCTSNode]],
    ) -> Dict[Tuple[int, int], List[MCTSNode]]:
        """Collect cached nodes grouped by initial rollout tree."""
        candidates: Dict[Tuple[int, int], List[MCTSNode]] = {}
        for root in roots:
            for node in candidate_pool.get(root.tree_idx, []):
                if node.parent is None or node.cached_entropy is None or getattr(node, "node_type", "step") != "step":
                    continue
                group_key = self._initial_tree_key(node)
                if group_key is not None:
                    candidates.setdefault(group_key, []).append(node)
        return candidates

    def _root_of(self, node: MCTSNode) -> MCTSNode:
        cur = node
        while cur.parent is not None:
            cur = cur.parent
        return cur

    def _initial_tree_key(self, node: MCTSNode) -> Optional[Tuple[int, int]]:
        if node.parent is None:
            return None
        cur = node
        while cur.parent is not None and cur.parent.parent is not None:
            cur = cur.parent
        return (node.tree_idx, cur.node_idx)

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
            root = self._root_of(step)
            response_used = max(0, len(step.state) - len(root.state))
            remaining_response_budget = self.max_token_num - response_used
            remaining_context_budget = self.max_model_len - len(step.state)
            branch_budget = min(self.branch_max_new_tokens, remaining_response_budget, remaining_context_budget)
            if branch_budget <= 0:
                skipped_steps += 1
                continue
            for _ in range(max(self.branch_repeats, 1)):
                branch_plans.append((step, branch_budget))

        if not branch_plans:
            if skipped_steps > 0:
                print(f"[StepTreeRL] Skipped {skipped_steps} selected nodes due to exhausted context budget")
            return []

        total_new_nodes = 0
        total_duplicates = 0
        created_nodes: List[MCTSNode] = []

        plans_by_budget: Dict[int, List[MCTSNode]] = defaultdict(list)
        for step, branch_budget in branch_plans:
            plans_by_budget[int(branch_budget)].append(step)

        for round_budget, active_steps in sorted(plans_by_budget.items()):
            group_nodes, group_duplicates = self._generate_continuations_for_budget(
                active_steps=active_steps,
                max_new_tokens=round_budget,
                generate_fn=generate_fn,
                device=device,
            )
            total_new_nodes += len(group_nodes)
            total_duplicates += group_duplicates
            created_nodes.extend(group_nodes)

        if skipped_steps > 0:
            print(f"[StepTreeRL] Skipped {skipped_steps} selected nodes due to exhausted context budget")
        if total_duplicates > 0:
            print(f"[StepTreeRL] Deduplicated {total_duplicates} duplicate steps, added {total_new_nodes} new nodes")
        return created_nodes

    def _generate_continuations_for_budget(
        self,
        active_steps: List[MCTSNode],
        max_new_tokens: int,
        generate_fn: Callable[[DataProto], DataProto],
        device: torch.device,
    ) -> Tuple[List[MCTSNode], int]:
        if not active_steps:
            return [], 0

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
            meta_info={
                "rollout_sampling_kwargs": {
                    "max_new_tokens": max_new_tokens,
                    "max_tokens": max_new_tokens,
                    "n": 1,
                },
            },
        )

        output = generate_fn(data)
        responses = output.batch["responses"]
        if responses.size(0) < padded_size:
            raise RuntimeError(
                f"StepTreeRL branch generation returned {responses.size(0)} responses for {padded_size} padded prompts."
            )
        if responses.size(0) > padded_size:
            print(
                f"[StepTreeRL] Branch generation returned {responses.size(0)} responses for {padded_size} padded prompts; "
                "truncating to the first sample per prompt. Check rollout_sampling_kwargs['n']."
            )
        responses = responses[:batch_size]  # slice back

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
            full_trajectory_text = step.accumulated_text + full_text
            segments = self._split_generation_segments(full_text, step_tokens_content)

            current_parent = step
            for seg_idx, segment in enumerate(segments):
                is_last_segment = seg_idx == len(segments) - 1
                is_answer = segment.node_type == "answer"
                is_terminal = is_answer or (is_last_segment and hit_eos) or (current_parent.depth + 1 > self.max_depth)

                accumulated_text = current_parent.accumulated_text + segment.text
                new_state = current_parent.state + segment.tokens

                # Deduplication is limited to sibling branches from the same parent.
                if self._is_duplicate_step(segment.tokens, current_parent):
                    total_duplicates += 1
                    continue

                child = MCTSNode(
                    state=new_state,
                    step_tokens=segment.tokens,
                    step_text=segment.text,
                    accumulated_text=accumulated_text,
                    trajectory_text=accumulated_text if is_answer else full_trajectory_text,
                    parent=current_parent,
                    depth=current_parent.depth + 1,
                    terminal=is_terminal,
                    visits=0,
                    tree_idx=step.tree_idx,
                    node_idx=self._next_node_idx(step.tree_idx),
                    node_type=segment.node_type,
                    process_reward=float(boxed_answer_format_correct(segment.text)) if is_answer else 0.0,
                )
                current_parent.children.append(child)
                current_parent = child
                created_nodes.append(child)
                if is_answer:
                    break

        return created_nodes, total_duplicates

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
        """Compute TreeRL-style RLOO leaf values and backpropagate them."""
        for root in roots:
            tree_idx = root.tree_idx
            gt = self._get_ground_truths(gen_batch, len(roots))[tree_idx]
            self._reset_tree_statistics(root)

            # Collect all leaves
            leaves = [n for n in collect_all_nodes(root) if not n.children]

            if not leaves:
                continue

            # Score each leaf using both full-trajectory format and answer correctness.
            for leaf in leaves:
                terminal_text = leaf.accumulated_text
                format_valid = bool(
                    terminal_text
                    and classify_trajectory_format(terminal_text, valid_choices="ABCDEF")["format_full"]
                )
                answer_correct = False
                if gt is not None and terminal_text:
                    try:
                        from verl.utils.reward_score.logi import compute_score
                        score, _ = compute_score(terminal_text, gt)
                        answer_correct = float(score) == 1.0
                    except Exception:
                        answer_correct = False

                leaf.is_correct = format_valid and answer_correct
                if not format_valid:
                    leaf.leaf_outcome = 0.0
                elif answer_correct:
                    leaf.leaf_outcome = 1.0
                else:
                    leaf.leaf_outcome = 0.1
                leaf.R = leaf.leaf_outcome  # base value before fusion / RLOO
                if leaf.is_correct:
                    leaf.main_chain = True

            # Optionally refine leaf outcome with LLM trajectory quality evaluation.
            # The external RM changes the value used by RLOO, but never overrides
            # format-aware correctness or main-chain membership.
            if self.trajectory_rm_enabled and self.trajectory_rm_url:
                self._evaluate_leaves_quality(roots, leaves, gen_batch)

            self._apply_rloo_to_leaves(leaves)
            for leaf in leaves:
                self._leaf_backpropagate_value(leaf)
            self._normalize_all_steps(root)

    def _reset_tree_statistics(self, root: MCTSNode) -> None:
        for node in collect_all_nodes(root):
            node.accumulated_value = 0.0
            node.terminal_in_subtree = 0
            node.correct_terminal_in_subtree = 0
            node.selected_terminal_in_subtree = 0
            node.state_value = 0.0
            node.segment_reward = 0.0
            node.leaf_outcome = 0.0
            node.llm_quality_score = 0.0
            node.is_correct = None
            node.main_chain = False

    def _apply_rloo_to_leaves(self, leaves: List[MCTSNode]) -> None:
        # Use leaf.R (may already be fused with LLM quality score) for RLOO normalization.
        outcomes = [float(leaf.R) for leaf in leaves]
        if len(leaves) <= 1:
            for leaf, outcome in zip(leaves, outcomes):
                leaf.R = outcome
                leaf.accumulated_value = leaf.R
            return

        total = sum(outcomes)
        denom = len(outcomes) - 1
        for leaf, outcome in zip(leaves, outcomes):
            mean_others = (total - outcome) / denom
            leaf.R = outcome - mean_others
            leaf.accumulated_value = leaf.R

    def _evaluate_leaves_quality(
        self,
        roots: List[MCTSNode],
        leaves: List[MCTSNode],
        gen_batch: DataProto,
    ) -> None:
        """Refine leaf outcome with LLM trajectory quality evaluation.

        Calls an external LLM (vLLM OpenAI-compatible server) to judge each
        leaf's trajectory for logical correctness, hallucination, and internal
        coherence.  The quality score (0 or 1) is blended into the leaf's
        accumulated_value before RLOO normalization:

            accumulated_value = binary_outcome + coeff * llm_quality_score

        This mirrors the original TreeRL external RM:
            accumulated_value = reward + 0.5 * sigmoid(value)
        """
        from verl.utils.trajectory_eval import evaluate_trajectories

        # Prepare per-leaf context
        questions: List[str] = []
        ground_truths: List[Optional[str]] = []
        trajectories: List[str] = []

        # Ensure question texts are populated for each tree
        self._ensure_question_texts(gen_batch)

        for leaf in leaves:
            tree_idx = leaf.tree_idx
            question_text = self._question_texts_by_tree.get(tree_idx, "")
            gt = self._get_ground_truths(gen_batch, len(roots))[tree_idx]
            trajectory = leaf.accumulated_text or ""
            questions.append(question_text)
            ground_truths.append(gt)
            trajectories.append(trajectory)

        print(
            f"[StepTreeRL] Evaluating {len(trajectories)} leaves via LLM RM "
            f"at {self.trajectory_rm_url} (model={self.trajectory_rm_model})"
        )

        quality_scores = evaluate_trajectories(
            questions=questions,
            ground_truths=ground_truths,
            trajectories=trajectories,
            rm_url=self.trajectory_rm_url,
            model_name=self.trajectory_rm_model,
            max_tokens=self.trajectory_rm_max_tokens,
            temperature=self.trajectory_rm_temperature,
            api_key=self.trajectory_rm_api_key,
        )

        # Log raw LLM RM scores for debugging
        base_outcomes = [float(leaf.leaf_outcome) for leaf in leaves]
        print(
            f"[StepTreeRL] LLM RM done. "
            f"leaf_outcomes={base_outcomes} "
            f"llm_quality_scores={quality_scores}"
        )

        # Blend quality score into leaf.accumulated_value BEFORE RLOO normalization.
        # This follows the original TreeRL formula: accumulated_value = reward + coeff * sigmoid(rm_score)
        for leaf, qscore in zip(leaves, quality_scores):
            leaf.llm_quality_score = float(qscore)
            fused = float(leaf.leaf_outcome) + self.trajectory_rm_coeff * leaf.llm_quality_score
            leaf.R = fused
            leaf.accumulated_value = fused

    def _leaf_backpropagate_value(self, leaf: MCTSNode) -> None:
        """Backpropagate a RLOO leaf value to all ancestors."""
        leaf.terminal_in_subtree += 1
        if leaf.is_correct:
            leaf.correct_terminal_in_subtree += 1
        node = leaf
        while node.parent is not None:
            node.parent.terminal_in_subtree += 1
            if leaf.is_correct:
                node.parent.correct_terminal_in_subtree += 1
            node.parent.accumulated_value += leaf.accumulated_value
            node = node.parent

    def _normalize_all_steps(self, root: MCTSNode) -> None:
        """TreeRL-style optional normalization over all nodes covered by candidate leaves."""
        style = self.overall_norm_style
        if style in {"", "none", "null"}:
            return

        all_steps = [
            node
            for node in collect_all_nodes(root)
            if node.terminal_in_subtree != 0 or node.terminal
        ]
        if not all_steps:
            return

        if style == "step":
            step_sum = sum(node.accumulated_value for node in all_steps)
            step_num = sum(node.terminal_in_subtree for node in all_steps)
            mean = step_sum / step_num if step_num > 0 else 0.0
            for node in all_steps:
                node.accumulated_value -= mean * node.terminal_in_subtree
            return

        if style == "token":
            step_sum = 0.0
            step_num = 0
            for node in all_steps:
                token_len = len(node.step_tokens)
                step_sum += node.accumulated_value * token_len
                step_num += node.terminal_in_subtree * token_len
            mean = step_sum / step_num if step_num > 0 else 0.0
            for node in all_steps:
                node.accumulated_value -= mean * node.terminal_in_subtree
            return

        raise ValueError(
            f"Unsupported trainer.step_treerl_config.overall_norm_style={self.overall_norm_style!r}. "
            "Expected 'none', 'step', or 'token'."
        )

    def _selected_backpropagate(self, leaf: MCTSNode) -> None:
        leaf.selected_terminal_in_subtree += 1
        node = leaf
        while node.parent is not None:
            node.parent.selected_terminal_in_subtree += 1
            node = node.parent

    def _compute_weighted_update(self, node: MCTSNode) -> None:
        if node.selected_terminal_in_subtree > 0:
            if self.weighted_value_style == "terminal_ratio":
                # Original TreeRL formula: upweight accumulated_value when a subtree
                # has many terminals but few of them are selected for training.
                # This concentrates the information from unselected terminals into
                # the selected ones.
                node.accumulated_value = (
                    node.accumulated_value
                    * node.terminal_in_subtree
                    / node.selected_terminal_in_subtree
                )
            elif self.weighted_value_style == "sqrt":
                node.accumulated_value = node.accumulated_value / math.sqrt(node.selected_terminal_in_subtree)
            elif self.weighted_value_style == "uniform":
                node.accumulated_value = node.accumulated_value / node.selected_terminal_in_subtree
            elif self.weighted_value_style == "original":
                node.accumulated_value = node.accumulated_value
            else:
                raise ValueError(
                    f"Unsupported trainer.step_treerl_config.weighted_value_style={self.weighted_value_style!r}. "
                    "Expected 'terminal_ratio', 'sqrt', 'uniform', or 'original'."
                )
        for child in node.children:
            self._compute_weighted_update(child)

    def _assign_segment_rewards(self, root: MCTSNode) -> None:
        nodes = [node for node in collect_all_nodes(root) if node.parent is not None]
        if not nodes:
            return
        max_depth = max(1, max(node.depth for node in nodes))
        for node in nodes:
            if node.terminal_in_subtree <= 0 or node.parent is None or node.parent.terminal_in_subtree <= 0:
                node.state_value = 0.0
                node.segment_reward = 0.0 if self.use_origin_advantage else self._length_penalty(node.depth, max_depth)
                continue
            parent_value = node.parent.accumulated_value / node.parent.terminal_in_subtree
            child_value = node.accumulated_value / node.terminal_in_subtree
            node.state_value = child_value
            if self.use_origin_advantage:
                if not self.use_branch_point_search:
                    node.segment_reward = 2.0 * child_value - parent_value
                elif node.parent.is_branch_point:
                    node.segment_reward = 2.0 * child_value - parent_value
                else:
                    branch_point = self._find_branch_point_ancestor(node)
                    if branch_point is not None and branch_point.terminal_in_subtree > 0:
                        branch_point_value = branch_point.accumulated_value / branch_point.terminal_in_subtree
                        node.segment_reward = child_value - branch_point_value
                    else:
                        node.segment_reward = 0.0
                continue
            tree_advantage = child_value - parent_value
            if getattr(node, "node_type", "step") == "answer":
                node.segment_reward = float(node.R)
            else:
                node.segment_reward = tree_advantage * float(node.process_reward) + self._length_penalty(node.depth, max_depth)

    def _length_penalty(self, step_index: int, max_step: int) -> float:
        if not self.length_penalty_enabled:
            return 0.0
        t = float(step_index) / max(float(max_step), 1.0)
        return -self.length_penalty_p_max * (1.0 / (1.0 + math.exp(-self.length_penalty_k * (t - self.length_penalty_t0))))

    def _find_branch_point_ancestor(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Return the nearest branch-point ancestor, or None if root is reached."""
        cur = node.parent
        while cur is not None:
            if cur.is_branch_point:
                return cur
            cur = cur.parent
        return None

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
        """Output selected terminal paths with GPU padding."""
        all_paths: List[List[MCTSNode]] = []
        all_gt: List[Optional[str]] = []
        self._metrics = self._collect_metrics(roots)

        for i, root in enumerate(roots):
            gt = ground_truths[i] if i < len(ground_truths) else None

            # Collect all leaves (initial + branched)
            candidate_leaves = [n for n in collect_all_nodes(root) if not n.children]

            if not candidate_leaves:
                # Fallback: deepest non-root node
                all_nodes = collect_all_nodes(root)
                non_root = [n for n in all_nodes if n.parent is not None]
                candidate_leaves = [max(non_root, key=lambda n: n.depth)] if non_root else [root]

            if self.path_selection == "selected_terminals":
                num_traces = int(self.selected_num_traces or self.rollout_n or len(candidate_leaves))
                leaves, terminal_padding = self._select_terminals(candidate_leaves, num_traces)
            elif self.path_selection != "all_leaves":
                raise ValueError(
                    f"Unsupported trainer.step_treerl_config.path_selection={self.path_selection!r}. "
                    "Expected 'all_leaves' or 'selected_terminals'."
                )
            else:
                leaves = list(candidate_leaves)
                terminal_padding = 0

            self._metrics.candidate_leaves += len(candidate_leaves)
            self._metrics.selected_traces += len(leaves)
            self._metrics.terminal_padding += terminal_padding
            for leaf in leaves:
                self._record_trace_format_metrics(leaf)
            for leaf in leaves:
                self._selected_backpropagate(leaf)
            if self.use_weighted_value:
                self._compute_weighted_update(root)
            self._assign_segment_rewards(root)

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

        # Average number of reasoning steps in the actual actor-training batch.
        # Answer nodes (the trailing boxed answer) are intentionally excluded;
        # repeated terminal/GPU-padding paths are included because they are trained.
        if all_paths:
            self._metrics.step_num = sum(
                sum(1 for node in path if getattr(node, "node_type", "step") == "step")
                for path in all_paths
            ) / len(all_paths)

        return _build_sampling_result(
            all_paths,
            all_gt,
            self.pad_token_id,
            device,
            len(roots),
            metrics=self._metrics,
        )

    def _select_terminals(self, leaves: List[MCTSNode], num_traces: int) -> Tuple[List[MCTSNode], int]:
        if num_traces <= 0:
            return [], 0
        if not leaves:
            return [], 0

        shuffled = list(leaves)
        random.shuffle(shuffled)
        selected: List[MCTSNode] = []
        remaining: List[MCTSNode] = []

        for leaf in shuffled:
            if leaf.main_chain and not selected:
                selected.append(leaf)
            else:
                remaining.append(leaf)

        needed = num_traces - len(selected)
        if needed > 0 and remaining:
            sampled = random.sample(remaining, min(needed, len(remaining)))
            selected.extend(sampled)
            needed = num_traces - len(selected)

        padding = 0
        fill_source = selected or remaining or shuffled
        while needed > 0 and fill_source:
            selected.append(random.choice(fill_source))
            padding += 1
            needed -= 1

        random.shuffle(selected)
        return selected[:num_traces], padding

    def _record_trace_format_metrics(self, leaf: MCTSNode) -> None:
        response_text = leaf.trajectory_text or leaf.accumulated_text
        classes = classify_trajectory_format(response_text)
        self._metrics.trace_total += 1
        self._metrics.full_format_correct_count += int(classes["format_full"])
        self._metrics.answer_format_only_count += int(classes["format_answer_only"])
        self._metrics.step_format_only_count += int(classes["format_step_only"])

    def _collect_metrics(self, roots: List[MCTSNode]) -> StepTreeRLMetrics:
        metrics = StepTreeRLMetrics()
        metrics.problem_count = len(roots)
        for root in roots:
            nodes = [node for node in collect_all_nodes(root) if node.parent is not None]
            step_nodes = [node for node in nodes if getattr(node, "node_type", "step") == "step"]
            metrics.total_steps += len(step_nodes)
            metrics.format_steps += sum(1 for node in step_nodes if format_step_reward(node.step_text) > 0.0)
            metrics.process_reward_sum += sum(float(node.process_reward) for node in step_nodes)
            metrics.process_reward_count += len(step_nodes)
            leaves = [node for node in nodes if not node.children]
            metrics.leaf_total += len(leaves)
            metrics.leaf_correct += sum(1 for leaf in leaves if bool(leaf.is_correct))
            # Accumulate LLM RM quality scores from leaves that have them
            for leaf in leaves:
                qs = getattr(leaf, "llm_quality_score", None)
                if qs is not None:
                    metrics.llm_rm_score_sum += float(qs)
                    metrics.llm_rm_score_count += 1
        return metrics

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
                "calculate_entropy": False,
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
    metrics: Optional[StepTreeRLMetrics] = None,
) -> SamplingResult:
    """Build padded tensors and SamplingResult from selected paths."""
    prompt_ids_list: List[torch.Tensor] = []
    full_seq_list: List[torch.Tensor] = []
    resp_ids_list: List[torch.Tensor] = []
    step_spans_list: List[List[Tuple[int, int]]] = []
    step_rewards_list: List[List[float]] = []
    step_correctness_scores_list: List[List[float]] = []
    response_lens: List[int] = []
    leaf_accuracy_list: List[float] = []

    for path, gt in zip(all_paths, all_gt):
        if not path:
            continue
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
            rewards.append(float(node.segment_reward))
            v_score = float(node.state_value)
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
        leaf_accuracy_list.append(1.0 if path[-1].is_correct else 0.0)

    if not prompt_ids_list:
        raise ValueError("StepTreeRL produced no non-empty training paths.")

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
    n_paths = len(prompt_ids_list)
    max_steps = max((len(s) for s in step_spans_list), default=1)
    score_ids = torch.full((n_paths, max_steps), -1, device=device, dtype=torch.long)
    reward_mask = torch.zeros(n_paths, max_steps, device=device, dtype=torch.float32)
    for i, (spans, rlen) in enumerate(zip(step_spans_list, response_lens)):
        for j, (_, end) in enumerate(spans[:max_steps]):
            end_pos = max(0, min(end - 1, rlen - 1)) if rlen > 0 else 0
            score_ids[i, j] = end_pos
            reward_mask[i, j] = 1.0

    # ORM leaf accuracy for logging/debugging. This is not used as a reward.
    max_resp_len = responses_padded.size(1)
    leaf_accuracy = torch.zeros(n_paths, max_resp_len, dtype=torch.float32, device=device)
    for i, rlen in enumerate(response_lens):
        if rlen > 0:
            leaf_accuracy[i, rlen - 1] = leaf_accuracy_list[i]

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
            "leaf_accuracy": leaf_accuracy,
            "step_correctness_scores": step_correctness_padded,
        },
        non_tensors={},
        meta_info={
            "step_treerl_metrics": {
                "format_steps": metrics.format_steps if metrics is not None else 0,
                "total_steps": metrics.total_steps if metrics is not None else 0,
                "problem_count": metrics.problem_count if metrics is not None else 0,
                "steps_per_problem": (
                    metrics.total_steps / metrics.problem_count
                    if metrics is not None and metrics.problem_count > 0
                    else 0.0
                ),
                "format_ratio": (metrics.format_steps / metrics.total_steps) if metrics is not None and metrics.total_steps > 0 else 0.0,
                "process_reward_mean": (
                    metrics.process_reward_sum / metrics.process_reward_count
                    if metrics is not None and metrics.process_reward_count > 0
                    else 0.0
                ),
                "leaf_acc": (metrics.leaf_correct / metrics.leaf_total) if metrics is not None and metrics.leaf_total > 0 else 0.0,
                "llm_rm_score": (
                    metrics.llm_rm_score_sum / metrics.llm_rm_score_count
                    if metrics is not None and metrics.llm_rm_score_count > 0
                    else 0.0
                ),
                "candidate_leaves": metrics.candidate_leaves if metrics is not None else 0,
                "selected_traces": metrics.selected_traces if metrics is not None else 0,
                "step_num": metrics.step_num if metrics is not None else 0.0,
                "terminal_padding": metrics.terminal_padding if metrics is not None else 0,
                "trace_total": metrics.trace_total if metrics is not None else 0,
                "full_format_correct_count": metrics.full_format_correct_count if metrics is not None else 0,
                "answer_format_only_count": metrics.answer_format_only_count if metrics is not None else 0,
                "step_format_only_count": metrics.step_format_only_count if metrics is not None else 0,
                "full_format_correct_ratio": (
                    metrics.full_format_correct_count / metrics.trace_total
                    if metrics is not None and metrics.trace_total > 0
                    else 0.0
                ),
                "answer_format_only_ratio": (
                    metrics.answer_format_only_count / metrics.trace_total
                    if metrics is not None and metrics.trace_total > 0
                    else 0.0
                ),
                "step_format_only_ratio": (
                    metrics.step_format_only_count / metrics.trace_total
                    if metrics is not None and metrics.trace_total > 0
                    else 0.0
                ),
            }
        },
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
