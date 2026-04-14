from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

from verl import DataProto
from verl.trainer.ppo.sampling.base import SamplingResult, SamplingStrategy


@contextmanager
def _timer(name: str, dest: dict):
    import time

    start = time.perf_counter()
    yield
    dest[name] = time.perf_counter() - start


class TreeSamplingStrategy(SamplingStrategy):
    """Step/token-level tree expansion strategy (TreeRL).

    Wraps :class:`~verl.utils.tree_structure.TreeManager` and exposes the full
    *initialize -> expand -> backpropagate -> build* pipeline through the
    unified :meth:`run` interface.
    """

    def __init__(self, config, tokenizer):
        from verl.utils.tree_structure import TreeManager

        self._config = config
        branch_level = config.trainer.get("branch_level", "step")
        step_reward_type = config.trainer.get("step_reward_type", "random")

        self._tree_manager = TreeManager(
            tokenizer=tokenizer,
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
            branch_level=branch_level,
            step_reward_type=step_reward_type,
        )
        self._branch_level = branch_level

    def run(
        self,
        gen_batch: DataProto,
        gen_batch_output: DataProto,
        generate_fn: Callable[[DataProto], DataProto],
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        timing_raw: dict,
    ) -> SamplingResult:
        tm = self._tree_manager
        tm.branch_level = self._branch_level

        tm.initialize_trees(
            gen_batch=gen_batch,
            gen_batch_output=gen_batch_output,
            compute_log_prob_fn=compute_log_prob_fn,
        )

        tree_rounds = self._config.trainer.get("tree_rounds", 1)
        tree_top_k = self._config.trainer.get("tree_top_k", 1)
        for _ in range(max(1, tree_rounds)):
            top_k_nodes = tm.get_top_k_entropy_nodes(tree_top_k)
            branch_plan = tm.prepare_branches(target_nodes=top_k_nodes)
            if branch_plan is not None and branch_plan.branch_batch is not None and branch_plan.batch_size > 0:
                with _timer("gen_branch", timing_raw):
                    branch_gen_output = generate_fn(branch_plan.branch_batch)
                tm.commit_branch_outputs(
                    branch_gen_output, branch_plan, compute_log_prob_fn=compute_log_prob_fn
                )

        for i in range(len(tm.trees)):
            tm.pretty_print_tree(i)

        tm.backpropagate_correctness()
        tm.compute_q_values(gamma=self._config.algorithm.get("gamma", 1))
        tm.apply_treerl_rewards()

        tree_batch_output = tm.build_response_batch(
            batch_index_list=None,
            gen_batch_output=gen_batch_output,
        )
        final_output = tree_batch_output if tree_batch_output is not None else gen_batch_output

        prompt_batch_size = int(gen_batch.batch.batch_size[0])
        expanded_batch_size = int(final_output.batch.batch_size[0])
        if prompt_batch_size <= 0 or expanded_batch_size % prompt_batch_size != 0:
            raise ValueError(
                f"Tree sampling output batch size {expanded_batch_size} "
                f"is not divisible by prompt batch size {prompt_batch_size}."
            )
        repeat_times = expanded_batch_size // prompt_batch_size

        return SamplingResult(gen_batch_output=final_output, repeat_times=repeat_times)
