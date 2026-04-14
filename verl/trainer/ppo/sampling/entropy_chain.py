from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

from verl import DataProto
from verl.trainer.ppo.sampling.base import SamplingResult, SamplingStrategy


@contextmanager
def _timer(name: str, dest: dict):
    """Reuse the same lightweight timer used in *ray_trainer*."""
    import time

    start = time.perf_counter()
    yield
    dest[name] = time.perf_counter() - start


class EntropyChainStrategy(SamplingStrategy):
    """Entropy-guided chain expansion strategy.

    Wraps :class:`~verl.utils.entropy_chain_expander.EntropyChainExpander` and
    exposes the full *initialize -> expand -> build* pipeline through the
    unified :meth:`run` interface.
    """

    def __init__(self, config, tokenizer):
        from verl.utils.entropy_chain_expander import EntropyChainExpander

        cfg = config.trainer.get("entropy_chain_config", {})
        self._expander = EntropyChainExpander(
            tokenizer=tokenizer,
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
            N=cfg.get("N", cfg.get("n", 4)),
            L=cfg.get("L", cfg.get("l", 3)),
            T=cfg.get("T", cfg.get("t", 1)),
            max_token_num=cfg.get("max_token_num", 4096),
            evaluation_strategy=cfg.get("evaluation_strategy", "token-entropy"),
            enforce_uniform_per_prompt=cfg.get("enforce_uniform_per_prompt", True),
        )

    def run(
        self,
        gen_batch: DataProto,
        gen_batch_output: DataProto,
        generate_fn: Callable[[DataProto], DataProto],
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        timing_raw: dict,
    ) -> SamplingResult:
        with _timer("entropy_chain", timing_raw):
            self._expander.initialize(
                gen_batch=gen_batch,
                gen_batch_output=gen_batch_output,
                compute_log_prob_fn=compute_log_prob_fn,
            )

            for _ in range(max(0, self._expander.L)):
                has_expansion = self._expander.expand_one_round(
                    generate_fn=generate_fn,
                    compute_log_prob_fn=compute_log_prob_fn,
                )
                if not has_expansion:
                    break

            expanded_output = self._expander.build_expanded_batch()

        prompt_batch_size = int(gen_batch.batch.batch_size[0])
        expanded_batch_size = int(expanded_output.batch.batch_size[0])
        if prompt_batch_size <= 0 or expanded_batch_size % prompt_batch_size != 0:
            raise ValueError(
                f"Entropy-chain output batch size {expanded_batch_size} "
                f"is not divisible by prompt batch size {prompt_batch_size}."
            )
        repeat_times = expanded_batch_size // prompt_batch_size

        return SamplingResult(gen_batch_output=expanded_output, repeat_times=repeat_times)
