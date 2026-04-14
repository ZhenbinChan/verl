from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from verl import DataProto


@dataclass
class SamplingResult:
    """Output returned by a :class:`SamplingStrategy`."""

    gen_batch_output: DataProto
    repeat_times: int


class SamplingStrategy(ABC):
    """Abstract interface for post-rollout sampling strategies.

    Concrete subclasses (e.g. *TreeSamplingStrategy*, *EntropyChainStrategy*)
    encapsulate all orchestration logic so that the main training loop only
    needs a single ``strategy.run(...)`` call.
    """

    @abstractmethod
    def run(
        self,
        gen_batch: DataProto,
        gen_batch_output: DataProto,
        generate_fn: Callable[[DataProto], DataProto],
        compute_log_prob_fn: Callable[[DataProto], DataProto],
        timing_raw: dict,
    ) -> SamplingResult:
        """Execute the full sampling pipeline.

        Args:
            gen_batch: Original prompt batch (before generation).
            gen_batch_output: Initial generation output (responses).
            generate_fn: Calls ``generate_sequences`` on the appropriate
                worker group.  Async wake/sleep is handled by the caller.
            compute_log_prob_fn: Calls ``compute_log_prob`` on the actor
                worker group.
            timing_raw: Dict for ``_timer`` performance bookkeeping.

        Returns:
            A :class:`SamplingResult` containing the (possibly modified)
            ``gen_batch_output`` and the ``repeat_times`` that the trainer
            should use when repeating the prompt batch.
        """
        ...
