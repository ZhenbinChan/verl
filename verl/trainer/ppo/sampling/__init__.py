"""Pluggable post-rollout sampling strategies.

Usage in the trainer::

    from verl.trainer.ppo.sampling import create_sampling_strategy

    strategy = create_sampling_strategy(config, tokenizer)
    if strategy is not None:
        result = strategy.run(gen_batch, gen_batch_output, generate_fn, compute_log_prob_fn, timing_raw)
"""

from __future__ import annotations

from typing import Optional

from verl.trainer.ppo.sampling.base import SamplingResult, SamplingStrategy

__all__ = [
    "SamplingStrategy",
    "SamplingResult",
    "create_sampling_strategy",
]

# Canonical strategy names -> strategy classes (lazy-imported).
_STRATEGY_REGISTRY = {
    "tree_search": "verl.trainer.ppo.sampling.tree_sampling.TreeSamplingStrategy",
    "treerl": "verl.trainer.ppo.sampling.entropy_chain.EntropyChainStrategy",
    # "mcts": "verl.trainer.ppo.sampling.mcts.MCTSStrategy",  # reserved
}


def create_sampling_strategy(config, tokenizer) -> Optional[SamplingStrategy]:
    """Instantiate the sampling strategy specified by ``trainer.sampling_strategy``.

    Accepted values (case-insensitive):
      - ``null`` / ``None`` / ``"none"`` -> no strategy (plain rollout)
      - ``"tree_search"`` -> :class:`TreeSamplingStrategy`
      - ``"treerl"``      -> :class:`EntropyChainStrategy`
      - ``"mcts"``        -> (reserved, not yet implemented)

    """
    name = config.trainer.get("sampling_strategy", None)

    if name is None or str(name).lower() in ("none", "null", "false"):
        return None

    name = str(name).lower().strip()
    if name not in _STRATEGY_REGISTRY:
        supported = ", ".join(sorted(_STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown sampling_strategy '{name}'. Supported: {supported}"
        )

    # Lazy import to avoid pulling heavy deps when not needed.
    import importlib

    module_path, cls_name = _STRATEGY_REGISTRY[name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls(config, tokenizer)
