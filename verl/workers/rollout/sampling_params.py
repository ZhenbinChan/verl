"""Helpers for per-request rollout sampling overrides."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ROLLOUT_SAMPLING_KWARG_KEYS = frozenset(
    {
        "best_of",
        "ignore_eos",
        "max_new_tokens",
        "max_tokens",
        "min_p",
        "n",
        "temperature",
        "top_k",
        "top_p",
    }
)


def extract_rollout_sampling_kwargs(meta_info: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return namespaced per-request sampling overrides from DataProto.meta_info."""
    if not meta_info:
        return {}

    overrides = meta_info.get("rollout_sampling_kwargs")
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise TypeError("meta_info['rollout_sampling_kwargs'] must be a mapping when provided.")

    return {key: value for key, value in overrides.items() if key in ROLLOUT_SAMPLING_KWARG_KEYS and value is not None}
