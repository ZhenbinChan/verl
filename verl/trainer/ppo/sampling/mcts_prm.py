from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from verl.utils.fol_verifier import FOLMetadata, FOLVerifier


def format_step_reward(step_text: str) -> float:
    """Returns 1.0 if step_text contains ≥1 <premise> and exactly 1 <conclusion>."""
    premises = re.findall(r"<premise>.*?</premise>", step_text, re.DOTALL)
    conclusions = re.findall(r"<conclusion>.*?</conclusion>", step_text, re.DOTALL)
    return 1.0 if (len(premises) >= 1 and len(conclusions) == 1) else 0.0


def fol_step_reward(
    step_text: str,
    *,
    metadata: "FOLMetadata",
    verifier: "FOLVerifier",
) -> float:
    """FOL/Z3-based step verification reward.

    Verifies the logical relationship between <premise> and <conclusion> tags.

    Args:
        step_text: The step text to verify.
        metadata: Pre-computed FOL metadata (context, declarations, etc.)
        verifier: FOL verifier instance.

    Returns:
        1.0 if unsat (conclusion follows from premises), 0.0 otherwise.
    """
    try:
        return verifier.verify_step(metadata, step_text, use_llm=True)
    except Exception:
        return 0.0


def fol_step_reward_with_context(
    step_text: str,
    *,
    sample_id: str,
    sample_metadata_map: Dict[str, "FOLMetadata"],
    verifier: "FOLVerifier",
) -> float:
    """FOL step reward with sample_id lookup.

    Used for batch verification where sample metadata is looked up by sample_id.
    """
    if sample_id not in sample_metadata_map:
        return 0.0
    metadata = sample_metadata_map[sample_id]
    return fol_step_reward(step_text, metadata=metadata, verifier=verifier)


def get_prm_fn(
    prm_type: str,
    **kwargs,
) -> Callable:
    """Return the PRM scoring function for the given type.

    Args:
        prm_type: Type of PRM ('format' or 'fol').
        **kwargs: Additional parameters for FOL PRM:
            - verifier: FOLVerifier instance (required for 'fol')
            - metadata_map: Dict[str, FOLMetadata] for batch lookup

    Supported:
        'format': checks <step>/<premise>/<conclusion> tag structure.
        'fol':    FOL/Z3 verification (requires verifier in kwargs).
    """
    if prm_type == "format":
        return format_step_reward

    elif prm_type == "fol":
        if "verifier" not in kwargs:
            raise ValueError(
                "FOL PRM requires 'verifier' parameter (FOLVerifier instance)"
            )
        verifier = kwargs["verifier"]

        if "metadata_map" in kwargs:
            # Batch mode: look up metadata by sample_id
            metadata_map = kwargs["metadata_map"]
            return lambda step_text, sample_id=None: fol_step_reward_with_context(
                step_text,
                sample_id=sample_id,
                sample_metadata_map=metadata_map,
                verifier=verifier,
            )
        else:
            # Single sample mode
            return lambda step_text, metadata=None: fol_step_reward(
                step_text, metadata=metadata, verifier=verifier
            )

    else:
        raise ValueError(f"Unknown PRM type: {prm_type!r}. Supported: 'format', 'fol'")
