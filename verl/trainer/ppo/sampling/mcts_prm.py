from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from verl.utils.fol_verifier import FOLMetadata, FOLVerifier


def format_step_reward(step_text: str) -> float:
    """Returns 1.0 if step_text contains ≥1 <premise> and exactly 1 <conclusion>."""
    premises = re.findall(r"<premise>.*?</premise>", step_text, re.DOTALL)
    conclusions = re.findall(r"<conclusion>.*?</conclusion>", step_text, re.DOTALL)
    return 1.0 if (len(premises) >= 1 and len(conclusions) == 1) else 0.0


def strict_step_xml_correct(step_text: str) -> bool:
    """Return True when a step is strictly composed of premise tags and one conclusion tag."""
    try:
        root = ET.fromstring(step_text.strip())
    except ET.ParseError:
        return False

    if root.tag != "step":
        return False
    if root.text and root.text.strip():
        return False

    premise_count = 0
    conclusion_count = 0
    for child in list(root):
        if child.tag == "premise":
            premise_count += 1
        elif child.tag == "conclusion":
            conclusion_count += 1
        else:
            return False
        if list(child):
            return False
        if child.tail and child.tail.strip():
            return False

    return premise_count >= 1 and conclusion_count == 1


def _boxed_answer_match(response_text: str):
    return re.search(r"\\boxed\{\{?\s*([A-Za-z])\s*\}?\}\s*$", response_text, re.DOTALL)


def boxed_answer_format_correct(response_text: str, valid_choices: Optional[str] = None) -> bool:
    """Return True when the response ends with a boxed single-letter answer."""
    match = _boxed_answer_match(response_text)
    if not match:
        return False
    answer = match.group(1).upper()
    if valid_choices is None:
        return True
    return answer in {choice.upper() for choice in valid_choices}


def _step_blocks_cover_text(text: str) -> tuple[bool, list[str]]:
    matches = list(re.finditer(r"<step>.*?</step>", text, re.DOTALL))
    if not matches:
        return False, []

    cursor = 0
    blocks = []
    for match in matches:
        if text[cursor:match.start()].strip():
            return False, []
        block = match.group(0)
        if not strict_step_xml_correct(block):
            return False, []
        blocks.append(block)
        cursor = match.end()

    if text[cursor:].strip():
        return False, []
    return True, blocks


def classify_trajectory_format(response_text: str, valid_choices: Optional[str] = None) -> Dict[str, float]:
    """Classify a full trajectory into mutually exclusive format buckets."""
    answer_match = _boxed_answer_match(response_text)
    answer_ok = False
    step_region = response_text
    if answer_match is not None:
        answer = answer_match.group(1).upper()
        answer_ok = valid_choices is None or answer in {choice.upper() for choice in valid_choices}
        step_region = response_text[:answer_match.start()]

    step_ok, _ = _step_blocks_cover_text(step_region)

    full = float(step_ok and answer_ok)
    answer_only = float(answer_ok and not step_ok)
    step_only = float(step_ok and not answer_ok)
    incorrect = float(not (full or answer_only or step_only))
    return {
        "format_full": full,
        "format_answer_only": answer_only,
        "format_step_only": step_only,
        "format_incorrect": incorrect,
        "format_trace_total": 1.0,
    }


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
    return verifier.verify_step(metadata, step_text, use_llm=True)


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
    if sample_id is None:
        raise ValueError("FOL step reward requires sample_id.")
    if sample_id not in sample_metadata_map:
        raise KeyError(f"Missing FOL metadata for sample_id={sample_id!r}.")
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
