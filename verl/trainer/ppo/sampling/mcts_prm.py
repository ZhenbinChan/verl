from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from verl.utils.fol_verifier import FOLMetadata, FOLVerifier


def format_step_reward(step_text: str) -> float:
    """Returns 1.0 if step_text is one strictly formatted step."""
    return 1.0 if strict_step_xml_correct(step_text) else 0.0


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


BOXED_ANSWER_RE = re.compile(r"\\boxed\{(?:\{\s*([A-Za-z])\s*\}|\s*([A-Za-z])\s*)\}\s*$", re.DOTALL)
STEP_BLOCK_RE = re.compile(r"<step\b[^>]*>.*?</step>", re.DOTALL)
STEP_TAG_RE = re.compile(r"</?step\b", re.DOTALL)
FORMAT_PRIMARY_CATEGORIES = (
    "full",
    "no_step",
    "text_outside_step",
    "step_xml_invalid",
    "step_schema_invalid",
    "boxed_missing",
    "boxed_invalid",
)


def _boxed_answer_match(response_text: str):
    return BOXED_ANSWER_RE.search(response_text or "")


def boxed_answer_format_correct(response_text: str, valid_choices: Optional[str] = None) -> bool:
    """Return True when the response ends with a boxed single-letter answer."""
    match = _boxed_answer_match(response_text)
    if not match:
        return False
    answer = (match.group(1) or match.group(2)).upper()
    if valid_choices is None:
        return True
    return answer in {choice.upper() for choice in valid_choices}


def _step_blocks_cover_text(text: str) -> tuple[bool, list[str]]:
    matches = list(STEP_BLOCK_RE.finditer(text or ""))
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
        answer = (answer_match.group(1) or answer_match.group(2)).upper()
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


def _split_answer_region(response_text: str, valid_choices: Optional[str] = None) -> tuple[str, str, str]:
    match = _boxed_answer_match(response_text)
    if match:
        answer = (match.group(1) or match.group(2)).upper()
        if valid_choices is None or answer in {choice.upper() for choice in valid_choices}:
            return response_text[: match.start()], "valid", answer
        return response_text[: match.start()], "invalid", ""

    boxed_start = response_text.rfind("\\boxed")
    if boxed_start >= 0:
        return response_text[:boxed_start], "invalid", ""

    return response_text, "missing", ""


def _step_block_status(step_text: str) -> str:
    try:
        root = ET.fromstring(step_text.strip())
    except ET.ParseError:
        return "xml_invalid"

    if root.tag != "step":
        return "schema_invalid"
    if root.text and root.text.strip():
        return "schema_invalid"

    premise_count = 0
    conclusion_count = 0
    for child in list(root):
        if child.tag == "premise":
            premise_count += 1
        elif child.tag == "conclusion":
            conclusion_count += 1
        else:
            return "schema_invalid"
        if list(child):
            return "schema_invalid"
        if child.tail and child.tail.strip():
            return "schema_invalid"

    if premise_count < 1 or conclusion_count != 1:
        return "schema_invalid"
    return "ok"


def classify_rollout_format(response_text: str, valid_choices: Optional[str] = None) -> Dict[str, object]:
    """Classify one complete rollout trajectory into exactly one primary format bucket."""
    response_text = response_text or ""
    step_region, boxed_status, boxed_answer = _split_answer_region(response_text, valid_choices)
    step_matches = list(STEP_BLOCK_RE.finditer(step_region))
    step_block_count = len(step_matches)

    if not step_matches:
        if STEP_TAG_RE.search(step_region):
            primary = "step_xml_invalid"
        else:
            primary = "no_step"
    else:
        cursor = 0
        text_outside_step = False
        for match in step_matches:
            if step_region[cursor : match.start()].strip():
                text_outside_step = True
                break
            cursor = match.end()
        if not text_outside_step and step_region[cursor:].strip():
            text_outside_step = True

        if text_outside_step:
            primary = "text_outside_step"
        else:
            statuses = [_step_block_status(match.group(0)) for match in step_matches]
            if "xml_invalid" in statuses:
                primary = "step_xml_invalid"
            elif "schema_invalid" in statuses:
                primary = "step_schema_invalid"
            elif boxed_status == "missing":
                primary = "boxed_missing"
            elif boxed_status == "invalid":
                primary = "boxed_invalid"
            else:
                primary = "full"

    return {
        "format_primary": primary,
        "boxed_status": boxed_status,
        "boxed_answer": boxed_answer,
        "step_block_count": float(step_block_count),
    }


def aggregate_rollout_format_metrics(format_infos: list[Dict[str, object]]) -> Dict[str, float]:
    """Aggregate trainer-level rollout format classifications."""
    total = float(len(format_infos))
    if total <= 0:
        return {}

    counts = {category: 0.0 for category in FORMAT_PRIMARY_CATEGORIES}
    for info in format_infos:
        primary = info.get("format_primary")
        if primary in counts:
            counts[primary] += 1.0

    metrics = {"rollout/format_primary/total": total}
    for category in FORMAT_PRIMARY_CATEGORIES:
        metrics[f"rollout/format_primary/{category}_ratio"] = counts[category] / total
    return metrics


def rollout_format_infos_to_columns(format_infos: list[Dict[str, object]]) -> Dict[str, list]:
    """Convert per-rollout format info to JSONL-compatible columns."""
    return {
        "format_primary": [str(info.get("format_primary", "")) for info in format_infos],
        "boxed_status": [str(info.get("boxed_status", "")) for info in format_infos],
        "boxed_answer": [str(info.get("boxed_answer", "")) for info in format_infos],
        "step_block_count": [float(info.get("step_block_count", 0.0)) for info in format_infos],
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
