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


BOXED_ANSWER_RE = re.compile(r"\\boxed\{(?:\{\s*(?:\(\s*([A-Za-z])\s*\)|([A-Za-z]))\s*\}|\s*(?:\(\s*([A-Za-z])\s*\)|([A-Za-z]))\s*)\}\s*$", re.DOTALL)
STEP_BLOCK_RE = re.compile(r"<step\b[^>]*>.*?</step>", re.DOTALL)
STEP_TAG_RE = re.compile(r"</?step\b", re.DOTALL)
LITERAL_WHITESPACE_ESCAPE_RE = re.compile(r"\\[nrtvf]")
FORMAT_PRIMARY_CATEGORIES = (
    "full",
    "no_step",
    "text_outside_step",
    "step_xml_invalid",
    "step_schema_invalid",
    "boxed_missing",
    "boxed_invalid",
)


def _outside_text_is_effectively_empty(text: str) -> bool:
    """Treat real whitespace and literal whitespace escapes as empty outside step blocks."""
    return not LITERAL_WHITESPACE_ESCAPE_RE.sub("", text or "").strip()


def _boxed_answer_match(response_text: str):
    return BOXED_ANSWER_RE.search(response_text or "")


def _boxed_answer_letter(match: re.Match) -> str:
    return next(group for group in match.groups() if group).upper()


def boxed_answer_format_correct(response_text: str, valid_choices: Optional[str] = None) -> bool:
    """Return True when the response ends with a boxed single-letter answer."""
    match = _boxed_answer_match(response_text)
    if not match:
        return False
    answer = _boxed_answer_letter(match)
    if valid_choices is None:
        return True
    return answer in {choice.upper() for choice in valid_choices}


def _split_answer_region(response_text: str, valid_choices: Optional[str] = None) -> tuple[str, str, str]:
    match = _boxed_answer_match(response_text)
    if match:
        answer = _boxed_answer_letter(match)
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


def _relaxed_format_correct(step_region: str, step_matches: list[re.Match], boxed_status: str) -> bool:
    if not step_matches or boxed_status != "valid":
        return False

    cursor = 0
    for match in step_matches:
        if STEP_TAG_RE.search(step_region[cursor : match.start()]):
            return False
        cursor = match.end()
    if STEP_TAG_RE.search(step_region[cursor:]):
        return False

    return all(_step_block_status(match.group(0)) == "ok" for match in step_matches)


def classify_rollout_format(response_text: str, valid_choices: Optional[str] = None) -> Dict[str, object]:
    """Classify one complete rollout trajectory into exactly one primary format bucket."""
    response_text = response_text or ""
    step_region, boxed_status, boxed_answer = _split_answer_region(response_text, valid_choices)
    step_matches = list(STEP_BLOCK_RE.finditer(step_region))
    step_block_count = len(step_matches)
    relaxed_format_correct = _relaxed_format_correct(step_region, step_matches, boxed_status)

    if not step_matches:
        if STEP_TAG_RE.search(step_region):
            primary = "step_xml_invalid"
        else:
            primary = "no_step"
    else:
        cursor = 0
        text_outside_step = False
        for match in step_matches:
            if not _outside_text_is_effectively_empty(step_region[cursor : match.start()]):
                text_outside_step = True
                break
            cursor = match.end()
        if not text_outside_step and not _outside_text_is_effectively_empty(step_region[cursor:]):
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
        "relaxed_format_correct": relaxed_format_correct,
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
    metrics["rollout/format_primary/relax_correct_ratio"] = sum(bool(info.get("relaxed_format_correct", False)) for info in format_infos) / total
    return metrics


def aggregate_rollout_answer_acc_metrics(answer_acc_values: list[float], format_infos: list[Dict[str, object]]) -> Dict[str, float]:
    """Aggregate answer accuracy overall and on format-correct trajectories only."""
    if not answer_acc_values or len(answer_acc_values) != len(format_infos):
        return {}

    binary_acc = [1.0 if float(value) > 0.5 else 0.0 for value in answer_acc_values]
    total = float(len(binary_acc))
    all_correct_count = float(sum(binary_acc))

    format_correct_acc = [
        acc
        for acc, info in zip(binary_acc, format_infos)
        if info.get("format_primary") == "full"
    ]
    format_correct_total = float(len(format_correct_acc))

    return {
        "rollout/answer_acc/all_correct_ratio": all_correct_count / total,
        "rollout/answer_acc/format_correct_only_ratio": float(sum(format_correct_acc)) / format_correct_total if format_correct_total > 0 else 0.0,
    }


def rollout_format_infos_to_columns(format_infos: list[Dict[str, object]]) -> Dict[str, list]:
    """Convert per-rollout format info to JSONL-compatible columns."""
    return {
        "format_primary": [str(info.get("format_primary", "")) for info in format_infos],
        "boxed_status": [str(info.get("boxed_status", "")) for info in format_infos],
        "boxed_answer": [str(info.get("boxed_answer", "")) for info in format_infos],
        "step_block_count": [float(info.get("step_block_count", 0.0)) for info in format_infos],
        "format_error_advantage_mask": [0.0 if info.get("format_primary") == "full" else 1.0 for info in format_infos],
    }


def rollout_format_infos_to_metric_columns(format_infos: list[Dict[str, object]]) -> Dict[str, list[float]]:
    """Convert rollout format info to numeric columns suitable for aggregation and validation metrics."""
    columns: Dict[str, list[float]] = {
        f"format_primary_{category}": [] for category in FORMAT_PRIMARY_CATEGORIES
    }
    columns.update(
        {
            "boxed_status_valid": [],
            "boxed_status_invalid": [],
            "boxed_status_missing": [],
            "relaxed_format_correct": [],
            "step_block_count": [],
            "format_error_advantage_mask": [],
        }
    )

    for info in format_infos:
        primary = str(info.get("format_primary", ""))
        boxed_status = str(info.get("boxed_status", ""))
        for category in FORMAT_PRIMARY_CATEGORIES:
            columns[f"format_primary_{category}"].append(1.0 if primary == category else 0.0)
        for status in ("valid", "invalid", "missing"):
            columns[f"boxed_status_{status}"].append(1.0 if boxed_status == status else 0.0)
        columns["relaxed_format_correct"].append(1.0 if bool(info.get("relaxed_format_correct", False)) else 0.0)
        columns["step_block_count"].append(float(info.get("step_block_count", 0.0)))
        columns["format_error_advantage_mask"].append(0.0 if primary == "full" else 1.0)

    return columns


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
