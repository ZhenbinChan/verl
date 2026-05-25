from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

ORIGINAL_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "answer", "raw_prompt", "sample_id", "extra_info"]
ScoreFn = Callable[[str, str], tuple[float, Any]]


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_messages(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError(f"Expected prompt to be a list of chat messages, got {type(value)}.")
    return [dict(message) for message in value]


def inject_prompt_instruction(messages: list[dict[str, Any]], prompt_instruction: str | None) -> list[dict[str, Any]]:
    messages = [dict(message) for message in messages]
    if not prompt_instruction:
        return messages
    for message in reversed(messages):
        if message.get("role") == "user":
            message["content"] = f"{prompt_instruction.rstrip()}\n\n{str(message.get('content', '')).lstrip()}"
            return messages
    raise ValueError("Could not find a user message for prompt instruction injection.")


def _default_logi_score(response: str, ground_truth: str) -> tuple[float, Any]:
    from verl.utils.reward_score.logi import compute_score as compute_logi_score

    return compute_logi_score(response, ground_truth)


def score_responses(responses: list[str], ground_truth: str, compute_score_fn: ScoreFn | None = None) -> int:
    score_fn = compute_score_fn or _default_logi_score
    correct_count = 0
    for response in responses:
        score, _ = score_fn(response or "", str(ground_truth))
        correct_count += int(float(score) > 0.0)
    return correct_count


def select_extreme_samples(scored_rows: list[dict[str, Any]], correct_size: int, error_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed_rows = [{**row, "_selection_index": idx} for idx, row in enumerate(scored_rows)]
    correct_rows = sorted(indexed_rows, key=lambda row: (-int(row["correct_count"]), int(row["_selection_index"])))[:correct_size]
    correct_ids = {row["sample_id"] for row in correct_rows}
    error_candidates = [row for row in indexed_rows if row["sample_id"] not in correct_ids]
    error_rows = sorted(error_candidates, key=lambda row: (int(row["correct_count"]), int(row["_selection_index"])))[:error_size]

    if len(correct_rows) < correct_size or len(error_rows) < error_size:
        distribution: dict[int, int] = {}
        for row in scored_rows:
            count = int(row["correct_count"])
            distribution[count] = distribution.get(count, 0) + 1
        raise RuntimeError(
            f"Could not select enough samples: correct={len(correct_rows)}/{correct_size}, error={len(error_rows)}/{error_size}, "
            f"correct_count_distribution={dict(sorted(distribution.items()))}"
        )

    return [_strip_selection_index(row) for row in correct_rows], [_strip_selection_index(row) for row in error_rows]


def _strip_selection_index(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.pop("_selection_index", None)
    return row


def rows_to_json_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        record = {column: to_jsonable(row.get(column)) for column in ORIGINAL_COLUMNS if column in row}
        record["correct_count"] = int(row["correct_count"])
        record["total_rollouts"] = int(row["total_rollouts"])
        records.append(record)
    return records


def filter_generated_dataset(
    generated: pd.DataFrame,
    correct_size: int,
    error_size: int,
    compute_score_fn: ScoreFn | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    scored_rows = []
    for row in generated.to_dict("records"):
        responses = to_jsonable(row.get("responses"))
        if not isinstance(responses, list):
            raise TypeError(f"Expected responses to be a list for sample_id={row.get('sample_id')}, got {type(responses)}.")
        reward_model = to_jsonable(row.get("reward_model"))
        ground_truth = row.get("answer")
        if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
            ground_truth = reward_model["ground_truth"]
        scored_rows.append(
            {
                **row,
                "correct_count": score_responses([str(response) for response in responses], str(ground_truth), compute_score_fn=compute_score_fn),
                "total_rollouts": len(responses),
            }
        )

    correct_rows, error_rows = select_extreme_samples(scored_rows, correct_size=correct_size, error_size=error_size)
    selected_ids = {row["sample_id"] for row in correct_rows + error_rows}
    remaining = generated.loc[~generated["sample_id"].isin(selected_ids), [column for column in generated.columns if column != "responses"]].copy()
    return correct_rows, error_rows, remaining
