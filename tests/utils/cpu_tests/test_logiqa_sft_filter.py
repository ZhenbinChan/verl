from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.logiqa_sft_filter_utils import filter_generated_dataset, rows_to_json_records, select_extreme_samples  # noqa: E402


def make_row(idx: int, correct_count: int) -> dict:
    return {
        "data_source": "logiqa",
        "prompt": [{"role": "user", "content": f"question {idx}"}],
        "ability": "logic",
        "reward_model": {"ground_truth": "A", "style": "rule"},
        "answer": "A",
        "raw_prompt": f"question {idx}",
        "sample_id": f"logiqa_{idx}",
        "extra_info": {"index": idx},
        "correct_count": correct_count,
        "total_rollouts": 64,
    }


def simple_logi_score(response: str, ground_truth: str):
    return (1.0 if rf"\boxed{{{ground_truth}}}" in response else 0.0), None


def test_select_extreme_samples_prefers_high_and_low_counts_without_overlap():
    rows = [make_row(idx, correct_count) for idx, correct_count in enumerate([64, 0, 63, 1, 62, 2])]

    correct_rows, error_rows = select_extreme_samples(rows, correct_size=2, error_size=2)

    assert [row["sample_id"] for row in correct_rows] == ["logiqa_0", "logiqa_2"]
    assert [row["sample_id"] for row in error_rows] == ["logiqa_1", "logiqa_3"]
    assert {row["sample_id"] for row in correct_rows}.isdisjoint({row["sample_id"] for row in error_rows})


def test_rows_to_json_records_drops_responses_and_keeps_selection_metadata():
    row = make_row(0, 64)
    row["responses"] = ["hidden"]

    records = rows_to_json_records([row])

    assert "responses" not in records[0]
    assert records[0]["correct_count"] == 64
    assert records[0]["total_rollouts"] == 64
    assert records[0]["sample_id"] == "logiqa_0"


def test_filter_generated_dataset_removes_selected_rows_from_remaining_train():
    generated = pd.DataFrame(
        [
            {**make_row(0, 0), "responses": [r"wrong \boxed{B}"]},
            {**make_row(1, 1), "responses": [r"right \boxed{A}"]},
            {**make_row(2, 0), "responses": [r"wrong \boxed{C}"]},
            {**make_row(3, 1), "responses": [r"right \boxed{A}"]},
        ]
    )

    correct_rows, error_rows, remaining = filter_generated_dataset(generated, correct_size=1, error_size=1, compute_score_fn=simple_logi_score)

    assert [row["sample_id"] for row in correct_rows] == ["logiqa_1"]
    assert [row["sample_id"] for row in error_rows] == ["logiqa_0"]
    assert set(remaining["sample_id"]) == {"logiqa_2", "logiqa_3"}
    assert "responses" not in remaining.columns
