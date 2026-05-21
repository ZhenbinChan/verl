from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_logiqa_format_qwen25 import (
    aggregate_metrics,
    evaluate_response,
    extract_final_boxed_answer,
    trajectory_format_correct,
)


VALID_RESPONSE = """
<step>
<premise>A implies B</premise>
<conclusion>B</conclusion>
</step>

<step>
<premise>B implies C</premise>
<conclusion>C</conclusion>
</step>

\\boxed{A}
"""


def test_extract_final_boxed_answer_accepts_single_and_double_braces():
    assert extract_final_boxed_answer("reasoning\n\\boxed{A}") == "A"
    assert extract_final_boxed_answer("reasoning\n\\boxed{{b}}") == "B"


def test_extract_final_boxed_answer_requires_single_letter_at_end():
    assert extract_final_boxed_answer("\\boxed{AB}") is None
    assert extract_final_boxed_answer("\\boxed{{A}") is None
    assert extract_final_boxed_answer("\\boxed{A}}") is None
    assert extract_final_boxed_answer("\\boxed{A}\ntrailing text") is None
    assert extract_final_boxed_answer("answer is A") is None


def test_trajectory_format_correct_requires_steps_and_final_boxed_letter():
    assert trajectory_format_correct(VALID_RESPONSE)

    assert not trajectory_format_correct(VALID_RESPONSE.replace("\\boxed{A}", "The answer is A."))
    assert not trajectory_format_correct("preface\n" + VALID_RESPONSE)
    assert not trajectory_format_correct(VALID_RESPONSE + "\nextra")
    assert not trajectory_format_correct(VALID_RESPONSE.replace("<premise>A implies B</premise>", "plain text"))
    assert not trajectory_format_correct("<step><premise>A</premise><conclusion>A</conclusion></step>\n\\boxed{AA}")


def test_evaluate_response_counts_step_format_separately_from_trajectory_format():
    response = """
<step>
<premise>A</premise>
<conclusion>A</conclusion>
</step>

extra explanation

\\boxed{C}
"""

    metrics = evaluate_response(response, "C")

    assert metrics["total_steps"] == 1
    assert metrics["format_correct_steps"] == 1
    assert metrics["step_format_ratio"] == 1.0
    assert metrics["trajectory_format_correct"] is False
    assert metrics["boxed_answer"] == "C"
    assert metrics["answer_correct"] is True


def test_aggregate_metrics_summarizes_model_results():
    records = [
        {
            "total_steps": 2,
            "format_correct_steps": 2,
            "trajectory_format_correct": True,
            "answer_correct": True,
        },
        {
            "total_steps": 1,
            "format_correct_steps": 0,
            "trajectory_format_correct": False,
            "answer_correct": False,
        },
    ]

    metrics = aggregate_metrics(records, model_path="/tmp/Qwen2.5-1.5B-Instruct")

    assert metrics["model_name"] == "Qwen2.5-1.5B-Instruct"
    assert metrics["total_trajectories"] == 2
    assert metrics["total_steps"] == 3
    assert metrics["step_format_correct_steps"] == 2
    assert metrics["step_format_ratio"] == 2 / 3
    assert metrics["trajectory_format_correct"] == 1
    assert metrics["trajectory_format_ratio"] == 0.5
    assert metrics["answer_correct"] == 1
    assert metrics["answer_accuracy"] == 0.5
