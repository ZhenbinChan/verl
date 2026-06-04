from __future__ import annotations

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.naive_plus import NaivePlusRewardManager


class CharOffsetTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(int(tok)) for tok in tokens if int(tok) != self.pad_token_id)


def make_data(responses: list[str]) -> DataProto:
    tokenizer = CharOffsetTokenizer()
    prompt = "Q"
    prompt_ids = tokenizer.encode(prompt)
    encoded_responses = [tokenizer.encode(response) for response in responses]
    max_response_len = max(len(response_ids) for response_ids in encoded_responses)

    response_rows = []
    attention_rows = []
    for response_ids in encoded_responses:
        padding = [tokenizer.pad_token_id] * (max_response_len - len(response_ids))
        response_rows.append(response_ids + padding)
        attention_rows.append([1] * len(prompt_ids) + [1] * len(response_ids) + [0] * len(padding))

    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([prompt_ids for _ in responses], dtype=torch.long),
            "responses": torch.tensor(response_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        },
        non_tensors={
            "reward_model": np.array([{"ground_truth": "Z"} for _ in responses], dtype=object),
            "data_source": np.array(["logiqa" for _ in responses], dtype=object),
            "extra_info": np.array([{} for _ in responses], dtype=object),
        },
    )


def test_naive_plus_default_keeps_original_reward_for_format_errors():
    invalid_format_correct_answer = "plain reasoning\n\\boxed{Z}"
    manager = NaivePlusRewardManager(
        tokenizer=CharOffsetTokenizer(),
        num_examine=0,
        compute_score=lambda solution_str, **_: 1.0 if "\\boxed{Z}" in solution_str else 0.0,
    )

    result = manager(make_data([invalid_format_correct_answer]), return_dict=True)

    assert result["reward_extra_info"]["answer_acc"] == [1.0]
    assert "format_full" not in result["reward_extra_info"]
    assert "format_error_advantage_mask" not in result["reward_extra_info"]
    assert result["outcome_reward"] == [1.0]
    assert result["reward_tensor"].sum(-1).tolist() == [1.0]


def test_naive_plus_penalizes_format_errors_when_enabled_without_recording_trainer_level_format_metrics():
    good_step = "<step><premise>a</premise><conclusion>b</conclusion></step>"
    valid_format_correct_answer = good_step + "\n\\boxed{Z}"
    valid_format_wrong_answer = good_step + "\n\\boxed{A}"
    invalid_format_correct_answer = "plain reasoning\n\\boxed{Z}"
    invalid_format_wrong_answer = "plain reasoning\n\\boxed{A}"
    manager = NaivePlusRewardManager(
        tokenizer=CharOffsetTokenizer(),
        num_examine=0,
        compute_score=lambda solution_str, **_: 1.0 if "\\boxed{Z}" in solution_str else 0.0,
        penalize_format_error=True,
    )

    result = manager(
        make_data(
            [
                valid_format_correct_answer,
                valid_format_wrong_answer,
                invalid_format_correct_answer,
                invalid_format_wrong_answer,
            ]
        ),
        return_dict=True,
    )
    reward_extra_info = result["reward_extra_info"]

    assert "format_full" not in reward_extra_info
    assert "format_primary" not in reward_extra_info
    assert reward_extra_info["answer_acc"] == [1.0, 0.0, 1.0, 0.0]
    assert "format_error_advantage_mask" not in reward_extra_info
    assert result["outcome_reward"] == [1.0, 0.0, -1.0, -1.0]
    assert result["reward_tensor"].sum(-1).tolist() == [1.0, 0.0, -1.0, -1.0]


def test_naive_plus_returns_step_tree_format_fields_when_enabled():
    good_step = "<step><premise>a</premise><conclusion>b</conclusion></step>"
    full = good_step + "\n\\boxed{Z}"
    answer_only = "plain reasoning\n\\boxed{Z}"
    step_only = good_step
    incorrect = "plain reasoning"
    manager = NaivePlusRewardManager(
        tokenizer=CharOffsetTokenizer(),
        num_examine=0,
        compute_score=lambda solution_str, **_: 1.0 if "\\boxed{Z}" in solution_str else 0.0,
        log_format_metrics=True,
    )

    result = manager(make_data([full, answer_only, step_only, incorrect]), return_dict=True)
    reward_extra_info = result["reward_extra_info"]

    assert reward_extra_info["format_full"] == [1.0, 0.0, 0.0, 0.0]
    assert reward_extra_info["format_answer_only"] == [0.0, 1.0, 0.0, 0.0]
    assert reward_extra_info["format_step_only"] == [0.0, 0.0, 1.0, 0.0]
    assert reward_extra_info["format_trace_total"] == [1.0, 1.0, 1.0, 1.0]
    assert "format_primary" not in reward_extra_info


def test_naive_plus_answer_acc_uses_explicit_answer_acc_before_penalty():
    manager = NaivePlusRewardManager(
        tokenizer=CharOffsetTokenizer(),
        num_examine=0,
        compute_score=lambda **_: {"score": 0.0, "answer_acc": 1.0},
        penalize_format_error=True,
    )

    result = manager(make_data(["plain reasoning\n\\boxed{Z}"]), return_dict=True)

    assert result["reward_extra_info"]["answer_acc"] == [1.0]
    assert "format_error_advantage_mask" not in result["reward_extra_info"]
    assert result["outcome_reward"] == [-1.0]


def test_naive_plus_answer_acc_ignores_non_binary_acc_field():
    manager = NaivePlusRewardManager(
        tokenizer=CharOffsetTokenizer(),
        num_examine=0,
        compute_score=lambda **_: {"score": 0.0, "acc": 0.8},
    )

    result = manager(make_data(["plain reasoning\n\\boxed{Z}"]), return_dict=True)

    assert result["reward_extra_info"]["answer_acc"] == [0.0]
    assert result["outcome_reward"] == [0.0]
