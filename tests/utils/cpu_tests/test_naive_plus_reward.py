from __future__ import annotations

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.sampling.mcts_prm import aggregate_trajectory_format_metrics
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


def test_naive_plus_records_trajectory_format_metrics_independent_of_answer_correctness():
    valid_format_wrong_answer = "<step><premise>a</premise><conclusion>b</conclusion></step>\n\\boxed{A}"
    answer_only = "plain reasoning\n\\boxed{B}"
    manager = NaivePlusRewardManager(
        tokenizer=CharOffsetTokenizer(),
        num_examine=0,
        compute_score=lambda **_: 0.0,
    )

    result = manager(make_data([valid_format_wrong_answer, answer_only]), return_dict=True)
    reward_extra_info = result["reward_extra_info"]

    assert reward_extra_info["format_full"] == [1.0, 0.0]
    assert reward_extra_info["format_answer_only"] == [0.0, 1.0]
    assert reward_extra_info["format_trace_total"] == [1.0, 1.0]
    assert result["outcome_reward"] == [0.0, 0.0]

    metrics = aggregate_trajectory_format_metrics(reward_extra_info)
    assert metrics["rollout/trajectory_format_correct_count"] == 1.0
    assert metrics["rollout/trajectory_format_total"] == 2.0
    assert metrics["rollout/trajectory_format_correct_ratio"] == 0.5
