from __future__ import annotations

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, apply_format_error_advantage_mask, compute_advantage


def make_grpo_batch() -> DataProto:
    return DataProto.from_dict(
        tensors={
            "responses": torch.ones((4, 3), dtype=torch.long),
            "attention_mask": torch.ones((4, 5), dtype=torch.long),
            "response_mask": torch.ones((4, 3), dtype=torch.float32),
            "token_level_rewards": torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.5],
                    [0.0, 0.0, -0.5],
                ],
                dtype=torch.float32,
            ),
        },
        non_tensors={
            "uid": np.array(["prompt_a", "prompt_a", "prompt_b", "prompt_b"], dtype=object),
        },
    )


def test_apply_format_error_advantage_mask_zeroes_only_selected_advantages():
    batch = compute_advantage(
        make_grpo_batch(),
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=False,
    )
    original_advantages = batch.batch["advantages"].clone()
    original_returns = batch.batch["returns"].clone()
    original_rewards = batch.batch["token_level_rewards"].clone()

    metrics = apply_format_error_advantage_mask(batch, {"format_error_advantage_mask": [0.0, 1.0, 0.0, 0.0]})

    assert torch.equal(batch.batch["advantages"][1], torch.zeros(3))
    assert torch.equal(batch.batch["advantages"][0], original_advantages[0])
    assert torch.equal(batch.batch["advantages"][2], original_advantages[2])
    assert torch.equal(batch.batch["advantages"][3], original_advantages[3])
    assert torch.equal(batch.batch["returns"], original_returns)
    assert torch.equal(batch.batch["token_level_rewards"], original_rewards)
    assert metrics == {
        "algorithm/format_error_advantage_mask/count": 1.0,
        "algorithm/format_error_advantage_mask/ratio": 0.25,
    }


def test_apply_format_error_advantage_mask_supports_step_tree():
    batch = compute_advantage(
        make_grpo_batch(),
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=False,
    )
    original_advantages = batch.batch["advantages"].clone()

    metrics = apply_format_error_advantage_mask(
        batch,
        {"format_error_advantage_mask": [1.0, 0.0, 1.0, 0.0]},
        reward_manager="step_tree",
    )

    assert torch.equal(batch.batch["advantages"][0], torch.zeros(3))
    assert torch.equal(batch.batch["advantages"][1], original_advantages[1])
    assert torch.equal(batch.batch["advantages"][2], torch.zeros(3))
    assert torch.equal(batch.batch["advantages"][3], original_advantages[3])
    assert metrics == {
        "algorithm/format_error_advantage_mask/count": 2.0,
        "algorithm/format_error_advantage_mask/ratio": 0.5,
    }


def test_apply_format_error_advantage_mask_requires_reward_extra_info():
    batch = compute_advantage(
        make_grpo_batch(),
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=False,
    )

    with pytest.raises(ValueError, match="format_error_advantage_mask"):
        apply_format_error_advantage_mask(batch, {})


def test_apply_format_error_advantage_mask_only_supports_expected_reward_managers():
    batch = compute_advantage(
        make_grpo_batch(),
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=False,
    )

    with pytest.raises(ValueError, match="naive_format"):
        apply_format_error_advantage_mask(
            batch,
            {"format_error_advantage_mask": [0.0, 1.0, 0.0, 0.0]},
            reward_manager="naive_plus",
        )


def test_apply_format_error_advantage_mask_checks_batch_size():
    batch = compute_advantage(
        make_grpo_batch(),
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=False,
    )

    with pytest.raises(ValueError, match="does not match batch size"):
        apply_format_error_advantage_mask(batch, {"format_error_advantage_mask": [1.0]})
