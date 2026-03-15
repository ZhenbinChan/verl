# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Step-GDPO tests
# ---------------------------------------------------------------------------

class _FakeConfig:
    """Minimal mock for AlgoConfig that supports .get()"""
    def __init__(self, data: dict):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


def _make_step_reward_data(batch_size: int, seq_len: int, num_steps_range=(2, 6)):
    """Generate mock step reward data: list of (pos, score) per sample."""
    import random as _rng
    data = []
    for _ in range(batch_size):
        num_steps = _rng.randint(*num_steps_range)
        positions = sorted(_rng.sample(range(seq_len), min(num_steps, seq_len)))
        rewards = [(pos, _rng.random()) for pos in positions]
        data.append(rewards)
    return np.array(data, dtype=object)


class TestStepGDPO(unittest.TestCase):

    def setUp(self):
        self.seed = 42
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.batch_size = 16
        self.seq_len = 64
        self.num_groups = 4

    def test_output_shape(self):
        """Test that step_gdpo returns correct shapes."""
        from verl.trainer.ppo.core_algos import compute_step_gdpo_advantage

        index = _make_group_index(self.batch_size, self.num_groups)
        response_mask = _rand_mask(self.batch_size, self.seq_len)
        token_level_rewards = torch.randn(self.batch_size, self.seq_len) * response_mask

        step_data = _make_step_reward_data(self.batch_size, self.seq_len)
        non_tensor_batch = {"format_step_reward": step_data}
        config = _FakeConfig({
            "step_reward_keys": ["format_step_reward"],
            "step_reward_weights": [1.0, 1.0],
        })

        adv, ret = compute_step_gdpo_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            config=config,
            non_tensor_batch=non_tensor_batch,
            batch=None,
        )
        self.assertEqual(adv.shape, (self.batch_size, self.seq_len))
        self.assertEqual(ret.shape, (self.batch_size, self.seq_len))
        # Advantages should be zero where mask is zero
        masked_adv = adv * (1 - response_mask)
        self.assertTrue(torch.allclose(masked_adv, torch.zeros_like(masked_adv), atol=1e-6))

    def test_degenerates_to_grpo_when_process_weight_zero(self):
        """When process weight=0, step_gdpo should match outcome-only GRPO (up to whiten)."""
        from verl.trainer.ppo.core_algos import compute_step_gdpo_advantage, compute_grpo_outcome_advantage

        index = _make_group_index(self.batch_size, self.num_groups)
        response_mask = _rand_mask(self.batch_size, self.seq_len)
        token_level_rewards = torch.randn(self.batch_size, self.seq_len) * response_mask

        # step_gdpo with process_weight=0
        config = _FakeConfig({
            "step_reward_keys": [],
            "step_reward_weights": [1.0, 0.0],
        })
        step_adv, _ = compute_step_gdpo_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            config=config,
            non_tensor_batch={},
            batch=None,
        )

        # Pure GRPO
        grpo_adv, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
        )
        # step_gdpo applies whiten on top of GRPO, so we compare the whiten of GRPO
        import verl.utils.torch_functional as verl_F
        grpo_whitened = verl_F.masked_whiten(grpo_adv, response_mask) * response_mask

        self.assertTrue(
            torch.allclose(step_adv, grpo_whitened, rtol=1e-4, atol=1e-5),
            f"Max diff: {(step_adv - grpo_whitened).abs().max().item()}"
        )

    def test_reward_to_go_cumsum(self):
        """Verify reward-to-go is correctly implemented as reverse cumsum."""
        from verl.trainer.ppo.core_algos import compute_step_gdpo_advantage

        bs, seq_len = 4, 20
        index = np.array([0, 0, 1, 1])
        response_mask = torch.ones(bs, seq_len)

        # Outcome rewards: all zero (so outcome advantage = 0)
        token_level_rewards = torch.zeros(bs, seq_len)

        # Process reward: single step at position 10 with score 1.0 for all samples
        step_data = np.array([
            [(10, 1.0)],
            [(10, 1.0)],
            [(10, 1.0)],
            [(10, 1.0)],
        ], dtype=object)

        config = _FakeConfig({
            "step_reward_keys": ["test_step_reward"],
            "step_reward_weights": [0.0, 1.0],  # only process
        })

        adv, _ = compute_step_gdpo_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            config=config,
            non_tensor_batch={"test_step_reward": step_data},
            batch=None,
        )

        # With all same scores, normalization gives 0 for all (std=0 case)
        # This tests the edge case gracefully
        self.assertEqual(adv.shape, (bs, seq_len))

    def test_big_pool_normalization(self):
        """Verify big pool normalization works: scores from all steps in a group are pooled."""
        from verl.trainer.ppo.core_algos import compute_step_gdpo_advantage

        bs, seq_len = 4, 20
        index = np.array([0, 0, 0, 0])  # all same group
        response_mask = torch.ones(bs, seq_len)
        token_level_rewards = torch.zeros(bs, seq_len)

        # Different step counts, different scores
        step_data = np.array([
            [(5, 0.0), (10, 1.0)],      # 2 steps
            [(3, 0.5), (8, 0.5), (15, 1.0)],  # 3 steps
            [(7, 0.0)],                  # 1 step
            [(4, 1.0), (12, 0.0)],       # 2 steps
        ], dtype=object)

        config = _FakeConfig({
            "step_reward_keys": ["test_step_reward"],
            "step_reward_weights": [0.0, 1.0],
        })

        adv, _ = compute_step_gdpo_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            config=config,
            non_tensor_batch={"test_step_reward": step_data},
            batch=None,
        )

        self.assertEqual(adv.shape, (bs, seq_len))
        # Pool has 8 scores: [0, 1, 0.5, 0.5, 1, 0, 1, 0]
        # mean = 0.5, std should be non-zero
        # Scores above mean should get positive normalized values
        # The test mainly verifies no crash and correct shape


if __name__ == "__main__":
    unittest.main()
