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

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

sys.modules.setdefault("ray", MagicMock())

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.sampling.mcts_node import MCTSNode
from verl.trainer.ppo.sampling.mcts_prm import (
    FORMAT_PRIMARY_CATEGORIES,
    aggregate_rollout_format_metrics,
    boxed_answer_format_correct,
    classify_rollout_format,
    format_step_reward,
    strict_step_xml_correct,
)
from verl.trainer.ppo.sampling.step_treerl import StepTreeRLStrategy, _build_sampling_result, _pad_sequences
from verl.utils.ppo_batch import build_padded_prompt_response_batch
from verl.utils.reward_score.logi import compute_score as logi_compute_score
from verl.workers.rollout.sampling_params import extract_rollout_sampling_kwargs
from verl.workers.reward_manager.step_tree import StepTreeRewardManager


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def decode(self, tokens, skip_special_tokens=True):
        return "decoded"

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class TextTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(int(tok)) for tok in tokens if int(tok) != self.pad_token_id)

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def make_strategy(
    top_k=1,
    iter_rounds=1,
    branch_repeats=1,
    process_reward_type="format",
    selected_num_traces=2,
    adv_estimator="step_treerl_reinforce",
    trajectory_rm_enabled=None,
    training_reward_mode="segment",
):
    config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": adv_estimator,
            },
            "trainer": {
                "process_reward": {
                    "type": process_reward_type,
                    "fol": {
                        "metadata_path": None,
                        "llm": {
                            "model_name": None,
                        },
                    },
                },
                "step_treerl_config": {
                    "training_reward_mode": training_reward_mode,
                    "n": top_k,
                    "l": iter_rounds,
                    "t": branch_repeats,
                    "top_k": top_k,
                    "iter_rounds": iter_rounds,
                    "max_token_num": 8,
                    "branch_max_new_tokens": 6,
                    "path_selection": "selected_terminals",
                    "selected_num_traces": selected_num_traces,
                    "overall_norm_style": "none",
                    "use_weighted_value": False,
                },
                "n_gpus_per_node": 1,
                "nnodes": 1,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "max_model_len": 32,
                    "n": 1,
                }
            },
        }
    )
    if trajectory_rm_enabled is not None:
        config.trainer.step_treerl_config.trajectory_rm_enabled = trajectory_rm_enabled
    return StepTreeRLStrategy(config, DummyTokenizer())


class TestStepTreeRLStrategy(unittest.TestCase):
    def test_compute_step_entropies_use_parent_prefix(self):
        strategy = make_strategy()
        device = torch.device("cpu")

        root = MCTSNode(state=[10], tree_idx=0)
        child = MCTSNode(state=[10, 20], step_tokens=[20], parent=root, tree_idx=0)
        grandchild = MCTSNode(state=[10, 20, 30], step_tokens=[30], parent=child, tree_idx=0)
        root.children = [child]
        child.children = [grandchild]

        captured = {}

        def compute_log_prob_fn(data):
            captured["prompts"] = data.batch["prompts"].clone()
            captured["responses"] = data.batch["responses"].clone()
            captured["calculate_entropy"] = data.meta_info.get("calculate_entropy")
            return SimpleNamespace(
                batch={
                    "old_log_probs": torch.tensor([[-0.5], [-1.5]], dtype=torch.float32),
                }
            )

        entropies = strategy._compute_step_entropies([child, grandchild], compute_log_prob_fn, device)

        self.assertEqual(entropies, [0.5, 1.5])
        self.assertEqual(captured["prompts"].tolist(), [[0, 10], [10, 20]])
        self.assertEqual(captured["responses"].tolist(), [[20], [30]])
        self.assertIs(captured["calculate_entropy"], False)

    def test_compute_step_entropies_pads_single_node_to_gpu_count(self):
        strategy = make_strategy()
        strategy._n_gpus = 4
        device = torch.device("cpu")
        root = MCTSNode(state=[10], tree_idx=0)
        child = MCTSNode(state=[10, 20], step_tokens=[20], parent=root, tree_idx=0)

        captured = {}

        def compute_log_prob_fn(data):
            captured["batch_size"] = len(data)
            captured["responses"] = data.batch["responses"].clone()
            return SimpleNamespace(batch={"old_log_probs": torch.tensor([[-0.5], [-9.0], [-9.0], [-9.0]])})

        entropies = strategy._compute_step_entropies([child], compute_log_prob_fn, device)

        self.assertEqual(entropies, [0.5])
        self.assertEqual(captured["batch_size"], 4)
        self.assertEqual(captured["responses"].tolist(), [[20], [20], [20], [20]])

    def test_build_sampling_result_keeps_branch_prefix_in_response(self):
        device = torch.device("cpu")
        short_root = MCTSNode(state=[1, 2], tree_idx=0)
        prefix_step = MCTSNode(
            state=[1, 2, 10, 11],
            step_tokens=[10, 11],
            parent=short_root,
            tree_idx=0,
        )
        branch_step = MCTSNode(
            state=[1, 2, 10, 11, 12, 13],
            step_tokens=[12, 13],
            parent=prefix_step,
            tree_idx=0,
            generation_source="branch",
            branch_round=2,
        )
        long_root = MCTSNode(state=[3, 4, 5, 6], tree_idx=1)
        long_root_step = MCTSNode(
            state=[3, 4, 5, 6, 20],
            step_tokens=[20],
            parent=long_root,
            tree_idx=1,
        )

        result = _build_sampling_result(
            all_paths=[[prefix_step, branch_step], [long_root_step]],
            all_gt=[None, None],
            pad_token_id=0,
            device=device,
            batch_size=2,
        )
        batch = result.gen_batch_output.batch

        self.assertEqual(batch["prompts"].tolist(), [[0, 0, 1, 2], [3, 4, 5, 6]])
        self.assertEqual(batch["responses"].tolist(), [[10, 11, 12, 13], [20, 0, 0, 0]])
        self.assertTrue(
            torch.equal(
                batch["input_ids"],
                torch.cat((batch["prompts"], batch["responses"]), dim=-1),
            )
        )
        self.assertEqual(
            batch["attention_mask"][:, -batch["responses"].shape[-1] :].tolist(),
            [[1, 1, 1, 1], [1, 0, 0, 0]],
        )
        self.assertEqual(result.gen_batch_output.non_tensor_batch["trace_source"].tolist(), ["branch", "origin"])
        self.assertEqual(result.gen_batch_output.non_tensor_batch["branch_round"].tolist(), [2, 0])

        with self.assertRaisesRegex(ValueError, "direct child of the root"):
            _build_sampling_result(
                all_paths=[[branch_step]],
                all_gt=[None],
                pad_token_id=0,
                device=device,
                batch_size=1,
            )

    def test_score_new_candidates_caches_only_unscored_nodes(self):
        strategy = make_strategy(top_k=1, iter_rounds=2)
        device = torch.device("cpu")

        root = MCTSNode(state=[1], tree_idx=0)
        scored = MCTSNode(state=[1, 2], step_tokens=[2], parent=root, tree_idx=0, cached_entropy=0.3)
        pending_a = MCTSNode(state=[1, 3], step_tokens=[3], parent=root, tree_idx=0)
        pending_b = MCTSNode(state=[1, 4], step_tokens=[4], parent=root, tree_idx=0)
        root.children = [scored, pending_a, pending_b]

        def fake_compute_step_entropies(nodes, _compute_log_prob_fn, _device):
            self.assertEqual(nodes, [pending_a, pending_b])
            return [0.7, 0.9]

        with patch.object(strategy, "_compute_step_entropies", side_effect=fake_compute_step_entropies) as mocked:
            strategy._score_new_candidates([scored, pending_a, pending_b], compute_log_prob_fn=None, device=device)

        mocked.assert_called_once()
        self.assertEqual(scored.cached_entropy, 0.3)
        self.assertEqual(pending_a.cached_entropy, 0.7)
        self.assertEqual(pending_b.cached_entropy, 0.9)

    def test_leaf_outcome_reward_is_sparse_at_last_valid_token(self):
        device = torch.device("cpu")
        short_root = MCTSNode(state=[1, 2], tree_idx=0)
        short_leaf = MCTSNode(
            state=[1, 2, 10, 11],
            step_tokens=[10, 11],
            segment_reward=9.0,
            process_reward=0.25,
            leaf_outcome=0.1,
            parent=short_root,
            tree_idx=0,
        )
        long_root = MCTSNode(state=[3], tree_idx=1)
        long_step = MCTSNode(
            state=[3, 20, 21],
            step_tokens=[20, 21],
            segment_reward=-4.0,
            process_reward=1.0,
            parent=long_root,
            tree_idx=1,
        )
        long_leaf = MCTSNode(
            state=[3, 20, 21, 22],
            step_tokens=[22],
            segment_reward=7.0,
            process_reward=99.0,
            leaf_outcome=1.0,
            parent=long_step,
            tree_idx=1,
            node_type="answer",
        )
        short_root.children = [short_leaf]
        long_root.children = [long_step]
        long_step.children = [long_leaf]

        result = _build_sampling_result(
            all_paths=[[short_leaf], [long_step, long_leaf]],
            all_gt=[None, None],
            pad_token_id=0,
            device=device,
            batch_size=2,
            training_reward_mode="leaf_outcome",
        )

        rewards = result.gen_batch_output.batch["reward_fn_scores"]
        self.assertTrue(torch.allclose(rewards, torch.tensor([[0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])))
        process_rewards = result.gen_batch_output.batch["process_reward_scores"]
        self.assertTrue(torch.allclose(process_rewards, torch.tensor([[0.0, 0.25, 0.0], [0.0, 1.0, 0.0]])))
        metrics = result.gen_batch_output.meta_info["step_treerl_metrics"]
        self.assertAlmostEqual(metrics["selected_process_reward_sum_mean"], 0.625)
        self.assertAlmostEqual(metrics["selected_leaf_outcome_mean"], 0.55)
        self.assertEqual(metrics["selected_leaf_outcome_invalid_ratio"], 0.0)
        self.assertEqual(metrics["selected_leaf_outcome_wrong_ratio"], 0.5)
        self.assertEqual(metrics["selected_leaf_outcome_correct_ratio"], 0.5)

    def test_leaf_outcome_scores_use_standard_grpo_group_normalization(self):
        rewards = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        response_mask = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        prompt_uids = np.asarray(["prompt_a", "prompt_a", "prompt_b", "prompt_b"], dtype=object)

        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=response_mask,
            index=prompt_uids,
            norm_adv_by_std_in_grpo=True,
        )

        expected_scale = 1.0 / (2.0**0.5)
        self.assertTrue(torch.allclose(advantages[0, :2], torch.full((2,), -expected_scale), atol=2e-5))
        self.assertTrue(torch.allclose(advantages[1], torch.full((3,), expected_scale), atol=2e-5))
        self.assertAlmostEqual(advantages[2, 0].item(), -expected_scale, places=5)
        self.assertTrue(torch.allclose(advantages[3], torch.full((3,), expected_scale), atol=2e-5))
        self.assertEqual(advantages[0, 2].item(), 0.0)
        self.assertEqual(advantages[2, 1:].sum().item(), 0.0)
        self.assertTrue(torch.equal(returns, advantages))

    def test_segment_mode_keeps_dense_training_reward_separate_from_process_reward(self):
        root = MCTSNode(state=[1], tree_idx=0)
        step = MCTSNode(
            state=[1, 2, 3],
            step_tokens=[2, 3],
            segment_reward=0.5,
            process_reward=1.0,
            parent=root,
            tree_idx=0,
        )
        answer = MCTSNode(
            state=[1, 2, 3, 4],
            step_tokens=[4],
            segment_reward=-0.25,
            process_reward=99.0,
            parent=step,
            tree_idx=0,
            node_type="answer",
        )
        root.children = [step]
        step.children = [answer]

        result = _build_sampling_result(
            all_paths=[[step, answer]],
            all_gt=[None],
            pad_token_id=0,
            device=torch.device("cpu"),
            batch_size=1,
            training_reward_mode="segment",
        )

        self.assertTrue(
            torch.allclose(
                result.gen_batch_output.batch["reward_fn_scores"],
                torch.tensor([[0.5, 0.5, -0.25]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                result.gen_batch_output.batch["process_reward_scores"],
                torch.tensor([[0.0, 1.0, 0.0]]),
            )
        )

    def test_leaf_outcome_scores_unique_selected_steps_and_weights_duplicate_paths(self):
        strategy = make_strategy(adv_estimator="grpo", training_reward_mode="leaf_outcome")
        strategy.path_selection = "all_leaves"
        strategy._n_gpus = 2
        root = MCTSNode(state=[1], tree_idx=0)
        shared_step = MCTSNode(
            state=[1, 2],
            step_tokens=[2],
            step_text="<step><premise>a</premise><conclusion>b</conclusion></step>",
            parent=root,
            tree_idx=0,
            node_idx=1,
        )
        answer_a = MCTSNode(
            state=[1, 2, 3],
            step_tokens=[3],
            step_text=r"\boxed{A}",
            process_reward=99.0,
            parent=shared_step,
            tree_idx=0,
            node_idx=2,
            node_type="answer",
        )
        answer_b = MCTSNode(
            state=[1, 2, 4],
            step_tokens=[4],
            step_text=r"\boxed{B}",
            process_reward=88.0,
            parent=shared_step,
            tree_idx=0,
            node_idx=3,
            node_type="answer",
        )
        root.children = [shared_step]
        shared_step.children = [answer_a, answer_b]

        with patch.object(strategy, "_score_process_rewards", return_value=[0.75]) as score_mock:
            result = strategy._build_output(
                gen_batch=None,
                roots=[root],
                device=torch.device("cpu"),
                ground_truths=["A"],
            )

        requests = score_mock.call_args.args[0]
        self.assertEqual(len(requests), 1)
        process_sums = result.gen_batch_output.batch["process_reward_scores"].sum(-1)
        self.assertTrue(torch.allclose(process_sums, torch.tensor([0.75, 0.75])))
        metrics = result.gen_batch_output.meta_info["step_treerl_metrics"]
        self.assertEqual(metrics["selected_process_reward_sum_mean"], 0.75)

    def test_branch_by_entropy_selects_top_k_per_tree_with_reselection(self):
        strategy = make_strategy(top_k=1, iter_rounds=2)
        device = torch.device("cpu")

        root0 = MCTSNode(state=[1], tree_idx=0)
        node0_internal = MCTSNode(state=[1, 11], step_tokens=[11], parent=root0, tree_idx=0, node_idx=1, step_text="0_internal", cached_entropy=0.9)
        node0_leaf = MCTSNode(state=[1, 11, 12], step_tokens=[12], parent=node0_internal, tree_idx=0, node_idx=2, step_text="0_leaf", cached_entropy=0.1)
        root0.children = [node0_internal]
        node0_internal.children = [node0_leaf]

        root1 = MCTSNode(state=[2], tree_idx=1)
        node1_internal = MCTSNode(state=[2, 21], step_tokens=[21], parent=root1, tree_idx=1, node_idx=1, step_text="1_internal", cached_entropy=0.8)
        node1_leaf = MCTSNode(state=[2, 21, 22], step_tokens=[22], parent=node1_internal, tree_idx=1, node_idx=2, step_text="1_leaf", cached_entropy=0.2)
        root1.children = [node1_internal]
        node1_internal.children = [node1_leaf]

        candidate_pool = {
            0: [node0_internal, node0_leaf],
            1: [node1_internal, node1_leaf],
        }
        selected_rounds = []
        new_nodes = [MCTSNode(state=[1, 11, 13], step_tokens=[13], parent=node0_internal, tree_idx=0, node_idx=3, cached_entropy=0.4)]

        def fake_continue_from_steps(nodes, _generate_fn, _device, branch_round):
            self.assertEqual(branch_round, len(selected_rounds) + 1)
            selected_rounds.append([node.step_text for node in nodes])
            return [] if len(selected_rounds) == 1 else new_nodes

        with patch.object(strategy, "_score_new_candidates") as mocked_score, patch.object(
            strategy, "_continue_from_steps", side_effect=fake_continue_from_steps
        ):
            strategy._branch_by_entropy(
                roots=[root0, root1],
                candidate_pool=candidate_pool,
                generate_fn=None,
                compute_log_prob_fn=None,
                device=device,
            )

        self.assertEqual(selected_rounds, [["0_internal", "1_internal"], ["0_internal", "1_internal"]])
        mocked_score.assert_called_with(new_nodes, None, device, timing_name="branch_entropy_logprob")
        self.assertTrue(node0_internal.is_branch_point)
        self.assertTrue(node1_internal.is_branch_point)

    def test_collect_branch_candidates_uses_cached_pool(self):
        strategy = make_strategy()
        root = MCTSNode(state=[1], tree_idx=0)
        scored = MCTSNode(state=[1, 2], step_tokens=[2], parent=root, tree_idx=0, node_idx=1, cached_entropy=0.5)
        pending = MCTSNode(state=[1, 3], step_tokens=[3], parent=root, tree_idx=0, node_idx=2)
        root.children = [scored, pending]

        candidates = strategy._collect_branch_candidates([root], {0: [scored, pending]})

        self.assertEqual(candidates, {(0, 1): [scored]})

    def test_collect_branch_candidates_groups_by_initial_rollout_tree(self):
        strategy = make_strategy(top_k=1)
        root = MCTSNode(state=[1], tree_idx=0)
        first_a = MCTSNode(state=[1, 2], step_tokens=[2], parent=root, tree_idx=0, node_idx=1, step_text="a", cached_entropy=0.1)
        first_b = MCTSNode(state=[1, 3], step_tokens=[3], parent=root, tree_idx=0, node_idx=2, step_text="b", cached_entropy=0.2)
        a_child = MCTSNode(state=[1, 2, 4], step_tokens=[4], parent=first_a, tree_idx=0, node_idx=3, step_text="a2", cached_entropy=0.9)
        b_child = MCTSNode(state=[1, 3, 5], step_tokens=[5], parent=first_b, tree_idx=0, node_idx=4, step_text="b2", cached_entropy=0.8)
        root.children = [first_a, first_b]
        first_a.children = [a_child]
        first_b.children = [b_child]

        selected_rounds = []

        def fake_continue_from_steps(nodes, _generate_fn, _device, branch_round):
            self.assertEqual(branch_round, 1)
            selected_rounds.append([node.step_text for node in nodes])
            return []

        with patch.object(strategy, "_continue_from_steps", side_effect=fake_continue_from_steps), patch.object(strategy, "_score_new_candidates"):
            strategy._branch_by_entropy(
                roots=[root],
                candidate_pool={0: [first_a, first_b, a_child, b_child]},
                generate_fn=None,
                compute_log_prob_fn=None,
                device=torch.device("cpu"),
            )

        self.assertEqual(selected_rounds, [["a2", "b2"]])

    def test_continue_from_internal_node_preserves_existing_children(self):
        strategy = make_strategy()
        device = torch.device("cpu")

        root = MCTSNode(state=[1], tree_idx=0)
        parent = MCTSNode(state=[1, 2], step_tokens=[2], parent=root, tree_idx=0, step_text="parent")
        existing_child = MCTSNode(state=[1, 2, 3], step_tokens=[3], parent=parent, tree_idx=0, step_text="existing")
        root.children = [parent]
        parent.children = [existing_child]

        generated_blocks = [
            ("<step><premise>a</premise><conclusion>b</conclusion></step>", [11, 12]),
            ("<step><premise>c</premise><conclusion>d</conclusion></step>", [13, 14]),
        ]

        def generate_fn(_data):
            return SimpleNamespace(batch={"responses": torch.tensor([[7, 8, 0]], dtype=torch.long)})

        with patch.object(strategy, "_split_by_step_end", return_value=generated_blocks):
            created = strategy._continue_from_steps([parent], generate_fn, device)

        self.assertEqual(len(parent.children), 2)
        self.assertIs(parent.children[0], existing_child)
        new_branch = parent.children[1]
        self.assertEqual(new_branch.step_tokens, [11, 12])
        self.assertEqual(new_branch.generation_source, "branch")
        self.assertEqual(new_branch.branch_round, 1)
        self.assertEqual(len(new_branch.children), 1)
        self.assertEqual(new_branch.children[0].step_tokens, [13, 14])
        self.assertEqual(new_branch.children[0].generation_source, "branch")
        self.assertEqual(new_branch.children[0].branch_round, 1)
        self.assertEqual(created, [new_branch, new_branch.children[0]])

    def test_generation_prompts_are_left_padded(self):
        input_ids, attention_mask, position_ids = _pad_sequences(
            [torch.tensor([11, 12]), torch.tensor([21, 22, 23, 24])],
            pad_token_id=0,
            device=torch.device("cpu"),
        )

        self.assertEqual(input_ids.tolist(), [[0, 0, 11, 12], [21, 22, 23, 24]])
        self.assertEqual(attention_mask.tolist(), [[0, 0, 1, 1], [1, 1, 1, 1]])
        self.assertEqual(position_ids.tolist(), [[0, 0, 0, 1], [0, 1, 2, 3]])

    def test_continue_from_steps_uses_dynamic_branch_budget(self):
        strategy = make_strategy()
        strategy.max_model_len = 10
        strategy.max_token_num = 20
        device = torch.device("cpu")

        root = MCTSNode(state=[1], tree_idx=0)
        almost_full = MCTSNode(state=[1, 2, 3, 4, 5, 6, 7, 8, 9], step_tokens=[9], parent=root, tree_idx=0)
        roomy = MCTSNode(state=[1, 2, 3], step_tokens=[3], parent=root, tree_idx=0)

        captured = []

        def generate_fn(data):
            captured.append((data.meta_info["rollout_sampling_kwargs"], data.batch["input_ids"].size(0)))
            return SimpleNamespace(batch={"responses": torch.tensor([[7, 0]], dtype=torch.long)})

        with patch.object(strategy, "_split_by_step_end", return_value=[]):
            created = strategy._continue_from_steps([almost_full, roomy], generate_fn, device)

        self.assertEqual(
            captured,
            [
                ({"max_new_tokens": 1, "max_tokens": 1, "n": 1}, 1),
                ({"max_new_tokens": 6, "max_tokens": 6, "n": 1}, 1),
            ],
        )
        self.assertEqual(created, [])

    def test_continue_from_steps_repeats_branch_generation(self):
        strategy = make_strategy(branch_repeats=2)
        device = torch.device("cpu")
        root = MCTSNode(state=[1], tree_idx=0)
        step = MCTSNode(state=[1, 2], step_tokens=[2], parent=root, tree_idx=0, node_idx=1)
        root.children = [step]
        calls = []

        def generate_fn(data):
            calls.append((data.meta_info["rollout_sampling_kwargs"], data.batch["input_ids"].size(0)))
            return SimpleNamespace(batch={"responses": torch.tensor([[7, 0], [8, 0]], dtype=torch.long)})

        with patch.object(strategy, "_split_by_step_end", return_value=[]):
            created = strategy._continue_from_steps([step], generate_fn, device)

        self.assertEqual(calls, [({"max_new_tokens": 6, "max_tokens": 6, "n": 1}, 2)])
        self.assertEqual(created, [])

    def test_extract_rollout_sampling_kwargs_uses_namespaced_overrides_only(self):
        meta_info = {
            "n": 6,
            "max_tokens": 4096,
            "rollout_sampling_kwargs": {
                "n": 1,
                "max_tokens": 32,
                "max_new_tokens": 32,
                "top_p": 0.9,
                "unsupported": "ignored",
                "temperature": None,
            },
        }

        self.assertEqual(
            extract_rollout_sampling_kwargs(meta_info),
            {"n": 1, "max_tokens": 32, "max_new_tokens": 32, "top_p": 0.9},
        )
        self.assertEqual(extract_rollout_sampling_kwargs({"n": 6, "max_tokens": 4096}), {})

    def test_generate_full_solutions_defers_process_reward_scoring(self):
        strategy = make_strategy()

        root = MCTSNode(state=[1], tree_idx=0)
        gen_batch_output = SimpleNamespace(
            batch={"responses": torch.tensor([[5, 6, 0]], dtype=torch.long)},
            meta_info={"n_samples": 1},
        )

        with patch.object(strategy, "_split_by_step_end", return_value=[("step-a", [11, 12])]):
            created = strategy._generate_full_solutions(
                gen_batch=None,
                gen_batch_output=gen_batch_output,
                roots=[root],
                batch_size=1,
            )

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].R, 0.0)

        with patch.object(strategy, "_score_process_rewards", return_value=[0.75]) as mocked_score:
            strategy._assign_process_rewards(created)

        mocked_score.assert_called_once()
        self.assertEqual(created[0].process_reward, 0.75)
        self.assertEqual(created[0].R, 0.75)

    def test_generate_full_solutions_keeps_final_boxed_answer_as_terminal_leaf(self):
        tokenizer = TextTokenizer()
        strategy = make_strategy(selected_num_traces=1)
        strategy.tokenizer = tokenizer
        strategy.pad_token_id = tokenizer.pad_token_id
        strategy.eos_token_id = 100000
        strategy.length_penalty_enabled = False

        response = "<step><premise>a</premise><conclusion>b</conclusion></step>\n\\boxed{A}"
        root = MCTSNode(state=tokenizer.encode("prompt"), tree_idx=0)
        gen_batch_output = SimpleNamespace(
            batch={"responses": torch.tensor([tokenizer.encode(response)], dtype=torch.long)},
            meta_info={"n_samples": 1},
        )

        created = strategy._generate_full_solutions(
            gen_batch=None,
            gen_batch_output=gen_batch_output,
            roots=[root],
            batch_size=1,
        )

        self.assertEqual(len(created), 2)
        step_node, answer_node = created
        self.assertEqual(step_node.node_type, "step")
        self.assertEqual(answer_node.node_type, "answer")
        self.assertIs(answer_node.parent, step_node)
        self.assertTrue(answer_node.terminal)
        self.assertIn("\\boxed{A}", answer_node.accumulated_text)
        self.assertEqual(tokenizer.decode(answer_node.step_tokens), "\n\\boxed{A}")

        with patch.object(strategy, "_score_process_rewards", return_value=[1.0]) as mocked_score:
            strategy._assign_process_rewards(created)
        mocked_score.assert_called_once()
        self.assertEqual(step_node.process_reward, 1.0)
        self.assertEqual(answer_node.process_reward, 1.0)

        gen_batch = SimpleNamespace(non_tensor_batch={"answer": ["A"]})
        strategy._backpropagate_all([root], gen_batch)
        strategy._assign_segment_rewards(root)
        self.assertTrue(answer_node.is_correct)
        self.assertEqual(root.correct_terminal_in_subtree, 1)
        self.assertEqual(answer_node.segment_reward, 1.0)

        result = strategy._build_output(
            gen_batch=None,
            roots=[root],
            device=torch.device("cpu"),
            ground_truths=["A"],
        )
        decoded_response = tokenizer.decode(result.gen_batch_output.batch["responses"][0].tolist())
        self.assertIn("\\boxed{A}", decoded_response)

        metrics = result.gen_batch_output.meta_info["step_treerl_metrics"]
        self.assertEqual(metrics["total_steps"], 1)
        self.assertEqual(metrics["format_steps"], 1)
        self.assertEqual(metrics["trace_total"], 1)
        self.assertEqual(metrics["format_primary_full_count"], 1)
        self.assertEqual(metrics["step_num"], 1.0)

    def test_leaf_outcome_combines_full_format_and_answer_correctness(self):
        valid = "<step><premise>a</premise><conclusion>b</conclusion></step>\\boxed{A}"
        invalid = "unwrapped reasoning \\boxed{A}"
        invalid_schema = "<step><conclusion>b</conclusion></step>\\boxed{A}"
        invalid_choice = "<step><premise>a</premise><conclusion>b</conclusion></step>\\boxed{G}"
        cases = [
            ("valid_correct", valid, 1.0, 1.0, True),
            ("valid_wrong", valid, 0.0, 0.1, False),
            ("invalid_correct", invalid, 1.0, 0.0, False),
            ("invalid_wrong", invalid, 0.0, 0.0, False),
            ("invalid_step_schema", invalid_schema, 1.0, 0.0, False),
            ("invalid_boxed_choice", invalid_choice, 1.0, 0.0, False),
        ]

        for name, trajectory, answer_score, expected_outcome, expected_correct in cases:
            with self.subTest(name=name):
                strategy = make_strategy()
                root = MCTSNode(state=[1], tree_idx=0)
                leaf = MCTSNode(
                    state=[1, 2],
                    step_tokens=[2],
                    accumulated_text=trajectory,
                    parent=root,
                    tree_idx=0,
                )
                root.children = [leaf]
                gen_batch = SimpleNamespace(non_tensor_batch={"answer": ["A"]})

                with patch("verl.utils.reward_score.logi.compute_score", return_value=(answer_score, {})):
                    strategy._backpropagate_all([root], gen_batch)

                self.assertEqual(leaf.leaf_outcome, expected_outcome)
                self.assertEqual(leaf.R, expected_outcome)
                self.assertIs(leaf.is_correct, expected_correct)
                self.assertIs(leaf.main_chain, expected_correct)

    def test_trajectory_rm_does_not_override_format_aware_correctness(self):
        strategy = make_strategy()
        self.assertTrue(strategy.trajectory_rm_enabled)
        strategy.trajectory_rm_url = "http://localhost:4869/v1"
        root = MCTSNode(state=[1], tree_idx=0)
        leaf = MCTSNode(
            state=[1, 2],
            step_tokens=[2],
            accumulated_text="unwrapped reasoning \\boxed{A}",
            parent=root,
            tree_idx=0,
        )
        root.children = [leaf]
        gen_batch = SimpleNamespace(non_tensor_batch={"answer": ["A"]})

        def apply_rm(_roots, leaves, _gen_batch):
            leaves[0].R = 0.6
            leaves[0].accumulated_value = 0.6

        with (
            patch("verl.utils.reward_score.logi.compute_score", return_value=(1.0, {})),
            patch.object(strategy, "_evaluate_leaves_quality", side_effect=apply_rm),
        ):
            strategy._backpropagate_all([root], gen_batch)

        self.assertEqual(leaf.leaf_outcome, 0.0)
        self.assertEqual(leaf.R, 0.6)
        self.assertFalse(leaf.is_correct)
        self.assertFalse(leaf.main_chain)

    def test_disabled_trajectory_rm_skips_request_and_uses_zero_score(self):
        strategy = make_strategy(trajectory_rm_enabled=False)
        strategy.trajectory_rm_url = "http://localhost:4869/v1"
        root = MCTSNode(state=[1], tree_idx=0)
        leaf = MCTSNode(
            state=[1, 2],
            step_tokens=[2],
            accumulated_text="<step><premise>a</premise><conclusion>b</conclusion></step>\\boxed{A}",
            parent=root,
            tree_idx=0,
            llm_quality_score=1.0,
        )
        root.children = [leaf]
        gen_batch = SimpleNamespace(non_tensor_batch={"answer": ["A"]})

        with (
            patch("verl.utils.reward_score.logi.compute_score", return_value=(1.0, {})),
            patch.object(strategy, "_evaluate_leaves_quality") as evaluate_mock,
        ):
            strategy._backpropagate_all([root], gen_batch)

        evaluate_mock.assert_not_called()
        self.assertEqual(leaf.llm_quality_score, 0.0)
        self.assertEqual(leaf.leaf_outcome, 1.0)
        self.assertEqual(leaf.R, 1.0)

    def test_step_num_counts_only_step_nodes_in_padded_training_paths(self):
        strategy = make_strategy()
        strategy.path_selection = "all_leaves"
        strategy._n_gpus = 4
        root = MCTSNode(state=[1], tree_idx=0)

        short_step = MCTSNode(state=[1, 2], step_tokens=[2], parent=root, tree_idx=0, node_idx=1)
        short_answer = MCTSNode(state=[1, 2, 3], step_tokens=[3], parent=short_step, tree_idx=0, node_idx=2, node_type="answer")
        long_step_1 = MCTSNode(state=[1, 4], step_tokens=[4], parent=root, tree_idx=0, node_idx=3)
        long_step_2 = MCTSNode(state=[1, 4, 5], step_tokens=[5], parent=long_step_1, tree_idx=0, node_idx=4)
        long_answer = MCTSNode(state=[1, 4, 5, 6], step_tokens=[6], parent=long_step_2, tree_idx=0, node_idx=5, node_type="answer")
        root.children = [short_step, long_step_1]
        short_step.children = [short_answer]
        long_step_1.children = [long_step_2]
        long_step_2.children = [long_answer]

        result = strategy._build_output(
            gen_batch=None,
            roots=[root],
            device=torch.device("cpu"),
            ground_truths=[None],
        )

        metrics = result.gen_batch_output.meta_info["step_treerl_metrics"]
        self.assertEqual(result.gen_batch_output.batch["responses"].shape[0], 4)
        self.assertEqual(metrics["step_num"], 1.75)

    def test_terminal_step_nodes_remain_branch_candidates_but_answer_nodes_do_not(self):
        strategy = make_strategy()
        root = MCTSNode(state=[1], tree_idx=0)
        step = MCTSNode(state=[1, 2], step_tokens=[2], parent=root, tree_idx=0, node_idx=1, cached_entropy=0.5, terminal=True)
        answer = MCTSNode(
            state=[1, 2, 3],
            step_tokens=[3],
            parent=step,
            tree_idx=0,
            node_idx=2,
            cached_entropy=0.9,
            terminal=True,
            node_type="answer",
        )
        root.children = [step]
        step.children = [answer]

        candidate_pool = {0: []}
        strategy._add_candidates(candidate_pool, [step, answer])
        self.assertEqual(candidate_pool, {0: [step]})
        self.assertEqual(strategy._collect_branch_candidates([root], candidate_pool), {(0, 1): [step]})

    def test_score_new_candidates_scores_terminal_steps_but_not_answer_nodes(self):
        strategy = make_strategy()
        root = MCTSNode(state=[1], tree_idx=0)
        terminal_step = MCTSNode(state=[1, 2], step_tokens=[2], parent=root, tree_idx=0, node_idx=1, terminal=True)
        answer = MCTSNode(state=[1, 3], step_tokens=[3], parent=root, tree_idx=0, node_idx=2, terminal=True, node_type="answer")
        root.children = [terminal_step, answer]

        def fake_compute_step_entropies(nodes, _compute_log_prob_fn, _device):
            self.assertEqual(nodes, [terminal_step])
            return [0.7]

        with patch.object(strategy, "_compute_step_entropies", side_effect=fake_compute_step_entropies) as mocked:
            strategy._score_new_candidates([terminal_step, answer], compute_log_prob_fn=None, device=torch.device("cpu"))

        mocked.assert_called_once()
        self.assertEqual(terminal_step.cached_entropy, 0.7)
        self.assertIsNone(answer.cached_entropy)

    def test_continue_from_steps_uses_cached_fol_sample_id(self):
        strategy = make_strategy()
        strategy.process_reward_type = "fol"
        strategy.process_reward_runtime = SimpleNamespace()
        strategy._sample_ids_by_tree = {0: "sample-0"}
        strategy.step_prm_fn = MagicMock(return_value=0.8)
        device = torch.device("cpu")

        root = MCTSNode(state=[1], tree_idx=0)
        parent = MCTSNode(state=[1, 2], step_tokens=[2], parent=root, tree_idx=0, step_text="parent")
        root.children = [parent]

        def generate_fn(_data):
            return SimpleNamespace(batch={"responses": torch.tensor([[7, 8, 0]], dtype=torch.long)})

        with patch.object(
            strategy,
            "_split_by_step_end",
            return_value=[("<step><premise>a</premise><conclusion>b</conclusion></step>", [11, 12])],
        ):
            created = strategy._continue_from_steps([parent], generate_fn, device)
            strategy._assign_process_rewards(created)

        strategy.step_prm_fn.assert_called_once_with("<step><premise>a</premise><conclusion>b</conclusion></step>", sample_id="sample-0")
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].R, 0.8)

    def test_self_eval_assign_process_rewards_uses_actor_generation(self):
        tokenizer = TextTokenizer()
        strategy = make_strategy(process_reward_type="self_eval")
        strategy.tokenizer = tokenizer
        strategy.max_model_len = 4096
        strategy._question_texts_by_tree = {0: "Question text"}
        device = torch.device("cpu")

        root = MCTSNode(state=tokenizer.encode("prompt"), tree_idx=0)
        node = MCTSNode(
            state=tokenizer.encode("prompt<step>1</step>"),
            step_tokens=tokenizer.encode("<step>1</step>"),
            step_text="<step><premise>a</premise><conclusion>b</conclusion></step>",
            accumulated_text="<step><premise>previous</premise><conclusion>a</conclusion></step><step><premise>a</premise><conclusion>b</conclusion></step>",
            parent=root,
            tree_idx=0,
            node_idx=1,
        )
        captured = {}

        def generate_fn(data):
            captured["sampling_kwargs"] = data.meta_info["rollout_sampling_kwargs"]
            prompt_ids = data.batch["input_ids"][0][data.batch["attention_mask"][0].bool()].tolist()
            captured["prompt"] = tokenizer.decode(prompt_ids)
            return SimpleNamespace(
                batch={"responses": torch.tensor([tokenizer.encode(r"\boxed{1}")], dtype=torch.long)}
            )

        strategy._assign_process_rewards([node], generate_fn=generate_fn, device=device)

        self.assertEqual(node.process_reward, 1.0)
        self.assertEqual(node.R, 1.0)
        self.assertEqual(captured["sampling_kwargs"]["n"], 1)
        self.assertEqual(captured["sampling_kwargs"]["max_new_tokens"], 32)
        self.assertIn("Question text", captured["prompt"])
        self.assertIn(node.accumulated_text, captured["prompt"])

    def test_self_eval_scores_boxed_zero_and_unparseable_outputs(self):
        tokenizer = TextTokenizer()
        strategy = make_strategy(process_reward_type="self_eval")
        strategy.tokenizer = tokenizer
        strategy.max_model_len = 4096
        device = torch.device("cpu")

        requests = [
            SimpleNamespace(response=r"\boxed{0}", expected=0.0),
            SimpleNamespace(response="not boxed", expected=0.0),
        ]
        for case in requests:
            node = MCTSNode(
                state=tokenizer.encode("prompt"),
                step_tokens=tokenizer.encode("<step>1</step>"),
                step_text="<step><premise>a</premise><conclusion>b</conclusion></step>",
                accumulated_text=f"<step><premise>a</premise><conclusion>{case.response}</conclusion></step>",
                parent=MCTSNode(state=tokenizer.encode("prompt"), tree_idx=0),
                tree_idx=0,
                node_idx=1,
            )

            def generate_fn(_data, response=case.response):
                return SimpleNamespace(
                    batch={"responses": torch.tensor([tokenizer.encode(response)], dtype=torch.long)}
                )

            strategy._assign_process_rewards([node], generate_fn=generate_fn, device=device)
            self.assertEqual(node.process_reward, case.expected)

    def test_rloo_backprop_and_segment_reward(self):
        strategy = make_strategy()
        strategy.length_penalty_enabled = False
        root = MCTSNode(state=[1], tree_idx=0)
        valid_a = "<step><premise>a</premise><conclusion>b</conclusion></step>\\boxed{A}"
        valid_b = "<step><premise>c</premise><conclusion>d</conclusion></step>\\boxed{B}"
        a = MCTSNode(state=[1, 2], step_tokens=[2], step_text="a", accumulated_text=valid_a, parent=root, tree_idx=0, process_reward=1.0)
        b = MCTSNode(state=[1, 3], step_tokens=[3], step_text="b", accumulated_text=valid_b, parent=root, tree_idx=0, process_reward=1.0)
        root.children = [a, b]

        gen_batch = SimpleNamespace(non_tensor_batch={"answer": ["A"]})
        with patch("verl.utils.reward_score.logi.compute_score", side_effect=[(0.0, {}), (1.0, {})]):
            strategy._backpropagate_all([root], gen_batch)
        strategy._assign_segment_rewards(root)

        self.assertAlmostEqual(a.R, -0.9)
        self.assertAlmostEqual(b.R, 0.9)
        self.assertEqual(root.terminal_in_subtree, 2)
        self.assertEqual(root.correct_terminal_in_subtree, 1)
        self.assertAlmostEqual(root.accumulated_value, 0.0)
        self.assertAlmostEqual(a.segment_reward, -0.9)
        self.assertAlmostEqual(b.segment_reward, 0.9)

    def test_leaf_outcome_mode_stops_before_tree_backprop_and_trajectory_rm(self):
        strategy = make_strategy(
            adv_estimator="grpo",
            training_reward_mode="leaf_outcome",
            trajectory_rm_enabled=True,
        )
        strategy.trajectory_rm_url = "http://unused.invalid/v1"
        root = MCTSNode(state=[1], tree_idx=0)
        invalid = MCTSNode(
            state=[1, 2],
            step_tokens=[2],
            accumulated_text=r"plain reasoning \boxed{A}",
            parent=root,
            tree_idx=0,
        )
        valid_wrong = MCTSNode(
            state=[1, 3],
            step_tokens=[3],
            accumulated_text=r"<step><premise>a</premise><conclusion>b</conclusion></step>\boxed{B}",
            parent=root,
            tree_idx=0,
        )
        valid_correct = MCTSNode(
            state=[1, 4],
            step_tokens=[4],
            accumulated_text=r"<step><premise>a</premise><conclusion>b</conclusion></step>\boxed{A}",
            parent=root,
            tree_idx=0,
        )
        root.children = [invalid, valid_wrong, valid_correct]
        gen_batch = SimpleNamespace(non_tensor_batch={"answer": ["A"]})

        with (
            patch("verl.utils.reward_score.logi.compute_score", side_effect=[(1.0, {}), (0.0, {}), (1.0, {})]),
            patch.object(strategy, "_evaluate_leaves_quality") as evaluate_mock,
            patch.object(strategy, "_apply_rloo_to_leaves") as rloo_mock,
        ):
            strategy._backpropagate_all([root], gen_batch)

        self.assertEqual([invalid.leaf_outcome, valid_wrong.leaf_outcome, valid_correct.leaf_outcome], [0.0, 0.1, 1.0])
        self.assertEqual([invalid.R, valid_wrong.R, valid_correct.R], [0.0, 0.1, 1.0])
        self.assertEqual(root.terminal_in_subtree, 0)
        self.assertTrue(valid_correct.main_chain)
        evaluate_mock.assert_not_called()
        rloo_mock.assert_not_called()

    def test_origin_segment_reward_uses_value_formula_without_prm_or_length_penalty(self):
        strategy = make_strategy(adv_estimator="step_treerl_origin")
        strategy.length_penalty_enabled = True
        strategy.length_penalty_p_max = 10.0

        root = MCTSNode(state=[1], tree_idx=0, accumulated_value=4.0, terminal_in_subtree=4)
        step = MCTSNode(
            state=[1, 2],
            step_tokens=[2],
            step_text="step",
            accumulated_text="step",
            parent=root,
            tree_idx=0,
            process_reward=0.25,
            accumulated_value=3.0,
            terminal_in_subtree=2,
        )
        root.children = [step]

        strategy._assign_segment_rewards(root)

        # V(root)=1.0, V(step)=1.5, V(parent)=V(root)=1.0
        self.assertEqual(step.state_value, 1.5)
        self.assertEqual(step.segment_reward, 1.0)

    def test_origin_answer_leaf_uses_value_formula_instead_of_leaf_r(self):
        strategy = make_strategy(adv_estimator="step_treerl_origin")
        strategy.length_penalty_enabled = True

        root = MCTSNode(state=[1], tree_idx=0, accumulated_value=4.0, terminal_in_subtree=4)
        step = MCTSNode(
            state=[1, 2],
            step_tokens=[2],
            step_text="step",
            accumulated_text="step",
            parent=root,
            tree_idx=0,
            accumulated_value=3.0,
            terminal_in_subtree=2,
        )
        answer = MCTSNode(
            state=[1, 2, 3],
            step_tokens=[3],
            step_text="\\boxed{A}",
            accumulated_text="step\\boxed{A}",
            parent=step,
            tree_idx=0,
            node_type="answer",
            R=99.0,
            accumulated_value=2.0,
            terminal_in_subtree=1,
        )
        root.children = [step]
        step.children = [answer]

        strategy._assign_segment_rewards(root)

        # V(root)=1.0, V(step)=1.5, V(answer)=2.0
        self.assertEqual(answer.state_value, 2.0)
        self.assertEqual(answer.segment_reward, 1.5)

    def test_step_treerl_origin_advantage_returns_masked_dense_rewards(self):
        rewards = torch.tensor([[1.0, 1.0, -0.5, -0.5, 9.0]], dtype=torch.float32)
        response_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.float32)

        advantages, returns = core_algos.compute_step_treerl_origin_advantage(
            token_level_rewards=rewards,
            response_mask=response_mask,
        )

        expected = torch.tensor([[1.0, 1.0, -0.5, -0.5, 0.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(advantages, expected))
        self.assertTrue(torch.equal(returns, expected))

    def test_select_terminals_keeps_correct_leaf_and_pads_to_num_traces(self):
        strategy = make_strategy(selected_num_traces=4)
        leaves = [
            MCTSNode(state=[1, 2], step_tokens=[2], tree_idx=0, node_idx=1, main_chain=True),
            MCTSNode(state=[1, 3], step_tokens=[3], tree_idx=0, node_idx=2, main_chain=False),
        ]

        selected, padding = strategy._select_terminals(leaves, 4)

        self.assertEqual(len(selected), 4)
        self.assertEqual(padding, 2)
        self.assertTrue(any(leaf.main_chain for leaf in selected))

    def test_weighted_update_uses_selected_terminal_counts(self):
        strategy = make_strategy()
        strategy.weighted_value_style = "sqrt"
        root = MCTSNode(state=[1], tree_idx=0, accumulated_value=8.0, terminal_in_subtree=2)
        a = MCTSNode(state=[1, 2], step_tokens=[2], parent=root, tree_idx=0, node_idx=1, accumulated_value=6.0, terminal_in_subtree=1)
        b = MCTSNode(state=[1, 3], step_tokens=[3], parent=root, tree_idx=0, node_idx=2, accumulated_value=2.0, terminal_in_subtree=1)
        root.children = [a, b]

        for leaf in [a, a, b]:
            strategy._selected_backpropagate(leaf)
        strategy._compute_weighted_update(root)

        self.assertAlmostEqual(root.accumulated_value, 8.0 / (3 ** 0.5))
        self.assertAlmostEqual(a.accumulated_value, 6.0 / (2 ** 0.5))
        self.assertAlmostEqual(b.accumulated_value, 2.0)

    def test_length_penalty_is_negative_and_larger_late(self):
        strategy = make_strategy()
        early = strategy._length_penalty(step_index=1, max_step=10)
        late = strategy._length_penalty(step_index=10, max_step=10)

        self.assertLess(early, 0.0)
        self.assertLess(late, early)

    def test_tree_loss_matches_treerl_clipped_objective(self):
        old_log_prob = torch.tensor([[0.0, 0.0]])
        log_prob = torch.log(torch.tensor([[1.5, 0.5]]))
        advantages = torch.tensor([[2.0, -1.0]])
        mask = torch.tensor([[1.0, 1.0]])
        config = SimpleNamespace(clip_ratio=0.2)

        loss_fn = core_algos.get_policy_loss_fn("tree_loss")
        loss, metrics = loss_fn(old_log_prob, log_prob, advantages, mask, config=config)

        clipped = torch.tensor([[1.26, 0.8]])
        expected = -torch.minimum(torch.tensor([[1.5, 0.5]]) * advantages, clipped * advantages).mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertIn("actor/pg_clipfrac", metrics)

    def test_strict_trajectory_format_helpers(self):
        good_step = "<step><premise>a</premise><premise>b</premise><conclusion>c</conclusion></step>"
        self.assertTrue(strict_step_xml_correct(good_step))
        self.assertFalse(strict_step_xml_correct("<step><premise>a</premise><conclusion>b</conclusion><conclusion>c</conclusion></step>"))
        self.assertFalse(strict_step_xml_correct("<step>extra<premise>a</premise><conclusion>b</conclusion></step>"))
        self.assertFalse(strict_step_xml_correct("<step><premise>a</premise><foo>b</foo><conclusion>c</conclusion></step>"))
        self.assertEqual(format_step_reward(good_step), 1.0)
        self.assertEqual(format_step_reward("<step><premise>a</premise><conclusion>b</conclusion><conclusion>c</conclusion></step>"), 0.0)
        self.assertEqual(format_step_reward("<step>extra<premise>a</premise><conclusion>b</conclusion></step>"), 0.0)
        self.assertEqual(format_step_reward("<step><premise>a</premise><foo>b</foo><conclusion>c</conclusion></step>"), 0.0)

        self.assertTrue(boxed_answer_format_correct(r"\boxed{A}"))
        self.assertTrue(boxed_answer_format_correct(r"final \boxed{{B}}"))
        self.assertTrue(boxed_answer_format_correct(r"\boxed{(A)}"))
        self.assertTrue(boxed_answer_format_correct(r"final \boxed{ (B) }"))
        self.assertTrue(boxed_answer_format_correct(r"final \boxed{{(C)}}"))
        self.assertFalse(boxed_answer_format_correct(r"\boxed{AB}"))
        self.assertFalse(boxed_answer_format_correct(r"\boxed{(AB)}"))
        self.assertFalse(boxed_answer_format_correct(r"\boxed{A}}"))
        self.assertFalse(boxed_answer_format_correct(r"\boxed{{A}"))
        self.assertFalse(boxed_answer_format_correct(r"\boxed{(A}}"))
        self.assertFalse(boxed_answer_format_correct(r"\boxed{A)}"))
        self.assertFalse(boxed_answer_format_correct(r"\boxed{A} trailing"))
        self.assertFalse(boxed_answer_format_correct(r"\boxed{(A)}."))

        full = good_step + r"\boxed{A}"
        full_parenthesized = good_step + r"\boxed{(A)}"
        answer_only = r"Reasoning without XML. \boxed{B}"
        step_only = good_step
        bad = "Reasoning without the requested structure."
        self.assertEqual(logi_compute_score(r"reasoning \boxed{(A)}", "A"), (1.0, None))
        self.assertEqual(logi_compute_score(r"reasoning \boxed{{(b)}}", "B"), (1.0, None))
        self.assertEqual(logi_compute_score(r"reasoning \boxed{(A)}", "B"), (0.0, None))
        self.assertEqual(logi_compute_score(r"reasoning \boxed{(AB)}", "A"), (0.0, None))
        self.assertEqual(logi_compute_score(r"reasoning \boxed{A}}", "A"), (0.0, None))
        self.assertEqual(logi_compute_score(r"reasoning \boxed{A} trailing text", "a"), (1.0, None))
        self.assertEqual(logi_compute_score(r"first \boxed{A}, final \boxed{B}", "B"), (1.0, None))
        self.assertEqual(logi_compute_score(r"first \boxed{A}, final \boxed{B}", "A"), (0.0, None))
        self.assertEqual(logi_compute_score("no boxed answer", "A"), (0.0, None))

        self.assertEqual(classify_rollout_format(full)["format_primary"], "full")
        self.assertEqual(classify_rollout_format(full_parenthesized)["format_primary"], "full")
        self.assertEqual(classify_rollout_format(answer_only)["format_primary"], "no_step")
        self.assertEqual(classify_rollout_format("prefix\n" + full)["format_primary"], "text_outside_step")
        self.assertEqual(classify_rollout_format("<step><premise>a</premise>")["format_primary"], "step_xml_invalid")
        self.assertEqual(classify_rollout_format("<step><premise>a</premise></step>" + r"\boxed{A}")["format_primary"], "step_schema_invalid")
        self.assertEqual(classify_rollout_format(good_step)["format_primary"], "boxed_missing")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{AA}")["format_primary"], "boxed_invalid")

        rollout_metrics = aggregate_rollout_format_metrics(
            [
                classify_rollout_format(full),
                classify_rollout_format(answer_only),
            ]
        )
        self.assertEqual(rollout_metrics["rollout/format_primary/total"], 2.0)
        self.assertEqual(rollout_metrics["rollout/format_primary/full_ratio"], 0.5)
        self.assertEqual(rollout_metrics["rollout/format_primary/no_step_ratio"], 0.5)
        self.assertEqual(rollout_metrics["rollout/format_primary/relax_correct_ratio"], 0.5)
        ratio_sum = sum(rollout_metrics[f"rollout/format_primary/{category}_ratio"] for category in FORMAT_PRIMARY_CATEGORIES)
        self.assertEqual(ratio_sum, 1.0)

    def test_build_output_tracks_trace_format_metrics_before_gpu_padding(self):
        strategy = make_strategy()
        strategy.path_selection = "all_leaves"
        strategy._n_gpus = 2
        strategy.length_penalty_enabled = False
        device = torch.device("cpu")

        root = MCTSNode(state=[1], tree_idx=0, terminal_in_subtree=3)
        full_leaf = MCTSNode(
            state=[1, 2],
            step_tokens=[2],
            step_text="<step><premise>a</premise><conclusion>b</conclusion></step>",
            accumulated_text="<step><premise>a</premise><conclusion>b</conclusion></step>",
            trajectory_text="<step><premise>a</premise><conclusion>b</conclusion></step>\\boxed{A}",
            parent=root,
            process_reward=1.0,
            tree_idx=0,
            node_idx=1,
            terminal_in_subtree=1,
        )
        answer_leaf = MCTSNode(
            state=[1, 3],
            step_tokens=[3],
            step_text="plain",
            accumulated_text="plain",
            trajectory_text="plain \\boxed{B}",
            parent=root,
            process_reward=0.0,
            tree_idx=0,
            node_idx=2,
            terminal_in_subtree=1,
        )
        step_leaf = MCTSNode(
            state=[1, 4],
            step_tokens=[4],
            step_text="<step><premise>a</premise><conclusion>b</conclusion></step>",
            accumulated_text="<step><premise>a</premise><conclusion>b</conclusion></step>",
            trajectory_text="<step><premise>a</premise><conclusion>b</conclusion></step>",
            parent=root,
            process_reward=1.0,
            tree_idx=0,
            node_idx=3,
            terminal_in_subtree=1,
        )
        root.children = [full_leaf, answer_leaf, step_leaf]

        result = strategy._build_output(
            gen_batch=None,
            roots=[root],
            device=device,
            ground_truths=[None],
        )

        metrics = result.gen_batch_output.meta_info["step_treerl_metrics"]
        self.assertEqual(metrics["problem_count"], 1)
        self.assertAlmostEqual(metrics["steps_per_problem"], 3.0)
        self.assertEqual(metrics["trace_total"], 3)
        self.assertEqual(metrics["selected_traces"], 3)
        self.assertEqual(metrics["format_primary_full_count"], 1)
        self.assertEqual(metrics["format_primary_no_step_count"], 1)
        self.assertEqual(metrics["format_primary_boxed_missing_count"], 1)
        self.assertAlmostEqual(metrics["format_primary_full_ratio"], 1 / 3)
        self.assertAlmostEqual(metrics["selected_process_reward_sum_mean"], 0.75)
        self.assertEqual(result.gen_batch_output.batch["responses"].shape[0], 4)
        self.assertTrue(
            torch.allclose(
                result.gen_batch_output.batch["process_reward_scores"].sum(-1),
                torch.tensor([1.0, 0.0, 1.0, 1.0]),
            )
        )

    def test_step_tree_reward_manager_returns_validation_format_metrics(self):
        tokenizer = TextTokenizer()
        response = "<step><premise>a</premise><conclusion>b</conclusion></step>\\boxed{A}"
        prompt_ids = tokenizer.encode("prompt")
        response_ids = tokenizer.encode(response)
        attention_mask = torch.ones((1, len(prompt_ids) + len(response_ids)), dtype=torch.long)
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([prompt_ids], dtype=torch.long),
                "responses": torch.tensor([response_ids], dtype=torch.long),
                "attention_mask": attention_mask,
            },
            non_tensors={
                "answer": np.array(["A"], dtype=object),
                "data_source": np.array(["reclor"], dtype=object),
            },
        )
        manager = StepTreeRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=lambda **_: 1.0,
            process_reward_cfg={"type": "format"},
        )

        result = manager(data, return_dict=True)

        extra = result["reward_extra_info"]
        self.assertEqual(extra["acc"], [1.0])
        self.assertEqual(extra["prm_score"], [1.0])
        self.assertEqual(result["reward_tensor"].sum().item(), 1.0)
        self.assertEqual(extra["format_primary_full"], [1.0])
        self.assertEqual(extra["format_error_advantage_mask"], [0.0])
        self.assertEqual(extra["boxed_status_valid"], [1.0])
        self.assertEqual(extra["relaxed_format_correct"], [1.0])

    def test_step_tree_reward_manager_decodes_mixed_lengths_with_response_mask(self):
        tokenizer = TextTokenizer()
        schema_invalid = "<step> </step>\\boxed{C}"
        full = "<step><premise>a</premise><conclusion>b</conclusion></step>\\boxed{C}"
        padded = build_padded_prompt_response_batch(
            prompt_sequences=[
                torch.tensor(tokenizer.encode("short"), dtype=torch.long),
                torch.tensor(tokenizer.encode("a much longer prompt"), dtype=torch.long),
            ],
            response_sequences=[
                torch.tensor(tokenizer.encode(schema_invalid), dtype=torch.long),
                torch.tensor(tokenizer.encode(full), dtype=torch.long),
            ],
            pad_token_id=tokenizer.pad_token_id,
        )
        data = DataProto.from_dict(
            tensors={
                "prompts": padded.prompts,
                "responses": padded.responses,
                "input_ids": padded.input_ids,
                "attention_mask": padded.attention_mask,
                "position_ids": padded.position_ids,
                "response_mask": padded.response_mask,
                "reward_fn_scores": torch.zeros_like(padded.responses, dtype=torch.float32),
            },
            non_tensors={
                "answer": np.array(["C", "C"], dtype=object),
                "data_source": np.array(["reclor", "reclor"], dtype=object),
            },
        )
        manager = StepTreeRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=lambda **_: 1.0,
            process_reward_cfg={"type": "format"},
        )

        result = manager(data, return_dict=True)
        extra = result["reward_extra_info"]

        self.assertEqual(result["response"], [schema_invalid, full])
        self.assertEqual(extra["format_primary_step_schema_invalid"], [1.0, 0.0])
        self.assertEqual(extra["format_primary_full"], [0.0, 1.0])
        self.assertEqual(extra["boxed_status_valid"], [1.0, 1.0])
        self.assertEqual(extra["relaxed_format_correct"], [0.0, 1.0])

    def test_step_tree_reward_manager_rejects_misaligned_attention_mask(self):
        tokenizer = TextTokenizer()
        response = "<step><premise>a</premise><conclusion>b</conclusion></step>\\boxed{C}"
        prompt_ids = tokenizer.encode("prompt")
        response_ids = tokenizer.encode(response)
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([prompt_ids], dtype=torch.long),
                "responses": torch.tensor([response_ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(prompt_ids) + len(response_ids) - 1), dtype=torch.long),
            },
            non_tensors={
                "answer": np.array(["C"], dtype=object),
                "data_source": np.array(["reclor"], dtype=object),
            },
        )
        manager = StepTreeRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=lambda **_: 1.0,
            process_reward_cfg={"type": "format"},
        )

        with self.assertRaisesRegex(ValueError, "Invalid prompt/response batch layout"):
            manager(data, return_dict=True)

    def test_step_tree_reward_manager_marks_format_error_advantage_mask(self):
        tokenizer = TextTokenizer()
        response = "plain reasoning\n\\boxed{A}"
        prompt_ids = tokenizer.encode("prompt")
        response_ids = tokenizer.encode(response)
        attention_mask = torch.ones((1, len(prompt_ids) + len(response_ids)), dtype=torch.long)
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([prompt_ids], dtype=torch.long),
                "responses": torch.tensor([response_ids], dtype=torch.long),
                "attention_mask": attention_mask,
            },
            non_tensors={
                "answer": np.array(["A"], dtype=object),
                "data_source": np.array(["reclor"], dtype=object),
            },
        )
        manager = StepTreeRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=lambda **_: 1.0,
            process_reward_cfg={"type": "format"},
        )

        result = manager(data, return_dict=True)

        extra = result["reward_extra_info"]
        self.assertEqual(extra["format_primary_no_step"], [1.0])
        self.assertEqual(extra["format_error_advantage_mask"], [1.0])

    def test_step_tree_reward_manager_accepts_precomputed_self_eval_scores(self):
        tokenizer = TextTokenizer()
        prompt_ids = tokenizer.encode("prompt")
        response_ids = tokenizer.encode("<step><premise>a</premise><conclusion>b</conclusion></step>")
        attention_mask = torch.ones((1, len(prompt_ids) + len(response_ids)), dtype=torch.long)
        reward_fn_scores = torch.zeros((1, len(response_ids)), dtype=torch.float32)
        reward_fn_scores[0, -1] = 1.0
        process_reward_scores = torch.zeros((1, len(response_ids)), dtype=torch.float32)
        process_reward_scores[0, 0] = 0.25
        process_reward_scores[0, -1] = 0.5
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([prompt_ids], dtype=torch.long),
                "responses": torch.tensor([response_ids], dtype=torch.long),
                "attention_mask": attention_mask,
                "reward_fn_scores": reward_fn_scores,
                "process_reward_scores": process_reward_scores,
            },
            non_tensors={
                "answer": np.array(["A"], dtype=object),
                "data_source": np.array(["reclor"], dtype=object),
            },
        )
        manager = StepTreeRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=lambda **_: 0.0,
            process_reward_cfg={"type": "self_eval"},
        )

        result = manager(data, return_dict=True)

        self.assertTrue(torch.equal(result["reward_tensor"], reward_fn_scores))
        self.assertEqual(result["reward_extra_info"]["prm_score"], [0.75])


if __name__ == "__main__":
    unittest.main()
