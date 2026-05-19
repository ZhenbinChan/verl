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
    boxed_answer_format_correct,
    classify_trajectory_format,
    strict_step_xml_correct,
)
from verl.trainer.ppo.sampling.step_treerl import StepTreeRLStrategy
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
):
    config = OmegaConf.create(
        {
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

        def fake_continue_from_steps(nodes, _generate_fn, _device):
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

        def fake_continue_from_steps(nodes, _generate_fn, _device):
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
        self.assertEqual(len(new_branch.children), 1)
        self.assertEqual(new_branch.children[0].step_tokens, [13, 14])
        self.assertEqual(created, [new_branch, new_branch.children[0]])

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

    def test_rloo_backprop_and_segment_reward(self):
        strategy = make_strategy()
        strategy.length_penalty_enabled = False
        root = MCTSNode(state=[1], tree_idx=0)
        a = MCTSNode(state=[1, 2], step_tokens=[2], step_text="a", accumulated_text="wrong", parent=root, tree_idx=0, process_reward=1.0)
        b = MCTSNode(state=[1, 3], step_tokens=[3], step_text="b", accumulated_text="right", parent=root, tree_idx=0, process_reward=1.0)
        root.children = [a, b]

        gen_batch = SimpleNamespace(non_tensor_batch={"answer": ["A"]})
        with patch("verl.utils.reward_score.logi.compute_score", side_effect=[(0.0, {}), (1.0, {})]):
            strategy._backpropagate_all([root], gen_batch)
        strategy._assign_segment_rewards(root)

        self.assertEqual(a.R, -1.0)
        self.assertEqual(b.R, 1.0)
        self.assertEqual(root.terminal_in_subtree, 2)
        self.assertEqual(root.correct_terminal_in_subtree, 1)
        self.assertEqual(root.accumulated_value, 0.0)
        self.assertEqual(a.segment_reward, -1.0)
        self.assertEqual(b.segment_reward, 1.0)

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

        self.assertTrue(boxed_answer_format_correct(r"\boxed{A}"))
        self.assertTrue(boxed_answer_format_correct(r"final \boxed{{B}}"))
        self.assertFalse(boxed_answer_format_correct(r"\boxed{AB}"))
        self.assertFalse(boxed_answer_format_correct(r"\boxed{A} trailing"))

        full = good_step + r"\boxed{A}"
        answer_only = r"Reasoning without XML. \boxed{B}"
        step_only = good_step
        bad = "Reasoning without the requested structure."
        self.assertEqual(classify_trajectory_format(full)["format_full"], 1.0)
        self.assertEqual(classify_trajectory_format(answer_only)["format_answer_only"], 1.0)
        self.assertEqual(classify_trajectory_format(step_only)["format_step_only"], 1.0)
        self.assertEqual(classify_trajectory_format(bad)["format_incorrect"], 1.0)

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
        self.assertEqual(metrics["full_format_correct_count"], 1)
        self.assertEqual(metrics["answer_format_only_count"], 1)
        self.assertEqual(metrics["step_format_only_count"], 1)
        self.assertAlmostEqual(metrics["full_format_correct_ratio"], 1 / 3)
        self.assertEqual(result.gen_batch_output.batch["responses"].shape[0], 4)

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
        self.assertEqual(extra["format_full"], [1.0])
        self.assertEqual(extra["format_answer_only"], [0.0])
        self.assertEqual(extra["format_step_only"], [0.0])
        self.assertEqual(extra["format_trace_total"], [1.0])


if __name__ == "__main__":
    unittest.main()
