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

import torch
from omegaconf import OmegaConf

sys.modules.setdefault("ray", MagicMock())

from verl.trainer.ppo.sampling.mcts_node import MCTSNode
from verl.trainer.ppo.sampling.step_treerl import StepTreeRLStrategy


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def decode(self, tokens, skip_special_tokens=True):
        return "decoded"

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def make_strategy(top_k=1, iter_rounds=1, process_reward_type="format"):
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
                    "top_k": top_k,
                    "iter_rounds": iter_rounds,
                    "max_token_num": 8,
                    "branch_max_new_tokens": 6,
                },
                "n_gpus_per_node": 1,
                "nnodes": 1,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "max_model_len": 32,
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
            return SimpleNamespace(
                batch={
                    "old_log_probs": torch.tensor([[-0.5], [-1.5]], dtype=torch.float32),
                }
            )

        entropies = strategy._compute_step_entropies([child, grandchild], compute_log_prob_fn, device)

        self.assertEqual(entropies, [0.5, 1.5])
        self.assertEqual(captured["prompts"].tolist(), [[0, 10], [10, 20]])
        self.assertEqual(captured["responses"].tolist(), [[20], [30]])

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
        node0_internal = MCTSNode(state=[1, 11], step_tokens=[11], parent=root0, tree_idx=0, step_text="0_internal", cached_entropy=0.9)
        node0_leaf = MCTSNode(state=[1, 11, 12], step_tokens=[12], parent=node0_internal, tree_idx=0, step_text="0_leaf", cached_entropy=0.1)
        root0.children = [node0_internal]
        node0_internal.children = [node0_leaf]

        root1 = MCTSNode(state=[2], tree_idx=1)
        node1_internal = MCTSNode(state=[2, 21], step_tokens=[21], parent=root1, tree_idx=1, step_text="1_internal", cached_entropy=0.8)
        node1_leaf = MCTSNode(state=[2, 21, 22], step_tokens=[22], parent=node1_internal, tree_idx=1, step_text="1_leaf", cached_entropy=0.2)
        root1.children = [node1_internal]
        node1_internal.children = [node1_leaf]

        candidate_pool = {
            0: [node0_internal, node0_leaf],
            1: [node1_internal, node1_leaf],
        }
        selected_rounds = []
        new_nodes = [MCTSNode(state=[1, 11, 13], step_tokens=[13], parent=node0_internal, tree_idx=0, cached_entropy=0.4)]

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
        mocked_score.assert_called_with(new_nodes, None, device)
        self.assertTrue(node0_internal.is_branch_point)
        self.assertTrue(node1_internal.is_branch_point)

    def test_collect_branch_candidates_uses_cached_pool(self):
        strategy = make_strategy()
        root = MCTSNode(state=[1], tree_idx=0)
        scored = MCTSNode(state=[1, 2], step_tokens=[2], parent=root, tree_idx=0, cached_entropy=0.5)
        pending = MCTSNode(state=[1, 3], step_tokens=[3], parent=root, tree_idx=0)
        root.children = [scored, pending]

        candidates = strategy._collect_branch_candidates([root], {0: [scored, pending]})

        self.assertEqual(candidates, {0: [scored]})

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
        device = torch.device("cpu")

        root = MCTSNode(state=[1], tree_idx=0)
        almost_full = MCTSNode(state=[1, 2, 3, 4, 5, 6, 7, 8, 9], step_tokens=[9], parent=root, tree_idx=0)
        roomy = MCTSNode(state=[1, 2, 3], step_tokens=[3], parent=root, tree_idx=0)

        captured = {}

        def generate_fn(data):
            captured["max_new_tokens"] = data.meta_info["max_new_tokens"]
            captured["batch_size"] = data.batch["input_ids"].size(0)
            return SimpleNamespace(batch={"responses": torch.tensor([[7, 0], [8, 0]], dtype=torch.long)})

        with patch.object(strategy, "_split_by_step_end", return_value=[]):
            created = strategy._continue_from_steps([almost_full, roomy], generate_fn, device)

        self.assertEqual(captured["max_new_tokens"], 1)
        self.assertEqual(captured["batch_size"], 2)
        self.assertEqual(created, [])

    def test_generate_full_solutions_uses_shared_step_reward(self):
        strategy = make_strategy()

        root = MCTSNode(state=[1], tree_idx=0)
        gen_batch_output = SimpleNamespace(
            batch={"responses": torch.tensor([[5, 6, 0]], dtype=torch.long)},
            meta_info={"n_samples": 1},
        )

        with patch.object(strategy, "_split_by_step_end", return_value=[("step-a", [11, 12])]), patch.object(
            strategy, "_score_step_reward", return_value=0.75
        ) as mocked_score:
            created = strategy._generate_full_solutions(
                gen_batch=None,
                gen_batch_output=gen_batch_output,
                roots=[root],
                batch_size=1,
            )

        mocked_score.assert_called_once_with("step-a", 0)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].R, 0.75)

    def test_continue_from_steps_uses_cached_fol_sample_id(self):
        strategy = make_strategy()
        strategy.process_reward_type = "fol"
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

        strategy.step_prm_fn.assert_called_once_with(
            "<step><premise>a</premise><conclusion>b</conclusion></step>",
            sample_id="sample-0",
        )
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].R, 0.8)


if __name__ == "__main__":
    unittest.main()
