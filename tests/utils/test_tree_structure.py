"""Unit tests for tree_structure.py (TreeRL EPTree implementation)."""

import math
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from verl.utils.tree_structure import (
    SearchTree,
    TreeManager,
    TreeNode,
    default_split_fn,
)


class TestTreeNode(unittest.TestCase):
    """Tests for TreeNode data structure."""

    def test_basic_creation(self):
        node = TreeNode(node_id=1, token_ids=[10, 20, 30], log_probs=[-0.5, -1.0, -0.3])
        self.assertEqual(node.node_id, 1)
        self.assertEqual(node.token_ids, [10, 20, 30])
        self.assertTrue(node.is_leaf)
        self.assertTrue(node.is_root)

    def test_parent_child_relationship(self):
        parent = TreeNode(node_id=1, token_ids=[10])
        child = TreeNode(node_id=2, token_ids=[20], parent=parent)
        parent.children.append(child)

        self.assertFalse(parent.is_leaf)
        self.assertTrue(parent.is_root)
        self.assertTrue(child.is_leaf)
        self.assertFalse(child.is_root)

    def test_entropy_scores(self):
        node = TreeNode(node_id=1, log_probs=[-0.5, -1.0, -0.3])
        self.assertEqual(node.entropy_scores, [0.5, 1.0, 0.3])
        self.assertAlmostEqual(node.max_entropy, 1.0)
        self.assertEqual(node.max_entropy_token_idx, 1)

    def test_descendant_leaves(self):
        root = TreeNode(node_id=1)
        child1 = TreeNode(node_id=2, parent=root)
        child2 = TreeNode(node_id=3, parent=root)
        grandchild = TreeNode(node_id=4, parent=child1)
        root.children = [child1, child2]
        child1.children = [grandchild]

        leaves = root.descendant_leaves()
        self.assertEqual(len(leaves), 2)
        leaf_ids = {l.node_id for l in leaves}
        self.assertEqual(leaf_ids, {3, 4})

    def test_path_from_root(self):
        root = TreeNode(node_id=1, token_ids=[10])
        child = TreeNode(node_id=2, token_ids=[20], parent=root)
        grandchild = TreeNode(node_id=3, token_ids=[30], parent=child)
        root.children = [child]
        child.children = [grandchild]

        path = grandchild.path_from_root()
        self.assertEqual([n.node_id for n in path], [1, 2, 3])

    def test_full_token_ids(self):
        root = TreeNode(node_id=1, token_ids=[10, 11])
        child = TreeNode(node_id=2, token_ids=[20, 21], parent=root)
        root.children = [child]

        ids = child.full_token_ids()
        self.assertEqual(ids, [10, 11, 20, 21])


class TestSearchTree(unittest.TestCase):

    def test_all_leaves(self):
        root = TreeNode(node_id=1)
        child1 = TreeNode(node_id=2, parent=root)
        child2 = TreeNode(node_id=3, parent=root)
        root.children = [child1, child2]

        tree = SearchTree(tree_idx=0, root=root, all_nodes=[root, child1, child2])
        self.assertEqual(tree.num_leaves, 2)

    def test_single_node_tree(self):
        root = TreeNode(node_id=1)
        tree = SearchTree(tree_idx=0, root=root, all_nodes=[root])
        self.assertEqual(tree.num_leaves, 1)


class TestBackpropagate(unittest.TestCase):
    """Test V(sn) = correct_leaves / total_leaves."""

    def _make_tree_manager(self):
        mgr = TreeManager.__new__(TreeManager)
        mgr.trees = []
        mgr._node_counter = 0
        return mgr

    def test_simple_binary_tree(self):
        """
        root
        ├── A (leaf, correct)
        └── B (leaf, incorrect)

        V(root) = 1/2 = 0.5
        V(A) = 1.0
        V(B) = 0.0
        """
        root = TreeNode(node_id=1)
        a = TreeNode(node_id=2, parent=root, correctness=1.0)
        b = TreeNode(node_id=3, parent=root, correctness=0.0)
        root.children = [a, b]

        tree = SearchTree(tree_idx=0, root=root, all_nodes=[root, a, b])

        mgr = self._make_tree_manager()
        mgr.trees = [tree]
        mgr.backpropagate()

        self.assertAlmostEqual(root.value, 0.5)
        self.assertAlmostEqual(a.value, 1.0)
        self.assertAlmostEqual(b.value, 0.0)

    def test_deeper_tree(self):
        """
        root
        ├── A
        │   ├── C (leaf, correct)
        │   └── D (leaf, incorrect)
        └── B (leaf, correct)

        V(A) = 1/2
        V(root) = 2/3
        """
        root = TreeNode(node_id=1)
        a = TreeNode(node_id=2, parent=root)
        b = TreeNode(node_id=3, parent=root, correctness=1.0)
        c = TreeNode(node_id=4, parent=a, correctness=1.0)
        d = TreeNode(node_id=5, parent=a, correctness=0.0)
        root.children = [a, b]
        a.children = [c, d]

        tree = SearchTree(tree_idx=0, root=root, all_nodes=[root, a, b, c, d])

        mgr = self._make_tree_manager()
        mgr.trees = [tree]
        mgr.backpropagate()

        self.assertAlmostEqual(root.value, 2 / 3, places=5)
        self.assertAlmostEqual(a.value, 0.5)
        self.assertAlmostEqual(b.value, 1.0)
        self.assertAlmostEqual(c.value, 1.0)
        self.assertAlmostEqual(d.value, 0.0)


class TestComputeStepRewards(unittest.TestCase):
    """Test R(sn) = (GA + LA) / sqrt(|L(sn)|)."""

    def _make_tree_manager(self):
        mgr = TreeManager.__new__(TreeManager)
        mgr.trees = []
        mgr._node_counter = 0
        return mgr

    def test_reward_formula(self):
        """
        root (V=0.5)
        ├── A (V=1.0, leaf, correct)
        └── B (V=0.0, leaf, incorrect)

        GA(A) = V(A) - V(root) = 1.0 - 0.5 = 0.5
        LA(A) = V(A) - V(root) = 1.0 - 0.5 = 0.5
        |L(A)| = 1
        R(A) = (0.5 + 0.5) / sqrt(1) = 1.0

        GA(B) = 0.0 - 0.5 = -0.5
        LA(B) = 0.0 - 0.5 = -0.5
        R(B) = (-0.5 + -0.5) / sqrt(1) = -1.0
        """
        root = TreeNode(node_id=1)
        a = TreeNode(node_id=2, parent=root, correctness=1.0)
        b = TreeNode(node_id=3, parent=root, correctness=0.0)
        root.children = [a, b]

        tree = SearchTree(tree_idx=0, root=root, all_nodes=[root, a, b])

        mgr = self._make_tree_manager()
        mgr.trees = [tree]
        mgr.backpropagate()
        mgr.compute_step_rewards()

        self.assertAlmostEqual(a.reward, 1.0, places=5)
        self.assertAlmostEqual(b.reward, -1.0, places=5)

    def test_reweight_factor(self):
        """
        root (V=2/3)
        ├── A (V=1/2)
        │   ├── C (leaf, correct, V=1)
        │   └── D (leaf, incorrect, V=0)
        └── B (leaf, correct, V=1)

        For root:
            GA(root) = V(root) - V(root) = 0
            LA(root) = V(root) - V(root) = 0  (root has no parent, use root)
            |L(root)| = 3
            R(root) = 0 / sqrt(3) = 0

        For A:
            GA(A) = 1/2 - 2/3 = -1/6
            LA(A) = 1/2 - 2/3 = -1/6
            |L(A)| = 2
            R(A) = (-1/6 + -1/6) / sqrt(2) = -1/3 / sqrt(2)
        """
        root = TreeNode(node_id=1)
        a = TreeNode(node_id=2, parent=root)
        b = TreeNode(node_id=3, parent=root, correctness=1.0)
        c = TreeNode(node_id=4, parent=a, correctness=1.0)
        d = TreeNode(node_id=5, parent=a, correctness=0.0)
        root.children = [a, b]
        a.children = [c, d]

        tree = SearchTree(tree_idx=0, root=root, all_nodes=[root, a, b, c, d])

        mgr = self._make_tree_manager()
        mgr.trees = [tree]
        mgr.backpropagate()
        mgr.compute_step_rewards()

        expected_root_reward = 0.0
        expected_a_reward = (-1 / 3) / math.sqrt(2)
        self.assertAlmostEqual(root.reward, expected_root_reward, places=5)
        self.assertAlmostEqual(a.reward, expected_a_reward, places=5)


class TestDefaultSplitFn(unittest.TestCase):

    def test_basic_split(self):
        text = "Step 1\n\nStep 2\n\nStep 3"
        result = default_split_fn(text)
        self.assertEqual(result, ["Step 1", "Step 2", "Step 3"])

    def test_empty_string(self):
        self.assertEqual(default_split_fn(""), [""])

    def test_no_separator(self):
        self.assertEqual(default_split_fn("single block"), ["single block"])


class TestSelectForkingPoints(unittest.TestCase):
    """Test entropy-based forking point selection."""

    def test_selects_highest_entropy(self):
        """Should select tokens with highest entropy (= -log_prob)."""
        root = TreeNode(
            node_id=1,
            token_ids=[10, 20, 30, 40, 50],
            log_probs=[-0.1, -2.0, -0.3, -1.5, -0.2],
            tree_idx=0,
            token_start=0,
            token_end=5,
        )
        tree = SearchTree(tree_idx=0, root=root, all_nodes=[root])

        mgr = TreeManager.__new__(TreeManager)
        mgr.trees = [tree]
        mgr.tree_top_n = 2
        mgr.mask_tail_ratio = 0.1

        points = mgr.select_forking_points(top_n=2)
        # Should select token_idx=1 (entropy=2.0) and token_idx=3 (entropy=1.5)
        selected_indices = {(n.node_id, t_idx) for _, n, t_idx in points}
        self.assertIn((1, 1), selected_indices)
        self.assertIn((1, 3), selected_indices)


class TestTreeGAEAdvantage(unittest.TestCase):
    """Test tree_gae advantage estimator."""

    def test_output_shape(self):
        from verl.trainer.ppo.core_algos import compute_tree_gae_advantage

        bs, seq_len = 4, 20
        index = np.array([0, 0, 1, 1])
        response_mask = torch.ones(bs, seq_len)
        token_level_rewards = torch.randn(bs, seq_len) * response_mask

        step_data = np.array([
            [(5, 0.5), (10, 0.8)],
            [(3, 0.2), (15, 1.0)],
            [(7, -0.3)],
            [(4, 0.6), (12, -0.5)],
        ], dtype=object)

        class FakeConfig:
            def get(self, key, default=None):
                return {"step_reward_weights": [1.0]}.get(key, default)

        adv, ret = compute_tree_gae_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            config=FakeConfig(),
            non_tensor_batch={"treerl_step_reward": step_data},
            batch=None,
        )
        self.assertEqual(adv.shape, (bs, seq_len))
        self.assertEqual(ret.shape, (bs, seq_len))

    def test_masked_positions_are_zero(self):
        from verl.trainer.ppo.core_algos import compute_tree_gae_advantage

        bs, seq_len = 4, 20
        index = np.array([0, 0, 1, 1])
        response_mask = torch.zeros(bs, seq_len)
        response_mask[:, :10] = 1.0  # Only first 10 tokens are valid

        token_level_rewards = torch.randn(bs, seq_len) * response_mask

        step_data = np.array([
            [(3, 0.5), (7, 0.8)],
            [(2, 0.2), (8, 1.0)],
            [(4, -0.3)],
            [(1, 0.6), (6, -0.5)],
        ], dtype=object)

        adv, ret = compute_tree_gae_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            non_tensor_batch={"treerl_step_reward": step_data},
            batch=None,
        )

        # Advantage should be zero where mask is zero
        masked_adv = adv[:, 10:]
        self.assertTrue(torch.allclose(masked_adv, torch.zeros_like(masked_adv), atol=1e-6))

    def test_assert_requires_treerl_step_reward(self):
        """tree_gae should assert if treerl_step_reward is missing."""
        from verl.trainer.ppo.core_algos import compute_tree_gae_advantage

        bs, seq_len = 4, 20
        index = np.array([0, 0, 1, 1])
        response_mask = torch.ones(bs, seq_len)
        token_level_rewards = torch.randn(bs, seq_len)

        with self.assertRaises(AssertionError):
            compute_tree_gae_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                non_tensor_batch={},  # no treerl_step_reward
                batch=None,
            )

    def test_with_external_prm(self):
        """Verify external PRM keys affect advantage output."""
        from verl.trainer.ppo.core_algos import compute_tree_gae_advantage

        bs, seq_len = 4, 20
        index = np.array([0, 0, 1, 1])
        response_mask = torch.ones(bs, seq_len)
        token_level_rewards = torch.randn(bs, seq_len) * response_mask

        step_data = np.array([
            [(5, 0.5), (10, 0.8)],
            [(3, 0.2), (15, 1.0)],
            [(7, -0.3)],
            [(4, 0.6), (12, -0.5)],
        ], dtype=object)

        # External PRM data: format_step_reward
        ext_data = np.array([
            [(5, 1.0), (10, 0.0)],
            [(3, 1.0), (15, 1.0)],
            [(7, 0.0)],
            [(4, 1.0), (12, 0.0)],
        ], dtype=object)

        class FakeConfig:
            def get(self, key, default=None):
                return {
                    "step_reward_weights": [1.0, 1.0],
                    "step_reward_type": "format",
                }.get(key, default)

        # Without external PRM
        adv_tree_only, _ = compute_tree_gae_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            config=FakeConfig(),
            non_tensor_batch={"treerl_step_reward": step_data},
            batch=None,
        )

        # With external PRM
        adv_with_ext, _ = compute_tree_gae_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            config=FakeConfig(),
            non_tensor_batch={
                "treerl_step_reward": step_data,
                "format_step_reward": ext_data,
            },
            batch=None,
        )

        # They should differ because external PRM adds a second dimension
        self.assertFalse(
            torch.allclose(adv_tree_only, adv_with_ext, atol=1e-6),
            "External PRM should change the advantage values",
        )

    def test_multi_reward_weights(self):
        """Verify asymmetric weights produce different results."""
        from verl.trainer.ppo.core_algos import compute_tree_gae_advantage

        bs, seq_len = 4, 20
        index = np.array([0, 0, 1, 1])
        response_mask = torch.ones(bs, seq_len)
        token_level_rewards = torch.randn(bs, seq_len) * response_mask

        step_data = np.array([
            [(5, 0.5), (10, 0.8)],
            [(3, 0.2), (15, 1.0)],
            [(7, -0.3)],
            [(4, 0.6), (12, -0.5)],
        ], dtype=object)

        ext_data = np.array([
            [(5, 1.0), (10, 0.0)],
            [(3, 1.0), (15, 1.0)],
            [(7, 0.0)],
            [(4, 1.0), (12, 0.0)],
        ], dtype=object)

        non_tensor = {
            "treerl_step_reward": step_data,
            "format_step_reward": ext_data,
        }

        class ConfigA:
            def get(self, key, default=None):
                return {
                    "step_reward_weights": [1.0, 0.5],
                    "step_reward_type": "format",
                }.get(key, default)

        class ConfigB:
            def get(self, key, default=None):
                return {
                    "step_reward_weights": [0.5, 1.0],
                    "step_reward_type": "format",
                }.get(key, default)

        adv_a, _ = compute_tree_gae_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            config=ConfigA(),
            non_tensor_batch=non_tensor,
            batch=None,
        )

        adv_b, _ = compute_tree_gae_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            config=ConfigB(),
            non_tensor_batch=non_tensor,
            batch=None,
        )

        # Different weights should produce different advantages
        self.assertFalse(
            torch.allclose(adv_a, adv_b, atol=1e-6),
            "Different weight configurations should produce different advantages",
        )


if __name__ == "__main__":
    unittest.main()
