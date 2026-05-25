import unittest

from omegaconf import OmegaConf

from verl.trainer.ppo.sampling import _STRATEGY_REGISTRY, create_sampling_strategy
from verl.trainer.ppo.sampling.mcts_prm import (
    FORMAT_PRIMARY_CATEGORIES,
    aggregate_rollout_answer_acc_metrics,
    aggregate_rollout_format_metrics,
    classify_rollout_format,
    rollout_format_infos_to_columns,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids if token_id > 2)


def make_sampling_config(strategy_name):
    return OmegaConf.create(
        {
            "trainer": {
                "sampling_strategy": strategy_name,
                "branch_level": "step",
                "step_reward_type": "random",
                "tree_rounds": 0,
                "tree_top_k": 1,
                "entropy_chain_config": {
                    "N": 1,
                    "L": 0,
                    "T": 1,
                    "max_token_num": 32,
                    "evaluation_strategy": "token-entropy",
                    "enforce_uniform_per_prompt": True,
                },
                "parallel_mcts_config": {
                    "max_nodes": 1,
                    "max_depth": 1,
                    "max_children": 1,
                    "concurrent_num": 1,
                    "pass_k": 1,
                    "num_traces": 1,
                    "max_token_num": 32,
                },
                "step_treerl_config": {
                    "n": 1,
                    "top_k": 1,
                    "iter_rounds": 0,
                    "max_depth": 1,
                    "max_token_num": 32,
                },
                "ig_config": {
                    "top_k": 1,
                    "iter_rounds": 0,
                    "max_depth": 1,
                    "max_token_num": 32,
                },
                "process_reward": {
                    "type": "format",
                },
                "n_gpus_per_node": 1,
                "nnodes": 1,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "max_model_len": 128,
                },
            },
            "algorithm": {
                "gamma": 1.0,
            },
            "reward_model": {
                "reward_manager": "auto",
            },
        }
    )


class TestTrainerRolloutFormatMetrics(unittest.TestCase):
    def test_all_registered_sampling_strategies_construct(self):
        expected_classes = {
            "tree_search": "TreeSamplingStrategy",
            "treerl": "EntropyChainStrategy",
            "parallel_mcts": "ParallelMCTSStrategy",
            "step_treerl": "StepTreeRLStrategy",
            "information_gain": "InformationGainStrategy",
        }
        self.assertEqual(set(_STRATEGY_REGISTRY), set(expected_classes))
        self.assertIsNone(create_sampling_strategy(make_sampling_config(None), DummyTokenizer()))

        for strategy_name, class_name in expected_classes.items():
            with self.subTest(sampling_strategy=strategy_name):
                strategy = create_sampling_strategy(make_sampling_config(strategy_name), DummyTokenizer())
                self.assertEqual(type(strategy).__name__, class_name)

    def test_rollout_metrics_are_computed_on_post_sampling_outputs_for_each_strategy(self):
        good_step = "<step><premise>a</premise><conclusion>b</conclusion></step>"
        post_sampling_outputs = [
            good_step + r"\boxed{A}",
            "branch prefix\n" + good_step + r"\boxed{A}",
        ]

        for strategy_name in [None, *_STRATEGY_REGISTRY.keys()]:
            with self.subTest(sampling_strategy=strategy_name):
                format_infos = [classify_rollout_format(output) for output in post_sampling_outputs]
                metrics = aggregate_rollout_format_metrics(format_infos)

                self.assertEqual(metrics["rollout/format_primary/total"], 2.0)
                self.assertEqual(metrics["rollout/format_primary/full_ratio"], 0.5)
                self.assertEqual(metrics["rollout/format_primary/text_outside_step_ratio"], 0.5)
                self.assertNotIn("rollout/trajectory_format_total", metrics)

    def test_answer_acc_metrics_keep_only_two_ratios(self):
        good_step = "<step><premise>a</premise><conclusion>b</conclusion></step>"
        format_infos = [
            classify_rollout_format(good_step + r"\boxed{A}"),
            classify_rollout_format(good_step + r"\boxed{B}"),
            classify_rollout_format("plain reasoning " + r"\boxed{A}"),
            classify_rollout_format("plain reasoning " + r"\boxed{B}"),
        ]
        metrics = aggregate_rollout_answer_acc_metrics([1.0, 0.0, 1.0, 0.0], format_infos)

        self.assertEqual(metrics["rollout/answer_acc/all_correct_ratio"], 0.5)
        self.assertEqual(metrics["rollout/answer_acc/format_correct_only_ratio"], 0.5)
        self.assertEqual(set(metrics), {"rollout/answer_acc/all_correct_ratio", "rollout/answer_acc/format_correct_only_ratio"})

    def test_answer_acc_format_correct_only_ratio_is_zero_without_full_format(self):
        format_infos = [
            classify_rollout_format("plain reasoning " + r"\boxed{A}"),
            classify_rollout_format("plain reasoning " + r"\boxed{B}"),
        ]
        metrics = aggregate_rollout_answer_acc_metrics([1.0, 0.0], format_infos)

        self.assertEqual(metrics["rollout/answer_acc/all_correct_ratio"], 0.5)
        self.assertEqual(metrics["rollout/answer_acc/format_correct_only_ratio"], 0.0)

    def test_trainer_extracts_only_explicit_answer_acc(self):
        self.assertEqual(RayPPOTrainer._extract_rollout_answer_acc(None, {"answer_acc": [1.0, 0.0]}, 2), [1.0, 0.0])
        self.assertEqual(RayPPOTrainer._extract_rollout_answer_acc(None, {"acc": [1.0, 0.0]}, 2), [])
        self.assertEqual(RayPPOTrainer._extract_rollout_answer_acc(None, {}, 2), [])

    def test_primary_categories_are_mutually_exclusive(self):
        good_step = "<step><premise>a</premise><conclusion>b</conclusion></step>"
        cases = {
            "full": good_step + r"\boxed{A}",
            "no_step": r"plain reasoning \boxed{A}",
            "text_outside_step": "prefix\n" + good_step + r"\boxed{A}",
            "step_xml_invalid": "<step><premise>a</premise>",
            "step_schema_invalid": "<step><premise>a</premise></step>" + r"\boxed{A}",
            "boxed_missing": good_step,
            "boxed_invalid": good_step + r"\boxed{AA}",
        }

        for expected, response in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(classify_rollout_format(response)["format_primary"], expected)

    def test_boxed_answer_boundaries(self):
        good_step = "<step><premise>a</premise><conclusion>b</conclusion></step>"
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{A}")["format_primary"], "full")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{{A}}")["format_primary"], "full")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{A}}")["format_primary"], "boxed_invalid")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{{A}")["format_primary"], "boxed_invalid")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{AB}")["format_primary"], "boxed_invalid")

    def test_aggregate_metrics_are_trainer_level_and_reward_manager_independent(self):
        reward_managers = [
            "naive",
            "naive_plus",
            "naive_math220k",
            "naive_format",
            "tree",
            "step_tree",
            "mcts",
            "ig",
            "dapo",
            "batch",
            "prime",
            "entropy",
        ]
        format_infos = [
            classify_rollout_format("<step><premise>a</premise><conclusion>b</conclusion></step>" + r"\boxed{A}"),
            classify_rollout_format("plain reasoning"),
        ]

        for reward_manager in reward_managers:
            with self.subTest(reward_manager=reward_manager):
                metrics = aggregate_rollout_format_metrics(format_infos)
                self.assertEqual(metrics["rollout/format_primary/total"], 2.0)
                self.assertEqual(metrics["rollout/format_primary/full_ratio"], 0.5)
                self.assertEqual(metrics["rollout/format_primary/no_step_ratio"], 0.5)
                self.assertNotIn("rollout/trajectory_format_total", metrics)
                ratio_sum = sum(metrics[f"rollout/format_primary/{category}_ratio"] for category in FORMAT_PRIMARY_CATEGORIES)
                self.assertEqual(ratio_sum, 1.0)

    def test_jsonl_columns_include_only_detail_fields(self):
        columns = rollout_format_infos_to_columns(
            [classify_rollout_format("<step><premise>a</premise><conclusion>b</conclusion></step>" + r"\boxed{A}")]
        )

        self.assertEqual(columns["format_primary"], ["full"])
        self.assertEqual(columns["boxed_status"], ["valid"])
        self.assertEqual(columns["boxed_answer"], ["A"])
        self.assertEqual(columns["step_block_count"], [1.0])


if __name__ == "__main__":
    unittest.main()
