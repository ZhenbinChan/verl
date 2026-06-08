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
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _build_step_treerl_sampling_metrics, _validation_metric_section


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
                self.assertEqual(metrics["rollout/format_primary/relax_correct_ratio"], 1.0)
                self.assertNotIn("rollout/trajectory_format_total", metrics)

    def test_validation_metric_section_uses_canonical_core_accuracy_and_reward(self):
        available_vars = {"acc": {}, "answer_acc": {}, "reward": {}, "verifiable_reward": {}, "outcome_reward": {}, "prm_score": {}, "format_full": {}}

        self.assertEqual(_validation_metric_section("acc", available_vars), "val-core")
        self.assertEqual(_validation_metric_section("reward", available_vars), "val-core")
        self.assertEqual(_validation_metric_section("verifiable_reward", available_vars), "val-core")
        self.assertEqual(_validation_metric_section("answer_acc", available_vars), "val-aux")
        self.assertEqual(_validation_metric_section("outcome_reward", available_vars), "val-aux")
        self.assertEqual(_validation_metric_section("prm_score", available_vars), "val-aux")
        self.assertEqual(_validation_metric_section("format_full", available_vars), "val-aux")

        self.assertEqual(_validation_metric_section("answer_acc", {"answer_acc": {}, "reward": {}}), "val-core")

    def test_step_treerl_sampling_metrics_use_tree_prefix_except_reward(self):
        metrics = _build_step_treerl_sampling_metrics(
            {
                "format_steps": 3,
                "total_steps": 6,
                "steps_per_problem": 1.5,
                "format_ratio": 0.5,
                "process_reward_mean": 0.25,
                "leaf_acc": 0.75,
                "candidate_leaves": 4,
                "selected_traces": 2,
                "terminal_padding": 1,
                "trace_total": 2,
                "full_format_correct_count": 1,
                "answer_format_only_count": 0,
                "step_format_only_count": 1,
                "full_format_correct_ratio": 0.5,
                "answer_format_only_ratio": 0.0,
                "step_format_only_ratio": 0.5,
            },
            {"branch_generation": 2.0},
        )

        self.assertEqual(metrics["Tree/format_steps"], 3)
        self.assertEqual(metrics["Tree/leaf_acc"], 0.75)
        self.assertEqual(metrics["Tree/time_branch_generation"], 2.0)
        self.assertEqual(metrics["reward/step_treerl_process_reward_mean"], 0.25)
        self.assertFalse(any(key.startswith("training/step_treerl") for key in metrics))

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

    def test_literal_whitespace_escapes_outside_steps_are_accepted(self):
        good_step = "<step><premise>a</premise><conclusion>b</conclusion></step>"
        literal_whitespace = (r"\n", r"\r", r"\t", r"\v", r"\f")

        for escape in literal_whitespace:
            cases = (
                escape + good_step + r"\boxed{A}",
                good_step + escape + good_step + r"\boxed{A}",
                good_step + escape + r"\boxed{A}",
                " \n" + escape + "\t" + good_step + r"\boxed{A}",
            )
            for response in cases:
                with self.subTest(escape=escape, response=response):
                    self.assertEqual(classify_rollout_format(response)["format_primary"], "full")

        self.assertEqual(
            classify_rollout_format(r"\n prefix " + good_step + r"\boxed{A}")["format_primary"],
            "text_outside_step",
        )
        self.assertEqual(
            classify_rollout_format(r"<step>\n<premise>a</premise><conclusion>b</conclusion></step>\boxed{A}")["format_primary"],
            "step_schema_invalid",
        )

    def test_relaxed_format_ignores_outside_text_but_validates_remaining_format(self):
        good_step = "<step><premise>a</premise><conclusion>b</conclusion></step>"
        cases = {
            "full": (good_step + r"\boxed{A}", True),
            "outside_text": ("prefix" + good_step + "suffix" + r"\boxed{A}", True),
            "no_step": (r"plain reasoning \boxed{A}", False),
            "unmatched_step_tag": ("<step>" + good_step + r"\boxed{A}", False),
            "schema_invalid": ("prefix<step><premise>a</premise></step>" + r"\boxed{A}", False),
            "boxed_missing": ("prefix" + good_step, False),
            "boxed_invalid": ("prefix" + good_step + r"\boxed{AA}", False),
        }

        for name, (response, expected) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(classify_rollout_format(response)["relaxed_format_correct"], expected)

    def test_relax_correct_ratio_is_derived_and_not_a_primary_category(self):
        good_step = "<step><premise>a</premise><conclusion>b</conclusion></step>"
        format_infos = [
            classify_rollout_format(good_step + r"\boxed{A}"),
            classify_rollout_format("prefix" + good_step + r"\boxed{A}"),
            classify_rollout_format("prefix<step><premise>a</premise></step>" + r"\boxed{A}"),
            classify_rollout_format(r"plain reasoning \boxed{A}"),
        ]

        metrics = aggregate_rollout_format_metrics(format_infos)

        self.assertEqual(metrics["rollout/format_primary/relax_correct_ratio"], 0.5)
        self.assertNotIn("relax_correct", FORMAT_PRIMARY_CATEGORIES)
        ratio_sum = sum(metrics[f"rollout/format_primary/{category}_ratio"] for category in FORMAT_PRIMARY_CATEGORIES)
        self.assertEqual(ratio_sum, 1.0)

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
                self.assertEqual(metrics["rollout/format_primary/relax_correct_ratio"], 0.5)
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
        self.assertEqual(set(columns), {"format_primary", "boxed_status", "boxed_answer", "step_block_count"})


if __name__ == "__main__":
    unittest.main()
