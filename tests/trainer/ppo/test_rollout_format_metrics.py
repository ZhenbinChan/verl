import unittest
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.sampling import _STRATEGY_REGISTRY, create_sampling_strategy
from verl.trainer.ppo.sampling.mcts_prm import (
    FORMAT_PRIMARY_CATEGORIES,
    aggregate_rollout_answer_acc_metrics,
    aggregate_rollout_format_metrics,
    classify_rollout_format,
    rollout_format_infos_to_columns,
    rollout_format_infos_to_metric_columns,
)
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _build_step_treerl_sampling_metrics,
    _should_aggregate_validation_reward,
    _validation_metric_section,
)


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
        available_vars = {
            "acc": {},
            "answer_acc": {},
            "reward": {},
            "verifiable_reward": {},
            "outcome_reward": {},
            "prm_score": {},
            "format_primary_full": {},
        }

        self.assertEqual(_validation_metric_section("acc", available_vars), "val-core")
        self.assertEqual(_validation_metric_section("reward", available_vars), "val-core")
        self.assertEqual(_validation_metric_section("verifiable_reward", available_vars), "val-core")
        self.assertEqual(_validation_metric_section("answer_acc", available_vars), "val-aux")
        self.assertEqual(_validation_metric_section("outcome_reward", available_vars), "val-aux")
        self.assertEqual(_validation_metric_section("prm_score", available_vars), "val-aux")
        self.assertEqual(_validation_metric_section("format_primary_full", available_vars), "val-aux")

        self.assertEqual(_validation_metric_section("answer_acc", {"answer_acc": {}, "reward": {}}), "val-core")

    def test_step_treerl_validation_aggregates_acc_but_not_reward(self):
        self.assertFalse(_should_aggregate_validation_reward("step_treerl"))
        self.assertTrue(_should_aggregate_validation_reward("none"))

        grouped = process_validation_metrics(
            data_sources=np.asarray(["logiqa", "logiqa", "logiqa"], dtype=object),
            sample_inputs=["question-1", "question-2", "question-3"],
            infos_dict={"acc": [1.0, 0.0, 1.0]},
        )

        self.assertAlmostEqual(grouped["logiqa"]["acc"]["mean@1"], 2 / 3)
        self.assertNotIn("reward", grouped["logiqa"])

    def test_step_treerl_sampling_metrics_use_tree_prefix_except_reward(self):
        metrics = _build_step_treerl_sampling_metrics(
            {
                "format_steps": 3,
                "total_steps": 6,
                "steps_per_problem": 1.5,
                "format_ratio": 0.5,
                "selected_process_reward_sum_mean": 1.25,
                "leaf_acc": 0.75,
                "candidate_leaves": 4,
                "selected_traces": 2,
                "step_num": 2.5,
                "terminal_padding": 1,
                "trace_total": 2,
                "selected_leaf_outcome_mean": 0.55,
                "selected_leaf_outcome_invalid_ratio": 0.25,
                "selected_leaf_outcome_wrong_ratio": 0.25,
                "selected_leaf_outcome_correct_ratio": 0.5,
                "format_primary_full_count": 1,
                "format_primary_full_ratio": 0.5,
                "format_primary_no_step_count": 0,
                "format_primary_no_step_ratio": 0.0,
                "format_primary_text_outside_step_count": 0,
                "format_primary_text_outside_step_ratio": 0.0,
                "format_primary_step_xml_invalid_count": 0,
                "format_primary_step_xml_invalid_ratio": 0.0,
                "format_primary_step_schema_invalid_count": 0,
                "format_primary_step_schema_invalid_ratio": 0.0,
                "format_primary_boxed_missing_count": 1,
                "format_primary_boxed_missing_ratio": 0.5,
                "format_primary_boxed_invalid_count": 0,
                "format_primary_boxed_invalid_ratio": 0.0,
                "relaxed_format_correct_count": 1,
                "relaxed_format_correct_ratio": 0.5,
            },
            {"branch_generation": 2.0},
        )

        self.assertEqual(metrics["Tree/format_steps"], 3)
        self.assertEqual(metrics["Tree/leaf_acc"], 0.75)
        self.assertEqual(metrics["Tree/llm_rm_score"], 0.0)
        self.assertEqual(metrics["Tree/format_primary/full_count"], 1)
        self.assertEqual(metrics["Tree/format_primary/full_ratio"], 0.5)
        self.assertEqual(metrics["Tree/format_primary/boxed_missing_count"], 1)
        self.assertEqual(metrics["Tree/format_primary/boxed_missing_ratio"], 0.5)
        self.assertEqual(metrics["Tree/format_primary/relaxed_format_correct_count"], 1)
        self.assertEqual(metrics["Tree/format_primary/relaxed_format_correct_ratio"], 0.5)
        self.assertEqual(metrics["Tree/time_branch_generation"], 2.0)
        self.assertEqual(metrics["reward/step_treerl_process_reward_mean"], 1.25)
        self.assertEqual(metrics["rollout/step_num"], 2.5)
        self.assertEqual(metrics["reward/outcome_reward"], 0.55)
        self.assertNotIn("reward/step_treerl_selected_leaf_outcome_mean", metrics)
        self.assertEqual(metrics["Tree/selected_leaf_outcome_invalid_ratio"], 0.25)
        self.assertEqual(metrics["Tree/selected_leaf_outcome_wrong_ratio"], 0.25)
        self.assertEqual(metrics["Tree/selected_leaf_outcome_correct_ratio"], 0.5)
        self.assertFalse(any(key.startswith("val-core/") for key in metrics))
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
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{(A)}")["format_primary"], "full")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{ (A) }")["format_primary"], "full")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{{(A)}}")["format_primary"], "full")
        self.assertEqual(classify_rollout_format(r"plain reasoning \boxed{(A)}")["format_primary"], "no_step")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{A}}")["format_primary"], "boxed_invalid")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{{A}")["format_primary"], "boxed_invalid")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{AB}")["format_primary"], "boxed_invalid")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{(AB)}")["format_primary"], "boxed_invalid")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{(A}}")["format_primary"], "boxed_invalid")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{A)}")["format_primary"], "boxed_invalid")
        self.assertEqual(classify_rollout_format(good_step + r"\boxed{(A)}.")["format_primary"], "boxed_invalid")

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
        self.assertEqual(columns["format_error_advantage_mask"], [0.0])
        self.assertEqual(
            set(columns),
            {"format_primary", "boxed_status", "boxed_answer", "step_block_count", "format_error_advantage_mask"},
        )

    def test_rollout_format_metric_columns_are_numeric_one_hot_fields(self):
        good_step = "<step><premise>a</premise><conclusion>b</conclusion></step>"
        columns = rollout_format_infos_to_metric_columns(
            [
                classify_rollout_format(good_step + r"\boxed{A}"),
                classify_rollout_format("plain reasoning " + r"\boxed{A}"),
            ]
        )

        self.assertEqual(columns["format_primary_full"], [1.0, 0.0])
        self.assertEqual(columns["format_primary_no_step"], [0.0, 1.0])
        self.assertEqual(columns["boxed_status_valid"], [1.0, 1.0])
        self.assertEqual(columns["relaxed_format_correct"], [1.0, 0.0])
        self.assertEqual(columns["step_block_count"], [1.0, 0.0])
        self.assertEqual(columns["format_error_advantage_mask"], [0.0, 1.0])

    def test_trainer_metrics_include_rollout_provenance(self):
        tokenizer = DummyTokenizer()
        good_step = "<step><premise>a</premise><conclusion>b</conclusion></step>"
        outputs = [good_step + r"\boxed{A}", "prefix" + good_step + r"\boxed{A}"]
        encoded = [tokenizer.encode(output) for output in outputs]
        response_width = max(len(tokens) for tokens in encoded)
        responses = torch.zeros((2, response_width), dtype=torch.long)
        response_mask = torch.zeros_like(responses)
        for index, tokens in enumerate(encoded):
            responses[index, : len(tokens)] = torch.tensor(tokens)
            response_mask[index, : len(tokens)] = 1

        prompts = torch.tensor([[10, 11], [12, 13]])
        attention_mask = torch.cat([torch.ones_like(prompts), response_mask], dim=-1)
        batch = DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
            },
            non_tensors={
                "trace_source": np.asarray(["origin", "branch"], dtype=object),
                "branch_round": np.asarray([0, 2], dtype=np.int64),
            },
        )
        trainer = SimpleNamespace(tokenizer=tokenizer)
        trainer._decode_rollout_responses = lambda rollout_batch: RayPPOTrainer._decode_rollout_responses(
            trainer, rollout_batch
        )

        metrics, columns, format_infos = RayPPOTrainer._compute_rollout_format_metrics(trainer, batch)

        self.assertEqual([info["format_primary"] for info in format_infos], ["full", "text_outside_step"])
        self.assertEqual(columns["trace_source"], ["origin", "branch"])
        self.assertEqual(columns["branch_round"], [0, 2])
        self.assertEqual(metrics["rollout/format_by_source/origin/strict_correct_ratio"], 1.0)
        self.assertEqual(metrics["rollout/format_by_source/branch/strict_correct_ratio"], 0.0)
        self.assertEqual(metrics["rollout/format_by_source/branch/relaxed_correct_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
