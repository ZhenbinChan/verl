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

import json
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf

from examples.data_preprocess.mcq_preprocess import FOLPreprocessor, extract_fields_from_record
from verl.trainer.ppo.sampling.mcts_prm import fol_step_reward_with_context
from verl.utils.fol_verifier import (
    EntityGroupsSchema,
    FOLMetadata,
    FOLVerifier,
    LLMClient,
    PredicateExtractionSchema,
    load_fol_metadata,
)
from verl.utils.process_reward import (
    ProcessRewardRuntime,
    StepRewardRequest,
    build_process_reward_runtime,
    build_generation_non_tensor_keys_to_pop,
    get_batch_sample_id,
    get_batch_question_text,
    resolve_process_reward_config,
)
from verl.workers.reward_manager.step_tree import StepTreeRewardManager


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def decode(self, tokens, skip_special_tokens=True):
        return ""

    def encode(self, text, add_special_tokens=False):
        return []


class StubLLMClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def constrain_generate(self, prompt, format, args=None):
        self.calls.append((prompt, format, args))
        payload = self.responses[format]
        if isinstance(payload, list):
            payload = payload.pop(0)
        return format(**payload)

    def generate(self, prompt, args=None):
        self.calls.append((prompt, "generate", args))
        return "<p1>ctx</p1>"


class TestProcessRewardConfig(unittest.TestCase):
    def test_resolve_process_reward_config_sets_auto_reward_manager(self):
        config = OmegaConf.create(
            {
                "trainer": {
                    "sampling_strategy": "step_treerl",
                    "process_reward": {"type": "format", "fol": {"llm": {"model_name": None}}},
                },
                "reward_model": {
                    "reward_manager": "auto",
                },
            }
        )

        process_reward_cfg = resolve_process_reward_config(config)

        self.assertEqual(process_reward_cfg.type, "format")
        self.assertEqual(config.reward_model.reward_manager, "step_tree")

    def test_resolve_process_reward_config_defaults_step_treerl_none_to_format(self):
        config = OmegaConf.create(
            {
                "trainer": {
                    "sampling_strategy": "step_treerl",
                    "process_reward": {"type": "none", "fol": {"llm": {"model_name": None}}},
                },
                "reward_model": {
                    "reward_manager": "auto",
                },
            }
        )

        process_reward_cfg = resolve_process_reward_config(config)

        self.assertEqual(process_reward_cfg.type, "format")

    def test_build_process_reward_runtime_fol_requires_model_name(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(
                [
                    {
                        "sample_id": "sample-0",
                        "fol_metadata": {
                            "sample_id": "sample-0",
                            "rephrased_context": "ctx",
                            "z3_declaration_code": "x = Int('x')",
                        },
                    }
                ],
                handle,
            )
            metadata_path = handle.name

        cfg = {
            "type": "fol",
            "fol": {
                "metadata_path": metadata_path,
                "llm": {
                    "api_base_url": "http://localhost:4869/v1",
                    "api_key": "EMPTY",
                    "model_name": None,
                },
            },
        }

        with self.assertRaises(ValueError):
            build_process_reward_runtime(cfg)

    def test_build_process_reward_runtime_self_eval_loads_prompt(self):
        runtime = build_process_reward_runtime({"type": "self_eval"})

        self.assertEqual(runtime.reward_type, "self_eval")
        self.assertIn("{question_text}", runtime.self_eval_prompt_template)
        self.assertIn("{reasoning_steps}", runtime.self_eval_prompt_template)
        self.assertEqual(runtime.self_eval_max_new_tokens, 32)

    def test_build_process_reward_runtime_self_eval_rejects_bad_prompt(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("missing placeholders")
            prompt_path = handle.name

        with self.assertRaises(ValueError):
            build_process_reward_runtime(
                {
                    "type": "self_eval",
                    "self_eval": {
                        "prompt_path": prompt_path,
                    },
                }
            )

    def test_resolve_process_reward_config_accepts_self_eval(self):
        config = OmegaConf.create(
            {
                "trainer": {
                    "sampling_strategy": "step_treerl",
                    "process_reward": {"type": "self_eval"},
                },
                "reward_model": {
                    "reward_manager": "auto",
                },
            }
        )

        process_reward_cfg = resolve_process_reward_config(config)

        self.assertEqual(process_reward_cfg.type, "self_eval")
        self.assertEqual(config.reward_model.reward_manager, "step_tree")

    def test_self_eval_generation_pop_keys_preserve_data_source(self):
        non_tensor_batch = {
            "raw_prompt_ids": np.array([1], dtype=object),
            "answer": np.array(["A"], dtype=object),
            "question_text": np.array(["q"], dtype=object),
            "extra_info": np.array([{"context": "ctx"}], dtype=object),
            "index": np.array([0], dtype=object),
            "data_source": np.array(["reclor"], dtype=object),
        }

        keys = build_generation_non_tensor_keys_to_pop(non_tensor_batch, "self_eval")

        self.assertIn("question_text", keys)
        self.assertIn("extra_info", keys)
        self.assertIn("index", keys)
        self.assertNotIn("data_source", keys)

    def test_fol_generation_pop_keys_keep_data_source_for_sample_id_fallback(self):
        non_tensor_batch = {
            "raw_prompt_ids": np.array([1], dtype=object),
            "index": np.array([0], dtype=object),
            "data_source": np.array(["reclor"], dtype=object),
        }

        keys = build_generation_non_tensor_keys_to_pop(non_tensor_batch, "fol")

        self.assertIn("data_source", keys)
        self.assertIn("index", keys)


class TestFOLVerifierStrictMode(unittest.TestCase):
    def test_llm_client_bypasses_proxy_for_local_base_url(self):
        self.assertTrue(LLMClient(base_url="http://localhost:4869/v1")._should_bypass_env_proxy())
        self.assertTrue(LLMClient(base_url="http://127.0.0.1:4869/v1")._should_bypass_env_proxy())
        self.assertFalse(LLMClient(base_url="https://api.openai.com/v1")._should_bypass_env_proxy())

    def test_wrap_z3_code_checks_entailment_via_negated_conclusion(self):
        verifier = FOLVerifier(llm_client=object())
        wrapped = verifier.wrap_z3_code("x = Bool('x')", "premise_fol = x\nconclusion_fol = x")
        self.assertIn("s.add(Not(conclusion_fol))", wrapped)

    def test_global_logic_step_verifies_last_step_entailment(self):
        verifier = FOLVerifier(llm_client=object())
        code = """
premises_1 = [x]
conclusion_1 = x
premises_2 = [conclusion_1]
conclusion_2 = x
"""
        parsed = verifier.parse_global_logic_steps(code)
        self.assertEqual(len(parsed), 2)
        result = verifier.run_global_logic_step("x = Bool('x')", parsed[-1])
        self.assertTrue(result["success"])
        self.assertIn("SUCCESS_ENTAILED", result["output"])

    def test_fol_metadata_preserves_global_prm_fields(self):
        metadata = FOLMetadata(
            sample_id="sample-0",
            rephrased_context="",
            question_text="<Context>ctx</Context>",
            prm_mode="global_fol_prm",
            z3_declaration_code="x = Bool('x')",
        )
        restored = FOLMetadata.from_dict(metadata.to_dict())
        self.assertEqual(restored.question_text, "<Context>ctx</Context>")
        self.assertEqual(restored.prm_mode, "global_fol_prm")

    def test_debug_dump_writes_json_and_python_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = FOLVerifier(llm_client=object(), debug_dir=tmpdir)
            metadata = FOLMetadata(
                sample_id="sample-0",
                rephrased_context="ctx",
                z3_declaration_code="x = Bool('x')",
            )
            dump_path = verifier._dump_debug_artifacts(
                metadata=metadata,
                step_text="<step><premise>a</premise><conclusion>a</conclusion></step>",
                payload={"wrapped_z3_code": "print('hello')", "final_score": 1.0},
            )
            self.assertIsNotNone(dump_path)
            self.assertTrue(dump_path.exists())
            self.assertTrue(dump_path.with_suffix(".py").exists())

    def test_build_z3_declaration_code_uses_sort_symbols_and_unique_constants(self):
        verifier = FOLVerifier(llm_client=object())
        entities = {
            "entity_type": ["protein", "hormone", "injury"],
            "entity_value": ["protein", "hormone", "new_arthritis_medication"],
        }
        predicates = {
            "inhibit_functioning": ["protein", "hormone"],
            "activated_by": ["hormone", "injury"],
        }
        code = verifier.build_z3_declaration_code(entities, predicates)

        self.assertIn("protein_sort = DeclareSort('protein')", code)
        self.assertIn("hormone_sort = DeclareSort('hormone')", code)
        self.assertIn("protein = Const('protein', protein_sort)", code)
        self.assertIn("hormone = Const('hormone', hormone_sort)", code)
        self.assertIn(
            "inhibit_functioning = Function('inhibit_functioning', protein_sort, hormone_sort, BoolSort())",
            code,
        )
        self.assertEqual(code.count("protein = Const("), 1)
        self.assertNotIn("Function('inhibit_functioning', protein, hormone)", code)

    def test_build_z3_declaration_code_supports_structured_schema(self):
        verifier = FOLVerifier(llm_client=object())
        entities = {
            "entity_groups": [
                {"type": "partner", "sort_kind": "uninterpreted", "values": ["Hodges", "Nader"]},
                {"type": "year", "sort_kind": "int", "values": [1961, 1962]},
            ]
        }
        predicates = {
            "predicates": [
                {"name": "join_year", "arg_types": ["partner"], "return_type": "year"},
                {"name": "earlier_than", "arg_types": ["partner", "partner"], "return_type": "bool"},
            ]
        }
        code = verifier.build_z3_declaration_code(entities, predicates)

        self.assertIn("partner_sort = DeclareSort('partner')", code)
        self.assertNotIn("year_sort = DeclareSort('year')", code)
        self.assertIn("hodges = Const('Hodges', partner_sort)", code)
        self.assertIn("sym_1961 = IntVal(1961)", code)
        self.assertIn("join_year = Function('join_year', partner_sort, IntSort())", code)
        self.assertIn(
            "earlier_than = Function('earlier_than', partner_sort, partner_sort, BoolSort())",
            code,
        )

    def test_strict_extractors_use_structured_prompts_and_schemas(self):
        client = StubLLMClient(
            {
                EntityGroupsSchema: {
                    "entity_groups": [
                        {"type": "person", "sort_kind": "uninterpreted", "values": ["Paula", "Bill"]}
                    ]
                },
                PredicateExtractionSchema: {
                    "predicates": [
                        {"name": "earlier_than", "arg_types": ["person", "person"], "return_type": "bool"}
                    ]
                },
            }
        )
        verifier = FOLVerifier(llm_client=client)

        entities = verifier.object_extract("ctx", "q", "opts", schema_variant="strict_v1")
        predicates = verifier.predicate_extract(
            "ctx",
            "q",
            "opts",
            obj_list=entities,
            schema_variant="strict_v1",
        )

        self.assertEqual(entities["entity_groups"][0]["type"], "person")
        self.assertEqual(predicates["predicates"][0]["name"], "earlier_than")
        self.assertEqual(client.calls[0][1], EntityGroupsSchema)
        self.assertEqual(client.calls[1][1], PredicateExtractionSchema)

    def test_strict_validator_augments_missing_predicate_types(self):
        preprocessor = FOLPreprocessor(llm_client=None, schema_variant="strict_v1")
        entities = {
            "entity_groups": [
                {"type": "person", "sort_kind": "uninterpreted", "values": ["Paula", "Bill"]},
            ]
        }
        predicates = {
            "predicates": [
                {"name": "goes_with", "arg_types": ["person", "direction"], "return_type": "bool"},
            ]
        }
        normalized_entities = preprocessor._validate_and_normalize_entities(entities)
        normalized_predicates = preprocessor._validate_and_normalize_predicates(predicates, normalized_entities)
        self.assertEqual(normalized_predicates["predicates"][0]["arg_types"], ["person", "direction"])
        self.assertIn(
            "direction",
            {group["type"] for group in normalized_entities["entity_groups"]},
        )

    def test_strict_extract_fol_metadata_auto_retries_on_validation_failure(self):
        client = StubLLMClient(
            {
                EntityGroupsSchema: [
                    {
                        "entity_groups": [
                            {"type": "entity", "sort_kind": "uninterpreted", "values": ["foo"]},
                        ]
                    },
                    {
                        "entity_groups": [
                            {"type": "person", "sort_kind": "uninterpreted", "values": ["Paula", "Bill"]},
                        ]
                    },
                    {
                        "entity_groups": [
                            {"type": "person", "sort_kind": "uninterpreted", "values": ["Paula", "Bill"]},
                        ]
                    },
                ],
                PredicateExtractionSchema: [
                    {
                        "predicates": [],
                    },
                    {
                        "predicates": [
                            {"name": "meets", "arg_types": ["person", "person"], "return_type": "bool"},
                        ]
                    },
                ],
            }
        )
        preprocessor = FOLPreprocessor(
            llm_client=client,
            schema_variant="strict_v1",
            schema_validation_retries=3,
        )
        metadata = preprocessor.extract_fol_metadata("ctx", "q", "opts", sample_id="sample-0")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.entities["entity_groups"][0]["type"], "person")
        self.assertEqual(metadata.predicates["predicates"][0]["name"], "meets")

    def test_load_fol_metadata_preserves_stored_declaration_code(self):
        payload = [
            {
                "sample_id": "sample-0",
                "fol_metadata": {
                    "sample_id": "sample-0",
                    "rephrased_context": "ctx",
                    "entities": {
                        "entity_type": ["protein", "hormone"],
                        "entity_value": ["protein", "hormone"],
                    },
                    "predicates": {"inhibit_functioning": ["protein", "hormone"]},
                    "z3_declaration_code": "BROKEN",
                },
            }
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            metadata_path = handle.name

        loaded = load_fol_metadata(metadata_path)
        self.assertIn("sample-0", loaded)
        self.assertEqual("BROKEN", loaded["sample-0"].z3_declaration_code)

    def test_refresh_fol_metadata_rebuilds_declaration_code(self):
        preprocessor = FOLPreprocessor(llm_client=None)
        refreshed = preprocessor.refresh_fol_metadata(
            {
                "sample_id": "sample-0",
                "rephrased_context": "ctx",
                "entities": {
                    "entity_type": ["protein", "hormone"],
                    "entity_value": ["protein", "hormone"],
                },
                "predicates": {"inhibit_functioning": ["protein", "hormone"]},
                "z3_declaration_code": "BROKEN",
            },
            sample_id="sample-0",
        )
        self.assertIn(
            "inhibit_functioning = Function('inhibit_functioning', protein_sort, hormone_sort, BoolSort())",
            refreshed.z3_declaration_code,
        )

    def test_refresh_fol_metadata_normalizes_numpy_arrays(self):
        preprocessor = FOLPreprocessor(llm_client=None)
        refreshed = preprocessor.refresh_fol_metadata(
            {
                "sample_id": "sample-0",
                "rephrased_context": "ctx",
                "entities": {
                    "entity_type": np.array(["protein", "hormone"]),
                    "entity_value": np.array(["protein", "hormone"]),
                },
                "predicates": {"inhibit_functioning": np.array(["protein", "hormone"])},
                "z3_declaration_code": "BROKEN",
            },
            sample_id="sample-0",
        )
        self.assertIsInstance(refreshed.entities["entity_type"], list)
        self.assertIsInstance(refreshed.predicates["inhibit_functioning"], list)

    def test_extract_fields_from_record_parses_json_string_fol_metadata(self):
        record = {
            "context": "ctx",
            "question": "q",
            "answers": ["a", "b", "c", "d"],
            "label": 0,
            "fol_metadata": json.dumps({"sample_id": "s0", "entities": {"entity_groups": []}}),
        }
        extracted = extract_fields_from_record(
            record,
            "context",
            "question",
            "answers",
            "label",
            ["A", "B", "C", "D"],
            "<Context>{context}</Context><Question>{question}</Question><Options>{answers}</Options>",
        )
        self.assertIsInstance(extracted["fol_metadata"], dict)
        self.assertEqual(extracted["fol_metadata"]["sample_id"], "s0")

    def test_verify_step_requires_llm_client(self):
        verifier = FOLVerifier(llm_client=None)
        metadata = FOLMetadata(
            sample_id="sample-0",
            rephrased_context="ctx",
            z3_declaration_code="x = Int('x')",
        )
        step_text = "<step><premise>a</premise><conclusion>b</conclusion></step>"

        with self.assertRaises(RuntimeError):
            verifier.verify_step(metadata, step_text, use_llm=True)

    def test_verify_step_requires_use_llm(self):
        verifier = FOLVerifier(llm_client=object())
        metadata = FOLMetadata(
            sample_id="sample-0",
            rephrased_context="ctx",
            z3_declaration_code="x = Int('x')",
        )
        step_text = "<step><premise>a</premise><conclusion>b</conclusion></step>"

        with self.assertRaises(RuntimeError):
            verifier.verify_step(metadata, step_text, use_llm=False)

    def test_fol_step_reward_with_context_requires_sample_id(self):
        verifier = FOLVerifier(llm_client=object())
        metadata = FOLMetadata(
            sample_id="sample-0",
            rephrased_context="ctx",
            z3_declaration_code="x = Int('x')",
        )

        with self.assertRaises(ValueError):
            fol_step_reward_with_context(
                "<step><premise>a</premise><conclusion>b</conclusion></step>",
                sample_id=None,
                sample_metadata_map={"sample-0": metadata},
                verifier=verifier,
            )


class TestRewardManagerFallback(unittest.TestCase):
    def test_step_tree_fallback_passes_sample_id_for_fol(self):
        manager = StepTreeRewardManager(
            tokenizer=DummyTokenizer(),
            num_examine=0,
            process_reward_cfg={"type": "format"},
        )
        manager.process_reward_type = "fol"
        manager._step_prm_fn = lambda step_text, sample_id=None: 1.0 if sample_id == "sample-1" else 0.0

        scores = manager._fallback_step_scores(
            "<step><premise>a</premise><conclusion>b</conclusion></step>",
            sample_id="sample-1",
        )

        self.assertEqual(scores, [1.0])

    def test_step_tree_fallback_rejects_self_eval_without_precomputed_scores(self):
        manager = StepTreeRewardManager(
            tokenizer=DummyTokenizer(),
            num_examine=0,
            process_reward_cfg={"type": "self_eval"},
        )

        with self.assertRaises(ValueError):
            manager._fallback_step_scores(
                "<step><premise>a</premise><conclusion>b</conclusion></step>"
            )


class TestProcessRewardBatchScoring(unittest.TestCase):
    def test_global_fol_batch_scoring_deduplicates_and_preserves_order(self):
        class FakeVerifier:
            def __init__(self):
                self.calls = []

            def verify_global_step(self, metadata, reasoning_steps, question_text=None, args=None):
                self.calls.append((metadata.sample_id, reasoning_steps, question_text))
                return 1.0 if "good" in reasoning_steps else 0.0

        verifier = FakeVerifier()
        runtime = ProcessRewardRuntime(
            reward_type="fol",
            fol_verifier=verifier,
            fol_metadata_map={
                "s0": FOLMetadata(sample_id="s0", rephrased_context="", question_text="q0", z3_declaration_code="x = Bool('x')"),
                "s1": FOLMetadata(sample_id="s1", rephrased_context="", question_text="q1", z3_declaration_code="x = Bool('x')"),
            },
            fol_prm_mode="global_fol_prm",
            max_concurrency=2,
        )
        requests = [
            StepRewardRequest(step_text="a", accumulated_text="good", sample_id="s0", question_text="q0"),
            StepRewardRequest(step_text="b", accumulated_text="bad", sample_id="s1", question_text="q1"),
            StepRewardRequest(step_text="a", accumulated_text="good", sample_id="s0", question_text="q0"),
        ]

        self.assertEqual(runtime.score_steps(requests), [1.0, 0.0, 1.0])
        self.assertEqual(len(verifier.calls), 2)

    def test_get_batch_question_text_from_extra_info(self):
        batch = {
            "extra_info": [
                {
                    "context": "ctx",
                    "query": "q",
                    "options": "(A) a",
                }
            ]
        }
        self.assertEqual(
            get_batch_question_text(batch, 0),
            "<Context>ctx</Context><Question>q</Question><Options>(A) a</Options>",
        )

    def test_get_batch_sample_id_prefixes_extra_info_index_with_data_source(self):
        batch = {
            "data_source": ["reclor"],
            "extra_info": [{"index": 3}],
        }
        self.assertEqual(get_batch_sample_id(batch, 0), "reclor_3")


if __name__ == "__main__":
    unittest.main()
