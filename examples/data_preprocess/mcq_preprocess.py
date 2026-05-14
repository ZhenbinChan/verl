"""
Generic MCQ Dataset Preprocessor with Optional FOL Metadata Extraction

支持任意多选题数据集（ReClor, LogiQA, ARC, 等等），
通过命令行参数指定字段映射，统一输出 parquet 格式，
可选提取 FOL metadata 用于 step_treerl 训练。

Usage:
    # ReClor: 从 parquet 读取，跳过 FOL 提取
    python examples/data_preprocess/mcq_preprocess.py \
        --input_parquet data/reclor/train.parquet \
        --output_dir data/reclor_fol \
        --skip_fol_extraction

    # ReClor: 从 HuggingFace 加载，提取 FOL metadata
    python examples/data_preprocess/mcq_preprocess.py \
        --dataset metaeval/reclor \
        --output_dir data/reclor_fol \
        --context_field context \
        --question_field question \
        --answers_field answers \
        --label_field label \
        --api_key "your-api-key"

    # LogiQA: 从 parquet 读取，提取 FOL metadata
    python examples/data_preprocess/mcq_preprocess.py \
        --input_parquet data/logiqa/train.parquet \
        --output_dir data/logiqa_fol \
        --context_field context \
        --question_field query \
        --answers_field options \
        --label_field correct_option \
        --raw_prompt_template "{context}\\n\\n{question}\\n\\n{answers}" \
        --api_key "your-api-key"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verl.utils.fol_verifier import (
    FOLMetadata,
    FOLVerifier,
    LLMClient,
    save_fol_metadata,
)


# ─── Field mapping presets for common datasets ────────────────────────────

PRESETS = {
    "reclor": dict(
        context_field="context",
        question_field="question",
        answers_field="answers",
        label_field="label",
        raw_prompt_template="<Context>{context}</Context><Question>{question}</Question><Options>{answers}</Options>",
        prompt_template="<Context>{context}</Context><Question>{question}</Question><Options>{answers}</Options>",
        dataset_name="metaeval/reclor",
        train_split="train",
        test_split="validation",
    ),
    "logiqa": dict(
        context_field="context",
        question_field="query",
        answers_field="options",
        label_field="correct_option",
        raw_prompt_template="<Context>{context}</Context><Question>{question}</Question><Options>{answers}</Options>",
        prompt_template="<Context>{context}</Context><Question>{question}</Question><Options>{answers}</Options>",
        dataset_name="lucasmccabe/logiqa",
        train_split="train",
        test_split="validation",
    ),
}


def get_option_text(answers_field: str, option_mapping: List[str], raw_answers) -> str:
    """将 answers 列表格式化为带标签的选项文本"""
    if isinstance(raw_answers, list):
        return "\n".join(
            f"({option_mapping[i]}): " + raw_answers[i] + "."
            for i in range(len(raw_answers))
        )
    return str(raw_answers)


def build_question(
    context: str,
    question: str,
    answers_str: str,
    raw_prompt_template: str,
) -> str:
    """根据模板构建 question 字符串"""
    return raw_prompt_template.format(
        context=context,
        question=question,
        answers=answers_str,
    )


def load_from_parquet(parquet_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """从现有 parquet 文件加载数据"""
    df = pd.read_parquet(parquet_path)
    if num_samples:
        df = df.head(num_samples)
    return df.to_dict("records")


def _parse_full_question(full_text: str) -> tuple:
    """从完整 prompt 文本中解析出 context, question, options"""
    ctx_m = re.search(r'<Context>(.*?)</Context>', full_text, re.DOTALL)
    q_m = re.search(r'<Question>(.*?)</Question>', full_text, re.DOTALL)
    opt_m = re.search(r'<Options>(.*?)</Options>', full_text, re.DOTALL)
    return (
        ctx_m.group(1).strip() if ctx_m else "",
        q_m.group(1).strip() if q_m else "",
        opt_m.group(1).strip() if opt_m else "",
    )


def extract_fields_from_record(
    record: Dict,
    context_field: str,
    question_field: str,
    answers_field: str,
    label_field: str,
    option_mapping: List[str],
    raw_prompt_template: str,
) -> Dict:
    """从原始记录中提取并标准化字段

    优先从顶层字段取值；若为空，则：
    1. 尝试从 extra_info 嵌套结构中取
    2. 若 extra_info.question 存在（完整带标签文本），用正则解析出三个字段
    """
    # 顶层字段
    raw_context = record.get(context_field, "")
    raw_question = record.get(question_field, "")
    raw_answers = record.get(answers_field, [])
    raw_label = record.get(label_field, 0)

    # 顶层为空时，从 extra_info 中取
    if not raw_context or not isinstance(raw_context, str):
        raw_context = ""
    if not raw_question or not isinstance(raw_question, str):
        raw_question = ""
    if not raw_answers or not isinstance(raw_answers, list):
        raw_answers = []

    extra_info = record.get("extra_info", {})
    if isinstance(extra_info, dict):
        # 顶层为空时从 extra_info 取
        if not raw_context:
            raw_context = extra_info.get(context_field, "") or extra_info.get("context", "") or ""
        if not raw_question:
            raw_question = extra_info.get(question_field, "") or extra_info.get("question", "") or ""
        if not raw_answers:
            raw_answers = extra_info.get(answers_field, []) or extra_info.get("options", []) or []
        if not raw_label:
            raw_label = extra_info.get(label_field, 0) or extra_info.get("label", 0)

        # 若仍为空，尝试从 extra_info.question 解析完整 prompt
        if not raw_context or not raw_question or not raw_answers:
            full_q = extra_info.get("question", "") or extra_info.get("raw_prompt", "") or ""
            if full_q and "<Context>" in full_q:
                raw_context, raw_question, raw_answers = _parse_full_question(full_q)

    if not isinstance(raw_context, str):
        raw_context = str(raw_context) if raw_context else ""
    if not isinstance(raw_question, str):
        raw_question = str(raw_question) if raw_question else ""
    if not isinstance(raw_answers, list):
        raw_answers = [str(raw_answers)] if raw_answers else []

    answers_str = get_option_text(answers_field, option_mapping, raw_answers)
    question = build_question(raw_context, raw_question, answers_str, raw_prompt_template)

    # Map label to letter
    try:
        label_int = int(raw_label)
    except (ValueError, TypeError):
        label_int = 0
    solution = option_mapping[label_int]

    fol_metadata = record.get("fol_metadata", None)
    if isinstance(fol_metadata, str):
        try:
            fol_metadata = json.loads(fol_metadata)
        except json.JSONDecodeError:
            pass

    return {
        "context": raw_context,
        "question": raw_question,
        "answers": answers_str,
        "answer": solution,
        "question_full": question,
        "fol_metadata": fol_metadata,
    }


def convert_parquet_record(
    record: Dict,
    idx: int,
    context_field: str,
    question_field: str,
    answers_field: str,
    label_field: str,
    option_mapping: List[str],
    raw_prompt_template: str,
    data_source: str = "mcq",
    skip_fol: bool = False,
    split_name: str = "train",
) -> Dict:
    """将现有 parquet 记录转换为 verl 训练格式"""
    extracted = extract_fields_from_record(
        record, context_field, question_field, answers_field,
        label_field, option_mapping, raw_prompt_template,
    )

    sample_id = str(record.get("sample_id", f"mcq_{idx}"))
    if split_name != "train" and not sample_id.startswith(f"{split_name}_"):
        sample_id = f"{split_name}_{sample_id}"
    extra_info = record.get("extra_info", {})
    if isinstance(extra_info, dict):
        index = extra_info.get("index", idx)
    else:
        index = idx

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": extracted["question_full"]}],
        "ability": "logic",
        "reward_model": {"style": "rule", "ground_truth": extracted["answer"]},
        "answer": extracted["answer"],
        "raw_prompt": extracted["question_full"],
        "sample_id": sample_id,
        "extra_info": {
            "split": extra_info.get("split", split_name) if isinstance(extra_info, dict) else split_name,
            "index": index,
            "answer": extracted["answer"],
            "question": extracted["question_full"],
            "context": extracted["context"],
            "query": extracted["question"],
            "options": extracted["answers"],
        },
        "fol_metadata": extracted["fol_metadata"] if not skip_fol else None,
    }


class FOLPreprocessor:
    """FOL 元数据预处理器"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        default_args: Optional[Dict] = None,
        schema_variant: str = "legacy",
        schema_validation_retries: int = 3,
    ):
        self.llm_client = llm_client
        self.schema_variant = schema_variant
        self.schema_validation_retries = max(1, int(schema_validation_retries))
        self.default_args = default_args or {
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.8,
        }
        self.fol_verifier = FOLVerifier(llm_client=llm_client) if llm_client else FOLVerifier()
        self._forbidden_entity_types = {
            "entity",
            "entities",
            "target",
            "targets",
            "object",
            "objects",
            "thing",
            "things",
            "misc",
            "unknown",
            "key_predicate",
            "key_relation",
            "key_relations",
            "predicate",
            "predicates",
            "relation",
            "relations",
            "assumption",
            "conclusion",
        }
        self._forbidden_predicate_names = {
            "entity",
            "entities",
            "key_predicate",
            "key_predicates",
            "key_relation",
            "key_relations",
            "predicate",
            "predicates",
            "relation",
            "relations",
            "assumption",
            "conclusion",
        }

    def build_z3_declaration_code(
        self,
        entities: Dict,
        predicates: Dict,
    ) -> str:
        return self.fol_verifier.build_z3_declaration_code(entities, predicates)

    def _normalize_type_name(self, value: Any) -> str:
        text = self.fol_verifier._sanitize_symbol_name(value)
        return self.fol_verifier._singularize_symbol(text)

    @staticmethod
    def _normalize_sort_kind(value: Any) -> str:
        normalized = str(value or "uninterpreted").strip().lower()
        if normalized in {"int", "intsort()"}:
            return "int"
        if normalized in {"real", "realsort()"}:
            return "real"
        return "uninterpreted"

    @staticmethod
    def _normalize_return_type(value: Any) -> str:
        normalized = str(value or "bool").strip().lower()
        if normalized in {"bool", "boolsort()", "boolean"}:
            return "bool"
        if normalized in {"int", "intsort()", "integer"}:
            return "int"
        if normalized in {"real", "realsort()", "float"}:
            return "real"
        return normalized

    @staticmethod
    def _dedupe_preserve_order(values: List[Any]) -> List[Any]:
        result = []
        seen = set()
        for value in values:
            key = repr(value)
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    def _normalize_value_for_sort(self, value: Any, sort_kind: str) -> Any:
        if sort_kind == "int":
            return int(value)
        if sort_kind == "real":
            return float(value)
        return str(value).strip()

    def _validate_and_normalize_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        if self.schema_variant != "strict_v1":
            return entities
        groups = entities.get("entity_groups") if isinstance(entities, dict) else None
        if not isinstance(groups, list) or not groups:
            raise ValueError("entity_groups must be a non-empty list")

        normalized_groups: List[Dict[str, Any]] = []
        by_type: Dict[str, Dict[str, Any]] = {}
        for group in groups:
            if not isinstance(group, dict):
                continue
            type_name = self._normalize_type_name(group.get("type", ""))
            if not type_name or type_name in self._forbidden_entity_types:
                continue
            sort_kind = self._normalize_sort_kind(group.get("sort_kind", "uninterpreted"))
            raw_values = group.get("values") or []
            if not isinstance(raw_values, list):
                raw_values = [raw_values]

            values = []
            for value in raw_values:
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                try:
                    normalized_value = self._normalize_value_for_sort(value, sort_kind)
                except (TypeError, ValueError):
                    continue
                if isinstance(normalized_value, str) and not normalized_value:
                    continue
                values.append(normalized_value)
            values = self._dedupe_preserve_order(values)
            if not values:
                continue

            existing = by_type.get(type_name)
            if existing is None:
                existing = {
                    "type": type_name,
                    "sort_kind": sort_kind,
                    "values": [],
                }
                by_type[type_name] = existing
                normalized_groups.append(existing)
            elif existing["sort_kind"] != sort_kind:
                raise ValueError(f"entity type {type_name} has conflicting sort kinds: {existing['sort_kind']} vs {sort_kind}")

            existing["values"].extend(values)
            existing["values"] = self._dedupe_preserve_order(existing["values"])

        if not normalized_groups:
            raise ValueError("no valid entity_groups remained after normalization")
        return {"entity_groups": normalized_groups}

    def _validate_and_normalize_predicates(
        self,
        predicates: Dict[str, Any],
        entities: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.schema_variant != "strict_v1":
            return predicates
        specs = predicates.get("predicates") if isinstance(predicates, dict) else None
        if not isinstance(specs, list) or not specs:
            raise ValueError("predicates must be a non-empty list")

        entity_types = {
            self._normalize_type_name(group.get("type", ""))
            for group in entities.get("entity_groups", [])
            if isinstance(group, dict)
        }
        if not entity_types:
            raise ValueError("predicate validation requires non-empty entity_groups")

        normalized_specs: List[Dict[str, Any]] = []
        seen = set()
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            name = self.fol_verifier._sanitize_symbol_name(spec.get("name", ""))
            if not name or name in self._forbidden_predicate_names:
                continue

            raw_arg_types = spec.get("arg_types") or []
            if not isinstance(raw_arg_types, list):
                raw_arg_types = [raw_arg_types]
            arg_types = [self._normalize_type_name(t) for t in raw_arg_types if str(t).strip()]
            return_type = self._normalize_return_type(spec.get("return_type", "bool"))
            referenced_types = {
                arg_type for arg_type in arg_types if arg_type not in entity_types
            }
            if return_type not in {"bool", "int", "real"} and return_type not in entity_types:
                referenced_types.add(return_type)
            for missing_type in sorted(referenced_types):
                if not missing_type or missing_type in self._forbidden_entity_types:
                    raise ValueError(f"predicate {name} references unknown or forbidden type: {missing_type}")
                entities.setdefault("entity_groups", []).append(
                    {
                        "type": missing_type,
                        "sort_kind": "uninterpreted",
                        "values": [],
                    }
                )
                entity_types.add(missing_type)

            key = (name, tuple(arg_types), return_type)
            if key in seen:
                continue
            seen.add(key)
            normalized_specs.append(
                {
                    "name": name,
                    "arg_types": arg_types,
                    "return_type": return_type,
                }
            )

        if not normalized_specs:
            raise ValueError("no valid predicates remained after normalization")
        return {"predicates": normalized_specs}

    def _validate_declaration_code(self, z3_declaration_code: str) -> None:
        namespace: Dict[str, Any] = {}
        exec("from z3 import *\n" + z3_declaration_code, namespace, namespace)

    def refresh_fol_metadata(
        self,
        fol_metadata: Dict,
        sample_id: str,
    ) -> FOLMetadata:
        metadata = FOLMetadata.from_dict(fol_metadata)
        if not isinstance(metadata.entities, dict) or not metadata.entities:
            raise ValueError(f"sample {sample_id} is missing entities; cannot rebuild z3_declaration_code")
        if not isinstance(metadata.predicates, dict) or not metadata.predicates:
            raise ValueError(f"sample {sample_id} is missing entities/predicates; cannot rebuild z3_declaration_code")
        metadata.sample_id = sample_id
        metadata.z3_declaration_code = self.build_z3_declaration_code(
            metadata.entities,
            metadata.predicates,
        )
        return metadata

    def extract_fol_metadata(
        self,
        context: str,
        question: str,
        options: str,
        sample_id: str,
    ) -> Optional[FOLMetadata]:
        """提取单个样本的 FOL 元数据"""
        if self.llm_client is None:
            return None

        try:
            rephrased_context = self.fol_verifier.rephrase(
                context=context,
                question=question,
                options=options,
                args=self.default_args,
            )
            last_error: Optional[Exception] = None
            feedback: Optional[str] = None
            for _attempt_idx in range(self.schema_validation_retries):
                try:
                    entities = self.fol_verifier.object_extract(
                        context=context,
                        question=question,
                        options=options,
                        args=self.default_args,
                        schema_variant=self.schema_variant,
                        feedback=feedback,
                    )
                    entities = self._validate_and_normalize_entities(entities)
                    predicates = self.fol_verifier.predicate_extract(
                        context=context,
                        question=question,
                        options=options,
                        obj_list=entities,
                        args=self.default_args,
                        schema_variant=self.schema_variant,
                        feedback=feedback,
                    )
                    predicates = self._validate_and_normalize_predicates(predicates, entities)
                    z3_declaration_code = self.build_z3_declaration_code(entities, predicates)
                    self._validate_declaration_code(z3_declaration_code)

                    return FOLMetadata(
                        sample_id=sample_id,
                        rephrased_context=rephrased_context,
                        entities=entities,
                        predicates=predicates,
                        z3_declaration_code=z3_declaration_code,
                        ground_truth="",
                        axioms=[],
                    )
                except Exception as e:
                    last_error = e
                    if self.schema_variant == "strict_v1":
                        feedback = (
                            "The previous output violated validation constraints. "
                            f"Reason: {e}. "
                            "Entity groups must use meaningful semantic types and predicates must only reference existing entity types or builtin return types."
                        )
                    else:
                        break

            if last_error is not None:
                raise last_error
        except Exception as e:
            print(f"[Error] Failed to extract FOL metadata for {sample_id}: {e}")
            return None


def extract_fol_batch(
    samples: List[Dict],
    preprocessor: FOLPreprocessor,
    batch_save: int = 50,
    max_retries: int = 3,
    verbose: bool = False,
    refresh_declarations_only: bool = False,
) -> List[Dict]:
    """批量提取 FOL 元数据"""
    import json

    results = []
    for i, sample in enumerate(samples):
        if sample.get("fol_metadata") is not None:
            if refresh_declarations_only:
                sample_id = sample.get("sample_id", f"sample_{i}")
                try:
                    sample["fol_metadata"] = preprocessor.refresh_fol_metadata(
                        sample["fol_metadata"],
                        sample_id=sample_id,
                    ).to_dict()
                except Exception as e:
                    print(f"  [Dropped] {sample_id} — failed to refresh fol_metadata: {e}")
                    continue
            results.append(sample)
            continue

        extra_info = sample.get("extra_info", {})
        context = extra_info.get("context", "") or ""
        query = extra_info.get("query", "") or ""
        options = extra_info.get("options", "") or ""

        # Fallback: parse from raw_prompt if fields are missing
        if not context or not query or not options:
            # raw_prompt can be in sample['raw_prompt'] or extra_info['question']
            raw_prompt = str(sample.get("raw_prompt", "") or extra_info.get("question", "") or "")
            ctx_m = re.search(r'<Context>(.*?)</Context>', raw_prompt, re.DOTALL)
            q_m = re.search(r'<Question>(.*?)</Question>', raw_prompt, re.DOTALL)
            opt_m = re.search(r'<Options>(.*?)</Options>', raw_prompt, re.DOTALL)
            context = ctx_m.group(1).strip() if ctx_m else ""
            query = q_m.group(1).strip() if q_m else ""
            options = opt_m.group(1).strip() if opt_m else ""
            if verbose:
                print(f"  [DEBUG] fallback parsed: ctx={repr(context[:20])}, q={repr(query[:20])}, opt={repr(options[:20])}")

        sample_id = sample.get("sample_id", f"sample_{i}")

        print(f"[{i + 1}/{len(samples)}] Processing {sample_id}...")

        fol_metadata = None
        attempt = 0
        while fol_metadata is None:
            attempt += 1
            if max_retries != -1 and attempt > max_retries:
                print(f"  [Skip] Max retries ({max_retries}) reached, skipping {sample_id}")
                break
            try:
                fol_metadata = preprocessor.extract_fol_metadata(
                    context=context,
                    question=query,
                    options=options,
                    sample_id=sample_id,
                )
                if fol_metadata is None:
                    retry_msg = f"  [Retry {attempt}] Returned None" if max_retries == -1 or attempt < max_retries else f"  [Retry {attempt}/{max_retries}] Returned None"
                    print(f"{retry_msg}, retrying..." if max_retries == -1 or attempt < max_retries else f"{retry_msg}, skipping...")
            except Exception as e:
                print(f"  [Retry {attempt}/{max_retries}] Failed: {e}" if max_retries != -1 else f"  [Retry {attempt}] Failed: {e}")

        if fol_metadata:
            fol_metadata.ground_truth = sample.get("answer", "")
            sample["fol_metadata"] = fol_metadata.to_dict()
            if verbose:
                print(f"  -> rephrased_context: {fol_metadata.rephrased_context[:100] if fol_metadata.rephrased_context else '(empty)'}")
                entities_str = str(fol_metadata.entities)[:100]
                print(f"  -> entities: {entities_str}")
                predicates_str = str(fol_metadata.predicates)[:100]
                print(f"  -> predicates: {predicates_str}")
            results.append(sample)
        else:
            # max_retries exhausted and still failed — drop this sample
            print(f"  [Dropped] {sample_id} — {attempt - 1} attempts failed, removed from dataset")
            continue

        if batch_save > 0 and (i + 1) % batch_save == 0:
            print(f"[Progress] Processed {i + 1}/{len(samples)}")

    return results


def _make_parquet_safe_records(records: List[Dict], schema_variant: str) -> List[Dict]:
    if schema_variant != "strict_v1":
        return records

    parquet_safe_records: List[Dict] = []
    for record in records:
        safe_record = dict(record)
        fol_metadata = safe_record.get("fol_metadata")
        if isinstance(fol_metadata, dict):
            safe_record["fol_metadata"] = json.dumps(fol_metadata, ensure_ascii=False)
        parquet_safe_records.append(safe_record)
    return parquet_safe_records

def main():
    parser = argparse.ArgumentParser(
        description="Generic MCQ Preprocessor with Optional FOL Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Dataset source ────────────────────────────────────────────────────
    parser.add_argument(
        "--input_parquet",
        default=None,
        help="Path to existing train.parquet. If set, skips HuggingFace download.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="HuggingFace dataset name (e.g. metaeval/reclor, lucasmccabe/logiqa). "
             "Used when --input_parquet is not set.",
    )

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        default="./data/mcq",
        help="Output directory for processed parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS directory (optional).",
    )

    # ── Field mapping ──────────────────────────────────────────────────────
    parser.add_argument(
        "--preset",
        default=None,
        choices=list(PRESETS.keys()),
        help="Use preset field mappings for common datasets (overrides --context_field etc.).",
    )
    parser.add_argument(
        "--context_field", default="context",
        help="Field name for context in the source dataset.",
    )
    parser.add_argument(
        "--question_field", default="question",
        help="Field name for question in the source dataset.",
    )
    parser.add_argument(
        "--answers_field", default="answers",
        help="Field name for answer choices in the source dataset.",
    )
    parser.add_argument(
        "--label_field", default="label",
        help="Field name for ground-truth label in the source dataset.",
    )
    parser.add_argument(
        "--raw_prompt_template",
        default="<Context>{context}</Context><Question>{question}</Question><Options>{answers}</Options>",
        help="Template string to build full question prompt. "
             "Must contain {context}, {question}, {answers} placeholders.",
    )

    # ── FOL extraction ──────────────────────────────────────────────────────
    parser.add_argument(
        "--skip_fol_extraction",
        action="store_true",
        help="Skip FOL metadata extraction.",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key for LLM service (or set DASHSCOPE_API_KEY env var).",
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:4869/v1",
        help="Base URL for LLM service.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-3b",
        help="Model name for LLM.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retry attempts per sample. -1 means retry forever until success.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print first 10 chars of each extracted FOL field.",
    )
    parser.add_argument(
        "--fol_schema_variant",
        default="legacy",
        choices=["legacy", "strict_v1"],
        help="Prompt/schema variant for FOL entity/predicate extraction.",
    )
    parser.add_argument(
        "--schema_validation_retries",
        type=int,
        default=3,
        help="Internal retries for schema validation failures during a single sample extraction.",
    )
    parser.add_argument(
        "--extract_fol_for_validate",
        action="store_true",
        help="When validate.parquet exists, also extract FOL metadata for validate and merge it into fol_metadata.json.",
    )
    parser.add_argument(
        "--refresh_declarations_only",
        action="store_true",
        help="Rebuild z3_declaration_code from existing fol_metadata entities/predicates without new LLM extraction.",
    )

    args = parser.parse_args()

    # Apply preset if specified
    if args.preset:
        preset = PRESETS[args.preset]
        args.context_field = preset.get("context_field", args.context_field)
        args.question_field = preset.get("question_field", args.question_field)
        args.answers_field = preset.get("answers_field", args.answers_field)
        args.label_field = preset.get("label_field", args.label_field)
        args.raw_prompt_template = preset.get("raw_prompt_template", args.raw_prompt_template)
        print(f"[INFO] Using preset: {args.preset}")

    option_mapping = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    # ── Load data ────────────────────────────────────────────────────────────
    if args.input_parquet:
        print(f"[INFO] Loading from existing parquet: {args.input_parquet}")
        records = load_from_parquet(args.input_parquet, args.num_samples)
        train_records = [convert_parquet_record(
            r, i,
            args.context_field, args.question_field, args.answers_field, args.label_field,
            option_mapping, args.raw_prompt_template,
            data_source=os.path.basename(os.path.dirname(args.input_parquet)),
            skip_fol=args.skip_fol_extraction,
            split_name="train",
        ) for i, r in enumerate(records)]
        print(f"[INFO] Loaded {len(train_records)} records")

        validate_data = []
        validate_parquet = os.path.join(os.path.dirname(args.input_parquet), "validate.parquet")
        if os.path.exists(validate_parquet):
            print(f"[INFO] Loading validate set from: {validate_parquet}")
            validate_records = load_from_parquet(validate_parquet, args.num_samples)
            validate_data = [convert_parquet_record(
                r, i,
                args.context_field, args.question_field, args.answers_field, args.label_field,
                option_mapping, args.raw_prompt_template,
                data_source=os.path.basename(os.path.dirname(args.input_parquet)),
                skip_fol=not args.extract_fol_for_validate,
                split_name="validate",
            ) for i, r in enumerate(validate_records)]
        else:
            print("[INFO] No validate.parquet found, skipping validate split")

        # Load test set
        test_parquet = os.path.join(os.path.dirname(args.input_parquet), "test.parquet")
        if os.path.exists(test_parquet):
            print(f"[INFO] Loading test set from: {test_parquet}")
            test_records = load_from_parquet(test_parquet, args.num_samples)
            test_data = [convert_parquet_record(
                r, i,
                args.context_field, args.question_field, args.answers_field, args.label_field,
                option_mapping, args.raw_prompt_template,
                data_source=os.path.basename(os.path.dirname(args.input_parquet)),
                skip_fol=True,
                split_name="test",
            ) for i, r in enumerate(test_records)]
        else:
            print("[INFO] No test.parquet found, creating empty test set")
            test_data = []

        train_data = train_records

    else:
        if not args.dataset:
            parser.error("--dataset is required when --input_parquet is not set")
        print(f"[INFO] Loading from HuggingFace: {args.dataset}")
        dataset = datasets.load_dataset(args.dataset, "default")
        train_split = PRESETS.get(args.preset, {}).get("train_split", "train")
        test_split = PRESETS.get(args.preset, {}).get("test_split", "validation")

        ds_train = dataset[train_split]
        if args.num_samples:
            ds_train = ds_train.select(range(min(args.num_samples, len(ds_train))))
        ds_test = dataset.get(test_split, None)

        def make_map_fn(split):
            def process_fn(example, idx):
                raw_context = example.get(args.context_field, "")
                raw_question = example.get(args.question_field, "")
                raw_answers = example.get(args.answers_field, [])
                raw_label = example.get(args.label_field, 0)

                answers_str = get_option_text(args.answers_field, option_mapping, raw_answers)
                question = build_question(
                    raw_context, raw_question, answers_str, args.raw_prompt_template
                )
                try:
                    solution = option_mapping[int(raw_label)]
                except (ValueError, TypeError):
                    solution = option_mapping[0]

                sample_id = f"{args.preset or 'mcq'}_{idx}"
                return {
                    "data_source": args.preset or args.dataset,
                    "prompt": [{"role": "user", "content": question}],
                    "ability": "logic",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "answer": solution,
                    "raw_prompt": question,
                    "sample_id": sample_id,
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "answer": solution,
                        "question": question,
                        "context": raw_context,
                        "query": raw_question,
                        "options": answers_str,
                    },
                    "fol_metadata": None,
                }
            return process_fn

        train_data = [make_map_fn("train")(ex, i) for i, ex in enumerate(ds_train)]
        test_data = [make_map_fn("test")(ex, i) for i, ex in enumerate(ds_test)] if ds_test else []
        validate_data = []

    # ── FOL extraction ──────────────────────────────────────────────────────
    if args.refresh_declarations_only:
        preprocessor = FOLPreprocessor(
            llm_client=None,
            schema_variant=args.fol_schema_variant,
            schema_validation_retries=args.schema_validation_retries,
        )
        print(f"[INFO] Refreshing z3_declaration_code for {len(train_data)} samples from existing fol_metadata...")
        train_data = extract_fol_batch(
            train_data,
            preprocessor,
            batch_save=50,
            max_retries=args.max_retries,
            verbose=args.verbose,
            refresh_declarations_only=True,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        fol_path = os.path.join(args.output_dir, "fol_metadata.json")
        save_data = {
            s["sample_id"]: FOLMetadata.from_dict(s["fol_metadata"])
            for s in train_data if s.get("fol_metadata")
        }
        save_fol_metadata(save_data, fol_path)
        print(f"[INFO] Saved refreshed FOL metadata to {fol_path}")
    elif not args.skip_fol_extraction:
        api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY", "EMPTY")
        llm_client = LLMClient(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
        )
        preprocessor = FOLPreprocessor(
            llm_client=llm_client,
            schema_variant=args.fol_schema_variant,
            schema_validation_retries=args.schema_validation_retries,
        )
        print(f"[INFO] Extracting FOL metadata for {len(train_data)} samples...")

        train_data = extract_fol_batch(train_data, preprocessor, batch_save=50, max_retries=args.max_retries, verbose=args.verbose)
        if args.extract_fol_for_validate and validate_data:
            print(f"[INFO] Extracting FOL metadata for {len(validate_data)} validate samples...")
            validate_data = extract_fol_batch(
                validate_data,
                preprocessor,
                batch_save=50,
                max_retries=args.max_retries,
                verbose=args.verbose,
            )

        # Save fol_metadata.json
        os.makedirs(args.output_dir, exist_ok=True)
        fol_path = os.path.join(args.output_dir, "fol_metadata.json")
        save_data = {
            s["sample_id"]: FOLMetadata.from_dict(s["fol_metadata"])
            for s in train_data if s.get("fol_metadata")
        }
        if args.extract_fol_for_validate and validate_data:
            save_data.update(
                {
                    s["sample_id"]: FOLMetadata.from_dict(s["fol_metadata"])
                    for s in validate_data if s.get("fol_metadata")
                }
            )
        save_fol_metadata(save_data, fol_path)
        print(f"[INFO] Saved FOL metadata to {fol_path}")

        # Filter out samples without FOL metadata (should already be dropped by extract_fol_batch)
        n_before = len(train_data)
        train_data = [s for s in train_data if s.get("fol_metadata") is not None]
        n_dropped = n_before - len(train_data)
        if n_dropped > 0:
            print(f"[INFO] Filtered: {n_before} -> {len(train_data)} samples ({n_dropped} failed samples dropped)")
        else:
            print(f"[INFO] Filtered: {n_before} samples with valid fol_metadata")

        if args.extract_fol_for_validate and validate_data:
            n_val_before = len(validate_data)
            validate_data = [s for s in validate_data if s.get("fol_metadata") is not None]
            n_val_dropped = n_val_before - len(validate_data)
            if n_val_dropped > 0:
                print(f"[INFO] Validate filtered: {n_val_before} -> {len(validate_data)} samples ({n_val_dropped} failed samples dropped)")
            else:
                print(f"[INFO] Validate filtered: {n_val_before} samples with valid fol_metadata")

    # ── Save parquet ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    print("[INFO] Saving parquet files...")
    train_parquet_data = _make_parquet_safe_records(train_data, args.fol_schema_variant)
    validate_parquet_data = _make_parquet_safe_records(validate_data, args.fol_schema_variant)
    test_parquet_data = _make_parquet_safe_records(test_data, args.fol_schema_variant)
    pd.DataFrame(train_parquet_data).to_parquet(os.path.join(args.output_dir, "train.parquet"))
    if validate_parquet_data:
        pd.DataFrame(validate_parquet_data).to_parquet(os.path.join(args.output_dir, "validate.parquet"))
    pd.DataFrame(test_parquet_data).to_parquet(os.path.join(args.output_dir, "test.parquet"))
    print(f"[INFO] Saved:")
    print(f"       - {args.output_dir}/train.parquet ({len(train_data)} samples)")
    if validate_parquet_data:
        print(f"       - {args.output_dir}/validate.parquet ({len(validate_data)} samples)")
    print(f"       - {args.output_dir}/test.parquet ({len(test_data)} samples)")

    if not args.skip_fol_extraction:
        print(f"       - {args.output_dir}/fol_metadata.json")

    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)
        print(f"[INFO] Copied to HDFS: {args.hdfs_dir}")


if __name__ == "__main__":
    main()
