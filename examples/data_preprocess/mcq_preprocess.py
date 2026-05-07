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
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

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

    return {
        "context": raw_context,
        "question": raw_question,
        "answers": answers_str,
        "answer": solution,
        "question_full": question,
        "fol_metadata": record.get("fol_metadata", None),
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
) -> Dict:
    """将现有 parquet 记录转换为 verl 训练格式"""
    extracted = extract_fields_from_record(
        record, context_field, question_field, answers_field,
        label_field, option_mapping, raw_prompt_template,
    )

    sample_id = record.get("sample_id", f"mcq_{idx}")
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
            "split": extra_info.get("split", "train") if isinstance(extra_info, dict) else "train",
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
    ):
        self.llm_client = llm_client
        self.default_args = default_args or {
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.8,
        }
        self.fol_verifier = FOLVerifier(llm_client=llm_client) if llm_client else FOLVerifier()

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
            entities = self.fol_verifier.object_extract(
                context=context,
                question=question,
                options=options,
                args=self.default_args,
            )
            predicates = self.fol_verifier.predicate_extract(
                context=context,
                question=question,
                options=options,
                obj_list=entities,
                args=self.default_args,
            )
            z3_declaration_code = self.fol_verifier.generate_z3_declarations(entities)
            z3_function_code = self.fol_verifier.generate_z3_functions(predicates)
            z3_declaration_code = z3_declaration_code + "\n\n" + z3_function_code

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
            print(f"[Error] Failed to extract FOL metadata for {sample_id}: {e}")
            return None


def extract_fol_batch(
    samples: List[Dict],
    preprocessor: FOLPreprocessor,
    batch_save: int = 50,
    max_retries: int = 3,
    verbose: bool = False,
) -> List[Dict]:
    """批量提取 FOL 元数据"""
    import json

    results = []
    for i, sample in enumerate(samples):
        if sample.get("fol_metadata") is not None:
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
        ) for i, r in enumerate(records)]
        print(f"[INFO] Loaded {len(train_records)} records")

        # Load test set
        test_parquet = os.path.join(os.path.dirname(args.input_parquet), "test.parquet")
        if os.path.exists(test_parquet):
            print(f"[INFO] Loading test set from: {test_parquet}")
            test_records = load_from_parquet(test_parquet)
            test_data = [convert_parquet_record(
                r, i,
                args.context_field, args.question_field, args.answers_field, args.label_field,
                option_mapping, args.raw_prompt_template,
                data_source=os.path.basename(os.path.dirname(args.input_parquet)),
                skip_fol=True,
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

    # ── FOL extraction ──────────────────────────────────────────────────────
    if not args.skip_fol_extraction:
        api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY", "EMPTY")
        llm_client = LLMClient(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
        )
        preprocessor = FOLPreprocessor(llm_client=llm_client)
        print(f"[INFO] Extracting FOL metadata for {len(train_data)} samples...")

        train_data = extract_fol_batch(train_data, preprocessor, batch_save=50, max_retries=args.max_retries, verbose=args.verbose)

        # Save fol_metadata.json
        os.makedirs(args.output_dir, exist_ok=True)
        fol_path = os.path.join(args.output_dir, "fol_metadata.json")
        save_data = {
            s["sample_id"]: FOLMetadata.from_dict(s["fol_metadata"])
            for s in train_data if s.get("fol_metadata")
        }
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

    # ── Save parquet ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    print("[INFO] Saving parquet files...")
    pd.DataFrame(train_data).to_parquet(os.path.join(args.output_dir, "train.parquet"))
    pd.DataFrame(test_data).to_parquet(os.path.join(args.output_dir, "test.parquet"))
    print(f"[INFO] Saved:")
    print(f"       - {args.output_dir}/train.parquet ({len(train_data)} samples)")
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
