"""
FOL Metadata Extraction for LogiQA Dataset

功能:
1. 加载 LogiQA 数据集
2. 对每个样本调用 FOL 元数据提取管道
3. 输出带 fol_metadata 字段的 parquet 文件

Usage:
    # 提取少量样本的 FOL 元数据 (用于测试)
    python examples/data_preprocess/logiqa_fol_preprocess.py \
        --local_dir data/logiqa_fol \
        --num_samples 10 \
        --api_key "your-api-key"

    # 跳过 FOL 提取，直接预处理数据 (用于大规模训练)
    python examples/data_preprocess/logiqa_fol_preprocess.py \
        --local_dir data/logiqa_fol \
        --skip_fol_extraction

    # 单独运行 FOL 元数据提取 (后台运行)
    nohup python examples/data_preprocess/logiqa_fol_preprocess.py \
        --local_dir data/logiqa_fol \
        --num_samples 1000 \
        > fol_preprocess.log 2>&1 &
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verl.utils.fol_verifier import (
    FOLMetadata,
    FOLVerifier,
    LLMClient,
    load_fol_metadata,
    save_fol_metadata,
)


def parse_raw_prompt(raw_prompt: str) -> Tuple[str, str, str]:
    """从 raw_prompt 解析出 context, query, options

    Args:
        raw_prompt: 格式如 "<Context>...</Context><Question>...</Question><Options>...</Options>"

    Returns:
        (context, query, options) 元组
    """
    context_match = re.search(r'<Context>(.*?)</Context>', raw_prompt, re.DOTALL)
    question_match = re.search(r'<Question>(.*?)</Question>', raw_prompt, re.DOTALL)
    options_match = re.search(r'<Options>(.*?)</Options>', raw_prompt, re.DOTALL)

    context = context_match.group(1).strip() if context_match else ""
    query = question_match.group(1).strip() if question_match else ""
    options = options_match.group(1).strip() if options_match else ""

    return context, query, options


def load_from_parquet(parquet_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """从现有 parquet 文件加载数据

    Args:
        parquet_path: parquet 文件路径
        num_samples: 限制加载的样本数量（从开头取）

    Returns:
        数据记录列表
    """
    df = pd.read_parquet(parquet_path)
    if num_samples:
        df = df.head(num_samples)
    return df.to_dict("records")


def convert_parquet_record(record: Dict, idx: int) -> Dict:
    """将现有 parquet 记录转换为统一格式

    Args:
        record: 原始 parquet 记录
        idx: 样本索引

    Returns:
        转换后的记录
    """
    raw_prompt = record.get("raw_prompt", "")
    if not isinstance(raw_prompt, str):
        raw_prompt = str(raw_prompt) if raw_prompt else ""

    context, query, options = parse_raw_prompt(raw_prompt)

    sample_id = record.get("sample_id", f"logiqa_{idx}")
    # 尝试从 extra_info 获取 index
    extra_info = record.get("extra_info", {})
    if isinstance(extra_info, dict) and "index" in extra_info:
        sample_id = f"logiqa_{extra_info['index']}"

    answer = record.get("answer", "")
    if not isinstance(answer, str):
        answer = str(answer) if answer else ""

    return {
        "data_source": record.get("data_source", "lucasmccabe/logiqa"),
        "prompt": record.get("prompt", [{"role": "user", "content": raw_prompt}]),
        "ability": record.get("ability", "logical_reasoning"),
        "reward_model": record.get("reward_model", {"style": "rule", "ground_truth": answer}),
        "answer": answer,
        "raw_prompt": raw_prompt,
        "sample_id": sample_id,
        "extra_info": {
            "split": extra_info.get("split", "train") if isinstance(extra_info, dict) else "train",
            "index": extra_info.get("index", idx) if isinstance(extra_info, dict) else idx,
            "answer": answer,
            "question": context + query,
            "context": context,
            "query": query,
            "options": options,
        },
        "fol_metadata": record.get("fol_metadata", None),
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

        # 创建 FOL 验证器
        if llm_client:
            self.fol_verifier = FOLVerifier(llm_client=llm_client)
        else:
            self.fol_verifier = FOLVerifier()

    def extract_fol_metadata(
        self,
        context: str,
        question: str,
        options: str,
        sample_id: str,
    ) -> Optional[FOLMetadata]:
        """提取单个样本的 FOL 元数据

        需要 LLM 调用:
        - rephrase()         1次
        - object_extract()   1次 (OutputSchema)
        - predicate_extract()1次 (OutputSchema)

        Returns:
            FOLMetadata 对象，失败时返回 None
        """
        if self.llm_client is None:
            print(f"[Warning] No LLM client, skipping FOL extraction for {sample_id}")
            return None

        try:
            # Step 1: rephrase - 重述问题
            rephrased_context = self.fol_verifier.rephrase(
                context=context,
                question=question,
                options=options,
                args=self.default_args,
            )

            # Step 2: object_extract - 提取实体
            entities = self.fol_verifier.object_extract(
                context=context,
                question=question,
                options=options,
                args=self.default_args,
            )

            # Step 3: predicate_extract - 提取谓词
            predicates = self.fol_verifier.predicate_extract(
                context=context,
                question=question,
                options=options,
                obj_list=entities,
                args=self.default_args,
            )

            # Step 4: 生成 Z3 declarations (纯计算)
            z3_declaration_code = self.fol_verifier.generate_z3_declarations(entities)
            z3_function_code = self.fol_verifier.generate_z3_functions(predicates)
            z3_declaration_code = z3_declaration_code + "\n\n" + z3_function_code

            metadata = FOLMetadata(
                sample_id=sample_id,
                rephrased_context=rephrased_context,
                entities=entities,
                predicates=predicates,
                z3_declaration_code=z3_declaration_code,
                ground_truth="",  # 稍后在 make_map_fn 中填充
                axioms=[],
            )

            return metadata

        except Exception as e:
            print(f"[Error] Failed to extract FOL metadata for {sample_id}: {e}")
            return None


def make_base_map_fn(split: str):
    """创建基础数据集 map 函数（不包含 FOL 元数据）"""
    option_mapping = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def process_fn(example, idx):
        context = example.get("context", "")
        question_raw = example.get("query", "")
        answer_raw = example.get("options", [])
        solution = option_mapping[int(example.get("correct_option", 0))]

        answers = "\n\n".join([
            f"Option ({option_mapping[i]}):{answer_raw[i]}"
            for i in range(len(answer_raw))
        ])
        question = (
            "<Context>" + context + "</Context>" +
            "<Question>" + question_raw + "</Question>" +
            "<Options>" + answers + "</Options>"
        )

        sample_id = f"logiqa_{idx}"

        data = {
            "data_source": "lucasmccabe/logiqa",
            "prompt": [{"role": "user", "content": question}],
            "ability": "logical_reasoning",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "answer": solution,
            "raw_prompt": question,
            "sample_id": sample_id,
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": solution,
                "question": context + question_raw,
                "context": context,
                "query": question_raw,
                "options": answers,
            },
            # FOL 元数据字段 (稍后填充)
            "fol_metadata": None,
        }

        return data

    return process_fn


def extract_fol_metadata_batch(
    samples: List[Dict],
    preprocessor: FOLPreprocessor,
    batch_size: int = 100,
    checkpoint_path: Optional[str] = None,
    max_retries: int = 3,
) -> List[Dict]:
    """批量提取 FOL 元数据

    Args:
        samples: 样本列表
        preprocessor: FOL 预处理器
        batch_size: 每多少个样本保存一次 checkpoint
        checkpoint_path: checkpoint 文件路径
        max_retries: 每个样本最大重试次数

    Returns:
        带有 fol_metadata 的样本列表
    """
    results = []
    for i, sample in enumerate(samples):
        if sample.get("fol_metadata") is not None:
            # 已有元数据，跳过
            results.append(sample)
            continue

        # 提取元数据
        extra_info = sample.get("extra_info", {})
        context = extra_info.get("context", "")
        query = extra_info.get("query", "")
        options = extra_info.get("options", "")
        sample_id = sample.get("sample_id", f"sample_{i}")

        # 确保 context, query, options 都是字符串
        if not isinstance(context, str):
            context = str(context) if context else ""
        if not isinstance(query, str):
            query = str(query) if query else ""
        if not isinstance(options, str):
            options = str(options) if options else ""

        print(f"[{i+1}/{len(samples)}] Processing {sample_id}...")

        fol_metadata = None
        for attempt in range(max_retries):
            try:
                fol_metadata = preprocessor.extract_fol_metadata(
                    context=context,
                    question=query,
                    options=options,
                    sample_id=sample_id,
                )
                if fol_metadata is not None:
                    break  # 成功，跳出重试循环
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[Retry {attempt+2}/{max_retries}] Failed for {sample_id}: {e}")
                else:
                    print(f"[Error] Failed to extract FOL metadata for {sample_id} after {max_retries} attempts: {e}")

        if fol_metadata:
            # 填充 ground_truth
            fol_metadata.ground_truth = sample.get("answer", "")
            sample["fol_metadata"] = fol_metadata.to_dict()
        else:
            sample["fol_metadata"] = None

        results.append(sample)

        # 每 batch_size 个样本打印进度
        if (i + 1) % batch_size == 0:
            print(f"[Progress] Processed {i + 1}/{len(samples)} samples")
            if checkpoint_path:
                _save_checkpoint(results, checkpoint_path, i)

    return results


def _save_checkpoint(results: List[Dict], output_path: str, step: int) -> None:
    """保存检查点"""
    checkpoint_path = f"{output_path}.checkpoint_{step}.json"

    # 清理结果，移除不可 JSON 序列化的对象
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            # 转换为字符串
            return str(obj)

    cleaned_results = clean_for_json(results)

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        import json
        json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="FOL Metadata Extraction for LogiQA")
    parser.add_argument(
        "--local_dir",
        default="/home/chenzhb/Workspaces/verl/data/logiqa_fol",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS directory for processed data (optional)"
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key for LLM service"
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:4869/v1",
        help="Base URL for LLM service"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-3b",
        help="Model name for LLM"
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=None,
        help="Number of training samples to process (default: all)"
    )
    parser.add_argument(
        "--input_parquet",
        default=None,
        help="Path to existing parquet file (reuse existing data instead of downloading from HuggingFace)"
    )
    parser.add_argument(
        "--skip_fol_extraction",
        action="store_true",
        help="Skip FOL metadata extraction (for large-scale preprocessing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Checkpoint frequency during extraction"
    )
    parser.add_argument(
        "--skip_output_parquet",
        action="store_true",
        help="Skip saving parquet files (only generate fol_metadata.json)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts for each sample's FOL metadata extraction"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.local_dir, exist_ok=True)

    # 加载数据：优先从现有 parquet 读取
    if args.input_parquet and os.path.exists(args.input_parquet):
        print(f"[INFO] Loading from existing parquet: {args.input_parquet}")
        existing_data = load_from_parquet(args.input_parquet, args.num_train_samples)
        train_data = [convert_parquet_record(record, idx) for idx, record in enumerate(existing_data)]
        print(f"[INFO] Loaded {len(train_data)} samples from parquet")

        # 测试集：尝试从同一目录加载
        test_parquet = os.path.join(os.path.dirname(args.input_parquet), "test.parquet")
        if os.path.exists(test_parquet):
            print(f"[INFO] Loading test set from: {test_parquet}")
            test_records = load_from_parquet(test_parquet)
            test_data = [convert_parquet_record(record, idx) for idx, record in enumerate(test_records)]
        else:
            print("[INFO] No test.parquet found, creating empty test set")
            test_data = []

    else:
        # 从 HuggingFace 加载（原有逻辑）
        print("[INFO] Loading LogiQA dataset from HuggingFace...")
        dataset = datasets.load_dataset("lucasmccabe/logiqa", "default")

        # 处理训练集
        train_size = len(dataset["train"])
        if args.num_train_samples is not None:
            train_size = min(args.num_train_samples, train_size)
        train_dataset = dataset["train"].select(range(train_size))
        test_dataset = dataset["validation"]

        # 创建基础数据
        print("[INFO] Creating base dataset...")
        train_data = [make_base_map_fn("train")(ex, i) for i, ex in enumerate(train_dataset)]
        test_data = [make_base_map_fn("test")(ex, i) for i, ex in enumerate(test_dataset)]

    # 提取 FOL 元数据 (如果不跳过)
    if not args.skip_fol_extraction:
        # 创建 LLM 客户端
        api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY", "EMPTY")
        llm_client = LLMClient(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
        )

        # 创建预处理器
        preprocessor = FOLPreprocessor(llm_client=llm_client)

        print(f"[INFO] Extracting FOL metadata for {len(train_data)} samples...")
        print("[INFO] This may take a while due to LLM calls...")

        # 提取训练集 FOL 元数据
        train_with_fol = extract_fol_metadata_batch(
            train_data,
            preprocessor,
            batch_size=args.batch_size,
            checkpoint_path=os.path.join(args.local_dir, "train_checkpoint"),
            max_retries=args.max_retries,
        )

        # 保存 FOL 元数据到单独的文件
        fol_metadata_path = os.path.join(args.local_dir, "fol_metadata.json")
        fol_data = [
            {"sample_id": s.get("sample_id"), "fol_metadata": s.get("fol_metadata")}
            for s in train_with_fol
        ]
        save_fol_metadata({s.get("sample_id"): FOLMetadata.from_dict(s["fol_metadata"]) for s in train_with_fol if s.get("fol_metadata")}, fol_metadata_path)
        print(f"[INFO] Saved FOL metadata to {fol_metadata_path}")

        # 过滤掉没有 fol_metadata 的样本
        train_with_fol = [s for s in train_with_fol if s.get("fol_metadata") is not None]
        print(f"[INFO] Filtered to {len(train_with_fol)} samples with valid fol_metadata")

    else:
        train_with_fol = train_data

    # 保存数据集为 parquet（如果不跳过）
    if not args.skip_output_parquet:
        print("[INFO] Saving dataset to parquet...")
        train_df = pd.DataFrame(train_with_fol)
        train_df.to_parquet(os.path.join(args.local_dir, "train.parquet"))

        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(args.local_dir, "test.parquet"))

        print(f"[INFO] Dataset saved to {args.local_dir}")
        print(f"       - {args.local_dir}/train.parquet ({len(train_with_fol)} samples)")
        print(f"       - {args.local_dir}/test.parquet")
    else:
        print("[INFO] Skipped parquet output (--skip_output_parquet)")

    if not args.skip_fol_extraction:
        print(f"       - {args.local_dir}/fol_metadata.json")

    # 复制到 HDFS (如果指定)
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print(f"[INFO] Copied to HDFS: {args.hdfs_dir}")


if __name__ == "__main__":
    main()
