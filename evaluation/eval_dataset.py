from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import datasets


INSTRUCTION_FOLLOWING = 'Please reason step by step with steps separated by "\\n\\n", and put the index of the correct answer within \\\\boxed{{}}.'
OPTION_MAPPING = list(string.ascii_uppercase)


@dataclass
class DatasetLoadConfig:
    max_samples: int = 0
    local_data_dir: str = "./data"
    prefer_local_parquet: bool = True
    pubmedqa_data_dir: str = "./data/pubmedqa_origin/data"
    mathqa_data_dir: str = "./data/MathQA"
    qa4mre_dataset_name: str = "community-datasets/qa4mre"
    qa4mre_fallback_dataset_name: str = "qa4mre"
    qa4mre_config_name: str = "2013.main.EN"
    qa4mre_split: str = "train"
    gpqa_dataset_name: str = "Idavidrein/gpqa"
    gpqa_split: str = "train"
    gpqa_seed: int = 42


EvalRecord = dict[str, Any]


LOCAL_PARQUET_PATHS = {
    "logiqa": "logiqa/test.parquet",
    "reclor": "reclor/test.parquet",
    "arlsat": "arlsat/test.parquet",
    "pubmedqa": "pubmedqa/test.parquet",
    "medqa": "medqa/test.parquet",
    "mathqa": "mathqa/test.parquet",
    "mathqa_challenge": "mathqa/challenge_test.parquet",
    "gpqa_diamond": "gpqa/gpqa_diamond/test.parquet",
    "gpqa_main": "gpqa/gpqa_main/test.parquet",
    "openbookqa": "openbookqa/test.parquet",
    "truthfulqa": "truthfulqa/test.parquet",
    "qa4mre": "qa4mre/test.parquet",
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def _load_local_parquet_records(dataset_name: str, config: DatasetLoadConfig) -> list[EvalRecord] | None:
    rel_path = LOCAL_PARQUET_PATHS.get(dataset_name)
    if not rel_path:
        return None
    path = Path(config.local_data_dir) / rel_path
    if not path.is_file():
        return None

    import pandas as pd

    df = pd.read_parquet(path)
    records = [_normalize_local_record(dataset_name, idx, _json_safe(row)) for idx, row in enumerate(df.to_dict(orient="records"))]
    return _limit(records, config.max_samples)


def _normalize_local_record(dataset_name: str, idx: int, record: EvalRecord) -> EvalRecord:
    reward_model = record.get("reward_model") or {}
    extra_info = record.get("extra_info") or {}
    answer = record.get("answer")
    if answer is None:
        answer = reward_model.get("ground_truth", extra_info.get("answer"))
        record["answer"] = answer

    if not reward_model:
        record["reward_model"] = {"style": "rule", "ground_truth": answer}
    elif "ground_truth" not in reward_model:
        reward_model["ground_truth"] = answer
        record["reward_model"] = reward_model

    if "raw_prompt" not in record or record.get("raw_prompt") is None:
        raw_prompt = extra_info.get("question")
        if raw_prompt is None:
            prompt = record.get("prompt") or []
            if prompt and isinstance(prompt, list):
                raw_prompt = prompt[-1].get("content", "")
        record["raw_prompt"] = raw_prompt or ""

    if "prompt" not in record or not record.get("prompt"):
        record["prompt"] = [{"role": "user", "content": record.get("raw_prompt", "")}]

    if "sample_id" not in record or record.get("sample_id") is None:
        record["sample_id"] = f"{dataset_name}_{idx}"

    record.setdefault("data_source", dataset_name)
    record.setdefault("extra_info", extra_info)
    return record


def _limit(records: list[EvalRecord], max_samples: int) -> list[EvalRecord]:
    if max_samples and max_samples > 0:
        return records[:max_samples]
    return records


def _option_label(idx: int) -> str:
    if idx >= len(OPTION_MAPPING):
        raise ValueError(f"Too many answer options: {idx + 1}")
    return OPTION_MAPPING[idx]


def _format_letter_options(options: list[str], *, prefix: str = "") -> str:
    return "\n".join([f"{prefix}({_option_label(i)}): {text}.\n" for i, text in enumerate(options)])


def _record(
    *,
    data_source: str,
    sample_id: str,
    raw_prompt: str,
    solution: Any,
    split: str,
    index: int,
    ability: str = "logic",
    extra_info: dict[str, Any] | None = None,
) -> EvalRecord:
    extra = {
        "split": split,
        "index": index,
        "answer": solution,
        "question": raw_prompt,
    }
    if extra_info:
        extra.update(extra_info)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": raw_prompt}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": solution},
        "answer": solution,
        "raw_prompt": raw_prompt,
        "sample_id": sample_id,
        "extra_info": extra,
    }


def load_logiqa(config: DatasetLoadConfig) -> list[EvalRecord]:
    ds = datasets.load_dataset("lucasmccabe/logiqa", "default")["test"]
    records = []
    for idx, example in enumerate(ds):
        context = example["context"]
        question_raw = example["query"]
        answer_raw = list(example["options"])
        solution = _option_label(int(example["correct_option"]))
        answers = _format_letter_options(answer_raw)
        raw_prompt = f"<Context>{context}</Context><Question>{question_raw}</Question><Options>{answers}</Options>"
        records.append(_record(data_source="logiqa", sample_id=f"logiqa_{idx}", raw_prompt=raw_prompt, solution=solution, split="test", index=idx))
    return _limit(records, config.max_samples)


def load_reclor(config: DatasetLoadConfig) -> list[EvalRecord]:
    ds = datasets.load_dataset("metaeval/reclor", "default")["validation"]
    records = []
    for idx, example in enumerate(ds):
        context = example["context"]
        question_raw = example["question"]
        answer_raw = list(example["answers"])
        solution = _option_label(int(example["label"]))
        answers = _format_letter_options(answer_raw)
        raw_prompt = f"<Context>{context}</Context><Question>{question_raw}</Question><Options>{answers}</Options>"
        records.append(_record(data_source="reclor", sample_id=f"reclor_{idx}", raw_prompt=raw_prompt, solution=solution, split="test", index=idx))
    return _limit(records, config.max_samples)


def load_arlsat(config: DatasetLoadConfig) -> list[EvalRecord]:
    ds = datasets.load_dataset("olegbask/AR-LSAT", "default")["validation"]
    records = []
    for idx, example in enumerate(ds):
        context = example["context"]
        question_raw = example["question"]
        answer_raw = list(example["answers"])
        solution = str(example["label"])
        answers = "\n".join([f"{i}: {answer_raw[i]}.\n" for i in range(len(answer_raw))])
        raw_prompt = f"{context}\n\n{question_raw}\n\n{answers}"
        records.append(
            _record(
                data_source="arlsat",
                sample_id=f"arlsat_{idx}",
                raw_prompt=raw_prompt,
                solution=solution,
                split="test",
                index=idx,
                extra_info={"options": answer_raw},
            )
        )
    return _limit(records, config.max_samples)


PUBMEDQA_ANSWER_TO_OPTION = {"yes": "A", "no": "B", "maybe": "C"}
PUBMEDQA_OPTIONS = [("A", "yes"), ("B", "no"), ("C", "maybe")]


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_pubmedqa_context(example: dict[str, Any]) -> str:
    contexts = example.get("CONTEXTS") or []
    labels = example.get("LABELS") or []
    formatted = []
    for idx, context in enumerate(contexts):
        label = labels[idx] if idx < len(labels) else None
        formatted.append(f"{label}: {context}" if label else str(context))
    return "\n\n".join(formatted)


def load_pubmedqa(config: DatasetLoadConfig) -> list[EvalRecord]:
    data_dir = Path(config.pubmedqa_data_dir)
    test_set = _load_json(data_dir / "test_set.json")
    ground_truths = _load_json(data_dir / "test_ground_truth.json")
    option_text = "\n\n".join([f"({label}): {text}" for label, text in PUBMEDQA_OPTIONS])
    records = []
    for idx, (pmid, example) in enumerate(test_set.items()):
        answer_text = str(ground_truths[pmid]).lower()
        solution = PUBMEDQA_ANSWER_TO_OPTION[answer_text]
        context = _format_pubmedqa_context(example)
        question_raw = example.get("QUESTION", "")
        raw_prompt = f"<Context>{context}</Context><Question>{question_raw}</Question><Options>{option_text}</Options>"
        records.append(
            _record(
                data_source="pubmedqa",
                sample_id=f"pubmedqa_{pmid}",
                raw_prompt=raw_prompt,
                solution=solution,
                split="test",
                index=idx,
                extra_info={"pmid": pmid, "answer_text": answer_text, "contexts": example.get("CONTEXTS") or []},
            )
        )
    return _limit(records, config.max_samples)


def load_medqa(config: DatasetLoadConfig) -> list[EvalRecord]:
    ds = datasets.load_dataset("awinml/medqa", "questions")["test"]
    records = []
    for idx, example in enumerate(ds):
        question_raw = example["question"]
        options = example["options"]
        solution = example["answer_idx"]
        answers = "".join([f"({key}):{value}\n" for key, value in options.items()])
        raw_prompt = f"<Question>{question_raw}</Question>\n\n<Options>{answers}</Options>"
        records.append(_record(data_source="awinml/medqa", sample_id=f"medqa_{idx}", raw_prompt=raw_prompt, solution=solution, split="test", index=idx))
    return _limit(records, config.max_samples)


MATHQA_OPTION_LABELS = ["a", "b", "c", "d", "e"]
MATHQA_OPTION_MAPPING = {label: label.upper() for label in MATHQA_OPTION_LABELS}
MATHQA_OPTION_RE = re.compile(r"(?i)([a-e])\s*\)")
MATHQA_DUPLICATE_OPTION_RE = re.compile(r"(?i)\b([a-e])\s*\)\s*\1\s*\)")


def _clean_mathqa_option_text(label: str, text: str) -> str:
    text = text.strip(" ,")
    duplicate_label_re = re.compile(rf"(?i)^{label}\s*\)\s*")
    while True:
        cleaned = duplicate_label_re.sub("", text, count=1).strip(" ,")
        if cleaned == text:
            return cleaned
        text = cleaned


def _parse_mathqa_options(options_raw: str, idx: int) -> list[tuple[str, str]]:
    normalized_options = options_raw
    while True:
        cleaned = MATHQA_DUPLICATE_OPTION_RE.sub(r"\1 )", normalized_options)
        if cleaned == normalized_options:
            break
        normalized_options = cleaned
    matches = list(MATHQA_OPTION_RE.finditer(normalized_options))
    options: dict[str, str] = {}
    for match_idx, match in enumerate(matches):
        label = match.group(1).lower()
        if label in options:
            continue
        start = match.end()
        end = matches[match_idx + 1].start() if match_idx + 1 < len(matches) else len(normalized_options)
        options[label] = _clean_mathqa_option_text(label, normalized_options[start:end])
    missing = [label for label in MATHQA_OPTION_LABELS if label not in options or not options[label]]
    if missing:
        raise ValueError(f"Failed to parse MathQA options for sample {idx}: missing={missing}")
    return [(MATHQA_OPTION_MAPPING[label], options[label]) for label in MATHQA_OPTION_LABELS]


def _load_mathqa_file(config: DatasetLoadConfig, filename: str, split: str, data_source: str) -> list[EvalRecord]:
    examples = _load_json(Path(config.mathqa_data_dir) / filename)
    records = []
    for idx, example in enumerate(examples):
        options = _parse_mathqa_options(example["options"], idx)
        solution = MATHQA_OPTION_MAPPING[str(example["correct"]).lower()]
        question_raw = example["Problem"].strip()
        answers = "\n\n".join([f"Option ({label}): {text}" for label, text in options])
        raw_prompt = f"<Question>{question_raw}</Question><Options>{answers}</Options>"
        records.append(
            _record(
                data_source=data_source,
                sample_id=f"{data_source}_{idx}",
                raw_prompt=raw_prompt,
                solution=solution,
                split=split,
                index=idx,
                ability="math",
                extra_info={
                    "category": example.get("category"),
                    "rationale": example.get("Rationale"),
                    "annotated_formula": example.get("annotated_formula"),
                    "linear_formula": example.get("linear_formula"),
                    "raw_options": example["options"],
                    "options": [text for _, text in options],
                },
            )
        )
    return _limit(records, config.max_samples)


def load_mathqa(config: DatasetLoadConfig) -> list[EvalRecord]:
    return _load_mathqa_file(config, "test.json", "test", "mathqa")


def load_mathqa_challenge(config: DatasetLoadConfig) -> list[EvalRecord]:
    return _load_mathqa_file(config, "challenge_test.json", "challenge_test", "mathqa_challenge")


def _load_gpqa(config: DatasetLoadConfig, config_name: str) -> list[EvalRecord]:
    ds = datasets.load_dataset(config.gpqa_dataset_name, config_name)[config.gpqa_split]
    records = []
    for idx, example in enumerate(ds):
        options = [
            {"text": str(example["Correct Answer"]).strip(), "is_correct": True},
            {"text": str(example["Incorrect Answer 1"]).strip(), "is_correct": False},
            {"text": str(example["Incorrect Answer 2"]).strip(), "is_correct": False},
            {"text": str(example["Incorrect Answer 3"]).strip(), "is_correct": False},
        ]
        rng = random.Random(f"{config.gpqa_seed}:{config_name}:{idx}")
        rng.shuffle(options)
        correct_indices = [i for i, option in enumerate(options) if option["is_correct"]]
        solution = _option_label(correct_indices[0])
        answers = "\n\n".join([f"Option ({_option_label(i)}): {option['text']}" for i, option in enumerate(options)])
        question_raw = str(example["Question"]).strip()
        raw_prompt = f"<Question>{question_raw}</Question><Options>{answers}</Options>"
        records.append(
            _record(
                data_source=config_name,
                sample_id=f"gpqa_{config_name}_{idx}",
                raw_prompt=raw_prompt,
                solution=solution,
                split="test",
                index=idx,
                extra_info={
                    "config": config_name,
                    "record_id": example.get("Record ID"),
                    "domain": example.get("High-level domain"),
                    "subdomain": example.get("Subdomain"),
                    "answer_text": str(example["Correct Answer"]).strip(),
                    "incorrect_answers": [
                        str(example["Incorrect Answer 1"]).strip(),
                        str(example["Incorrect Answer 2"]).strip(),
                        str(example["Incorrect Answer 3"]).strip(),
                    ],
                    "options": [option["text"] for option in options],
                    "seed": config.gpqa_seed,
                },
            )
        )
    return _limit(records, config.max_samples)


def load_gpqa_diamond(config: DatasetLoadConfig) -> list[EvalRecord]:
    return _load_gpqa(config, "gpqa_diamond")


def load_gpqa_main(config: DatasetLoadConfig) -> list[EvalRecord]:
    return _load_gpqa(config, "gpqa_main")


def load_openbookqa(config: DatasetLoadConfig) -> list[EvalRecord]:
    ds = datasets.load_dataset("allenai/openbookqa", "main")["test"]
    records = []
    for idx, example in enumerate(ds):
        question_raw = example["question_stem"]
        answer_raw = list(example["choices"]["text"])
        solution = example["answerKey"]
        answers = "\n\n".join([f"({_option_label(i)}){answer_raw[i]}." for i in range(len(answer_raw))])
        raw_prompt = f"<Question>{question_raw}</Question>\n\n<Options>{answers}</Options>"
        records.append(_record(data_source="allenai/openbookqa", sample_id=f"openbookqa_{idx}", raw_prompt=raw_prompt, solution=solution, split="test", index=idx))
    return _limit(records, config.max_samples)


def load_truthfulqa(config: DatasetLoadConfig) -> list[EvalRecord]:
    ds = datasets.load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    records = []
    for idx, example in enumerate(ds):
        question_raw = example["question"]
        choices = list(example["mc1_targets"]["choices"])
        labels = list(example["mc1_targets"]["labels"])
        positive = [i for i, label in enumerate(labels) if int(label) == 1]
        solution = _option_label(positive[0])
        answers = "\n\n".join([f"Option ({_option_label(i)}): {choice}" for i, choice in enumerate(choices)])
        raw_prompt = f"<Question>{question_raw}</Question>\n\n<Options>{answers}</Options>"
        records.append(
            _record(
                data_source="truthfulqa",
                sample_id=f"truthfulqa_{idx}",
                raw_prompt=raw_prompt,
                solution=solution,
                split="test",
                index=idx,
                extra_info={"choices": choices, "labels": labels},
            )
        )
    return _limit(records, config.max_samples)


def _load_qa4mre_dataset(config: DatasetLoadConfig):
    try:
        return datasets.load_dataset(config.qa4mre_dataset_name, config.qa4mre_config_name)
    except Exception:
        return datasets.load_dataset(config.qa4mre_fallback_dataset_name, config.qa4mre_config_name)


def load_qa4mre(config: DatasetLoadConfig) -> list[EvalRecord]:
    ds = _load_qa4mre_dataset(config)[config.qa4mre_split]
    records = []
    normalized_config_name = config.qa4mre_config_name.replace(".", "_")
    for idx, example in enumerate(ds):
        answer_options = [(str(answer_id), str(answer_text)) for answer_id, answer_text in zip(example["answer_options"]["answer_id"], example["answer_options"]["answer_str"], strict=True)]
        correct_answer_id = str(example["correct_answer_id"])
        solution_idx = next(i for i, (answer_id, _) in enumerate(answer_options) if answer_id == correct_answer_id)
        solution = _option_label(solution_idx)
        answers = "\n\n".join([f"Option ({_option_label(i)}): {answer_text}" for i, (_, answer_text) in enumerate(answer_options)])
        context = example["document_str"].strip()
        question_raw = example["question_str"].strip()
        raw_prompt = f"<Context>{context}</Context><Question>{question_raw}</Question><Options>{answers}</Options>"
        records.append(
            _record(
                data_source="qa4mre",
                sample_id=f"qa4mre_{normalized_config_name}_{idx}",
                raw_prompt=raw_prompt,
                solution=solution,
                split="test",
                index=idx,
                extra_info={
                    "topic_id": example["topic_id"],
                    "topic_name": example["topic_name"],
                    "test_id": example["test_id"],
                    "document_id": example["document_id"],
                    "question_id": example["question_id"],
                    "answer_id": correct_answer_id,
                    "answer_text": example["correct_answer_str"],
                    "answer_option_ids": [answer_id for answer_id, _ in answer_options],
                    "answer_options": [answer_text for _, answer_text in answer_options],
                },
            )
        )
    return _limit(records, config.max_samples)


SUPPORTED_DATASETS: dict[str, Callable[[DatasetLoadConfig], list[EvalRecord]]] = {
    "logiqa": load_logiqa,
    "reclor": load_reclor,
    "arlsat": load_arlsat,
    "pubmedqa": load_pubmedqa,
    "medqa": load_medqa,
    "mathqa": load_mathqa,
    "mathqa_challenge": load_mathqa_challenge,
    "gpqa_diamond": load_gpqa_diamond,
    "gpqa_main": load_gpqa_main,
    "openbookqa": load_openbookqa,
    "truthfulqa": load_truthfulqa,
    "qa4mre": load_qa4mre,
}


def load_eval_records(dataset_name: str, config: DatasetLoadConfig) -> list[EvalRecord]:
    name = dataset_name.strip()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {name!r}. Available: {sorted(SUPPORTED_DATASETS)}")
    if config.prefer_local_parquet:
        local_records = _load_local_parquet_records(name, config)
        if local_records is not None:
            return local_records
    return SUPPORTED_DATASETS[name](config)


def _parse_dataset_list(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--local_data_dir", default="./data")
    parser.add_argument("--disable_local_parquet", action="store_true")
    parser.add_argument("--pubmedqa_data_dir", default="./data/pubmedqa_origin/data")
    parser.add_argument("--mathqa_data_dir", default="./data/MathQA")
    parser.add_argument("--gpqa_seed", type=int, default=42)
    args = parser.parse_args()

    config = DatasetLoadConfig(
        max_samples=args.max_samples,
        local_data_dir=args.local_data_dir,
        prefer_local_parquet=not args.disable_local_parquet,
        pubmedqa_data_dir=args.pubmedqa_data_dir,
        mathqa_data_dir=args.mathqa_data_dir,
        gpqa_seed=args.gpqa_seed,
    )
    summary = {}
    for dataset_name in _parse_dataset_list(args.datasets):
        records = load_eval_records(dataset_name, config)
        summary[dataset_name] = {
            "count": len(records),
            "answers": dict(Counter(str(record["answer"]) for record in records)),
            "config": asdict(config),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
