#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.logiqa_fol_pilot import (
    VLLMChatGenerator,
    answer_correct,
    build_model_prompt,
    extract_step_blocks,
    get_ground_truth,
    get_question_text,
    get_sample_id,
    step_format_correct,
)


BOXED_ANSWER_RE = re.compile(r"\\boxed\{(?:\{\s*([A-Za-z])\s*\}|\s*([A-Za-z])\s*)\}\s*$", re.DOTALL)
DEFAULT_MODEL_PATHS = [
    "/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct",
    "/home/chenzhb/Workspaces/LLMs/Qwen2.5-7B-Instruct",
]


def slugify_model_path(model_path: str) -> str:
    name = Path(model_path).name.lower()
    name = name.replace(".", "_")
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def extract_final_boxed_answer(response_text: str) -> str | None:
    match = BOXED_ANSWER_RE.search(response_text or "")
    if not match:
        return None
    return (match.group(1) or match.group(2)).upper()


def step_blocks_cover_region(text: str) -> bool:
    matches = list(re.finditer(r"<step\b[^>]*>.*?</step>", text or "", re.DOTALL))
    if not matches:
        return False

    cursor = 0
    for match in matches:
        if text[cursor : match.start()].strip():
            return False
        if not step_format_correct(match.group(0)):
            return False
        cursor = match.end()

    return not text[cursor:].strip()


def trajectory_format_correct(response_text: str, valid_choices: str = string.ascii_uppercase) -> bool:
    match = BOXED_ANSWER_RE.search(response_text or "")
    if not match:
        return False
    answer = (match.group(1) or match.group(2)).upper()
    if answer not in {choice.upper() for choice in valid_choices}:
        return False
    return step_blocks_cover_region(response_text[: match.start()])


def evaluate_response(response_text: str, ground_truth: str) -> dict[str, Any]:
    step_blocks = extract_step_blocks(response_text)
    step_records = [
        {
            "step_index": idx + 1,
            "text": step_text,
            "format_correct": step_format_correct(step_text),
        }
        for idx, step_text in enumerate(step_blocks)
    ]
    format_correct_steps = sum(1 for step in step_records if step["format_correct"])
    total_steps = len(step_records)
    return {
        "steps": step_records,
        "total_steps": total_steps,
        "format_correct_steps": format_correct_steps,
        "step_format_ratio": (format_correct_steps / total_steps) if total_steps else 0.0,
        "trajectory_format_correct": trajectory_format_correct(response_text),
        "boxed_answer": extract_final_boxed_answer(response_text),
        "answer_correct": answer_correct(response_text, ground_truth),
    }


def aggregate_metrics(records: list[dict[str, Any]], *, model_path: str) -> dict[str, Any]:
    total_trajectories = len(records)
    total_steps = sum(int(record["total_steps"]) for record in records)
    format_correct_steps = sum(int(record["format_correct_steps"]) for record in records)
    trajectory_format_correct_count = sum(1 for record in records if record["trajectory_format_correct"])
    answer_correct_count = sum(1 for record in records if record["answer_correct"])

    return {
        "model_path": model_path,
        "model_name": Path(model_path).name,
        "total_trajectories": total_trajectories,
        "total_steps": total_steps,
        "step_format_correct_steps": format_correct_steps,
        "step_format_ratio": (format_correct_steps / total_steps) if total_steps else 0.0,
        "trajectory_format_correct": trajectory_format_correct_count,
        "trajectory_format_ratio": (trajectory_format_correct_count / total_trajectories) if total_trajectories else 0.0,
        "answer_correct": answer_correct_count,
        "answer_accuracy": (answer_correct_count / total_trajectories) if total_trajectories else 0.0,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_single_model(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(args.data_path)
    sampled = dataset.sample(n=min(args.num_samples, len(dataset)), random_state=args.seed).reset_index(drop=True)
    sampled.to_parquet(output_dir / "sampled_inputs.parquet")

    prompt_template = Path(args.prompt_path).read_text(encoding="utf-8")
    records = sampled.to_dict("records")
    samples = []
    for idx, record in enumerate(records):
        question_text = get_question_text(record)
        samples.append(
            {
                "sample_id": get_sample_id(record, idx),
                "question_text": question_text,
                "ground_truth": get_ground_truth(record),
                "prompt_text": build_model_prompt(prompt_template, question_text),
            }
        )

    generator = VLLMChatGenerator(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )
    prompts = [generator.format_chat(user=sample["prompt_text"]) for sample in samples]
    responses = generator.generate(
        prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )

    result_records = []
    for sample, response in zip(samples, responses):
        evaluation = evaluate_response(response, sample["ground_truth"])
        result_records.append(
            {
                **sample,
                "response": response,
                **evaluation,
            }
        )

    metrics = aggregate_metrics(result_records, model_path=args.model_path)
    write_jsonl(output_dir / "generations.jsonl", result_records)
    pd.DataFrame(result_records).to_parquet(output_dir / "generations.parquet")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[compare_logiqa_format_qwen25] "
        f"model={metrics['model_name']} "
        f"step_format={metrics['step_format_correct_steps']}/{metrics['total_steps']} "
        f"({metrics['step_format_ratio']:.6f}) "
        f"trajectory_format={metrics['trajectory_format_correct']}/{metrics['total_trajectories']} "
        f"({metrics['trajectory_format_ratio']:.6f}) "
        f"answer={metrics['answer_correct']}/{metrics['total_trajectories']} "
        f"({metrics['answer_accuracy']:.6f})",
        flush=True,
    )
    return metrics


def run_parent(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for model_path in args.model_paths:
        model_output_dir = output_dir / slugify_model_path(model_path)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-model",
            "--model-path",
            model_path,
            "--model-output-dir",
            str(model_output_dir),
            "--data-path",
            args.data_path,
            "--prompt-path",
            args.prompt_path,
            "--num-samples",
            str(args.num_samples),
            "--seed",
            str(args.seed),
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--batch-size",
            str(args.batch_size),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
            "--dtype",
            args.dtype,
        ]
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")
        subprocess.run(cmd, check=True)

        metrics_path = model_output_dir / "metrics.json"
        summary.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("[compare_logiqa_format_qwen25] summary", flush=True)
    for metrics in summary:
        print(
            f"{metrics['model_name']}: "
            f"step_format={metrics['step_format_correct_steps']}/{metrics['total_steps']} "
            f"({metrics['step_format_ratio']:.6f}), "
            f"trajectory_format={metrics['trajectory_format_correct']}/{metrics['total_trajectories']} "
            f"({metrics['trajectory_format_ratio']:.6f}), "
            f"answer={metrics['answer_correct']}/{metrics['total_trajectories']} "
            f"({metrics['answer_accuracy']:.6f})",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LogiQA format ratios for Qwen2.5 instruct models.")
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "logiqa" / "test.parquet"))
    parser.add_argument("--prompt-path", default=str(REPO_ROOT / "prompts" / "premise_conclusion.txt"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "logiqa_format_compare"))
    parser.add_argument("--model-paths", nargs="+", default=DEFAULT_MODEL_PATHS)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--single-model", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-output-dir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single_model:
        if not args.model_path or not args.model_output_dir:
            raise ValueError("--single-model requires --model-path and --model-output-dir")
        run_single_model(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
