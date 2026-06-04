from __future__ import annotations

import argparse
import copy
import json
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from verl.utils.reward_score import default_compute_score

try:
    from evaluation.call_openai_api import ApiChoice, OpenAICompatibleClient, SamplingConfig, load_api_config, merge_sampling_config
    from evaluation.eval_dataset import DatasetLoadConfig, SUPPORTED_DATASETS, load_eval_records
except ModuleNotFoundError:
    from call_openai_api import ApiChoice, OpenAICompatibleClient, SamplingConfig, load_api_config, merge_sampling_config
    from eval_dataset import DatasetLoadConfig, SUPPORTED_DATASETS, load_eval_records


STRICT_BOXED_RE = re.compile(r"\\boxed\{\{?\s*([A-Za-z0-9]+)\s*\}?\}", re.IGNORECASE)
RELAXED_BOXED_RE = re.compile(r"\\boxed\{+\s*\(?\s*([A-Za-z0-9]+)\s*\)?\s*\}+", re.IGNORECASE)
HASH_ANSWER_RE = re.compile(r"####\s*\(?\s*([A-Za-z0-9]+)\s*\)?", re.IGNORECASE)
TEXT_ANSWER_RE = re.compile(
    r"(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)\s*\(?\s*([A-E0-9])\s*\)?",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"(<step>.*?</step>)", re.DOTALL)
THREAD_LOCAL = threading.local()


def _parse_dataset_list(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


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


def load_prompt_instruction(prompt_path: Optional[str]) -> Optional[str]:
    if prompt_path is None or str(prompt_path).strip() == "" or str(prompt_path).strip().lower() == "null":
        return None
    path = Path(prompt_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt instruction file not found: {path}")
    return path.read_text(encoding="utf-8")


def inject_prompt_instruction(messages: list[dict[str, str]], prompt_instruction: Optional[str]) -> list[dict[str, str]]:
    copied = [dict(message) for message in messages]
    if not prompt_instruction:
        return copied
    for message in reversed(copied):
        if message.get("role") == "user":
            message["content"] = prompt_instruction + "\n\n" + message.get("content", "")
            break
    return copied


def extract_strict_answer(text: str) -> Optional[str]:
    matches = STRICT_BOXED_RE.findall(text or "")
    if not matches:
        return None
    return str(matches[-1]).strip().upper()


def extract_relaxed_answer(text: str) -> Optional[str]:
    text = text or ""
    for pattern in (RELAXED_BOXED_RE, HASH_ANSWER_RE, TEXT_ANSWER_RE):
        matches = pattern.findall(text)
        if matches:
            return str(matches[-1]).strip().upper()
    return None


def score_relaxed_response(data_source: str, text: str, ground_truth: Any) -> float:
    pred = extract_relaxed_answer(text)
    if pred is not None:
        return 1.0 if pred == str(ground_truth).strip().upper() else 0.0
    try:
        return float(default_compute_score(data_source, text, ground_truth))
    except Exception:
        return 0.0


def score_strict_response(text: str, ground_truth: Any) -> float:
    pred = extract_strict_answer(text)
    if pred is None:
        return 0.0
    return 1.0 if pred == str(ground_truth).strip().upper() else 0.0


def score_response(data_source: str, text: str, ground_truth: Any) -> float:
    return score_relaxed_response(data_source, text, ground_truth)


def _choice_to_trajectory(choice: ApiChoice, *, trajectory_id: str, score_context: Optional[tuple[str, Any]] = None) -> dict[str, Any]:
    score = None
    strict_score = None
    if score_context is not None:
        data_source, ground_truth = score_context
        score = score_relaxed_response(data_source, choice.text, ground_truth)
        strict_score = score_strict_response(choice.text, ground_truth)
    return {
        "trajectory_id": trajectory_id,
        "text": choice.text,
        "extracted_answer": extract_relaxed_answer(choice.text),
        "strict_extracted_answer": extract_strict_answer(choice.text),
        "score": score,
        "strict_score": strict_score,
        "finish_reason": choice.finish_reason,
        "avg_logprob": choice.avg_logprob,
        "token_logprobs": choice.token_logprobs,
        "raw_index": choice.raw_index,
    }


def _answer_from_trajectory(trajectory: dict[str, Any]) -> Optional[str]:
    answer = trajectory.get("extracted_answer")
    return str(answer).upper() if answer else None


def select_normal_trajectory(
    trajectories: list[dict[str, Any]],
    *,
    method: str,
    data_source: str,
    ground_truth: Any,
    rng: random.Random,
) -> tuple[Optional[str], Optional[dict[str, Any]], dict[str, Any]]:
    if not trajectories:
        return None, None, {"oracle_selection": False}

    method = method.lower()
    if method == "reward_model":
        raise NotImplementedError("--normal_selection reward_model is reserved for future implementation.")

    if method == "random":
        valid = [trajectory for trajectory in trajectories if _answer_from_trajectory(trajectory) is not None] or trajectories
        chosen = rng.choice(valid)
        return _answer_from_trajectory(chosen), chosen, {"oracle_selection": False}

    if method == "best":
        scored = []
        for trajectory in trajectories:
            score = score_response(data_source, trajectory.get("text", ""), ground_truth)
            trajectory["score"] = score
            scored.append((score, trajectory))
        chosen = max(scored, key=lambda item: item[0])[1]
        return _answer_from_trajectory(chosen), chosen, {"oracle_selection": True}

    if method != "majority_vote":
        raise ValueError(f"Unsupported normal selection method: {method}")

    answers = [_answer_from_trajectory(trajectory) for trajectory in trajectories]
    valid_answers = [answer for answer in answers if answer is not None]
    if not valid_answers:
        return None, trajectories[0], {"oracle_selection": False}

    counts = Counter(valid_answers)
    best_count = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == best_count}
    final_answer = next(answer for answer in answers if answer in tied)
    chosen = next(trajectory for trajectory in trajectories if _answer_from_trajectory(trajectory) == final_answer)
    return final_answer, chosen, {"oracle_selection": False, "vote_counts": dict(counts)}


def _split_steps(text: str) -> list[str]:
    matches = STEP_RE.findall(text or "")
    if matches:
        suffix_start = (text or "").rfind(matches[-1]) + len(matches[-1])
        suffix = (text or "")[suffix_start:]
        segments = list(matches)
        if suffix.strip():
            segments.append(suffix)
        return segments
    return [text or ""]


def _candidate_score(trajectory: dict[str, Any], fallback_used: bool) -> float:
    avg_logprob = trajectory.get("avg_logprob")
    if avg_logprob is not None and not fallback_used:
        return -float(avg_logprob)
    return float(len(trajectory.get("text", "")))


def _build_branch_messages(messages: list[dict[str, str]], partial_text: str) -> list[dict[str, str]]:
    copied = [dict(message) for message in messages]
    continuation_instruction = (
        "\n\nContinue the following partial solution. Do not restart from the beginning. "
        "Continue with the same format and finish with the final answer in \\boxed{}.\n\n"
        f"<PartialSolution>\n{partial_text}\n</PartialSolution>"
    )
    for message in reversed(copied):
        if message.get("role") == "user":
            message["content"] = message.get("content", "") + continuation_instruction
            break
    return copied


def run_normal_inference(
    client: OpenAICompatibleClient,
    messages: list[dict[str, str]],
    *,
    rollout: int,
    data_source: str,
    ground_truth: Any,
    selection: str,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Optional[str], Optional[dict[str, Any]], dict[str, Any]]:
    response = client.chat(messages, n=rollout)
    trajectories = [
        _choice_to_trajectory(choice, trajectory_id=f"normal_{idx}", score_context=(data_source, ground_truth))
        for idx, choice in enumerate(response.choices)
    ]
    final_pred, selected, metadata = select_normal_trajectory(
        trajectories,
        method=selection,
        data_source=data_source,
        ground_truth=ground_truth,
        rng=rng,
    )
    metadata.update({"usage": response.usage, "api_model": response.raw_model})
    return trajectories, trajectories, final_pred, selected, metadata


def run_tree_inference(
    client: OpenAICompatibleClient,
    messages: list[dict[str, str]],
    *,
    rollout: int,
    data_source: str,
    ground_truth: Any,
    tree_rounds: int,
    top_k: int,
    branch_repeats: int,
    selected_num_traces: int,
    branch_max_tokens: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Optional[str], Optional[dict[str, Any]], dict[str, Any]]:
    initial = client.chat(messages, n=rollout)
    all_trajectories = [
        _choice_to_trajectory(choice, trajectory_id=f"tree_initial_{idx}", score_context=(data_source, ground_truth))
        for idx, choice in enumerate(initial.choices)
    ]
    logprob_fallback = all(trajectory.get("avg_logprob") is None for trajectory in all_trajectories)

    candidates = list(all_trajectories)
    for round_idx in range(max(0, tree_rounds)):
        unfinished = [
            trajectory
            for trajectory in candidates
            if trajectory.get("extracted_answer") is None and trajectory.get("text", "").strip()
        ]
        if not unfinished:
            break
        ranked = sorted(unfinished, key=lambda trajectory: _candidate_score(trajectory, logprob_fallback), reverse=True)
        selected = ranked[: max(1, top_k)]
        for selected_idx, trajectory in enumerate(selected):
            partial_text = trajectory["text"]
            branch_messages = _build_branch_messages(messages, partial_text)
            branch_response = client.chat(
                branch_messages,
                n=max(1, branch_repeats),
                sampling_overrides={"max_tokens": branch_max_tokens},
            )
            for branch_idx, choice in enumerate(branch_response.choices):
                merged_choice = copy.deepcopy(choice)
                merged_choice.text = partial_text + choice.text
                all_trajectories.append(
                    _choice_to_trajectory(
                        merged_choice,
                        trajectory_id=f"tree_r{round_idx}_s{selected_idx}_b{branch_idx}",
                        score_context=(data_source, ground_truth),
                    )
                )
        candidates = list(all_trajectories)

    selected_count = max(1, selected_num_traces)
    ranked_leaves = sorted(all_trajectories, key=lambda trajectory: _candidate_score(trajectory, logprob_fallback), reverse=True)
    selected_trajectories = ranked_leaves[: min(selected_count, len(ranked_leaves))]
    while len(selected_trajectories) < selected_count and selected_trajectories:
        selected_trajectories.append(rng.choice(selected_trajectories))

    final_pred, selected_final, metadata = select_normal_trajectory(
        selected_trajectories,
        method="majority_vote",
        data_source=data_source,
        ground_truth=ground_truth,
        rng=rng,
    )
    metadata.update(
        {
            "logprob_fallback": logprob_fallback,
            "initial_usage": initial.usage,
            "api_model": initial.raw_model,
            "tree_rounds": tree_rounds,
            "top_k": top_k,
            "branch_repeats": branch_repeats,
            "selected_num_traces": selected_num_traces,
        }
    )
    return all_trajectories, selected_trajectories, final_pred, selected_final, metadata


def build_result_record(
    record: dict[str, Any],
    *,
    mode: str,
    rollout: int,
    selection: str,
    all_trajectories: list[dict[str, Any]],
    selected_trajectories: list[dict[str, Any]],
    final_pred: Optional[str],
    selected_final: Optional[dict[str, Any]],
    metadata: dict[str, Any],
    sampling_config: SamplingConfig,
) -> dict[str, Any]:
    ground_truth = record["reward_model"]["ground_truth"]
    final_text = f"\\boxed{{{final_pred}}}" if final_pred is not None else ""
    selected_text = selected_final.get("text", "") if selected_final is not None else ""
    strict_pred = extract_strict_answer(selected_text)
    strict_is_correct = score_strict_response(selected_text, ground_truth) > 0.5
    relaxed_is_correct = score_relaxed_response(record["data_source"], final_text, ground_truth) > 0.5
    return _json_safe(
        {
            "sample_id": record.get("sample_id"),
            "dataset": record.get("data_source"),
            "prompt": record.get("raw_prompt"),
            "ground_truth": ground_truth,
            "final_pred": final_pred,
            "strict_final_pred": strict_pred,
            "is_correct": relaxed_is_correct,
            "relaxed_is_correct": relaxed_is_correct,
            "strict_is_correct": strict_is_correct,
            "mode": mode,
            "rollout": rollout,
            "selection": selection,
            "selected_final_trajectory": selected_final,
            "selected_trajectories": selected_trajectories,
            "all_trajectories": all_trajectories,
            "extra_info": record.get("extra_info", {}),
            "api_metadata": metadata,
            "sampling": asdict(sampling_config),
        }
    )


def build_failed_result_record(
    record: dict[str, Any],
    *,
    mode: str,
    rollout: int,
    selection: str,
    sampling_config: SamplingConfig,
    error: Exception,
    elapsed_seconds: float,
) -> dict[str, Any]:
    metadata = {
        "error": str(error),
        "error_type": type(error).__name__,
        "elapsed_seconds": elapsed_seconds,
    }
    return _json_safe(
        {
            "sample_id": record.get("sample_id"),
            "dataset": record.get("data_source"),
            "prompt": record.get("raw_prompt"),
            "ground_truth": record["reward_model"]["ground_truth"],
            "final_pred": None,
            "strict_final_pred": None,
            "is_correct": False,
            "relaxed_is_correct": False,
            "strict_is_correct": False,
            "failed": True,
            "mode": mode,
            "rollout": rollout,
            "selection": selection,
            "selected_final_trajectory": None,
            "selected_trajectories": [],
            "all_trajectories": [],
            "extra_info": record.get("extra_info", {}),
            "api_metadata": metadata,
            "sampling": asdict(sampling_config),
        }
    )


def _result_key(record: dict[str, Any]) -> Optional[str]:
    sample_id = record.get("sample_id")
    if sample_id is None:
        return None
    return str(sample_id)


def _load_existing_result_map(output_dir: Path, dataset_name: str, model_name: str) -> dict[str, dict[str, Any]]:
    dataset_dir = output_dir / dataset_name
    result_map: dict[str, dict[str, Any]] = {}
    for suffix in ("correct", "incorrect"):
        path = dataset_dir / f"{model_name}_{suffix}.json"
        if not path.is_file():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            key = _result_key(record)
            if key is not None and key not in result_map:
                result_map[key] = record
    return result_map


def _get_thread_client(api_config, sampling_config: SamplingConfig) -> OpenAICompatibleClient:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAICompatibleClient(api_config, sampling_config)
        THREAD_LOCAL.client = client
    return client


def _selection_name(*, effective_mode: str, rollout: int, normal_selection: str) -> str:
    if effective_mode == "tree":
        return "tree_majority_vote"
    return "single" if rollout <= 1 else normal_selection


def run_record_inference(
    *,
    idx: int,
    record: dict[str, Any],
    prompt_instruction: Optional[str],
    api_config,
    sampling_config: SamplingConfig,
    args: argparse.Namespace,
    effective_mode: str,
    selected_num_traces: int,
) -> dict[str, Any]:
    started_at = time.monotonic()
    selection = _selection_name(effective_mode=effective_mode, rollout=args.rollout, normal_selection=args.normal_selection)
    rng = random.Random(args.seed + idx)
    client = _get_thread_client(api_config, sampling_config)
    try:
        messages = inject_prompt_instruction(record["prompt"], prompt_instruction)
        ground_truth = record["reward_model"]["ground_truth"]
        if effective_mode == "tree":
            all_trajectories, selected_trajectories, final_pred, selected_final, metadata = run_tree_inference(
                client,
                messages,
                rollout=args.rollout,
                data_source=record["data_source"],
                ground_truth=ground_truth,
                tree_rounds=args.tree_rounds,
                top_k=args.top_k_nodes,
                branch_repeats=args.branch_repeats,
                selected_num_traces=selected_num_traces,
                branch_max_tokens=args.branch_max_tokens,
                rng=rng,
            )
        else:
            all_trajectories, selected_trajectories, final_pred, selected_final, metadata = run_normal_inference(
                client,
                messages,
                rollout=max(1, args.rollout),
                data_source=record["data_source"],
                ground_truth=ground_truth,
                selection="majority_vote" if args.rollout <= 1 else args.normal_selection,
                rng=rng,
            )
        metadata["elapsed_seconds"] = time.monotonic() - started_at
        return build_result_record(
            record,
            mode=effective_mode,
            rollout=args.rollout,
            selection=selection,
            all_trajectories=all_trajectories,
            selected_trajectories=selected_trajectories,
            final_pred=final_pred,
            selected_final=selected_final,
            metadata=metadata,
            sampling_config=sampling_config,
        )
    except Exception as exc:
        return build_failed_result_record(
            record,
            mode=effective_mode,
            rollout=args.rollout,
            selection=selection,
            sampling_config=sampling_config,
            error=exc,
            elapsed_seconds=time.monotonic() - started_at,
        )


def evaluate_dataset_records(
    *,
    dataset_name: str,
    eval_records: list[dict[str, Any]],
    prompt_instruction: Optional[str],
    api_config,
    sampling_config: SamplingConfig,
    args: argparse.Namespace,
    effective_mode: str,
    selected_num_traces: int,
    output_dir: Path,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_at = time.monotonic()
    result_by_idx: dict[int, dict[str, Any]] = {}
    skipped = 0
    existing_results = _load_existing_result_map(output_dir, dataset_name, model_name) if args.resume else {}

    pending: list[tuple[int, dict[str, Any]]] = []
    for idx, record in enumerate(eval_records):
        existing = existing_results.get(_result_key(record))
        if existing is not None:
            result_by_idx[idx] = existing
            skipped += 1
        else:
            pending.append((idx, record))

    if skipped:
        print(f"[{dataset_name}] skipped {skipped}/{len(eval_records)} existing results", flush=True)

    completed = skipped
    max_workers = max(1, int(args.concurrency))
    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    run_record_inference,
                    idx=idx,
                    record=record,
                    prompt_instruction=prompt_instruction,
                    api_config=api_config,
                    sampling_config=sampling_config,
                    args=args,
                    effective_mode=effective_mode,
                    selected_num_traces=selected_num_traces,
                ): idx
                for idx, record in pending
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result_by_idx[idx] = future.result()
                completed += 1
                failed_note = " failed" if result_by_idx[idx].get("failed") else ""
                print(f"[{dataset_name}] {completed}/{len(eval_records)} done (idx={idx}){failed_note}", flush=True)

    result_records = [result_by_idx[idx] for idx in range(len(eval_records)) if idx in result_by_idx]
    elapsed_seconds = time.monotonic() - started_at
    metadata = {
        "elapsed_seconds": elapsed_seconds,
        "average_seconds_per_sample": elapsed_seconds / len(result_records) if result_records else 0.0,
        "average_seconds_per_new_sample": elapsed_seconds / len(pending) if pending else 0.0,
        "concurrency": max_workers,
        "resumed": bool(args.resume),
        "skipped": skipped,
        "new_samples": len(pending),
    }
    return result_records, metadata


def write_dataset_outputs(
    *,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    result_records: list[dict[str, Any]],
    summary: dict[str, Any],
    output_backend: str,
) -> None:
    if output_backend not in {"local", "both"}:
        return
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    correct = [record for record in result_records if record["is_correct"]]
    incorrect = [record for record in result_records if not record["is_correct"]]
    (dataset_dir / f"{model_name}_correct.json").write_text(json.dumps(correct, indent=2, ensure_ascii=False), encoding="utf-8")
    (dataset_dir / f"{model_name}_incorrect.json").write_text(json.dumps(incorrect, indent=2, ensure_ascii=False), encoding="utf-8")
    (dataset_dir / f"{model_name}_summary.json").write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")


def log_wandb(
    *,
    project: str,
    run_name: str,
    summaries: dict[str, dict[str, Any]],
    records_by_dataset: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> None:
    import wandb

    run = wandb.init(project=project, name=run_name, config=_json_safe(config))
    summary_rows = []
    for dataset_name, summary in summaries.items():
        summary_rows.append([dataset_name, summary["total"], summary["correct"], summary["accuracy"], summary["mode"], summary["rollout"]])
        run.summary[f"{dataset_name}/accuracy"] = summary["accuracy"]
    wandb.log({"summary": wandb.Table(columns=["dataset", "total", "correct", "accuracy", "mode", "rollout"], data=summary_rows)})

    table_columns = ["sample_id", "ground_truth", "final_pred", "prompt", "selected_trajectories", "all_trajectories"]
    for dataset_name, records in records_by_dataset.items():
        for bucket_name, bucket_records in {
            "correct": [record for record in records if record["is_correct"]],
            "incorrect": [record for record in records if not record["is_correct"]],
        }.items():
            rows = [
                [
                    record["sample_id"],
                    str(record["ground_truth"]),
                    str(record["final_pred"]),
                    record["prompt"],
                    json.dumps(record["selected_trajectories"], ensure_ascii=False),
                    json.dumps(record["all_trajectories"], ensure_ascii=False),
                ]
                for record in bucket_records
            ]
            wandb.log({f"{dataset_name}/{bucket_name}": wandb.Table(columns=table_columns, data=rows)})
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--prompt_path", default="prompts/base.txt")
    parser.add_argument("--rollout", type=int, default=1)
    parser.add_argument("--mode", choices=["normal", "tree"], default="normal")
    parser.add_argument("--normal_selection", choices=["majority_vote", "reward_model", "random", "best"], default="majority_vote")
    parser.add_argument("--output_dir", default="eval_output/cross_domain")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_backend", choices=["local", "wandb", "both"], default="local")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="verl")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", type=_parse_bool, default=False)

    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--logprobs", type=_parse_bool, default=None)
    parser.add_argument("--top_logprobs", type=int, default=None)

    parser.add_argument("--tree_rounds", type=int, default=3)
    parser.add_argument("--top_k_nodes", type=int, default=2)
    parser.add_argument("--branch_repeats", type=int, default=1)
    parser.add_argument("--selected_num_traces", type=int, default=None)
    parser.add_argument("--branch_max_tokens", type=int, default=512)

    parser.add_argument("--pubmedqa_data_dir", default="./data/pubmedqa_origin/data")
    parser.add_argument("--mathqa_data_dir", default="./data/MathQA")
    parser.add_argument("--gpqa_seed", type=int, default=42)
    parser.add_argument("--disable_local_parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_config, sampling_config = load_api_config(args.config)

    if args.api_base_url is not None:
        api_config.base_url = args.api_base_url
    if args.api_key is not None:
        api_config.api_key = args.api_key
    if args.model is not None:
        api_config.model = args.model

    output_backend = "both" if args.wandb and args.output_backend == "local" else args.output_backend
    model_name = args.model_name or api_config.model.replace("/", "_").replace(":", "_")
    effective_mode = "normal" if args.rollout <= 1 else args.mode
    selected_num_traces = args.selected_num_traces or args.rollout

    sampling_config = merge_sampling_config(
        sampling_config,
        {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "logprobs": args.logprobs,
            "top_logprobs": args.top_logprobs,
        },
    )
    if args.logprobs is None:
        sampling_config.logprobs = effective_mode == "tree"
    if sampling_config.top_logprobs is not None and int(sampling_config.top_logprobs) <= 0:
        sampling_config.top_logprobs = None
    if not sampling_config.logprobs:
        sampling_config.top_logprobs = None

    data_config = DatasetLoadConfig(
        max_samples=args.max_samples,
        prefer_local_parquet=not args.disable_local_parquet,
        pubmedqa_data_dir=args.pubmedqa_data_dir,
        mathqa_data_dir=args.mathqa_data_dir,
        gpqa_seed=args.gpqa_seed,
    )
    prompt_instruction = load_prompt_instruction(args.prompt_path)

    records_by_dataset: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    output_dir = Path(args.output_dir)

    for dataset_name in _parse_dataset_list(args.datasets):
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset {dataset_name!r}. Available: {sorted(SUPPORTED_DATASETS)}")
        eval_records = load_eval_records(dataset_name, data_config)
        result_records, timing = evaluate_dataset_records(
            dataset_name=dataset_name,
            eval_records=eval_records,
            prompt_instruction=prompt_instruction,
            api_config=api_config,
            sampling_config=sampling_config,
            args=args,
            effective_mode=effective_mode,
            selected_num_traces=selected_num_traces,
            output_dir=output_dir,
            model_name=model_name,
        )

        correct = sum(1 for record in result_records if record["is_correct"])
        strict_correct = sum(1 for record in result_records if record.get("strict_is_correct"))
        relaxed_correct = sum(1 for record in result_records if record.get("relaxed_is_correct", record["is_correct"]))
        failed = sum(1 for record in result_records if record.get("failed"))
        summary = {
            "dataset": dataset_name,
            "model": api_config.model,
            "model_name": model_name,
            "total": len(result_records),
            "correct": correct,
            "incorrect": len(result_records) - correct,
            "failed": failed,
            "accuracy": correct / len(result_records) if result_records else 0.0,
            "strict_correct": strict_correct,
            "strict_acc": strict_correct / len(result_records) if result_records else 0.0,
            "relaxed_correct": relaxed_correct,
            "relaxed_acc": relaxed_correct / len(result_records) if result_records else 0.0,
            "mode": effective_mode,
            "rollout": args.rollout,
            "normal_selection": args.normal_selection,
            "oracle_selection": any(record.get("api_metadata", {}).get("oracle_selection") for record in result_records),
            "prompt_path": args.prompt_path,
            **timing,
        }
        records_by_dataset[dataset_name] = result_records
        summaries[dataset_name] = summary
        write_dataset_outputs(
            output_dir=output_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            result_records=result_records,
            summary=summary,
            output_backend=output_backend,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    if output_backend in {"wandb", "both"}:
        log_wandb(
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"{model_name}_cross_domain_eval",
            summaries=summaries,
            records_by_dataset=records_by_dataset,
            config={
                "api": asdict(api_config),
                "sampling": asdict(sampling_config),
                "args": vars(args),
            },
        )


if __name__ == "__main__":
    main()
