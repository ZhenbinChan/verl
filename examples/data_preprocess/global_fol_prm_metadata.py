"""Generate metadata for global_fol_prm with external LLM APIs.

This script intentionally avoids the older entity/predicate extraction
preprocessor. It directly asks the declaration prompt to produce executable
Z3 declaration code for each MCQ sample.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from verl.utils.fol_verifier import FOLMetadata, FOLVerifier, LLMClient, save_fol_metadata


PROMPT_PATH = REPO_ROOT / "mcts_utils" / "prompts" / "Z3DeclarationsGeneration1.txt"


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_python(v) for v in value]
    return value


def _extract_prompt_text(value: Any) -> Optional[str]:
    value = _to_python(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, dict):
        content = value.get("content")
        return str(content) if content is not None and str(content).strip() else None
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _extract_prompt_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None
    text = str(value)
    return text if text.strip() else None


def _question_from_extra_info(extra_info: Any) -> Optional[str]:
    extra_info = _to_python(extra_info)
    if not isinstance(extra_info, dict):
        return None
    for key in ("question", "raw_prompt", "prompt"):
        text = _extract_prompt_text(extra_info.get(key))
        if text and "<Context>" in text:
            return text
    context = extra_info.get("context", "")
    query = extra_info.get("query", "")
    options = extra_info.get("options", "")
    if context or query or options:
        return f"<Context>{context}</Context><Question>{query}</Question><Options>{options}</Options>"
    for key in ("question", "raw_prompt", "prompt"):
        text = _extract_prompt_text(extra_info.get(key))
        if text:
            return text
    return None


def get_question_text(record: Dict[str, Any]) -> str:
    for key in ("raw_prompt", "question_text"):
        text = _extract_prompt_text(record.get(key))
        if text:
            return text
    text = _extract_prompt_text(record.get("prompt"))
    if text:
        return text
    text = _question_from_extra_info(record.get("extra_info"))
    if text:
        return text
    raise ValueError("Could not find question text in record")


def get_ground_truth(record: Dict[str, Any]) -> str:
    answer = record.get("answer")
    if answer is not None:
        return str(answer)
    reward_model = _to_python(record.get("reward_model"))
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    extra_info = _to_python(record.get("extra_info"))
    if isinstance(extra_info, dict) and extra_info.get("answer") is not None:
        return str(extra_info["answer"])
    return ""


def extract_python_code(text: str) -> str:
    matches = re.findall(r"```python\s+(.*?)```", text or "", flags=re.DOTALL)
    return matches[-1].strip() if matches else (text or "").strip()


def build_llm_client(args) -> LLMClient:
    api_key = (
        args.api_key
        or os.getenv("MINIMAX_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )
    return LLMClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        provider=args.provider,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        deployment_name=args.deployment_name,
        request_timeout=args.request_timeout,
        default_args={
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )


def process_record(
    record: Dict[str, Any],
    idx: int,
    llm_client: LLMClient,
    system_prompt: str,
    args,
) -> tuple[int, Dict[str, Any], Optional[FOLMetadata], Optional[str]]:
    sample = dict(record)
    sample_id = str(sample.get("sample_id", f"sample_{idx}"))
    question_text = get_question_text(sample)
    ground_truth = get_ground_truth(sample)
    last_error = None

    for attempt in range(1, args.max_retries + 1):
        try:
            raw_response = llm_client.generate(question_text, system_prompt=system_prompt)
            declaration_code = extract_python_code(raw_response)
            FOLVerifier.validate_z3_declaration_code(declaration_code)
            metadata = FOLMetadata(
                sample_id=sample_id,
                rephrased_context="",
                question_text=question_text,
                prm_mode="global_fol_prm",
                z3_declaration_code=declaration_code,
                ground_truth=ground_truth,
            )
            payload = metadata.to_dict()
            if args.save_raw_response:
                payload["raw_declaration_response"] = raw_response
            sample["fol_metadata"] = payload
            sample.setdefault("sample_id", sample_id)
            return idx, sample, metadata, None
        except Exception as exc:
            last_error = f"attempt {attempt}: {exc}"

    sample["fol_metadata"] = None
    sample.setdefault("sample_id", sample_id)
    return idx, sample, None, last_error


def write_outputs(records: List[Dict[str, Any]], metadata: Dict[str, FOLMetadata], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_records = []
    for record in records:
        safe_record = dict(record)
        if isinstance(safe_record.get("fol_metadata"), dict):
            safe_record["fol_metadata"] = json.dumps(safe_record["fol_metadata"], ensure_ascii=False)
        parquet_records.append(safe_record)
    pd.DataFrame(parquet_records).to_parquet(output_dir / "train.parquet")
    save_fol_metadata(metadata, str(output_dir / "fol_metadata.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate global_fol_prm metadata with API batching.")
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--provider", default="minimax", choices=["minimax", "openai_compatible", "azure_openai"])
    parser.add_argument("--base_url", default="https://api.minimaxi.com/v1")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default="MiniMax-M2.7")
    parser.add_argument("--azure_endpoint", default=None)
    parser.add_argument("--api_version", default=None)
    parser.add_argument("--deployment_name", default=None)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--request_timeout", type=float, default=60)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--save_raw_response", action="store_true")
    parser.add_argument("--save_every", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_parquet)
    if args.num_samples is not None:
        df = df.head(args.num_samples)
    raw_records = df.to_dict("records")

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    llm_client = build_llm_client(args)
    output_dir = Path(args.output_dir)
    results: List[Optional[Dict[str, Any]]] = [None] * len(raw_records)
    metadata_map: Dict[str, FOLMetadata] = {}
    failures: List[str] = []

    workers = max(1, int(args.max_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_record, record, idx, llm_client, system_prompt, args)
            for idx, record in enumerate(raw_records)
        ]
        completed = 0
        for future in as_completed(futures):
            idx, sample, metadata, error = future.result()
            results[idx] = sample
            if metadata is not None:
                metadata_map[metadata.sample_id] = metadata
            elif error:
                failures.append(f"{sample.get('sample_id', idx)}: {error}")
            completed += 1
            print(
                f"[global_fol_prm_metadata] {completed}/{len(raw_records)} "
                f"success={len(metadata_map)} failed={len(failures)}",
                flush=True,
            )
            if args.save_every > 0 and completed % args.save_every == 0:
                partial = [record for record in results if record is not None]
                write_outputs(partial, metadata_map, output_dir)

    final_records = [record for record in results if record is not None]
    write_outputs(final_records, metadata_map, output_dir)
    if failures:
        with open(output_dir / "metadata_failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
    print(
        f"[global_fol_prm_metadata] saved {len(final_records)} samples, "
        f"{len(metadata_map)} metadata entries to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
