"""Generate global_fol_prm metadata for all dataset splits in one run."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.data_preprocess.global_fol_prm_metadata import build_llm_client, load_api_config, process_record, write_outputs
from verl.utils.fol_verifier import FOLMetadata, save_fol_metadata


SPLIT_ALIASES = {
    "train": ("train.parquet",),
    "test": ("test.parquet",),
    "validate": ("validate.parquet", "val.parquet", "validation.parquet"),
}


def _parse_splits(raw_splits: Optional[str]) -> Optional[list[str]]:
    if not raw_splits:
        return None
    splits = [part.strip() for part in raw_splits.split(",") if part.strip()]
    unknown = [split for split in splits if split not in SPLIT_ALIASES]
    if unknown:
        raise ValueError(f"Unsupported split(s): {', '.join(unknown)}. Supported: {', '.join(SPLIT_ALIASES)}")
    return splits


def _discover_split_files(input_dir: Path, requested_splits: Optional[Iterable[str]]) -> Dict[str, Path]:
    split_names = list(requested_splits or SPLIT_ALIASES.keys())
    discovered: Dict[str, Path] = {}
    missing: list[str] = []
    explicit = requested_splits is not None
    for split in split_names:
        match = None
        for filename in SPLIT_ALIASES[split]:
            candidate = input_dir / filename
            if candidate.exists():
                match = candidate
                break
        if match is not None:
            discovered[split] = match
        elif explicit:
            missing.append(split)
    if missing:
        raise FileNotFoundError(f"Requested split(s) not found in {input_dir}: {', '.join(missing)}")
    return discovered


def _build_split_args(base_args, split: str, split_file: Path, dataset_namespace: str) -> Namespace:
    return Namespace(
        input_parquet=str(split_file),
        output_dir=str(base_args.output_dir),
        provider=base_args.provider,
        base_url=base_args.base_url,
        api_key=base_args.api_key,
        model=base_args.model,
        azure_endpoint=base_args.azure_endpoint,
        api_version=base_args.api_version,
        deployment_name=base_args.deployment_name,
        max_workers=base_args.max_workers,
        request_timeout=base_args.request_timeout,
        max_retries=base_args.max_retries,
        max_tokens=base_args.max_tokens,
        temperature=base_args.temperature,
        top_p=base_args.top_p,
        default_args=base_args.default_args,
        extra_body=base_args.extra_body,
        bypass_env_proxy=base_args.bypass_env_proxy,
        num_samples=base_args.num_samples_per_split,
        save_raw_response=base_args.save_raw_response,
        save_every=base_args.save_every,
        output_parquet_name=f"{split}.parquet",
        metadata_filename=f"fol_metadata_{split}.json",
        failures_filename=f"metadata_failures_{split}.json",
        sample_id_prefix=f"{dataset_namespace}_{split}",
    )


def _process_split(split: str, split_file: Path, split_args: Namespace, system_prompt: str, llm_client) -> dict[str, Any]:
    df = pd.read_parquet(split_file)
    original_len = len(df)
    if split_args.num_samples is not None:
        df = df.head(split_args.num_samples)
    raw_records = df.to_dict("records")

    results: list[Optional[dict[str, Any]]] = [None] * len(raw_records)
    metadata_map: dict[str, FOLMetadata] = {}
    failures: list[str] = []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = max(1, int(split_args.max_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_record, record, idx, llm_client, system_prompt, split_args)
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
                f"[global_fol_prm_metadata_splits:{split}] {completed}/{len(raw_records)} "
                f"success={len(metadata_map)} failed={len(failures)}",
                flush=True,
            )
            if split_args.save_every > 0 and completed % split_args.save_every == 0:
                partial = [record for record in results if record is not None]
                write_outputs(partial, metadata_map, Path(split_args.output_dir), split_args.output_parquet_name, split_args.metadata_filename)

    final_records = [record for record in results if record is not None]
    output_dir = Path(split_args.output_dir)
    write_outputs(final_records, metadata_map, output_dir, split_args.output_parquet_name, split_args.metadata_filename)
    if failures:
        with (output_dir / split_args.failures_filename).open("w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
    else:
        failures_path = output_dir / split_args.failures_filename
        if failures_path.exists():
            failures_path.unlink()

    return {
        "split": split,
        "input_file": str(split_file),
        "output_parquet": split_args.output_parquet_name,
        "metadata_file": split_args.metadata_filename,
        "failures_file": split_args.failures_filename if failures else None,
        "original_num_records": original_len,
        "processed_num_records": len(final_records),
        "metadata_entries": len(metadata_map),
        "failures": len(failures),
        "metadata_map": metadata_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate global_fol_prm metadata for train/test/validation splits.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_namespace", default=None)
    parser.add_argument("--splits", default=None, help="Comma-separated subset of train,test,validate. Missing implicit splits are skipped.")
    parser.add_argument("--provider", default="minimax", choices=["minimax", "openai_compatible", "azure_openai"])
    parser.add_argument("--base_url", default="https://api.minimaxi.com/v1")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default="MiniMax-M2.7")
    parser.add_argument("--api_config", default=None)
    parser.add_argument("--use_api_config", default=None)
    parser.add_argument("--use_minimax_config", default=None)
    parser.add_argument("--azure_endpoint", default=None)
    parser.add_argument("--api_version", default=None)
    parser.add_argument("--deployment_name", default=None)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--request_timeout", type=float, default=60)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--bypass_env_proxy", action="store_true")
    parser.add_argument("--num_samples_per_split", type=int, default=None)
    parser.add_argument("--save_raw_response", action="store_true")
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()
    args.default_args = {}
    args.extra_body = None

    api_config_path = args.api_config or args.use_api_config or args.use_minimax_config
    if api_config_path:
        api_cfg = load_api_config(Path(api_config_path))
        args.provider = api_cfg["provider"] or args.provider
        args.base_url = api_cfg["base_url"] or args.base_url
        args.api_key = args.api_key or api_cfg["api_key"]
        args.model = api_cfg["model"] or args.model
        args.azure_endpoint = api_cfg["azure_endpoint"] or args.azure_endpoint
        args.api_version = api_cfg["api_version"] or args.api_version
        args.deployment_name = api_cfg["deployment_name"] or args.deployment_name
        args.request_timeout = api_cfg["request_timeout"] or args.request_timeout
        if api_cfg["bypass_env_proxy"] is not None:
            args.bypass_env_proxy = bool(api_cfg["bypass_env_proxy"])
        args.default_args = api_cfg["default_args"]
        args.extra_body = api_cfg["extra_body"]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_namespace = args.dataset_namespace or input_dir.name
    requested_splits = _parse_splits(args.splits)
    split_files = _discover_split_files(input_dir, requested_splits)
    if not split_files:
        raise FileNotFoundError(f"No supported split parquet files found in {input_dir}")

    from examples.data_preprocess.global_fol_prm_metadata import PROMPT_PATH

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    llm_client = build_llm_client(args)
    combined_metadata: dict[str, FOLMetadata] = {}
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dataset_namespace": dataset_namespace,
        "provider": args.provider,
        "model": args.model,
        "splits": [],
        "skipped_splits": [split for split in SPLIT_ALIASES if split not in split_files],
        "combined_metadata_file": "fol_metadata_all.json",
    }

    for split, split_file in split_files.items():
        split_args = _build_split_args(args, split, split_file, dataset_namespace)
        result = _process_split(split, split_file, split_args, system_prompt, llm_client)
        for sample_id, metadata in result.pop("metadata_map").items():
            if sample_id in combined_metadata:
                raise ValueError(f"Duplicate sample_id={sample_id!r} while combining split metadata.")
            combined_metadata[sample_id] = metadata
        manifest["splits"].append(result)

    save_fol_metadata(combined_metadata, str(output_dir / "fol_metadata_all.json"))
    with (output_dir / "metadata_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(
        f"[global_fol_prm_metadata_splits] saved {len(combined_metadata)} metadata entries "
        f"for {len(manifest['splits'])} split(s) to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
