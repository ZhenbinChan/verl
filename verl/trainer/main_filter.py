"""
Generate rollouts and select extreme buckets for SFT data construction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from pprint import pprint
from typing import Any

import hydra
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from scripts.logiqa_sft_filter_utils import as_bool, filter_generated_dataset, inject_prompt_instruction, normalize_messages, rows_to_json_records
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def group_decoded_outputs(output: DataProto, tokenizer, original_batch_size: int, padded_batch_size: int, rollout_n: int) -> list[list[str]]:
    expected_outputs = padded_batch_size * rollout_n
    if len(output) != expected_outputs:
        raise RuntimeError(f"Expected {expected_outputs} generated rows for padded batch size {padded_batch_size} and rollout.n={rollout_n}, got {len(output)}.")

    responses: list[list[str]] = [[] for _ in range(original_batch_size)]
    for prompt_idx in range(original_batch_size):
        for sample_idx in range(rollout_n):
            data_item = output[prompt_idx * rollout_n + sample_idx]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = data_item.batch["responses"][:valid_response_length]
            responses[prompt_idx].append(tokenizer.decode(valid_response_ids, skip_special_tokens=True))
    return responses


def generate_responses(config, dataset: pd.DataFrame) -> pd.DataFrame:
    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompt_instruction = None
    if config.data.get("prompt_path", None):
        prompt_path = Path(str(config.data.prompt_path)).expanduser()
        prompt_instruction = prompt_path.read_text(encoding="utf-8")

    chat_lst = [
        inject_prompt_instruction(normalize_messages(prompt), prompt_instruction)
        for prompt in dataset[config.data.prompt_key].tolist()
    ]

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        max_colocate_count=config.trainer.max_colocate_count,
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = int(config.data.batch_size)
    rollout_n = int(config.rollout.n)
    num_batch = -(-total_samples // config_batch_size)
    output_lst: list[list[str]] = []

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.", flush=True)
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        data = DataProto.from_dict({"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids})
        data_padded, _ = pad_dataproto_to_divisor(data, wg.world_size)

        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.", flush=True)
        output_padded = wg.generate_sequences(data_padded)
        batch_outputs = group_decoded_outputs(
            output=output_padded,
            tokenizer=tokenizer,
            original_batch_size=len(batch_chat_lst),
            padded_batch_size=len(data_padded),
            rollout_n=rollout_n,
        )
        output_lst.extend(batch_outputs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generated = dataset.copy()
    generated["responses"] = output_lst
    return generated


def run_filter(config) -> None:
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    output_dir = Path(str(config.filter.output_dir)).expanduser()
    makedirs(str(output_dir), exist_ok=True)
    generated_path = Path(str(config.filter.generated_path or output_dir / "generations.parquet")).expanduser()

    if as_bool(config.filter.run_generation):
        dataset = pd.read_parquet(copy_to_local(config.data.path))
        generated = generate_responses(config, dataset)
        if as_bool(config.filter.save_generations):
            makedirs(str(generated_path.parent), exist_ok=True)
            generated.to_parquet(generated_path)
    else:
        generated = pd.read_parquet(copy_to_local(str(generated_path)))

    correct_rows, error_rows, remaining = filter_generated_dataset(
        generated=generated,
        correct_size=int(config.filter.correct_size),
        error_size=int(config.filter.error_size),
    )

    write_json(output_dir / "correct.json", rows_to_json_records(correct_rows))
    write_json(output_dir / "error.json", rows_to_json_records(error_rows))
    remaining.to_parquet(output_dir / "train.parquet")

    print(
        f"Saved correct={len(correct_rows)} error={len(error_rows)} remaining={len(remaining)} to {output_dir}",
        flush=True,
    )


@ray.remote(num_cpus=1)
def main_task(config) -> None:
    run_filter(config)


@hydra.main(config_path="config", config_name="filter", version_base=None)
def main(config) -> None:
    if not as_bool(config.filter.run_generation):
        run_filter(config)
        return

    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )
    ray.get(main_task.remote(config))


if __name__ == "__main__":
    main()
