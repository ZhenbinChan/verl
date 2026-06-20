#!/usr/bin/env bash

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY='verl-fol'

set -xeuo pipefail

HOME=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/qwen3-8b-base_warmup_115_v2
PROMPT_PATH=$HOME/prompts/premise_conclusion_v2.txt
TRAIN_FILE=$HOME/data/logiqa/train.parquet
OUTPUT_DIR=$HOME/eval_output/main_generation/qwen3_8b_base_warmup_115_v2_logiqa_train_50_n2
SUBSET_PATH=$OUTPUT_DIR/logiqa_train_50.parquet
GENERATED_PATH=$OUTPUT_DIR/logiqa_train_50_n2_generated.parquet
JSON_PATH=$OUTPUT_DIR/logiqa_train_50_n2_generated.json

MAX_SAMPLES=50
N_SAMPLES=2
N_GPUS_PER_NODE=2
BATCH_SIZE=8
TEMPERATURE=0.8
TOP_P=0.95
PROMPT_LENGTH=2048
RESPONSE_LENGTH=4096
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.5
MAX_COLOCATE_COUNT=1
RAY_NUM_CPUS=2

mkdir -p $OUTPUT_DIR

python3 - "$TRAIN_FILE" "$SUBSET_PATH" "$MAX_SAMPLES" <<'PY'
import sys

import pandas as pd

src, dst, max_samples = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_parquet(src)
subset = df.head(max_samples)
subset.to_parquet(dst)
print(f"Saved {len(subset)} rows to {dst}")
PY

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.max_colocate_count=$MAX_COLOCATE_COUNT \
    data.path=$SUBSET_PATH \
    data.prompt_key=prompt \
    data.prompt_path=$PROMPT_PATH \
    data.batch_size=$BATCH_SIZE \
    data.n_samples=$N_SAMPLES \
    data.output_path=$GENERATED_PATH \
    model.path=$MODEL_PATH \
    rollout.n=1 \
    rollout.temperature=$TEMPERATURE \
    rollout.top_p=$TOP_P \
    rollout.prompt_length=$PROMPT_LENGTH \
    rollout.response_length=$RESPONSE_LENGTH \
    rollout.tensor_model_parallel_size=$N_GPUS_PER_NODE \
    rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    rollout.max_model_len=$MAX_MODEL_LEN \
    rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    rollout.enforce_eager=False \
    rollout.free_cache_engine=False \
    ray_init.num_cpus=$RAY_NUM_CPUS

python3 - "$GENERATED_PATH" "$JSON_PATH" <<'PY'
import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src)
df.to_json(dst, orient="records", force_ascii=False, indent=2)
print(f"Saved {len(df)} records to {dst}")
PY

echo "Subset parquet: $SUBSET_PATH"
echo "Generated parquet: $GENERATED_PATH"
echo "Generated JSON: $JSON_PATH"
