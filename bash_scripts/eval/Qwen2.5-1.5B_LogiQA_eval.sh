#!/usr/bin/env bash
set -xeuo pipefail

# Resource note:
# - Ray requests about CPU:(N_GPUS * MAX_COLOCATE_COUNT) + GPU:N_GPUS.
# - Example: N_GPUS=2 and MAX_COLOCATE_COUNT=10 needs about CPU:20 + GPU:2.
# - Example: N_GPUS=2 and MAX_COLOCATE_COUNT=1 needs about CPU:2 + GPU:2.


MAX_COLOCATE_COUNT=1
VERL_LOGI_DEBUG=0 # Set to 1 to enable debug mode for LogiQA evaluation, which may print more detailed logs.

conda activate verl

ROOT_DIR=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct
DATA_PATH=${DATA_PATH:-$ROOT_DIR/data/logiqa/test.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/eval_output/main_eval/qwen2.5_1.5b_instruct_logiqa}
DATASET_NAME=logiqa
REWARD_FN_PATH=$ROOT_DIR/custom_module.py

MAX_SAMPLES=${MAX_SAMPLES:-0}
RUN_GENERATION=${RUN_GENERATION:-1}
RUN_EVAL=${RUN_EVAL:-1}

N_GPUS=${N_GPUS:-2}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_COLOCATE_COUNT=${MAX_COLOCATE_COUNT:-10}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}

BATCH_SIZE=${BATCH_SIZE:-8}
N_SAMPLES=${N_SAMPLES:-1}
SAMPLE_AGG=${SAMPLE_AGG:-best}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.95}
PROMPT_LENGTH=${PROMPT_LENGTH:-1024}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-512}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}

GENERATED_PATH=$OUTPUT_DIR/${DATASET_NAME}_generated.parquet
EVAL_LOG_PATH=$OUTPUT_DIR/${DATASET_NAME}_main_eval.log
EVAL_DATA_PATH=$DATA_PATH

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=true

mkdir -p "$OUTPUT_DIR"

if [ "$MAX_SAMPLES" != "0" ]; then
    EVAL_DATA_PATH=$OUTPUT_DIR/${DATASET_NAME}_subset_${MAX_SAMPLES}.parquet
    python - "$DATA_PATH" "$EVAL_DATA_PATH" "$MAX_SAMPLES" <<'PY'
import sys
import pandas as pd

src, dst, max_samples = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_parquet(src)
df.head(max_samples).to_parquet(dst)
print(f"Saved {min(len(df), max_samples)} rows to {dst}")
PY
fi

if [ "$RUN_GENERATION" = "1" ]; then
    python -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.max_colocate_count=$MAX_COLOCATE_COUNT \
        data.path=$EVAL_DATA_PATH \
        data.prompt_key=prompt \
        data.batch_size=$BATCH_SIZE \
        data.n_samples=$N_SAMPLES \
        data.output_path=$GENERATED_PATH \
        model.path=$MODEL_PATH \
        rollout.temperature=$TEMPERATURE \
        rollout.top_p=$TOP_P \
        rollout.prompt_length=$PROMPT_LENGTH \
        rollout.response_length=$RESPONSE_LENGTH \
        rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
        rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
        rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
        rollout.enforce_eager=False \
        rollout.free_cache_engine=False \
        ray_init.num_cpus=$RAY_NUM_CPUS
fi

if [ "$RUN_EVAL" = "1" ]; then
    python -m verl.trainer.main_eval \
        data.path=$GENERATED_PATH \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model \
        sample_agg=$SAMPLE_AGG \
        custom_reward_function.path=$REWARD_FN_PATH \
        custom_reward_function.name=compute_score \
        ray_init.num_cpus=$RAY_NUM_CPUS | tee $EVAL_LOG_PATH
fi

echo "Generated parquet: $GENERATED_PATH"
echo "Evaluation log: $EVAL_LOG_PATH"
