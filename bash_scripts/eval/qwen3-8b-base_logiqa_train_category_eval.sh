#!/usr/bin/env bash

set -euo pipefail
set -x

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY='verl-fol'
export VERL_LOGI_DEBUG=0

ROOT_DIR=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen3-8B-Base
DATA_PATH=$ROOT_DIR/data/logiqa/train.parquet
OUTPUT_DIR=$ROOT_DIR/eval_output/main_eval/qwen3_8b_base_logiqa_train_n2
DATASET_NAME=logiqa_train_n2
REWARD_FN_PATH=$ROOT_DIR/bash_scripts/eval/custom_module.py
PROMPT_PATH=$ROOT_DIR/prompts/base.txt

RUN_GENERATION=1
RUN_MAIN_EVAL=1
RUN_CATEGORY_EVAL=1
N_GPUS=2
TENSOR_PARALLEL_SIZE=1
MAX_COLOCATE_COUNT=1
RAY_NUM_CPUS=8
BATCH_SIZE=8
N_SAMPLES=1
SAMPLE_AGG=best
TEMPERATURE=0.8
TOP_P=1.0
PROMPT_LENGTH=2048
RESPONSE_LENGTH=4096
GPU_MEMORY_UTILIZATION=0.65
MAX_NUM_BATCHED_TOKENS=8192

GENERATED_PATH=$OUTPUT_DIR/${DATASET_NAME}_generated.parquet
MAIN_EVAL_LOG_PATH=$OUTPUT_DIR/${DATASET_NAME}_main_eval.log
CATEGORY_OUTPUT_DIR=$OUTPUT_DIR/category_eval

cd "$ROOT_DIR"
export PYTHONPATH=$ROOT_DIR:$ROOT_DIR/bash_scripts/eval

mkdir -p "$OUTPUT_DIR"

is_hf_model_dir() {
    local model_dir="$1"
    [ -f "$model_dir/model.safetensors" ] || \
        [ -f "$model_dir/pytorch_model.bin" ] || \
        [ -f "$model_dir/model.safetensors.index.json" ] || \
        compgen -G "$model_dir/model-*.safetensors" > /dev/null
}

if ! is_hf_model_dir "$MODEL_PATH"; then
    echo "ERROR: MODEL_PATH is not a HuggingFace model directory: $MODEL_PATH" >&2
    exit 1
fi

if [ "$RUN_GENERATION" = "1" ]; then
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node="$N_GPUS" \
        trainer.max_colocate_count="$MAX_COLOCATE_COUNT" \
        data.path="$DATA_PATH" \
        data.prompt_key=prompt \
        data.prompt_path="$PROMPT_PATH" \
        data.batch_size="$BATCH_SIZE" \
        data.n_samples="$N_SAMPLES" \
        data.output_path="$GENERATED_PATH" \
        model.path="$MODEL_PATH" \
        rollout.n=1 \
        rollout.temperature="$TEMPERATURE" \
        rollout.top_p="$TOP_P" \
        rollout.prompt_length="$PROMPT_LENGTH" \
        rollout.response_length="$RESPONSE_LENGTH" \
        rollout.tensor_model_parallel_size="$TENSOR_PARALLEL_SIZE" \
        rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
        rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
        rollout.enforce_eager=False \
        rollout.free_cache_engine=False \
        ray_init.num_cpus="$RAY_NUM_CPUS"
fi

if [ "$RUN_MAIN_EVAL" = "1" ]; then
    python3 -m verl.trainer.main_eval \
        data.path="$GENERATED_PATH" \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model \
        sample_agg="$SAMPLE_AGG" \
        custom_reward_function.path="$REWARD_FN_PATH" \
        custom_reward_function.name=compute_score \
        ray_init.num_cpus="$RAY_NUM_CPUS" | tee "$MAIN_EVAL_LOG_PATH"
fi

if [ "$RUN_CATEGORY_EVAL" = "1" ]; then
    python3 scripts/logiqa_category_eval.py \
        --input-path "$GENERATED_PATH" \
        --output-dir "$CATEGORY_OUTPUT_DIR" \
        --sample-agg "$SAMPLE_AGG"
fi

echo "Generated parquet: $GENERATED_PATH"
echo "Main evaluation log: $MAIN_EVAL_LOG_PATH"
echo "Category accuracy CSV: $CATEGORY_OUTPUT_DIR/category_accuracy.csv"
