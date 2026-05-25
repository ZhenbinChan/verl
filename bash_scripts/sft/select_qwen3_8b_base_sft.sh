#!/bin/bash


set -x

HOME=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen3-8B-Base
DATA_PATH=$HOME/data/logiqa/train.parquet
PROMPT_PATH=$HOME/prompts/premise_conclusion.txt
OUTPUT_DIR=$HOME/data/logiqa_sft

N_GPUS_PER_NODE=2
BATCH_SIZE=64
ROLLOUT_N=64
TEMPERATURE=0.8
TOP_P=1.0
TOP_K=-1
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.5
CORRECT_SIZE=500
ERROR_SIZE=500
RUN_GENERATION=True
SAVE_GENERATIONS=True
GENERATED_PATH=$OUTPUT_DIR/generations.parquet

cd "$HOME"
export PYTHONPATH="$HOME:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN

python3 -m verl.trainer.main_filter \
    data.path=$DATA_PATH \
    data.prompt_path=$PROMPT_PATH \
    data.batch_size=$BATCH_SIZE \
    model.path=$MODEL_PATH \
    rollout.n=$ROLLOUT_N \
    rollout.temperature=$TEMPERATURE \
    rollout.top_p=$TOP_P \
    rollout.top_k=$TOP_K \
    rollout.prompt_length=$MAX_PROMPT_LENGTH \
    rollout.response_length=$MAX_RESPONSE_LENGTH \
    rollout.max_model_len=$MAX_MODEL_LEN \
    rollout.tensor_model_parallel_size=$N_GPUS_PER_NODE \
    rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    filter.output_dir=$OUTPUT_DIR \
    filter.correct_size=$CORRECT_SIZE \
    filter.error_size=$ERROR_SIZE \
    filter.run_generation=$RUN_GENERATION \
    filter.save_generations=$SAVE_GENERATIONS \
    filter.generated_path=$GENERATED_PATH $@
