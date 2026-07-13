#!/usr/bin/env bash

export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY='verl-fol'

HOME=/2024133105/Workspaces/verl
MODEL_PATH=/2024133105/Workspaces/verl/hf_model/qwen3-8b_grpo360_promptv1_wothink
DATASET_NAME=logiqa
MODEL_NAME=qwen3-8b_grpo360_promptv1_wothink


N_GPUS=4
BATCH_SIZE=16
N_SAMPLES=1
TEMPERATURE=0.8
TOP_P=1.0
PROMPT_LENGTH=2048
RESPONSE_LENGTH=4096
GPU_MEMORY_UTILIZATION=0.4
MAX_NUM_BATCHED_TOKENS=8192
TENSOR_PARALLEL_SIZE=4
RAY_NUM_CPUS=8
MAX_COLOCATE_COUNT=1
SAMPLE_AGG='best'

OUTPUT_DIR=$HOME/eval_output/main_eval/${MODEL_NAME}_${DATASET_NAME}
GENERATED_PATH=$OUTPUT_DIR/generated.parquet
EVAL_LOG_PATH=$OUTPUT_DIR/main_eval.log
DATA_PATH=$HOME/data/${DATASET_NAME}/test.parquet
REWARD_FN_PATH=$HOME/verl/utils/reward_score/logi_eval.py
PROMPT_PATH=$HOME/prompts/base.txt

cd "$HOME"
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory already exists: $OUTPUT_DIR"
else
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.max_colocate_count=${MAX_COLOCATE_COUNT} \
    data.path=$DATA_PATH \
    data.prompt_key=prompt \
    data.prompt_path=$PROMPT_PATH \
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

python3 -m verl.trainer.main_eval \
    data.path=$GENERATED_PATH \
    data.prompt_key=prompt \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.reward_model_key=reward_model \
    sample_agg=${SAMPLE_AGG} \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    ray_init.num_cpus=$RAY_NUM_CPUS | tee "$EVAL_LOG_PATH"

echo "Generated parquet: $GENERATED_PATH"
echo "Evaluation log: $EVAL_LOG_PATH"
