#!/usr/bin/env bash

set -ex

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
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Llama-3.1-8B
CONFIG_PATH=$ROOT_DIR/evaluation/configs/local_vllm_eval_llama3_1.json
PROMPT_PATH=$ROOT_DIR/prompts/base.txt
OUTPUT_DIR=$ROOT_DIR/eval_output/cross_domain_lighteval_settings_llama3_1_8b

API_PORT=4869
API_BASE_URL=http://localhost:$API_PORT/v1
SERVED_MODEL_NAME=eval-model
MODEL_NAME=llama3_1_8b

DATASETS=logiqa,reclor,arlsat
MAX_SAMPLES=0
ROLLOUT=1
MODE=normal
NORMAL_SELECTION=majority_vote
OUTPUT_BACKEND=local
SEED=42
CONCURRENCY=32
RESUME=false

TEMPERATURE=0.8
TOP_P=1.0
TOP_K=-1
MAX_TOKENS=4096
LOGPROBS=false
TOP_LOGPROBS=0

TREE_ROUNDS=3
TREE_TOP_K=2
BRANCH_REPEATS=1
SELECTED_NUM_TRACES=1
BRANCH_MAX_TOKENS=512

cd "$ROOT_DIR"
export PYTHONPATH=$ROOT_DIR

source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate verl

python3 evaluation/cross_domain_eval.py \
    --config $CONFIG_PATH \
    --api_base_url $API_BASE_URL \
    --model $SERVED_MODEL_NAME \
    --datasets $DATASETS \
    --prompt_path $PROMPT_PATH \
    --max_samples $MAX_SAMPLES \
    --rollout $ROLLOUT \
    --mode $MODE \
    --normal_selection $NORMAL_SELECTION \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --output_backend $OUTPUT_BACKEND \
    --seed $SEED \
    --concurrency $CONCURRENCY \
    --resume $RESUME \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --max_tokens $MAX_TOKENS \
    --logprobs $LOGPROBS \
    --top_logprobs $TOP_LOGPROBS \
    --tree_rounds $TREE_ROUNDS \
    --top_k_nodes $TREE_TOP_K \
    --branch_repeats $BRANCH_REPEATS \
    --selected_num_traces $SELECTED_NUM_TRACES \
    --branch_max_tokens $BRANCH_MAX_TOKENS \
    --disable_local_parquet

echo "Llama-3.1-8B rollout1 logic evaluation finished. Per-dataset summaries are under:"
echo "$OUTPUT_DIR/<dataset>/${MODEL_NAME}_summary.json"
