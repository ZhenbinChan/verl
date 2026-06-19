#!/usr/bin/env bash

module load cuda/12.8


unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY='verl-fol'

# ======================= Start VLLM as RM ======================== #
CACHE_ROOT=~/run/Workspaces/vllm_runtime_cache

export HF_HOME=$CACHE_ROOT/huggingface
export HF_MODULES_CACHE=$HF_HOME/modules
export TRANSFORMERS_CACHE=$HF_HOME/transformers

export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export TORCH_HOME=$CACHE_ROOT/torch
export VLLM_CACHE_ROOT=$CACHE_ROOT/vllm
export XDG_CACHE_HOME=$CACHE_ROOT/xdg

export VLLM_NO_USAGE_STATS=1

mkdir -p \
  "$HF_HOME" \
  "$HF_MODULES_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$XDG_CACHE_HOME"

VLLM_PORT=4869
CUDA_VISIBLE_DEVICES=0,1 nohup python -m vllm.entrypoints.openai.api_server \
    --model /data/home/scyb224/run/Workspaces/LLMs/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --gpu-memory-utilization 0.5 \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --served-model-name eval-model \
    --trust-remote-code \
    --guided-decoding-backend xgrammar > vllm_server_outlines_backend.log 2>&1 &

echo $! > vllm_server_outlines_backend.pid

echo "[info] Waiting for VLLM server to be ready on port ${VLLM_PORT}..."
for i in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 http://localhost:${VLLM_PORT}/v1/models)
    if [ "$code" = "200" ]; then
        echo "[info] VLLM server started after ${i}s"
        VLLM_READY=1
        break
    fi
    sleep 1
done

if [ "$VLLM_READY" != "1" ]; then
    echo "[ERROR] VLLM server failed to start within 180s. Check vllm_server_outlines_backend.log"
    exit 1
fi


echo "[info] Start RL Training"

# ======================== End VLLM as RM ======================== #


HOME=/data/home/scyb224/run/Workspaces/verl
MODEL_PATH=/data/home/scyb224/run/Workspaces/LLMs/Qwen3-8B-Base
PROMPT_PATH=$HOME/prompts/premise_conclusion_v2.txt

N_GPUS_PER_NODE=4
BATCH_SIZE=4
PPO_MICRO_BATCH_SIZE_PER_GPU=4
LOGPROB_MICRO_BATCH_SIZE_PER_GPU=4


M=4
N=2
L=2
T=2
NUM_TRACES=16

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
MAX_MODEL_LEN=8192
PPO_MAX_TOKEN_LEN_PER_GPU=16384
GPU_MEMORY_UTILIZATION=0.4
TEMPERATURE=0.8
TOP_P=1.0
LR=1e-6
LOG_FORMAT_METRICS=True
LENGTH_PENALTY_ENABLED=false

EXPERIMENT_NAME='qwen3-8b_steprl_originv2_n16_m4n2l2t2_warmup_v2'



CUDA_VISIBLE_DEVICES=2,3,4,5 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_treerl_origin \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/logiqa/train.parquet \
    data.val_files=$HOME/data/logiqa/validate.parquet \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_path=${PROMPT_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.policy_loss=tree_loss \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS_PER_NODE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.n=${M} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=false \
    reward_model.reward_manager='step_tree' \
    trainer.val_before_train=True \
    trainer.sampling_strategy=step_treerl \
    trainer.process_reward.type=format \
    trainer.step_treerl_config.max_depth=15 \
    trainer.step_treerl_config.max_token_num=4096 \
    trainer.step_treerl_config.m=${M} \
    trainer.step_treerl_config.n=${N} \
    trainer.step_treerl_config.l=${L} \
    trainer.step_treerl_config.t=${T} \
    trainer.log_format_metrics=${LOG_FORMAT_METRICS} \
    trainer.step_treerl_config.selected_num_traces=${NUM_TRACES} \
    trainer.step_treerl_config.path_selection=selected_terminals \
    trainer.step_treerl_config.use_weighted_value=true \
    trainer.step_treerl_config.weighted_value_style=terminal_ratio \
    trainer.step_treerl_config.overall_norm_style=none \
    trainer.step_treerl_config.length_penalty.enabled=false \
    +trainer.step_treerl_config.trajectory_rm_url=http://localhost:4869/v1 \
    +trainer.step_treerl_config.trajectory_rm_model=eval-model \
    trainer.step_treerl_config.length_penalty.enabled=$LENGTH_PENALTY_ENABLED \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=9999999 \
    trainer.test_freq=20 \
    trainer.total_epochs=1  $@