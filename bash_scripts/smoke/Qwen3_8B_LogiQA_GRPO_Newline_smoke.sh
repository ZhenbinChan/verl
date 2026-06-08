#!/usr/bin/env bash

set -euo pipefail

# Run inside an allocated shell:
#   srun -G 2 --cpus-per-task=2 -t 120:00:00 --pty bash -i
#   conda activate verl
#   bash bash_scripts/smoke/Qwen3_8B_LogiQA_GRPO_Newline_smoke.sh

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY='verl-fol'

set -x

HOME=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen3-8B-Base
TRAIN_FILE=$HOME/data/logiqa/train.parquet
VAL_FILE=$HOME/data/logiqa/test.parquet
PROMPT_PATH=$HOME/prompts/premise_conclusion.txt
ROLLOUT_DATA_DIR=$HOME/record
EXPERIMENT_NAME=qwen3-8b_logiqa_grpo_newline_smoke

N_GPUS_PER_NODE=2
BATCH_SIZE=2
MICRO_BATCH_SIZE_PER_GPU=1
ROLLOUT_N=4
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=1024
MAX_MODEL_LEN=4096
PPO_MAX_TOKEN_LEN_PER_GPU=8192
GPU_MEMORY_UTILIZATION=0.5
TEMPERATURE=0.8
TOP_P=1.0
TOTAL_TRAINING_STEPS=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.prompt_path=$PROMPT_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS_PER_NODE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=naive_plus \
    reward_model.model.fsdp_config.optimizer_offload=True \
    reward_model.reward_kwargs.reward_style=null \
    +reward_model.reward_kwargs.penalize_format_error=True \
    trainer.critic_warmup=2 \
    "trainer.logger=['console']" \
    trainer.project_name=verl \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
    trainer.log_format_metrics=True \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.total_epochs=1 \
    "$@"

python3 $HOME/bash_scripts/smoke/check_rollout_newlines.py \
    $ROLLOUT_DATA_DIR/$EXPERIMENT_NAME
