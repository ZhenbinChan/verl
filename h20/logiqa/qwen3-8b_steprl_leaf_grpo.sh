#!/usr/bin/env bash

unset ROCR_VISIBLE_DEVICES

export VLLM_LOGGING_LEVEL=WARN
export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY='verl-fol'

PROJECT_ROOT=/2024133105/Workspaces/verl
MODEL_PATH=/2024133105/Workspaces/llms/Qwen3-8B
PROMPT_PATH=$PROJECT_ROOT/prompts/premise_conclusion.txt

N_GPUS_PER_NODE=4
BATCH_SIZE=16
PPO_MICRO_BATCH_SIZE_PER_GPU=16
LOGPROB_MICRO_BATCH_SIZE_PER_GPU=16

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
THINK_MODE=false
EXPERIMENT_NAME='qwen3-8b_steprl_leaf_grpo_wothk'
RESUME_MODE=disable

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files=$PROJECT_ROOT/data/logiqa/train.parquet \
    data.val_files=$PROJECT_ROOT/data/logiqa/validate.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=$THINK_MODE \
    data.prompt_path=$PROMPT_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS_PER_NODE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$M \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=false \
    reward_model.reward_manager='step_tree' \
    trainer.val_before_train=True \
    trainer.sampling_strategy=step_treerl \
    trainer.process_reward.type=format \
    trainer.step_treerl_config.training_reward_mode=leaf_outcome \
    trainer.step_treerl_config.max_depth=15 \
    trainer.step_treerl_config.max_token_num=4096 \
    trainer.step_treerl_config.m=$M \
    trainer.step_treerl_config.n=$N \
    trainer.step_treerl_config.l=$L \
    trainer.step_treerl_config.t=$T \
    trainer.log_format_metrics=$LOG_FORMAT_METRICS \
    trainer.step_treerl_config.selected_num_traces=$NUM_TRACES \
    trainer.step_treerl_config.path_selection=selected_terminals \
    +trainer.step_treerl_config.dedup_sibling_steps=false \
    trainer.step_treerl_config.trajectory_rm_enabled=false \
    trainer.rollout_data_dir=$PROJECT_ROOT/rollout/$EXPERIMENT_NAME/ \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.resume_mode=$RESUME_MODE \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    "$@"
