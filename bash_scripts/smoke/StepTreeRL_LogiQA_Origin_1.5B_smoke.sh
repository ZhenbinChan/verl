#!/usr/bin/env bash

# Run inside an allocated shell with the `verl` conda env:
#   srun -G 2 --cpus-per-task=2 -t 120:00:00 --pty bash -i
#   conda activate verl
#   bash bash_scripts/smoke/StepTreeRL_LogiQA_Origin_1.5B_smoke.sh

set -euo pipefail

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN

set -x

HOME=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct
PROMPT_PATH=$HOME/prompts/premise_conclusion.txt
TRAIN_FILE=$HOME/data/logiqa/train.parquet
VAL_FILE=$HOME/data/logiqa/test.parquet

N_GPUS_PER_NODE=2
M=2
N=1
L=1
T=1
NUM_TRACES=4
TRAIN_BSZ=2
MINI_BSZ=1
LR=1e-6
TOTAL_TRAINING_STEPS=5
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=512
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.8

EXPERIMENT_NAME=StepTreeRL_LogiQA_origin_terminal_ratio_smoke

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_treerl_origin \
    algorithm.use_kl_in_reward=False \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_path=$PROMPT_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.policy_loss=tree_loss \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MINI_BSZ \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MINI_BSZ \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS_PER_NODE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$M \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MINI_BSZ \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.enable=false \
    reward_model.reward_manager=step_tree \
    trainer.val_before_train=False \
    trainer.sampling_strategy=step_treerl \
    trainer.process_reward.type=format \
    trainer.step_treerl_config.max_depth=15 \
    trainer.step_treerl_config.max_token_num=512 \
    trainer.step_treerl_config.branch_max_new_tokens=128 \
    trainer.step_treerl_config.m=$M \
    trainer.step_treerl_config.n=$N \
    trainer.step_treerl_config.l=$L \
    trainer.step_treerl_config.t=$T \
    trainer.step_treerl_config.selected_num_traces=$NUM_TRACES \
    trainer.step_treerl_config.path_selection=selected_terminals \
    trainer.step_treerl_config.use_weighted_value=true \
    trainer.step_treerl_config.weighted_value_style=terminal_ratio \
    trainer.step_treerl_config.overall_norm_style=none \
    trainer.step_treerl_config.length_penalty.enabled=false \
    +trainer.step_treerl_config.trajectory_rm_url=http://localhost:4869/v1 \
    +trainer.step_treerl_config.trajectory_rm_model=eval-model \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.total_epochs=1 \
    "$@"
