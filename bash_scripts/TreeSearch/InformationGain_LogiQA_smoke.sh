#!/usr/bin/env bash
set -x

# ----------------------------------------------------------
# LogiQA + InformationGain (KL-StepTreeRL) smoke run
# sampling_strategy=information_gain
# PRM: format reward (<step>/<premise>/<conclusion> tags)
#
# Simplified parameters:
#   rollout.n     — initial generation count per prompt
#   top_k         — number of high-KL steps to select per round
#   iter_rounds   — number of branching rounds
#
# Final samples per tree: 1 + top_k × iter_rounds
# Example: 1 + 3 × 1 = 4 samples per tree ✅ (divisible by n_gpus)
# ----------------------------------------------------------

HOME=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct

# Single-node GPU count
N_GPUS_PER_NODE=2

# ----------------------------------------------------------
# Simplified parameters
# ----------------------------------------------------------
ROLLOUT_N=4          # Initial generation count per prompt
TOP_K=3               # Number of high-KL steps to select per round
ITER_ROUNDS=1        # Number of branching rounds
# Final: 1 + 3 × 1 = 4 samples per tree (divisible by 2 GPUs ✅)

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ig_grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/reclor/train.parquet \
    data.val_files=$HOME/data/reclor/test.parquet \
    data.train_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_path=$HOME/prompts/premise_conclusion.txt \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS_PER_NODE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.enable=false \
    reward_model.reward_manager='ig' \
    reward_model.reward_kwargs.reward_style=format \
    trainer.val_before_train=True \
    trainer.sampling_strategy=information_gain \
    trainer.ig_config.max_depth=20 \
    trainer.ig_config.max_token_num=512 \
    trainer.ig_config.top_k=${TOP_K} \
    trainer.ig_config.iter_rounds=${ITER_ROUNDS} \
    trainer.ig_config.prm=format \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl' \
    trainer.experiment_name='InformationGain_Reclor' \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=1  $@
