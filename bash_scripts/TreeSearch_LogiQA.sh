#!/bin/bash
# TreeRL Example: EPTree-based tree search for RL training
# Based on TreeRL paper (arXiv:2506.11902)
# EPTree params: (M=6, N=2, L=1, T=2) -> 30 leaf paths per prompt

set -x

# Model and data paths
MODEL_PATH=${MODEL_PATH:-"~/run/models/Qwen2.5-14B-SFT"}
DATASET_PATH=${DATASET_PATH:-"~/run/data/logiqa"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=tree_gae \
    algorithm.step_reward_weights="[1.0, 1.0]" \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    \
    data.train_files=${DATASET_PATH}/train.parquet \
    data.val_files=${DATASET_PATH}/val.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1.5e-6 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    \
    reward.reward_manager.name=tree \
    \
    trainer.tree_sampling=True \
    trainer.tree_rounds=1 \
    trainer.tree_top_n=2 \
    trainer.tree_branches=2 \
    trainer.tree_mask_tail_ratio=0.1 \
    trainer.total_epochs=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name=treerl_logiqa \
    trainer.experiment_name=treerl_m6_n2_l1_t2 \
    trainer.logger=console
