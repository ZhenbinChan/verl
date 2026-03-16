#!/bin/bash
set -x

HOME=~
MODEL_PATH=~/run/models/Qwen2.5-1.5B-Instruct
DATA_DIR="$HOME/run/work/verl/data/gsm8k"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_gdpo \
    +algorithm.step_reward_keys='["format_step_reward"]' \
    +algorithm.step_reward_weights='[0.5, 0.5]' \
    reward.reward_manager.name=step \
    reward.reward_manager.source=register \
    reward.step_reward_type=format \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.logger=['console', 'wandb'] \
    trainer.project_name="verl-step-gdpo-qwen" \
    trainer.experiment_name="normal-run" \
    trainer.total_epochs=5 \
    trainer.test_freq=5 \
    trainer.print_sample_freq=10 \
    trainer.default_local_dir="./checkpoints/step_gdpo"
