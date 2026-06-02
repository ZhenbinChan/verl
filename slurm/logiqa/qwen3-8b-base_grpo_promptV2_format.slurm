#!/bin/bash
#SBATCH -J promptV2_50steps
#SBATCH -o /home/chenzhenbin/Workspaces/verl/slurm/logs/logiqa/qwen3-8b-base_grpo_promptV2_format_step50.out
#SBATCH --qos=lv0b
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=320G

module add cuda12.8/12.8

export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY='verl-fol'

set -x


HOME=/share/nlp/chenzhenbin/Workspaces/verl
MODEL_PATH=/share/nlp/chenzhenbin/Workspaces/LLMs/Qwen3-8B-Base
EXPERIMENT_NAME='qwen3-8b_logiqa_grpo_promptV2_format_50steps'
N_GPUS_PER_NODE=4
BATCH_SIZE=64
MICRO_BSZ=2
ROLLOUT_N=64
TEMPERATURE=0.8
TOP_P=1.0
PROMPT_PATH=$HOME/prompts/premise_conclusion_simple.txt
REWARD_MANAGER='naive_format'
REWARD_STYLE='null'
LOG_FORMAT_METRICS=True

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/logiqa/train.parquet \
    data.val_files=$HOME/data/logiqa/validate.parquet \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.prompt_path=${PROMPT_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS_PER_NODE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=${REWARD_MANAGER} \
    reward_model.model.fsdp_config.optimizer_offload=True \
    reward_model.reward_kwargs.reward_style=${REWARD_STYLE} \
    +reward_model.reward_kwargs.penalize_format_error=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.log_format_metrics=${LOG_FORMAT_METRICS} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.val_before_train=True \
    trainer.total_epochs=1 \
    trainer.training_steps=50 $@