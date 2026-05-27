#!/usr/bin/env bash

if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

# Request GPUs first, then run this script inside the allocated shell:
#   srun -G 2 --cpus-per-task=2 -t 120:00:00 --pty bash -i
#   conda activate verl
#   bash bash_scripts/smoke/LogiQA_NaivePlus_Format_Metrics_1.5B_smoke.sh

unset ROCR_VISIBLE_DEVICES
export VLLM_LOGGING_LEVEL=WARN
export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY='verl-fol'

if PYTORCH_CUDA_ALLOC_CONF_VALUE=$(printenv PYTORCH_CUDA_ALLOC_CONF); then
    case "$PYTORCH_CUDA_ALLOC_CONF_VALUE" in
        *expandable_segments:True*)
            echo "Unset PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True because it is incompatible with vLLM memory pool."
            unset PYTORCH_CUDA_ALLOC_CONF
            ;;
    esac
fi

set -x

HOME=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct
TRAIN_FILE=$HOME/data/logiqa/train.parquet
VAL_FILE=$HOME/data/logiqa/validate.parquet
PROMPT_PATH=$HOME/prompts/premise_conclusion.txt

GPUS=2
TRAIN_BSZ=2
ROLLOUT_N=2
MINI_BSZ=1
LR=1e-6

MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=256
MAX_MODEL_LEN=1280
PPO_MAX_TOKEN_LEN_PER_GPU=4096
ROLLOUT_GPU_MEMORY_UTILIZATION=0.25
EXPERIMENT_NAME=logiqa_naive_plus_format_metrics_1p5b_smoke

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.prompt_path=$PROMPT_PATH \
    data.truncation=error \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BSZ \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MINI_BSZ \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MINI_BSZ \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GPUS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MINI_BSZ \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=naive_plus \
    reward_model.micro_batch_size_per_gpu=$GPUS \
    reward_model.model.fsdp_config.optimizer_offload=True \
    reward_model.reward_kwargs.reward_style=null \
    +reward_model.reward_kwargs.penalize_format_error=True \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=verl \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.rollout_data_dir=$HOME/record/ \
    trainer.log_format_metrics=True \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.val_before_train=True \
    trainer.total_training_steps=5 \
    trainer.total_epochs=1 "$@"
