export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY=${WANDB_ENTITY:-verl-fol}

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "Unset PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True because it is incompatible with vLLM memory pool."
    unset PYTORCH_CUDA_ALLOC_CONF
fi

set -x


PROJECT_ROOT=${PROJECT_ROOT:-/home/chenzhb/Workspaces/verl}
MODEL_PATH=${MODEL_PATH:-/home/chenzhb/Workspaces/LLMs/Qwen2.5-7B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_ROOT}/data/logiqa/train.parquet}
VAL_FILE=${VAL_FILE:-${PROJECT_ROOT}/data/logiqa/validate.parquet}
PROMPT_PATH=${PROMPT_PATH:-${PROJECT_ROOT}/prompts/base.txt}
TRAIN_BSZ=${TRAIN_BSZ:-8}
N=${N:-8}
GPUS=${GPUS:-2}
MINI_BSZ=${MINI_BSZ:-2}
LR=${LR:-1e-6}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.prompt_path=$PROMPT_PATH \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BSZ \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MINI_BSZ \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode='seq-mean-token-mean' \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MINI_BSZ \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GPUS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MINI_BSZ \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    reward_model.reward_manager='naive_plus' \
    reward_model.micro_batch_size_per_gpu=$GPUS \
    reward_model.model.fsdp_config.optimizer_offload=True \
    reward_model.reward_kwargs.reward_style=null \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name='verl' \
    trainer.experiment_name='Qwen2.5-7B_LogiQA_GRPO_only' \
    trainer.rollout_data_dir=$PROJECT_ROOT/record/ \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100000000 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 $@
