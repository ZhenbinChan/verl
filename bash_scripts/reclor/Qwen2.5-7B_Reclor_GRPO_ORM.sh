export WANDB_API_KEY='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'
export WANDB_MODE=online

set -x


HOME=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/
TRAIN_BSZ=8
N=8
GPUS=2
MINI_BSZ=8
LR=1e-6

MODEL_NAME=Qwen2.5-7B-Instruct

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/reclor/train.parquet \
    data.val_files=$HOME/data/reclor/test.parquet \
    data.train_batch_size=$TRAIN_BSZ \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.prompt_path=$HOME/prompts/base.txt \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MINI_BSZ \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.loss_agg_mode='seq-mean-token-mean' \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MINI_BSZ \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GPUS \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MINI_BSZ \
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
    trainer.experiment_name='Qwen2.5-7B_Reclor_GRPO_ORM' \
    trainer.rollout_data_dir=$HOME/record/ \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 $@
