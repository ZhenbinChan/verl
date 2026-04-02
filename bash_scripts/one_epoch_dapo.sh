set -x

HOME=~
MODEL_PATH=~/run/models/Qwen2.5-1.5B-Instruct
DATA_NAME=logiqa2k
DATA_DIR="$HOME/run/work/verl/data/${DATA_NAME}"
export VLLM_ATTENTION_BACKEND=XFORMERS
# ray stop --force
# ! For some reason, great lakes sets these two env vars
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

echo "Using $NNODES nodes for training..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# overlong_buffer_cfg.len: 超长惩罚，也就是如果超过 max_resp_len - overlong_buffer_len，那么就扣除 超出/overlong_buffer*penalty_factor
# overlong_buffer_cfg 必须开启，否则模型模式崩溃 重复生成token
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/validation.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=2048 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl-fol' \
    trainer.experiment_name="qwen1.5b_dapo_1epo_${DATA_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    ++data.seed=42 \
    ++actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.model.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    critic.model.seed=42 \
    critic.data_loader_seed=42 $@