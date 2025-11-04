set -x

HOME=.

MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct

PRM_PATH=jinachris/PURE-PRM-7B


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_grpo \
    data.train_files=$HOME/data/logiqa/train.parquet \
    data.val_files=$HOME/data/logiqa/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=True \
    reward_model.worker_type='prm' \
    reward_model.reward_manager='prime' \
    reward_model.model.path=${PRM_PATH} \
    reward_model.micro_batch_size_per_gpu=10 \
    reward_model.model.fsdp_config.optimizer_offload=True \
    +reward_model.model.fsdp_config.model_dtype=bfloat16 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl' \
    trainer.experiment_name='Qwen2.5-1.5B_GRPO_LogiQA_PRM_eps3' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@

# GRPO:
# verl
# actor, ref(kl), reward
# actor (rollout: vllm, gradient: policy)
# ref (log_prob) 
# reward (prm)


# 32B: logiqa: 60 -> 62
# 1.5B: 40~


# E
# A) -> 0.00x


# 1,0,1,0,1
# 1,1,0,0,0<-0


# 1,1,0,0,0 ---- PRM800K 1,0,0,-1,0,1