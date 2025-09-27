set -x

HOME=./

MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct

PRM_PATH=jinachris/PURE-PRM-7B

python -m verl.trainer.main_ppo \
  data.train_files=$HOME/data/logiqa/train.parquet \
  data.val_files=$HOME/data/logiqa/test.parquet \
  data.train_batch_size=16 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.optim.lr=1e-5 \
  critic.model.path=${MODEL_PATH} \
  critic.ppo_micro_batch_size_per_gpu=8 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  reward_model.enable=True \
  reward_model.model.path=$PRM_PATH \
  reward_model.worker_type='judge' \
  reward_model.reward_manager='naive_plus' \
  reward_model.micro_batch_size_per_gpu=8 \
  reward_model.model.fsdp_config.param_offload=True \
  reward_model.model.fsdp_config.optimizer_offload=True \
  trainer.logger=['console','wandb'] \
  trainer.project_name='verl' \
  trainer.experiment_name='Qwen2.5-1.5b_PRM_LogiQA_ALL_epoch5' \
  trainer.val_before_train=True \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=10 \
  trainer.total_epochs=5 2>&1 | tee verl_demo.log

