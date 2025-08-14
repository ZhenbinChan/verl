#!/bin/bash
#SBATCH -J PRM                # 作业名为 xxx
#SBATCH -o ./logs/LogiQA.out           # 屏幕上的输出文件重定向到 xxx.out
#SBATCH --qos=lv0b               # 作业使用的 QoS 为 lv0b
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=16        # 单任务使用的 CPU 核心数为 4
#SBATCH -t 48:00:00                # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:2             # 单个节点使用 1 块 GPU 卡
#SBATCH --mem=160G                  #--mem=<size[units]>：设定每个节点的内存大小，后缀可以为[K|M|G|T]，默认为MB

# 设置运行环境
# module add /miniconda     # 添加 anaconda/3-5.0.0.1 模块

HOME=/home/chenzhenbin/Workspaces/verl

MODEL_PATH=/home/chenzhenbin/Workspaces/LLMs/Qwen2.5-1.5B-Instruct

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/logiqa/train.parquet \
    data.val_files=$HOME/data/logiqa/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager='naive_plus' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl' \
    trainer.experiment_name='Qwen2.5-1.5B_GRPO_LogiQA_epoch5' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@