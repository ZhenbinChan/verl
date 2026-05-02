#!/usr/bin/env bash
set -x

# ----------------------------------------------------------
# LogiQA + ParallelMCTS smoke run
# sampling_strategy=parallel_mcts
# PRM: format reward (<step>/<premise>/<conclusion> tags)
# ----------------------------------------------------------

HOME=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct

TOTAL_TRAINING_STEPS=2

# ----------------------------------------------------------
# MCTS parameters
#   MAX_NODES     — 每棵树最多扩展的节点数
#   MAX_CHILDREN  — 每个节点每轮并行生成的子节点数
#   CONCURRENT    — 每轮 UCT 选取的节点数/棵树
#   PASS_K        — 每棵树达到多少个终止叶子后停止搜索
#   NUM_TRACES    — 每棵树选多少条路径用于训练
#                   real_train_batch_size = train_batch_size × NUM_TRACES
#                   需满足 real_train_batch_size % N_GPUS_PER_NODE == 0
# ----------------------------------------------------------
MAX_NODES=10
MAX_CHILDREN=3
CONCURRENT=2
PASS_K=4
NUM_TRACES=4    # real_train_batch_size = 2 × 4 = 8

# 单机 GPU 数（8 % 2 = 0 满足整除要求）
N_GPUS_PER_NODE=2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=mcts_grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/logiqa_action_1k/train.parquet \
    data.val_files=$HOME/data/logiqa_action_1k/test.parquet \
    data.train_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS_PER_NODE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.enable=false \
    reward_model.reward_manager='mcts' \
    reward_model.reward_kwargs.reward_style=format \
    trainer.val_before_train=False \
    trainer.sampling_strategy=parallel_mcts \
    trainer.parallel_mcts_config.max_nodes=${MAX_NODES} \
    trainer.parallel_mcts_config.max_depth=20 \
    trainer.parallel_mcts_config.max_children=${MAX_CHILDREN} \
    trainer.parallel_mcts_config.concurrent_num=${CONCURRENT} \
    trainer.parallel_mcts_config.pass_k=${PASS_K} \
    trainer.parallel_mcts_config.num_traces=${NUM_TRACES} \
    trainer.parallel_mcts_config.exploration_constant=1.0 \
    trainer.parallel_mcts_config.gamma=0.9 \
    trainer.parallel_mcts_config.max_token_num=256 \
    trainer.parallel_mcts_config.backprop=true \
    trainer.parallel_mcts_config.random_pick=false \
    trainer.parallel_mcts_config.selection_policy=importance_sampling \
    trainer.parallel_mcts_config.prm=format \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl' \
    trainer.experiment_name='ParallelMCTS_LogiQA_smoke' \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} $@
