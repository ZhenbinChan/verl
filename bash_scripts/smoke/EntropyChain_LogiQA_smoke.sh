#!/usr/bin/env bash
set -x

# -------------------------------
# LogiQA + EntropyChain smoke run
# 默认 2 step；设置 TOTAL_TRAINING_STEPS=1 即可只跑 1 step
# -------------------------------

HOME=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct

TOTAL_TRAINING_STEPS=2

# ------------------------------------------------------------
#   N — 每一轮、每棵树最多选多少个「高熵 token 位置」做扩展（全树候选按熵排序后取 top-N）
#   L — 熵引导扩展的轮数（EntropyChainStrategy 里循环 expand_one_round 的次数）
#   T — 每个扩展点重复采样次数（同一分叉任务在 batch 里重复 T 次）
N=2
L=1
T=1
# ------------------------------------------------------------
# 单机 GPU 数；需满足 ray_trainer 里 real_train_batch_size % n_gpus == 0
# （默认 train_batch=1、rollout.n=2、tree_rounds=1、tree_top_k=1 → real_train_batch_size=4，可被 2 整除）
#  假设 train_batch = bs, rollout.n = n, tree_rounds = r, tree_top_k = k
#  Rollout 之后的大小是 bs * n; 第1轮每棵树扩展 k 个分叉，那么得到 bs * n * (1+k) 个分叉
#  第2轮每棵树又扩展 k 个分叉，那么得到 bs * n * (1+k*2)
#  所以 real_train_batch_size = bs * n * (1+k*r)
N_GPUS_PER_NODE=2
# ------------------------------------------------------------

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=entropy_reinforce \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/logiqa_action/train.parquet \
    data.val_files=$HOME/data/logiqa_action/test.parquet \
    data.train_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
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
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.enable=false \
    reward_model.reward_manager='entropy' \
    reward_model.reward_kwargs.reward_style='state_value' \
    reward_model.reward_kwargs.print_entropy_tree=true \
    reward_model.reward_kwargs.print_entropy_tree_max_preview_chars=20 \
    reward_model.reward_kwargs.print_entropy_tree_local_rank_only=true \
    reward_model.reward_kwargs.entropy_tree_graphviz_dir=/home/chenzhb/Workspaces/verl/visualization \
    reward_model.reward_kwargs.entropy_tree_graphviz_view=false \
    reward_model.reward_kwargs.entropy_tree_graphviz_max_label_chars=512 \
    trainer.val_before_train=False \
    trainer.sampling_strategy=treerl \
    trainer.entropy_chain_config.N=${N} \
    trainer.entropy_chain_config.L=${L} \
    trainer.entropy_chain_config.T=${T} \
    trainer.entropy_chain_config.max_token_num=4096 \
    trainer.entropy_chain_config.evaluation_strategy='token-entropy' \
    trainer.entropy_chain_config.enforce_uniform_per_prompt=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl' \
    trainer.experiment_name='EntropyChain_LogiQA_smoke' \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} $@
