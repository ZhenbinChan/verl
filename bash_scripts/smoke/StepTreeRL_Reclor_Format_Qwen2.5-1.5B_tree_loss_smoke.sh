#!/usr/bin/env bash
set -x

HOME=/home/chenzhb/Workspaces/verl
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct

VERL_LOGI_DEBUG=disabled

N_GPUS_PER_NODE=2
M=6
N=2
L=1
T=2
NUM_TRACES=16
LOGPROB_MICRO_BSZ=${LOGPROB_MICRO_BSZ:-2}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_treerl_reinforce \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/reclor/train.parquet \
    data.val_files=$HOME/data/reclor/test.parquet \
    data.train_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_path=$HOME/prompts/premise_conclusion.txt \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.policy_loss=tree_loss \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BSZ} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS_PER_NODE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.n=${M} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BSZ} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.enable=false \
    reward_model.reward_manager='auto' \
    trainer.val_before_train=True \
    trainer.sampling_strategy=step_treerl \
    trainer.process_reward.type=format \
    trainer.step_treerl_config.max_depth=20 \
    trainer.step_treerl_config.max_token_num=4096 \
    trainer.step_treerl_config.m=${M} \
    trainer.step_treerl_config.n=${N} \
    trainer.step_treerl_config.l=${L} \
    trainer.step_treerl_config.t=${T} \
    trainer.step_treerl_config.selected_num_traces=${NUM_TRACES} \
    trainer.step_treerl_config.path_selection=selected_terminals \
    trainer.step_treerl_config.use_weighted_value=true \
    trainer.step_treerl_config.weighted_value_style=sqrt \
    trainer.step_treerl_config.overall_norm_style=none \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl' \
    trainer.experiment_name='StepTreeRL_Reclor_tree_loss_smoke' \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=99999 \
    trainer.test_freq=2 \
    trainer.total_training_steps=5 \
    trainer.total_epochs=1  $@
