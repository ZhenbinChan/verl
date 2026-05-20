#!/bin/bash
#SBATCH -J Tree_format
#SBATCH -o /home/chenzhenbin/Workspaces/verl/slurm/logs/logiqa/qwen3-8b-base_Tree_format.out
#SBATCH --qos=lv0b
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=320G

module add cuda12.8/12.8

export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY='verl-fol'

set -x

HOME=/share/nlp/chenzhenbin/Workspaces/verl
MODEL_PATH=/share/nlp/chenzhenbin/Workspaces/LLMs/Qwen3-8B-Base
EXPERIMENT_NAME='qwen3-8b_logiqa_Tree_format'
N_GPUS_PER_NODE=4
BATCH_SIZE=16
MICRO_BSZ=8
TEMPERATURE=0.8
TOP_P=0.95
PROMPT_PATH=$HOME/prompts/premise_conclusion_simple.txt
REWARD_MANAGER='auto'
REWARD_STYLE='null'

VERL_LOGI_DEBUG=disabled

M=6
N=2
L=1
T=2
NUM_TRACES=16

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_treerl_reinforce \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/reclor/train.parquet \
    data.val_files=$HOME/data/reclor/test.parquet \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_path=${PROMPT_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.policy_loss=tree_loss \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS_PER_NODE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.n=${M} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.enable=false \
    reward_model.reward_manager=${REWARD_MANAGER} \
    trainer.val_before_train=True \
    trainer.sampling_strategy=step_treerl \
    trainer.process_reward.type=format \
    trainer.step_treerl_config.max_depth=16 \
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
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=100000000 \
    trainer.test_freq=50 \
    trainer.total_epochs=1  $@
