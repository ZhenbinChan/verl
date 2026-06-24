#!/bin/bash
#SBATCH -J OriginWarmup
#SBATCH -o /share/nlp/chenzhenbin/Workspaces/verl/slurm/logs/logiqa/qwen3-8b-base_steprlv2_promptv3.out
#SBATCH --qos=lv0b
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=320G

module add cuda12.8/12.8

# ======================= Start VLLM as RM ======================== #
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m vllm.entrypoints.openai.api_server \
    --model /share/nlp/chenzhenbin/Workspaces/LLMs/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 4869 \
    --gpu-memory-utilization 0.1 \
    --max-model-len 8192 \
    --tensor-parallel-size 4 \
    --served-model-name eval-model \
    --trust-remote-code \
    --guided-decoding-backend xgrammar > vllm_server_outlines_backend.log 2>&1 &

echo $! > vllm_server_outlines_backend.pid

for i in $(seq 1 180); do
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "[info] VLLM server Started after ${i}s"
        VLLM_READY=1
        break
    fi
    sleep 1
done

echo "[info] Start RL Training"

# =============================================================== #

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export WANDB_API_KEY='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'
export WANDB_MODE=online
export WANDB_ENTITY='verl-fol'

set -x

HOME=/share/nlp/chenzhenbin/Workspaces/verl
MODEL_PATH=/share/nlp/chenzhenbin/Workspaces/LLMs/Qwen3-8B-Base
PROMPT_PATH=$HOME/prompts/premise_conclusion_v2.txt

N_GPUS_PER_NODE=4
BATCH_SIZE=4
PPO_MICRO_BATCH_SIZE_PER_GPU=4
LOGPROB_MICRO_BATCH_SIZE_PER_GPU=4


M=2
N=2
L=1
T=2
NUM_TRACES=8

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
MAX_MODEL_LEN=8192
PPO_MAX_TOKEN_LEN_PER_GPU=16384
GPU_MEMORY_UTILIZATION=0.4
TEMPERATURE=0.8
TOP_P=1.0
LR=1e-6
LOG_FORMAT_METRICS=True
LENGTH_PENALTY_ENABLED=false

EXPERIMENT_NAME='qwen3-8b_steprl_originv2_n16_m4n2l1t2_warmup_v2'



CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_treerl_origin \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/logiqa/train.parquet \
    data.val_files=$HOME/data/logiqa/validate.parquet \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_path=${PROMPT_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.policy_loss=tree_loss \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS_PER_NODE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.n=${M} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=false \
    reward_model.reward_manager='step_tree' \
    trainer.val_before_train=True \
    trainer.sampling_strategy=step_treerl \
    trainer.process_reward.type=format \
    trainer.step_treerl_config.max_depth=15 \
    trainer.step_treerl_config.max_token_num=4096 \
    trainer.step_treerl_config.m=${M} \
    trainer.step_treerl_config.n=${N} \
    trainer.step_treerl_config.l=${L} \
    trainer.step_treerl_config.t=${T} \
    trainer.log_format_metrics=${LOG_FORMAT_METRICS} \
    trainer.step_treerl_config.selected_num_traces=${NUM_TRACES} \
    trainer.step_treerl_config.path_selection=selected_terminals \
    trainer.step_treerl_config.use_weighted_value=true \
    trainer.step_treerl_config.weighted_value_style=terminal_ratio \
    trainer.step_treerl_config.overall_norm_style=none \
    trainer.step_treerl_config.length_penalty.enabled=false \
    +trainer.step_treerl_config.trajectory_rm_url=http://localhost:4869/v1 \
    +trainer.step_treerl_config.trajectory_rm_model=eval-model \
    trainer.step_treerl_config.length_penalty.enabled=$LENGTH_PENALTY_ENABLED \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=9999999 \
    trainer.test_freq=20 \
    trainer.total_epochs=1  $@
