set -x

# FOL Step-GDPO — Local judge boost mode
#
# Uses an already-running local judge at OPENAI_BASE_URL (default: localhost:4872).
# This script is meant for the current 2-GPU setup:
#   - training on one GPU
#   - judge served separately on another GPU

HOME=~
MODEL_PATH=~/run/models/Qwen2.5-1.5B-Instruct
DATA_NAME=logiqa2k
DATA_DIR="$HOME/run/work/verl/data/${DATA_NAME}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

echo "Using $NNODES nodes for training..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export OPENAI_API_KEY=${OPENAI_API_KEY:-"EMPTY"}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-"http://127.0.0.1:4872/v1"}
export FOL_MODEL=${FOL_MODEL:-"Qwen3.6-35B-A3B"}
export FOL_RPM=${FOL_RPM:-0}
export FOL_OPENAI_TPM=${FOL_OPENAI_TPM:-0}
export FOL_OPENAI_MAX_INFLIGHT=${FOL_OPENAI_MAX_INFLIGHT:-128}
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}
export STEP_REWARD_MAX_WORKERS=${STEP_REWARD_MAX_WORKERS:-16}

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_gdpo \
    +algorithm.step_reward_type=fol \
    +algorithm.fol_max_tries=1 \
    +algorithm.fol_timeout=10 \
    +algorithm.api_timeout=200 \
    algorithm.use_xml_steps=true \
    +algorithm.step_reward_weights='[0.5, 0.5]' \
    reward_model.reward_manager=step \
    reward.num_workers=${REWARD_NUM_WORKERS} \
    +algorithm.step_reward_max_workers=${STEP_REWARD_MAX_WORKERS} \
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
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl-fol' \
    trainer.experiment_name="qwen1.5b_step_gdpo_1epo_${DATA_NAME}_fol_boost" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=9999 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    trainer.val_before_train=false \
    ++data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    critic.data_loader_seed=42 $@
