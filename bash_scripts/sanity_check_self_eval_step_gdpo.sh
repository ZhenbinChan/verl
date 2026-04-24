set -x

# Self-Eval Step-GDPO Sanity Check — fast pipeline verification
#
# Key differences from full training scripts:
#   - val_max_samples=8   (only 8 val samples instead of 651 → ~2 min validation vs 10+ min)
#   - total_training_steps=3
#   - console-only logging (no wandb)
#   - Judge vLLM on same GPU with low memory budget
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   bash sanity_check_self_eval_step_gdpo.sh
#
# With 2 GPUs (faster, judge on second GPU):
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash sanity_check_self_eval_step_gdpo.sh

HOME=~
MODEL_PATH=~/run/models/Qwen2.5-1.5B-Instruct
DATA_NAME=logiqa2k
DATA_DIR="$HOME/run/work/verl/data/${DATA_NAME}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ray stop --force
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Launch vLLM judge server
SELF_EVAL_PORT=${SELF_EVAL_PORT:-8199}
export SELF_EVAL_MODEL=${SELF_EVAL_MODEL:-$(basename $MODEL_PATH)}
SELF_EVAL_MAX_MODEL_LEN=${SELF_EVAL_MAX_MODEL_LEN:-8192}

echo "==> Launching local vLLM judge server (port $SELF_EVAL_PORT)..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $SELF_EVAL_MODEL \
    --port $SELF_EVAL_PORT \
    --gpu-memory-utilization 0.15 \
    --tensor-parallel-size 1 \
    --max-model-len $SELF_EVAL_MAX_MODEL_LEN \
    --no-enable-log-requests > vllm_server.log 2>&1 &
VLLM_PID=$!
echo "vLLM server log: vllm_server.log"
trap "echo 'Killing vLLM server (PID=$VLLM_PID)'; kill $VLLM_PID 2>/dev/null" EXIT
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

echo "Waiting for vLLM server to start..."
VLLM_READY=0
set +x
for i in $(seq 1 600); do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${SELF_EVAL_PORT}/health', timeout=2).read()" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        VLLM_READY=1
        break
    fi
    sleep 1
done
set -x
if [ "$VLLM_READY" -eq 0 ]; then
    echo "ERROR: vLLM server failed to start within 600s"
    exit 1
fi

export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:${SELF_EVAL_PORT}/v1"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_gdpo \
    +algorithm.step_reward_type=self_eval \
    algorithm.use_xml_steps=true \
    +algorithm.step_reward_weights='[0.5, 0.5]' \
    reward_model.reward_manager=step \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/validation.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=8 \
    data.val_max_samples=8 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl-fol' \
    trainer.experiment_name="qwen1.5b_self_eval_step_gdpo_sanity_${DATA_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.test_freq=3 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3 \
    +reward.api_config.temperature=0.0 \
    ++data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    critic.data_loader_seed=42 $@
