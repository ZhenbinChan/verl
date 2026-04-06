set -x

# FOL-SLM Step-GDPO — 2 GPUs: training on GPU 0, FOL-SLM vLLM on GPU 1
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash fol_slm_step_gdpo_local.sh

HOME=~
MODEL_PATH=~/run/models/Qwen2.5-1.5B-Instruct
FOL_SLM_MODEL_PATH=${FOL_SLM_MODEL_PATH:-~/run/models/Qwen2.5-3B-Instruct}
DATA_NAME=logiqa2k
DATA_DIR="$HOME/run/work/verl/data/${DATA_NAME}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ray stop --force
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ── Launch FOL-SLM vLLM server on GPU 1 ──
FOL_SLM_PORT=${FOL_SLM_PORT:-4869}
export FOL_SLM_MODEL=${FOL_SLM_MODEL:-$(basename $FOL_SLM_MODEL_PATH)}

echo "==> Launching FOL-SLM vLLM server on GPU 1 (port $FOL_SLM_PORT)..."
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model $FOL_SLM_MODEL_PATH \
    --served-model-name $FOL_SLM_MODEL \
    --port $FOL_SLM_PORT \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --no-enable-log-requests > fol_slm_vllm_server.log 2>&1 &
FOL_VLLM_PID=$!
echo "FOL-SLM vLLM server log: fol_slm_vllm_server.log"
trap "echo 'Killing FOL-SLM vLLM server (PID=$FOL_VLLM_PID)'; kill $FOL_VLLM_PID 2>/dev/null" EXIT

echo "Waiting for FOL-SLM vLLM server to start..."
VLLM_READY=0
set +x
for i in $(seq 1 180); do
    if curl -s http://localhost:${FOL_SLM_PORT}/health > /dev/null 2>&1; then
        echo "FOL-SLM vLLM server ready after ${i}s"
        VLLM_READY=1
        break
    fi
    sleep 1
done
set -x
if [ "$VLLM_READY" -eq 0 ]; then
    echo "ERROR: FOL-SLM vLLM server failed to start within 180s"
    exit 1
fi

# FOL-SLM uses these env vars (see nl2fol_slm.py defaults)
export OPENAI_API_KEY="EMPTY"
export FOL_SLM_BASE_URL="http://localhost:${FOL_SLM_PORT}/v1"

# ── Step-GDPO training on GPU 0 ──
# +algorithm.fol_verify_with_cumulative_steps=true to enable step history on FOL evaluation
CUDA_VISIBLE_DEVICES=0 python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_gdpo \
    +algorithm.step_reward_type=fol \
    +algorithm.fol_preprocess=structured \
    +algorithm.fol_translation=assertion \
    +algorithm.fol_max_tries=1 \
    algorithm.use_xml_steps=true \
    +algorithm.step_reward_weights='[0.5, 0.5]' \
    reward_model.reward_manager=step \
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
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl-fol' \
    trainer.experiment_name="qwen1.5b_step_gdpo_fol_slm_1epo_${DATA_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    ++data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    critic.data_loader_seed=42 $@
