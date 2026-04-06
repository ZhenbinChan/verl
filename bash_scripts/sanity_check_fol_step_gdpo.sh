set -x

# FOL Step-GDPO Sanity Check — fast pipeline verification
#
# Parallel structure to sanity_check_self_eval_step_gdpo.sh, but uses FOL
# reward (Z3-based verification) instead of self-eval scoring.
#
# Key differences from full training scripts:
#   - val_max_samples=8   (8 val samples instead of full dataset)
#   - total_training_steps=3
#   - console-only logging (no wandb)
#   - FOL judge vLLM on same GPU (low memory)
#   - fol_max_tries=1 (no retry loop, faster)
#
# Usage (1 GPU):
#   export CUDA_VISIBLE_DEVICES=0
#   bash sanity_check_fol_step_gdpo.sh
#
# Usage (2 GPUs, judge on second GPU, more room for retry loop):
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash sanity_check_fol_step_gdpo.sh

HOME=~
MODEL_PATH=~/run/models/Qwen2.5-1.5B-Instruct
# Use 1.5B as the FOL judge too to minimize memory in sanity check.
# For full runs use Qwen2.5-3B-Instruct as the judge.
FOL_MODEL_PATH=${FOL_MODEL_PATH:-~/run/models/Qwen2.5-1.5B-Instruct}
DATA_NAME=logiqa2k
DATA_DIR="$HOME/run/work/verl/data/${DATA_NAME}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ray stop --force
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Launch FOL vLLM judge server
FOL_PORT=${FOL_PORT:-4869}
export FOL_MODEL=${FOL_MODEL:-$(basename $FOL_MODEL_PATH)}

echo "==> Launching FOL vLLM judge server (port $FOL_PORT)..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $FOL_MODEL_PATH \
    --served-model-name $FOL_MODEL \
    --port $FOL_PORT \
    --gpu-memory-utilization 0.15 \
    --tensor-parallel-size 1 \
    --no-enable-log-requests > fol_vllm_server.log 2>&1 &
FOL_VLLM_PID=$!
echo "FOL vLLM server log: fol_vllm_server.log"
trap "echo 'Killing FOL vLLM server (PID=$FOL_VLLM_PID)'; kill $FOL_VLLM_PID 2>/dev/null" EXIT

echo "Waiting for FOL vLLM server to start..."
VLLM_READY=0
set +x
for i in $(seq 1 180); do
    if curl -s http://localhost:${FOL_PORT}/health > /dev/null 2>&1; then
        echo "FOL vLLM server ready after ${i}s"
        VLLM_READY=1
        break
    fi
    sleep 1
done
set -x
if [ "$VLLM_READY" -eq 0 ]; then
    echo "ERROR: FOL vLLM server failed to start within 180s"
    exit 1
fi

# Point FOL API calls to local vLLM server
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://localhost:${FOL_PORT}/v1"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=step_gdpo \
    +algorithm.step_reward_type=fol \
    +algorithm.fol_max_tries=1 \
    algorithm.use_xml_steps=true \
    +algorithm.step_reward_weights='[0.5, 0.5]' \
    reward_model.reward_manager=step \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/validation.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=8 \
    +data.val_max_samples=8 \
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
    trainer.experiment_name="qwen1.5b_fol_step_gdpo_sanity_${DATA_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.test_freq=3 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3 \
    ++data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    critic.data_loader_seed=42 $@
