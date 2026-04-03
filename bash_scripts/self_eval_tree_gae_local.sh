set -x

# Self-Evaluate Tree-GAE — Mode B: 2 GPUs (training on GPU 0, vLLM reference on GPU 1)
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash self_eval_tree_gae_local.sh

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

# Launch vLLM server on GPU 1 with reference model weights
SELF_EVAL_PORT=${SELF_EVAL_PORT:-8199}
export SELF_EVAL_MODEL=${SELF_EVAL_MODEL:-$(basename $MODEL_PATH)}

echo "==> Launching local vLLM server on GPU 1 (port $SELF_EVAL_PORT)..."
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $SELF_EVAL_MODEL \
    --port $SELF_EVAL_PORT \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --no-enable-log-requests &
VLLM_PID=$!
trap "echo 'Killing vLLM server (PID=$VLLM_PID)'; kill $VLLM_PID 2>/dev/null" EXIT

echo "Waiting for vLLM server to start..."
VLLM_READY=0
set +x
for i in $(seq 1 180); do
    if curl -s http://localhost:${SELF_EVAL_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        VLLM_READY=1
        break
    fi
    sleep 1
done
set -x
if [ "$VLLM_READY" -eq 0 ]; then
    echo "ERROR: vLLM server failed to start within 180s"
    exit 1
fi

export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://localhost:${SELF_EVAL_PORT}/v1"

# EPTree params: (M=6, N=2, L=1, T=2) -> 30 leaf paths per prompt
CUDA_VISIBLE_DEVICES=0 python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=tree_gae \
    +algorithm.step_reward_type=self_eval \
    algorithm.use_xml_steps=true \
    +algorithm.step_reward_weights='[0.5, 0.5]' \
    reward_model.reward_manager=tree \
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
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    +trainer.tree_sampling=True \
    +trainer.tree_rounds=1 \
    +trainer.tree_top_n=2 \
    +trainer.tree_branches=2 \
    +trainer.tree_mask_tail_ratio=0.1 \
    +trainer.tree_step_reward_mode=la \
    +trainer.tree_overall_norm_style=token \
    +trainer.tree_use_weighted_value=False \
    +trainer.tree_weighted_value_style=sqrt \
    +algorithm.tree_ext_reward_dedup=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl-fol' \
    trainer.experiment_name="qwen1.5b_tree_gae_1epo_${DATA_NAME}_self_eval" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    ++data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    critic.data_loader_seed=42 $@
