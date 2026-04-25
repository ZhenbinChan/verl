set -x

# FOL Tree-GAE — Remote: uses external API (no local vLLM needed)
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   export OPENAI_BASE_URL=https://api.openai.com/v1  # or compatible endpoint
#   bash fol_tree_gae_remote.sh
HOME=~
MODEL_PATH=~/run/models/Qwen2.5-1.5B-Instruct
DATA_NAME=logiqa2k
DATA_DIR="$HOME/run/work/verl/data/${DATA_NAME}"
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
# ray stop --force
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Sanity check
echo "Using $NNODES nodes for training..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# API configuration for LLM-based step rewards (FOL, self_eval, etc.)
# Script-local defaults. These intentionally override stale ~/.bashrc values.
# If you need different settings for one run, edit this block or override in CLI.
export OPENAI_API_KEY=${OPENAI_API_KEY:-"sk-YOUR-KEY-HERE"}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-"https://api.siliconflow.cn/v1"}
export FOL_MODEL=${FOL_MODEL:-"Qwen/Qwen3.6-35B-A3B"}
export FOL_RPM=${FOL_RPM:-200}
export FOL_OPENAI_TPM=${FOL_OPENAI_TPM:-60000}
export FOL_OPENAI_MAX_INFLIGHT=${FOL_OPENAI_MAX_INFLIGHT:-4}

if [[ "$OPENAI_BASE_URL" =~ ^https?://(127\.0\.0\.1|localhost)(:|/|$) ]]; then
    echo "Using local API endpoint at $OPENAI_BASE_URL; skipping SSH tunnel and proxy setup."
    export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
    export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
else
    # 1. 建立隧道到 login node 上的 mihomo (端口 17897)
    echo "Building SSH tunnel to ${SLURM_SUBMIT_HOST}:17897 ..."
    pkill -f "L 17897:127.0.0.1:17897" 2>/dev/null || true
    sleep 2
    ssh -i ~/.ssh/id_ed25519 \
        -o StrictHostKeyChecking=no \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o TCPKeepAlive=yes \
        -N -f -L 17897:127.0.0.1:17897 scyb676@$SLURM_SUBMIT_HOST 2>/dev/null
    sleep 2

    if ! ss -tln 2>/dev/null | grep -q ':17897'; then
        echo "ERROR: SSH tunnel to 17897 failed to bind. Aborting." >&2
        exit 1
    fi
    echo "Tunnel up."

    export http_proxy=http://127.0.0.1:17897
    export https_proxy=http://127.0.0.1:17897
    export HTTP_PROXY=http://127.0.0.1:17897
    export HTTPS_PROXY=http://127.0.0.1:17897
    export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1,.local,10.*,192.168.*,*.sock"
    export no_proxy="$NO_PROXY"
fi

# Tree-GAE training (remote API for FOL rewards)
# EPTree params: (M=4, N=1, L=1, T=3) -> 16 leaf paths per prompt
# +algorithm.fol_verify_with_cumulative_steps=true to enable step history on FOL evaluation
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=tree_gae \
    +algorithm.step_reward_type=fol \
    +algorithm.fol_max_tries=1 \
    +algorithm.fol_timeout=10 \
    +algorithm.api_timeout=200 \
    algorithm.use_xml_steps=true \
    +algorithm.step_reward_weights='[0.8, 0.2]' \
    +algorithm.penalty_max_steps=12 \
    +algorithm.penalty_on_truncated=true \
    +algorithm.penalty_on_multi_boxed=true \
    +algorithm.penalty_on_bad_format=true \
    +algorithm.penalty_score=-1.0 \
    reward_model.reward_manager=tree \
    reward.num_workers=4 \
    +algorithm.step_reward_max_workers=4 \
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
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    +trainer.tree_sampling=True \
    +trainer.tree_rounds=1 \
    +trainer.tree_top_n=1 \
    +trainer.tree_branches=3 \
    +trainer.tree_mask_tail_ratio=0.1 \
    +trainer.tree_step_reward_mode=la \
    +trainer.tree_overall_norm_style=token \
    +trainer.tree_use_weighted_value=False \
    +trainer.tree_weighted_value_style=sqrt \
    +algorithm.tree_ext_reward_dedup=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl-fol' \
    trainer.experiment_name="qwen1.5b_tree_gae_1epo_${DATA_NAME}_fol_4_1_3" \
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
