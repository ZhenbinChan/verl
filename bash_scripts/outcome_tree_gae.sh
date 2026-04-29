set -x

HOME=~
MODEL_PATH=~/run/models/Qwen2.5-1.5B-Instruct
DATA_NAME=${DATA_NAME:-logiqa2k_prompt_v2}
DATA_DIR=${DATA_DIR:-"$HOME/run/work/verl/data/${DATA_NAME}"}
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_ENTITY=${WANDB_ENTITY:-verl-fol}
# ray stop --force
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Sanity check
echo "Using $NNODES nodes for training..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# TreeRL Example: EPTree-based tree search for RL training
# Based on TreeRL paper (arXiv:2506.11902)
# EPTree params: (M=4, N=1, L=1, T=3) -> 16 leaf paths per prompt

# Outcome-Only Tree-GAE training
# 相比外接 PRM 的版本，这里去掉了 +algorithm.step_reward_type 和 weights 的注入。
# 底层会自动退化为只使用基于叶子正确率推导出来的 (GA+LA)/\sqrt{n} 作为唯一的 advantage 信号。
#
# Previous checkpoint/eval schedule:
# trainer.save_freq=-1
# trainer.max_actor_ckpt_to_keep=0
# trainer.test_freq=20
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=tree_gae \
    algorithm.use_xml_steps=true \
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
    trainer.project_name='verl-fol-2' \
    trainer.experiment_name="qwen1.5b_tree_gae_1epo_${DATA_NAME}_outcome_only_4_1_3" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    ++data.seed=42 \
    actor_rollout_ref.actor.data_loader_seed=42 \
    critic.data_loader_seed=42 $@
