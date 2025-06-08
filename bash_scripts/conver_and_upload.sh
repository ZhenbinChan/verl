
checkpoint_dir="/data/home/scyb224/Workspace/verl/checkpoints/Verl/Qwen2.5-1.5b_GRPO_math220k_500_function_rm/global_step_15/actor/"
output_dir="/data/home/scyb224/Workspace/verl/output_models/Qwen2.5-1.5B-GRPO-Math220K"


python /data/home/scyb224/Workspace/verl/scripts/model_merger.py \
    --backend "fsdp" \
    --hf_upload_path 'BunnyNLP/Qwen2.5-1.5B-GRPO-Math220K' \
    --hf_model_path ${checkpoint_dir} \
    --local_dir ${checkpoint_dir} \
    --target_dir ${output_dir} 