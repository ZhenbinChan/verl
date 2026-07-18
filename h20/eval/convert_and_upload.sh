base_model_dir="/2024133105/Workspaces/llms/Qwen3-8B"

checkpoint_dir="/2024133105/Workspaces/verl/ckpt/verl/qwen3-8b_steprl_wothk/global_step_320/actor"


output_dir="/2024133105/Workspaces/verl/hf_model/qwen3-8b_steprl_wothk_s320"

# python /home/chenzhb/Workspaces/verl/scripts/model_merger.py \
#     --backend "fsdp" \
#     --hf_upload_path 'BunnyNLP/Qwen2.5-1.5B-GRPO-Math220K' \
#     --hf_model_path ${checkpoint_dir} \
#     --local_dir ${checkpoint_dir} \
#     --target_dir ${output_dir} 


# 不上传 hf
python /2024133105/Workspaces/verl/scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path ${base_model_dir} \
    --local_dir ${checkpoint_dir} \
    --target_dir ${output_dir} 
#    --test \
#    --test_hf_dir ${base_model_dir}