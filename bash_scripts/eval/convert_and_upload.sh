
checkpoint_dir="/share/nlp/chenzhenbin/Workspaces/verl/checkpoints/verl/Qwen2.5-1.5B_GRPO_LogiQA_PRM_eps3_outline/global_step_1383/actor/"


output_dir="/share/nlp/chenzhenbin/Workspaces/verl/output_models/Qwen2.5-1.5B_GRPO_LogiQA_PRM_eps3_outline"

# python /home/chenzhb/Workspaces/verl/scripts/model_merger.py \
#     --backend "fsdp" \
#     --hf_upload_path 'BunnyNLP/Qwen2.5-1.5B-GRPO-Math220K' \
#     --hf_model_path ${checkpoint_dir} \
#     --local_dir ${checkpoint_dir} \
#     --target_dir ${output_dir} 


# 不上传 hf
python /share/nlp/chenzhenbin/Workspaces/verl/scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path ${checkpoint_dir} \
    --local_dir ${checkpoint_dir} \
    --target_dir ${output_dir} 
