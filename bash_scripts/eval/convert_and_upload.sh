base_model_dir="/share/nlp/chenzhenbin/Workspaces/LLMs/Qwen3-8B-Base"

checkpoint_dir="/share/nlp/chenzhenbin/Workspaces/verl/ckpt/verl/qwen3-8b_logiqa_grpo_promptV1_n64_formatp/global_step_115/actor"


output_dir="/share/nlp/chenzhenbin/Workspaces/verl/output_models/qwen3-8b-base_warmup_115"

# python /home/chenzhb/Workspaces/verl/scripts/model_merger.py \
#     --backend "fsdp" \
#     --hf_upload_path 'BunnyNLP/Qwen2.5-1.5B-GRPO-Math220K' \
#     --hf_model_path ${checkpoint_dir} \
#     --local_dir ${checkpoint_dir} \
#     --target_dir ${output_dir} 


# 不上传 hf
python /share/nlp/chenzhenbin/Workspaces/verl/scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path ${base_model_dir} \
    --local_dir ${checkpoint_dir} \
    --target_dir ${output_dir} \
    --test \
    --test_hf_dir ${base_model_dir}

