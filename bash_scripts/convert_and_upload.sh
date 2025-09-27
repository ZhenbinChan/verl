
checkpoint_dir="./checkpoints/verl/Qwen2.5-Math-1.5B_PPO_MATH_ALL_epoch10/global_step_2340/actor"


output_dir="./output_models/Qwen2.5-Math-1.5B_PPO_MATH_ALL_epoch10"

# python ./scripts/model_merger.py \
#     --backend "fsdp" \
#     --hf_upload_path 'BunnyNLP/Qwen2.5-1.5B-GRPO-Math220K' \
#     --hf_model_path ${checkpoint_dir} \
#     --local_dir ${checkpoint_dir} \
#     --target_dir ${output_dir} 


# 不上传 hf
python ./scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path ${checkpoint_dir} \
    --local_dir ${checkpoint_dir} \
    --target_dir ${output_dir} 