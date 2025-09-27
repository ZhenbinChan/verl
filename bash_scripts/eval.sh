export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
MODEL=Qwen/Qwen2.5-1.5B

# MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"


MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"

# OUTPUT_DIR=./eval_output/$MODEL
OUTPUT_DIR=./eval_output/Qwen2.5-1.5B


# LogiQA
TASK=agieval:ogiqa-en
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR


# # GSM8K
# TASK=gsm8k
# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # AIME 2024
# TASK=aime24
# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # MATH-500
# TASK=math_500
# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # GPQA Diamond
# TASK=gpqa:diamond
# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # LiveCodeBench
# lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 