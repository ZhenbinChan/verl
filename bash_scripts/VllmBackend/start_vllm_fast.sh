python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name 'judge-model' \
    --gpu-memory-utilization 0.3 \
    --max-model-len 8192 \
    --trust-remote-code