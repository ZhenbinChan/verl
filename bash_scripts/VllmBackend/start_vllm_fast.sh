python -m vllm.entrypoints.openai.api_server \
    --model /data/home/scyb224/Workspace/LLMs/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name 'judge-model' \
    --gpu-memory-utilization 0.2 \
    --max-model-len 8192 \
    --trust-remote-code