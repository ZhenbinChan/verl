CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model /home/chenzhb/Workspaces/LLMs/Qwen2.5-Coder-7B-Instruct \
    --host 0.0.0.0 \
    --port 4869 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --served-model-name qwen2.5-3b \
    --trust-remote-code \
    --structured-outputs-config '{"backend":"outlines"}'