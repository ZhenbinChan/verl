#!/usr/bin/env bash

# 更高的版本，启用 outlines： --structured-outputs-config '{"backend":"outlines"}' 

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model /home/chenzhb/Workspaces/LLMs/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 4869 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --served-model-name eval-model \
    --trust-remote-code \
    --guided-decoding-backend xgrammar
