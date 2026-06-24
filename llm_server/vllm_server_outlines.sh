#!/usr/bin/env bash

# 更高的版本，启用 outlines： --structured-outputs-config '{"backend":"outlines"}' 
# 所有运行时缓存都放到 ~/run 下，避免写 ~/.cache 或 home quota
CACHE_ROOT=/share/nlp/chenzhenbin/Workspaces/vllm_runtime_cache

export HF_HOME=$CACHE_ROOT/huggingface
export HF_MODULES_CACHE=$HF_HOME/modules
export TRANSFORMERS_CACHE=$HF_HOME/transformers

export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export TORCH_HOME=$CACHE_ROOT/torch
export VLLM_CACHE_ROOT=$CACHE_ROOT/vllm
export XDG_CACHE_HOME=$CACHE_ROOT/xdg

export VLLM_NO_USAGE_STATS=1

mkdir -p \
  "$HF_HOME" \
  "$HF_MODULES_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$XDG_CACHE_HOME"


CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model /share/nlp/chenzhenbin/Workspaces/LLMs/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 4869 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --served-model-name eval-model \
    --trust-remote-code \
    --guided-decoding-backend xgrammar
