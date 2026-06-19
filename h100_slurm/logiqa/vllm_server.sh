# ======================= Start VLLM as RM ======================== #
CACHE_ROOT=~/run/Workspaces/vllm_runtime_cache

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

VLLM_PORT=4869
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model /data/home/scyb224/run/Workspaces/LLMs/Qwen3-8B-Base \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --gpu-memory-utilization 0.5 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --served-model-name eval-model \
    --trust-remote-code > vllm_server_outlines_backend.log 2>&1 &

echo $! > vllm_server_outlines_backend.pid
echo "[info] Start RL Training Done"

# ======================== End VLLM as RM ======================== #