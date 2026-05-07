MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen2.5-Coder-7B-Instruct
HOST=0.0.0.0
VLLM_PORT=4869
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=8192
TENSOR_PARALLEL_SIZE=2
SERVED_MODEL_NAME=qwen2.5-coder-7b-instruct

nohup CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --host ${HOST} \
    --port ${VLLM_PORT} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --served-model-name ${SERVED_MODEL_NAME} \
    --trust-remote-code \
    --structured-outputs-config '{"backend":"outlines"}' > vllm_server_outlines_backend.log 2>&1 &

echo $! > vllm_server_outlines_backend.pid

for i in $(seq 1 180); do
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "VLLM server Started after ${i}s"
        VLLM_READY=1
        break
    fi
    sleep 1
done