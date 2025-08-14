MODEL_PATH=/share/nlp/chenzhenbin/Workspaces/LLMs/Qwen2.5-7B-Instruct
VLLM_HOST=0.0.0.0
VLLM_PORT=8000
GPU_MEMORY_UTIL=0.2
MAX_MODEL_LEN=4096
TENSOR_PARALLEL_SIZE=2

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}


print_info "启动vLLM服务器...  "
print_info "模型路径: $MODEL_PATH"
print_info "服务地址: http://${VLLM_HOST}:${VLLM_PORT}"


nohup python ./start_vllm_server.py  \
    --model "$MODEL_PATH" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" > vllm.log 2>&1 & echo $! > run.pid
