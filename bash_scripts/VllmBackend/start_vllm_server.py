#!/usr/bin/env python3
"""
Simple vLLM server startup script for LLM-as-a-Judge inference
"""

import argparse
import subprocess
import sys
import os


def start_vllm_server(model_path, host="0.0.0.0", port=8000, gpu_memory_utilization=0.9, 
                     max_model_len=4096, tensor_parallel_size=1):
    """
    Start vLLM server for online inference
    
    Args:
        model_path: Path to the model directory
        host: Server host (default: 0.0.0.0)
        port: Server port (default: 8000)
        gpu_memory_utilization: GPU memory utilization ratio (default: 0.9)
        max_model_len: Maximum model sequence length (default: 4096)
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
    """
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--served-model-name", "judge-model",
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes"
    ]
    
    print(f"Starting vLLM server with command:")
    print(" ".join(cmd))
    print(f"Server will be available at: http://{host}:{port}")
    print(f"Model: {model_path}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Start vLLM server for LLM-as-a-Judge")
    parser.add_argument("--model", required=True, help="Path to the model directory")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, 
                       help="GPU memory utilization ratio (default: 0.9)")
    parser.add_argument("--max-model-len", type=int, default=4096, 
                       help="Maximum model sequence length (default: 4096)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Number of GPUs for tensor parallelism (default: 1)")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model path does not exist: {args.model}")
        sys.exit(1)
    
    start_vllm_server(
        model_path=args.model,
        host=args.host,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size
    )


if __name__ == "__main__":
    main()
