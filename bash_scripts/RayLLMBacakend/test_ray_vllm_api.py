#!/usr/bin/env python3
"""
Ray内部vLLM API测试脚本 - 简化版
"""

import ray
import asyncio
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(__file__))
from ray_internal_vllm_api import ray_generate, ray_batch_generate

async def test_tensor_parallel():
    """测试张量并行vLLM"""
    
    print("🚀 测试张量并行vLLM（模型切片到2张GPU）...")
    
    if not ray.is_initialized():
        ray.init()
    
    # 单个测试
    print("\n🔍 单个推理测试...")
    response = await ray_generate("请解释什么是机器学习？")
    print(f"回复: {response}")
    
    # 批量测试
    print("\n📦 批量推理测试...")
    prompts = ["1+1=?", "什么是深度学习？", "Python的优势是什么？"]
    responses = await ray_batch_generate(prompts)
    
    for prompt, response in zip(prompts, responses):
        print(f"Q: {prompt}")
        print(f"A: {response}\n")
    
    print("✅ 测试完成！")

if __name__ == "__main__":
    asyncio.run(test_tensor_parallel())
