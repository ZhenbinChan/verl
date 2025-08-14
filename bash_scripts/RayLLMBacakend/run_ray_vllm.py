#!/usr/bin/env python3
"""
Ray vLLM服务管理器 - 纯异步版本
"""

import asyncio
import sys
import os

# 添加当前目录到路径，这样可以导入ray_internal_vllm_api
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def start_service():
    """启动服务并保持运行"""
    print("🚀 启动Ray张量并行vLLM服务...")
    
    # ✅ 直接导入并调用，而不是用subprocess
    from ray_internal_vllm_api import main
    await main()

async def test_service():
    """测试服务"""
    print("🧪 测试服务...")
    
    from ray_internal_vllm_api import ray_generate, ray_batch_generate
    
    try:
        # 测试单个推理
        response = await ray_generate("Hello, how are you?")
        print(f"✅ 单个推理成功: {response}")
        
        # 测试批量推理
        responses = await ray_batch_generate(["What is AI?", "How are you?"])
        print(f"✅ 批量推理成功:")
        for i, resp in enumerate(responses):
            print(f"  {i+1}. {resp}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def quick_test():
    """快速启动并测试"""
    print("🚀 快速启动并测试...")
    
    from ray_internal_vllm_api import get_service, ray_generate, ray_batch_generate
    
    # 启动服务（只初始化，不进入无限循环）
    await get_service()
    print("✅ 服务启动完成")
    
    # 运行测试
    print("🧪 开始测试...")
    
    # 单个测试
    response = await ray_generate("Hello")
    print(f"单个推理: {response}")
    
    # 批量测试
    responses = await ray_batch_generate(["What is machine learning?", "Explain Python"])
    print("批量推理结果:")
    for i, resp in enumerate(responses):
        print(f"  {i+1}. {resp}")
    
    print("✅ 测试完成")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "test", "quick"], 
                       help="start: 启动并保持服务, test: 测试服务, quick: 快速测试")
    args = parser.parse_args()
    
    if args.action == "start":
        # 运行服务（会一直运行）
        asyncio.run(start_service())
    elif args.action == "test":
        # 只运行测试（假设服务已经在其他地方启动）
        asyncio.run(test_service())
    elif args.action == "quick":
        # 启动服务并测试，然后退出
        asyncio.run(quick_test())
