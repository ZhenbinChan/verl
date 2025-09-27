#!/usr/bin/env python3
"""
Ray集群内部vLLM推理API - 张量并行版本
将7B模型切片到2张GPU上
"""

import ray
import asyncio
import logging
from typing import List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_gpus=2)  # 使用2张完整GPU进行张量并行
class VLLMTensorParallelActor:
    """vLLM张量并行推理actor - 模型切片到2张GPU"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None
        
    def initialize(self):
        """初始化vLLM引擎（张量并行）"""
        from vllm import LLM, SamplingParams
        
        # 关键：tensor_parallel_size=2 将模型切片到2张GPU
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=2,  # 🔥 模型切片到2张GPU
            gpu_memory_utilization=0.3,
            trust_remote_code=True,
            max_model_len=8192,
        )
        
        self.sampling_params = SamplingParams(
            max_tokens=1024,  # 每次推理最多生成1024个token
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            stop=["<|im_end|>", "\n\n"]
        )
        
        logger.info("✅ 模型已切片到2张GPU")
        return "Tensor parallel model loaded"
    
    def generate(self, prompt: str) -> str:
        """单个推理"""
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """批量推理"""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

@ray.remote
class RayVLLMService:
    """简化的Ray vLLM服务管理器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.actor = None
        
    async def start_service(self):
        """启动服务"""
        logger.info(f"启动张量并行vLLM服务: {self.model_path}")
        
        # 创建单个张量并行actor
        self.actor = VLLMTensorParallelActor.remote(self.model_path)
        
        # ✅ 修正：只用 await，不用 ray.get()
        result = await self.actor.initialize.remote()
        logger.info(f"初始化结果: {result}")
        logger.info("✅ 服务启动成功")
        return True
    
    async def generate(self, prompt: str) -> str:
        """单个推理"""
        # ✅ 修正：只用 await
        return await self.actor.generate.remote(prompt)
    
    async def batch_generate(self, prompts: List[str]) -> List[str]:
        """批量推理"""
        # ✅ 修正：只用 await
        return await self.actor.batch_generate.remote(prompts)

# 全局服务实例
_service = None

async def get_service(model_path: str = "Qwen/Qwen2.5-7B-Instruct"):
    """获取服务实例"""
    global _service
    if _service is None:
        _service = RayVLLMService.remote(model_path)
        await _service.start_service.remote()
    return _service

# 简化的API函数
async def ray_generate(prompt: str) -> str:
    """推理单个prompt"""
    service = await get_service()
    return await service.generate.remote(prompt)

async def ray_batch_generate(prompts: List[str]) -> List[str]:
    """批量推理"""
    service = await get_service()
    return await service.batch_generate.remote(prompts)

async def start_service_only():
    """只启动服务，不进入无限循环"""
    if not ray.is_initialized():
        ray.init()
    
    # 初始化服务
    await get_service()
    print("✅ Ray vLLM服务启动完成，可以接受请求")

# 启动脚本
async def main():
    """启动服务并保持运行"""
    await start_service_only()
    
    # 运行一个简单测试
    response = await ray_generate("Hello, how are you?")
    print(f"测试响应: {response}")
    
    # 保持运行
    print("🔄 服务保持运行中... (Ctrl+C 停止)")
    try:
        while True:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        print("Service stopped")

if __name__ == "__main__":
    asyncio.run(main())
