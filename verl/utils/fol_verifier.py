"""
FOL Verifier Module for MCTS Training

核心功能:
1. 管理预计算的 Z3 Declarations
2. 将 NL 推理链转换为 Z3 FOL 代码
3. 执行 Z3 验证并返回 sat/unsat 结果

Usage:
    # 预计算模式 (训练时使用)
    from verl.utils.fol_verifier import FOLVerifier, FOLMetadata

    metadata = FOLMetadata(
        sample_id="logiqa_0",
        rephrased_context="...",
        entities={"Person": ["Alice", "Bob"]},
        predicates={"married_to": ["Person", "Person"]},
        z3_declaration_code="from z3 import *\n...",
        ground_truth="A",
        axioms=[]
    )

    verifier = FOLVerifier()
    reward = verifier.verify_step(metadata, step_text, use_llm=True)
"""

from __future__ import annotations

import json
import os
import re
import string
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel


# =============================================================================
# 数据类
# =============================================================================

class OutputSchema(BaseModel):
    """LLM 约束生成的输出格式"""
    data: Dict[str, List[Any]]


@dataclass
class FOLMetadata:
    """预计算的 FOL 元数据"""
    sample_id: str
    rephrased_context: str
    entities: Dict[str, List[str]] = field(default_factory=dict)
    predicates: Dict[str, List[str]] = field(default_factory=dict)
    z3_declaration_code: str = ""
    ground_truth: str = ""
    axioms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "rephrased_context": self.rephrased_context,
            "entities": self.entities,
            "predicates": self.predicates,
            "z3_declaration_code": self.z3_declaration_code,
            "ground_truth": self.ground_truth,
            "axioms": self.axioms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FOLMetadata":
        return cls(
            sample_id=d.get("sample_id", ""),
            rephrased_context=d.get("rephrased_context", ""),
            entities=d.get("entities", {}),
            predicates=d.get("predicates", {}),
            z3_declaration_code=d.get("z3_declaration_code", ""),
            ground_truth=d.get("ground_truth", ""),
            axioms=d.get("axioms", []),
        )


# =============================================================================
# LLM 客户端
# =============================================================================

class LLMClient:
    """LLM 客户端封装"""

    def __init__(
        self,
        base_url: str = "http://localhost:4869/v1",
        api_key: str = "EMPTY",
        model: str = "qwen2.5-3b",
        default_args: Optional[Dict] = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.default_args = default_args or {
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.8,
        }
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        args: Optional[Dict] = None,
    ) -> str:
        """生成文本"""
        model = model or self.model
        merged_args = {**self.default_args, **(args or {})}

        chat_response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **merged_args,
        )
        return chat_response.choices[0].message.content

    def constrain_generate(
        self,
        prompt: str,
        format: type[BaseModel],
        model: Optional[str] = None,
        args: Optional[Dict] = None,
    ) -> BaseModel:
        """约束生成，返回结构化输出"""
        model = model or self.model
        merged_args = {**self.default_args, **(args or {})}

        chat_response = self.client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=format,
            **merged_args,
        )
        return chat_response.choices[0].message.parsed


# =============================================================================
# FOL 验证器
# =============================================================================

class FOLVerifier:
    """FOL 验证器 - 封装 Z3 验证逻辑

    支持两种模式:
    1. 在线模式: 使用 LLM 翻译 NL step 为 Z3 代码
    2. 预计算模式: 使用预编译的 Z3 代码验证
    """

    # prompts 路径
    PROMPT_DIR = Path(__file__).parent.parent.parent / "mcts_utils" / "prompts"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        verify_timeout: float = 10.0,
        max_retries: int = 3,
    ):
        self.llm_client = llm_client
        self.verify_timeout = verify_timeout
        self.max_retries = max_retries
        self._prompt_templates: Dict[str, str] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """加载 prompt 模板"""
        prompt_files = {
            "rephrase": "rephrase.txt",
            "object_extract": "object_exctract.txt",
            "predicate_extract": "predicate_extraction.txt",
            "translate_step": "translate_step.txt",
            "correct_code": "correct_code.txt",
        }

        for name, filename in prompt_files.items():
            path = self.PROMPT_DIR / filename
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self._prompt_templates[name] = f.read()

    def _get_prompt(self, name: str, **kwargs) -> str:
        """获取并填充 prompt 模板"""
        if name not in self._prompt_templates:
            raise ValueError(f"Unknown prompt: {name}")
        template = Template(self._prompt_templates[name])
        return template.safe_substitute(**kwargs)

    # =========================================================================
    # 预计算相关的纯函数
    # =========================================================================

    def generate_z3_declarations(self, entities: Dict[str, List[str]]) -> str:
        """生成 Z3 类型和常量声明

        Args:
            entities: 实体字典，格式为 {entity_type: [entity_names]}
                例如: {"Person": ["Alice", "Bob"], "University": ["MIT"]}

        Returns:
            Z3 声明代码字符串
        """
        code_lines = []

        # Z3 Type Declaration
        code_lines.append("# Z3 Type Declaration")
        for entity_type in entities.keys():
            line = f"{entity_type} = DeclareSort('{entity_type}')"
            code_lines.append(line)

        # Constants Definition
        code_lines.append("\n# Constants Definition")
        for entity_type, names in entities.items():
            for name in names:
                # 处理空格，将实体名中的空格替换为下划线
                formatted_name = name.replace(" ", "_")
                line = f"{formatted_name} = Const('{formatted_name}', {entity_type})"
                code_lines.append(line)

        # Variable Declarations
        code_lines.append("\n# Variable Declarations")
        alphabet = string.ascii_lowercase

        for i, entity_type in enumerate(entities.keys()):
            if i < len(alphabet):
                var_name = alphabet[i]
                line = f"{var_name} = Const('{var_name}', {entity_type})"
                code_lines.append(line)
            else:
                break

        return "\n".join(code_lines)

    def generate_z3_functions(self, predicates: Dict[str, List[str]]) -> str:
        """生成 Z3 函数/谓词声明

        Args:
            predicates: 谓词字典，格式为 {predicate_name: [arg_types]}
                例如: {"married_to": ["Person", "Person"]}

        Returns:
            Z3 函数声明代码字符串
        """
        code_lines = []
        code_lines.append("# Z3 Function/Predicate Declaration")

        for func_name, types in predicates.items():
            types_str = ", ".join(types)
            line = f"{func_name} = Function('{func_name}', {types_str})"
            code_lines.append(line)
        return "\n".join(code_lines)

    def get_step_list(self, text_content: str) -> List[str]:
        """从文本中提取 <step>...</step> 块

        Args:
            text_content: 包含 <step> 标签的文本

        Returns:
            step 块内容列表
        """
        pattern = r"<step.*?>(.*?)</step>"
        matches = re.findall(pattern, text_content, flags=re.DOTALL)
        return [content.strip() for content in matches]

    def get_premise_conclusion(self, step_content: str) -> Tuple[List[str], Optional[str]]:
        """从 step 中提取 premise 和 conclusion

        Args:
            step_content: 单个 step 的内容

        Returns:
            (premise_list, conclusion) 元组
        """
        premise_list = []
        pattern = r"<premise>(.*?)</premise>"
        matches = re.findall(pattern, step_content, flags=re.DOTALL)
        premise_list = [content.strip() for content in matches]

        pattern = r"<conclusion>(.*?)</conclusion>"
        matches = re.findall(pattern, step_content, flags=re.DOTALL)
        conclusion = matches[-1] if matches else None

        return premise_list, conclusion

    # =========================================================================
    # 需要 LLM 的函数
    # =========================================================================

    def rephrase(
        self,
        context: str,
        question: str,
        options: str,
        args: Optional[Dict] = None,
    ) -> str:
        """将上下文和问题改写成结构化的逻辑描述

        Args:
            context: 题目背景
            question: 问题
            options: 选项
            args: LLM 生成参数

        Returns:
            改写后的文本
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client required for rephrase")

        prompt = self._get_prompt("rephrase", context=context, question=question, options=options)
        return self.llm_client.generate(prompt, args=args)

    def object_extract(
        self,
        context: str,
        question: str,
        options: str,
        args: Optional[Dict] = None,
    ) -> Dict[str, List[str]]:
        """提取实体类型和常量

        Args:
            context: 题目背景
            question: 问题
            options: 选项
            args: LLM 生成参数

        Returns:
            实体字典，格式为 {entity_type: [entity_names]}
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client required for object_extract")

        prompt = self._get_prompt("object_extract", context=context, question=question, options=options)
        result = self.llm_client.constrain_generate(prompt, OutputSchema, args=args)
        return result.data

    def predicate_extract(
        self,
        context: str,
        question: str,
        options: str,
        obj_list: Dict[str, List[str]],
        args: Optional[Dict] = None,
    ) -> Dict[str, List[str]]:
        """提取谓词/关系

        Args:
            context: 题目背景
            question: 问题
            options: 选项
            obj_list: 实体字典
            args: LLM 生成参数

        Returns:
            谓词字典，格式为 {predicate_name: [arg_types]}
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client required for predicate_extract")

        prompt = self._get_prompt(
            "predicate_extract",
            context=context,
            question=question,
            options=options,
            obj_list=obj_list,
        )
        result = self.llm_client.constrain_generate(prompt, OutputSchema, args=args)
        return result.data

    def translate_step_to_z3(
        self,
        rephrased_context: str,
        declaration_code: str,
        step_content: str,
        args: Optional[Dict] = None,
    ) -> str:
        """将 NL step 翻译为 Z3 FOL 代码

        Args:
            rephrased_context: 改写后的上下文
            declaration_code: Z3 声明代码
            step_content: NL step 内容
            args: LLM 生成参数

        Returns:
            Z3 代码字符串
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client required for translate_step_to_z3")

        prompt = self._get_prompt(
            "translate_step",
            context=rephrased_context,
            declaration=declaration_code,
            step=step_content,
        )
        result = self.llm_client.generate(prompt, args=args)
        return self._extract_python_block(result)

    def correct_z3_code(
        self,
        code: str,
        error: str,
        args: Optional[Dict] = None,
    ) -> str:
        """修正有错误的 Z3 代码

        Args:
            code: 原始代码
            error: 错误信息
            args: LLM 生成参数

        Returns:
            修正后的代码
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client required for correct_z3_code")

        prompt = self._get_prompt("correct_code", code=code, error=error)
        result = self.llm_client.generate(prompt, args=args)
        return self._extract_python_block(result)

    def _extract_python_block(self, code: str) -> str:
        """从代码中提取 python 块"""
        py_pattern = r"```python\s+(.*?)```"
        matches = re.findall(py_pattern, code, re.DOTALL)
        return matches[-1].strip() if matches else code.strip()

    # =========================================================================
    # 纯计算函数
    # =========================================================================

    def wrap_z3_code(self, declaration: str, expression: str) -> str:
        """包装为完整的可执行 Z3 代码

        Args:
            declaration: Z3 声明代码
            expression: Z3 表达式代码

        Returns:
            完整的 Z3 验证代码
        """
        z3_code = ""
        z3_code += "from z3 import *\n\n"
        z3_code += "s = Solver()\n\n"
        z3_code += "s.reset()\n\n"
        z3_code += "# --- Declarations ---\n\n"
        z3_code += declaration + "\n\n"
        z3_code += "# --- Expressions ---\n\n"
        z3_code += expression + "\n\n"
        z3_code += "s.add(premise_fol)\n\n"
        z3_code += "s.add(conclusion_fol)\n\n"
        z3_code += "result = s.check()\n"
        z3_code += "print(f'Result: {result}')\n"
        z3_code += "if result == sat:\n"
        z3_code += "    print('SAT')\n"
        z3_code += "elif result == unsat:\n"
        z3_code += "    print('UNSAT')\n"
        z3_code += "else:\n"
        z3_code += "    print('UNKNOWN')\n"
        return z3_code

    def run_code(self, code_string: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """执行 Z3 代码

        Args:
            code_string: Z3 代码字符串
            timeout: 超时时间（秒）

        Returns:
            {"success": bool, "output": str, "error": str or None}
        """
        timeout = timeout or self.verify_timeout
        try:
            result = subprocess.run(
                [sys.executable, "-c", code_string],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                return {"success": True, "output": result.stdout, "error": None}
            else:
                return {"success": False, "output": result.stdout, "error": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "TimeoutError: 代码执行超时"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def correct_loop(
        self,
        code: str,
        args: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """代码修正循环

        Args:
            code: 初始代码
            args: LLM 生成参数

        Returns:
            {"success": bool, "output": str, "error": str or None}
        """
        res = self.run_code(code)
        tries = 0

        while not res["success"] and tries < self.max_retries:
            code = self.correct_z3_code(code, res["error"], args=args)
            res = self.run_code(code)
            tries += 1
            # 每次迭代增加温度，鼓励模型生成更多样化的修正方案
            if args and "temperature" in args:
                args = {**args, "temperature": min(args["temperature"] + 0.1, 1.0)}

        return res

    # =========================================================================
    # 主验证函数
    # =========================================================================

    def verify_step(
        self,
        metadata: FOLMetadata,
        step_text: str,
        use_llm: bool = True,
        args: Optional[Dict] = None,
    ) -> float:
        """验证单个 step

        Args:
            metadata: FOL 元数据
            step_text: 需要验证的 step 文本
            use_llm: 是否使用 LLM 翻译（False 则只做格式检查）
            args: LLM 生成参数

        Returns:
            1.0 if unsat (结论从前提推出), 0.0 otherwise
        """
        # 1. 解析 premise/conclusion
        premises, conclusion = self.get_premise_conclusion(step_text)

        if not premises or not conclusion:
            return 0.0

        if not use_llm or self.llm_client is None:
            # 只做格式检查
            return 1.0 if (len(premises) >= 1 and conclusion) else 0.0

        try:
            # 2. 调用 translate_step_to_z3 (LLM)
            trans_code = self.translate_step_to_z3(
                rephrased_context=metadata.rephrased_context,
                declaration_code=metadata.z3_declaration_code,
                step_content=step_text,
                args=args,
            )

            # 3. 包装为完整代码
            wrapped_code = self.wrap_z3_code(metadata.z3_declaration_code, trans_code)

            # 4. 执行 Z3 验证
            result = self.run_code(wrapped_code)

            if not result["success"]:
                # 5. 如失败，调用 correct_loop
                corrected_result = self.correct_loop(wrapped_code, args=args)
                if corrected_result["success"]:
                    result = corrected_result
                else:
                    return 0.0

            # 6. 解析结果: UNSAT = 正确 (premises 推出 conclusion)
            #            SAT/UNKNOWN = 错误
            output = result["output"]
            if "UNSAT" in output:
                return 1.0
            else:
                return 0.0

        except Exception as e:
            print(f"[FOL Warning] Verification failed: {e}")
            return 0.0

    def verify_step_batch(
        self,
        metadata_map: Dict[str, FOLMetadata],
        step_texts: List[str],
        sample_ids: List[str],
        use_llm: bool = True,
        args: Optional[Dict] = None,
    ) -> List[float]:
        """批量验证 step

        Args:
            metadata_map: sample_id -> FOLMetadata 的映射
            step_texts: step 文本列表
            sample_ids: 对应的 sample_id 列表
            use_llm: 是否使用 LLM 翻译
            args: LLM 生成参数

        Returns:
            奖励列表
        """
        rewards = []
        for step_text, sample_id in zip(step_texts, sample_ids):
            if sample_id not in metadata_map:
                rewards.append(0.0)
                continue

            metadata = metadata_map[sample_id]
            reward = self.verify_step(metadata, step_text, use_llm=use_llm, args=args)
            rewards.append(reward)

        return rewards


# =============================================================================
# 预计算模式的 FOL 验证器
# =============================================================================

class FOLVerifierPrecomputed(FOLVerifier):
    """预计算模式的 FOL 验证器 - 不需要 LLM 调用

    用于训练阶段，直接使用预计算的元数据进行验证
    """

    def __init__(self, precomputed_data: Dict[str, FOLMetadata]):
        """初始化预计算验证器

        Args:
            precomputed_data: sample_id -> FOLMetadata 的映射
        """
        self.precomputed_data = precomputed_data
        # 强制不使用 LLM
        super().__init__(llm_client=None)

    def verify_sample(self, sample_id: str, step_text: str) -> float:
        """验证单个样本

        Args:
            sample_id: 样本 ID (用于查找预计算数据)
            step_text: MCTS 生成的完整推理链

        Returns:
            1.0 if correct, 0.0 if incorrect
        """
        if sample_id not in self.precomputed_data:
            return 0.0

        metadata = self.precomputed_data[sample_id]
        return self.verify_step(metadata, step_text, use_llm=False)


# =============================================================================
# 工具函数
# =============================================================================

def load_fol_metadata(metadata_path: str) -> Dict[str, FOLMetadata]:
    """从 JSON 文件加载 FOL 元数据

    Args:
        metadata_path: 元数据文件路径

    Returns:
        sample_id -> FOLMetadata 的映射
    """
    if not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for item in data:
        if item.get("fol_metadata"):
            sample_id = item.get("sample_id", item.get("extra_info", {}).get("index"))
            result[str(sample_id)] = FOLMetadata.from_dict(item["fol_metadata"])

    return result


def save_fol_metadata(
    metadata_map: Dict[str, FOLMetadata],
    output_path: str,
) -> None:
    """保存 FOL 元数据到 JSON 文件

    Args:
        metadata_map: sample_id -> FOLMetadata 的映射
        output_path: 输出文件路径
    """
    data = []
    for sample_id, metadata in metadata_map.items():
        data.append({
            "sample_id": sample_id,
            "fol_metadata": metadata.to_dict(),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
