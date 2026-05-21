"""
FOL Verifier Module for MCTS Training

核心功能:
1. 管理预计算的 Z3 Declarations
2. 将 NL 推理链转换为 Z3 FOL 代码
3. 执行 Z3 验证并返回 sat/unsat 结果

    Usage:
        from verl.utils.fol_verifier import FOLVerifier, FOLMetadata, LLMClient

        metadata = FOLMetadata(
            sample_id="logiqa_0",
            rephrased_context="...",
            entities={"Person": ["Alice", "Bob"]},
            predicates={"married_to": ["Person", "Person"]},
            z3_declaration_code="from z3 import *\n...",
            ground_truth="A",
            axioms=[]
        )

        llm_client = LLMClient(model="qwen2.5-7b-coder")
        verifier = FOLVerifier(llm_client=llm_client)
        reward = verifier.verify_step(metadata, step_text, use_llm=True)
"""

from __future__ import annotations

import json
import keyword
import os
import re
import string
import subprocess
import sys
import time
import ast
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field


# =============================================================================
# 数据类
# =============================================================================

class OutputSchema(BaseModel):
    """LLM 约束生成的输出格式"""
    data: Dict[str, List[Any]]


class EntityGroupSchema(BaseModel):
    type: str
    sort_kind: Literal["uninterpreted", "int", "real"] = "uninterpreted"
    values: List[Any] = Field(default_factory=list)


class EntityGroupsSchema(BaseModel):
    entity_groups: List[EntityGroupSchema] = Field(default_factory=list)


class PredicateSpecSchema(BaseModel):
    name: str
    arg_types: List[str] = Field(default_factory=list)
    return_type: str = "bool"


class PredicateExtractionSchema(BaseModel):
    predicates: List[PredicateSpecSchema] = Field(default_factory=list)


@dataclass
class FOLMetadata:
    """预计算的 FOL 元数据"""
    sample_id: str
    rephrased_context: str
    question_text: str = ""
    prm_mode: str = ""
    entities: Dict[str, List[str]] = field(default_factory=dict)
    predicates: Dict[str, List[str]] = field(default_factory=dict)
    z3_declaration_code: str = ""
    ground_truth: str = ""
    axioms: List[str] = field(default_factory=list)

    @staticmethod
    def _normalize_json_value(value: Any) -> Any:
        if hasattr(value, "tolist"):
            value = value.tolist()
        elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                value = value.item()
            except Exception:
                pass

        if isinstance(value, dict):
            return {str(k): FOLMetadata._normalize_json_value(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return [FOLMetadata._normalize_json_value(v) for v in value]
        if isinstance(value, list):
            return [FOLMetadata._normalize_json_value(v) for v in value]
        return value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "rephrased_context": self.rephrased_context,
            "question_text": self.question_text,
            "prm_mode": self.prm_mode,
            "entities": self._normalize_json_value(self.entities),
            "predicates": self._normalize_json_value(self.predicates),
            "z3_declaration_code": self.z3_declaration_code,
            "ground_truth": self.ground_truth,
            "axioms": self._normalize_json_value(self.axioms),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FOLMetadata":
        return cls(
            sample_id=d.get("sample_id", ""),
            rephrased_context=d.get("rephrased_context", ""),
            question_text=d.get("question_text", d.get("raw_prompt", "")),
            prm_mode=d.get("prm_mode", d.get("pipeline", "")),
            entities=cls._normalize_json_value(d.get("entities", {})),
            predicates=cls._normalize_json_value(d.get("predicates", {})),
            z3_declaration_code=d.get("z3_declaration_code", ""),
            ground_truth=d.get("ground_truth", ""),
            axioms=cls._normalize_json_value(d.get("axioms", [])),
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
        provider: str = "openai_compatible",
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        request_timeout: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        bypass_env_proxy: bool = False,
    ):
        self.provider = str(provider or "openai_compatible").lower()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.request_timeout = request_timeout
        self.extra_body = extra_body or {}
        self.bypass_env_proxy = bypass_env_proxy
        self.default_args = default_args or {
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.8,
        }
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import httpx
            if self.provider == "azure_openai":
                from openai import AzureOpenAI

                client_kwargs = {
                    "azure_endpoint": self.azure_endpoint,
                    "api_key": self.api_key,
                    "api_version": self.api_version,
                }
                if self.request_timeout is not None:
                    client_kwargs["timeout"] = self.request_timeout
                self._client = AzureOpenAI(**client_kwargs)
            else:
                from openai import OpenAI

                client_kwargs = {
                    "base_url": self.base_url,
                    "api_key": self.api_key,
                }
                if self.request_timeout is not None:
                    client_kwargs["timeout"] = self.request_timeout
                if self._should_bypass_env_proxy():
                    client_kwargs["http_client"] = httpx.Client(trust_env=False)
                self._client = OpenAI(**client_kwargs)
        return self._client

    def _should_bypass_env_proxy(self) -> bool:
        if self.bypass_env_proxy:
            return True
        hostname = urlparse(self.base_url).hostname
        return hostname in {"localhost", "127.0.0.1", "::1"}

    def _resolve_model(self, model: Optional[str] = None) -> str:
        if self.provider == "azure_openai":
            return model or self.deployment_name or self.model
        return model or self.model

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        args: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """生成文本"""
        model = self._resolve_model(model)
        merged_args = {**self.default_args, **(args or {})}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        create_kwargs = {
            "model": model,
            "messages": messages,
            **merged_args,
        }
        if self.extra_body:
            create_kwargs["extra_body"] = self.extra_body

        chat_response = self.client.chat.completions.create(**create_kwargs)
        return chat_response.choices[0].message.content

    def constrain_generate(
        self,
        prompt: str,
        format: type[BaseModel],
        model: Optional[str] = None,
        args: Optional[Dict] = None,
    ) -> BaseModel:
        """约束生成，返回结构化输出"""
        model = self._resolve_model(model)
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
        debug_dir: Optional[str] = None,
    ):
        self.llm_client = llm_client
        self.verify_timeout = verify_timeout
        self.max_retries = max_retries
        self.debug_dir = Path(debug_dir).expanduser() if debug_dir else None
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_templates: Dict[str, str] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """加载 prompt 模板"""
        prompt_files = {
            "rephrase": "rephrase.txt",
            "object_extract": "object_exctract.txt",
            "object_extract_strict": "object_exctract_structured.txt",
            "predicate_extract": "predicate_extraction.txt",
            "predicate_extract_strict": "predicate_extraction_structured.txt",
            "translate_step": "translate_step.txt",
            "correct_code": "correct_code.txt",
            "z3_declaration_global": "Z3DeclarationsGeneration1.txt",
            "z3_implication_global": "Z3ImplicationConversion1.txt",
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

    @staticmethod
    def _sanitize_symbol_name(name: Any) -> str:
        text = str(name or "").strip()
        text = re.sub(r"\W+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_").lower()
        if not text:
            text = "sym"
        if text[0].isdigit():
            text = f"sym_{text}"
        if keyword.iskeyword(text):
            text = f"{text}_sym"
        return text

    @staticmethod
    def _make_unique_symbol(base: str, used: set[str]) -> str:
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        return candidate

    @staticmethod
    def _singularize_symbol(name: str) -> str:
        if name.endswith("ies") and len(name) > 3:
            return name[:-3] + "y"
        if name.endswith("s") and not name.endswith("ss") and len(name) > 1:
            return name[:-1]
        return name

    @staticmethod
    def _builtin_sort_expr(sort_kind: Any) -> Optional[str]:
        text = str(sort_kind or "").strip()
        if not text:
            return None
        normalized = text.lower()
        if normalized in {"bool", "boolsort()"}:
            return "BoolSort()"
        if normalized in {"int", "intsort()"}:
            return "IntSort()"
        if normalized in {"real", "realsort()"}:
            return "RealSort()"
        return None

    def _build_entity_schema(self, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        generic_schema = {"entity_type", "entity_value"}.issubset(set(entities.keys()))
        structured_groups = entities.get("entity_groups")
        used_symbols: set[str] = set()

        sort_specs: list[dict[str, str]] = []
        sort_symbol_by_norm: dict[str, str] = {}

        def register_sort(sort_label: str, sort_kind: str = "uninterpreted") -> str:
            norm = self._sanitize_symbol_name(sort_label)
            if norm in sort_symbol_by_norm:
                return sort_symbol_by_norm[norm]
            builtin_expr = self._builtin_sort_expr(sort_kind)
            if builtin_expr is not None:
                sort_symbol_by_norm[norm] = builtin_expr
                sort_specs.append(
                    {
                        "label": sort_label,
                        "expr": builtin_expr,
                        "kind": sort_kind,
                        "declare": "false",
                    }
                )
                return builtin_expr

            sort_symbol = self._make_unique_symbol(f"{norm}_sort", used_symbols)
            sort_symbol_by_norm[norm] = sort_symbol
            sort_specs.append(
                {
                    "label": sort_label,
                    "expr": sort_symbol,
                    "kind": "uninterpreted",
                    "declare": "true",
                }
            )
            return sort_symbol

        constant_entries: list[tuple[str, str, str]] = []
        declared_constant_names: set[str] = set()

        fallback_sort_symbol: Optional[str] = None

        def ensure_fallback_sort() -> str:
            nonlocal fallback_sort_symbol
            if fallback_sort_symbol is None:
                fallback_sort_symbol = self._make_unique_symbol("entity_sort", used_symbols)
                sort_specs.append(
                    {
                        "label": "entity",
                        "expr": fallback_sort_symbol,
                        "kind": "uninterpreted",
                        "declare": "true",
                    }
                )
            return fallback_sort_symbol

        def infer_sort_symbol(name: str) -> str:
            norm = self._sanitize_symbol_name(name)
            singular_norm = self._singularize_symbol(norm)
            if norm in sort_symbol_by_norm:
                return sort_symbol_by_norm[norm]
            if singular_norm in sort_symbol_by_norm:
                return sort_symbol_by_norm[singular_norm]

            tokens = [self._singularize_symbol(tok) for tok in norm.split("_") if tok]
            best_match: Optional[tuple[int, str]] = None
            for sort_norm, sort_symbol in sort_symbol_by_norm.items():
                sort_singular = self._singularize_symbol(sort_norm)
                if sort_singular in tokens or sort_singular in singular_norm:
                    score = len(sort_singular)
                    if best_match is None or score > best_match[0]:
                        best_match = (score, sort_symbol)
            if best_match is not None:
                return best_match[1]
            return ensure_fallback_sort()

        if isinstance(structured_groups, list):
            for group in structured_groups:
                if not isinstance(group, dict):
                    continue
                sort_label = str(group.get("type", "")).strip()
                if not sort_label:
                    continue
                sort_expr = register_sort(sort_label, str(group.get("sort_kind", "uninterpreted")))
                for const_name in group.get("values") or []:
                    const_name = str(const_name).strip()
                    if not const_name or const_name in declared_constant_names:
                        continue
                    declared_constant_names.add(const_name)
                    const_symbol = self._make_unique_symbol(self._sanitize_symbol_name(const_name), used_symbols)
                    constant_entries.append((const_name, const_symbol, sort_expr))
        elif generic_schema:
            for sort_label in entities.get("entity_type", []) or []:
                if str(sort_label).strip():
                    register_sort(str(sort_label).strip())
            for const_name in entities.get("entity_value", []) or []:
                const_name = str(const_name).strip()
                if not const_name or const_name in declared_constant_names:
                    continue
                declared_constant_names.add(const_name)
                const_symbol = self._make_unique_symbol(self._sanitize_symbol_name(const_name), used_symbols)
                constant_entries.append((const_name, const_symbol, infer_sort_symbol(const_name)))
        else:
            for sort_label, names in entities.items():
                sort_label = str(sort_label).strip()
                if not sort_label:
                    continue
                sort_symbol = register_sort(sort_label)
                for const_name in names or []:
                    const_name = str(const_name).strip()
                    if not const_name or const_name in declared_constant_names:
                        continue
                    declared_constant_names.add(const_name)
                    const_symbol = self._make_unique_symbol(self._sanitize_symbol_name(const_name), used_symbols)
                    constant_entries.append((const_name, const_symbol, sort_symbol))

        if not sort_specs:
            ensure_fallback_sort()

        return {
            "sort_specs": sort_specs,
            "sort_symbol_by_norm": sort_symbol_by_norm,
            "constant_entries": constant_entries,
            "fallback_sort_symbol": fallback_sort_symbol or ensure_fallback_sort(),
            "used_symbols": used_symbols,
        }

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
        schema = self._build_entity_schema(entities)
        code_lines = ["# Z3 Type Declaration"]

        for sort_spec in schema["sort_specs"]:
            if sort_spec["declare"] == "true":
                code_lines.append(f"{sort_spec['expr']} = DeclareSort({sort_spec['label']!r})")

        code_lines.append("\n# Constants Definition")
        for original_name, const_symbol, sort_symbol in schema["constant_entries"]:
            if sort_symbol == "IntSort()":
                try:
                    code_lines.append(f"{const_symbol} = IntVal({int(original_name)})")
                    continue
                except (TypeError, ValueError):
                    pass
            elif sort_symbol == "RealSort()":
                try:
                    code_lines.append(f"{const_symbol} = RealVal({float(original_name)!r})")
                    continue
                except (TypeError, ValueError):
                    pass
            code_lines.append(f"{const_symbol} = Const({original_name!r}, {sort_symbol})")

        code_lines.append("\n# Variable Declarations")
        alphabet = string.ascii_lowercase
        sort_symbols = [sort_spec["expr"] for sort_spec in schema["sort_specs"]]
        for i, sort_symbol in enumerate(sort_symbols[: len(alphabet)]):
            var_name = alphabet[i]
            code_lines.append(f"{var_name} = Const({var_name!r}, {sort_symbol})")

        return "\n".join(code_lines)

    def generate_z3_functions(
        self,
        predicates: Dict[str, List[str]],
        entities: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """生成 Z3 函数/谓词声明

        Args:
            predicates: 谓词字典，格式为 {predicate_name: [arg_types]}
                例如: {"married_to": ["Person", "Person"]}

        Returns:
            Z3 函数声明代码字符串
        """
        schema = self._build_entity_schema(entities or {})
        sort_symbol_by_norm = schema["sort_symbol_by_norm"]
        fallback_sort_symbol = schema["fallback_sort_symbol"]
        used_symbols = set(schema["used_symbols"])

        def resolve_arg_sort(arg_hint: Any) -> str:
            builtin_expr = self._builtin_sort_expr(arg_hint)
            if builtin_expr is not None:
                return builtin_expr
            norm = self._sanitize_symbol_name(arg_hint)
            singular_norm = self._singularize_symbol(norm)
            if norm in sort_symbol_by_norm:
                return sort_symbol_by_norm[norm]
            if singular_norm in sort_symbol_by_norm:
                return sort_symbol_by_norm[singular_norm]
            for sort_norm, sort_symbol in sort_symbol_by_norm.items():
                sort_singular = self._singularize_symbol(sort_norm)
                tokens = [self._singularize_symbol(tok) for tok in norm.split("_") if tok]
                if sort_singular in tokens or sort_singular in singular_norm:
                    return sort_symbol
            return fallback_sort_symbol

        code_lines = ["# Z3 Function/Predicate Declaration"]

        structured_predicates = predicates.get("predicates") if isinstance(predicates, dict) else None
        if isinstance(structured_predicates, list):
            for spec in structured_predicates:
                if not isinstance(spec, dict):
                    continue
                func_name = str(spec.get("name", "")).strip()
                if not func_name:
                    continue
                func_symbol = self._make_unique_symbol(self._sanitize_symbol_name(func_name), used_symbols)
                sort_args = [resolve_arg_sort(arg_hint) for arg_hint in (spec.get("arg_types") or [])]
                return_sort = resolve_arg_sort(spec.get("return_type", "bool"))
                args_str = ", ".join([*sort_args, return_sort])
                code_lines.append(f"{func_symbol} = Function({func_name!r}, {args_str})")
            return "\n".join(code_lines)

        for func_name, arg_types in predicates.items():
            func_symbol = self._make_unique_symbol(self._sanitize_symbol_name(func_name), used_symbols)
            sort_args = [resolve_arg_sort(arg_hint) for arg_hint in (arg_types or [])]
            args_str = ", ".join([*sort_args, "BoolSort()"])
            code_lines.append(f"{func_symbol} = Function({func_name!r}, {args_str})")
        return "\n".join(code_lines)

    def build_z3_declaration_code(
        self,
        entities: Dict[str, List[str]],
        predicates: Dict[str, List[str]],
    ) -> str:
        z3_declaration_code = self.generate_z3_declarations(entities)
        z3_function_code = self.generate_z3_functions(predicates, entities=entities)
        return z3_declaration_code + "\n\n" + z3_function_code

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
        schema_variant: str = "legacy",
        feedback: Optional[str] = None,
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

        if schema_variant == "strict_v1":
            prompt = self._get_prompt("object_extract_strict", context=context, question=question, options=options)
            if feedback:
                prompt += (
                    "\n\n## Previous attempt failed validation\n"
                    f"{feedback}\n"
                    "Regenerate the full output and obey the schema exactly."
                )
            result = self.llm_client.constrain_generate(prompt, EntityGroupsSchema, args=args)
            return result.model_dump()

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
        schema_variant: str = "legacy",
        feedback: Optional[str] = None,
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

        prompt_name = "predicate_extract"
        response_schema: type[BaseModel] = OutputSchema
        if schema_variant == "strict_v1":
            prompt_name = "predicate_extract_strict"
            response_schema = PredicateExtractionSchema

        prompt = self._get_prompt(
            prompt_name,
            context=context,
            question=question,
            options=options,
            obj_list=json.dumps(obj_list, ensure_ascii=False, indent=2),
        )
        if feedback:
            prompt += (
                "\n\n## Previous attempt failed validation\n"
                f"{feedback}\n"
                "Regenerate the full output and obey the schema exactly."
            )
        result = self.llm_client.constrain_generate(prompt, response_schema, args=args)
        if schema_variant == "strict_v1":
            return result.model_dump()
        return result.data

    def translate_step_to_z3(
        self,
        rephrased_context: str,
        declaration_code: str,
        step_content: str,
        args: Optional[Dict] = None,
        debug_record: Optional[Dict[str, Any]] = None,
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
        extracted = self._extract_python_block(result)
        if debug_record is not None:
            debug_record["translate_prompt"] = prompt
            debug_record["translate_raw_response"] = result
            debug_record["translated_z3_code"] = extracted
        return extracted

    def correct_z3_code(
        self,
        code: str,
        error: str,
        args: Optional[Dict] = None,
        debug_record: Optional[Dict[str, Any]] = None,
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
        extracted = self._extract_python_block(result)
        if debug_record is not None:
            debug_record.setdefault("correction_attempts", []).append(
                {
                    "prompt": prompt,
                    "raw_response": result,
                    "corrected_z3_code": extracted,
                    "error": error,
                }
            )
        return extracted

    def _extract_python_block(self, code: str) -> str:
        """从代码中提取 python 块"""
        py_pattern = r"```python\s+(.*?)```"
        matches = re.findall(py_pattern, code, re.DOTALL)
        return matches[-1].strip() if matches else code.strip()

    def generate_global_declaration(
        self,
        question_text: str,
        args: Optional[Dict] = None,
        debug_record: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a global FOL PRM Z3 declaration from the full question text."""
        if self.llm_client is None:
            raise RuntimeError("LLM client required for global FOL declaration generation")
        system_prompt = self._get_prompt("z3_declaration_global")
        result = self.llm_client.generate(question_text, system_prompt=system_prompt, args=args)
        extracted = self._extract_python_block(result)
        self.validate_z3_declaration_code(extracted)
        if debug_record is not None:
            debug_record["declaration_prompt"] = question_text
            debug_record["declaration_raw_response"] = result
            debug_record["z3_declaration_code"] = extracted
        return extracted

    @staticmethod
    def validate_z3_declaration_code(code: str) -> None:
        """Validate that declaration code is executable and contains no solver calls."""
        forbidden_patterns = [
            r"\bsolver\s*\.add\s*\(",
            r"\bs\s*\.add\s*\(",
            r"\.check\s*\(",
            r"\bSolver\s*\(",
        ]
        for pattern in forbidden_patterns:
            if re.search(pattern, code):
                raise ValueError(f"Declaration code contains forbidden solver logic: {pattern}")
        namespace: Dict[str, Any] = {}
        exec("from z3 import *\n" + code, namespace, namespace)

    def _translate_global_reasoning_to_z3(
        self,
        question_text: str,
        declaration_code: str,
        reasoning_steps: str,
        args: Optional[Dict] = None,
        debug_record: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.llm_client is None:
            raise RuntimeError("LLM client required for global FOL implication conversion")
        system_prompt = self._get_prompt("z3_implication_global")
        full_input = (
            f"Question:\n{question_text}\n\n"
            f"Z3 Declaration:\n{declaration_code}\n\n"
            f"Reasoning steps:\n{reasoning_steps}"
        )
        result = self.llm_client.generate(full_input, system_prompt=system_prompt, args=args)
        extracted = self._extract_python_block(result)
        if debug_record is not None:
            debug_record["implication_prompt"] = full_input
            debug_record["implication_raw_response"] = result
            debug_record["implication_code"] = extracted
        return extracted

    @staticmethod
    def parse_global_logic_steps(code_str: str) -> List[Dict[str, Any]]:
        """Parse premises_i/conclusion_i assignments from generated Python code."""
        tree = ast.parse(code_str)
        raw_data: Dict[int, Dict[str, List[str]]] = {}

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                if not name.startswith(("premises_", "conclusion_")):
                    continue
                try:
                    ptype, idx_text = name.rsplit("_", 1)
                    idx = int(idx_text)
                except ValueError:
                    continue
                raw_data.setdefault(idx, {"premises": [], "conclusion": []})
                if ptype == "premises" and isinstance(node.value, ast.List):
                    raw_data[idx]["premises"] = [ast.unparse(e) for e in node.value.elts]
                else:
                    key = "premises" if ptype == "premises" else "conclusion"
                    raw_data[idx][key] = [ast.unparse(node.value)]

        for idx in sorted(raw_data.keys()):
            new_premises = []
            for premise in raw_data[idx]["premises"]:
                p_strip = premise.strip()
                if p_strip.startswith("conclusion_"):
                    try:
                        ref_idx = int(p_strip.rsplit("_", 1)[-1])
                        new_premises.append(raw_data[ref_idx]["conclusion"][0])
                        continue
                    except Exception:
                        pass
                new_premises.append(premise)
            raw_data[idx]["premises"] = new_premises

        return [
            {
                "step_index": idx,
                "premises": raw_data[idx]["premises"],
                "conclusion": raw_data[idx]["conclusion"],
            }
            for idx in sorted(raw_data.keys())
            if raw_data[idx]["conclusion"]
        ]

    def run_global_logic_step(
        self,
        declaration_code: str,
        parsed_step: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        premises = parsed_step.get("premises") or []
        conclusion = (parsed_step.get("conclusion") or [""])[0]
        premises_str = ", ".join(premises) if premises else "True"
        z3_code = f"""
from z3 import *
{declaration_code}
solver = Solver()
solver.add(And({premises_str}))
solver.add(Not({conclusion}))

check_res = solver.check()
if check_res == unsat:
    print("SUCCESS_ENTAILED")
elif check_res == sat:
    print("FAILED_CONTRADICT")
else:
    print("UNKNOWN")
""".strip()
        result = self.run_code(z3_code, timeout=timeout)
        result["code"] = z3_code
        return result

    def verify_global_step(
        self,
        metadata: FOLMetadata,
        reasoning_steps: str,
        question_text: Optional[str] = None,
        args: Optional[Dict] = None,
        debug_dump: bool = False,
    ) -> float:
        """Verify the final step in a full reasoning prefix for global_fol_prm."""
        if not reasoning_steps or "<step" not in reasoning_steps:
            return 0.0
        declaration_code = metadata.z3_declaration_code
        question = question_text or metadata.question_text
        if not declaration_code:
            if not question:
                return 0.0
            declaration_code = self.generate_global_declaration(question, args=args)
            metadata.z3_declaration_code = declaration_code
        if not question:
            question = metadata.rephrased_context
        if not question:
            return 0.0

        debug_record: Dict[str, Any] = {
            "prm_mode": "global_fol_prm",
            "reasoning_steps": reasoning_steps,
            "question_text": question,
            "llm_args": dict(args or {}),
        }
        try:
            implication_code = self._translate_global_reasoning_to_z3(
                question_text=question,
                declaration_code=declaration_code,
                reasoning_steps=reasoning_steps,
                args=args,
                debug_record=debug_record,
            )
            parsed_steps = self.parse_global_logic_steps(implication_code)
            debug_record["parsed_steps"] = parsed_steps
            if not parsed_steps:
                debug_record["final_score"] = 0.0
                if debug_dump:
                    self._dump_debug_artifacts(metadata=metadata, step_text=reasoning_steps, payload=debug_record)
                return 0.0
            result = self.run_global_logic_step(declaration_code, parsed_steps[-1])
            debug_record["run_result"] = result
            output = result.get("output") or ""
            final_score = 1.0 if result.get("success") and "SUCCESS_ENTAILED" in output else 0.0
            debug_record["final_score"] = final_score
            if debug_dump:
                self._dump_debug_artifacts(metadata=metadata, step_text=reasoning_steps, payload=debug_record)
            return final_score
        except Exception:
            if debug_dump:
                debug_record["exception"] = repr(sys.exc_info()[1])
                self._dump_debug_artifacts(metadata=metadata, step_text=reasoning_steps, payload=debug_record)
            raise

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
        # Entailment check: premises /\ not(conclusion) is UNSAT iff premises entail conclusion.
        z3_code += "s.add(Not(conclusion_fol))\n\n"
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

    def _debug_step_stem(self, metadata: FOLMetadata, step_text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", step_text.strip())[:48].strip("_") or "step"
        timestamp = int(time.time() * 1000)
        return f"{metadata.sample_id}_{timestamp}_{slug}"

    def _dump_debug_artifacts(
        self,
        *,
        metadata: FOLMetadata,
        step_text: str,
        payload: Dict[str, Any],
    ) -> Optional[Path]:
        if self.debug_dir is None:
            return None

        stem = self._debug_step_stem(metadata, step_text)
        json_path = self.debug_dir / f"{stem}.json"
        py_path = self.debug_dir / f"{stem}.py"

        serializable_payload = {
            "sample_id": metadata.sample_id,
            "step_text": step_text,
            **payload,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_payload, f, ensure_ascii=False, indent=2)

        wrapped_code = payload.get("wrapped_z3_code", None)
        if wrapped_code:
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(wrapped_code)

        return json_path

    def correct_loop(
        self,
        code: str,
        args: Optional[Dict] = None,
        debug_record: Optional[Dict[str, Any]] = None,
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
            code = self.correct_z3_code(code, res["error"], args=args, debug_record=debug_record)
            res = self.run_code(code)
            tries += 1
            # 每次迭代增加温度，鼓励模型生成更多样化的修正方案
            if args and "temperature" in args:
                args = {**args, "temperature": min(args["temperature"] + 0.1, 1.0)}

        res["code"] = code
        res["tries"] = tries
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
        debug_dump: bool = False,
    ) -> float:
        """验证单个 step

        Args:
            metadata: FOL 元数据
            step_text: 需要验证的 step 文本
            use_llm: 是否使用 LLM 翻译
            args: LLM 生成参数

        Returns:
            1.0 if unsat (结论从前提推出), 0.0 otherwise
        """
        # 1. 解析 premise/conclusion
        premises, conclusion = self.get_premise_conclusion(step_text)

        if not premises or not conclusion:
            return 0.0

        if not use_llm:
            raise RuntimeError(
                "FOL verification requires use_llm=True. "
                "Format-only fallback has been removed."
            )
        if self.llm_client is None:
            raise RuntimeError(
                "FOL verification requires an LLM client. "
                "Format-only fallback has been removed."
            )

        debug_record: Dict[str, Any] = {
            "premises": premises,
            "conclusion": conclusion,
            "use_llm": use_llm,
            "llm_args": dict(args or {}),
        }
        try:
            # 2. 调用 translate_step_to_z3 (LLM)
            trans_code = self.translate_step_to_z3(
                rephrased_context=metadata.rephrased_context,
                declaration_code=metadata.z3_declaration_code,
                step_content=step_text,
                args=args,
                debug_record=debug_record,
            )

            # 3. 包装为完整代码
            wrapped_code = self.wrap_z3_code(metadata.z3_declaration_code, trans_code)
            debug_record["wrapped_z3_code"] = wrapped_code

            # 4. 执行 Z3 验证
            result = self.run_code(wrapped_code)
            debug_record["initial_run_result"] = result

            if not result["success"]:
                # 5. 如失败，调用 correct_loop
                corrected_result = self.correct_loop(wrapped_code, args=args, debug_record=debug_record)
                debug_record["corrected_run_result"] = corrected_result
                if corrected_result["success"]:
                    result = corrected_result
                    debug_record["wrapped_z3_code_after_correction"] = corrected_result.get("code", wrapped_code)
                else:
                    debug_record["final_score"] = 0.0
                    dump_path = self._dump_debug_artifacts(metadata=metadata, step_text=step_text, payload=debug_record)
                    if dump_path is not None:
                        debug_record["debug_dump_path"] = str(dump_path)
                    return 0.0

            # 6. 解析结果: UNSAT = 正确 (premises 推出 conclusion)
            #            SAT/UNKNOWN = 错误
            output = result["output"]
            final_score = 1.0 if "UNSAT" in output else 0.0
            debug_record["final_output"] = output
            debug_record["final_score"] = final_score
            if debug_dump:
                dump_path = self._dump_debug_artifacts(metadata=metadata, step_text=step_text, payload=debug_record)
                if dump_path is not None:
                    debug_record["debug_dump_path"] = str(dump_path)
            return final_score

        except Exception:
            if debug_dump:
                debug_record["exception"] = repr(sys.exc_info()[1])
                self._dump_debug_artifacts(metadata=metadata, step_text=step_text, payload=debug_record)
            raise

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
                raise KeyError(f"Missing FOL metadata for sample_id={sample_id!r}.")

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
        raise NotImplementedError(
            "FOLVerifierPrecomputed is no longer supported. "
            "Training-time FOL step verification now requires an LLM client."
        )

    def verify_sample(self, sample_id: str, step_text: str) -> float:
        raise NotImplementedError(
            "FOLVerifierPrecomputed.verify_sample is no longer supported."
        )


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
            metadata = FOLMetadata.from_dict(item["fol_metadata"])
            result[str(sample_id)] = metadata

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
