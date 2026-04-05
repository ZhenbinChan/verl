"""
NL to FOL translation utilities — Small LLM (SLM) version.

Adapted from ZhenbinChan/verl pipeline branch (ZhenbinChan's approach).
Uses a local vLLM-served small model (e.g. qwen2.5-3b) to:
  1. Rephrase the problem for clarity
  2. Extract objects and predicates (structured output)
  3. Generate Z3 declarations from extracted entities (code-based, not LLM)
  4. Translate reasoning steps into Z3 code
  5. Auto-correct Z3 code errors with LLM retry loop

All API calls are parameterized via ``api_config`` dict.
"""

import concurrent.futures
import json
import os
import re
import string
import subprocess
import sys
import tempfile
import threading
from functools import wraps
from pathlib import Path
from string import Template
from typing import Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# OpenAI client cache — reuse connections across calls
# ---------------------------------------------------------------------------
_client_cache: dict[tuple, OpenAI] = {}
_client_cache_lock = threading.Lock()


def _get_client(api_key: str, base_url: str | None, timeout: float) -> OpenAI:
    """Return a cached OpenAI client, creating one if needed."""
    cache_key = (api_key, base_url, timeout)
    with _client_cache_lock:
        client = _client_cache.get(cache_key)
        if client is None:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=1,
            )
            _client_cache[cache_key] = client
        return client


# ---------------------------------------------------------------------------
# Prompt files — ship with verl under verl/prompts/
# ---------------------------------------------------------------------------
PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts"
REPHRASE_PROMPT = PROMPT_ROOT / "rephrase.txt"
OBJECT_EXTRACT_PROMPT = PROMPT_ROOT / "object_extract.txt"
PREDICATE_EXTRACTION_PROMPT = PROMPT_ROOT / "predicate_extraction.txt"
GENERATE_PROMPT = PROMPT_ROOT / "generate.txt"
TRANSLATE_STEP_PROMPT = PROMPT_ROOT / "translate_step.txt"
CORRECT_CODE_PROMPT = PROMPT_ROOT / "correct_code.txt"


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# LLM call wrapper (local vLLM compatible)
# ---------------------------------------------------------------------------

def _get_default_api_config() -> dict:
    """Build default API config for local vLLM."""
    return {
        "model": os.environ.get("FOL_SLM_MODEL", "qwen2.5-3b"),
        "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
        "base_url": os.environ.get("FOL_SLM_BASE_URL", "http://localhost:4869/v1"),
        "temperature": 0.1,
        "max_tokens": 4096,
        "top_p": 0.8,
    }


def _call_llm(
    user_prompt: str,
    *,
    api_config: Optional[dict] = None,
) -> str:
    """Call an OpenAI-compatible chat API (local vLLM or remote)."""
    cfg = _get_default_api_config()
    if api_config:
        cfg.update({k: v for k, v in api_config.items() if v is not None})

    timeout = cfg.pop("timeout", 120)
    client = _get_client(cfg["api_key"], cfg.get("base_url"), timeout)
    completion = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": user_prompt}],
        temperature=cfg.get("temperature", 0.1),
        max_tokens=cfg.get("max_tokens", 4096),
        top_p=cfg.get("top_p", 0.8),
    )
    return completion.choices[0].message.content or ""


def _call_llm_structured(
    user_prompt: str,
    *,
    api_config: Optional[dict] = None,
) -> Optional[dict]:
    """Call LLM and try to parse the response as JSON dict.

    Falls back to regex extraction if structured parsing is unavailable.
    """
    response = _call_llm(user_prompt, api_config=api_config)
    # Try to extract JSON from response
    json_pattern = re.compile(r"\{[\s\S]*\}", re.DOTALL)
    match = json_pattern.search(response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def rephrase(
    context: str,
    question: str,
    options: str = "",
    *,
    api_config: Optional[dict] = None,
) -> str:
    """Rephrase the problem for clarity using LLM."""
    template = _load_prompt(REPHRASE_PROMPT)
    prompt = Template(template).safe_substitute(
        context=context, question=question, options=options
    )
    return _call_llm(prompt, api_config=api_config)


def object_extract(
    context: str,
    question: str,
    options: str = "",
    *,
    api_config: Optional[dict] = None,
) -> dict:
    """Extract key entities from the problem using structured LLM output.

    Returns:
        dict mapping entity_type -> list[entity_name]
    """
    template = _load_prompt(OBJECT_EXTRACT_PROMPT)
    prompt = Template(template).safe_substitute(
        context=context, question=question, options=options
    )
    result = _call_llm_structured(prompt, api_config=api_config)
    if result and isinstance(result, dict):
        return result
    return {}


def predicate_extract(
    context: str,
    question: str,
    options: str = "",
    objectives: Optional[dict] = None,
    *,
    api_config: Optional[dict] = None,
) -> dict:
    """Extract predicates/relations using structured LLM output.

    Returns:
        dict mapping predicate_name -> list[type_str]
    """
    template = _load_prompt(PREDICATE_EXTRACTION_PROMPT)
    obj_list = str(objectives) if objectives else ""
    prompt = Template(template).safe_substitute(
        context=context, question=question, options=options, obj_list=obj_list
    )
    result = _call_llm_structured(prompt, api_config=api_config)
    if result and isinstance(result, dict):
        return result
    return {}


def generate_z3_declarations(entities: dict) -> str:
    """Generate Z3 type/constant/variable declarations from extracted entities.

    This is code-based (not LLM-based) — deterministic generation from
    structured entity extraction results.

    Args:
        entities: dict mapping entity_type -> list[entity_name]

    Returns:
        Z3 Python declaration code string.
    """
    code_lines = []

    code_lines.append("# Z3 Type Declaration")
    for entity_type in entities.keys():
        code_lines.append(f"{entity_type} = DeclareSort('{entity_type}')")

    code_lines.append("\n# Constants Definition")
    for entity_type, names in entities.items():
        # Ensure names is a flat list of strings (LLM may return ints, nested lists, etc.)
        if not isinstance(names, list):
            names = [names]
        for name in names:
            if isinstance(name, list):
                # Flatten nested lists
                for sub in name:
                    formatted_name = str(sub).replace(" ", "_")
                    code_lines.append(f"{formatted_name} = Const('{formatted_name}', {entity_type})")
                continue
            formatted_name = str(name).replace(" ", "_")
            code_lines.append(f"{formatted_name} = Const('{formatted_name}', {entity_type})")

    code_lines.append("\n# Variable Declarations")
    alphabet = string.ascii_lowercase
    for i, entity_type in enumerate(entities.keys()):
        if i < len(alphabet):
            var_name = alphabet[i]
            code_lines.append(f"{var_name} = Const('{var_name}', {entity_type})")
        else:
            break

    return "\n".join(code_lines)


def generate_z3_functions(predicates: dict) -> str:
    """Generate Z3 Function declarations from extracted predicates.

    Args:
        predicates: dict mapping func_name -> list[type_str]

    Returns:
        Z3 Python function declaration code string.
    """
    code_lines = ["# Z3 Function/Predicate Declaration"]
    for func_name, types in predicates.items():
        # Ensure types is a list of strings
        if not isinstance(types, list):
            types = [str(types)]
        else:
            types = [str(t) for t in types]
        types_str = ", ".join(types)
        code_lines.append(f"{func_name} = Function('{func_name}', {types_str})")
    return "\n".join(code_lines)


# ---------------------------------------------------------------------------
# Step translation and Z3 execution
# ---------------------------------------------------------------------------

def translate_step_to_z3(
    rephrased_context: str,
    declaration_code: str,
    step_content: str,
    *,
    api_config: Optional[dict] = None,
) -> str:
    """Translate a single reasoning step into executable Z3 code.

    Returns the wrapped Z3 code ready for execution.
    """
    template = _load_prompt(TRANSLATE_STEP_PROMPT)
    prompt = Template(template).safe_substitute(
        context=rephrased_context, declaration=declaration_code, step=step_content
    )
    trans_output = _call_llm(prompt, api_config=api_config)
    trans_code = extract_python_block(trans_output)
    return wrap_z3_code(declaration_code, trans_code)


def extract_python_block(code: str) -> str:
    """Extract last Python code block from LLM response."""
    py_pattern = r"```python\s+(.*?)```"
    matches = re.findall(py_pattern, code, re.DOTALL)
    if matches:
        return matches[-1]
    return code.strip()


def wrap_z3_code(declaration: str, expression: str) -> str:
    """Assemble a complete Z3 script from declarations and expressions."""
    z3_code = "from z3 import *\n\n"
    z3_code += "s = Solver()\n\n"
    z3_code += "s.reset()\n"
    z3_code += "# --- Declarations ---\n\n"
    z3_code += declaration + "\n\n"
    z3_code += "# --- Expressions ---\n\n"
    z3_code += expression + "\n\n"
    z3_code += "s.add(premise_fol)\n\n"
    z3_code += "s.add(conclusion_fol)\n\n"
    z3_code += "result = s.check()\n"
    z3_code += "print(f'Result: {result}')\n"
    z3_code += "if result == sat:\n"
    z3_code += "    print('Model:', s.model())\n"
    z3_code += "    print(1.0)\n"
    z3_code += "else:\n"
    z3_code += "    print(0.0)\n"
    return z3_code


def get_step_list(text_content: str) -> list[str]:
    """Extract step contents from ``<step>...</step>`` tags."""
    pattern = r"<step.*?>(.*?)</step>"
    matches = re.findall(pattern, text_content, flags=re.DOTALL)
    return [content.strip() for content in matches]


def get_premise_conclusion(step_content: str) -> tuple[list[str], Optional[str]]:
    """Parse premise and conclusion from a step block."""
    premise_list = []
    pattern = r"<premise>(.*?)</premise>"
    matches = re.findall(pattern, step_content, flags=re.DOTALL)
    for content in matches:
        premise_list.append(content.strip())

    pattern = r"<conclusion>(.*?)</conclusion>"
    matches = re.findall(pattern, step_content, flags=re.DOTALL)
    conclusion = matches[-1].strip() if matches else None
    return premise_list, conclusion


# ---------------------------------------------------------------------------
# Code execution and auto-correction
# ---------------------------------------------------------------------------

def run_code(code_string: str, timeout: float = 10.0) -> dict:
    """Execute Python code string in a subprocess.

    Returns:
        dict with keys: success (bool), output (str), error (str or None)
    """
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
        return {"success": False, "output": "", "error": "RuntimeError: code execution timeout"}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def correct_z3_code(
    code: str,
    error: str,
    *,
    api_config: Optional[dict] = None,
) -> str:
    """Use LLM to fix erroneous Z3 code.

    Returns the corrected Python code.
    """
    template = _load_prompt(CORRECT_CODE_PROMPT)
    prompt = Template(template).safe_substitute(code=code, error=error)
    fix_output = _call_llm(prompt, api_config=api_config)
    return extract_python_block(fix_output)


def correct_loop(
    code: str,
    *,
    api_config: Optional[dict] = None,
    max_tries: int = 8,
) -> dict:
    """Execute Z3 code with auto-correction loop.

    If execution fails, use LLM to fix the code and retry, up to ``max_tries``.
    Temperature increases by 0.1 each retry to encourage diversity.

    Returns:
        dict with keys: success, output, error
    """
    cfg = dict(api_config or {})
    res = run_code(code)
    tries = 0

    while not res["success"] and tries < max_tries:
        error_msg = res.get("error", "Unknown error")
        code = correct_z3_code(code, error_msg, api_config=cfg)
        res = run_code(code)
        tries += 1
        # Increase temperature each retry
        cfg["temperature"] = cfg.get("temperature", 0.1) + 0.1

    return res


# ---------------------------------------------------------------------------
# High-level API (used by step reward)
# ---------------------------------------------------------------------------

_preprocess_cache: dict[tuple, tuple[str, str]] = {}
_preprocess_locks: dict[tuple, threading.Lock] = {}
_preprocess_global_lock = threading.Lock()

def thread_safe_cache(func):
    """Decorator to add thread-safe caching to a function."""
    @wraps(func)
    def wrapper(context: str, question: str, options: str = "", *, api_config: Optional[dict] = None):
        # 1. Build a unique cache key
        cache_key = (context, question, options)
        # 2. Check cache without lock first, if hit, return immediately
        if cache_key in _preprocess_cache:
            return _preprocess_cache[cache_key]
        # Ensure only one thread preprocesses the same input at a time
        # 3. Get or create a specialized lock for this cache key
        with _preprocess_global_lock:
            if cache_key not in _preprocess_locks:
                _preprocess_locks[cache_key] = threading.Lock()
            key_lock = _preprocess_locks[cache_key]
        # 4. Acquire the key-specific lock to preprocess
        with key_lock:
            # 5. Check cache again inside lock (double-checked locking)
            if cache_key in _preprocess_cache:
                return _preprocess_cache[cache_key]
            # 6. Call the original function to preprocess -- cache miss, so LLM calls will happen here
            result = func(context, question, options, api_config=api_config)
            # 7. Store result in cache before releasing lock
            _preprocess_cache[cache_key] = result
            return result
        # 8. Release the lock automatically with 'with' statement
    return wrapper

@thread_safe_cache
def slm_preprocess(
    context: str,
    question: str,
    options: str = "",
    *,
    api_config: Optional[dict] = None,
) -> tuple[str, str]:
    """Full SLM preprocessing pipeline.

    Returns:
        (rephrased_context, declaration_code)
    """
    # rephrased = rephrase(context, question, options, api_config=api_config)
    # entities = object_extract(context, question, options, api_config=api_config)
    # Parallelize rephrase and object extraction since they are independent
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        rephrase_future = executor.submit(rephrase, context, question, options, api_config=api_config)
        object_future = executor.submit(object_extract, context, question, options, api_config=api_config)
        rephrased = rephrase_future.result()
        entities = object_future.result()
    predicates = predicate_extract(
        context, question, options, objectives=entities, api_config=api_config
    )
    z3_decl = generate_z3_declarations(entities)
    z3_funcs = generate_z3_functions(predicates)
    declaration_code = z3_decl + "\n" + z3_funcs
    return rephrased, declaration_code


def translate_and_verify_step_slm(
    rephrased_context: str,
    declaration_code: str,
    step_text: str,
    *,
    api_config: Optional[dict] = None,
) -> float:
    """Translate a step to Z3 and verify with auto-correction.

    Returns:
        1.0 if SAT (step is consistent), 0.0 otherwise.
    """
    max_tries = (api_config or {}).get("max_tries", 8)
    try:
        exe_code = translate_step_to_z3(
            rephrased_context, declaration_code, step_text, api_config=api_config
        )
        result = correct_loop(exe_code, api_config=api_config, max_tries=max_tries)

        if result["success"] and result.get("output"):
            output = result["output"].strip()
            lines = output.splitlines()
            for line in lines:
                if "1.0" in line:
                    return 1.0
            # Try last line as numeric
            try:
                return float(lines[-1])
            except (ValueError, IndexError):
                pass
        return 0.0
    except Exception:
        return 0.0
