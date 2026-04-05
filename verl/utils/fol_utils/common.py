"""
Shared infrastructure for FOL (First-Order Logic) verification.

Extracted from nl2fol.py and nl2fol_slm.py to eliminate duplication.
All FOL pipelines (direct, structured) and translation modes (implication, assertion)
share these utilities.
"""

import ast
import concurrent.futures
import contextlib
import io
import json
import multiprocessing
import os
import re
import signal
import string
import subprocess
import sys
import tempfile
import threading
import traceback
from functools import wraps
from pathlib import Path
from string import Template
from typing import Any, Optional, Union

from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompt path constants
# ---------------------------------------------------------------------------
PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts"
# Direct pipeline
Z3_DECLARATION_PROMPT = PROMPT_ROOT / "z3_declaration_generation.txt"
Z3_IMPLICATION_PROMPT = PROMPT_ROOT / "z3_implication_conversion.txt"
# Structured pipeline
REPHRASE_PROMPT = PROMPT_ROOT / "rephrase.txt"
OBJECT_EXTRACT_PROMPT = PROMPT_ROOT / "object_extract.txt"
PREDICATE_EXTRACTION_PROMPT = PROMPT_ROOT / "predicate_extraction.txt"
TRANSLATE_STEP_PROMPT = PROMPT_ROOT / "translate_step.txt"
CORRECT_CODE_PROMPT = PROMPT_ROOT / "correct_code.txt"


def load_prompt(path: Union[str, Path]) -> str:
    """Load and strip a prompt template file."""
    return Path(path).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# OpenAI client management (connection pooling)
# ---------------------------------------------------------------------------
_client_cache: dict[tuple, OpenAI] = {}
_client_cache_lock = threading.Lock()


def get_client(api_key: str, base_url: str | None, timeout: float) -> OpenAI:
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
# LLM call wrappers
# ---------------------------------------------------------------------------

def _get_default_api_config() -> dict:
    """Build default API config from environment variables."""
    return {
        "model": os.environ.get("FOL_MODEL", os.environ.get("FOL_SLM_MODEL", "gpt-4o-mini")),
        "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
        "base_url": os.environ.get("OPENAI_BASE_URL", os.environ.get("FOL_SLM_BASE_URL", None)),
        "temperature": 0.2,
        "max_tokens": 4096,
        "top_p": 0.8,
    }


def call_llm(
    user_prompt: str,
    *,
    api_config: Optional[dict] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Call an OpenAI-compatible chat API with connection pooling.

    Args:
        user_prompt: User message content.
        api_config: Dict with keys: model, api_key, base_url, temperature, max_tokens, top_p.
        system_prompt: Optional system message.

    Returns:
        The assistant's response text.
    """
    cfg = _get_default_api_config()
    if api_config:
        cfg.update({k: v for k, v in api_config.items() if v is not None})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    timeout = cfg.pop("timeout", 120)
    top_p = cfg.pop("top_p", 0.8)
    client = get_client(cfg["api_key"], cfg.get("base_url"), timeout)
    completion = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=cfg.get("temperature", 0.2),
        max_tokens=cfg.get("max_tokens", 4096),
        top_p=top_p,
        n=1,
    )
    return completion.choices[0].message.content or ""


def call_llm_structured(
    user_prompt: str,
    *,
    api_config: Optional[dict] = None,
) -> Optional[dict]:
    """Call LLM and parse response as JSON dict.

    Falls back to regex extraction if structured parsing is unavailable.
    """
    response = call_llm(user_prompt, api_config=api_config)
    json_pattern = re.compile(r"\{[\s\S]*\}", re.DOTALL)
    match = json_pattern.search(response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Text extraction utilities
# ---------------------------------------------------------------------------

def extract_python_block(text: str, strategy: str = "last") -> str:
    """Extract Python code from fenced code blocks.

    Args:
        text: LLM response text.
        strategy: "last" returns the last code block,
                  "all" joins all blocks.

    Returns:
        Extracted code, or stripped text if no blocks found.
    """
    pattern = re.compile(r"```python\s+(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return text.strip()
    if strategy == "all":
        return "\n\n".join(matches)
    return matches[-1]


def parse_step_tags(step_text: str) -> dict:
    """Parse <premise> and <conclusion> tags from a step block.

    Returns:
        dict with keys 'premises' (list[str]) and 'conclusion' (str | None).
    """
    premise_pattern = re.compile(r"<premise>(.*?)</premise>", re.DOTALL)
    premises = [p.strip() for p in premise_pattern.findall(step_text)]

    conclusion_pattern = re.compile(r"<conclusion>(.*?)</conclusion>", re.DOTALL)
    conclusion_matches = conclusion_pattern.findall(step_text)
    conclusion = conclusion_matches[-1].strip() if conclusion_matches else None

    return {"premises": premises, "conclusion": conclusion}


def get_step_list(text_content: str) -> list[str]:
    """Extract step contents from ``<step>...</step>`` tags."""
    pattern = r"<step.*?>(.*?)</step>"
    matches = re.findall(pattern, text_content, flags=re.DOTALL)
    return [content.strip() for content in matches]


def parse_python_logic_steps(code_str: str) -> list[dict]:
    """Parse Python code with ``premises_N`` and ``conclusion_N`` assignments.

    Used after the implication-mode LLM translates reasoning steps into Z3 code.
    Returns a list of dicts, each with ``premises`` and ``conclusion`` in FOL string form.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return []

    raw_data = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name.startswith(("premises_", "conclusion_")):
                ptype, idx = name.split("_")[0], int(name.split("_")[-1])
                raw_data.setdefault(idx, {"premises": [], "conclusion": []})
                if ptype == "premises" and isinstance(node.value, ast.List):
                    raw_data[idx]["premises"] = [ast.unparse(e) for e in node.value.elts]
                else:
                    raw_data[idx][ptype] = [ast.unparse(node.value)]

    # Dereference: replace ``conclusion_X`` references in premises
    for idx in sorted(raw_data.keys()):
        new_premises = []
        for p in raw_data[idx]["premises"]:
            p_strip = p.strip()
            if p_strip.startswith("conclusion_"):
                ref_idx = int(p_strip.split("_")[-1])
                if ref_idx in raw_data and raw_data[ref_idx]["conclusion"]:
                    new_premises.append(raw_data[ref_idx]["conclusion"][0])
                else:
                    new_premises.append(p)
            else:
                new_premises.append(p)
        raw_data[idx]["premises"] = new_premises

    return [
        {
            "step_index": i,
            "premises": raw_data[i]["premises"],
            "conclusion": raw_data[i]["conclusion"],
        }
        for i in sorted(raw_data.keys())
    ]


# ---------------------------------------------------------------------------
# Problem extraction (from prompt or extra_info)
# ---------------------------------------------------------------------------

def extract_fol_problem(
    prompt_text: str, extra_info: dict | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Extract (context, question, options) from extra_info or prompt XML tags.

    Priority: structured fields in extra_info > regex on prompt_text XML tags.
    """
    extra_info = extra_info or {}

    context = extra_info.get("fol_context", None)
    question = extra_info.get("fol_question", None)
    options = extra_info.get("fol_options", None)

    if not context:
        m = re.search(r"<Context>(.*?)</Context>", prompt_text, re.DOTALL)
        context = m.group(1).strip() if m else None
    if not question:
        m = re.search(r"<Question>(.*?)</Question>", prompt_text, re.DOTALL)
        question = m.group(1).strip() if m else None
    if not options:
        m = re.search(r"<Options>(.*?)</Options>", prompt_text, re.DOTALL)
        options = m.group(1).strip() if m else None

    return context, question, options


# ---------------------------------------------------------------------------
# Code execution sandbox
# ---------------------------------------------------------------------------
_USE_FAST_EXEC = hasattr(signal, "alarm")
_mp_pool = None
_mp_pool_lock = threading.Lock()


def _get_mp_pool():
    """Get a single-process sandbox pool (persistent, prevents OOM)."""
    global _mp_pool
    if _mp_pool is None:
        with _mp_pool_lock:
            if _mp_pool is None:
                ctx = multiprocessing.get_context("spawn")
                _mp_pool = ctx.Pool(processes=1, maxtasksperchild=50)
    return _mp_pool


def _worker_execute(code_string: str, timeout_sec: int) -> dict:
    """Execute code in a subprocess worker with signal-based timeout (Linux only)."""
    def handler(signum, frame):
        raise TimeoutError("code execution timeout")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_sec)

    stdout_io = io.StringIO()
    stderr_io = io.StringIO()
    success = False

    try:
        with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
            exec(code_string, {})
        success = True
    except TimeoutError as e:
        stderr_io.write(f"RuntimeError: {str(e)}")
    except Exception:
        traceback.print_exc(file=stderr_io)
    finally:
        signal.alarm(0)

    return {
        "success": success,
        "output": stdout_io.getvalue(),
        "error": stderr_io.getvalue() if not success else None,
    }


def _run_code_pool(code_string: str, timeout: float = 30.0) -> dict:
    """Execute via multiprocessing pool with signal.alarm (Linux)."""
    pool = _get_mp_pool()
    timeout_int = max(1, int(timeout))
    try:
        async_result = pool.apply_async(_worker_execute, (code_string, timeout_int))
        return async_result.get(timeout=timeout_int + 5.0)
    except multiprocessing.TimeoutError:
        return {"success": False, "output": "", "error": "RuntimeError: worker process hung or timeout"}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def _run_code_subprocess(code_string: str, timeout: float = 30.0) -> dict:
    """Execute via subprocess (Windows-compatible fallback)."""
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


def run_code(code_string: str, timeout: float = 30.0) -> dict:
    """Execute Python code safely in a sandbox.

    Uses multiprocessing pool with signal.alarm on Linux,
    falls back to subprocess on Windows.

    Returns:
        dict with keys: success (bool), output (str), error (str | None)
    """
    if _USE_FAST_EXEC:
        return _run_code_pool(code_string, timeout)
    else:
        return _run_code_subprocess(code_string, timeout)


# ---------------------------------------------------------------------------
# Z3 code auto-correction loop
# ---------------------------------------------------------------------------

def correct_z3_code(
    code: str,
    error: str,
    *,
    api_config: Optional[dict] = None,
) -> str:
    """Use LLM to fix erroneous Z3 code.

    Returns the corrected Python code.
    """
    template = load_prompt(CORRECT_CODE_PROMPT)
    prompt = Template(template).safe_substitute(code=code, error=error)
    fix_output = call_llm(prompt, api_config=api_config)
    return extract_python_block(fix_output)


def correct_loop(
    code: str,
    *,
    api_config: Optional[dict] = None,
    max_tries: int = 3,
    timeout: float = 30.0,
) -> dict:
    """Execute Z3 code with auto-correction loop.

    If execution fails, use LLM to fix the code and retry, up to ``max_tries``.
    Temperature increases by 0.05 each retry to encourage diversity.

    Returns:
        dict with keys: success, output, error
    """
    cfg = dict(api_config or {})
    res = run_code(code, timeout=timeout)
    tries = 0

    while not res["success"] and tries < max_tries:
        error_msg = res.get("error", "Unknown error")
        code = correct_z3_code(code, error_msg, api_config=cfg)
        res = run_code(code, timeout=timeout)
        tries += 1
        cfg["temperature"] = cfg.get("temperature", 0.1) + 0.05

    return res


# ---------------------------------------------------------------------------
# Thread-safe preprocessing cache
# ---------------------------------------------------------------------------
_preprocess_cache: dict[tuple, Any] = {}
_preprocess_locks: dict[tuple, threading.Lock] = {}
_preprocess_global_lock = threading.Lock()


def thread_safe_cache(func):
    """Decorator: thread-safe double-checked locking cache.

    Caches on (context, question, options) — the first three positional args.
    """
    @wraps(func)
    def wrapper(context: str, question: str, options: str = "", *, api_config: Optional[dict] = None):
        cache_key = (context, question, options)

        # Fast path: check without lock
        if cache_key in _preprocess_cache:
            return _preprocess_cache[cache_key]

        # Get or create a key-specific lock
        with _preprocess_global_lock:
            if cache_key not in _preprocess_locks:
                _preprocess_locks[cache_key] = threading.Lock()
            key_lock = _preprocess_locks[cache_key]

        # Double-checked locking
        with key_lock:
            if cache_key in _preprocess_cache:
                return _preprocess_cache[cache_key]
            result = func(context, question, options, api_config=api_config)
            _preprocess_cache[cache_key] = result
            return result

    return wrapper


# ---------------------------------------------------------------------------
# Structured pipeline helpers (from nl2fol_slm.py)
# ---------------------------------------------------------------------------

def rephrase(
    context: str, question: str, options: str = "",
    *, api_config: Optional[dict] = None,
) -> str:
    """Rephrase the problem for clarity using LLM."""
    template = load_prompt(REPHRASE_PROMPT)
    prompt = Template(template).safe_substitute(
        context=context, question=question, options=options
    )
    return call_llm(prompt, api_config=api_config)


def object_extract(
    context: str, question: str, options: str = "",
    *, api_config: Optional[dict] = None,
) -> dict:
    """Extract key entities from the problem using structured LLM output."""
    template = load_prompt(OBJECT_EXTRACT_PROMPT)
    prompt = Template(template).safe_substitute(
        context=context, question=question, options=options
    )
    result = call_llm_structured(prompt, api_config=api_config)
    if result and isinstance(result, dict):
        return result
    return {}


def predicate_extract(
    context: str, question: str, options: str = "",
    objectives: Optional[dict] = None,
    *, api_config: Optional[dict] = None,
) -> dict:
    """Extract predicates/relations using structured LLM output."""
    template = load_prompt(PREDICATE_EXTRACTION_PROMPT)
    obj_list = str(objectives) if objectives else ""
    prompt = Template(template).safe_substitute(
        context=context, question=question, options=options, obj_list=obj_list
    )
    result = call_llm_structured(prompt, api_config=api_config)
    if result and isinstance(result, dict):
        return result
    return {}


def generate_z3_declarations_from_entities(entities: dict) -> str:
    """Generate Z3 type/constant/variable declarations from extracted entities.

    Code-based (deterministic) — not LLM-based.
    """
    code_lines = []

    code_lines.append("# Z3 Type Declaration")
    for entity_type in entities.keys():
        code_lines.append(f"{entity_type} = DeclareSort('{entity_type}')")

    code_lines.append("\n# Constants Definition")
    for entity_type, names in entities.items():
        if not isinstance(names, list):
            names = [names]
        for name in names:
            if isinstance(name, list):
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
    """Generate Z3 Function declarations from extracted predicates."""
    code_lines = ["# Z3 Function/Predicate Declaration"]
    for func_name, types in predicates.items():
        if not isinstance(types, list):
            types = [str(types)]
        else:
            types = [str(t) for t in types]
        types_str = ", ".join(types)
        code_lines.append(f"{func_name} = Function('{func_name}', {types_str})")
    return "\n".join(code_lines)


# ---------------------------------------------------------------------------
# Format checking (kept from original fol.py)
# ---------------------------------------------------------------------------

def check_step_format_fol(step_text: str) -> bool:
    """Check if a reasoning step strictly follows the format with <step>, <premise>, <conclusion>."""
    step_text = step_text.strip()

    if not step_text.startswith("<step>") or not step_text.endswith("</step>"):
        return False

    step_open = step_text.count("<step>")
    step_close = step_text.count("</step>")
    premise_open = step_text.count("<premise>")
    premise_close = step_text.count("</premise>")
    conclusion_open = step_text.count("<conclusion>")
    conclusion_close = step_text.count("</conclusion>")

    if step_open != 1 or step_close != 1:
        return False
    if premise_open <= 0 or premise_open != premise_close:
        return False
    if conclusion_open <= 0 or conclusion_open != conclusion_close:
        return False

    first_premise_pos = step_text.find("<premise>")
    first_conclusion_pos = step_text.find("<conclusion>")
    if first_premise_pos > first_conclusion_pos:
        return False

    # Check tag nesting
    tag_pattern = r"<(/?\w+)>"
    matches = list(re.finditer(tag_pattern, step_text))
    stack = []
    for match in matches:
        tag = match.group(1)
        if tag.startswith("/"):
            closing_tag = tag[1:]
            if not stack or stack[-1] != closing_tag:
                return False
            stack.pop()
        else:
            if tag == "conclusion" and "premise" in stack:
                pass
            stack.append(tag)

    if len(stack) != 0:
        return False

    # Check tags have content
    for tag_name in ["premise", "conclusion"]:
        matches_content = re.findall(f"<{tag_name}>(.*?)</{tag_name}>", step_text, re.DOTALL)
        for content in matches_content:
            if not content.strip():
                return False

    return True
