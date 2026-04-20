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
import logging
import multiprocessing
import os
import random
import re
import signal
import string
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from functools import wraps
from pathlib import Path
from string import Template
from typing import Any, Optional, Union

from openai import OpenAI

logger = logging.getLogger(__name__)

# Error-message fragments that indicate a transient Gemini API failure worth retrying.
# Kept as substrings (case-sensitive against upstream messages) so we can fall back
# to string matching when google.genai.errors types are unavailable or subclassed oddly.
_GEMINI_TRANSIENT_MARKERS = (
    # HTTP status / google API status
    "503", "502", "500", "504", "429",
    "UNAVAILABLE", "RESOURCE_EXHAUSTED",
    "DEADLINE_EXCEEDED", "INTERNAL", "overloaded",
    # httpx / connection-level (proxy flap, tunnel drop, keep-alive expired,
    # 机场 RST under concurrent-connection limits)
    "Server disconnected", "disconnected without sending",
    "RemoteProtocolError", "ConnectError", "ConnectTimeout",
    "ReadError", "ReadTimeout", "WriteError", "WriteTimeout",
    "PoolTimeout", "Connection reset", "Connection aborted",
    "ProxyError", "SSLError", "Errno 104",
    "timed out", "Timeout",
)


def _is_transient_gemini_error(exc: BaseException) -> bool:
    try:
        from google.genai import errors as genai_errors  # type: ignore

        for cls_name in ("ServerError", "ServiceUnavailableError", "ResourceExhaustedError"):
            cls = getattr(genai_errors, cls_name, None)
            if cls is not None and isinstance(exc, cls):
                return True
        api_error_cls = getattr(genai_errors, "APIError", None)
        if api_error_cls is not None and isinstance(exc, api_error_cls):
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if code in (429, 500, 502, 503, 504):
                return True
    except Exception:
        pass
    msg = str(exc)
    return any(marker in msg for marker in _GEMINI_TRANSIENT_MARKERS)

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
_gemini_client_cache: dict[str, Any] = {}
_client_cache_lock = threading.Lock()
_gemini_last_call = 0.0
_gemini_rate_lock = threading.Lock()
# Process-wide cap on concurrent in-flight Gemini calls. Prevents one worker's
# 16-thread pool from bursting 16 simultaneous TCP connections through a
# low-capacity proxy (Clash/机场 typically limits concurrent connections per
# account). Override with env FOL_GEMINI_MAX_INFLIGHT if mihomo + your 机场
# can handle more (or less).
_gemini_inflight_sem = threading.Semaphore(
    int(os.environ.get("FOL_GEMINI_MAX_INFLIGHT", "8"))
)
# Same pattern for OpenAI-compatible APIs (SiliconFlow, vLLM, etc.)
_openai_inflight_sem = threading.Semaphore(
    int(os.environ.get("FOL_OPENAI_MAX_INFLIGHT", "32"))
)
_openai_last_call = 0.0
_openai_rate_lock = threading.Lock()

_OPENAI_TRANSIENT_MARKERS = ("429", "rate limit", "502", "503", "504", "connection", "timeout")


def _is_transient_openai_error(exc: BaseException) -> bool:
    """Check if an OpenAI-compatible API error is transient and retryable."""
    try:
        from openai import RateLimitError, APIStatusError
        if isinstance(exc, RateLimitError):
            return True
        if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503, 504):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return any(marker in msg for marker in _OPENAI_TRANSIENT_MARKERS)


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
                max_retries=3,
            )
            _client_cache[cache_key] = client
        return client


def get_gemini_client(api_key: str) -> Any:
    """Return a cached Gemini client (google-genai SDK). Lazy import."""
    if api_key in _gemini_client_cache:
        return _gemini_client_cache[api_key]

    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "The 'google-genai' package is required for Gemini models. "
            "Please install it with 'pip install google-genai'."
        )

    with _client_cache_lock:
        if api_key not in _gemini_client_cache:
            client = genai.Client(api_key=api_key)
            _gemini_client_cache[api_key] = client
        return _gemini_client_cache[api_key]


# ---------------------------------------------------------------------------
# LLM call wrappers
# ---------------------------------------------------------------------------

def _get_default_api_config() -> dict:
    """Build default API config from environment variables."""
    return {
        "model": os.environ.get("FOL_MODEL", os.environ.get("FOL_SLM_MODEL", "gpt-4o-mini")),
        "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
        "base_url": os.environ.get("OPENAI_BASE_URL", os.environ.get("FOL_SLM_BASE_URL", None)),
        "rpm": float(os.environ.get("FOL_RPM", 10)),
        "temperature": 0.2,
        "max_tokens": 1024,
        "top_p": 0.8,
    }


_llm_call_counter = 0
_llm_call_counter_lock = threading.Lock()

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
    global _llm_call_counter
    with _llm_call_counter_lock:
        _llm_call_counter += 1
        call_id = _llm_call_counter

    cfg = _get_default_api_config()
    if api_config:
        cfg.update({k: v for k, v in api_config.items() if v is not None})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    timeout = cfg.pop("api_timeout", 200)
    top_p = cfg.pop("top_p", 0.8)

    # _tid = threading.current_thread().name
    # _prompt_preview = user_prompt[:80].replace('\n', '\\n')
    # print(f"[LLM][{_tid}] #{call_id} → {cfg['model']}@{cfg.get('base_url','?')}  prompt={_prompt_preview!r}...", flush=True)
    # _t = time.time()
    if cfg["model"].startswith("gemini"):
        return call_gemini(user_prompt, cfg, system_prompt=system_prompt)

    # RPM rate limiting (same as Gemini path)
    rpm = cfg.get("rpm", 0)
    if rpm > 0:
        global _openai_last_call
        min_interval = 60.0 / max(rpm, 0.1)
        with _openai_rate_lock:
            dt = time.time() - _openai_last_call
            if dt < min_interval:
                time.sleep(min_interval - dt)
            _openai_last_call = time.time()

    client = get_client(cfg["api_key"], cfg.get("base_url"), timeout)
    max_retries = int(cfg.get("api_max_retries", 5))
    base_delay = float(cfg.get("api_retry_base_delay", 10.0))
    max_delay = float(cfg.get("api_retry_max_delay", 300.0))

    attempt = 0
    while True:
        try:
            with _openai_inflight_sem:
                completion = client.chat.completions.create(
                    model=cfg["model"],
                    messages=messages,
                    temperature=cfg.get("temperature", 0.2),
                    max_tokens=cfg.get("max_tokens", 1024),
                    top_p=top_p,
                    n=1,
                )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            if attempt >= max_retries or not _is_transient_openai_error(exc):
                logger.error("Max retries exceeded at OpenAI SDK. Rewarding 0.0.")
                exit()
                raise
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            # logger.warning(
            #     "OpenAI API transient error (attempt %d/%d), sleeping %.1fs: %s",
            #     attempt + 1, max_retries, delay, str(exc)[:200],
            # )
            time.sleep(delay)
            attempt += 1


def call_gemini(
    user_prompt: str,
    cfg: dict,
    system_prompt: Optional[str] = None,
) -> str:
    """Call Google Gemini API with rate limiting."""
    global _gemini_last_call

    client = get_gemini_client(cfg["api_key"])
    rpm = cfg.get("rpm", 10)
    min_interval = 60.0 / max(rpm, 0.1)

    with _gemini_rate_lock:
        dt = time.time() - _gemini_last_call
        if dt < min_interval:
            time.sleep(min_interval - dt)
        _gemini_last_call = time.time()

    from google.genai import types
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=cfg.get("temperature", 0.2),
        max_output_tokens=cfg.get("max_tokens", 4096),
        top_p=cfg.get("top_p", 0.8),
    )

    # thinking_budget support
    thinking_budget = cfg.get("thinking_budget", 0)
    if thinking_budget > 0:
        config.thinking_config = types.ThinkingConfig(include_thoughts=True)
        # Note: thinking_budget is not yet standardized across SDK versions,
        # using a common setting if available.

    # Independent from Z3 correct_loop `max_tries`: this is Gemini API-level
    # transient-error retry (429 / 5xx / UNAVAILABLE / conn-reset), not logical retries.
    # Default 3 instead of 5: sustained link outages (>30s) are not fixed by more retries,
    # just waste wall-clock time per failing call (5 retries = ~65s, 3 retries = ~14s).
    max_retries = int(cfg.get("gemini_api_max_retries", 3))
    base_delay = float(cfg.get("gemini_api_retry_base_delay", 2.0))
    max_delay = float(cfg.get("gemini_api_retry_max_delay", 30.0))

    attempt = 0
    while True:
        try:
            with _gemini_inflight_sem:
                response = client.models.generate_content(
                    model=cfg["model"],
                    contents=user_prompt,
                    config=config,
                )
            text = getattr(response, "text", None)
            return text or ""
        except Exception as exc:
            if attempt >= max_retries:
                logger.error("Gemini transient error: Max retries exceeded, rewarding 0.0")
            if attempt >= max_retries or not _is_transient_gemini_error(exc):
                raise
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            logger.warning(
                "Gemini transient error (attempt %d/%d), sleeping %.1fs: %s",
                attempt + 1, max_retries, delay, str(exc)[:200],
            )
            time.sleep(delay)
            attempt += 1


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
# _USE_FAST_EXEC = hasattr(signal, "alarm")
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
    """
    Execute via multiprocessing pool with signal.alarm (Linux).
    Not used since tasks must queue behind a single worker, causing bottlenecks. Kept as reference for the signal-based timeout approach.
    """
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

    Always uses subprocess — each Z3 run gets its own process so:
    - No OOM accumulation (process dies after each run)
    - Naturally parallel (no single-process pool bottleneck)
    - Works on both Linux and Windows

    Returns:
        dict with keys: success (bool), output (str), error (str | None)
    """
    # _tid = threading.current_thread().name
    # _code_preview = code_string[:100].replace('\n', '\\n')
    # print(f"[EXEC][{_tid}] → subprocess  timeout={timeout}  code={_code_preview!r}...", flush=True)
    # _t = time.time()
    # res = _run_code_subprocess(code_string, timeout)
    # _elapsed = time.time() - _t
    # _warn = "  ⚠ SLOW" if _elapsed > 5.0 else ""
    # print(f"[EXEC][{_tid}] ← {_elapsed:.2f}s  success={res['success']}  out={res.get('output','')[:60]!r}{_warn}", flush=True)
    # if not res['success']:
    #     print(f"[EXEC][{_tid}]   err={res.get('error','')[:500]!r}", flush=True)
    # return res
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
    # _tid = threading.current_thread().name
    cfg = dict(api_config or {})
    # print(f"[Z3LOOP][{_tid}] → run_code (initial, timeout={timeout}s)...", flush=True)
    # _t = time.time()
    res = run_code(code, timeout=timeout)
    # print(f"[Z3LOOP][{_tid}] ← run_code {time.time()-_t:.2f}s  success={res['success']}  out={res.get('output','')[:80]!r}", flush=True)
    tries = 0

    while not res["success"] and tries < max_tries:
        error_msg = res.get("error", "Unknown error")
        # print(f"[Z3LOOP][{_tid}] ⟳ retry {tries+1}/{max_tries}  err={error_msg[:100]!r}", flush=True)
        # _tc = time.time()
        code = correct_z3_code(code, error_msg, api_config=cfg)
        # print(f"[Z3LOOP][{_tid}]   correct_z3 {time.time()-_tc:.2f}s → run_code...", flush=True)
        # _tr = time.time()
        res = run_code(code, timeout=timeout)
        # print(f"[Z3LOOP][{_tid}]   run_code {time.time()-_tr:.2f}s  success={res['success']}  out={res.get('output','')[:80]!r}", flush=True)
        tries += 1
        cfg["temperature"] = cfg.get("temperature", 0.1) + 0.05

    # if tries > 0:
    #     print(f"[Z3LOOP][{_tid}] ✓ loop done after {tries} retries, total={time.time()-_t:.2f}s", flush=True)
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
        # _tid = threading.current_thread().name
        # _key_hash = hash(cache_key) % 10000

        # Fast path: check without lock
        if cache_key in _preprocess_cache:
            # print(f"[CACHE][{_tid}] HIT  key={_key_hash}  fn={func.__name__}  cache_size={len(_preprocess_cache)}", flush=True)
            return _preprocess_cache[cache_key]

        # Get or create a key-specific lock
        with _preprocess_global_lock:
            if cache_key not in _preprocess_locks:
                _preprocess_locks[cache_key] = threading.Lock()
            key_lock = _preprocess_locks[cache_key]

        # Double-checked locking
        # print(f"[CACHE][{_tid}] MISS key={_key_hash}  fn={func.__name__}  waiting lock...", flush=True)
        # _tw = time.time()
        with key_lock:
            if cache_key in _preprocess_cache:
                # print(f"[CACHE][{_tid}] HIT-after-wait  key={_key_hash}  waited={time.time()-_tw:.2f}s", flush=True)
                return _preprocess_cache[cache_key]
            # print(f"[CACHE][{_tid}] COMPUTING  key={_key_hash}  fn={func.__name__}...", flush=True)
            # _tc = time.time()
            result = func(context, question, options, api_config=api_config)
            # print(f"[CACHE][{_tid}] STORED  key={_key_hash}  fn={func.__name__}  {time.time()-_tc:.2f}s  cache_size={len(_preprocess_cache)+1}", flush=True)
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
