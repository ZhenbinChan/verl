"""Self-evaluate step reward.

Uses an LLM (typically the reference model served via vLLM/SGLang) to
score the quality of each reasoning step on a 0-10 rubric, then
normalizes to [0, 1].

The API calling pattern mirrors ``verl.utils.fol_utils.nl2fol._call_llm``.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts" / "self_eval"
_NON_TERMINAL_PROMPT_PATH = PROMPT_ROOT / "non_terminal.txt"
_TERMINAL_PROMPT_PATH = PROMPT_ROOT / "terminal.txt"

_prompt_cache: dict[str, str] = {}
_prompt_logged: set[str] = set()


def _load_prompt(path: Path) -> str:
    key = str(path)
    if key not in _prompt_cache:
        _prompt_cache[key] = path.read_text(encoding="utf-8").strip()
        # Log system prompt once on first load
    if key not in _prompt_logged:
        _prompt_logged.add(key)
        logger.info("=== [Self-Eval System Prompt] %s ===\n%s\n===", path.name, _prompt_cache[key])
    return _prompt_cache[key]


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

_SCORE_RE = re.compile(r"\*{0,2}Overall\s+Score:?\*{0,2}\s*\[?([0-9]*\.?[0-9]+)\]?", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{\{?([0-9]*\.?[0-9]+)\}?\}")


def _extract_score(text: str) -> Optional[float]:
    """Extract ``Overall Score: <float>`` from LLM output, fallback to ``\\boxed{<float>}``."""
    match = _SCORE_RE.search(text)
    if match:
        return float(match.group(1))
    # Fallback: model may output \boxed{number} instead of following rubric
    match = _BOXED_RE.search(text)
    if match:
        return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _get_default_api_config() -> dict:
    return {
        "model": os.environ.get("SELF_EVAL_MODEL", os.environ.get("FOL_MODEL", "gpt-4o-mini")),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", None),
        "temperature": 0.0,
        "max_tokens": 1024,
    }


def _call_llm(prompt: str, *, api_config: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
    """Call an OpenAI-compatible chat API (vLLM / SGLang / OpenAI)."""
    cfg = _get_default_api_config()
    if api_config:
        cfg.update({k: v for k, v in api_config.items() if v is not None})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    timeout = cfg.pop("api_timeout", 200)
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        timeout=timeout,
        max_retries=1,
    )
    completion = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 1024),
    )
    return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public API — pluggable step reward function
# ---------------------------------------------------------------------------

def compute_step_reward_self_eval(
    step_text: str,
    prompt_text: str,
    step_history: list[str],
    *,
    api_config: Optional[dict] = None,
    extra_info: Optional[dict] = None,
) -> float:
    """Self-evaluate reward: let the reference model score a reasoning step.

    Signature matches the pluggable step-reward interface used by
    ``StepRewardManager`` and ``TreeRewardManager``.

    Args:
        step_text: The current reasoning step text.
        prompt_text: The original problem / user prompt.
        step_history: All steps so far (including current).
        api_config: OpenAI-compatible API configuration dict.
        extra_info: Optional dict; if ``is_terminal`` key is present it
            overrides the heuristic terminal detection.

    Returns:
        Float in [0, 1].  0.0 on any failure.
    """
    extra_info = extra_info or {}

    # Determine if this is a terminal (final) step
    is_terminal = extra_info.get("is_terminal", None)
    if is_terminal is None:
        # Heuristic: check if the step contains a boxed answer
        is_terminal = r"\boxed" in step_text

    # Select system prompt
    sys_prompt = _load_prompt(_TERMINAL_PROMPT_PATH if is_terminal else _NON_TERMINAL_PROMPT_PATH)

    # Build user prompt with accumulated reasoning
    accumulated = "\n\n".join(step_history)
    user_prompt = f"Problem: {prompt_text}\n\nStudent's Reasoning Process: {accumulated}\n\n"
    # logger.info("=== [Self-Eval Started] is_terminal=%s step_length=%d ===", is_terminal, len(step_text))
    # logger.info("--- [Self-Eval User Prompt] ---\n%s\n---", user_prompt)

    try:
        response_text = _call_llm(user_prompt, api_config=api_config, system_prompt=sys_prompt)
        score = _extract_score(response_text)
        if score is not None:
            reward = max(0.0, min(10.0, score)) / 10.0
            # logger.info("--- [Self-Eval Judge Response] score=%.1f/10 ---", score)
            # logger.info("--- [Self-Eval Judge Response] ---\n%s\n----------------------------------", response_text)
            # logger.info("--- [Self-Eval Finished] reward=%.3f ---", reward)

            return reward
        # logger.info("--- [Self-Eval Judge] could not extract score ---")
                     
        # logger.info("--- [Self-Eval Judge Response] ---\n%s\n----------------------------------", response_text)
        # logger.info("--- [Self-Eval Finished] reward=%.3f ---", 0.0)
        return 0.0
    except Exception as e:
        logger.warning("--- [Self-Eval Judge Response] API call failed: %s ---", e)
        # logger.info("--- [Self-Eval Finished] reward=%.3f ---", 0.0)
        return 0.0
