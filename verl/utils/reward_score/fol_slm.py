"""
FOL-based step reward -- Small LLM (SLM) version.

Adapted from ZhenbinChan/verl pipeline branch (ZhenbinChan's approach).
Uses a local vLLM-served small model with structured extraction and
Z3 auto-correction loop.

Step reward type: ``fol_slm``
"""

import logging
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt-level preprocessing cache
# ---------------------------------------------------------------------------
# slm_preprocess depends only on (context, question, options), NOT on step text.
# All steps from the same prompt can share the same preprocessed result,
# saving ~3 LLM calls per step after the first one.
_preprocess_cache: dict[tuple[str, str, str], tuple[str, str]] = {}
_preprocess_cache_lock = threading.Lock()
_CACHE_MAX_SIZE = 512


def _get_cached_preprocess(
    context: str, question: str, options: str, api_config: dict | None
) -> tuple[str, str]:
    """Get (rephrased_context, declaration_code) with prompt-level caching."""
    cache_key = (context, question, options)

    with _preprocess_cache_lock:
        if cache_key in _preprocess_cache:
            return _preprocess_cache[cache_key]

    from verl.utils.fol_utils.nl2fol_slm import slm_preprocess

    result = slm_preprocess(context, question, options, api_config=api_config)

    with _preprocess_cache_lock:
        if len(_preprocess_cache) > _CACHE_MAX_SIZE:
            _preprocess_cache.clear()
        _preprocess_cache[cache_key] = result

    return result


def compute_step_reward_fol_slm(
    step_text: str,
    prompt_text: str,
    step_history: list[str],
    *,
    api_config: dict | None = None,
    extra_info: dict | None = None,
) -> float:
    """FOL-based process reward — SLM version.

    Uses a local small LLM (via vLLM) to:
      1. Rephrase problem for clarity
      2. Extract objects & predicates (structured output)
      3. Generate Z3 declarations from code (deterministic)
      4. Translate step to Z3 via LLM
      5. Execute with auto-correction loop (up to ``max_tries`` retries)

    Extraction priority for context/question/options:
      1. Structured fields in extra_info (fol_context, fol_question, fol_options)
      2. Fallback: regex on prompt_text for <Context>/<Question>/<Options> XML tags
      3. If neither found: return 0.0 (not a logic problem)
    """
    extra_info = extra_info or {}

    # Extract context/question/options
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

    if not context or not question:
        return 0.0
    
    from verl.utils.fol_utils.nl2fol_slm import translate_and_verify_step_slm
    try:
        is_cumulative = (api_config or {}).get("cumulative", False)
        if is_cumulative and step_history:
            step_text_to_translate = "\n".join(step_history)
        else:
            step_text_to_translate = step_text
        # print(f"FOL-SLM reward: Translating step: {step_text_to_translate}", flush=True)
        rephrased_context, declaration_code = _get_cached_preprocess(
            context, question, options or "", api_config
        )
        # print(f"FOL-SLM reward: Rephrased context: {rephrased_context}", flush=True)
        # print(f"FOL-SLM reward: Declaration code: {declaration_code}", flush=True)

        reward = translate_and_verify_step_slm(
            rephrased_context, declaration_code, step_text_to_translate, api_config=api_config
        )
        print(f"FOL-SLM reward: Reward computed: {reward}", flush=True)
        return float(reward)
    except Exception as e:
        logger.warning("FOL-SLM reward computation failed: %s", e)
        return 0.0
