"""
FOL-based step reward -- Small LLM (SLM) version.

Adapted from ZhenbinChan/verl pipeline branch (ZhenbinChan's approach).
Uses a local vLLM-served small model with structured extraction and
Z3 auto-correction loop.

Step reward type: ``fol_slm``
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


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
      5. Execute with auto-correction loop (up to 8 retries)

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

    try:
        from verl.utils.fol_utils.nl2fol_slm import (
            slm_preprocess,
            translate_and_verify_step_slm,
        )

        rephrased_context, declaration_code = slm_preprocess(
            context, question, options or "", api_config=api_config
        )
        reward = translate_and_verify_step_slm(
            rephrased_context, declaration_code, step_text, api_config=api_config
        )
        return float(reward)
    except Exception as e:
        logger.warning("FOL-SLM reward computation failed: %s", e)
        return 0.0
