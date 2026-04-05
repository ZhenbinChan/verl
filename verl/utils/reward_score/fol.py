"""
FOL-based step reward — API large-model version.

Adapted from ZhenbinChan/verl pipeline branch (T0nglinziyong's approach).
Uses an external LLM to translate reasoning steps into Z3 and verify entailment.

Exports the same interface as the old ``fol.py`` so callers are unaffected:
  - ``check_step_format_fol``
  - ``compute_step_reward_format_fol``
  - ``compute_step_reward_fol``
"""

import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format checking (unchanged from original)
# ---------------------------------------------------------------------------

def check_step_format_fol(step_text: str) -> bool:
    """
    Check if a reasoning step strictly follows the format and contains <step>, <premise>, <conclusion>.
    """
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


def compute_step_reward_format_fol(step_text: str, prompt_text: str, step_history: list[str], **kwargs) -> float:
    """Format-check process reward ensuring strict step/premise/conclusion tags."""
    return 1.0 if check_step_format_fol(step_text) else 0.0


# ---------------------------------------------------------------------------
# FOL entailment reward (API large-model version)
# ---------------------------------------------------------------------------

def compute_step_reward_fol(
    step_text: str,
    prompt_text: str,
    step_history: list[str],
    *,
    api_config: dict | None = None,
    extra_info: dict | None = None,
) -> float:
    """FOL-based process reward — API large-model version.

    Uses an external LLM to:
      1. Generate Z3 declarations from the problem context
      2. Translate the current step's premises/conclusion into Z3 code
      3. Check entailment via Z3 solver (UNSAT of negated conclusion → entailed)

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
        from verl.utils.fol_utils.nl2fol import (
            fol_preprocess_declarations,
            translate_and_verify_step,
        )

        is_cumulative = (api_config or {}).get("cumulative", False)
        if is_cumulative and step_history:
            step_text_to_translate = "\n".join(step_history)
        else:
            step_text_to_translate = step_text

        declarations = fol_preprocess_declarations(
            context, question, options, api_config=api_config
        )
        reward = translate_and_verify_step(
            context, declarations, step_text_to_translate, api_config=api_config
        )
        return float(reward)
    except Exception as e:
        logger.warning("FOL reward computation failed: %s", e)
        return 0.0
