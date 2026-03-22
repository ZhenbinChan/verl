import logging
import re

def check_step_format_fol(step_text: str) -> bool:
    """
    Check if a reasoning step strictly follows the format and contains <step>, <premise>, <conclusion>.
    Adapted from mcts_utils/evaluation.py
    """
    step_text = step_text.strip()
    
    # 1. 检查是否以 <step> 开头和结尾
    if not step_text.startswith("<step>"):
        return False
    if not step_text.endswith("</step>"):
        return False
    
    # 2. 检查标签是否平衡和成对出现
    step_open_count = step_text.count("<step>")
    step_close_count = step_text.count("</step>")
    premise_open_count = step_text.count("<premise>")
    premise_close_count = step_text.count("</premise>")
    conclusion_open_count = step_text.count("<conclusion>")
    conclusion_close_count = step_text.count("</conclusion>")
    
    if step_open_count != 1 or step_close_count != 1:
        return False
    if premise_open_count <= 0 or premise_open_count != premise_close_count:
        return False
    if conclusion_open_count <= 0 or conclusion_open_count != conclusion_close_count:
        return False
    
    # 3. 检查 premise 是否在 conclusion 之前
    first_premise_pos = step_text.find("<premise>")
    first_conclusion_pos = step_text.find("<conclusion>")
    if first_premise_pos > first_conclusion_pos:
        return False
        
    # 4. 检查标签嵌套顺序
    tag_pattern = r'<(/?\w+)>'
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
        
    # 5. 检查标签内是否有内容
    for tag_name in ["premise", "conclusion"]:
        matches_content = re.findall(f'<{tag_name}>(.*?)</{tag_name}>', step_text, re.DOTALL)
        for content in matches_content:
            if not content.strip():
                return False
                
    return True


def compute_step_reward_format_fol(step_text: str, prompt_text: str, step_history: list[str], **kwargs) -> float:
    """Format-check process reward ensuring strict step/premise/conclusion tags."""
    if check_step_format_fol(step_text):
        return 1.0
    return 0.0


def compute_step_reward_fol(
    step_text: str,
    prompt_text: str,
    step_history: list[str],
    *,
    api_config: dict | None = None,
    extra_info: dict | None = None,
) -> float:
    """FOL-based process reward.

    Uses an external LLM to translate the problem premises into Z3 FOL
    constraints, then checks satisfiability of the current step.

    Extraction priority for context/question/options:
      1. Structured fields in extra_info (fol_context, fol_question, fol_options)
      2. Fallback: regex on prompt_text for <Context>/<Question>/<Options> XML tags
      3. If neither found: return 0.0 (not a logic problem)
    """
    extra_info = extra_info or {}

    # Priority 1: structured fields from extra_info
    context = extra_info.get("fol_context", None)
    question = extra_info.get("fol_question", None)
    options = extra_info.get("fol_options", None)

    # Priority 2: fallback to XML tag parsing from prompt text
    if not context:
        m = re.search(r"<Context>(.*?)</Context>", prompt_text, re.DOTALL)
        context = m.group(1).strip() if m else None
    if not question:
        m = re.search(r"<Question>(.*?)</Question>", prompt_text, re.DOTALL)
        question = m.group(1).strip() if m else None
    if not options:
        m = re.search(r"<Options>(.*?)</Options>", prompt_text, re.DOTALL)
        options = m.group(1).strip() if m else None

    # If we still can't extract context/question, not a logic problem
    if not context or not question:
        return 0.0

    try:
        from verl.utils.fol_utils.nl2fol import fol_preprocessing, translate_and_execute_fol

        declaration = fol_preprocessing(context, question, options, api_config=api_config)
        reward = translate_and_execute_fol(declaration, step_text, api_config=api_config)
        return float(reward)
    except Exception as e:
        logging.getLogger(__name__).warning("FOL reward computation failed: %s", e)
        return 0.0
