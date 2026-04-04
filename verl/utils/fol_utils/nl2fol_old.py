"""
NL to FOL translation utilities.

Adapted from verl-binlp/mcts_utils/nl2fol.py.
Provides LLM-assisted premise extraction, declaration extraction, and
FOL translation. All API calls are parameterized via `api_config` dict.
"""

import os
import re
from pathlib import Path
from typing import Optional, Sequence, Union

from openai import OpenAI

# Prompt files ship with verl under verl/prompts/
PROMPT_ROOT = Path(__file__).resolve().parents[4] / "prompts"
PREMISE_EXTRACTION_PROMPT = PROMPT_ROOT / "premise_extraction.txt"
DECLARATION_PROMPT = PROMPT_ROOT / "extract_declaration.txt"
TRANSLATE_FOL_PROMPT = PROMPT_ROOT / "translate2fol.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_text(value: Optional[str]) -> str:
    return value if value is not None else ""


def extract_plain_text_block(text: str) -> str:
    """Extract content from ```plain_text ... ``` blocks."""
    pattern = r"```plain_text\s+([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""


def load_prompt(path: Optional[Union[str, Path]] = None) -> str:
    prompt_path = Path(path) if path else PREMISE_EXTRACTION_PROMPT
    return prompt_path.read_text(encoding="utf-8").strip()


def _format_options(options: Optional[Union[str, Sequence[str]]]) -> str:
    if options is None:
        return ""
    if isinstance(options, str):
        return options.strip()
    return "\n".join(str(opt).strip() for opt in options)


def _fill_prompt(template: str, context: str, question: str, options) -> str:
    return (
        template.replace("{{Context}}", _ensure_text(context).strip())
        .replace("{{Question}}", _ensure_text(question).strip())
        .replace("{{Options}}", _format_options(options))
    )


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------

def _get_default_api_config() -> dict:
    """Build default API config from environment variables."""
    return {
        "model": os.environ.get("FOL_MODEL", "gpt-4o-mini"),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", None),
        "temperature": 0.6,
        "max_tokens": 1024,
    }


def _call_llm(prompt: str, *, api_config: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
    """Call an OpenAI-compatible chat API.

    Args:
        prompt: User message content.
        api_config: Dict with keys: model, api_key, base_url, temperature, max_tokens.
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
    messages.append({"role": "user", "content": prompt})

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg.get("base_url"))
    completion = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=cfg.get("temperature", 0.6),
        max_tokens=cfg.get("max_tokens", 1024),
    )
    return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def preprocess_premises(
    *,
    context: str,
    question: str,
    options=None,
    api_config: Optional[dict] = None,
) -> str:
    """Extract premises using the bundled premise-extraction prompt."""
    prompt_template = load_prompt(PREMISE_EXTRACTION_PROMPT)
    filled_prompt = _fill_prompt(prompt_template, context, question, options)
    return _call_llm(filled_prompt, api_config=api_config)


def extract_declaration(
    *,
    context: str,
    question: str,
    options=None,
    api_config: Optional[dict] = None,
) -> str:
    """Extract declarations using the extract_declaration prompt."""
    prompt_template = load_prompt(DECLARATION_PROMPT)
    filled_prompt = _fill_prompt(prompt_template, context, question, options)
    return _call_llm(filled_prompt, api_config=api_config)


def translate_to_fol(
    *,
    declarations: str,
    sentences: str,
    api_config: Optional[dict] = None,
) -> str:
    """Translate natural language sentences to FOL using translate2fol prompt."""
    prompt_template = load_prompt(TRANSLATE_FOL_PROMPT)
    filled_prompt = (
        prompt_template + "\n" + _ensure_text(declarations).strip() + "\n" + _ensure_text(sentences).strip()
    )
    return _call_llm(filled_prompt, api_config=api_config)


# ---------------------------------------------------------------------------
# High-level convenient functions (used by step reward)
# ---------------------------------------------------------------------------

def fol_preprocessing(
    context: str,
    question: str,
    options: Optional[str] = None,
    *,
    api_config: Optional[dict] = None,
) -> str:
    """Run premise extraction + declaration extraction. Returns declaration text."""
    premise_text = preprocess_premises(
        context=context, question=question, options=options, api_config=api_config
    )
    declaration_raw = extract_declaration(
        context=premise_text, question=question, options=options, api_config=api_config
    )
    declaration = extract_plain_text_block(declaration_raw)
    return declaration


def translate_and_execute_fol(
    declaration: str,
    sentences: str,
    *,
    api_config: Optional[dict] = None,
) -> float:
    """Translate sentences to FOL, convert to Z3 code, and execute.

    Returns:
        1.0 if the Z3 solver finds SAT, 0.0 otherwise.
    """
    fol_raw = translate_to_fol(declarations=declaration, sentences=sentences, api_config=api_config)
    constraints = extract_plain_text_block(fol_raw)
    if not constraints:
        return 0.0

    from .fol_to_python_converter import convert_and_execute_fol

    _code, result, error = convert_and_execute_fol(declaration, constraints)
    if result:
        try:
            return float(result[-1])
        except (ValueError, IndexError):
            return 0.0
    return 0.0
