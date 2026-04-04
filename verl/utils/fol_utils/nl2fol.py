"""
NL to FOL translation utilities -- API large-model version.

Adapted from ZhenbinChan/verl pipeline branch (T0nglinziyong's approach).
Uses an external LLM (e.g. GPT-4o-mini) to:
  1. Generate Z3 declarations from context/question/options
  2. Translate a reasoning step (premise/conclusion) into Z3 implication code
  3. Execute Z3 to check entailment

All API calls are parameterized via ``api_config`` dict.
"""

import ast
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompt files — ship with verl under verl/prompts/
# ---------------------------------------------------------------------------
PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts"
Z3_DECLARATION_PROMPT = PROMPT_ROOT / "z3_declaration_generation.txt"
Z3_IMPLICATION_PROMPT = PROMPT_ROOT / "z3_implication_conversion.txt"
# Legacy prompts kept for fallback / alternative pipeline
PREMISE_EXTRACTION_PROMPT = PROMPT_ROOT / "premise_extraction.txt"
EXTRACT_DECLARATION_PROMPT = PROMPT_ROOT / "extract_declaration.txt"
TRANSLATE2FOL_PROMPT = PROMPT_ROOT / "translate2fol.txt"


def _load_prompt(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------

def _get_default_api_config() -> dict:
    """Build default API config from environment variables."""
    return {
        "model": os.environ.get("FOL_MODEL", "gpt-4o-mini"),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", None),
        "temperature": 0.2,
        "max_tokens": 2048,
    }


def _call_llm(
    user_prompt: str,
    *,
    api_config: Optional[dict] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Call an OpenAI-compatible chat API.

    Args:
        user_prompt: User message content.
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
    messages.append({"role": "user", "content": user_prompt})

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg.get("base_url"))
    completion = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=cfg.get("temperature", 0.2),
        max_tokens=cfg.get("max_tokens", 2048),
        n=1,
    )
    return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_python_code(text: str) -> str:
    """Extract content from ```python ... ``` fenced code blocks."""
    pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    code_blocks = pattern.findall(text)
    return "\n\n".join(code_blocks) if code_blocks else text.strip()


def parse_reasoning_step(step_text: str) -> dict:
    """Parse a single ``<step>`` block into premises and conclusion.

    Args:
        step_text: Text that may contain ``<premise>`` and ``<conclusion>`` tags.

    Returns:
        dict with keys ``premises`` (list[str]) and ``conclusion`` (str).
    """
    premise_pattern = re.compile(r"<premise>(.*?)</premise>", re.DOTALL)
    premises = [p.strip() for p in premise_pattern.findall(step_text)]

    conclusion_pattern = re.compile(r"<conclusion>(.*?)</conclusion>", re.DOTALL)
    conclusion_match = conclusion_pattern.search(step_text)
    conclusion = conclusion_match.group(1).strip() if conclusion_match else ""

    return {"premises": premises, "conclusion": conclusion}


def parse_python_logic_steps(code_str: str) -> list[dict]:
    """Parse Python code with ``premises_N`` and ``conclusion_N`` assignments.

    This is used after the LLM translates reasoning steps into Z3 code.
    The output is a list of dicts, each with ``premises`` and ``conclusion``
    in FOL string form.
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
# Pipeline stages
# ---------------------------------------------------------------------------

def generate_z3_declarations(
    context: str,
    question: str,
    options: Optional[str] = None,
    *,
    api_config: Optional[dict] = None,
) -> str:
    """Use LLM to generate Z3 variable/function declarations from context.

    Returns the raw Z3 declaration code (Python).
    """
    system_prompt = _load_prompt(Z3_DECLARATION_PROMPT)
    user_input = f"<Context>{context}</Context>\n<Question>{question}</Question>"
    if options:
        user_input += f"\n<Options>{options}</Options>"

    response = _call_llm(user_input, api_config=api_config, system_prompt=system_prompt)
    return extract_python_code(response)


def translate_step_to_z3_implication(
    context: str,
    declarations: str,
    step_text: str,
    *,
    api_config: Optional[dict] = None,
) -> str:
    """Use LLM to translate a single reasoning step into Z3 implication code.

    The prompt includes the original context, Z3 declarations, and the step text.
    Returns the raw Z3 implication code.
    """
    system_prompt = _load_prompt(Z3_IMPLICATION_PROMPT)
    user_input = (
        f"Z3 Declarations:\n```python\n{declarations}\n```\n\n"
        f"Context:\n{context}\n\n"
        f"Reasoning Step:\n{step_text}"
    )
    response = _call_llm(user_input, api_config=api_config, system_prompt=system_prompt)
    return extract_python_code(response)


def verify_step_fol(
    declarations: str,
    premises_fol: list[str],
    conclusion_fol: str,
) -> tuple[Optional[list[str]], str]:
    """Build Z3 code to check if premises entail conclusion, then execute it.

    Checks entailment by adding premises and NOT(conclusion); if UNSAT, entailed.

    Returns:
        (result_lines, error_message)
    """
    premises_str = ", ".join(premises_fol) if premises_fol else "True"
    z3_code = f"""\
from z3 import *

{declarations}

solver = Solver()
solver.add(And({premises_str}))
solver.add(Not({conclusion_fol}))

check_res = solver.check()
if check_res == unsat:
    print("SUCCESS_ENTAILED")
    print(1.0)
elif check_res == sat:
    print("FAILED_CONTRADICT")
    print(0.0)
else:
    print("UNKNOWN")
    print(0.0)
"""
    return execute_program(z3_code)


# ---------------------------------------------------------------------------
# Program execution
# ---------------------------------------------------------------------------

def execute_program(
    python_code: str,
    timeout: float = 5.0,
    filter_warnings: bool = True,
) -> tuple[Optional[list[str]], str]:
    """Execute Python code in a subprocess and capture output.

    Returns:
        (result_lines, error_message)  — result_lines is None on failure.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_filename = f.name
        if filter_warnings:
            f.write("import warnings\n")
            f.write("warnings.filterwarnings('ignore')\n")
        f.write(python_code)

    try:
        process = subprocess.Popen(
            ["python", temp_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(timeout=timeout)
        output = stdout.decode("utf-8").strip()

        if process.returncode != 0:
            error_output = stderr.decode("utf-8").strip()
            error_lines = error_output.splitlines()
            error_msg = error_lines[-1] if error_lines else "Unknown Error"
            return None, error_msg

        result = output.splitlines() if output else []
        if len(result) == 0:
            return None, "No Output"
        return result, ""

    except subprocess.TimeoutExpired:
        process.kill()
        return None, "TimeoutError"
    except Exception as e:
        return None, str(e)
    finally:
        try:
            os.unlink(temp_filename)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# High-level API (used by step reward)
# ---------------------------------------------------------------------------

def fol_preprocess_declarations(
    context: str,
    question: str,
    options: Optional[str] = None,
    *,
    api_config: Optional[dict] = None,
) -> str:
    """Generate Z3 declarations from context. One-time per problem."""
    return generate_z3_declarations(context, question, options, api_config=api_config)


def translate_and_verify_step(
    context: str,
    declarations: str,
    step_text: str,
    *,
    api_config: Optional[dict] = None,
) -> float:
    """Translate a step to Z3 and verify entailment.

    Full pipeline for a single step:
      1. Parse NL step text for premise/conclusion tags
      2. Translate the whole step to Z3 using LLM
      3. Parse the Z3 code for per-step FOL expressions
      4. Verify via Z3 solver

    Returns:
        1.0 if entailed, 0.0 otherwise.
    """
    # Strategy 1: Try direct LLM translation of the step to Z3
    try:
        z3_code = translate_step_to_z3_implication(
            context, declarations, step_text, api_config=api_config
        )
        if z3_code:
            # Try to parse the translated code for premises/conclusion
            parsed_steps = parse_python_logic_steps(z3_code)
            if parsed_steps:
                step_data = parsed_steps[0]
                premises_fol = step_data["premises"]
                conclusion_fol = step_data["conclusion"]
                if conclusion_fol:
                    conclusion_str = conclusion_fol[0] if isinstance(conclusion_fol, list) else conclusion_fol
                    result, error = verify_step_fol(declarations, premises_fol, conclusion_str)
                    if result:
                        try:
                            return float(result[-1])
                        except (ValueError, IndexError):
                            pass

            # Fallback: try executing the translated code directly
            # (some LLMs produce self-contained Z3 scripts)
            full_code = f"from z3 import *\n\n{declarations}\n\n{z3_code}"
            result, error = execute_program(full_code)
            if result:
                # Look for entailment signals
                for line in result:
                    if "SUCCESS_ENTAILED" in line or "unsat" in line.lower():
                        return 1.0
                    if "FAILED" in line or "sat" == line.strip().lower():
                        return 0.0
                # Try last line as numeric
                try:
                    return float(result[-1])
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass

    return 0.0
