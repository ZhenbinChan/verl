"""
Configurable FOL verification engine.

Supports two preprocessing pipelines (direct / structured) and
two translation modes (implication / assertion). Verification semantics
is always entailment: UNSAT of (premises AND NOT conclusion) -> 1.0.
"""

import concurrent.futures
import logging
from dataclasses import dataclass, field
from enum import Enum
from string import Template
from typing import Optional

from verl.utils.fol_utils.common import (
    # Prompt paths
    Z3_DECLARATION_PROMPT,
    Z3_IMPLICATION_PROMPT,
    TRANSLATE_STEP_PROMPT,
    # LLM calls
    call_llm,
    call_llm_structured,
    # Text extraction
    extract_python_block,
    load_prompt,
    parse_python_logic_steps,
    # Structured pipeline helpers
    rephrase,
    object_extract,
    predicate_extract,
    generate_z3_declarations_from_entities,
    generate_z3_functions,
    # Execution
    correct_loop,
    run_code,
    # Caching
    thread_safe_cache,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class PreprocessPipeline(Enum):
    DIRECT = "direct"          # 1 LLM call -> z3_declaration_generation.txt
    STRUCTURED = "structured"  # rephrase || object_extract -> predicate_extract -> code-gen


class TranslationMode(Enum):
    IMPLICATION = "implication"  # z3_implication_conversion.txt -> premises_N/conclusion_N
    ASSERTION = "assertion"      # translate_step.txt -> premise_fol/conclusion_fol


@dataclass
class FOLConfig:
    preprocess: PreprocessPipeline = PreprocessPipeline.DIRECT
    translation: TranslationMode = TranslationMode.IMPLICATION
    max_tries: int = 1
    timeout: float = 30.0
    cumulative: bool = False
    api_config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cached preprocessing
# ---------------------------------------------------------------------------

@thread_safe_cache
def _preprocess_direct(
    context: str, question: str, options: str = "",
    *, api_config: Optional[dict] = None,
) -> tuple[str, str]:
    """Direct pipeline: 1 LLM call to generate rich Z3 declarations.

    Returns (context, declarations).
    """
    system_prompt = load_prompt(Z3_DECLARATION_PROMPT)
    user_input = f"<Context>{context}</Context>\n<Question>{question}</Question>"
    if options:
        user_input += f"\n<Options>{options}</Options>"

    response = call_llm(user_input, api_config=api_config, system_prompt=system_prompt)
    declarations = extract_python_block(response, strategy="all")
    return context, declarations


@thread_safe_cache
def _preprocess_structured(
    context: str, question: str, options: str = "",
    *, api_config: Optional[dict] = None,
) -> tuple[str, str]:
    """Structured pipeline: rephrase + extract entities/predicates + code-gen.

    Returns (rephrased_context, declarations).
    """
    # Parallelize rephrase and object extraction
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        rephrase_future = executor.submit(
            rephrase, context, question, options, api_config=api_config
        )
        object_future = executor.submit(
            object_extract, context, question, options, api_config=api_config
        )
        rephrased = rephrase_future.result()
        entities = object_future.result()

    predicates = predicate_extract(
        context, question, options, objectives=entities, api_config=api_config
    )
    z3_decl = generate_z3_declarations_from_entities(entities)
    z3_funcs = generate_z3_functions(predicates)
    declaration_code = z3_decl + "\n" + z3_funcs
    return rephrased, declaration_code


# ---------------------------------------------------------------------------
# Translation: implication mode
# ---------------------------------------------------------------------------

def _translate_implication(
    context: str,
    declarations: str,
    step_text: str,
    *,
    api_config: Optional[dict] = None,
) -> str:
    """Implication-mode translation.

    Uses z3_implication_conversion.txt to translate step into premises_N/conclusion_N.
    Then builds a complete Z3 entailment-check script:
      solver.add(And(premises))
      solver.add(Not(conclusion))
      UNSAT -> 1.0

    Returns executable Z3 Python code string.
    """
    system_prompt = load_prompt(Z3_IMPLICATION_PROMPT)
    user_input = (
        f"Z3 Declarations:\n```python\n{declarations}\n```\n\n"
        f"Context:\n{context}\n\n"
        f"Reasoning Step:\n{step_text}"
    )
    response = call_llm(user_input, api_config=api_config, system_prompt=system_prompt)
    z3_code = extract_python_block(response, strategy="all")

    # Parse for premises_N / conclusion_N
    parsed_steps = parse_python_logic_steps(z3_code)
    if parsed_steps:
        step_data = parsed_steps[0]
        premises_fol = step_data["premises"]
        conclusion_fol = step_data["conclusion"]
        if conclusion_fol:
            conclusion_str = conclusion_fol[0] if isinstance(conclusion_fol, list) else conclusion_fol
            return _build_entailment_code(declarations, premises_fol, conclusion_str)

    # Fallback: try executing the translated code directly
    return f"from z3 import *\n\n{declarations}\n\n{z3_code}"


def _build_entailment_code(
    declarations: str,
    premises_fol: list[str],
    conclusion_fol: str,
) -> str:
    """Build Z3 entailment-check script.

    Adds premises and NOT(conclusion); if UNSAT, entailed -> 1.0.
    """
    premises_str = ", ".join(premises_fol) if premises_fol else "True"
    return f"""\
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
    print("FAILED_NOT_ENTAILED")
    print(0.0)
else:
    print("UNKNOWN")
    print(0.0)
"""


# ---------------------------------------------------------------------------
# Translation: assertion mode
# ---------------------------------------------------------------------------

def _translate_assertion(
    context: str,
    declarations: str,
    step_text: str,
    *,
    api_config: Optional[dict] = None,
) -> str:
    """Assertion-mode translation.

    Uses translate_step.txt to translate step into premise_fol/conclusion_fol.
    The prompt instructs the LLM to negate the conclusion, so conclusion_fol
    already contains Not(...).

    Wraps into a complete Z3 script that checks UNSAT -> 1.0.

    Returns executable Z3 Python code string.
    """
    template = load_prompt(TRANSLATE_STEP_PROMPT)
    prompt = Template(template).safe_substitute(
        context=context, declaration=declarations, step=step_text
    )
    trans_output = call_llm(prompt, api_config=api_config)
    trans_code = extract_python_block(trans_output)
    return _wrap_assertion_z3_code(declarations, trans_code)


def _wrap_assertion_z3_code(declaration: str, expression: str) -> str:
    """Assemble complete Z3 script from assertion-mode translation.

    The LLM has already negated the conclusion in conclusion_fol,
    so we check UNSAT -> entailed -> 1.0.
    """
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
    # UNSAT = conclusion is entailed (premise AND NOT(conclusion) unsatisfiable)
    z3_code += "if result == unsat:\n"
    z3_code += "    print('SUCCESS_ENTAILED')\n"
    z3_code += "    print(1.0)\n"
    z3_code += "else:\n"
    z3_code += "    print(0.0)\n"
    return z3_code


# ---------------------------------------------------------------------------
# FOL Engine
# ---------------------------------------------------------------------------

class FOLEngine:
    """Unified FOL verification engine.

    Composes preprocessing, translation, and verification stages
    based on FOLConfig. Verification semantics is always entailment:
    UNSAT of (premises AND NOT conclusion) -> 1.0.
    """

    def __init__(self, config: FOLConfig):
        self.config = config

    def preprocess(
        self, context: str, question: str, options: str = "",
    ) -> tuple[str, str]:
        """Run preprocessing pipeline.

        Returns:
            (context_for_translation, declaration_code)
        """
        if self.config.preprocess == PreprocessPipeline.DIRECT:
            return _preprocess_direct(
                context, question, options, api_config=self.config.api_config
            )
        else:
            return _preprocess_structured(
                context, question, options, api_config=self.config.api_config
            )

    def verify_step(
        self, processed_context: str, declarations: str, step_text: str,
    ) -> float:
        """Translate step to Z3, execute with correction loop, return reward.

        Returns:
            1.0 if entailed, 0.0 otherwise.
        """
        try:
            # Step 1: Translate to Z3 code
            if self.config.translation == TranslationMode.IMPLICATION:
                z3_code = _translate_implication(
                    processed_context, declarations, step_text,
                    api_config=self.config.api_config,
                )
            else:
                z3_code = _translate_assertion(
                    processed_context, declarations, step_text,
                    api_config=self.config.api_config,
                )

            # Step 2: Execute with auto-correction loop
            result = correct_loop(
                z3_code,
                api_config=self.config.api_config,
                max_tries=self.config.max_tries,
                timeout=self.config.timeout,
            )

            # Step 3: Parse result
            if result["success"] and result.get("output"):
                output = result["output"].strip()
                lines = output.splitlines()
                for line in lines:
                    if "SUCCESS_ENTAILED" in line:
                        return 1.0
                    if "1.0" in line:
                        return 1.0
                # Try last line as numeric
                try:
                    return float(lines[-1])
                except (ValueError, IndexError):
                    pass

            return 0.0

        except Exception as e:
            logger.warning("FOL engine verify_step failed: %s", e)
            return 0.0
