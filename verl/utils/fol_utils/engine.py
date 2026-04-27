"""
Configurable FOL verification engine.

Supports two preprocessing pipelines (direct / structured) and
two translation modes (implication / assertion). Verification semantics
is always entailment: UNSAT of (premises AND NOT conclusion) -> 1.0.
"""

import ast
import concurrent.futures
import json
import keyword
import logging
import re
import time
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
    extract_structured_python_code,
    load_prompt,
    # Structured pipeline helpers
    rephrase,
    object_extract,
    predicate_extract,
    generate_z3_declarations_from_entities,
    generate_z3_functions,
    # Execution
    correct_loop,
    run_code,
    use_outlines,
    # Caching
    thread_safe_cache,
)

logger = logging.getLogger(__name__)

_DECLARATION_SCHEMA = {
    "type": "object",
    "properties": {
        "sorts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "enum_values": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "enum_values"],
                "additionalProperties": False,
            },
        },
        "variables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "sort": {"type": "string"},
                },
                "required": ["name", "sort"],
                "additionalProperties": False,
            },
        },
        "constants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "sort": {"type": "string"},
                },
                "required": ["name", "sort"],
                "additionalProperties": False,
            },
        },
        "functions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arg_sorts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "return_sort": {"type": "string"},
                },
                "required": ["name", "arg_sorts", "return_sort"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sorts", "variables", "constants", "functions"],
    "additionalProperties": False,
}

_IMPLICATION_TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "new_variables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "sort": {"type": "string"},
                },
                "required": ["name", "sort"],
                "additionalProperties": False,
            },
        },
        "background_axioms": {
            "type": "array",
            "items": {"type": "string"},
        },
        "previous_conclusions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "current_premises": {
            "type": "array",
            "items": {"type": "string"},
        },
        "conclusion": {"type": "string"},
    },
    "required": ["new_variables", "background_axioms", "previous_conclusions", "current_premises", "conclusion"],
    "additionalProperties": False,
}

_Z3_EXPR_OPERATOR_NAMES = {
    "And",
    "Or",
    "Not",
    "Implies",
    "ForAll",
    "Exists",
    "If",
    "Distinct",
    "Sum",
    "IntVal",
    "RealVal",
    "BoolVal",
    "True",
    "False",
}

_DECLARATION_BUILTIN_RETURN_SORTS = {"BoolSort()", "IntSort()", "RealSort()"}

_ASSERTION_TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "new_variables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "sort": {"type": "string"},
                },
                "required": ["name", "sort"],
                "additionalProperties": False,
            },
        },
        "premise_fol": {
            "type": "array",
            "items": {"type": "string"},
        },
        "conclusion_fol": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["new_variables", "premise_fol", "conclusion_fol"],
    "additionalProperties": False,
}


_PYTHON_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "python_code": {"type": "string"},
    },
    "required": ["python_code"],
    "additionalProperties": False,
}


def _judge_use_outlines(api_config: Optional[dict]) -> bool:
    """Whether to request structured JSON output from the FOL judge."""
    return use_outlines(api_config)


def _structured_python_fallback(
    prompt: str,
    *,
    api_config: Optional[dict] = None,
    system_prompt: Optional[str] = None,
    usage_info: Optional[dict] = None,
    debug_info: Optional[dict] = None,
    response_name: str,
) -> str:
    """Request executable Python code via a strict schema.

    This is a fail-closed fallback for translation paths that previously fell
    back to free-form text, which could leak natural language into run_code().
    """
    payload = call_llm_structured(
        (
            f"{prompt}\n\n"
            "Return a JSON object with a single field `python_code` containing "
            "only executable Python/Z3 code. Do not include explanations."
        ),
        api_config=api_config,
        system_prompt=system_prompt,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": response_name,
                "schema": _PYTHON_CODE_SCHEMA,
            },
        },
        usage_info=usage_info,
    )
    if debug_info is not None and payload is not None:
        debug_info["translation_response"] = json.dumps(payload, ensure_ascii=False, indent=2)
    return extract_structured_python_code(payload)


def _render_const_declarations(items: list[dict], *, section_name: str) -> str:
    """Render variable / constant declarations from schema payload."""
    lines = [f"# {section_name}"]
    for item in items:
        name = item.get("name")
        sort = item.get("sort")
        if isinstance(name, str) and isinstance(sort, str):
            lines.append(f"{name} = Const('{name}', {sort})")
    return "\n".join(lines) if len(lines) > 1 else ""


def _render_declarations_from_schema(payload: Optional[dict]) -> str:
    """Render Z3 declaration code from a structured declaration schema."""
    if not isinstance(payload, dict):
        return ""

    lines = ["from z3 import *", "", "# Declare Sorts & Enums"]
    for sort_item in payload.get("sorts", []):
        if not isinstance(sort_item, dict):
            continue
        name = sort_item.get("name")
        enum_values = sort_item.get("enum_values", [])
        if not isinstance(name, str):
            continue
        if isinstance(enum_values, list) and len(enum_values) > 0 and all(isinstance(v, str) for v in enum_values):
            enum_names = ", ".join(enum_values)
            enum_literals = ", ".join(f"'{value}'" for value in enum_values)
            lines.append(f"{name}, ({enum_names}) = EnumSort('{name}', [{enum_literals}])")
        else:
            lines.append(f"{name} = DeclareSort('{name}')")

    variable_block = _render_const_declarations(payload.get("variables", []), section_name="Declare Variables")
    if variable_block:
        lines.extend(["", variable_block])

    constant_block = _render_const_declarations(payload.get("constants", []), section_name="Declare Constants")
    if constant_block:
        lines.extend(["", constant_block])

    lines.extend(["", "# Declare Functions"])
    for func_item in payload.get("functions", []):
        if not isinstance(func_item, dict):
            continue
        name = func_item.get("name")
        arg_sorts = func_item.get("arg_sorts", [])
        return_sort = func_item.get("return_sort")
        if not isinstance(name, str) or not isinstance(return_sort, str) or not isinstance(arg_sorts, list):
            continue
        if not all(isinstance(sort, str) for sort in arg_sorts):
            continue
        signature = ", ".join([*arg_sorts, return_sort])
        lines.append(f"{name} = Function('{name}', {signature})")

    return "\n".join(lines).strip()


def _is_valid_python_identifier(name: object) -> bool:
    """Whether a schema identifier can be emitted as a Python variable name."""
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def _collect_declaration_payload_errors(payload: Optional[dict]) -> list[dict[str, object]]:
    """Validate declaration JSON before rendering it into executable Z3 code."""
    errors: list[dict[str, object]] = []
    if not isinstance(payload, dict):
        return [{"field": "$", "error": "payload is not a JSON object"}]

    for field_name in ("sorts", "variables", "constants", "functions"):
        if not isinstance(payload.get(field_name), list):
            errors.append({"field": field_name, "error": "field must be an array"})

    sorts = payload.get("sorts", []) if isinstance(payload.get("sorts"), list) else []
    variables = payload.get("variables", []) if isinstance(payload.get("variables"), list) else []
    constants = payload.get("constants", []) if isinstance(payload.get("constants"), list) else []
    functions = payload.get("functions", []) if isinstance(payload.get("functions"), list) else []

    if not sorts:
        errors.append({"field": "sorts", "error": "at least one sort is required"})

    declared_sorts: set[str] = set()
    emitted_names: set[str] = set()

    def add_name(name: object, field: str) -> None:
        if not _is_valid_python_identifier(name):
            errors.append({"field": field, "name": name, "error": "invalid Python identifier"})
            return
        if name in emitted_names:
            errors.append({"field": field, "name": name, "error": "duplicate emitted identifier"})
            return
        emitted_names.add(str(name))

    for idx, item in enumerate(sorts):
        if not isinstance(item, dict):
            errors.append({"field": f"sorts[{idx}]", "error": "item must be an object"})
            continue
        name = item.get("name")
        add_name(name, f"sorts[{idx}].name")
        if isinstance(name, str):
            declared_sorts.add(name)
        enum_values = item.get("enum_values", [])
        if not isinstance(enum_values, list) or not all(isinstance(value, str) for value in enum_values):
            errors.append({"field": f"sorts[{idx}].enum_values", "error": "enum_values must be an array of strings"})
            continue
        for enum_idx, value in enumerate(enum_values):
            add_name(value, f"sorts[{idx}].enum_values[{enum_idx}]")

    for field_name, items in (("variables", variables), ("constants", constants)):
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append({"field": f"{field_name}[{idx}]", "error": "item must be an object"})
                continue
            add_name(item.get("name"), f"{field_name}[{idx}].name")
            sort = item.get("sort")
            if not isinstance(sort, str) or sort not in declared_sorts:
                errors.append({"field": f"{field_name}[{idx}].sort", "sort": sort, "error": "unknown sort"})

    for idx, item in enumerate(functions):
        if not isinstance(item, dict):
            errors.append({"field": f"functions[{idx}]", "error": "item must be an object"})
            continue
        add_name(item.get("name"), f"functions[{idx}].name")
        arg_sorts = item.get("arg_sorts")
        if not isinstance(arg_sorts, list) or not all(isinstance(sort, str) for sort in arg_sorts):
            errors.append({"field": f"functions[{idx}].arg_sorts", "error": "arg_sorts must be an array of strings"})
        else:
            for sort_idx, sort in enumerate(arg_sorts):
                if sort not in declared_sorts:
                    errors.append({
                        "field": f"functions[{idx}].arg_sorts[{sort_idx}]",
                        "sort": sort,
                        "error": "unknown sort",
                    })
        return_sort = item.get("return_sort")
        if not isinstance(return_sort, str) or (
            return_sort not in declared_sorts and return_sort not in _DECLARATION_BUILTIN_RETURN_SORTS
        ):
            errors.append({"field": f"functions[{idx}].return_sort", "sort": return_sort, "error": "unknown return sort"})

    return errors


def _render_validated_declarations_from_schema(payload: Optional[dict]) -> tuple[str, list[dict[str, object]]]:
    """Render declaration code only if schema and executable syntax validate."""
    errors = _collect_declaration_payload_errors(payload)
    if errors:
        return "", errors
    declarations = _render_declarations_from_schema(payload)
    if not declarations:
        return "", [{"field": "$", "error": "rendered declarations are empty"}]
    res = run_code(f"{declarations}\n\nprint('DECLARATION_OK')", timeout=5.0)
    if not res.get("success"):
        return "", [{"field": "$", "error": "rendered declaration code failed", "detail": res.get("error") or res.get("output")}]
    return declarations, []


def _repair_declaration_payload(
    payload: Optional[dict],
    errors: list[dict[str, object]],
    *,
    api_config: Optional[dict] = None,
    system_prompt: Optional[str] = None,
) -> Optional[dict]:
    """Ask the judge to repair only the declaration JSON schema payload."""
    cfg = dict(api_config or {})
    max_tries = max(0, int(cfg.get("max_tries", 0) or 0))
    if max_tries <= 0:
        return payload

    current = payload if isinstance(payload, dict) else {
        "sorts": [],
        "variables": [],
        "constants": [],
        "functions": [],
    }
    repair_prompt_base = (
        "Repair this Z3 declaration JSON object so it follows the provided schema and renders as valid Z3-Python declarations.\n"
        "Do not output Python code. Do not add solver.add, axioms, facts, premises, conclusions, or background knowledge.\n"
        "Only repair identifiers, duplicate names, missing/unknown sort references, and return_sort spellings.\n"
        "Use return_sort values like BoolSort(), IntSort(), RealSort(), or a declared custom sort name.\n\n"
    )
    for attempt in range(max_tries):
        cfg["temperature"] = float(cfg.get("temperature", 0.2)) + 0.05 * (attempt + 1)
        prompt = (
            repair_prompt_base +
            f"Declaration JSON object:\n{json.dumps(current, ensure_ascii=False, indent=2)}\n\n"
            f"Validation errors:\n{json.dumps(errors, ensure_ascii=False, indent=2)}"
        )
        repaired = call_llm_structured(
            prompt,
            api_config=cfg,
            system_prompt=system_prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "fol-z3-declaration-repair",
                    "schema": _DECLARATION_SCHEMA,
                },
            },
        )
        declarations, repaired_errors = _render_validated_declarations_from_schema(repaired)
        if declarations and not repaired_errors:
            return repaired
        errors = repaired_errors
        current = repaired if isinstance(repaired, dict) else current
    return payload


def _render_structured_implication(
    payload: Optional[dict],
    declarations: str,
    debug_info: Optional[dict] = None,
) -> str:
    """Render complete entailment code from implication schema payload."""
    if not isinstance(payload, dict):
        return ""

    var_block = _render_const_declarations(payload.get("new_variables", []), section_name="New Variables")
    background_axioms = payload.get("background_axioms", [])
    previous_conclusions = payload.get("previous_conclusions", [])
    current_premises = payload.get("current_premises", [])
    conclusion = payload.get("conclusion")
    premise_groups = [background_axioms, previous_conclusions, current_premises]
    if any(not isinstance(group, list) or not all(isinstance(item, str) for item in group) for group in premise_groups):
        return ""
    if not isinstance(conclusion, str) or not conclusion.strip():
        return ""

    full_declarations = declarations
    if var_block:
        full_declarations = f"{declarations}\n\n{var_block}"
    premises = [*background_axioms, *previous_conclusions, *current_premises]
    expression_errors = _collect_z3_expression_errors(
        {
            "background_axioms": background_axioms,
            "previous_conclusions": previous_conclusions,
            "current_premises": current_premises,
            "conclusion": [conclusion],
        }
    )
    if expression_errors:
        if debug_info is not None:
            debug_info["invalid_expression_syntax"] = expression_errors
        return _build_fail_closed_code(full_declarations, "FAILED_INVALID_EXPRESSION")
    unknown_identifier_errors = _collect_unknown_identifier_errors(
        {
            "background_axioms": background_axioms,
            "previous_conclusions": previous_conclusions,
            "current_premises": current_premises,
            "conclusion": [conclusion],
        },
        full_declarations,
        payload.get("new_variables", []),
    )
    if unknown_identifier_errors:
        if debug_info is not None:
            debug_info["unknown_translation_identifiers"] = unknown_identifier_errors
        return _build_fail_closed_code(full_declarations, "FAILED_INVALID_TRANSLATION")
    leaked_sources = _find_exact_conclusion_leaks(
        conclusion,
        {
            "background_axioms": background_axioms,
            "previous_conclusions": previous_conclusions,
            "current_premises": current_premises,
        },
    )
    if leaked_sources:
        if debug_info is not None:
            debug_info["conclusion_leakage_detected"] = True
            debug_info["conclusion_leakage_sources"] = leaked_sources
        return _build_fail_closed_code(full_declarations, "FAILED_LEAKED_CONCLUSION")
    return _build_entailment_code(full_declarations, premises, conclusion)


def _validate_z3_expression_syntax(expr: str) -> Optional[str]:
    """Return a syntax error message if an expression is not Z3-Python syntax."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp):
            return "Python boolean operators are not valid for symbolic Z3 expressions; use And(...) or Or(...)."
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return "Python 'not' is not valid for symbolic Z3 expressions; use Not(...)."
    return None


def _collect_z3_expression_errors(expr_sources: dict[str, list[str]]) -> list[dict[str, object]]:
    """Collect Z3 expression syntax errors by source field and item index."""
    errors = []
    for source, expressions in expr_sources.items():
        for idx, expr in enumerate(expressions):
            error = _validate_z3_expression_syntax(expr)
            if error:
                errors.append({"source": source, "index": idx, "expr": expr, "error": error})
    return errors


def _identifier_set(expr: str) -> set[str]:
    """Return non-operator identifiers appearing in a Z3 expression string."""
    names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr))
    return names - _Z3_EXPR_OPERATOR_NAMES


def _collect_assignment_target_names(target: ast.AST, names: set[str]) -> None:
    """Collect Python names bound by assignment targets in rendered Z3 code."""
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            _collect_assignment_target_names(element, names)


def _declared_identifier_names(declarations: str, new_variables: list[dict[str, str]]) -> set[str]:
    """Return identifiers available to implication expressions."""
    names = set(_Z3_EXPR_OPERATOR_NAMES)
    try:
        tree = ast.parse(declarations)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _collect_assignment_target_names(target, names)
    for item in new_variables:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.add(item["name"])
    return names


def _collect_unknown_identifier_errors(
    expr_sources: dict[str, list[str]],
    declarations: str,
    new_variables: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Collect expression identifiers that are not declared in the Z3 vocabulary."""
    declared_names = _declared_identifier_names(declarations, new_variables)
    errors: list[dict[str, object]] = []
    for source, expressions in expr_sources.items():
        for idx, expr in enumerate(expressions):
            unknown = sorted(_identifier_set(expr) - declared_names)
            if unknown:
                errors.append({"source": source, "index": idx, "expr": expr, "unknown_identifiers": unknown})
    return errors


def _implication_payload_repair_is_conservative(original: dict, repaired: dict) -> bool:
    """Check that expression repair did not add/remove facts or identifiers."""
    if not isinstance(repaired, dict):
        return False
    if repaired.get("new_variables", []) != original.get("new_variables", []):
        return False
    for field_name in ("background_axioms", "previous_conclusions", "current_premises"):
        before = original.get(field_name, [])
        after = repaired.get(field_name, [])
        if not isinstance(before, list) or not isinstance(after, list) or len(before) != len(after):
            return False
        for before_expr, after_expr in zip(before, after):
            if not isinstance(before_expr, str) or not isinstance(after_expr, str):
                return False
            if _identifier_set(before_expr) != _identifier_set(after_expr):
                return False
    before_conclusion = original.get("conclusion")
    after_conclusion = repaired.get("conclusion")
    if not isinstance(before_conclusion, str) or not isinstance(after_conclusion, str):
        return False
    return _identifier_set(before_conclusion) == _identifier_set(after_conclusion)


def _repair_implication_expressions(
    payload: Optional[dict],
    errors: list[dict[str, object]],
    *,
    api_config: Optional[dict] = None,
    usage_info: Optional[dict] = None,
    debug_info: Optional[dict] = None,
) -> Optional[dict]:
    """Ask the judge to repair only malformed Z3 expression strings."""
    if not isinstance(payload, dict) or not errors:
        return payload
    cfg = dict(api_config or {})
    max_tries = max(0, int(cfg.get("max_tries", 0) or 0))
    if max_tries <= 0:
        return payload

    current = payload
    repair_prompt_base = (
        "Repair only the malformed Z3-Python expression strings in this JSON object.\n"
        "Do not add, remove, or reorder premises. Do not change new_variables.\n"
        "Do not change the logical meaning, identifiers, predicates, functions, constants, or source fields.\n"
        "Only convert invalid Python/Z3 expression syntax into valid Z3 API syntax, for example "
        "`A And B` -> `And(A, B)`.\n\n"
    )
    for attempt in range(max_tries):
        cfg["temperature"] = float(cfg.get("temperature", 0.2)) + 0.05 * (attempt + 1)
        prompt = (
            repair_prompt_base +
            f"JSON object:\n{json.dumps(current, ensure_ascii=False, indent=2)}\n\n"
            f"Expression syntax errors:\n{json.dumps(errors, ensure_ascii=False, indent=2)}"
        )
        repaired = call_llm_structured(
            prompt,
            api_config=cfg,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "fol-implication-expression-repair",
                    "schema": _IMPLICATION_TRANSLATION_SCHEMA,
                },
            },
            usage_info=usage_info,
        )
        if debug_info is not None:
            debug_info["expression_correction_attempts"] = attempt + 1
            debug_info["expression_correction_response"] = (
                json.dumps(repaired, ensure_ascii=False, indent=2) if repaired is not None else None
            )
        if not _implication_payload_repair_is_conservative(current, repaired):
            if debug_info is not None:
                debug_info["expression_correction_rejected"] = "non_conservative_repair"
            continue
        repaired_errors = _collect_z3_expression_errors(
            {
                "background_axioms": repaired.get("background_axioms", []),
                "previous_conclusions": repaired.get("previous_conclusions", []),
                "current_premises": repaired.get("current_premises", []),
                "conclusion": [repaired.get("conclusion", "")],
            }
        )
        if not repaired_errors:
            return repaired
        errors = repaired_errors
        current = repaired
    if debug_info is not None:
        debug_info["expression_correction_failed"] = True
    return payload


def _parse_json_object_response(response: str) -> Optional[dict]:
    """Parse a JSON object from a plain LLM response."""
    if not isinstance(response, str):
        return None
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        payload = json.loads(response[start:end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_fol_expr(expr: str) -> str:
    """Normalize a FOL expression for conservative exact leakage checks."""
    return "".join(str(expr).split())


def _find_exact_conclusion_leaks(conclusion: str, premise_sources: dict[str, list[str]]) -> list[str]:
    """Return source names whose premises contain the conclusion verbatim.

    This intentionally uses exact normalized equality only. Stronger substring
    checks would incorrectly reject legitimate axioms such as Implies(P, C).
    """
    conclusion_norm = _normalize_fol_expr(conclusion)
    if not conclusion_norm:
        return []
    leaked = []
    for source, premises in premise_sources.items():
        if any(_normalize_fol_expr(premise) == conclusion_norm for premise in premises):
            leaked.append(source)
    return leaked


def _build_fail_closed_code(declarations: str, reason: str) -> str:
    """Build executable Z3 code that deterministically returns reward 0."""
    return f"""\
from z3 import *

{declarations}

print("{reason}")
print(0.0)
"""


def _render_structured_assertion(
    payload: Optional[dict],
    declarations: str,
) -> str:
    """Render assertion-mode helper code from structured schema payload."""
    if not isinstance(payload, dict):
        return ""

    premise_fol = payload.get("premise_fol", [])
    conclusion_fol = payload.get("conclusion_fol", [])
    if not isinstance(premise_fol, list) or not all(isinstance(item, str) for item in premise_fol):
        return ""
    if not isinstance(conclusion_fol, list) or not all(isinstance(item, str) for item in conclusion_fol):
        return ""

    expression_lines = []
    var_block = _render_const_declarations(payload.get("new_variables", []), section_name="New Variables")
    if var_block:
        expression_lines.append(var_block)
        expression_lines.append("")
    expression_lines.append("premise_fol = [")
    for expr in premise_fol:
        expression_lines.append(f"    {expr},")
    expression_lines.append("]")
    expression_lines.append("")
    expression_lines.append("conclusion_fol = [")
    for expr in conclusion_fol:
        expression_lines.append(f"    {expr},")
    expression_lines.append("]")
    return _wrap_assertion_z3_code(declarations, "\n".join(expression_lines))


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
    old_max_tries: int = 0
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

    if _judge_use_outlines(api_config):
        structured_input = (
            f"{user_input}\n\n"
            "Return a JSON object that follows the provided schema exactly. "
            "Use empty arrays instead of omitted fields."
        )
        payload = call_llm_structured(
            structured_input,
            api_config=api_config,
            system_prompt=system_prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "fol-z3-declarations",
                    "schema": _DECLARATION_SCHEMA,
                },
            },
        )
        declarations, declaration_errors = _render_validated_declarations_from_schema(payload)
        if not declarations:
            repaired_payload = _repair_declaration_payload(
                payload,
                declaration_errors,
                api_config=api_config,
                system_prompt=system_prompt,
            )
            declarations, declaration_errors = _render_validated_declarations_from_schema(repaired_payload)
        if not declarations:
            return context, ""
    else:
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
    debug_info: Optional[dict] = None,
) -> str:
    """Implication-mode translation.

    Uses z3_implication_conversion.txt to translate step into source-separated
    premise groups and a conclusion. Then builds a complete Z3 entailment-check
    script:
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
    usage_info = None
    if debug_info is not None:
        usage_info = debug_info.setdefault(
            "judge_usage",
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
    if _judge_use_outlines(api_config):
        structured_input = (
            f"{user_input}\n\n"
            "Return a JSON object that follows the provided schema exactly. "
            "Use only strings built from the provided Z3 declarations."
        )
        payload = call_llm_structured(
            structured_input,
            api_config=api_config,
            system_prompt=system_prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "fol-implication-translation",
                    "schema": _IMPLICATION_TRANSLATION_SCHEMA,
                },
            },
            usage_info=usage_info,
        )
        if debug_info is not None and payload is not None:
            debug_info["translation_response"] = json.dumps(payload, ensure_ascii=False, indent=2)
        expression_errors = _collect_z3_expression_errors(
            {
                "background_axioms": payload.get("background_axioms", []) if isinstance(payload, dict) else [],
                "previous_conclusions": payload.get("previous_conclusions", []) if isinstance(payload, dict) else [],
                "current_premises": payload.get("current_premises", []) if isinstance(payload, dict) else [],
                "conclusion": [payload.get("conclusion", "")] if isinstance(payload, dict) else [],
            }
        )
        if expression_errors:
            if debug_info is not None:
                debug_info["invalid_expression_syntax_initial"] = expression_errors
            payload = _repair_implication_expressions(
                payload,
                expression_errors,
                api_config=api_config,
                usage_info=usage_info,
                debug_info=debug_info,
            )
        structured_code = _render_structured_implication(payload, declarations, debug_info=debug_info)
        if structured_code:
            return structured_code
        if debug_info is not None and debug_info.get("translation_response") is None:
            debug_info["translation_response"] = None
        if debug_info is not None:
            debug_info["translation_failed_closed"] = "invalid_structured_implication"
        return _build_fail_closed_code(declarations, "FAILED_INVALID_TRANSLATION")
    else:
        response = call_llm(
            user_input,
            api_config=api_config,
            system_prompt=system_prompt,
            usage_info=usage_info,
        )
        if debug_info is not None:
            debug_info["translation_response"] = response
        payload = _parse_json_object_response(response)
        expression_errors = _collect_z3_expression_errors(
            {
                "background_axioms": payload.get("background_axioms", []) if isinstance(payload, dict) else [],
                "previous_conclusions": payload.get("previous_conclusions", []) if isinstance(payload, dict) else [],
                "current_premises": payload.get("current_premises", []) if isinstance(payload, dict) else [],
                "conclusion": [payload.get("conclusion", "")] if isinstance(payload, dict) else [],
            }
        )
        if expression_errors:
            if debug_info is not None:
                debug_info["invalid_expression_syntax_initial"] = expression_errors
            payload = _repair_implication_expressions(
                payload,
                expression_errors,
                api_config=api_config,
                usage_info=usage_info,
                debug_info=debug_info,
            )
        structured_code = _render_structured_implication(payload, declarations, debug_info=debug_info)
        if structured_code:
            return structured_code
        if debug_info is not None:
            debug_info["translation_failed_closed"] = "invalid_json_implication"
        return _build_fail_closed_code(declarations, "FAILED_INVALID_TRANSLATION")


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
    debug_info: Optional[dict] = None,
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
    usage_info = None
    if debug_info is not None:
        usage_info = debug_info.setdefault(
            "judge_usage",
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
    if _judge_use_outlines(api_config):
        structured_prompt = (
            f"{prompt}\n\n"
            "Return a JSON object that follows the provided schema exactly. "
            "Use empty arrays instead of omitted fields."
        )
        payload = call_llm_structured(
            structured_prompt,
            api_config=api_config,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "fol-assertion-translation",
                    "schema": _ASSERTION_TRANSLATION_SCHEMA,
                },
            },
            usage_info=usage_info,
        )
        if debug_info is not None and payload is not None:
            debug_info["translation_response"] = json.dumps(payload, ensure_ascii=False, indent=2)
        structured_code = _render_structured_assertion(payload, declarations)
        if structured_code:
            return structured_code
        trans_code = _structured_python_fallback(
            prompt,
            api_config=api_config,
            usage_info=usage_info,
            debug_info=debug_info,
            response_name="fol-assertion-python-fallback",
        )
    else:
        trans_output = call_llm(prompt, api_config=api_config, usage_info=usage_info)
        if debug_info is not None:
            debug_info["translation_response"] = trans_output
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
        self, processed_context: str, declarations: str, step_text: str, debug_info: Optional[dict] = None,
    ) -> float:
        """Translate step to Z3, execute with correction loop, return reward.

        Returns:
            1.0 if entailed, 0.0 otherwise.
        """
        try:
            verify_t0 = time.perf_counter()
            # Step 1: Translate to Z3 code
            translation_t0 = time.perf_counter()
            if self.config.translation == TranslationMode.IMPLICATION:
                z3_code = _translate_implication(
                    processed_context, declarations, step_text,
                    api_config=self.config.api_config,
                    debug_info=debug_info,
                )
            else:
                z3_code = _translate_assertion(
                    processed_context, declarations, step_text,
                    api_config=self.config.api_config,
                    debug_info=debug_info,
                )
            translation_s = time.perf_counter() - translation_t0

            # Step 2: Execute with auto-correction loop
            expression_correction_attempts = 0
            if debug_info is not None:
                expression_correction_attempts = int(debug_info.get("expression_correction_attempts", 0) or 0)
            old_style_correction_tries = max(0, int(self.config.old_max_tries))
            result = correct_loop(
                z3_code,
                api_config=self.config.api_config,
                max_tries=old_style_correction_tries,
                timeout=self.config.timeout,
                debug_info=debug_info,
            )
            if debug_info is not None:
                old_correction_attempts = int(debug_info.get("correction_attempts", 0) or 0)
                debug_info["old_correction_attempts"] = old_correction_attempts
                debug_info["correction_attempts"] = expression_correction_attempts + old_correction_attempts
                debug_info["translation_s"] = translation_s
                debug_info["verify_step_s"] = time.perf_counter() - verify_t0
                debug_info["z3_output"] = result.get("output")
                debug_info["z3_error"] = result.get("error")

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
