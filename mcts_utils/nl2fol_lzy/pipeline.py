from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    from .utils import extract_final_answer, extract_python_code, parse_python_logic_steps, parse_reasoning_steps
except ImportError:
    from utils import extract_final_answer, extract_python_code, parse_python_logic_steps, parse_reasoning_steps


GenerateFn = Callable[[str, str], str]

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "mcts_utils" / "prompts"


@dataclass
class FOLPrefixResult:
    declaration_code: str
    implication_code: str
    parsed_steps: list[dict]
    step_outputs: list[Optional[list[str]]] = field(default_factory=list)
    step_errors: list[str] = field(default_factory=list)

    @property
    def final_step_output(self) -> Optional[list[str]]:
        return self.step_outputs[-1] if self.step_outputs else None

    @property
    def final_step_correct(self) -> bool:
        output = self.final_step_output or []
        return any("SUCCESS_ENTAILED" in line for line in output)


def load_prompt(name: str, prompt_dir: Path = PROMPT_DIR) -> str:
    return (prompt_dir / name).read_text(encoding="utf-8")


def build_openai_generate_fn(
    *,
    model: str = "gpt-4o-mini-2024-07-18",
    temperature: float = 0.2,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> GenerateFn:
    from openai import OpenAI

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    client = OpenAI(**kwargs)

    def generate(user_input: str, system_prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            n=1,
            temperature=temperature,
        )
        return response.choices[0].message.content

    return generate


def generate_reasoning(question_nl: str, generate_fn: GenerateFn) -> str:
    return generate_fn(question_nl, load_prompt("Generation1.txt"))


def generate_declaration(question_nl: str, generate_fn: GenerateFn) -> str:
    response = generate_fn(question_nl, load_prompt("Z3DeclarationsGeneration1.txt"))
    return extract_python_code(response)


def generate_implication(question_nl: str, declaration_code: str, reasoning_steps: str, generate_fn: GenerateFn) -> str:
    full_input = f"Question:\n{question_nl}\n\nZ3 Declaration:\n{declaration_code}\n\nReasoning steps:\n{reasoning_steps}"
    response = generate_fn(full_input, load_prompt("Z3ImplicationConversion1.txt"))
    return extract_python_code(response)


def verify_fol_step(declaration_code: str, parsed_step: dict, timeout: float = 1.0) -> tuple[Optional[list[str]], str]:
    try:
        try:
            from .utils import execute_program
        except ImportError:
            from utils import execute_program

        premises = parsed_step.get("premises") or []
        conclusion = (parsed_step.get("conclusion") or [""])[0]
        premises_str = ", ".join(premises) if premises else "True"
        z3_code = f"""
from z3 import *
{declaration_code}
solver = Solver()
solver.add(And({premises_str}))
solver.add(Not({conclusion}))

check_res = solver.check()
if check_res == unsat:
    print("SUCCESS_ENTAILED")
elif check_res == sat:
    print("FAILED_CONTRADICT")
else:
    print("UNKNOWN")
""".strip()
        return execute_program(z3_code, timeout=timeout)
    except Exception as exc:
        return None, repr(exc)


def score_reasoning_prefix(
    question_nl: str,
    reasoning_steps: str,
    generate_fn: GenerateFn,
    *,
    declaration_code: Optional[str] = None,
    timeout: float = 1.0,
) -> FOLPrefixResult:
    declaration = declaration_code or generate_declaration(question_nl, generate_fn)
    implication = generate_implication(question_nl, declaration, reasoning_steps, generate_fn)
    parsed_steps = parse_python_logic_steps(implication)

    step_outputs: list[Optional[list[str]]] = []
    step_errors: list[str] = []
    for step in parsed_steps:
        output, error = verify_fol_step(declaration, step, timeout=timeout)
        step_outputs.append(output)
        step_errors.append(error)

    return FOLPrefixResult(
        declaration_code=declaration,
        implication_code=implication,
        parsed_steps=parsed_steps,
        step_outputs=step_outputs,
        step_errors=step_errors,
    )


def run_demo() -> None:
    question_nl = """
<Context>Zhao Ming, Qian Hong and Sun Jie were admitted to Peking University, Tsinghua University and Beijing Normal University. Which schools were they admitted to? The students made the following guesses? Classmate A guessed? Zhao Ming was admitted to Tsinghua University and Sun Jie was admitted to Beijing Normal University. Student B guess? Zhao Ming was admitted to Beijing Normal University, Qian Hong was admitted to Tsinghua University. Student C guess? Zhao Ming was admitted to Peking University, Sun Jie was admitted to Tsinghua University. As a result, the students' guesses were half correct.</Context><Question>Well, their admission status is.</Question><Options>Option (A):Zhao Ming, Qian Hong and Sun Jie were accepted by Peking University, Tsinghua University and Beijing Normal University respectively.

Option (B):Zhao Ming, Qian Hong and Sun Jie were admitted to Tsinghua University, Beijing Normal University and Peking University respectively.

Option (C):Zhao Ming, Qian Hong and Sun Jie were admitted to Beijing Normal University, Tsinghua University and Peking University respectively.

Option (D):Zhao Ming, Qian Hong and Sun Jie were accepted by Peking University, Beijing Normal University and Tsinghua University respectively.</Options>
""".strip()
    gold_answer = "A"
    generate_fn = build_openai_generate_fn(
        model=os.getenv("FOL_DEMO_MODEL", "gpt-4o-mini-2024-07-18"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    answer_nl = generate_reasoning(question_nl, generate_fn)
    print("=" * 60)
    print(answer_nl)

    result = score_reasoning_prefix(question_nl, answer_nl, generate_fn)
    parsed_chain = parse_reasoning_steps(answer_nl)
    extracted_answer = extract_final_answer(answer_nl)

    for idx, (nl, fol, result_fol) in enumerate(zip(parsed_chain, result.parsed_steps, result.step_outputs)):
        print(f"Step {idx + 1}:")
        print("Premises (NL):")
        for premise in nl["premises"]:
            print(f"- {premise}")
        print(f"Conclusion (NL): {nl['conclusion'][0]}")
        print("Premises (FOL):")
        for premise in fol["premises"]:
            print(f"- {premise}")
        print(f"Conclusion (FOL): {fol['conclusion'][0]}")
        print(f"Verification Result (FOL): {result_fol}")
        print("-" * 40)

    print(f"Extracted Answer: {extracted_answer}")
    print(f"Gold Answer: {gold_answer}")
    print(f"Answer Correct: {extracted_answer == gold_answer}")


if __name__ == "__main__":
    run_demo()
