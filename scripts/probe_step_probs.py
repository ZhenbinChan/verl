#!/usr/bin/env python3
"""
Probe the model's answer probabilities at each reasoning step.

Generates a logical reasoning question, solves it n times with step-by-step
reasoning, then at each step boundary injects "So far, the most likely answer is"
and reads the next-token probabilities for A/B/C/D/E.
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────
MODEL_PATH = "/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct"
INSTRUCTION_PATH = "/home/chenzhb/Workspaces/verl/prompts/instruction_following_prompt.txt"
N_SOLUTIONS = 5
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────

# Token IDs for " A", " B", " C", " D", " E" (space-prefixed) in Qwen2.5
# After "is" (token 374: " is"), the next token naturally has a leading space.
ANSWER_TOKEN_IDS = [362, 425, 356, 422, 468]  # " A", " B", " C", " D", " E"
ANSWER_LABELS = ["A", "B", "C", "D", "E"]
PROBE_PHRASE = "\n\nSo far, the most likely answer is"


def load_model(model_path: str):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()
    return model, tokenizer


def load_instruction(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def generate_question(model, tokenizer, instruction: str) -> str:
    """Generate a logical reasoning question with 5 answer options."""
    prompt = (
        "Generate a challenging logical reasoning question in English. "
        "The question should have exactly 5 answer choices labeled A through E. "
        "Include the correct answer. Output in the following format:\n\n"
        "<Question>[your question]</Question>\n"
        "<Options>\n(A) [option A]\n(B) [option B]\n(C) [option C]\n(D) [option D]\n(E) [option E]\n</Options>\n"
        "<Answer>[correct letter]</Answer>"
    )
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Generated question:\n{generated}\n")
    return generated


def generate_solutions(model, tokenizer, instruction: str, question: str, n: int) -> list[str]:
    """Generate n independent solutions for the given question."""
    solutions = []
    user_content = instruction + "\n\n" + question
    messages = [{"role": "user", "content": user_content}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    for i in range(n):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        solution = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        solutions.append(solution)
        print(f"Solution {i + 1}/{n} generated ({len(solution)} chars)")

    return solutions


def parse_steps(solution: str) -> list[str]:
    """Extract individual step blocks from a solution string.

    Returns a list of step contents (text between <step> and </step>).
    """
    # Find all <step>...</step> blocks (non-greedy, DOTALL for multiline)
    pattern = r"<step>\s*(.*?)\s*</step>"
    matches = re.findall(pattern, solution, re.DOTALL)
    return [m.strip() for m in matches]


def probe_step_probabilities(
    model, tokenizer, instruction: str, question: str, accumulated_steps: list[str]
) -> dict[str, float]:
    """Run a forward pass with accumulated steps + probe phrase, return A-E probabilities."""
    user_content = instruction + "\n\n" + question
    messages = [{"role": "user", "content": user_content}]
    # Get the formatted chat prefix ending with assistant start
    prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # Build the full text: prefix + accumulated steps + probe phrase
    steps_text = "\n\n".join(
        f"<step>\n{step}\n</step>" for step in accumulated_steps
    )
    if steps_text:
        full_text = prefix + steps_text + PROBE_PHRASE
    else:
        full_text = prefix + PROBE_PHRASE.lstrip("\n")

    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # last token position

    probs = torch.softmax(logits.float(), dim=-1)
    result = {}
    for label, tid in zip(ANSWER_LABELS, ANSWER_TOKEN_IDS):
        result[label] = probs[tid].item()

    return result


def main():
    model, tokenizer = load_model(MODEL_PATH)
    instruction = load_instruction(INSTRUCTION_PATH)

    # Step 1: Generate a question
    question = generate_question(model, tokenizer, instruction)
    print("=" * 60)
    print(f"QUESTION:\n{question}")
    print("=" * 60)

    # Step 2: Generate n solutions
    print(f"\nGenerating {N_SOLUTIONS} solutions...")
    solutions = generate_solutions(model, tokenizer, instruction, question, N_SOLUTIONS)

    # Step 3 & 4: Parse steps and probe probabilities
    for sol_idx, solution in enumerate(solutions):
        steps = parse_steps(solution)
        if not steps:
            print(f"\nSolution {sol_idx + 1}: No steps found, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Solution {sol_idx + 1} ({len(steps)} steps)")
        print(f"{'=' * 60}")

        accumulated: list[str] = []
        for step_idx, step in enumerate(steps):
            accumulated.append(step)
            probs = probe_step_probabilities(model, tokenizer, instruction, question, accumulated)

            # Print step summary and probabilities
            step_preview = step[:80].replace("\n", " ")
            print(f"\n--- Step {step_idx + 1}: {step_preview}...")
            prob_str = "  ".join(
                f"{label}: {probs[label]:.4f}" for label in ANSWER_LABELS
            )
            best = max(probs, key=probs.get)
            print(f"    {prob_str}")
            print(f"    Best: {best} ({probs[best]:.4f})")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
