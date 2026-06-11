#!/usr/bin/env python3
"""
Run inference on the LogiQA test set using two models (DeepSeek-V4-Pro and
Qwen-3.5) via the ShanghaiTech API gateway, using the premise_conclusion_v2
prompt template.  Saves all responses to JSONL and prints summary statistics
(response length, step count, accuracy, and the same metrics restricted to
correctly-answered samples).
"""

import json
import re
import time
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
LOGiQA_PATH = REPO_ROOT / "mcts_utils" / "data" / "logiqa.jsonl"
PROMPT_PATH = REPO_ROOT / "prompts" / "premise_conclusion_v2.txt"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# ---------------------------------------------------------------------------
# API configs (ShanghaiTech gateway)
# ---------------------------------------------------------------------------
BASE_URL = "https://genaiapi.shanghaitech.edu.cn/api/v1/start"

MODELS = {
    "deepseek-pro": {
        "api_key": "f7f4b08f3abf4632afdb26baacbfb76e",
        "max_tokens": 4096,
        "extra_kwargs": {
            "reasoning_effort": "high",
            "extra_body": {"thinking": {"type": "enabled"}},
        },
    },
    "qwen-instruct": {
        "api_key": "a7db49a1c59a44b2b255c8a0fb83dda4",
        "max_tokens": 16384,
        "extra_kwargs": {},
    },
}

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
CONCURRENCY = 8
MAX_RETRIES = 3
REQUEST_TIMEOUT = 180  # seconds
MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Regex helpers (aligned with the existing codebase)
# ---------------------------------------------------------------------------
STEP_RE = re.compile(r"<step>.*?</step>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{[^}]*?([A-Ea-e])[^}]*?\}")


def count_steps(text: str) -> int:
    """Count <step>...</step> blocks in the response."""
    return len(STEP_RE.findall(text))


def extract_answer(text: str) -> Optional[str]:
    """Extract the answer letter from \\boxed{...}.  Returns the last match."""
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip().upper()


def load_dataset(path: Path) -> list[dict]:
    """Load LogiQA JSONL, one JSON object per line."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_system_prompt(path: Path) -> str:
    """Read the prompt template from a text file."""
    return path.read_text(encoding="utf-8")


def format_user_message(sample: dict) -> str:
    """Build the user message with XML-tagged context / question / options."""
    ctx = sample["context"].strip()
    query = sample["query"].strip()
    options = sample["options"].strip()
    return (
        f"<Context>\n{ctx}\n</Context>\n\n"
        f"<Question>\n{query}\n</Question>\n\n"
        f"<Options>\n{options}\n</Options>"
    )


def call_api(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_message: str,
    extra_kwargs: dict,
    max_tokens: int,
) -> str:
    """Single API call with retries.  Returns the response text."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                stream=False,
                timeout=REQUEST_TIMEOUT,
                **extra_kwargs,
            )
            msg = response.choices[0].message
            # Some thinking models return content=None when thinking consumes all
            # tokens.  Fall back to the reasoning field if content is empty.
            content = msg.content
            if not content:
                reasoning = getattr(msg, "reasoning", None)
                if reasoning:
                    content = reasoning
            return content or ""
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            print(f"  [attempt {attempt}/{MAX_RETRIES}] error: {e} — retrying in {wait}s")
            time.sleep(wait)
    raise last_error  # type: ignore


def process_one(
    idx: int,
    sample: dict,
    system_prompt: str,
    model_name: str,
    client: OpenAI,
    extra_kwargs: dict,
    max_tokens: int,
) -> dict:
    """Process a single sample: call the API and return a result record."""
    user_msg = format_user_message(sample)
    try:
        response_text = call_api(client, model_name, system_prompt, user_msg, extra_kwargs, max_tokens)
    except Exception as e:
        response_text = f"__ERROR__: {e}"

    num_steps = count_steps(response_text)
    resp_len = len(response_text)
    extracted = extract_answer(response_text)
    gt = sample["answer"].strip().upper()
    is_correct = (extracted == gt) if extracted is not None else False

    return {
        "id": sample["id"],
        "ground_truth": gt,
        "extracted_answer": extracted,
        "is_correct": is_correct,
        "num_steps": num_steps,
        "response_length": resp_len,
        "response_text": response_text,
    }


def run_inference(
    samples: list[dict],
    system_prompt: str,
    model_name: str,
    api_key: str,
    extra_kwargs: dict,
    max_tokens: int,
    output_path: Path,
) -> list[dict]:
    """Run inference on all samples with the given model, save to JSONL."""
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    results = []

    t_start = time.time()
    completed = 0
    total = len(samples)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_map = {}
        for i, sample in enumerate(samples):
            future = executor.submit(
                process_one,
                i,
                sample,
                system_prompt,
                model_name,
                client,
                extra_kwargs,
                max_tokens,
            )
            future_map[future] = i

        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  [{model_name}] {completed}/{total} ({rate:.1f}/s)")

    # Sort by id so the output is deterministic
    results.sort(key=lambda r: r["id"])

    # Save JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Saved to {output_path}")
    return results


def print_stats(results: list[dict], model_name: str):
    """Print summary statistics for a model's results."""
    total = len(results)
    correct = [r for r in results if r["is_correct"]]
    num_correct = len(correct)
    acc = num_correct / total * 100 if total > 0 else 0

    avg_len = sum(r["response_length"] for r in results) / total if total > 0 else 0
    avg_steps = sum(r["num_steps"] for r in results) / total if total > 0 else 0

    avg_len_correct = (
        sum(r["response_length"] for r in correct) / num_correct
        if num_correct > 0
        else 0
    )
    avg_steps_correct = (
        sum(r["num_steps"] for r in correct) / num_correct
        if num_correct > 0
        else 0
    )

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Total samples:      {total}")
    print(f"  Correct:            {num_correct} ({acc:.2f}%)")
    print(f"  Avg response len:   {avg_len:.1f} chars")
    print(f"  Avg steps:          {avg_steps:.2f}")
    print(f"  --- correct only ---")
    print(f"  Avg response len:   {avg_len_correct:.1f} chars")
    print(f"  Avg steps:          {avg_steps_correct:.2f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading dataset ...")
    samples = load_dataset(LOGiQA_PATH)
    print(f"  Loaded {len(samples)} samples from {LOGiQA_PATH}")

    print("Loading prompt template ...")
    system_prompt = load_system_prompt(PROMPT_PATH)
    print(f"  Prompt loaded ({len(system_prompt)} chars) from {PROMPT_PATH}")

    for model_name, cfg in MODELS.items():
        print(f"\n{'#'*60}")
        print(f"# Running inference with model: {model_name}")
        print(f"{'#'*60}")

        output_path = OUTPUT_DIR / f"logiqa_{model_name}.jsonl"
        results = run_inference(
            samples=samples,
            system_prompt=system_prompt,
            model_name=model_name,
            api_key=cfg["api_key"],
            extra_kwargs=cfg["extra_kwargs"],
            max_tokens=cfg["max_tokens"],
            output_path=output_path,
        )
        print_stats(results, model_name)

    print("All done.")


if __name__ == "__main__":
    main()
