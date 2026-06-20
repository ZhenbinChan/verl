#!/usr/bin/env python3
"""Rerun DeepSeek-V4-Pro on LogiQA with max_tokens=16384 (same as Qwen)."""
import json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGiQA_PATH = REPO_ROOT / "mcts_utils" / "data" / "logiqa.jsonl"
PROMPT_PATH = REPO_ROOT / "prompts" / "premise_conclusion_v2.txt"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "logiqa_deepseek-pro_16384.jsonl"

BASE_URL = "https://genaiapi.shanghaitech.edu.cn/api/v1/start"
API_KEY = "f7f4b08f3abf4632afdb26baacbfb76e"
MODEL = "deepseek-pro"
MAX_TOKENS = 16384
CONCURRENCY = 8
MAX_RETRIES = 3
REQUEST_TIMEOUT = 300

STEP_RE = re.compile(r"<step>.*?</step>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{[^}]*?([A-Ea-e])[^}]*?\}")


def count_steps(text):
    return len(STEP_RE.findall(text))


def extract_answer(text):
    matches = BOXED_RE.findall(text)
    return matches[-1].strip().upper() if matches else None


def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_system_prompt(path):
    return path.read_text(encoding="utf-8")


def format_user_message(sample):
    return (
        f"<Context>\n{sample['context'].strip()}\n</Context>\n\n"
        f"<Question>\n{sample['query'].strip()}\n</Question>\n\n"
        f"<Options>\n{sample['options'].strip()}\n</Options>"
    )


def call_api(client, system_prompt, user_message):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=MAX_TOKENS,
                stream=False,
                timeout=REQUEST_TIMEOUT,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )
            msg = response.choices[0].message
            content = msg.content
            if not content:
                reasoning = getattr(msg, "reasoning", None)
                if reasoning:
                    content = reasoning
            return content or ""
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [attempt {attempt}/{MAX_RETRIES}] error: {e} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("All retries exhausted")


def process_one(idx, sample, system_prompt, client):
    user_msg = format_user_message(sample)
    try:
        response_text = call_api(client, system_prompt, user_msg)
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


def print_stats(results, model_name):
    total = len(results)
    correct = [r for r in results if r["is_correct"]]
    nc = len(correct)
    acc = nc / total * 100 if total > 0 else 0
    avg_len = sum(r["response_length"] for r in results) / total if total > 0 else 0
    avg_steps = sum(r["num_steps"] for r in results) / total if total > 0 else 0
    avg_len_c = sum(r["response_length"] for r in correct) / nc if nc > 0 else 0
    avg_steps_c = sum(r["num_steps"] for r in correct) / nc if nc > 0 else 0
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Total:              {total}")
    print(f"  Correct:            {nc} ({acc:.2f}%)")
    print(f"  Avg response len:   {avg_len:.1f} chars")
    print(f"  Avg steps:          {avg_steps:.2f}")
    print(f"  --- correct only ---")
    print(f"  Avg response len:   {avg_len_c:.1f} chars")
    print(f"  Avg steps:          {avg_steps_c:.2f}")
    print(f"{'='*60}\n")


def main():
    print("Loading dataset ...")
    samples = load_dataset(LOGiQA_PATH)
    print(f"  Loaded {len(samples)} samples")

    print("Loading prompt ...")
    system_prompt = load_system_prompt(PROMPT_PATH)
    print(f"  Prompt: {len(system_prompt)} chars")

    print(f"Settings: max_tokens={MAX_TOKENS}, timeout={REQUEST_TIMEOUT}s")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    results = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_map = {}
        for i, sample in enumerate(samples):
            future = executor.submit(process_one, i, sample, system_prompt, client)
            future_map[future] = i

        completed = 0
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 10 == 0 or completed == len(samples):
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  [{MODEL}] {completed}/{len(samples)} ({rate:.1f}/s)")

    results.sort(key=lambda r: r["id"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved to {OUTPUT_PATH}")
    print_stats(results, MODEL)

    # Also re-print the Qwen stats from the existing file for side-by-side comparison
    qwen_path = OUTPUT_PATH.parent / "logiqa_qwen-instruct.jsonl"
    if qwen_path.exists():
        with open(qwen_path, encoding="utf-8") as f:
            qwen_results = [json.loads(line) for line in f if line.strip()]
        print("\n### Qwen (existing) for comparison ###")
        print_stats(qwen_results, "qwen-instruct")

    print("Done.")


if __name__ == "__main__":
    main()
