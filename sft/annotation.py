#!/usr/bin/env python3
"""annotation.py

使用 DeepSeek-V4-Pro 为选中的 200 道题目各生成 8 条 step-reasoning response，
筛选出答案正确且格式符合要求的样本，保存为 JSONL。

支持断点续传：每条 response 生成后实时写入 checkpoint，中断后重启自动跳过已完成部分。

用法：
    cd /2024133105/Workspaces/verl
    python3 sft/annotation.py              # 正常/续传
    python3 sft/annotation.py --no-resume  # 从头重跑，忽略 checkpoint

参考：llm_server/deepseek.py
"""

import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# 将 verl 仓库加入 sys.path，以便导入 mcts_prm 中的 format 判定函数
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verl.trainer.ppo.sampling.mcts_prm import classify_rollout_format

# ---------------------------------------------------------------------------
# DeepSeek API 配置（与 llm_server/deepseek.py 保持一致）
# ---------------------------------------------------------------------------
BASE_URL = "https://api.deepseek.com"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-4873a3bf077241e2a70e130104f12a91")
MODEL_NAME = "deepseek-v4-pro"

if not API_KEY:
    print("[annotation] ERROR: DEEPSEEK_API_KEY environment variable is not set.")
    print("           export DEEPSEEK_API_KEY=sk-xxx")
    sys.exit(1)

DEFAULT_ARGS = {
    "reasoning_effort": "high",
    "temperature": 0.8,
    "top_p": 1.0,
    "max_tokens": 8192,
}
EXTRA_BODY = {"thinking": {"type": "enabled"}}

# ---------------------------------------------------------------------------
# 文件路径
# ---------------------------------------------------------------------------
SFT_DIR = ROOT / "sft"
SELECTED_JSONL = SFT_DIR / "selected_200.jsonl"
PROMPT_PATH = ROOT / "prompts" / "premise_conclusion.txt"
CHECKPOINT_JSONL = SFT_DIR / "dsv4_checkpoint.jsonl"          # 实时追加的 raw response
CHECKPOINT_PROGRESS = SFT_DIR / "dsv4_checkpoint_progress.json"  # prompt_id → done_count
RAW_OUTPUT_JSONL = SFT_DIR / "dsv4_responses_raw.jsonl"       # 最终合并后的输出
FILTERED_OUTPUT_JSONL = SFT_DIR / "dsv4_responses_filtered.jsonl"  # 筛选+去重后

N_RESPONSES = 8          # 每道题生成 8 条
MAX_RETRIES = 3          # 单次 API 调用最大重试
API_TIMEOUT = 120.0      # 单次 API 调用超时（秒）
CONCURRENCY = 2          # 并发线程数（DeepSeek 限速，不宜过高）
REQUEST_INTERVAL = 0.5   # 同线程两次请求间隔（秒）

# 线程安全锁，保护 checkpoint 写入
import threading
_checkpoint_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Checkpoint 进度读写
# ---------------------------------------------------------------------------
def load_progress() -> dict[str, int]:
    """读取 checkpoint 进度文件，返回 {prompt_id: completed_count}。"""
    if not CHECKPOINT_PROGRESS.exists():
        return {}
    try:
        with open(CHECKPOINT_PROGRESS, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: int(v) for k, v in data.items()}
    except Exception as e:
        print(f"[annotation] WARNING: failed to load progress file: {e}")
    return {}


def save_progress(progress: dict[str, int]) -> None:
    """原子写入进度文件。"""
    with _checkpoint_lock:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(SFT_DIR), suffix=".json.tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False)
            os.replace(tmp_path, str(CHECKPOINT_PROGRESS))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def append_to_checkpoint(record: dict) -> None:
    """线程安全地将一条 record 追加写入 checkpoint JSONL。"""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _checkpoint_lock:
        with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
            f.write(line)


def increment_progress(prompt_id: str) -> None:
    """将指定 prompt_id 的完成数 +1，并保存。"""
    progress = load_progress()
    progress[prompt_id] = progress.get(prompt_id, 0) + 1
    save_progress(progress)


def clear_checkpoints() -> None:
    """删除 checkpoint 文件（--no-resume 时使用）。"""
    for p in [CHECKPOINT_JSONL, CHECKPOINT_PROGRESS]:
        if p.exists():
            p.unlink()
            print(f"[annotation] removed {p.name}")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def load_prompt_template() -> str:
    """加载 prompt 模板。"""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_message(prompt_template: str, raw_prompt: str) -> list[dict]:
    """构造给 DeepSeek 的 messages。"""
    user_content = f"{prompt_template}\n\n{raw_prompt}"
    return [{"role": "user", "content": user_content}]


def extract_boxed_answer(response_text: str) -> str | None:
    """从 response 中提取 \\boxed{...} 中的答案字母（大写）。"""
    m = re.search(r"\\boxed\{\s*(?:\\boxed\{\s*)?\s*([A-Za-z])\s*(?:\}\s*)?\}", response_text)
    if m:
        return m.group(1).upper()
    return None


def is_answer_correct(response_text: str, ground_truth: str) -> bool:
    """检查 response 中的 \\boxed 答案是否与 ground_truth 一致。"""
    extracted = extract_boxed_answer(response_text)
    if extracted is None:
        return False
    return extracted == ground_truth.upper()


def is_format_valid(response_text: str) -> bool:
    """使用 StepRL 的 classify_rollout_format 严格判定格式是否为 'full'。"""
    info = classify_rollout_format(response_text, valid_choices="ABCDEF")
    return info.get("format_primary") == "full"


def normalize_response_text(text: str) -> str:
    """标准化 response 文本用于比较去重。"""
    text = text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text


# ---------------------------------------------------------------------------
# API 调用
# ---------------------------------------------------------------------------
def call_deepseek(messages: list[dict], client: OpenAI) -> str | None:
    """调用 DeepSeek API，返回 response 文本；失败返回 None。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False,
                timeout=API_TIMEOUT,
                **DEFAULT_ARGS,
                extra_body=EXTRA_BODY,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  [API] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    return None


def generate_responses_for_one(
    prompt_id: str,
    raw_prompt: str,
    ground_truth: str,
    prompt_template: str,
    start_from: int = 0,
) -> int:
    """对一道题完成剩余 response，返回本次新生成的条数。"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = build_message(prompt_template, raw_prompt)
    question_text = raw_prompt

    new_count = 0
    for i in range(start_from, N_RESPONSES):
        response_text = call_deepseek(messages, client)
        if response_text is None:
            print(f"  [{prompt_id}] response {i+1}/{N_RESPONSES}: FAILED (all retries exhausted)")
            continue
        record = {
            "prompt_id": prompt_id,
            "question": question_text,
            "response": response_text,
            "ground_truth": ground_truth,
        }
        # 实时持久化
        append_to_checkpoint(record)
        increment_progress(prompt_id)
        new_count += 1

        correct = is_answer_correct(response_text, ground_truth)
        fmt = classify_rollout_format(response_text, valid_choices="ABCDEF")["format_primary"]
        print(f"  [{prompt_id}] response {i+1}/{N_RESPONSES}: correct={correct}, format={fmt}")
        # 请求间隔（避免触发 DeepSeek 限速）
        time.sleep(REQUEST_INTERVAL)
    return new_count


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    no_resume = "--no-resume" in sys.argv or "--force" in sys.argv

    if no_resume:
        clear_checkpoints()
        print("[annotation] --no-resume: starting from scratch")

    # 1. 加载进度
    progress = load_progress()
    done_ids = {pid for pid, count in progress.items() if count >= N_RESPONSES}
    partial_ids = {pid: count for pid, count in progress.items() if 0 < count < N_RESPONSES}

    print(f"[annotation] checkpoint: {len(done_ids)} fully done, {len(partial_ids)} partially done")

    # 2. 加载全部题目
    with open(SELECTED_JSONL, "r", encoding="utf-8") as f:
        problems = [json.loads(line) for line in f]
    print(f"[annotation] loaded {len(problems)} problems from {SELECTED_JSONL}")

    prompt_template = load_prompt_template()
    print(f"[annotation] prompt template loaded ({len(prompt_template)} chars)")

    # 3. 筛选需要跑的任务
    tasks: list[dict] = []
    skipped = 0
    for prob in problems:
        pid = prob["sample_id"]
        if pid in done_ids:
            skipped += 1
            continue
        start = partial_ids.get(pid, 0)
        tasks.append({
            "prompt_id": pid,
            "raw_prompt": prob["raw_prompt"],
            "ground_truth": str(prob["answer"]),
            "start_from": start,
        })

    if skipped > 0:
        print(f"[annotation] skipping {skipped} already-completed problems")
    if not tasks:
        print("[annotation] all problems already completed, proceeding to post-processing")

    # 4. 并发生成
    if tasks:
        print(f"[annotation] generating {len(tasks)} remaining problems, concurrency={CONCURRENCY} ...")
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {
                executor.submit(
                    generate_responses_for_one,
                    prompt_id=t["prompt_id"],
                    raw_prompt=t["raw_prompt"],
                    ground_truth=t["ground_truth"],
                    prompt_template=prompt_template,
                    start_from=t["start_from"],
                ): t["prompt_id"]
                for t in tasks
            }

            for fut in as_completed(futures):
                prompt_id = futures[fut]
                try:
                    new_count = fut.result()
                    print(f"  [{prompt_id}] done: {new_count} new responses")
                except Exception as e:
                    print(f"  [{prompt_id}] ERROR: {e}")

    # 5. 加载 checkpoint 中所有 record 作为 raw output
    all_responses: list[dict] = []
    if CHECKPOINT_JSONL.exists():
        with open(CHECKPOINT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_responses.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    print(f"[annotation] total raw responses from checkpoint: {len(all_responses)}")

    # 6. 保存最终 raw output（合并后）
    with open(RAW_OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in all_responses:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[annotation] raw responses saved: {len(all_responses)} → {RAW_OUTPUT_JSONL}")

    # 7. 筛选：答案正确 + 格式 = full
    filtered: list[dict] = []
    correct_count = 0
    format_count = 0
    for rec in all_responses:
        resp = rec["response"]
        gt = rec["ground_truth"]
        ok_answer = is_answer_correct(resp, gt)
        ok_format = is_format_valid(resp)
        if ok_answer:
            correct_count += 1
        if ok_format:
            format_count += 1
        if ok_answer and ok_format:
            filtered.append(rec)

    print(f"[annotation] filtering stats:")
    print(f"    total responses:       {len(all_responses)}")
    print(f"    answer-correct:        {correct_count} ({correct_count/max(len(all_responses),1)*100:.1f}%)")
    print(f"    format-valid (full):   {format_count} ({format_count/max(len(all_responses),1)*100:.1f}%)")
    print(f"    both OK (kept):        {len(filtered)} ({len(filtered)/max(len(all_responses),1)*100:.1f}%)")

    # 8. 去重
    deduped: list[dict] = []
    duplicate_count = 0
    seen_hashes: dict[str, set[str]] = {}

    for rec in filtered:
        pid = rec["prompt_id"]
        normalized = normalize_response_text(rec["response"])
        text_hash = hash(normalized)
        if pid not in seen_hashes:
            seen_hashes[pid] = set()
        if text_hash in seen_hashes[pid]:
            duplicate_count += 1
            continue
        seen_hashes[pid].add(text_hash)
        deduped.append(rec)

    print(f"[annotation] dedup stats:")
    print(f"    before dedup:   {len(filtered)}")
    print(f"    duplicates:     {duplicate_count}")
    print(f"    after dedup:    {len(deduped)}")

    per_problem = {}
    for rec in deduped:
        pid = rec["prompt_id"]
        per_problem[pid] = per_problem.get(pid, 0) + 1
    if per_problem:
        counts = list(per_problem.values())
        print(f"    per-problem:    min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")

    # 9. 保存筛选+去重后结果
    with open(FILTERED_OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in deduped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[annotation] filtered + deduped responses saved: {len(deduped)} → {FILTERED_OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
