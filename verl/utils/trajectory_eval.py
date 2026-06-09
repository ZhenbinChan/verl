"""Trajectory quality evaluation via external LLM RM.

Evaluates reasoning trajectories for logical correctness, hallucination,
and internal coherence by prompting an external LLM (vLLM server).

Usage::

    from verl.utils.trajectory_eval import evaluate_trajectories

    scores = evaluate_trajectories(
        questions=["...", "..."],
        ground_truths=["D", "A"],
        trajectories=["<step>...</step>", "..."],
        rm_url="http://localhost:4869/v1",
        model_name="eval-model",
    )
    # scores: [1.0, 0.0, 1.0, ...]
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# Default prompt template path
_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "trajectory_eval.txt"


def load_trajectory_eval_template(prompt_path: Optional[str] = None) -> str:
    """Load the trajectory evaluation prompt template from disk."""
    path = Path(prompt_path) if prompt_path else _DEFAULT_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Trajectory eval prompt not found: {path}")
    return path.read_text(encoding="utf-8")


_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert evaluator for logical reasoning traces. "
    "Your task is to judge whether a reasoning trajectory is logically correct, "
    "free of hallucinations, and internally coherent.\n\n"
    "Evaluation criteria:\n"
    "1. Logical Correctness: each conclusion must logically follow from the premises.\n"
    "2. No Hallucination: no fabricated facts not present in the given input.\n"
    "3. Internal Coherence: the chain must be consistent, no contradictions.\n\n"
    "Output ONLY a single digit 1 or 0. Do NOT output any explanation or extra text."
)


def _build_eval_user_message(
    question: str,
    ground_truth: Optional[str],
    trajectory: str,
) -> str:
    """Build the user message with the problem, answer, and trajectory to evaluate."""
    gt_str = str(ground_truth) if ground_truth is not None else "N/A"
    return (
        f"Problem:\n{question}\n\n"
        f"Ground Truth Answer: {gt_str}\n\n"
        f"Reasoning Trajectory:\n{trajectory}\n\n"
        f"Score (1 or 0):"
    )


def _call_llm(
    prompt: str,
    rm_url: str,
    model_name: str,
    max_tokens: int = 32,
    temperature: float = 0.0,
    timeout: int = 60,
    max_retries: int = 2,
    system_prompt: str = "",
) -> float:
    """Send a single prompt to the LLM RM and parse the 0/1 response."""
    url = f"{rm_url.rstrip('/')}/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                # Try first: the very first non-whitespace character is 0 or 1
                first_char = content.lstrip()
                if first_char and first_char[0] in ("0", "1"):
                    return float(first_char[0])
                # Try second: regex search for 0 or 1 anywhere in first 50 chars
                match = re.search(r"\b([01])\b", content[:50])
                if match:
                    return float(match.group(1))
                logger.warning(
                    "Could not parse 0/1 from LLM RM response: %r. Defaulting to 0.0.",
                    content,
                )
                return 0.0
            else:
                logger.warning(
                    "LLM RM returned status %d (attempt %d): %s",
                    resp.status_code, attempt + 1, resp.text[:200],
                )
        except Exception:
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
            else:
                logger.error("LLM RM request failed after %d attempts.", max_retries + 1)

    return 0.0


def evaluate_trajectories(
    questions: List[str],
    ground_truths: List[Optional[str]],
    trajectories: List[str],
    rm_url: str,
    model_name: str,
    max_tokens: int = 32,
    temperature: float = 0.0,
    max_workers: int = 8,
) -> List[float]:
    """Batch-evaluate reasoning trajectories via an external LLM.

    Args:
        questions: Problem descriptions (includes Context + Question + Options).
        ground_truths: Correct answer labels (may be None for some items).
        trajectories: Full reasoning traces for each leaf to evaluate.
        rm_url: Base URL of the vLLM / OpenAI-compatible server.
        model_name: Model name registered on the server.
        max_tokens: Max tokens for the LLM response (should be small, default 32).
        temperature: Sampling temperature (zero for deterministic).
        max_workers: Number of concurrent HTTP workers.

    Returns:
        A list of float scores (0.0 or 1.0), one per trajectory.
    """
    if not trajectories:
        return []

    prompts = [
        _build_eval_user_message(q, gt, traj)
        for q, gt, traj in zip(questions, ground_truths, trajectories)
    ]

    scores: List[float] = [0.0] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _call_llm,
                prompt,
                rm_url,
                model_name,
                max_tokens,
                temperature,
                60,     # timeout
                2,      # max_retries
                _DEFAULT_SYSTEM_PROMPT,
            ): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                scores[idx] = future.result()
            except Exception:
                scores[idx] = 0.0

    return scores
