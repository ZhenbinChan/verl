"""
Unified FOL-based step reward.

Uses a configurable engine supporting two preprocessing pipelines
(direct / structured) and two translation modes (implication / assertion).
Verification semantics is always entailment: UNSAT -> 1.0.

Configurable via api_config keys:
  - fol_preprocess: "direct" (default) | "structured"
  - fol_translation: "implication" (default) | "assertion"
  - max_tries: int (default 3)
  - timeout: float (default 30.0)
  - cumulative: bool (default False)

Exports:
  - check_step_format_fol
  - compute_step_reward_format_fol
  - compute_step_reward_fol
"""

import logging
# import threading
# import time

from verl.utils.fol_utils.common import check_step_format_fol, extract_fol_problem
from verl.utils.fol_utils.engine import (
    FOLConfig,
    FOLEngine,
    PreprocessPipeline,
    TranslationMode,
)

logger = logging.getLogger(__name__)


def compute_step_reward_format_fol(
    step_text: str, prompt_text: str, step_history: list[str], **kwargs,
) -> float:
    """Format-check process reward ensuring strict step/premise/conclusion tags."""
    return 1.0 if check_step_format_fol(step_text) else 0.0


def compute_step_reward_fol(
    step_text: str,
    prompt_text: str,
    step_history: list[str],
    *,
    api_config: dict | None = None,
    extra_info: dict | None = None,
) -> float:
    """Unified FOL entailment process reward.

    Configurable via api_config:
      fol_preprocess: "direct" | "structured"
      fol_translation: "implication" | "assertion"
      max_tries, timeout, cumulative
    """
    # _t0 = time.time()
    # _tid = threading.current_thread().name
    # print(f"[FOL][{_tid}] ▶ enter  step={step_text[:60]!r}...", flush=True)

    context, question, options = extract_fol_problem(prompt_text, extra_info)
    if not context or not question:
        # print(f"[FOL][{_tid}] ✗ no context/question, skip", flush=True)
        return 0.0

    cfg = api_config or {}

    try:
        preprocess = PreprocessPipeline(cfg.get("fol_preprocess", "direct"))
    except ValueError:
        preprocess = PreprocessPipeline.DIRECT

    try:
        translation = TranslationMode(cfg.get("fol_translation", "implication"))
    except ValueError:
        translation = TranslationMode.IMPLICATION

    fol_config = FOLConfig(
        preprocess=preprocess,
        translation=translation,
        max_tries=int(cfg.get("max_tries", 3)),
        timeout=float(cfg.get("timeout", 30.0)),
        cumulative=bool(cfg.get("cumulative", False)),
        api_config=cfg,
    )

    engine = FOLEngine(fol_config)

    try:
        # Handle cumulative mode
        if fol_config.cumulative and step_history:
            step_to_translate = "\n".join(step_history)
        else:
            step_to_translate = step_text

        # print(f"[FOL][{_tid}] → preprocess({fol_config.preprocess.value})...", flush=True)
        # _t1 = time.time()
        processed_ctx, declarations = engine.preprocess(
            context, question, options or ""
        )
        # print(f"[FOL][{_tid}] ← preprocess done  {time.time()-_t1:.2f}s  decl_len={len(declarations)}", flush=True)

        # print(f"[FOL][{_tid}] → verify_step({fol_config.translation.value})...", flush=True)
        # _t2 = time.time()
        reward = engine.verify_step(processed_ctx, declarations, step_to_translate)
        # print(f"[FOL][{_tid}] ◀ done  reward={reward}  verify={time.time()-_t2:.2f}s  total={time.time()-_t0:.2f}s", flush=True)
        return float(reward)
    except Exception as e:
        # print(f"[FOL][{_tid}] ✗ EXCEPTION after {time.time()-_t0:.2f}s: {e}", flush=True)
        logger.warning("FOL reward computation failed: %s", e)
        return 0.0
