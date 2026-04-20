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


def _build_fol_config(api_config: dict | None = None) -> FOLConfig:
    """Construct a FOLConfig from reward API config."""
    cfg = api_config or {}

    try:
        preprocess = PreprocessPipeline(cfg.get("fol_preprocess", "direct"))
    except ValueError:
        preprocess = PreprocessPipeline.DIRECT

    try:
        translation = TranslationMode(cfg.get("fol_translation", "implication"))
    except ValueError:
        translation = TranslationMode.IMPLICATION

    return FOLConfig(
        preprocess=preprocess,
        translation=translation,
        max_tries=int(cfg.get("max_tries", 1)),
        timeout=float(cfg.get("timeout", 30.0)),
        cumulative=bool(cfg.get("cumulative", False)),
        api_config=cfg,
    )


def prepare_fol_shared_state(
    prompt_text: str,
    *,
    api_config: dict | None = None,
    extra_info: dict | None = None,
) -> dict | None:
    """Precompute response-level FOL state reusable across all steps."""
    context, question, options = extract_fol_problem(prompt_text, extra_info)
    if not context or not question:
        return None

    fol_config = _build_fol_config(api_config)
    engine = FOLEngine(fol_config)
    processed_ctx, declarations = engine.preprocess(context, question, options or "")
    return {
        "config": fol_config,
        "processed_context": processed_ctx,
        "declarations": declarations,
    }


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
    fol_shared_state: dict | None = None,
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

    try:
        shared_state = fol_shared_state or prepare_fol_shared_state(
            prompt_text, api_config=api_config, extra_info=extra_info
        )
        if shared_state is None:
            return 0.0

        fol_config = shared_state["config"]
        engine = FOLEngine(fol_config)

        # Handle cumulative mode
        if fol_config.cumulative and step_history:
            step_to_translate = "\n".join(step_history)
        else:
            step_to_translate = step_text

        # print(f"[FOL][{_tid}] → verify_step({fol_config.translation.value})...", flush=True)
        # _t2 = time.time()
        reward = engine.verify_step(
            shared_state["processed_context"],
            shared_state["declarations"],
            step_to_translate,
        )
        # print(f"[FOL][{_tid}] ◀ done  reward={reward}  verify={time.time()-_t2:.2f}s  total={time.time()-_t0:.2f}s", flush=True)
        return float(reward)
    except Exception as e:
        # print(f"[FOL][{_tid}] ✗ EXCEPTION after {time.time()-_t0:.2f}s: {e}", flush=True)
        logger.warning("FOL reward computation failed: %s", e)
        return 0.0
