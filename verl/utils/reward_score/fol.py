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

import atexit
import logging
import os
import threading
from collections import OrderedDict
from hashlib import sha1
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

_FOL_SHARED_STATE_CACHE_MAX_SIZE = max(1, int(os.environ.get("FOL_SHARED_PREPROCESS_CACHE_SIZE", "512")))
_fol_shared_state_cache: OrderedDict[tuple, dict] = OrderedDict()
_fol_shared_state_cache_lock = threading.Lock()
_FOL_VERIFY_CACHE_MAX_SIZE = max(1, int(os.environ.get("FOL_VERIFY_CACHE_SIZE", "4096")))
_fol_verify_cache: OrderedDict[tuple, float] = OrderedDict()
_fol_verify_cache_lock = threading.Lock()
_fol_verify_cache_stats = {"hits": 0, "misses": 0}
_fol_verify_cache_stats_lock = threading.Lock()
_fol_verify_cache_step_stats: OrderedDict[int, dict[str, int]] = OrderedDict()
_fol_verify_cache_summary_registered = False


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
    cache_key = (
        context,
        question,
        options or "",
        fol_config.preprocess.value,
        (fol_config.api_config or {}).get("model"),
        (fol_config.api_config or {}).get("base_url"),
        (fol_config.api_config or {}).get("temperature"),
        (fol_config.api_config or {}).get("max_tokens"),
        (fol_config.api_config or {}).get("top_p"),
        bool((fol_config.api_config or {}).get("fol_judge_use_outlines", False)),
    )

    with _fol_shared_state_cache_lock:
        cached = _fol_shared_state_cache.get(cache_key)
        if cached is not None:
            _fol_shared_state_cache.move_to_end(cache_key)
            return cached

    engine = FOLEngine(fol_config)
    processed_ctx, declarations = engine.preprocess(context, question, options or "")
    shared_state = {
        "config": fol_config,
        "processed_context": processed_ctx,
        "declarations": declarations,
    }
    with _fol_shared_state_cache_lock:
        existing = _fol_shared_state_cache.get(cache_key)
        if existing is not None:
            _fol_shared_state_cache.move_to_end(cache_key)
            return existing
        _fol_shared_state_cache[cache_key] = shared_state
        if len(_fol_shared_state_cache) > _FOL_SHARED_STATE_CACHE_MAX_SIZE:
            _fol_shared_state_cache.popitem(last=False)
    return shared_state


def _digest_text(text: str) -> str:
    """Return a stable digest for large cache-key strings."""
    return sha1(text.encode("utf-8")).hexdigest()


def _build_verify_cache_key(
    *,
    shared_state: dict,
    fol_config: FOLConfig,
    step_to_translate: str,
) -> tuple:
    """Build a strict cache key for one FOL verify_step call."""
    api_cfg = fol_config.api_config or {}
    return (
        _digest_text(shared_state["processed_context"]),
        _digest_text(shared_state["declarations"]),
        _digest_text(step_to_translate),
        fol_config.preprocess.value,
        fol_config.translation.value,
        fol_config.max_tries,
        fol_config.timeout,
        fol_config.cumulative,
        api_cfg.get("model"),
        api_cfg.get("base_url"),
        api_cfg.get("temperature"),
        api_cfg.get("max_tokens"),
        api_cfg.get("top_p"),
        bool(api_cfg.get("fol_judge_use_outlines", False)),
    )


def _verify_cache_log_enabled() -> bool:
    """Whether verify-cache summary logging is enabled."""
    return str(os.environ.get("FOL_VERIFY_CACHE_LOG", "0")).strip().lower() not in {"", "0", "false", "no", "off"}


def _print_verify_cache_summary() -> None:
    """Print a one-shot verify-cache summary when the worker process exits."""
    if not _verify_cache_log_enabled():
        return

    with _fol_verify_cache_stats_lock:
        hits = _fol_verify_cache_stats["hits"]
        misses = _fol_verify_cache_stats["misses"]
        step_items = list(_fol_verify_cache_step_stats.items())
    total = hits + misses
    if total <= 0:
        return

    per_step_parts = []
    for step_idx, stats in step_items:
        step_total = stats["hits"] + stats["misses"]
        if step_total <= 0:
            continue
        step_rate = stats["hits"] / step_total
        per_step_parts.append(f"step{step_idx}: {stats['hits']}/{step_total}={step_rate:.1%}")
    suffix = f" | {', '.join(per_step_parts)}" if per_step_parts else ""
    print(
        f"[FOLVerifyCacheSummary][pid={os.getpid()}] "
        f"hits={hits} misses={misses} hit_rate={hits / total:.1%}{suffix}",
        flush=True,
    )


def _register_verify_cache_summary() -> None:
    """Register the summary printer exactly once."""
    global _fol_verify_cache_summary_registered
    with _fol_verify_cache_stats_lock:
        if _fol_verify_cache_summary_registered:
            return
        atexit.register(_print_verify_cache_summary)
        _fol_verify_cache_summary_registered = True


def _log_verify_cache_event(hit: bool, step_index: int) -> None:
    """Accumulate verify-cache hit statistics by logical step index."""
    if not _verify_cache_log_enabled():
        return

    _register_verify_cache_summary()
    with _fol_verify_cache_stats_lock:
        stat_key = "hits" if hit else "misses"
        _fol_verify_cache_stats[stat_key] += 1
        step_stats = _fol_verify_cache_step_stats.setdefault(step_index, {"hits": 0, "misses": 0})
        step_stats[stat_key] += 1


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
    return_debug: bool = False,
) -> float | dict:
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
        debug_info = {
            "cache_hit": False,
            "translation_response": None,
            "correction_attempts": 0,
            "z3_output": None,
            "z3_error": None,
            "judge_usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        # Format precheck: if the step contains <step> but has bad format
        # (missing <premise>/<conclusion>, mismatched tags, etc.), skip the
        # expensive FOL judge call and return 0.0 directly.
        if "<step>" in step_text and not check_step_format_fol(step_text):
            return {"score": 0.0, "debug": debug_info} if return_debug else 0.0
        shared_state = fol_shared_state or prepare_fol_shared_state(
            prompt_text, api_config=api_config, extra_info=extra_info
        )
        if shared_state is None:
            return {"score": 0.0, "debug": debug_info} if return_debug else 0.0

        fol_config = shared_state["config"]
        engine = FOLEngine(fol_config)

        # Handle cumulative mode
        if fol_config.cumulative and step_history:
            step_to_translate = "\n".join(step_history)
        else:
            step_to_translate = step_text
        step_index = max(0, len(step_history) - 1)

        verify_cache_key = _build_verify_cache_key(
            shared_state=shared_state,
            fol_config=fol_config,
            step_to_translate=step_to_translate,
        )
        with _fol_verify_cache_lock:
            cached_reward = _fol_verify_cache.get(verify_cache_key)
            if cached_reward is not None:
                _fol_verify_cache.move_to_end(verify_cache_key)
                debug_info["cache_hit"] = True
                _log_verify_cache_event(True, step_index)
                return {"score": cached_reward, "debug": debug_info} if return_debug else cached_reward
        _log_verify_cache_event(False, step_index)

        # print(f"[FOL][{_tid}] → verify_step({fol_config.translation.value})...", flush=True)
        # _t2 = time.time()
        reward = engine.verify_step(
            shared_state["processed_context"],
            shared_state["declarations"],
            step_to_translate,
            debug_info=debug_info,
        )
        reward = float(reward)
        with _fol_verify_cache_lock:
            existing_reward = _fol_verify_cache.get(verify_cache_key)
            if existing_reward is not None:
                _fol_verify_cache.move_to_end(verify_cache_key)
                debug_info["cache_hit"] = True
                return {"score": existing_reward, "debug": debug_info} if return_debug else existing_reward
            _fol_verify_cache[verify_cache_key] = reward
            if len(_fol_verify_cache) > _FOL_VERIFY_CACHE_MAX_SIZE:
                _fol_verify_cache.popitem(last=False)
        # print(f"[FOL][{_tid}] ◀ done  reward={reward}  verify={time.time()-_t2:.2f}s  total={time.time()-_t0:.2f}s", flush=True)
        return {"score": reward, "debug": debug_info} if return_debug else reward
    except Exception as e:
        # print(f"[FOL][{_tid}] ✗ EXCEPTION after {time.time()-_t0:.2f}s: {e}", flush=True)
        logger.warning("FOL reward computation failed: %s", e)
        if return_debug:
            return {
                "score": 0.0,
                "debug": {
                    "cache_hit": False,
                    "translation_response": None,
                    "correction_attempts": 0,
                    "z3_output": None,
                    "z3_error": str(e),
                    "judge_usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                },
            }
        return 0.0
