from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from omegaconf import OmegaConf, open_dict

from verl.trainer.ppo.sampling.mcts_prm import format_step_reward, get_prm_fn
from verl.utils.fol_verifier import FOLVerifier, LLMClient, load_fol_metadata


PROCESS_REWARD_DEFAULTS: Dict[str, Any] = {
    "type": "none",
    "fol": {
        "metadata_path": None,
        "verify_timeout": 10.0,
        "max_retries": 3,
        "debug_dir": None,
        "llm": {
            "api_base_url": "http://localhost:4869/v1",
            "api_key": "EMPTY",
            "model_name": None,
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.8,
        },
    },
}

AUTO_REWARD_MANAGER_BY_STRATEGY = {
    "tree_search": "tree",
    "treerl": "entropy",
    "step_treerl": "step_tree",
    "parallel_mcts": "mcts",
    "information_gain": "ig",
}

SUPPORTED_PROCESS_REWARD_TYPES = {"none", "format", "fol"}


@dataclass
class ProcessRewardRuntime:
    reward_type: str
    step_prm_fn: Optional[Callable] = None
    llm_client: Optional[LLMClient] = None
    fol_verifier: Optional[FOLVerifier] = None
    fol_metadata_map: Dict[str, Any] = field(default_factory=dict)


def _normalize_reward_type(value: Any) -> str:
    reward_type = str(value or "none").lower().strip()
    if reward_type not in SUPPORTED_PROCESS_REWARD_TYPES:
        supported = ", ".join(sorted(SUPPORTED_PROCESS_REWARD_TYPES))
        raise ValueError(
            f"Unsupported trainer.process_reward.type '{value}'. Supported: {supported}."
        )
    return reward_type


def resolve_process_reward_config(config):
    """Canonicalize process-reward config and resolve reward-manager defaults."""

    with open_dict(config):
        if "reward_model" not in config or config.reward_model is None:
            config.reward_model = OmegaConf.create({})
        merged_cfg = OmegaConf.merge(
            OmegaConf.create(copy.deepcopy(PROCESS_REWARD_DEFAULTS)),
            config.trainer.get("process_reward", {}) or {},
        )
        config.trainer.process_reward = merged_cfg

    strategy_name = str(config.trainer.get("sampling_strategy", "") or "").lower().strip()
    canonical = config.trainer.process_reward
    canonical.type = _normalize_reward_type(canonical.get("type", "none"))

    reward_manager_name = str(config.reward_model.get("reward_manager", "auto") or "auto").lower().strip()
    if reward_manager_name == "auto":
        resolved_manager = AUTO_REWARD_MANAGER_BY_STRATEGY.get(strategy_name, "naive")
        with open_dict(config):
            config.reward_model.reward_manager = resolved_manager

    if strategy_name in {"step_treerl", "parallel_mcts", "information_gain"} and canonical.type == "none":
        raise ValueError(
            f"sampling_strategy={strategy_name!r} requires trainer.process_reward.type to be "
            "'format' or 'fol'; got 'none'."
        )

    return config.trainer.process_reward


def build_process_reward_runtime(process_reward_cfg: Mapping[str, Any]) -> ProcessRewardRuntime:
    reward_type = _normalize_reward_type(process_reward_cfg.get("type", "none"))
    if reward_type == "none":
        return ProcessRewardRuntime(reward_type=reward_type, step_prm_fn=None)

    if reward_type == "format":
        return ProcessRewardRuntime(reward_type=reward_type, step_prm_fn=format_step_reward)

    fol_cfg = process_reward_cfg.get("fol", {}) or {}
    metadata_path = fol_cfg.get("metadata_path", None)
    if not metadata_path:
        raise ValueError(
            "trainer.process_reward.type='fol' requires trainer.process_reward.fol.metadata_path."
        )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"FOL metadata path not found: {metadata_path}"
        )

    llm_cfg = fol_cfg.get("llm", {}) or {}
    api_base_url = llm_cfg.get("api_base_url", None)
    model_name = llm_cfg.get("model_name", None)
    if not api_base_url:
        raise ValueError(
            "trainer.process_reward.type='fol' requires trainer.process_reward.fol.llm.api_base_url."
        )
    if not model_name:
        raise ValueError(
            "trainer.process_reward.type='fol' requires trainer.process_reward.fol.llm.model_name."
        )

    default_args = {
        "max_tokens": llm_cfg.get("max_tokens", 4096),
        "temperature": llm_cfg.get("temperature", 0.1),
        "top_p": llm_cfg.get("top_p", 0.8),
    }
    llm_client = LLMClient(
        base_url=api_base_url,
        api_key=llm_cfg.get("api_key") or "EMPTY",
        model=model_name,
        default_args=default_args,
    )
    fol_verifier = FOLVerifier(
        llm_client=llm_client,
        verify_timeout=float(fol_cfg.get("verify_timeout", 10.0)),
        max_retries=int(fol_cfg.get("max_retries", 3)),
        debug_dir=fol_cfg.get("debug_dir", None),
    )
    fol_metadata_map = load_fol_metadata(metadata_path)
    if not fol_metadata_map:
        raise ValueError(
            f"No FOL metadata entries loaded from {metadata_path}."
        )

    step_prm_fn = get_prm_fn(
        "fol",
        verifier=fol_verifier,
        metadata_map=fol_metadata_map,
    )
    return ProcessRewardRuntime(
        reward_type=reward_type,
        step_prm_fn=step_prm_fn,
        llm_client=llm_client,
        fol_verifier=fol_verifier,
        fol_metadata_map=fol_metadata_map,
    )


def normalize_process_reward_cfg(process_reward_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    base = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.create(copy.deepcopy(PROCESS_REWARD_DEFAULTS)),
            process_reward_cfg,
        ),
        resolve=True,
    )
    base["type"] = _normalize_reward_type(base.get("type", "none"))
    return base


def get_batch_sample_id(non_tensor_batch: Optional[Mapping[str, Any]], index: int) -> Optional[str]:
    if not non_tensor_batch:
        return None

    sample_ids = non_tensor_batch.get("sample_id", None)
    if sample_ids is not None:
        try:
            if index < len(sample_ids):
                sample_id = sample_ids[index]
                if sample_id is not None:
                    return str(sample_id)
        except TypeError:
            if index == 0 and sample_ids is not None:
                return str(sample_ids)

    extra_info = non_tensor_batch.get("extra_info", None)
    if extra_info is not None:
        try:
            if index < len(extra_info):
                info = extra_info[index]
                if isinstance(info, Mapping):
                    for key in ("sample_id", "index", "id"):
                        value = info.get(key, None)
                        if value is not None:
                            return str(value)
        except TypeError:
            if isinstance(extra_info, Mapping):
                for key in ("sample_id", "index", "id"):
                    value = extra_info.get(key, None)
                    if value is not None:
                        return str(value)

    return None


def require_batch_sample_id(
    non_tensor_batch: Optional[Mapping[str, Any]],
    index: int,
    *,
    context: str,
) -> str:
    sample_id = get_batch_sample_id(non_tensor_batch, index)
    if sample_id is None:
        raise ValueError(f"{context} requires sample_id for batch index {index}.")
    return sample_id


def get_item_sample_id(non_tensor_batch: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not non_tensor_batch:
        return None

    sample_id = non_tensor_batch.get("sample_id", None)
    if sample_id is not None:
        return str(sample_id)

    extra_info = non_tensor_batch.get("extra_info", None)
    if isinstance(extra_info, Mapping):
        for key in ("sample_id", "index", "id"):
            value = extra_info.get(key, None)
            if value is not None:
                return str(value)

    return None
