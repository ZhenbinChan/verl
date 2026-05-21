from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from omegaconf import OmegaConf, open_dict

from verl.trainer.ppo.sampling.mcts_prm import format_step_reward, get_prm_fn
from verl.utils.fol_verifier import FOLVerifier, LLMClient, load_fol_metadata


DEFAULT_SELF_EVAL_PROMPT_PATH = str(Path(__file__).resolve().parents[1] / "prompts" / "self_eval_reward.txt")

PROCESS_REWARD_DEFAULTS: Dict[str, Any] = {
    "type": "none",
    "fol": {
        "prm_mode": "global_fol_prm",
        "metadata_path": None,
        "online_declaration_fallback": True,
        "fail_on_missing_metadata": False,
        "verify_timeout": 10.0,
        "max_retries": 3,
        "debug_dir": None,
        "llm": {
            "api_config": None,
            "provider": "openai_compatible",
            "api_base_url": "http://localhost:4869/v1",
            "api_key": "EMPTY",
            "model_name": None,
            "azure_endpoint": None,
            "api_version": None,
            "deployment_name": None,
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.8,
            "default_args": {},
            "max_concurrency": 8,
            "request_timeout": 60,
            "extra_body": {},
            "bypass_env_proxy": False,
        },
    },
    "self_eval": {
        "prompt_path": DEFAULT_SELF_EVAL_PROMPT_PATH,
        "max_new_tokens": 32,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_batch_size": None,
        "fail_on_parse_error": False,
    },
}

AUTO_REWARD_MANAGER_BY_STRATEGY = {
    "tree_search": "tree",
    "treerl": "entropy",
    "step_treerl": "step_tree",
    "parallel_mcts": "mcts",
    "information_gain": "ig",
}

SUPPORTED_PROCESS_REWARD_TYPES = {"none", "format", "fol", "self_eval"}


@dataclass
class StepRewardRequest:
    step_text: str
    sample_id: Optional[str] = None
    question_text: Optional[str] = None
    accumulated_text: str = ""
    tree_idx: int = 0
    node_idx: int = 0


@dataclass
class ProcessRewardRuntime:
    reward_type: str
    step_prm_fn: Optional[Callable] = None
    llm_client: Optional[LLMClient] = None
    fol_verifier: Optional[FOLVerifier] = None
    fol_metadata_map: Dict[str, Any] = field(default_factory=dict)
    fol_prm_mode: str = "format"
    fol_online_declaration_fallback: bool = True
    fol_fail_on_missing_metadata: bool = False
    llm_default_args: Dict[str, Any] = field(default_factory=dict)
    max_concurrency: int = 8
    self_eval_prompt_template: str = ""
    self_eval_max_new_tokens: int = 32
    self_eval_temperature: float = 0.0
    self_eval_top_p: float = 1.0
    self_eval_max_batch_size: Optional[int] = None
    self_eval_fail_on_parse_error: bool = False
    _score_cache: Dict[tuple, float] = field(default_factory=dict)
    _metadata_cache: Dict[str, Any] = field(default_factory=dict)

    def score_steps(self, requests: list[StepRewardRequest]) -> list[float]:
        if self.reward_type == "format":
            return [float(self.step_prm_fn(req.step_text)) for req in requests]
        if self.reward_type == "fol":
            return self._score_fol_steps(requests)
        raise ValueError(f"Process reward type {self.reward_type!r} does not support step scoring.")

    def _get_or_create_metadata(self, req: StepRewardRequest):
        if req.sample_id is None:
            raise ValueError("FOL process reward requires sample_id for batch step scoring.")
        sample_id = str(req.sample_id)
        metadata = self.fol_metadata_map.get(sample_id) or self._metadata_cache.get(sample_id)
        if metadata is not None:
            if req.question_text and not getattr(metadata, "question_text", ""):
                metadata.question_text = req.question_text
            return metadata

        if not self.fol_online_declaration_fallback:
            if self.fol_fail_on_missing_metadata:
                raise KeyError(f"Missing FOL metadata for sample_id={sample_id!r}.")
            return None
        if not req.question_text:
            if self.fol_fail_on_missing_metadata:
                raise KeyError(
                    f"Missing FOL metadata and question_text for sample_id={sample_id!r}."
                )
            return None

        from verl.utils.fol_verifier import FOLMetadata

        metadata = FOLMetadata(
            sample_id=sample_id,
            rephrased_context="",
            question_text=req.question_text,
            prm_mode=self.fol_prm_mode,
            z3_declaration_code="",
        )
        self._metadata_cache[sample_id] = metadata
        return metadata

    def _score_one_fol(self, req: StepRewardRequest) -> float:
        metadata = self._get_or_create_metadata(req)
        if metadata is None:
            return 0.0
        if self.fol_prm_mode == "global_fol_prm":
            text = req.accumulated_text or req.step_text
            cache_key = (self.fol_prm_mode, req.sample_id, text)
            if cache_key in self._score_cache:
                return self._score_cache[cache_key]
            score = float(
                self.fol_verifier.verify_global_step(
                    metadata,
                    text,
                    question_text=req.question_text,
                    args=self.llm_default_args,
                )
            )
            self._score_cache[cache_key] = score
            return score
        if self.fol_prm_mode == "local_fol_prm":
            cache_key = (self.fol_prm_mode, req.sample_id, req.step_text)
            if cache_key in self._score_cache:
                return self._score_cache[cache_key]
            if not getattr(metadata, "z3_declaration_code", "") and req.question_text:
                metadata.z3_declaration_code = self.fol_verifier.generate_global_declaration(
                    req.question_text,
                    args=self.llm_default_args,
                )
            score = float(
                self.fol_verifier.verify_step(
                    metadata,
                    req.step_text,
                    use_llm=True,
                    args=self.llm_default_args,
                )
            )
            self._score_cache[cache_key] = score
            return score
        raise ValueError(
            "trainer.process_reward.fol.prm_mode must be "
            "'global_fol_prm' or 'local_fol_prm', "
            f"but got {self.fol_prm_mode!r}."
        )

    def _score_fol_steps(self, requests: list[StepRewardRequest]) -> list[float]:
        if not requests:
            return []
        key_to_req: Dict[tuple, StepRewardRequest] = {}
        request_keys = []
        for req in requests:
            if req.sample_id is None:
                raise ValueError("FOL process reward requires sample_id for batch step scoring.")
            text_key = req.accumulated_text if self.fol_prm_mode == "global_fol_prm" else req.step_text
            key = (self.fol_prm_mode, str(req.sample_id), text_key)
            request_keys.append(key)
            if key not in key_to_req:
                key_to_req[key] = req

        scores_by_key: Dict[tuple, float] = {}
        pending = []
        for key, req in key_to_req.items():
            if key in self._score_cache:
                scores_by_key[key] = self._score_cache[key]
            else:
                pending.append((key, req))

        if pending:
            workers = max(1, min(int(self.max_concurrency or 1), len(pending)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_key = {
                    executor.submit(self._score_one_fol, req): key
                    for key, req in pending
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        score = float(future.result())
                    except Exception:
                        if self.fol_fail_on_missing_metadata:
                            raise
                        score = 0.0
                    self._score_cache[key] = score
                    scores_by_key[key] = score

        return [float(scores_by_key.get(key, 0.0)) for key in request_keys]


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
    if strategy_name == "step_treerl" and canonical.type == "none":
        canonical.type = "format"

    reward_manager_name = str(config.reward_model.get("reward_manager", "auto") or "auto").lower().strip()
    if reward_manager_name == "auto":
        resolved_manager = AUTO_REWARD_MANAGER_BY_STRATEGY.get(strategy_name, "naive")
        with open_dict(config):
            config.reward_model.reward_manager = resolved_manager

    if strategy_name in {"step_treerl", "parallel_mcts", "information_gain"} and canonical.type == "none":
        raise ValueError(
            f"sampling_strategy={strategy_name!r} requires trainer.process_reward.type to be "
            "'format', 'fol', or 'self_eval'; got 'none'."
        )

    return config.trainer.process_reward


def _load_self_eval_prompt_template(prompt_path: Any) -> str:
    if not prompt_path:
        raise ValueError("trainer.process_reward.self_eval.prompt_path is required.")
    path = Path(str(prompt_path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Self-eval reward prompt path not found: {path}")
    template = path.read_text(encoding="utf-8")
    required_placeholders = ("{question_text}", "{reasoning_steps}")
    missing = [placeholder for placeholder in required_placeholders if placeholder not in template]
    if missing:
        raise ValueError(
            "Self-eval reward prompt must contain placeholders: "
            f"{', '.join(required_placeholders)}. Missing: {', '.join(missing)}."
        )
    return template


def _load_llm_api_config(config_path: Any) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(str(config_path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"FOL LLM API config not found: {path}")
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"FOL LLM API config must be a mapping: {path}")
    return {
        "provider": config.get("provider", None),
        "api_base_url": config.get("api_base_url", config.get("base_url", None)),
        "api_key": config.get("api_key", None),
        "model_name": config.get("model_name", config.get("model", None)),
        "azure_endpoint": config.get("azure_endpoint", None),
        "api_version": config.get("api_version", None),
        "deployment_name": config.get("deployment_name", None),
        "request_timeout": config.get("request_timeout", None),
        "bypass_env_proxy": config.get("bypass_env_proxy", None),
        "default_args": config.get("default_args", {}) or {},
        "extra_body": config.get("extra_body", {}) or {},
    }


def _merge_llm_cfg_with_api_config(llm_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    llm_cfg = dict(llm_cfg or {})
    api_config_path = llm_cfg.get("api_config", None)
    if not api_config_path:
        return llm_cfg

    merged = _load_llm_api_config(api_config_path)
    defaults = PROCESS_REWARD_DEFAULTS["fol"]["llm"]
    for key, value in llm_cfg.items():
        if key == "api_config":
            continue
        if value is None:
            continue
        if key not in defaults or value != defaults.get(key):
            merged[key] = value
    merged["api_config"] = api_config_path
    return merged


def build_process_reward_runtime(process_reward_cfg: Mapping[str, Any]) -> ProcessRewardRuntime:
    reward_type = _normalize_reward_type(process_reward_cfg.get("type", "none"))
    if reward_type == "none":
        return ProcessRewardRuntime(reward_type=reward_type, step_prm_fn=None)

    if reward_type == "format":
        return ProcessRewardRuntime(reward_type=reward_type, step_prm_fn=format_step_reward)

    if reward_type == "self_eval":
        self_eval_cfg = process_reward_cfg.get("self_eval", {}) or {}
        max_batch_size = self_eval_cfg.get("max_batch_size", None)
        if max_batch_size is not None:
            max_batch_size = int(max_batch_size)
            if max_batch_size <= 0:
                raise ValueError("trainer.process_reward.self_eval.max_batch_size must be positive when set.")
        return ProcessRewardRuntime(
            reward_type=reward_type,
            step_prm_fn=None,
            self_eval_prompt_template=_load_self_eval_prompt_template(self_eval_cfg.get("prompt_path") or DEFAULT_SELF_EVAL_PROMPT_PATH),
            self_eval_max_new_tokens=int(self_eval_cfg.get("max_new_tokens", 32)),
            self_eval_temperature=float(self_eval_cfg.get("temperature", 0.0)),
            self_eval_top_p=float(self_eval_cfg.get("top_p", 1.0)),
            self_eval_max_batch_size=max_batch_size,
            self_eval_fail_on_parse_error=bool(self_eval_cfg.get("fail_on_parse_error", False)),
        )

    fol_cfg = process_reward_cfg.get("fol", {}) or {}
    prm_mode = str(fol_cfg.get("prm_mode", "global_fol_prm") or "global_fol_prm").lower()
    if prm_mode not in {"global_fol_prm", "local_fol_prm"}:
        raise ValueError(
            "trainer.process_reward.fol.prm_mode must be "
            "'global_fol_prm' or 'local_fol_prm'."
        )
    online_declaration_fallback = bool(fol_cfg.get("online_declaration_fallback", True))
    fail_on_missing_metadata = bool(fol_cfg.get("fail_on_missing_metadata", False))

    metadata_path = fol_cfg.get("metadata_path", None)
    if metadata_path and not os.path.exists(metadata_path):
        if not online_declaration_fallback or fail_on_missing_metadata:
            raise FileNotFoundError(
                f"FOL metadata path not found: {metadata_path}"
            )

    llm_cfg = _merge_llm_cfg_with_api_config(fol_cfg.get("llm", {}) or {})
    provider = str(llm_cfg.get("provider", "openai_compatible") or "openai_compatible").lower()
    api_base_url = llm_cfg.get("api_base_url", llm_cfg.get("base_url", None))
    model_name = llm_cfg.get("model_name", llm_cfg.get("model", None))
    deployment_name = llm_cfg.get("deployment_name", None)
    if provider != "azure_openai" and not api_base_url:
        raise ValueError(
            "trainer.process_reward.type='fol' requires trainer.process_reward.fol.llm.api_base_url."
        )
    if provider == "azure_openai" and not llm_cfg.get("azure_endpoint", None):
        raise ValueError(
            "trainer.process_reward.type='fol' with Azure requires trainer.process_reward.fol.llm.azure_endpoint."
        )
    if not model_name and not deployment_name:
        raise ValueError(
            "trainer.process_reward.type='fol' requires trainer.process_reward.fol.llm.model_name or deployment_name."
        )

    default_args = {
        "max_tokens": llm_cfg.get("max_tokens", 4096),
        "temperature": llm_cfg.get("temperature", 0.1),
        "top_p": llm_cfg.get("top_p", 0.8),
    }
    default_args.update(llm_cfg.get("default_args", {}) or {})
    llm_client = LLMClient(
        base_url=api_base_url,
        api_key=llm_cfg.get("api_key") or "EMPTY",
        model=model_name or deployment_name,
        default_args=default_args,
        provider=provider,
        azure_endpoint=llm_cfg.get("azure_endpoint", None),
        api_version=llm_cfg.get("api_version", None),
        deployment_name=deployment_name,
        request_timeout=llm_cfg.get("request_timeout", None),
        extra_body=llm_cfg.get("extra_body", None),
        bypass_env_proxy=bool(llm_cfg.get("bypass_env_proxy", False)),
    )
    fol_verifier = FOLVerifier(
        llm_client=llm_client,
        verify_timeout=float(fol_cfg.get("verify_timeout", 10.0)),
        max_retries=int(fol_cfg.get("max_retries", 3)),
        debug_dir=fol_cfg.get("debug_dir", None),
    )
    fol_metadata_map = load_fol_metadata(metadata_path) if metadata_path else {}
    if not fol_metadata_map and not online_declaration_fallback:
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
        fol_prm_mode=prm_mode,
        fol_online_declaration_fallback=online_declaration_fallback,
        fol_fail_on_missing_metadata=fail_on_missing_metadata,
        llm_default_args=default_args,
        max_concurrency=int(llm_cfg.get("max_concurrency", 8)),
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


def build_generation_non_tensor_keys_to_pop(
    non_tensor_batch: Mapping[str, Any],
    process_reward_type: str,
) -> list[str]:
    """Return non-tensor fields that should move from training batch to generation batch."""

    keys_to_pop = ["raw_prompt_ids"]
    for key in ("multi_modal_data", "multi_modal_inputs"):
        if key in non_tensor_batch:
            keys_to_pop.append(key)
    for key in ("raw_prompt", "tools_kwargs", "answer"):
        if key in non_tensor_batch:
            keys_to_pop.append(key)

    reward_type = str(process_reward_type or "none").lower().strip()
    if reward_type == "fol":
        process_reward_keys = ("sample_id", "question_text", "extra_info", "data_source", "index")
    elif reward_type == "self_eval":
        process_reward_keys = ("question_text", "extra_info", "index")
    else:
        process_reward_keys = ()

    for key in process_reward_keys:
        if key in non_tensor_batch and key not in keys_to_pop:
            keys_to_pop.append(key)
    return keys_to_pop


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
                    for key in ("sample_id", "id"):
                        value = info.get(key, None)
                        if value is not None:
                            return str(value)
                    value = info.get("index", None)
                    if value is not None:
                        data_source = _get_indexed_value(non_tensor_batch.get("data_source", None), index)
                        if data_source is not None:
                            return f"{data_source}_{value}"
                        return str(value)
        except TypeError:
            if isinstance(extra_info, Mapping):
                for key in ("sample_id", "id"):
                    value = extra_info.get(key, None)
                    if value is not None:
                        return str(value)
                value = extra_info.get("index", None)
                if value is not None:
                    data_source = _get_indexed_value(non_tensor_batch.get("data_source", None), index)
                    if data_source is not None:
                        return f"{data_source}_{value}"
                    return str(value)

    index_values = non_tensor_batch.get("index", None)
    sample_index = _get_indexed_value(index_values, index)
    if sample_index is not None:
        data_source = _get_indexed_value(non_tensor_batch.get("data_source", None), index)
        if data_source is not None:
            return f"{data_source}_{sample_index}"
        return str(sample_index)

    return None


def _get_indexed_value(container: Any, index: int) -> Any:
    if container is None:
        return None
    try:
        if isinstance(container, (list, tuple)) and index < len(container):
            return container[index]
        if hasattr(container, "__len__") and not isinstance(container, (str, bytes, Mapping)):
            if index < len(container):
                return container[index]
    except TypeError:
        pass
    if index == 0:
        return container
    return None


def _extract_prompt_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, Mapping):
        content = value.get("content", None)
        return str(content) if content is not None and str(content).strip() else None
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            text = _extract_prompt_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None
    try:
        if hasattr(value, "tolist"):
            return _extract_prompt_text(value.tolist())
    except Exception:
        pass
    text = str(value)
    return text if text.strip() else None


def _build_question_from_extra_info(info: Mapping[str, Any]) -> Optional[str]:
    for key in ("question", "raw_prompt", "prompt"):
        text = _extract_prompt_text(info.get(key, None))
        if text and "<Context>" in text:
            return text
    context = info.get("context", "")
    query = info.get("query", "")
    options = info.get("options", "")
    if context or query or options:
        return f"<Context>{context}</Context><Question>{query}</Question><Options>{options}</Options>"
    for key in ("question", "raw_prompt", "prompt"):
        text = _extract_prompt_text(info.get(key, None))
        if text:
            return text
    return None


def get_batch_question_text(non_tensor_batch: Optional[Mapping[str, Any]], index: int) -> Optional[str]:
    if not non_tensor_batch:
        return None

    for key in ("raw_prompt", "question_text"):
        text = _extract_prompt_text(_get_indexed_value(non_tensor_batch.get(key, None), index))
        if text:
            return text

    prompt_value = _get_indexed_value(non_tensor_batch.get("prompt", None), index)
    text = _extract_prompt_text(prompt_value)
    if text:
        return text

    extra_info = non_tensor_batch.get("extra_info", None)
    info = _get_indexed_value(extra_info, index)
    if isinstance(info, Mapping):
        text = _build_question_from_extra_info(info)
        if text:
            return text
    elif isinstance(extra_info, Mapping):
        text = _build_question_from_extra_info(extra_info)
        if text:
            return text

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
