from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ApiConfig:
    base_url: str = "http://localhost:4869/v1"
    api_key: str = "EMPTY"
    model: str = "eval-model"
    timeout: float = 180.0
    max_retries: int = 3
    retry_sleep: float = 2.0
    bypass_env_proxy: bool = True


@dataclass
class SamplingConfig:
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 20
    max_tokens: Optional[int] = 4096
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    seed: Optional[int] = None
    stop: Optional[list[str] | str] = None
    logprobs: Optional[bool] = True
    top_logprobs: Optional[int] = 5
    response_format: Optional[dict[str, Any]] = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApiChoice:
    text: str
    finish_reason: Optional[str] = None
    avg_logprob: Optional[float] = None
    token_logprobs: list[float] = field(default_factory=list)
    raw_index: int = 0


@dataclass
class ApiResponse:
    choices: list[ApiChoice]
    usage: dict[str, Any] = field(default_factory=dict)
    raw_model: Optional[str] = None


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return _expand_env(json.load(f))


def _dataclass_from_dict(cls, values: Optional[dict[str, Any]]):
    data = cls()
    if not values:
        return data
    valid_keys = set(data.__dataclass_fields__)  # type: ignore[attr-defined]
    for key, value in values.items():
        if key in valid_keys:
            setattr(data, key, value)
    return data


def load_api_config(path: str | Path) -> tuple[ApiConfig, SamplingConfig]:
    data = _load_json(path)
    return _dataclass_from_dict(ApiConfig, data.get("api")), _dataclass_from_dict(SamplingConfig, data.get("sampling"))


def merge_sampling_config(base: SamplingConfig, overrides: Optional[dict[str, Any]] = None) -> SamplingConfig:
    merged = copy.deepcopy(base)
    if not overrides:
        return merged
    valid_keys = set(merged.__dataclass_fields__)  # type: ignore[attr-defined]
    for key, value in overrides.items():
        if value is not None and key in valid_keys:
            setattr(merged, key, value)
    return merged


def _deep_merge_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def sampling_to_create_kwargs(sampling: SamplingConfig, *, n: int = 1, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    sampling = merge_sampling_config(sampling, overrides)
    kwargs: dict[str, Any] = {}
    extra_body = copy.deepcopy(sampling.extra_body or {})

    direct_keys = [
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
        "logprobs",
        "response_format",
    ]
    for key in direct_keys:
        value = getattr(sampling, key)
        if value is not None:
            kwargs[key] = value

    if sampling.logprobs and sampling.top_logprobs is not None:
        kwargs["top_logprobs"] = sampling.top_logprobs

    if sampling.top_k is not None and int(sampling.top_k) >= 0:
        extra_body["top_k"] = sampling.top_k
    if sampling.repetition_penalty is not None:
        extra_body["repetition_penalty"] = sampling.repetition_penalty

    if n is not None and n > 1:
        kwargs["n"] = int(n)
    if extra_body:
        kwargs["extra_body"] = _deep_merge_dict(kwargs.get("extra_body", {}), extra_body)
    return kwargs


def _model_dump(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return {}


def _extract_choice(choice: Any, idx: int) -> ApiChoice:
    message = getattr(choice, "message", None)
    text = getattr(message, "content", None)
    if text is None:
        text = ""

    token_logprobs: list[float] = []
    logprobs = getattr(choice, "logprobs", None)
    content_logprobs = getattr(logprobs, "content", None) if logprobs is not None else None
    if content_logprobs:
        for token_info in content_logprobs:
            value = getattr(token_info, "logprob", None)
            if value is not None:
                token_logprobs.append(float(value))

    avg_logprob = None
    if token_logprobs:
        avg_logprob = sum(token_logprobs) / len(token_logprobs)

    return ApiChoice(
        text=str(text),
        finish_reason=getattr(choice, "finish_reason", None),
        avg_logprob=avg_logprob,
        token_logprobs=token_logprobs,
        raw_index=idx,
    )


class OpenAICompatibleClient:
    def __init__(self, api_config: ApiConfig, sampling_config: Optional[SamplingConfig] = None):
        import httpx
        from openai import OpenAI

        self.api_config = api_config
        self.sampling_config = sampling_config or SamplingConfig()
        http_client = httpx.Client(
            timeout=api_config.timeout,
            trust_env=not bool(api_config.bypass_env_proxy),
        )
        self.client = OpenAI(
            api_key=api_config.api_key,
            base_url=api_config.base_url,
            timeout=api_config.timeout,
            http_client=http_client,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        n: int = 1,
        sampling_overrides: Optional[dict[str, Any]] = None,
    ) -> ApiResponse:
        create_kwargs = sampling_to_create_kwargs(self.sampling_config, n=n, overrides=sampling_overrides)
        last_error: Optional[Exception] = None
        for attempt in range(max(1, int(self.api_config.max_retries))):
            try:
                response = self.client.chat.completions.create(
                    model=self.api_config.model,
                    messages=messages,
                    **create_kwargs,
                )
                choices = [_extract_choice(choice, idx) for idx, choice in enumerate(response.choices)]
                return ApiResponse(
                    choices=choices,
                    usage=_model_dump(getattr(response, "usage", None)),
                    raw_model=getattr(response, "model", None),
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= max(1, int(self.api_config.max_retries)):
                    break
                time.sleep(float(self.api_config.retry_sleep))
        raise RuntimeError(f"OpenAI-compatible API call failed after {self.api_config.max_retries} attempts") from last_error


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--print_config", action="store_true")
    args = parser.parse_args()

    api_config, sampling_config = load_api_config(args.config)
    if args.print_config:
        print(json.dumps({"api": asdict(api_config), "sampling": asdict(sampling_config)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
