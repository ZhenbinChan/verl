#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mcts_utils.nl2fol_lzy.pipeline import load_prompt, verify_fol_step
from mcts_utils.nl2fol_lzy.utils import extract_python_code, parse_python_logic_steps


STEP_PATTERN = re.compile(r"<step\b[^>]*>.*?</step>", re.DOTALL)


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_python(v) for v in value]
    return value


def _extract_prompt_text(value: Any) -> Optional[str]:
    value = _to_python(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, dict):
        content = value.get("content")
        return str(content) if content is not None and str(content).strip() else None
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _extract_prompt_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None
    text = str(value)
    return text if text.strip() else None


def get_question_text(record: dict[str, Any]) -> str:
    for key in ("raw_prompt", "question_text"):
        text = _extract_prompt_text(record.get(key))
        if text:
            return text
    extra_info = _to_python(record.get("extra_info"))
    if isinstance(extra_info, dict):
        for key in ("question", "raw_prompt", "prompt"):
            text = _extract_prompt_text(extra_info.get(key))
            if text:
                return text
    text = _extract_prompt_text(record.get("prompt"))
    if text:
        return text
    raise ValueError("Could not find question text in record.")


def get_ground_truth(record: dict[str, Any]) -> str:
    answer = record.get("answer")
    if answer is not None:
        return str(answer)
    reward_model = _to_python(record.get("reward_model"))
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    extra_info = _to_python(record.get("extra_info"))
    if isinstance(extra_info, dict) and extra_info.get("answer") is not None:
        return str(extra_info["answer"])
    return ""


def get_sample_id(record: dict[str, Any], fallback_idx: int) -> str:
    sample_id = record.get("sample_id")
    if sample_id is not None:
        return str(sample_id)
    extra_info = _to_python(record.get("extra_info"))
    if isinstance(extra_info, dict):
        for key in ("sample_id", "id"):
            if extra_info.get(key) is not None:
                return str(extra_info[key])
        if extra_info.get("index") is not None:
            data_source = record.get("data_source")
            if data_source is not None:
                return f"{data_source}_{extra_info['index']}"
            return str(extra_info["index"])
    return f"sample_{fallback_idx}"


def build_model_prompt(prompt_template: str, question_text: str) -> str:
    return f"{prompt_template.rstrip()}\n\n{question_text.strip()}"


def extract_step_blocks(response_text: str) -> list[str]:
    return [match.group(0).strip() for match in STEP_PATTERN.finditer(response_text or "")]


def build_reasoning_prefix(step_blocks: list[str], stop_index: int) -> str:
    return "\n\n".join(step_blocks[: stop_index + 1])


def answer_correct(response_text: str, ground_truth: str) -> bool:
    try:
        from verl.utils.reward_score.logi import compute_score as compute_logi_score

        score, _ = compute_logi_score(response_text or "", str(ground_truth))
        return bool(score)
    except Exception:
        matches = re.findall(r"\\boxed\{\{?([A-Za-z])\}?\}", response_text or "")
        if not matches:
            matches = re.findall(r"The answer is\s*([A-Za-z])\s*\.", response_text or "")
        return bool(matches and matches[-1].upper() == str(ground_truth).upper())


def step_format_correct(step_text: str) -> bool:
    try:
        from verl.trainer.ppo.sampling.mcts_prm import strict_step_xml_correct

        return strict_step_xml_correct(step_text)
    except Exception:
        try:
            root = ET.fromstring(step_text.strip())
        except ET.ParseError:
            return False
        if root.tag != "step":
            return False
        if root.text and root.text.strip():
            return False
        premise_count = 0
        conclusion_count = 0
        for child in list(root):
            if child.tag == "premise":
                premise_count += 1
            elif child.tag == "conclusion":
                conclusion_count += 1
            else:
                return False
            if list(child):
                return False
            if child.tail and child.tail.strip():
                return False
        return premise_count >= 1 and conclusion_count == 1


class VLLMChatGenerator:
    def __init__(
        self,
        *,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        trust_remote_code: bool,
        dtype: str,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
        )

    def format_chat(self, *, user: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        batch_size: int,
    ) -> list[str]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=1,
        )
        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            batch_outputs = self.llm.generate(batch, sampling_params)
            outputs.extend(output.outputs[0].text for output in batch_outputs)
        return outputs


def load_fol_api_config(config_path: str) -> dict[str, Any]:
    import yaml

    path = Path(config_path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"FOL API config must be a mapping: {path}")

    base_url = config.get("api_base_url") or config.get("base_url")
    model = config.get("model_name") or config.get("model")
    if not base_url:
        raise ValueError(f"FOL API config missing base_url/api_base_url: {path}")
    if not model:
        raise ValueError(f"FOL API config missing model_name/model: {path}")

    return {
        "provider": config.get("provider", "openai_compatible"),
        "base_url": base_url,
        "api_key": config.get("api_key") or os.getenv("OPENAI_API_KEY") or "EMPTY",
        "model": model,
        "request_timeout": config.get("request_timeout"),
        "bypass_env_proxy": bool(config.get("bypass_env_proxy", False)),
        "default_args": dict(config.get("default_args") or {}),
        "extra_body": dict(config.get("extra_body") or {}),
    }


def build_api_request_kwargs(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    default_args: Optional[dict[str, Any]] = None,
    extra_body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    request_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    request_kwargs.update(default_args or {})
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    return request_kwargs


class OpenAICompatibleFOLGenerator:
    def __init__(
        self,
        *,
        api_config: dict[str, Any],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        import httpx
        from openai import OpenAI

        self.model = str(api_config["model"])
        self.default_args = dict(api_config.get("default_args") or {})
        self.extra_body = dict(api_config.get("extra_body") or {})
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        client_kwargs = {
            "base_url": api_config["base_url"],
            "api_key": api_config["api_key"],
        }
        if api_config.get("request_timeout") is not None:
            client_kwargs["timeout"] = api_config["request_timeout"]
        if api_config.get("bypass_env_proxy"):
            client_kwargs["http_client"] = httpx.Client(trust_env=False)
        self.client = OpenAI(**client_kwargs)

    def generate_one(self, *, user: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response = self.client.chat.completions.create(
            **build_api_request_kwargs(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                default_args=self.default_args,
                extra_body=self.extra_body,
            )
        )
        return response.choices[0].message.content

    def generate(self, users: list[str], *, system: str, max_workers: int) -> list[str]:
        workers = max(1, min(int(max_workers or 1), len(users) or 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(lambda user: self.generate_one(user=user, system=system), users))


def init_wandb(args: argparse.Namespace):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )


def log_wandb(run, metrics: dict[str, Any], trajectories: list[dict[str, Any]]) -> None:
    if run is None:
        return
    import wandb

    table = wandb.Table(
        columns=[
            "sample_id",
            "fol_correct_steps",
            "total_steps",
            "answer_correct",
            "format_correct_steps",
            "format_ratio",
            "response",
        ]
    )
    for item in trajectories:
        table.add_data(
            item["sample_id"],
            item["fol_correct_steps"],
            item["total_steps"],
            item["answer_correct"],
            item["format_correct_steps"],
            item["format_ratio"],
            item["response"],
        )
    numeric_metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float, bool))}
    run.log({**numeric_metrics, "trajectories": table})
    run.finish()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    model_name = Path(args.model_path).name.lower().replace("-", "_")
    split_name = Path(args.data_path).stem
    return REPO_ROOT / "outputs" / "logiqa_fol_pilot" / f"{model_name}_logiqa_{split_name}_{args.num_samples}_seed{args.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 100-sample LogiQA vLLM + FOL process-reward pilot.")
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "logiqa" / "test.parquet"))
    parser.add_argument("--model-path", default="/home/chenzhb/Workspaces/LLMs/Qwen3-8B-Base")
    parser.add_argument("--prompt-path", default=str(REPO_ROOT / "prompts" / "premise_conclusion.txt"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--fol-max-tokens", type=int, default=4096)
    parser.add_argument("--fol-temperature", type=float, default=0.1)
    parser.add_argument("--fol-top-p", type=float, default=0.8)
    parser.add_argument("--fol-batch-size", type=int, default=4, help="Deprecated; FOL API calls use --fol-max-workers.")
    parser.add_argument("--fol-api-config", default=str(REPO_ROOT / "llm_server" / "configs" / "deepseek.yaml"))
    parser.add_argument("--fol-max-workers", type=int, default=8)
    parser.add_argument("--z3-timeout", type=float, default=1.0)
    parser.add_argument("--wandb-project", default="verl-logiqa-pilot")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data_path)
    sampled = df.sample(n=min(args.num_samples, len(df)), random_state=args.seed).reset_index(drop=True)
    sampled.to_parquet(output_dir / "sampled_inputs.parquet")

    prompt_template = Path(args.prompt_path).read_text(encoding="utf-8")
    records = sampled.to_dict("records")
    samples = []
    for idx, record in enumerate(records):
        question_text = get_question_text(record)
        ground_truth = get_ground_truth(record)
        samples.append(
            {
                "sample_id": get_sample_id(record, idx),
                "question_text": question_text,
                "ground_truth": ground_truth,
                "prompt_text": build_model_prompt(prompt_template, question_text),
            }
        )

    generator = VLLMChatGenerator(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )

    print(f"[logiqa_fol_pilot] generating {len(samples)} trajectories", flush=True)
    model_prompts = [generator.format_chat(user=sample["prompt_text"]) for sample in samples]
    responses = generator.generate(
        model_prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )

    trajectories: list[dict[str, Any]] = []
    for sample, response in zip(samples, responses):
        step_blocks = extract_step_blocks(response)
        steps = []
        for step_index, step_text in enumerate(step_blocks):
            is_format_correct = step_format_correct(step_text)
            steps.append(
                {
                    "step_index": step_index + 1,
                    "text": step_text,
                    "format_correct": is_format_correct,
                    "fol_score": 0.0,
                    "fol_error": "",
                    "fol_output": None,
                    "implication_code": "",
                }
            )
        format_correct_steps = sum(1 for step in steps if step["format_correct"])
        total_steps = len(steps)
        trajectories.append(
            {
                **sample,
                "response": response,
                "answer_correct": answer_correct(response, sample["ground_truth"]),
                "steps": steps,
                "total_steps": total_steps,
                "format_correct_steps": format_correct_steps,
                "format_ratio": (format_correct_steps / total_steps) if total_steps else 0.0,
                "fol_correct_steps": 0,
                "declaration_code": "",
                "declaration_error": "",
            }
        )

    fol_api_config = load_fol_api_config(args.fol_api_config)
    fol_generator = OpenAICompatibleFOLGenerator(
        api_config=fol_api_config,
        max_tokens=args.fol_max_tokens,
        temperature=args.fol_temperature,
        top_p=args.fol_top_p,
    )
    print(
        "[logiqa_fol_pilot] using FOL API "
        f"model={fol_api_config['model']} config={args.fol_api_config}",
        flush=True,
    )

    declaration_prompt = load_prompt("Z3DeclarationsGeneration1.txt")
    declaration_inputs = [item["question_text"] for item in trajectories]
    print(f"[logiqa_fol_pilot] generating {len(declaration_inputs)} FOL declarations", flush=True)
    declaration_outputs = fol_generator.generate(
        declaration_inputs,
        system=declaration_prompt,
        max_workers=args.fol_max_workers,
    )

    valid_declarations = 0
    for item, declaration_output in zip(trajectories, declaration_outputs):
        declaration_code = extract_python_code(declaration_output)
        item["declaration_code"] = declaration_code
        try:
            from verl.utils.fol_verifier import FOLVerifier

            FOLVerifier.validate_z3_declaration_code(declaration_code)
            valid_declarations += 1
        except Exception as exc:
            item["declaration_error"] = repr(exc)

    implication_prompt = load_prompt("Z3ImplicationConversion1.txt")
    implication_tasks: list[tuple[int, int]] = []
    implication_inputs: list[str] = []
    for trajectory_idx, item in enumerate(trajectories):
        if item["declaration_error"]:
            for step in item["steps"]:
                if step["format_correct"]:
                    step["fol_error"] = f"declaration_error: {item['declaration_error']}"
            continue
        for step_idx, step in enumerate(item["steps"]):
            if not step["format_correct"]:
                step["fol_error"] = "format_incorrect"
                continue
            reasoning_prefix = build_reasoning_prefix(extract_step_blocks(item["response"]), step_idx)
            full_input = (
                f"Question:\n{item['question_text']}\n\n"
                f"Z3 Declaration:\n{item['declaration_code']}\n\n"
                f"Reasoning steps:\n{reasoning_prefix}"
            )
            implication_tasks.append((trajectory_idx, step_idx))
            implication_inputs.append(full_input)

    print(
        f"[logiqa_fol_pilot] scoring {len(implication_tasks)} formatted step prefixes "
        f"from {valid_declarations} valid declarations",
        flush=True,
    )
    implication_outputs = fol_generator.generate(
        implication_inputs,
        system=implication_prompt,
        max_workers=args.fol_max_workers,
    )

    for (trajectory_idx, step_idx), implication_output in zip(implication_tasks, implication_outputs):
        item = trajectories[trajectory_idx]
        step = item["steps"][step_idx]
        implication_code = extract_python_code(implication_output)
        step["implication_code"] = implication_code
        try:
            parsed_steps = parse_python_logic_steps(implication_code)
            if not parsed_steps:
                step["fol_error"] = "no_parsed_steps"
                continue
            output, error = verify_fol_step(item["declaration_code"], parsed_steps[-1], timeout=args.z3_timeout)
            step["fol_output"] = output
            step["fol_error"] = error
            step["fol_score"] = 1.0 if output and any("SUCCESS_ENTAILED" in line for line in output) else 0.0
        except Exception as exc:
            step["fol_error"] = repr(exc)

    for item in trajectories:
        item["fol_correct_steps"] = sum(1 for step in item["steps"] if step["fol_score"] == 1.0)
        print(
            "[logiqa_fol_pilot] "
            f"sample_id={item['sample_id']} "
            f"fol_correct_steps={item['fol_correct_steps']} "
            f"total_steps={item['total_steps']} "
            f"answer_correct={item['answer_correct']}",
            flush=True,
        )

    total_steps = sum(item["total_steps"] for item in trajectories)
    format_correct_steps = sum(item["format_correct_steps"] for item in trajectories)
    fol_correct_steps = sum(item["fol_correct_steps"] for item in trajectories)
    answer_correct_count = sum(1 for item in trajectories if item["answer_correct"])
    metrics = {
        "format/total_steps": total_steps,
        "format/correct_steps": format_correct_steps,
        "format/ratio": (format_correct_steps / total_steps) if total_steps else 0.0,
        "fol/total_steps": total_steps,
        "fol/correct_steps": fol_correct_steps,
        "fol/ratio": (fol_correct_steps / total_steps) if total_steps else 0.0,
        "answer/correct_count": answer_correct_count,
        "answer/total": len(trajectories),
        "answer/accuracy": (answer_correct_count / len(trajectories)) if trajectories else 0.0,
        "fol/api_model": fol_api_config["model"],
        "fol/api_config": args.fol_api_config,
    }

    write_jsonl(output_dir / "generations.jsonl", trajectories)
    pd.DataFrame(trajectories).to_parquet(output_dir / "generations.parquet")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    run = init_wandb(args)
    log_wandb(run, metrics, trajectories)

    print(f"[logiqa_fol_pilot] saved outputs to {output_dir}", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
