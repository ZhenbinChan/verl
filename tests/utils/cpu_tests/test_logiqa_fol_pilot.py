from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from mcts_utils.nl2fol_lzy.pipeline import verify_fol_step
from mcts_utils.nl2fol_lzy.utils import parse_python_logic_steps
from scripts.logiqa_fol_pilot import (
    OpenAICompatibleFOLGenerator,
    answer_correct,
    build_api_request_kwargs,
    build_model_prompt,
    build_reasoning_prefix,
    extract_step_blocks,
    get_ground_truth,
    get_question_text,
    get_sample_id,
    load_fol_api_config,
    step_format_correct,
)


def test_logiqa_record_helpers_extract_question_answer_and_sample_id():
    record = {
        "raw_prompt": "<Context>ctx</Context><Question>q</Question><Options>A. one</Options>",
        "answer": "A",
        "sample_id": "logiqa_7",
    }

    assert get_question_text(record).startswith("<Context>ctx")
    assert get_ground_truth(record) == "A"
    assert get_sample_id(record, 0) == "logiqa_7"


def test_logiqa_record_helpers_fallback_to_extra_info():
    record = {
        "data_source": "logiqa",
        "reward_model": {"ground_truth": "B"},
        "extra_info": {
            "index": 3,
            "question": "<Context>ctx</Context><Question>q</Question>",
        },
    }

    assert get_question_text(record) == "<Context>ctx</Context><Question>q</Question>"
    assert get_ground_truth(record) == "B"
    assert get_sample_id(record, 0) == "logiqa_3"


def test_prompt_and_step_extraction_format_ratio_helpers():
    prompt = build_model_prompt("Instruction", "<Context>ctx</Context>")
    assert prompt == "Instruction\n\n<Context>ctx</Context>"

    response = """
<step>
<premise>A</premise>
<conclusion>A</conclusion>
</step>

<step>
<conclusion>missing premise</conclusion>
</step>

\\boxed{A}
"""
    steps = extract_step_blocks(response)

    assert len(steps) == 2
    assert step_format_correct(steps[0])
    assert not step_format_correct(steps[1])
    assert build_reasoning_prefix(steps, 0) == steps[0]
    assert build_reasoning_prefix(steps, 1) == "\n\n".join(steps)


def test_answer_correct_uses_logiqa_reward_parser():
    assert answer_correct("Reasoning\n\\boxed{C}", "C")
    assert answer_correct("The answer is B.", "B")
    assert not answer_correct("\\boxed{A}", "D")


def test_pipeline_verify_fol_step_success_and_failure():
    declaration = "a = Bool('a')"
    implication_code = """
premises_0 = [a]
conclusion_0 = a
premises_1 = [a]
conclusion_1 = Not(a)
"""
    parsed_steps = parse_python_logic_steps(implication_code)

    success_output, success_error = verify_fol_step(declaration, parsed_steps[0])
    failed_output, failed_error = verify_fol_step(declaration, parsed_steps[1])

    assert success_error == ""
    assert failed_error == ""
    assert success_output == ["SUCCESS_ENTAILED"]
    assert failed_output == ["FAILED_CONTRADICT"]


def test_load_fol_api_config_normalizes_deepseek_yaml(tmp_path):
    config_path = tmp_path / "deepseek.yaml"
    config_path.write_text(
        """
provider: openai_compatible
base_url: https://api.deepseek.com
api_key: test-key
model_name: deepseek-v4-pro
request_timeout: 180
bypass_env_proxy: true
default_args:
  reasoning_effort: high
extra_body:
  thinking:
    type: enabled
""",
        encoding="utf-8",
    )

    config = load_fol_api_config(str(config_path))

    assert config["base_url"] == "https://api.deepseek.com"
    assert config["api_key"] == "test-key"
    assert config["model"] == "deepseek-v4-pro"
    assert config["request_timeout"] == 180
    assert config["bypass_env_proxy"] is True
    assert config["default_args"] == {"reasoning_effort": "high"}
    assert config["extra_body"] == {"thinking": {"type": "enabled"}}


def test_build_api_request_kwargs_preserves_default_args_and_extra_body():
    kwargs = build_api_request_kwargs(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=4096,
        temperature=0.1,
        top_p=0.8,
        default_args={"reasoning_effort": "high"},
        extra_body={"thinking": {"type": "enabled"}},
    )

    assert kwargs["model"] == "deepseek-v4-pro"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["temperature"] == 0.1
    assert kwargs["top_p"] == 0.8
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_fol_generator_uses_api_request_shape_with_fake_client():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)

            class Message:
                content = "api-output"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    generator = OpenAICompatibleFOLGenerator.__new__(OpenAICompatibleFOLGenerator)
    generator.model = "deepseek-v4-pro"
    generator.default_args = {"reasoning_effort": "high"}
    generator.extra_body = {"thinking": {"type": "enabled"}}
    generator.max_tokens = 123
    generator.temperature = 0.2
    generator.top_p = 0.9
    generator.client = FakeClient()

    output = generator.generate_one(user="Question", system="System")

    assert output == "api-output"
    assert calls[0]["model"] == "deepseek-v4-pro"
    assert calls[0]["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Question"},
    ]
    assert calls[0]["reasoning_effort"] == "high"
    assert calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}
