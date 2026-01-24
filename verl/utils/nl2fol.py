"""Utilities for converting natural language problems to FOL-ready inputs.

This module provides a reusable GPT chat helper and a preprocessing function
that formats the premise-extraction prompt and returns the model output. The
actual FOL translation can build on top of these helpers with different
prompts.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence, Union

from openai import OpenAI

PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts"
PREMISE_EXTRACTION_PROMPT = PROMPT_ROOT / "premise_extraction.txt"
DECLARATION_PROMPT = PROMPT_ROOT / "extract_declaration.txt"
TRANSLATE_FOL_PROMPT = PROMPT_ROOT / "translate2fol.txt"


import re
def extract_plain_text_block(text):
    pattern = r"```plain_text\s+([\s\S]*?)```"
    # 搜索匹配项
    match = re.search(pattern, text)
    if match:
        # group(1) 返回第一个括号内匹配的内容，即我们需要的部分
        return match.group(1).strip()
    else:
        return "CAN: 未找到匹配的内容"

def _ensure_text(value: Optional[str]) -> str:
	return value if value is not None else ""


def load_prompt(path: Optional[Union[str, Path]] = None) -> str:
	"""Load a prompt file as text.

	Args:
		path: Optional custom path. Defaults to `premise_extraction.txt`.
	"""
	prompt_path = Path(path) if path else PREMISE_EXTRACTION_PROMPT
	return prompt_path.read_text(encoding="utf-8").strip()


def _fill_prompt(template: str, context: str, question: str, options: Optional[Union[str, Sequence[str]]]) -> str:
	return (
		template.replace("{{Context}}", _ensure_text(context).strip())
		.replace("{{Question}}", _ensure_text(question).strip())
		.replace("{{Options}}", _format_options(options))
	)


def call_gpt_api(
	prompt: str,
	*,
	model: str = "gpt-4o-mini",
	temperature: float = 0.0,
	max_tokens: int = 1024,
	api_key: Optional[str] = None,
	base_url: Optional[str] = None,
	system_prompt: Optional[str] = None,
	client: Optional[OpenAI] = None,
) -> str:
	"""Call the GPT chat completion API with a user prompt.

	Args:
		prompt: The user message content.
		model: Chat model name.
		temperature: Sampling temperature.
		max_tokens: Max tokens to generate.
		api_key: Optional API key (falls back to OPENAI_API_KEY env var).
		base_url: Optional base URL (e.g., self-hosted OpenAI-compatible endpoint).
		system_prompt: Optional system message prepended to the chat.

	Returns:
		The text content of the first choice (empty string if missing).
	"""

	messages = []
	if system_prompt:
		messages.append({"role": "system", "content": system_prompt})
	messages.append({"role": "user", "content": prompt})

	client = client or OpenAI(api_key=api_key, base_url=base_url)
	completion = client.chat.completions.create(
		model=model,
		messages=messages,
		temperature=temperature,
		max_tokens=max_tokens,
	)
	return completion.choices[0].message.content or ""


def call_qwen_api(
	prompt: str,
	*,
	model: str = "qwen-plus",
	temperature: float = 0.0,
	max_tokens: int = 1024,
	api_key: Optional[str] = None,
	base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
	system_prompt: Optional[str] = None,
	client: Optional[OpenAI] = None,
) -> str:
	"""Call Tongyi Qianwen via OpenAI-compatible endpoint.

	Args mirror `call_gpt_api`, but default to qwen models and dashscope base URL.
	API key can be provided explicitly or via DASHSCOPE_API_KEY.
	"""

	messages = []
	if system_prompt:
		messages.append({"role": "system", "content": system_prompt})
	messages.append({"role": "user", "content": prompt})

	client = client or OpenAI(api_key=api_key or os.getenv("DASHSCOPE_API_KEY"), base_url=base_url)
	completion = client.chat.completions.create(
		model=model,
		messages=messages,
		temperature=temperature,
		max_tokens=max_tokens,
	)
	return completion.choices[0].message.content or ""


def _format_options(options: Optional[Union[str, Sequence[str]]]) -> str:
	if options is None:
		return ""
	if isinstance(options, str):
		return options.strip()
	return "\n".join(str(opt).strip() for opt in options)


def preprocess_premises(
	*,
	context: str,
	question: str,
	options: Optional[Union[str, Sequence[str]]] = None,
	model: str = "gpt-4o-mini",
	temperature: float = 0.0,
	max_tokens: int = 1024,
	api_key: Optional[str] = None,
	base_url: Optional[str] = None,
	system_prompt: Optional[str] = None,
	provider: str = "openai",
	client: Optional[OpenAI] = None,
) -> str:
	"""Extract premises using the bundled premise-extraction prompt.

	This fills the template at `prompts/premise_extraction.txt` with the given
	fields, sends it to the GPT API, and returns the raw model output.
	"""

	prompt_template = load_prompt(PREMISE_EXTRACTION_PROMPT)
	filled_prompt = _fill_prompt(prompt_template, context, question, options)

	caller = call_qwen_api if provider == "qwen" else call_gpt_api
	return caller(
		filled_prompt,
		model=model,
		temperature=temperature,
		max_tokens=max_tokens,
		api_key=api_key,
		base_url=base_url,
		system_prompt=system_prompt,
		client=client,
	)


def translate_to_fol(
	*,
	declarations: str,
	sentences: str,
	model: str = "gpt-4o-mini",
	temperature: float = 0.0,
	max_tokens: int = 1024,
	api_key: Optional[str] = None,
	base_url: Optional[str] = None,
	system_prompt: Optional[str] = None,
	provider: str = "openai",
	client: Optional[OpenAI] = None,
) -> str:
	"""Translate natural language sentences to FOL using `translate2fol.txt` prompt.

	Args:
	    declarations: Text block describing declarations (EnumSort/Function etc.).
	    sentences: Natural language sentences to translate.
	    provider/model/...: Same as other helpers; can reuse shared client.
	"""

	prompt_template = load_prompt(TRANSLATE_FOL_PROMPT)
	filled_prompt = prompt_template + "\n" + _ensure_text(declarations).strip() + "\n" + _ensure_text(sentences).strip()

	caller = call_qwen_api if provider == "qwen" else call_gpt_api
	return caller(
		filled_prompt,
		model=model,
		temperature=temperature,
		max_tokens=max_tokens,
		api_key=api_key,
		base_url=base_url,
		system_prompt=system_prompt,
		client=client,
	)


def extract_declaration(
	*,
	context: str,
	question: str,
	options: Optional[Union[str, Sequence[str]]] = None,
	model: str = "gpt-4o-mini",
	temperature: float = 0.0,
	max_tokens: int = 1024,
	api_key: Optional[str] = None,
	base_url: Optional[str] = None,
	system_prompt: Optional[str] = None,
	provider: str = "openai",
	client: Optional[OpenAI] = None,
) -> str:
	"""Extract declarations using `extract_declaration.txt` prompt."""

	prompt_template = load_prompt(DECLARATION_PROMPT)
	filled_prompt = _fill_prompt(prompt_template, context, question, options)

	caller = call_qwen_api if provider == "qwen" else call_gpt_api
	return caller(
		filled_prompt,
		model=model,
		temperature=temperature,
		max_tokens=max_tokens,
		api_key=api_key,
		base_url=base_url,
		system_prompt=system_prompt,
		client=client,
	)


def preprocess_and_extract_declaration(
	*,
	context: str,
	question: str,
	options: Optional[Union[str, Sequence[str]]] = None,
	model_premise: str = "gpt-4o-mini",
	provider: str = "openai",
	temperature: float = 0.0,
	max_tokens: int = 1024,
	api_key: Optional[str] = None,
	base_url: Optional[str] = None,
	system_prompt: Optional[str] = None,
	client: Optional[OpenAI] = None,
) -> str:
	"""Run only the preprocessing stage and return its result.

	Declaration extraction should be performed separately via `extract_declaration`,
	optionally reusing the same client/model created by the caller.
	"""

	caller = call_qwen_api if provider == "qwen" else call_gpt_api
	prem_template = load_prompt(PREMISE_EXTRACTION_PROMPT)
	prem_prompt = _fill_prompt(prem_template, context, question, options)
	return caller(
		prem_prompt,
		model=model_premise,
		temperature=temperature,
		max_tokens=max_tokens,
		api_key=api_key,
		base_url=base_url,
		system_prompt=system_prompt,
		client=client,
	)


def _build_cli() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Premise / declaration extraction")
	parser.add_argument("--provider", choices=["openai", "qwen"], default="qwen")
	parser.add_argument("--model", default=None, help="Model name or path (shared)")
	parser.add_argument("--model-premise", default=None, help="Model for premise stage (if different)")
	parser.add_argument("--model-decl", default=None, help="Model for declaration stage (if different)")
	parser.add_argument("--task", choices=["premise", "declaration", "translate"], default="premise")
	parser.add_argument("--temperature", type=float, default=0.6)
	parser.add_argument("--max-tokens", type=int, default=1024)
	parser.add_argument("--api-key", default="sk-094e8ca337e14ac690bde26268c06667")
	parser.add_argument("--base-url", default=None)
	parser.add_argument("--system-prompt", default=None)
	parser.add_argument("--context", default=None, help="Context text (premise/declaration)")
	parser.add_argument("--question", default=None, help="Question text (premise/declaration)")
	parser.add_argument("--options", nargs="*", default=None, help="Optional list of options (premise/declaration)")
	parser.add_argument("--declarations", default=None, help="Declaration text for translate task")
	parser.add_argument("--sentences", default=None, help="Sentences to translate for translate task")
	parser.add_argument("--no-defaults", action="store_true", help="Do not fall back to demo context/question/options")
	return parser.parse_args()



def main():
	context = """ A law firm has eight partners, namely Gregg, Hodges, Ivan, James, King, MacNeil, Nader and Owens. From 1961 to 1968, a partner joined the firm each year. Hodges joined the firm before Nader. King joined the firm before James. Nader and James joined the firm before Gregg. Nader joined the firm before Owens.James joined the firm before MacNeil. Gregg joined the firm before Ivan. """
	question = "Which of the following cannot be true?"
	options = """A.Hodges joined the law firm in 1961.\nB.Hodges joined the law firm in 1963.\nC.Gregg joined the law firm in 1964.\nD.MacNeil joined the law firm in 1964.\nE.Owens joined the law firm in 1964."""

	args = _build_cli()

	if args.provider == "qwen":
		default_model = "qwen-plus"
		api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
		base_url = args.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
	else:
		default_model = "gpt-4o-mini"
		api_key = args.api_key or os.getenv("OPENAI_API_KEY")
		base_url = args.base_url

	model_shared = args.model or default_model
	client = OpenAI(api_key=api_key, base_url=base_url) if (api_key or base_url) else None

	# Step 1: preprocess (returns premise extraction)
	premise_text = preprocess_premises(
		context=context,
		question=question,
		options=options,
		model=args.model_premise or model_shared,
		temperature=args.temperature,
		max_tokens=args.max_tokens,
		api_key=api_key,
		base_url=base_url,
		system_prompt=args.system_prompt,
		provider=args.provider,
		client=client,
	)
	print("=== Premise Extraction Output ===")
	print(premise_text)

	# Step 2: declaration extraction, using the preprocessed text as context
	declaration = extract_declaration(
		context=premise_text,
		question=question,
		options=options,
		model=args.model_decl or model_shared,
		temperature=args.temperature,
		max_tokens=args.max_tokens,
		api_key=api_key,
		base_url=base_url,
		system_prompt=args.system_prompt,
		provider=args.provider,
		client=client,
	)
	print("=== Declaration Extraction Output ===")
	declaration = extract_plain_text_block(declaration)
	print(declaration)

	# Step 3: Translate to FOL
	fol = translate_to_fol(
		declarations=declaration,
		# sentences="At no meal does Vladimir eat the same kind of food as Wendy",
		sentences=premise_text,
		model=model_shared,
		temperature=args.temperature,
		max_tokens=args.max_tokens,
		api_key=api_key,
		base_url=base_url,
		system_prompt=args.system_prompt,
		provider=args.provider,
		client=client,
	)
	print(fol)
	constraints = extract_plain_text_block(fol)
	from verl.utils.fol_to_python_converter import convert_and_execute_fol
	code, result, error = convert_and_execute_fol(declaration, constraints)

	if result:
		print("Execute successfully!")
		print("\nExecution result:")
		for line in result:
			print(f"  {line}")
	else:
		print(f"Execute Failed: {error}")

__all__ = [
	"load_prompt",
	"call_gpt_api",
	"call_qwen_api",
	"preprocess_premises",
	"extract_declaration",
	"preprocess_and_extract_declaration",
	"translate_to_fol",
]


if __name__ == "__main__":
	main()
