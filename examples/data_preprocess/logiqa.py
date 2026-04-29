"""
Preprocess the LogiQA 1.0/2.0 dataset to parquet format for verl
"""

import argparse
import os
from pathlib import Path
import datasets

# Default location for prompt files
PROMPT_DIR = Path(__file__).resolve().parents[2] / "verl" / "prompts"


def _load_prompt_file(path_or_name: str) -> str:
    """Load a prompt txt. Accepts an absolute path or a bare filename
    resolved relative to verl/prompts/."""
    p = Path(path_or_name)
    if not p.is_absolute():
        p = PROMPT_DIR / p
    return p.read_text(encoding="utf-8").strip()

def make_map_fn(split, format='flat', system_prompt=None, user_prompt_suffix=None):
    def process_fn(example, idx):
        # Extract fields from LogiQA (works for both 1.0 and 2.0)
        context = example.get('context')
        # LogiQA 1.0 uses 'query', 2.0 uses 'question'
        question_text = example.get('question') or example.get('query')
        options = example.get('options')
        # LogiQA 1.0 uses 'correct_option', 2.0 uses 'answer'
        answer_idx = example.get('answer', example.get('correct_option'))
        if isinstance(answer_idx, str): # Handle string indices if any
             answer_idx = int(answer_idx)
        
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        options_str = "\n".join([f"Option ({labels[i]}): {opt}" for i, opt in enumerate(options) if i < len(labels)])
        
        instruction_following = 'Please reason step by step with steps separated by "\n\n", and put the letter of the correct option within \\boxed{{}}.'
        
        if format == 'xml':
            prompt_content = f"<Context>\n{context}\n</Context>\n\n<Question>\n{question_text}\n</Question>\n\n<Options>\n{options_str}\n</Options>\n\n{instruction_following}"
        else:
            prompt_content = f"Context: {context}\n\nQuestion: {question_text}\n\nOptions:\n{options_str}\n\n{instruction_following}"
        
        ground_truth = labels[answer_idx] if answer_idx < len(labels) else 'A'

        if user_prompt_suffix:
            prompt_content = f"{prompt_content}\n\n{user_prompt_suffix}"

        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": prompt_content})

        data = {
            "data_source": "logiqa",
            "prompt": prompt_messages,
            "ability": "logic",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": ground_truth,
                "fol_context": context,
                "fol_question": question_text,
                "fol_options": options_str,
            },
        }
        return data

    return process_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="./data/logiqa2k", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--subset", default="en", help="The subset (en/zh for v2, default for v1).")
    parser.add_argument("--version", type=int, default=1, choices=[1, 2], help="LogiQA version (1 or 2).")
    parser.add_argument("--format", default="flat", choices=["flat", "xml"], help="Prompt format.")
    parser.add_argument("--num_samples", type=int, default=2000, help="The number of training samples to keep.")
    parser.add_argument(
        "--system_prompt_file",
        default=None,
        help=(
            "Path (or bare filename under verl/prompts/) of a txt file to use as the system prompt. "
            "Example: logical_reasoning.txt"
        ),
    )
    parser.add_argument(
        "--user_prompt_file",
        default=None,
        help=(
            "Path (or bare filename under verl/prompts/) of a txt file to append to the user prompt. "
            "Example: extra_user_instructions.txt"
        ),
    )

    args = parser.parse_args()
    
    local_save_dir = os.path.expanduser(args.local_save_dir)
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)

    system_prompt = _load_prompt_file(args.system_prompt_file) if args.system_prompt_file else None
    user_prompt_suffix = _load_prompt_file(args.user_prompt_file) if args.user_prompt_file else None

    # Load dataset based on version
    if args.version == 1:
        data_source = "lucasmccabe/logiqa"
        dataset = datasets.load_dataset(data_source, "default")
        # 1.0 uses 'validation' as the dev set
        val_key = "validation"
    else:
        data_source = "baber/logiqa2"
        dataset = datasets.load_dataset(data_source, args.subset)
        val_key = "validation"

    train_dataset = dataset["train"]
    val_dataset = dataset[val_key]
    test_dataset = dataset["test"]

    if args.num_samples is not None and args.num_samples != -1:
        # train_dataset = train_dataset.shuffle(seed=42)
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.num_samples)))

    # Transform datasets
    train_dataset = train_dataset.map(function=make_map_fn("train", args.format, system_prompt=system_prompt, user_prompt_suffix=user_prompt_suffix), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("validation", args.format, system_prompt=system_prompt, user_prompt_suffix=user_prompt_suffix), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", args.format, system_prompt=system_prompt, user_prompt_suffix=user_prompt_suffix), with_indices=True)

    # Save to parquet
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Dataset saved to {local_save_dir}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
