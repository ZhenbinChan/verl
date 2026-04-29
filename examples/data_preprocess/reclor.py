"""
Preprocess the ReClor dataset to parquet format for verl.

The Hugging Face dataset viewer cannot cast all ReClor splits into one schema
because the official test split is unlabeled. This script loads train/val JSON
files directly and emits only labeled splits by default.
"""

import argparse
import os
from pathlib import Path

import datasets
from tqdm.auto import tqdm


PROMPT_DIR = Path(__file__).resolve().parents[2] / "verl" / "prompts"
DATA_FILES = {
    "train": "https://huggingface.co/datasets/voidful/ReClor/resolve/main/train.json",
    "validation": "https://huggingface.co/datasets/voidful/ReClor/resolve/main/val.json",
    "test": "https://huggingface.co/datasets/voidful/ReClor/resolve/main/test.json",
}
LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def _load_prompt_file(path_or_name: str) -> str:
    """Load a prompt txt from an absolute path or verl/prompts/."""
    p = Path(path_or_name)
    if not p.is_absolute():
        p = PROMPT_DIR / p
    return p.read_text(encoding="utf-8").strip()


def _load_json_split(split: str):
    return datasets.load_dataset("json", data_files={split: DATA_FILES[split]})[split]


def make_map_fn(split, format="xml", system_prompt=None, user_prompt_suffix=None, require_label=True):
    def process_fn(example, idx):
        context = example.get("context", "")
        question_text = example.get("question", "")
        options = example.get("answers", [])

        answer_idx = example.get("label")
        if answer_idx is None:
            if require_label:
                raise ValueError(f"ReClor split {split} example {idx} has no label")
            ground_truth = ""
        else:
            answer_idx = int(answer_idx)
            ground_truth = LABELS[answer_idx] if answer_idx < len(LABELS) else "A"

        options_str = "\n".join(
            f"Option ({LABELS[i]}): {opt}" for i, opt in enumerate(options) if i < len(LABELS)
        )
        instruction_following = (
            'Please reason step by step with steps separated by "\\n\\n", '
            "and put the letter of the correct option within \\boxed{{}}."
        )

        if format == "xml":
            prompt_content = (
                f"<Context>\n{context}\n</Context>\n\n"
                f"<Question>\n{question_text}\n</Question>\n\n"
                f"<Options>\n{options_str}\n</Options>\n\n"
                f"{instruction_following}"
            )
        else:
            prompt_content = (
                f"Context: {context}\n\n"
                f"Question: {question_text}\n\n"
                f"Options:\n{options_str}\n\n"
                f"{instruction_following}"
            )

        if user_prompt_suffix:
            prompt_content = f"{prompt_content}\n\n{user_prompt_suffix}"

        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": prompt_content})

        return {
            "data_source": "reclor",
            "prompt": prompt_messages,
            "ability": "logic",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "id_string": example.get("id_string", ""),
                "answer": ground_truth,
                "fol_context": context,
                "fol_question": question_text,
                "fol_options": options_str,
            },
        }

    return process_fn


def map_with_progress(dataset, split, format, system_prompt=None, user_prompt_suffix=None, require_label=True):
    mapper = make_map_fn(
        split,
        format,
        system_prompt=system_prompt,
        user_prompt_suffix=user_prompt_suffix,
        require_label=require_label,
    )
    rows = []
    for idx, example in enumerate(tqdm(dataset, desc=f"Processing {split}", unit="example")):
        rows.append(mapper(example, idx))
    return datasets.Dataset.from_list(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="./data/reclor_prompt_v2",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument("--format", default="xml", choices=["flat", "xml"], help="Prompt format.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of train samples to keep.")
    parser.add_argument(
        "--system_prompt_file",
        default=None,
        help="Path or bare filename under verl/prompts/ for the system prompt.",
    )
    parser.add_argument(
        "--user_prompt_file",
        default=None,
        help="Path or bare filename under verl/prompts/ to append to the user prompt.",
    )
    parser.add_argument(
        "--save_unlabeled_test",
        action="store_true",
        help="Also save the unlabeled official test split as test_unlabeled.parquet.",
    )
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    system_prompt = _load_prompt_file(args.system_prompt_file) if args.system_prompt_file else None
    user_prompt_suffix = _load_prompt_file(args.user_prompt_file) if args.user_prompt_file else None

    train_dataset = _load_json_split("train")
    val_dataset = _load_json_split("validation")

    if args.num_samples is not None and args.num_samples != -1:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.num_samples)))

    train_dataset = map_with_progress(
        train_dataset,
        "train",
        args.format,
        system_prompt=system_prompt,
        user_prompt_suffix=user_prompt_suffix,
        require_label=True,
    )
    val_dataset = map_with_progress(
        val_dataset,
        "validation",
        args.format,
        system_prompt=system_prompt,
        user_prompt_suffix=user_prompt_suffix,
        require_label=True,
    )

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))

    print(f"Dataset saved to {local_save_dir}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    if args.save_unlabeled_test:
        test_dataset = _load_json_split("test")
        test_dataset = map_with_progress(
            test_dataset,
            "test",
            args.format,
            system_prompt=system_prompt,
            user_prompt_suffix=user_prompt_suffix,
            require_label=False,
        )
        test_dataset.to_parquet(os.path.join(local_save_dir, "test_unlabeled.parquet"))
        print(f"Unlabeled test samples: {len(test_dataset)}")
