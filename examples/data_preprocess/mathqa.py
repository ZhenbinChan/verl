# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the local MathQA test sets to parquet format.
"""

import argparse
import json
import os
import re
from collections import Counter

import datasets

from verl.utils.hdfs_io import copy, makedirs


INSTRUCTION_FOLLOWING = 'Please reason step by step with steps separated by "\n\n", and put the index of the correct answer within \\boxed{{}}.'
OPTION_LABELS = ["a", "b", "c", "d", "e"]
OPTION_MAPPING = {label: label.upper() for label in OPTION_LABELS}
OPTION_RE = re.compile(r"(?i)([a-e])\s*\)")
DUPLICATE_OPTION_RE = re.compile(r"(?i)\b([a-e])\s*\)\s*\1\s*\)")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_option_text(label, text):
    text = text.strip(" ,")
    duplicate_label_re = re.compile(rf"(?i)^{label}\s*\)\s*")
    while True:
        cleaned = duplicate_label_re.sub("", text, count=1).strip(" ,")
        if cleaned == text:
            return cleaned
        text = cleaned


def parse_options(options_raw, idx):
    normalized_options = options_raw
    while True:
        cleaned_options = DUPLICATE_OPTION_RE.sub(r"\1 )", normalized_options)
        if cleaned_options == normalized_options:
            break
        normalized_options = cleaned_options

    matches = list(OPTION_RE.finditer(normalized_options))
    options = {}

    for match_idx, match in enumerate(matches):
        label = match.group(1).lower()
        if label in options:
            continue

        start = match.end()
        end = matches[match_idx + 1].start() if match_idx + 1 < len(matches) else len(normalized_options)
        options[label] = clean_option_text(label, normalized_options[start:end])

    missing_labels = [label for label in OPTION_LABELS if label not in options]
    empty_labels = [label for label in OPTION_LABELS if not options.get(label)]
    if missing_labels or empty_labels:
        raise ValueError(
            f"Failed to parse MathQA options for sample {idx}: "
            f"missing={missing_labels}, empty={empty_labels}, options={options_raw!r}"
        )

    return [(OPTION_MAPPING[label], options[label]) for label in OPTION_LABELS]


def format_options(options):
    return "\n\n".join([f"Option ({label}): {text}" for label, text in options])


def make_record(example, idx, split, data_source):
    options = parse_options(example["options"], idx)
    solution = OPTION_MAPPING[str(example["correct"]).lower()]
    question_raw = example["Problem"].strip()
    answers = format_options(options)
    raw_prompt = f"<Question>{question_raw}</Question><Options>{answers}</Options>"
    prompt = raw_prompt + "\n\n" + INSTRUCTION_FOLLOWING

    return {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "answer": solution,
        "raw_prompt": raw_prompt,
        "sample_id": f"mathqa_{split}_{idx}",
        "extra_info": {
            "split": split,
            "index": idx,
            "category": example.get("category"),
            "answer": solution,
            "rationale": example.get("Rationale"),
            "annotated_formula": example.get("annotated_formula"),
            "linear_formula": example.get("linear_formula"),
            "raw_options": example["options"],
            "options": [option_text for _, option_text in options],
            "question": raw_prompt,
        },
    }


def build_dataset(data_dir, filename, split, data_source):
    path = os.path.join(data_dir, filename)
    examples = load_json(path)
    records = [make_record(example, idx, split, data_source) for idx, example in enumerate(examples)]
    return datasets.Dataset.from_list(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/MathQA")
    parser.add_argument("--local_dir", default="./data/mathqa/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    outputs = [
        ("test.json", "test", "mathqa", "test.parquet"),
        ("challenge_test.json", "challenge_test", "mathqa_challenge", "challenge_test.parquet"),
    ]

    os.makedirs(args.local_dir, exist_ok=True)

    for filename, split, data_source, output_name in outputs:
        dataset = build_dataset(args.data_dir, filename, split, data_source)
        output_path = os.path.join(args.local_dir, output_name)
        dataset.to_parquet(output_path)

        print("Save to :", output_path)
        print(f"MathQA {split} answer distribution:", dict(Counter(dataset["answer"])))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
