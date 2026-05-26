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
Preprocess the GPQA dataset to parquet format.
"""

import argparse
import os
import random
from collections import Counter

import datasets

from verl.utils.hdfs_io import copy, makedirs


INSTRUCTION_FOLLOWING = 'Please reason step by step with steps separated by "\n\n", and put the index of the correct answer within \\boxed{{}}.'
OPTION_MAPPING = ["A", "B", "C", "D"]
DEFAULT_CONFIGS = ["gpqa_diamond", "gpqa_main"]


def deterministic_shuffle_options(example, config_name, idx, seed):
    options = [
        {"text": str(example["Correct Answer"]).strip(), "is_correct": True},
        {"text": str(example["Incorrect Answer 1"]).strip(), "is_correct": False},
        {"text": str(example["Incorrect Answer 2"]).strip(), "is_correct": False},
        {"text": str(example["Incorrect Answer 3"]).strip(), "is_correct": False},
    ]

    rng = random.Random(f"{seed}:{config_name}:{idx}")
    rng.shuffle(options)
    return options


def get_solution(options, idx):
    correct_indices = [i for i, option in enumerate(options) if option["is_correct"]]
    if len(correct_indices) != 1:
        raise ValueError(f"GPQA sample {idx} should have exactly one correct answer, got {len(correct_indices)}")

    return OPTION_MAPPING[correct_indices[0]]


def format_options(options):
    return "\n\n".join([f"Option ({OPTION_MAPPING[i]}): {option['text']}" for i, option in enumerate(options)])


def make_map_fn(split, config_name, seed):
    def process_fn(example, idx):
        options = deterministic_shuffle_options(example, config_name, idx, seed)
        solution = get_solution(options, idx)
        answers = format_options(options)

        question_raw = str(example["Question"]).strip()
        raw_prompt = f"<Question>{question_raw}</Question><Options>{answers}</Options>"
        prompt = raw_prompt + "\n\n" + INSTRUCTION_FOLLOWING

        return {
            "data_source": config_name,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "ability": "logic",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "answer": solution,
            "raw_prompt": raw_prompt,
            "sample_id": f"gpqa_{config_name}_{idx}",
            "extra_info": {
                "split": split,
                "index": idx,
                "config": config_name,
                "record_id": example.get("Record ID"),
                "domain": example.get("High-level domain"),
                "subdomain": example.get("Subdomain"),
                "answer": solution,
                "answer_text": str(example["Correct Answer"]).strip(),
                "incorrect_answers": [
                    str(example["Incorrect Answer 1"]).strip(),
                    str(example["Incorrect Answer 2"]).strip(),
                    str(example["Incorrect Answer 3"]).strip(),
                ],
                "options": [option["text"] for option in options],
                "seed": seed,
                "question": raw_prompt,
            },
        }

    return process_fn


def build_dataset(dataset_name, config_name, split, seed):
    dataset = datasets.load_dataset(dataset_name, config_name)
    if split not in dataset:
        raise ValueError(f"Split {split!r} was not found in {dataset_name}/{config_name}. Available splits: {list(dataset.keys())}")

    return dataset[split].map(
        function=make_map_fn("test", config_name, seed),
        with_indices=True,
        remove_columns=dataset[split].column_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="Idavidrein/gpqa")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--local_dir", default="./data/gpqa/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    for config_name in args.configs:
        test_dataset = build_dataset(args.dataset_name, config_name, args.split, args.seed)

        config_dir = os.path.join(local_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        test_dataset.to_parquet(os.path.join(config_dir, "test.parquet"))

        print("Save to :", config_dir)
        print(f"GPQA {config_name} test answer distribution:", dict(Counter(test_dataset["answer"])))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
