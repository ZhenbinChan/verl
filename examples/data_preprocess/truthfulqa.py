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
Preprocess the TruthfulQA MC1 validation split to parquet format.
"""

import argparse
import os
import string
from collections import Counter

import datasets

from verl.utils.hdfs_io import copy, makedirs


INSTRUCTION_FOLLOWING = 'Please reason step by step with steps separated by "\n\n", and put the index of the correct answer within \\boxed{{}}.'


def get_solution(labels, idx):
    positive_indices = [i for i, label in enumerate(labels) if int(label) == 1]
    if len(positive_indices) != 1:
        raise ValueError(f"TruthfulQA MC1 sample {idx} should have exactly one correct answer, got labels={labels}")

    answer_idx = positive_indices[0]
    if answer_idx >= len(string.ascii_uppercase):
        raise ValueError(f"TruthfulQA MC1 sample {idx} has too many choices: {len(labels)}")

    return string.ascii_uppercase[answer_idx]


def format_options(choices):
    if len(choices) > len(string.ascii_uppercase):
        raise ValueError(f"TruthfulQA MC1 has too many choices: {len(choices)}")

    return "\n\n".join([f"Option ({string.ascii_uppercase[i]}): {choice}" for i, choice in enumerate(choices)])


def make_map_fn(split):
    def process_fn(example, idx):
        question_raw = example["question"]
        mc1_targets = example["mc1_targets"]
        answer_raw = mc1_targets["choices"]
        labels = mc1_targets["labels"]
        solution = get_solution(labels, idx)

        answers = format_options(answer_raw)
        raw_prompt = f"<Question>{question_raw}</Question>\n\n<Options>{answers}</Options>"
        question = raw_prompt + "\n\n" + INSTRUCTION_FOLLOWING
        sample_id = f"truthfulqa_{idx}"

        return {
            "data_source": "truthfulqa",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "logic",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "answer": solution,
            "raw_prompt": raw_prompt,
            "sample_id": sample_id,
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": solution,
                "question": raw_prompt,
                "choices": answer_raw,
                "labels": labels,
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/truthfulqa/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    dataset = datasets.load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    test_dataset = dataset["validation"].map(
        function=make_map_fn("test"),
        with_indices=True,
        remove_columns=dataset["validation"].column_names,
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print("Save to :", local_dir)
    print("TruthfulQA test answer distribution:", dict(Counter(test_dataset["answer"])))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
