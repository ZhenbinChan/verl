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
Preprocess the QA4MRE dataset to parquet format.
"""

import argparse
import os
import string
from collections import Counter

import datasets

from verl.utils.hdfs_io import copy, makedirs


INSTRUCTION_FOLLOWING = 'Please reason step by step with steps separated by "\n\n", and put the index of the correct answer within \\boxed{{}}.'


def load_qa4mre_dataset(dataset_name, fallback_dataset_name, config_name):
    try:
        return datasets.load_dataset(dataset_name, config_name)
    except Exception as primary_error:
        if not fallback_dataset_name or fallback_dataset_name == dataset_name:
            raise

        print(f"Failed to load {dataset_name}/{config_name}: {primary_error}")
        print(f"Falling back to {fallback_dataset_name}/{config_name}")
        return datasets.load_dataset(fallback_dataset_name, config_name)


def normalize_answer_options(answer_options):
    answer_ids = answer_options["answer_id"]
    answer_texts = answer_options["answer_str"]
    return [(str(answer_id), str(answer_text)) for answer_id, answer_text in zip(answer_ids, answer_texts, strict=True)]


def get_solution(answer_options, correct_answer_id, idx):
    for option_idx, (answer_id, _) in enumerate(answer_options):
        if answer_id == str(correct_answer_id):
            if option_idx >= len(string.ascii_uppercase):
                raise ValueError(f"QA4MRE sample {idx} has too many answer options: {len(answer_options)}")
            return string.ascii_uppercase[option_idx]

    raise ValueError(f"QA4MRE sample {idx} correct_answer_id={correct_answer_id} was not found in answer options")


def format_options(answer_options):
    if len(answer_options) > len(string.ascii_uppercase):
        raise ValueError(f"QA4MRE sample has too many answer options: {len(answer_options)}")

    return "\n\n".join(
        [f"Option ({string.ascii_uppercase[i]}): {answer_text}" for i, (_, answer_text) in enumerate(answer_options)]
    )


def make_map_fn(split, config_name):
    normalized_config_name = config_name.replace(".", "_")

    def process_fn(example, idx):
        answer_options = normalize_answer_options(example["answer_options"])
        solution = get_solution(answer_options, example["correct_answer_id"], idx)
        answers = format_options(answer_options)

        context = example["document_str"].strip()
        question_raw = example["question_str"].strip()
        raw_prompt = f"<Context>{context}</Context><Question>{question_raw}</Question><Options>{answers}</Options>"
        prompt = raw_prompt + "\n\n" + INSTRUCTION_FOLLOWING

        return {
            "data_source": "qa4mre",
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
            "sample_id": f"qa4mre_{normalized_config_name}_{idx}",
            "extra_info": {
                "split": split,
                "index": idx,
                "topic_id": example["topic_id"],
                "topic_name": example["topic_name"],
                "test_id": example["test_id"],
                "document_id": example["document_id"],
                "question_id": example["question_id"],
                "answer": solution,
                "answer_id": str(example["correct_answer_id"]),
                "answer_text": example["correct_answer_str"],
                "answer_option_ids": [answer_id for answer_id, _ in answer_options],
                "answer_options": [answer_text for _, answer_text in answer_options],
                "question": raw_prompt,
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="community-datasets/qa4mre")
    parser.add_argument("--fallback_dataset_name", default="qa4mre")
    parser.add_argument("--config_name", default="2013.main.EN")
    parser.add_argument("--split", default="train")
    parser.add_argument("--local_dir", default="./data/qa4mre/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    dataset = load_qa4mre_dataset(args.dataset_name, args.fallback_dataset_name, args.config_name)
    test_dataset = dataset[args.split].map(
        function=make_map_fn("test", args.config_name),
        with_indices=True,
        remove_columns=dataset[args.split].column_names,
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print("Save to :", local_dir)
    print("QA4MRE test answer distribution:", dict(Counter(test_dataset["answer"])))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
