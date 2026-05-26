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
Preprocess the local PubMedQA test set to parquet format.
"""

import argparse
import json
import os
from collections import Counter

import datasets

from verl.utils.hdfs_io import copy, makedirs


ANSWER_TO_OPTION = {
    "yes": "A",
    "no": "B",
    "maybe": "C",
}

OPTIONS = [
    ("A", "yes"),
    ("B", "no"),
    ("C", "maybe"),
]

INSTRUCTION_FOLLOWING = 'Please reason step by step with steps separated by "\n\n", and put the index of the correct answer within \\boxed{{}}.'


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_context(example):
    contexts = example.get("CONTEXTS") or []
    labels = example.get("LABELS") or []

    formatted_contexts = []
    for idx, context in enumerate(contexts):
        label = labels[idx] if idx < len(labels) else None
        if label:
            formatted_contexts.append(f"{label}: {context}")
        else:
            formatted_contexts.append(context)

    return "\n\n".join(formatted_contexts)


def format_options():
    return "\n\n".join([f"({option}): {answer}" for option, answer in OPTIONS])


def build_records(test_set, ground_truths):
    if set(test_set) != set(ground_truths):
        missing_ground_truth = sorted(set(test_set) - set(ground_truths))
        missing_examples = sorted(set(ground_truths) - set(test_set))
        raise ValueError(
            "PubMedQA test_set.json and test_ground_truth.json contain different PMIDs. "
            f"Missing ground truth: {missing_ground_truth[:5]}; missing examples: {missing_examples[:5]}"
        )

    data_source = "pubmedqa"
    option_text = format_options()
    records = []

    for idx, (pmid, example) in enumerate(test_set.items()):
        answer_text = str(ground_truths[pmid]).lower()
        if answer_text not in ANSWER_TO_OPTION:
            raise ValueError(f"Unsupported PubMedQA answer for PMID {pmid}: {ground_truths[pmid]}")

        final_decision = example.get("final_decision")
        if final_decision is not None and str(final_decision).lower() != answer_text:
            raise ValueError(
                f"PubMedQA final_decision does not match ground truth for PMID {pmid}: "
                f"{final_decision} != {answer_text}"
            )

        solution = ANSWER_TO_OPTION[answer_text]
        context = format_context(example)
        question_raw = example.get("QUESTION", "")
        raw_prompt = f"<Context>{context}</Context><Question>{question_raw}</Question><Options>{option_text}</Options>"
        prompt = raw_prompt + "\n\n" + INSTRUCTION_FOLLOWING

        records.append(
            {
                "data_source": data_source,
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
                "sample_id": f"pubmedqa_{pmid}",
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "pmid": pmid,
                    "answer": solution,
                    "answer_text": answer_text,
                    "question": raw_prompt,
                    "contexts": example.get("CONTEXTS") or [],
                },
            }
        )

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/pubmedqa_origin/data")
    parser.add_argument("--local_dir", default="./data/pubmedqa/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    test_set_path = os.path.join(args.data_dir, "test_set.json")
    ground_truth_path = os.path.join(args.data_dir, "test_ground_truth.json")

    test_set = load_json(test_set_path)
    ground_truths = load_json(ground_truth_path)
    test_records = build_records(test_set, ground_truths)
    test_dataset = datasets.Dataset.from_list(test_records)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print("Save to :", local_dir)
    print("PubMedQA test answer distribution:", dict(Counter(record["answer"] for record in test_records)))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
