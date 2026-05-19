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
Preprocess the OpenBookQA dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/openbookqa_action/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "allenai/openbookqa"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    validate_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    with open("./mcts_utils/prompts/Generation1.txt", "r", encoding="utf-8") as f:
        instruction_following = f.read()

    def make_map_fn(split):
        option_mapping = ["A", "B", "C", "D","E", "F", "G", "H", "I", "J"]
        def process_fn(example, idx):
            question_raw = example.pop("question_stem")
            answer_raw = example.pop("choices").pop("text")
            solution = example.pop("answerKey")

            answers = "\n\n".join(["(" + option_mapping[i] +")"+ answer_raw[i] + "." for i in range(len(answer_raw))])
            question = "## Task Instructions\n\n" + instruction_following + "\n\n" + "<Question>" + question_raw + "</Question>\n\n" + "<Options>" + answers + "</Options>"

            sample_id = f"openbookqa_{idx}"
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "logic",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "answer": solution,
                "raw_prompt": question,
                "sample_id": sample_id,
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    validate_dataset = validate_dataset.map(function=make_map_fn("validation"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    validate_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print("Save to :", local_dir)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
