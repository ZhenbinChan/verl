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
Preprocess the LogiQA dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/home/chenzhb/Workspaces/verl/data/logiqa_action/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "lucasmccabe/logiqa"

    dataset = datasets.load_dataset(data_source, "default")

    # train_dataset = dataset["train"].select([i for i in range(2000)])
    # test_dataset = dataset["validation"].select([i for i in range(100)])

    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]

    with open("/home/chenzhb/Workspaces/verl/mcts_utils/prompts/Generation1.txt", "r", encoding="utf-8") as f:
        instruction_following = f.read()
    instruction_following = str(instruction_following)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        option_mapping = ["A", "B", "C", "D","E", "F", "G", "H", "I", "J"]
        def process_fn(example, idx):
            context = example.pop("context")
            question_raw = example.pop("query")
            answer_raw = example.pop("options")# list
            solution = option_mapping[int(example.pop("correct_option"))]

            answers = "\n\n".join(["Option (" + option_mapping[i] +"):"+ answer_raw[i] + ".\n" for i in range(len(answer_raw))])
            question = "<Context>" + context + "</Context>" + "<Question>" + question_raw + "</Question>" + "<Options>" + answers + "</Options>"
            # solution = extract_solution(answer_raw)

            # import pdb;pdb.set_trace()
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "answer": solution,
                "raw_prompt": "<Context>" + context + "</Context>" + "<Question>" + question_raw + "</Question>" + "<Options>" + answers + "</Options>",
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": context + question_raw 
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
