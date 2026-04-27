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
Preprocess the Reclor dataset to parquet format
"""

import argparse
import json
import os

from verl.utils.hdfs_io import copy, makedirs


def load_jsonl(file_path):
    """Load JSONL file and return a list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/home/chenzhb/Workspaces/verl/data/logiqa2_action/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "logiqa2"

    # Load data from local JSONL files
    train_data = load_jsonl("/home/chenzhb/Workspaces/verl/data/logiqa2_ori/train.txt")
    dev_data = load_jsonl("/home/chenzhb/Workspaces/verl/data/logiqa2_ori/dev.txt")
    test_data = load_jsonl("/home/chenzhb/Workspaces/verl/data/logiqa2_ori/test.txt")

    # Create datasets-like objects from the data
    from datasets import Dataset

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    with open("/home/chenzhb/Workspaces/verl/mcts_utils/prompts/Generation1.txt", "r", encoding="utf-8") as f:
        instruction_following = f.read()
    instruction_following = str(instruction_following)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        option_mapping = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        def process_fn(example, idx):
            # Local data format: text=context, question, options=list, answer=int (index)
            context = example.pop("text")
            question_raw = example.pop("question")
            options_raw = example.pop("options")  # list of option strings
            answer_idx = example.pop("answer")  # int, index of correct answer
            solution = option_mapping[int(answer_idx)]

            answers = "\n\n".join(["Option (" + option_mapping[i] + ") " + options_raw[i] + ".\n" for i in range(len(options_raw))])
            question = instruction_following + "<Context>" + context + "</Context>" + "<Question>" + question_raw + "</Question>" + "<Options>" + answers + "</Options>"

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
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
