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

from openai import OpenAI

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    #api_key="sk-Yw3l_G6NFtbwd14Mj30HXg", # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    #base_url="https://litellm.mybigai.ac.cn/",
    base_url="http://127.0.0.1:1029/v1",
    api_key="no-key-required"
)

    


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def generate_outline(question):
    prompt = """
        You are an expert in logical reasoning. Please analyze the above problem and provide an outline for solving it. You do not need to provide specific steps, just using one short sentence to summarize what this step should do. For example:<outline>\n\nStep 1:...;\n\nStep 2:...;\n\n...;\n\nStep n: Obtain the answer.(n is the index of the final step!)\n\n<\outline>
    """
    completion = client.chat.completions.create(
        model="judge-model",# deepseek-r1-250528 deepseek-v3-250324
        messages=[
            {
                "role": "user",
                "content": question +  prompt,
            },
        ],
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "lucasmccabe/logiqa"

    dataset = datasets.load_dataset(data_source, "default")

    # train_dataset = dataset["train"].select([i for i in range(2000)])
    # test_dataset = dataset["validation"].select([i for i in range(100)])

    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]

    instruction_following = 'Please strictly follow the outline to solve the given problem step by step, and each step can be split by "\n\n", and put the index of the correct answer within \\boxed{{}}.'



    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        option_mapping = ["A", "B", "C", "D","E", "F", "G", "H", "I", "J"]
        def process_fn(example, idx):
            context = example.pop("context")
            question_raw = example.pop("query")
            answer_raw = example.pop("options")# list
            solution = option_mapping[int(example.pop("correct_option"))]

            answers = "\n\n".join(["Option (" + option_mapping[i] +"):"+ answer_raw[i] + ".\n" for i in range(len(answer_raw))])
            question = context + "\n\n" +question_raw + "\n\n" + answers + "\n\n"

            question_with_outlines = generate_outline(question)
            print(question_with_outlines)
            question = question + question_with_outlines + "\n\n" + instruction_following
            print(len(question))
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
                "raw_prompt": context + "\n\n" +question_raw + "\n\n" + answers,
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": context + "\n\n" +question_raw + "\n\n" + answers
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
