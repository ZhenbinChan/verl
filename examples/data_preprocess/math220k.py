"""
Preprocess the OpenR1-Math-220K dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from datasets import DatasetDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math220k')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = '/home/chenzhb/Workspaces/Datasets/OpenR1-Math-220k'

    dataset = datasets.load_dataset(data_source, 'default')

    # OpenR1-Math-220K does not have a test set, so we split the train set into train and test
    train_valid_split = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=42)

    train_dataset = train_valid_split['train'].select([i for i in range(500)])
    test_dataset = train_valid_split['test'].select([i for i in range(50)])


    instruction_following = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('problem')

            question = question_raw + ' ' + instruction_following

            answer_raw = example['solution']
            solution = example['answer']
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
