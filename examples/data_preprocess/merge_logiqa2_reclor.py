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
Merge LogiQA2 and Reclor datasets into a combined training set.
Train: LogiQA2 + Reclor
Test: Reclor
"""

import os
import datasets


if __name__ == "__main__":
    output_dir = "/home/chenzhb/Workspaces/verl/data/logiqa2_reclor_action/"
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets (using _action versions which have letter-mapped answers)
    logiqa2_train = datasets.load_dataset("parquet", data_files="/home/chenzhb/Workspaces/verl/data/logiqa2_action/train.parquet")["train"]
    reclor_train = datasets.load_dataset("parquet", data_files="/home/chenzhb/Workspaces/verl/data/reclor_action/train.parquet")["train"]
    reclor_test = datasets.load_dataset("parquet", data_files="/home/chenzhb/Workspaces/verl/data/reclor_action/test.parquet")["train"]

    
    # Merge train sets
    combined_train = datasets.concatenate_datasets([logiqa2_train, reclor_train])

    # Add sample_id for tracking
    def add_sample_id(example, idx):
        sample_id = f"{example['data_source']}_{example.get('extra_info', {}).get('index', idx)}"
        return {"sample_id": sample_id}

    combined_train = combined_train.map(add_sample_id, with_indices=True)
    # import pdb;pdb.set_trace()
    # Save
    combined_train.to_parquet(os.path.join(output_dir, "train.parquet"))
    reclor_test.to_parquet(os.path.join(output_dir, "test.parquet"))

    print(f"Combined train: {len(combined_train)} samples (logiqa2: {len(logiqa2_train)}, reclor: {len(reclor_train)})")
    print(f"Test: {len(reclor_test)} samples")
    print(f"Output: {output_dir}")