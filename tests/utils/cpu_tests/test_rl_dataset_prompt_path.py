# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import sys
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

sys.modules.setdefault("ray", MagicMock())

from verl.utils.dataset.rl_dataset import RLHFDataset


def test_rlhf_dataset_prompt_path_missing_file_raises():
    config = OmegaConf.create({"prompt_path": "/tmp/definitely_missing_prompt_instruction.txt"})

    with patch.object(RLHFDataset, "_download", return_value=None), patch.object(
        RLHFDataset, "_read_files_and_tokenize", return_value=None
    ):
        with pytest.raises(FileNotFoundError):
            RLHFDataset(data_files="unused.parquet", tokenizer=None, config=config)


def test_rlhf_dataset_prompt_path_none_is_allowed():
    config = OmegaConf.create({"prompt_path": None})

    with patch.object(RLHFDataset, "_download", return_value=None), patch.object(
        RLHFDataset, "_read_files_and_tokenize", return_value=None
    ):
        dataset = RLHFDataset(data_files="unused.parquet", tokenizer=None, config=config)

    assert dataset.prompt_instruction is None
