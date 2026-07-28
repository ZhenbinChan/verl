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
The base class for Actor
"""

from abc import ABC, abstractmethod
from typing import Dict, Mapping, Optional

import torch

from verl import DataProto
from verl.trainer.ppo.sampling.base import STEP_TREERL_INITIAL_ROLLOUT_N_KEY, STEP_TREERL_REPEAT_TIMES_KEY

__all__ = [
    "BasePPOActor",
    "STEP_TREERL_INITIAL_ROLLOUT_N_KEY",
    "STEP_TREERL_REPEAT_TIMES_KEY",
    "resolve_ppo_mini_batch_size",
]


def resolve_ppo_mini_batch_size(
    configured_mini_batch_size: int,
    meta_info: Mapping,
    *,
    local_batch_size: Optional[int] = None,
    micro_batch_size: Optional[int] = None,
) -> int:
    """Resolve the local actor mini-batch size for a StepTreeRL-expanded batch.

    Actor worker configuration is normalized once using ``rollout.n``. StepTreeRL
    may replace those initial rollouts with a different, fixed number of selected
    traces. Only batches carrying the StepTreeRL metadata below are rescaled;
    ordinary PPO/GRPO batches retain the configured value exactly.
    """
    mini_batch_size = int(configured_mini_batch_size)
    if STEP_TREERL_REPEAT_TIMES_KEY not in meta_info:
        return mini_batch_size

    repeat_times = int(meta_info[STEP_TREERL_REPEAT_TIMES_KEY])
    initial_rollout_n = int(meta_info.get(STEP_TREERL_INITIAL_ROLLOUT_N_KEY, 0))
    if repeat_times <= 0 or initial_rollout_n <= 0:
        raise ValueError(
            "StepTreeRL PPO mini-batch scaling requires positive "
            f"{STEP_TREERL_REPEAT_TIMES_KEY} and {STEP_TREERL_INITIAL_ROLLOUT_N_KEY}, "
            f"got {repeat_times} and {initial_rollout_n}."
        )

    scaled_size = mini_batch_size * repeat_times
    if scaled_size % initial_rollout_n != 0:
        raise ValueError(
            "StepTreeRL PPO mini-batch scaling is not integral: "
            f"{mini_batch_size} * {repeat_times} / {initial_rollout_n}."
        )
    effective_mini_batch_size = scaled_size // initial_rollout_n
    if effective_mini_batch_size <= 0:
        raise ValueError(f"StepTreeRL effective PPO mini-batch size must be positive, got {effective_mini_batch_size}.")
    if local_batch_size is not None and int(local_batch_size) % effective_mini_batch_size != 0:
        raise ValueError(
            "StepTreeRL local actor batch must be divisible by the effective PPO mini-batch size: "
            f"{int(local_batch_size)} % {effective_mini_batch_size} != 0."
        )
    if micro_batch_size is not None and effective_mini_batch_size % int(micro_batch_size) != 0:
        raise ValueError(
            "StepTreeRL effective PPO mini-batch size must be divisible by the actor micro-batch size: "
            f"{effective_mini_batch_size} % {int(micro_batch_size)} != 0."
        )
    return effective_mini_batch_size


class BasePPOActor(ABC):
    def __init__(self, config):
        """The base class for PPO actor

        Args:
            config (DictConfig): a config passed to the PPOActor. We expect the type to be
                DictConfig (https://omegaconf.readthedocs.io/), but it can be any namedtuple in general.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute logits given a batch of data.

        Args:
            data (DataProto): a batch of data represented by DataProto. It must contain key ```input_ids```,
                ```attention_mask``` and ```position_ids```.

        Returns:
            DataProto: a DataProto containing the key ```log_probs```


        """
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> Dict:
        """Update the policy with an iterator of DataProto

        Args:
            data (DataProto): an iterator over the DataProto that returns by
                ```make_minibatch_iterator```

        Returns:
            Dict: a dictionary contains anything. Typically, it contains the statistics during updating the model
            such as ```loss```, ```grad_norm```, etc,.

        """
        pass
