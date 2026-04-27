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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


class NaivePlusRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if "rm_scores" in data.batch.keys():
        if False:
            if return_dict:
                prompt = []
                gt = []
                response = []
                outcome_reward = []
                reward_extra_info = {}

                for i in range(len(data)):
                    data_item = data[i]  # DataProtoItem
                    prompt_ids = data_item.batch["prompts"]
                    prompt_length = prompt_ids.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                    response_ids = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_ids = response_ids[:valid_response_length]
                    # decode
                    prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch.get("extra_info", None)

                    score = data.batch["rm_scores"][i].sum(-1)

                    outcome_reward.append(float(score))# 每个 sample的最终reward
                    prompt.append(prompt_str)# 每个sample 的prompt
                    gt.append(ground_truth)# 每个sample 的gt
                    response.append(response_str)# 每个sample 的response

                    print("#"*60 + "\n" + f"[Response] {response_str} \n\n" + f"[GroundTrhth] {ground_truth} \n\n" + f"[Reward] {score} \n\n" + "#"*60)

                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": reward_extra_info,
                    "prompt": prompt,
                    "ground_truth": gt,
                    "response": response,
                    "outcome_reward": outcome_reward
                    }
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        prompt = []
        gt = []
        response = []
        outcome_reward = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )


            outcome_reward.append(score)# 每个 sample的最终reward
            prompt.append(prompt_str)# 每个sample 的prompt
            gt.append(ground_truth)# 每个sample 的gt
            response.append(response_str)# 每个sample 的response

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # reward_tensor[i, valid_response_length - 1] = reward
            reward_tensor[i, :valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "prompt": prompt,
                "ground_truth": gt,
                "response": response,
                "outcome_reward": outcome_reward
            }
        else:
            return reward_tensor
