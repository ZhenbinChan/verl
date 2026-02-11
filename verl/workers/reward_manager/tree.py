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


class TreeRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """Compute reward tensor.

        If `step_rewards` is provided per sample (list of floats), it will be broadcast to
        the corresponding step token spans (split by blank line "\n\n"). Otherwise falls
        back to the original single-outcome reward logic.
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        prompt = []
        gt = []
        response = []
        outcome_reward = []
        import pdb;pdb.set_trace()
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
            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, None)
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            step_rewards = data_item.non_tensor_batch.get("step_rewards")

            if step_rewards is not None:
                # use provided step rewards, broadcast to token spans split by blank line
                segments = response_str.split("\n\n") if response_str else [""]
                spans = []
                cursor = 0
                for seg in segments:
                    seg_tokens = self.tokenizer.encode(seg, add_special_tokens=False)
                    start, end = cursor, cursor + len(seg_tokens)
                    spans.append((start, end))
                    cursor = end

                for (s, e), r in zip(spans, step_rewards):
                    start = max(0, s)
                    end = max(start, min(e, valid_response_length))
                    if start < end:
                        reward_tensor[i, start:end] = float(r)

                # outcome reward: sum of step rewards
                score = sum(step_rewards)
            else:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                if isinstance(score, dict):
                    reward = score["score"]
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                # place outcome reward at last valid token
                if valid_response_length > 0:
                    reward_tensor[i, valid_response_length - 1] = reward

            outcome_reward.append(score)
            prompt.append(prompt_str)
            gt.append(ground_truth)
            response.append(response_str)

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
                "outcome_reward": outcome_reward,
            }
        else:
            return reward_tensor
