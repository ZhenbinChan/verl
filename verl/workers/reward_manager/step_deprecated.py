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

import random
import re
from collections import defaultdict
from typing_extensions import deprecated

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

@deprecated(version="0.1.0", reason="Use verl.experimental.reward_loop.reward_manager.step instead")
class StepRewardManager:
    """The Step Reward Manager that computes both outcome and process rewards."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", step_reward_type="random") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.step_reward_type = step_reward_type

    def _split_response_into_steps(self, response_tensor):
        """Split a response tensor into steps and return (step_text, token_span)."""
        resp_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
        segments = resp_text.split("\n\n") if resp_text else [""]
        steps = []
        cursor = 0
        for seg in segments:
            # We add \n\n back to length calculation if it's not the last segment
            # Actually simplest is just to encode the segment
            seg_tokens = self.tokenizer.encode(seg, add_special_tokens=False) if self.tokenizer is not None else []
            start, end = cursor, cursor + len(seg_tokens)
            # Adjust end to not exceed response_tensor length if encoding mismatch
            end = min(end, response_tensor.size(0))
            steps.append((seg, (start, end)))
            cursor = end
            
            # Account for \n\n tokens
            newline_tokens = self.tokenizer.encode("\n\n", add_special_tokens=False) if self.tokenizer is not None else []
            cursor += len(newline_tokens)

        return steps

    def _compute_step_reward(self, step_text: str, prompt_text: str, step_content: list[str]) -> float:
        """Compute the step-level process reward."""
        if self.step_reward_type == "random" or self.step_reward_type == "fol":
            # Support FOL logic later, fallback to random for now
            return float(random.randint(0, 1))
        elif self.step_reward_type == "format":
            # Simple format check (e.g., must contain some action tags)
            score = 0.0
            if re.search(r"<Action>.*</Action>", step_text, re.DOTALL):
                score += 1.0
            return score
        else:
            return 0.0

    def __call__(self, data: DataProto, return_dict=False):
        """Compute outcome and process rewards"""
        outcome_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        process_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
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

            # 1. Compute Outcome Reward
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            outcome_reward.append(score)
            prompt.append(prompt_str)
            gt.append(ground_truth)
            response.append(response_str)

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            outcome_reward_tensor[i, valid_response_length - 1] = reward

            # 2. Compute Process Reward (Step-level)
            if self.step_reward_type != "none":
                steps = self._split_response_into_steps(valid_response_ids)
                step_content = []
                for step_text, (start, end) in steps:
                    step_content.append(step_text)
                    step_score = self._compute_step_reward(
                        step_text=step_text,
                        prompt_text=prompt_str,
                        step_content=step_content
                    )
                    # Assign process reward to the end of the step
                    # Make sure not to exceed sequence length
                    pos = max(0, min(end - 1, valid_response_length - 1))
                    if end > 0:
                        process_reward_tensor[i, pos] = step_score

            # 3. Logging
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
                "outcome_reward_tensor": outcome_reward_tensor,
                "process_reward_tensor": process_reward_tensor,
                "reward_tensor": outcome_reward_tensor, # Fallback compatibility
                "reward_extra_info": reward_extra_info,
                "prompt": prompt,
                "ground_truth": gt,
                "response": response,
                "outcome_reward": outcome_reward
            }
        else:
            return outcome_reward_tensor
