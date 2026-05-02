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

import re
from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


class NaiveFormatRewardManager:
    """Reward manager with dual rewards:
    1. Format reward: placed at the last token of each correctly formatted <step>...</step> block.
       Normalized as: num_valid_steps / max(total_step_blocks, target_format_steps).
       This incentivizes multi-step reasoning (at least target_format_steps steps)
       while penalizing the 1-step shortcut.
    2. Answer reward: +1 at the last valid token if the final answer (extracted from \\boxed{})
       matches the ground truth.
    
    The GRPO advantage estimator sums all token-level rewards per sequence.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None,
                 reward_fn_key="data_source", target_format_steps=3, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.target_format_steps = target_format_steps

    # ------------------------------------------------------------------
    # Format parsing helpers
    # ------------------------------------------------------------------

    def _is_valid_step_content(self, step_content: str) -> bool:
        """Check that the content inside a <step>...</step> block has:
        - At least one <premise>...</premise> pair
        - At least one <conclusion>...</conclusion> pair
        """
        has_premise = bool(re.search(r'<premise>.*?</premise>', step_content, re.DOTALL))
        has_conclusion = bool(re.search(r'<conclusion>.*?</conclusion>', step_content, re.DOTALL))
        return has_premise and has_conclusion

    def _parse_step_end_positions(self, response_str: str):
        """Find the character positions of the closing '>' of each </step> tag
        that belongs to a correctly formatted <step>...</step> block.

        Uses a stack to handle nesting correctly.

        Returns:
            tuple: (valid_end_positions, total_steps)
                - valid_end_positions: List[int], character offsets for valid steps only.
                - total_steps: int, total number of matched <step>...</step> pairs
                  (regardless of content validity). Used as normalization denominator.
        """
        # Collect all <step> and </step> occurrences with their positions
        events = []
        for m in re.finditer(r'<step>', response_str):
            events.append((m.start(), 'open', m.end()))  # open_pos, type, close_tag_start_pos
        for m in re.finditer(r'</step>', response_str):
            events.append((m.start(), 'close', m.end()))  # start of </step>, type, position after >

        events.sort(key=lambda x: x[0])

        valid_end_positions = []
        total_steps = 0
        stack = []  # stack of (open_char_pos)

        for pos, etype, end_pos in events:
            if etype == 'open':
                stack.append(pos)
            elif etype == 'close':
                if stack:
                    open_pos = stack.pop()
                    total_steps += 1
                    # Extract the content between <step> and </step>
                    # end_pos is the character position right after '>', so
                    # the content goes from open_pos to (pos) (start of '</step>')
                    step_content = response_str[open_pos:pos]
                    if self._is_valid_step_content(step_content):
                        # The '>' of '</step>' is at end_pos - 1
                        valid_end_positions.append(end_pos - 1)
                # else: unmatched </step>, ignore

        return valid_end_positions, total_steps

    def _find_token_idx_for_char(self, char_pos: int,
                                  offsets) -> int:
        """Map a character position in response_str to a token index using
        the offset_mapping from tokenizer(return_offsets_mapping=True).

        Falls back to the last token if the character position is beyond all offsets.
        """
        for idx, (start, end) in enumerate(offsets):
            if start <= char_pos < end:
                return idx
        # fallback: last token
        return len(offsets) - 1 if offsets else 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, data: DataProto, return_dict=False):
        # If pre-computed rm_scores exist, delegate (path is effectively dead in our setup)
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": {},
                }
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        prompt_list = []
        gt_list = []
        response_list = []
        outcome_reward_list = []
        format_reward_counts = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # ----------------------------------------------------------
            # 1. Compute answer reward via the existing scoring function
            # ----------------------------------------------------------
            answer_score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(answer_score, dict):
                answer_reward = float(answer_score.get("score", 0.0))
            elif isinstance(answer_score, (list, tuple)):
                answer_reward = float(answer_score[0])
            else:
                answer_reward = float(answer_score)

            # ----------------------------------------------------------
            # 2. Parse format: find valid <step> blocks and their
            #    corresponding token positions
            # ----------------------------------------------------------
            step_end_char_positions, total_step_blocks = self._parse_step_end_positions(response_str)

            # Re-encode the response to get token-to-character offset mapping
            encoded = self.tokenizer(response_str, return_offsets_mapping=True,
                                     add_special_tokens=False)
            offsets = encoded["offset_mapping"]  # list of (start_char, end_char)

            step_format_rewards = 0
            norm_factor = 1.0 / max(total_step_blocks, self.target_format_steps) if total_step_blocks > 0 else 0.0
            for char_pos in step_end_char_positions:
                token_idx = self._find_token_idx_for_char(char_pos, offsets)
                # Safety: clamp to valid response length
                if 0 <= token_idx < valid_response_length:
                    reward_tensor[i, token_idx] = norm_factor
                    step_format_rewards += norm_factor

            # ----------------------------------------------------------
            # 3. Place answer reward at the last valid response token
            # ----------------------------------------------------------
            reward_tensor[i, valid_response_length - 1] = answer_reward

            # ----------------------------------------------------------
            # 4. Bookkeeping
            # ----------------------------------------------------------
            total_reward = step_format_rewards + answer_reward
            outcome_reward_list.append(total_reward)
            format_reward_counts.append(step_format_rewards)
            prompt_list.append(prompt_str)
            gt_list.append(ground_truth)
            response_list.append(response_str)

            # Logging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[answer_reward]", answer_reward)
                print("[format_reward_count]", step_format_rewards)
                print("[total_reward]", total_reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "prompt": prompt_list,
                "ground_truth": gt_list,
                "response": response_list,
                "outcome_reward": outcome_reward_list,
                "format_reward_count": format_reward_counts,
            }
        else:
            return reward_tensor