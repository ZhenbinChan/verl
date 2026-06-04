import unittest

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.naive_format import NaiveFormatRewardManager


class CharOffsetTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(int(tok)) for tok in tokens if int(tok) != self.pad_token_id)

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        result = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return result


def make_data(response: str, prompt: str = "Q") -> DataProto:
    tokenizer = CharOffsetTokenizer()
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)
    attention_mask = torch.ones((1, len(prompt_ids) + len(response_ids)), dtype=torch.long)
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([prompt_ids], dtype=torch.long),
            "responses": torch.tensor([response_ids], dtype=torch.long),
            "attention_mask": attention_mask,
        },
        non_tensors={
            "reward_model": np.array([{"ground_truth": "A"}], dtype=object),
            "data_source": np.array(["reclor"], dtype=object),
            "extra_info": np.array([{}], dtype=object),
        },
    )


def step_end_index(response: str, occurrence: int = 0) -> int:
    cursor = -1
    for _ in range(occurrence + 1):
        cursor = response.index("</step>", cursor + 1)
    return cursor + len("</step>") - 1


class TestNaiveFormatRewardManager(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharOffsetTokenizer()

    def make_manager(self, answer_score=0.0):
        return NaiveFormatRewardManager(
            tokenizer=self.tokenizer,
            num_examine=0,
            compute_score=lambda **_: answer_score,
        )

    def reward_for(self, response: str, answer_score=0.0):
        manager = self.make_manager(answer_score=answer_score)
        result = manager(make_data(response), return_dict=True)
        return result["reward_tensor"][0], result

    def test_two_valid_steps_each_get_half(self):
        step1 = "<step><premise>a</premise><conclusion>b</conclusion></step>"
        step2 = "<step><premise>c</premise><premise>d</premise><conclusion>e</conclusion></step>"
        response = step1 + step2 + "\\boxed{A}"

        rewards, result = self.reward_for(response, answer_score=1.0)

        self.assertAlmostEqual(float(rewards[step_end_index(response, 0)]), 0.5)
        self.assertAlmostEqual(float(rewards[step_end_index(response, 1)]), 0.5)
        self.assertAlmostEqual(float(rewards[len(response) - 1]), 1.0)
        self.assertAlmostEqual(result["format_reward_count"][0], 1.0)
        self.assertAlmostEqual(result["outcome_reward"][0], 2.0)
        self.assertEqual(result["reward_extra_info"]["format_error_advantage_mask"], [0.0])

    def test_invalid_step_counts_in_denominator_but_gets_no_reward(self):
        valid = "<step><premise>a</premise><conclusion>b</conclusion></step>"
        invalid = "<step><premise>c</premise><conclusion>d</conclusion><conclusion>e</conclusion></step>"
        response = valid + invalid + "\\boxed{A}"

        rewards, result = self.reward_for(response, answer_score=1.0)

        self.assertAlmostEqual(float(rewards[step_end_index(response, 0)]), 0.5)
        self.assertAlmostEqual(float(rewards[step_end_index(response, 1)]), 0.0)
        self.assertAlmostEqual(float(rewards[len(response) - 1]), 1.0)
        self.assertAlmostEqual(result["format_reward_count"][0], 0.5)
        self.assertAlmostEqual(result["outcome_reward"][0], 1.5)
        self.assertEqual(result["reward_extra_info"]["format_error_advantage_mask"], [1.0])

    def test_invalid_step_structures_receive_zero(self):
        responses = [
            "<step><conclusion>b</conclusion></step>",
            "<step><premise>a</premise></step>",
            "<step>extra<premise>a</premise><conclusion>b</conclusion></step>",
            "<step><premise>a</premise><foo>b</foo><conclusion>c</conclusion></step>",
            "<step><premise><inner>a</inner></premise><conclusion>b</conclusion></step>",
        ]

        for response in responses:
            with self.subTest(response=response):
                rewards, result = self.reward_for(response, answer_score=0.0)
                self.assertAlmostEqual(float(rewards.sum()), 0.0)
                self.assertAlmostEqual(result["format_reward_count"][0], 0.0)
                self.assertAlmostEqual(result["outcome_reward"][0], 0.0)
                self.assertEqual(result["reward_extra_info"]["format_error_advantage_mask"], [1.0])

    def test_answer_reward_accumulates_with_step_end_token(self):
        response = "<step><premise>a</premise><conclusion>b</conclusion></step>"

        rewards, result = self.reward_for(response, answer_score=1.0)

        self.assertAlmostEqual(float(rewards[len(response) - 1]), 2.0)
        self.assertAlmostEqual(result["format_reward_count"][0], 1.0)
        self.assertAlmostEqual(result["outcome_reward"][0], 2.0)
        self.assertEqual(result["reward_extra_info"]["format_error_advantage_mask"], [1.0])

    def test_plain_reasoning_is_masked_for_advantage(self):
        response = "plain reasoning\\boxed{A}"

        rewards, result = self.reward_for(response, answer_score=1.0)

        self.assertAlmostEqual(float(rewards.sum()), 1.0)
        self.assertAlmostEqual(result["format_reward_count"][0], 0.0)
        self.assertAlmostEqual(result["outcome_reward"][0], 1.0)
        self.assertEqual(result["reward_extra_info"]["format_error_advantage_mask"], [1.0])


if __name__ == "__main__":
    unittest.main()
