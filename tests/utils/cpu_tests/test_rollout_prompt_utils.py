import unittest

import torch

from verl.workers.rollout.prompt_utils import extract_prompt_token_ids


class TestExtractPromptTokenIds(unittest.TestCase):
    def test_extracts_left_right_and_two_sided_padding(self):
        cases = (
            ([0, 0, 11, 12], [0, 0, 1, 1]),
            ([11, 12, 0, 0], [1, 1, 0, 0]),
            ([0, 11, 12, 0], [0, 1, 1, 0]),
        )

        for input_ids, attention_mask in cases:
            with self.subTest(attention_mask=attention_mask):
                self.assertEqual(
                    extract_prompt_token_ids(torch.tensor(input_ids), torch.tensor(attention_mask)),
                    [11, 12],
                )

    def test_preserves_valid_token_equal_to_pad_token_id(self):
        self.assertEqual(
            extract_prompt_token_ids(torch.tensor([0, 11, 0, 12]), torch.tensor([0, 1, 1, 1])),
            [11, 0, 12],
        )

    def test_rejects_invalid_masks_and_shapes(self):
        invalid_cases = (
            (torch.tensor([[11, 12]]), torch.tensor([1, 1])),
            (torch.tensor([11, 12]), torch.tensor([1])),
            (torch.tensor([11, 12]), torch.tensor([0, 0])),
            (torch.tensor([11, 12]), torch.tensor([1, 2])),
            (torch.tensor([11, 12, 13]), torch.tensor([1, 0, 1])),
        )

        for input_ids, attention_mask in invalid_cases:
            with self.subTest(input_shape=tuple(input_ids.shape), attention_mask=attention_mask.tolist()):
                with self.assertRaises(ValueError):
                    extract_prompt_token_ids(input_ids, attention_mask)


if __name__ == "__main__":
    unittest.main()
