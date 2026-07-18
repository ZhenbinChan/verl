import pytest
import torch

from verl.utils.ppo_batch import build_padded_prompt_response_batch


def test_build_padded_prompt_response_batch_uses_canonical_layout():
    batch = build_padded_prompt_response_batch(
        prompt_sequences=[torch.tensor([11, 12]), torch.tensor([21, 22, 23, 24])],
        response_sequences=[torch.tensor([31, 32, 33]), torch.tensor([41])],
        pad_token_id=0,
    )

    expected_prompts = torch.tensor([[0, 0, 11, 12], [21, 22, 23, 24]])
    expected_responses = torch.tensor([[31, 32, 33], [41, 0, 0]])
    expected_response_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    expected_attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0],
        ]
    )
    expected_position_ids = torch.tensor(
        [
            [0, 0, 0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 4, 4],
        ]
    )

    assert torch.equal(batch.prompts, expected_prompts)
    assert torch.equal(batch.responses, expected_responses)
    assert torch.equal(batch.input_ids, torch.cat((expected_prompts, expected_responses), dim=-1))
    assert torch.equal(batch.response_mask, expected_response_mask)
    assert torch.equal(batch.attention_mask, expected_attention_mask)
    assert torch.equal(batch.position_ids, expected_position_ids)


def test_build_padded_prompt_response_batch_uses_lengths_instead_of_pad_token_values():
    batch = build_padded_prompt_response_batch(
        prompt_sequences=[torch.tensor([0, 11])],
        response_sequences=[torch.tensor([21, 0, 22])],
        pad_token_id=0,
    )

    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1]]
    assert batch.response_mask.tolist() == [[1, 1, 1]]


@pytest.mark.parametrize(
    ("prompts", "responses", "message"),
    [
        ([], [], "prompt_sequences must not be empty"),
        ([torch.tensor([1])], [], "must have the same length"),
        ([torch.tensor([])], [torch.tensor([2])], "prompt_sequences[0] must not be empty"),
        ([torch.tensor([1])], [torch.tensor([])], "response_sequences[0] must not be empty"),
        ([torch.tensor([[1]])], [torch.tensor([2])], "must be one-dimensional"),
    ],
)
def test_build_padded_prompt_response_batch_rejects_invalid_sequences(prompts, responses, message):
    with pytest.raises(ValueError, match=message):
        build_padded_prompt_response_batch(prompts, responses, pad_token_id=0)


def test_build_padded_prompt_response_batch_rejects_mixed_dtypes():
    with pytest.raises(ValueError, match="has dtype"):
        build_padded_prompt_response_batch(
            [torch.tensor([1], dtype=torch.long)],
            [torch.tensor([2], dtype=torch.int32)],
            pad_token_id=0,
        )
