from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from verl.utils.model import compute_position_id_with_mask


@dataclass(frozen=True)
class PaddedPromptResponseBatch:
    prompts: torch.Tensor
    responses: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    response_mask: torch.Tensor


def build_padded_prompt_response_batch(
    prompt_sequences: Sequence[torch.Tensor],
    response_sequences: Sequence[torch.Tensor],
    pad_token_id: int,
) -> PaddedPromptResponseBatch:
    """Build the canonical verl layout: left-padded prompts followed by right-padded responses."""
    if not prompt_sequences:
        raise ValueError("prompt_sequences must not be empty.")
    if len(prompt_sequences) != len(response_sequences):
        raise ValueError(
            "prompt_sequences and response_sequences must have the same length, "
            f"got {len(prompt_sequences)} and {len(response_sequences)}."
        )

    reference = prompt_sequences[0]
    if not isinstance(reference, torch.Tensor):
        raise TypeError("prompt_sequences[0] must be a torch.Tensor.")
    device = reference.device
    dtype = reference.dtype

    def _validate_sequence(sequence: torch.Tensor, kind: str, index: int) -> None:
        if not isinstance(sequence, torch.Tensor):
            raise TypeError(f"{kind}_sequences[{index}] must be a torch.Tensor.")
        if sequence.ndim != 1:
            raise ValueError(f"{kind}_sequences[{index}] must be one-dimensional, got shape {tuple(sequence.shape)}.")
        if sequence.numel() == 0:
            raise ValueError(f"{kind}_sequences[{index}] must not be empty.")
        if sequence.device != device:
            raise ValueError(
                f"{kind}_sequences[{index}] is on {sequence.device}, expected {device}."
            )
        if sequence.dtype != dtype:
            raise ValueError(
                f"{kind}_sequences[{index}] has dtype {sequence.dtype}, expected {dtype}."
            )

    for index, prompt in enumerate(prompt_sequences):
        _validate_sequence(prompt, "prompt", index)
    for index, response in enumerate(response_sequences):
        _validate_sequence(response, "response", index)

    batch_size = len(prompt_sequences)
    max_prompt_length = max(sequence.numel() for sequence in prompt_sequences)
    max_response_length = max(sequence.numel() for sequence in response_sequences)

    prompts = torch.full(
        (batch_size, max_prompt_length),
        pad_token_id,
        dtype=dtype,
        device=device,
    )
    responses = torch.full(
        (batch_size, max_response_length),
        pad_token_id,
        dtype=dtype,
        device=device,
    )
    prompt_mask = torch.zeros((batch_size, max_prompt_length), dtype=torch.long, device=device)
    response_mask = torch.zeros((batch_size, max_response_length), dtype=torch.long, device=device)

    for index, (prompt, response) in enumerate(zip(prompt_sequences, response_sequences)):
        prompt_length = prompt.numel()
        response_length = response.numel()
        prompts[index, max_prompt_length - prompt_length :] = prompt
        prompt_mask[index, max_prompt_length - prompt_length :] = 1
        responses[index, :response_length] = response
        response_mask[index, :response_length] = 1

    input_ids = torch.cat((prompts, responses), dim=-1)
    attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
    position_ids = compute_position_id_with_mask(attention_mask)

    expected_width = prompts.size(1) + responses.size(1)
    if input_ids.size(1) != expected_width or attention_mask.size(1) != expected_width:
        raise RuntimeError(
            "Invalid prompt/response batch layout: input_ids and attention_mask must contain "
            "the padded prompt segment followed by the padded response segment."
        )

    return PaddedPromptResponseBatch(
        prompts=prompts,
        responses=responses,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        response_mask=response_mask,
    )
