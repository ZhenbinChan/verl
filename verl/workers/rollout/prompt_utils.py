from __future__ import annotations

import torch


def extract_prompt_token_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[int]:
    """Extract valid prompt tokens using the attention mask as the source of truth."""
    if input_ids.ndim != 1 or attention_mask.ndim != 1:
        raise ValueError(
            "input_ids and attention_mask must be one-dimensional, "
            f"got {tuple(input_ids.shape)} and {tuple(attention_mask.shape)}."
        )
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            "input_ids and attention_mask must have the same shape, "
            f"got {tuple(input_ids.shape)} and {tuple(attention_mask.shape)}."
        )

    is_binary = torch.logical_or(attention_mask == 0, attention_mask == 1)
    if not bool(torch.all(is_binary)):
        raise ValueError("attention_mask must contain only 0 and 1 values.")

    valid_mask = attention_mask.bool()
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("Prompt attention_mask must contain at least one valid token.")

    first = int(valid_indices[0].item())
    last = int(valid_indices[-1].item())
    if not bool(torch.all(valid_mask[first : last + 1])):
        raise ValueError("Prompt attention_mask valid tokens must form one contiguous region.")

    return input_ids[valid_mask].tolist()
