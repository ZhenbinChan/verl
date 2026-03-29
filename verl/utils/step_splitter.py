"""
Shared step-splitting utilities for TreeRL and StepRewardManager.

Provides common functions for splitting LLM response text into reasoning steps,
both by plain-text delimiters (e.g., \\n\\n) and by XML tags (<step>...</step>).
Also provides token-position mapping for step boundaries.
"""

import re
from typing import Callable, Optional


def default_split_fn(response_text: str) -> list[str]:
    """Default step splitter: split by double newline."""
    if not response_text:
        return [""]
    return response_text.split("\n\n")


def split_response_into_steps(
    response_text: str,
    split_fn: Optional[Callable[[str], list[str]]] = None,
) -> list[tuple[str, int, int]]:
    """Split response text into steps using a given splitter function.

    Args:
        response_text: The full decoded response text.
        split_fn: A callable that splits the text into segments.
            Defaults to ``default_split_fn`` (split by ``\\n\\n``).

    Returns:
        List of (step_text, char_start, char_end) tuples.
    """
    if split_fn is None:
        split_fn = default_split_fn
    segments = split_fn(response_text)
    steps: list[tuple[str, int, int]] = []
    cursor = 0
    for seg in segments:
        start = response_text.find(seg, cursor)
        if start == -1:
            start = cursor
        end = start + len(seg)
        steps.append((seg, start, end))
        cursor = end
    return steps


def split_by_xml_step_tags(response_text: str) -> list[tuple[str, int, int]]:
    """Split response text into steps using ``<step>...</step>`` XML tags.

    If no ``<step>`` tags are found the returned list is empty, allowing
    callers to fall back to a delimiter-based splitter.

    Args:
        response_text: The full decoded response text.

    Returns:
        List of (step_text, char_start, char_end) tuples.
    """
    pattern = r"<step>.*?(?:</step>|$)"
    matches = list(re.finditer(pattern, response_text, flags=re.DOTALL))
    return [(m.group(0), m.start(), m.end()) for m in matches]


def _char_end_to_token_pos(response_ids, tokenizer, char_end: int, valid_response_length: int) -> int:
    """Binary search: return the 0-indexed token position whose decoded prefix covers ``char_end`` chars.

    Works directly on the actual token IDs so there is no BPE round-trip drift
    (``encode(decode(ids[:k]))`` can differ from ``k`` at BPE merge boundaries).
    """
    ids = list(response_ids[:valid_response_length])
    n = len(ids)
    if n == 0:
        return 0
    # Find smallest prefix length k (1-indexed) s.t. len(decode(ids[:k])) >= char_end.
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(ids[:mid], skip_special_tokens=True)) >= char_end:
            hi = mid
        else:
            lo = mid + 1
    return max(0, min(lo - 1, n - 1))  # convert to 0-indexed token position


def get_step_token_positions(
    response_text: str,
    valid_response_length: int,
    tokenizer,
    use_xml: bool = False,
    split_fn: Optional[Callable[[str], list[str]]] = None,
    response_ids=None,
) -> list[tuple[str, int]]:
    """Map character-level step boundaries to token positions.

    Tries XML ``<step>`` tag splitting first when *use_xml* is ``True``.
    Falls back to the delimiter-based splitter.

    Args:
        response_text: The full decoded response text.
        valid_response_length: Number of valid (non-padding) response tokens.
        tokenizer: HuggingFace tokenizer for encoding text.
        use_xml: If ``True``, attempt XML ``<step>`` tag splitting first.
        split_fn: Custom text splitter; defaults to ``default_split_fn``.
        response_ids: Optional actual token ID sequence (tensor or list).
            When provided, token positions are derived directly from the token
            IDs instead of re-encoding text prefixes, eliminating BPE drift.

    Returns:
        List of (step_text, token_end_pos) tuples where *token_end_pos* is the
        0-indexed position of the last token in this step within the response.
    """
    if response_ids is not None:
        if not use_xml:
            # Token-space splitting: no decode→encode round-trip at all.
            ids = list(response_ids[:valid_response_length])
            token_steps = split_tokens_by_delimiter(ids, tokenizer)
            result: list[tuple[str, int]] = []
            for _tok_start, tok_end, step_text in token_steps:
                token_end_pos = max(0, min(tok_end - 1, valid_response_length - 1))
                result.append((step_text, token_end_pos))
            return result
        else:
            # XML splitting: text-level boundaries, then binary-search token positions.
            steps = split_by_xml_step_tags(response_text)
            if not steps:
                steps = split_response_into_steps(response_text, split_fn)
            result = []
            for step_text, _char_start, char_end in steps:
                token_end_pos = _char_end_to_token_pos(response_ids, tokenizer, char_end, valid_response_length)
                result.append((step_text, token_end_pos))
            return result

    # Legacy path (no response_ids): original encode-prefix behaviour.
    steps: list[tuple[str, int, int]] = []
    if use_xml:
        steps = split_by_xml_step_tags(response_text)
    if not steps:
        steps = split_response_into_steps(response_text, split_fn)

    result = []
    for step_text, _char_start, char_end in steps:
        text_up_to_end = response_text[:char_end]
        tokens_up_to_end = tokenizer.encode(text_up_to_end, add_special_tokens=False)
        token_end_pos = min(len(tokens_up_to_end) - 1, valid_response_length - 1)
        token_end_pos = max(0, token_end_pos)
        result.append((step_text, token_end_pos))
    return result


def split_tokens_by_delimiter(
    token_ids,
    tokenizer,
    delimiter: str = "\n\n",
) -> list[tuple[int, int, str]]:
    """Split a token sequence at delimiter boundaries directly in token space.

    This avoids the decode |→| split |→| re-encode round-trip that causes BPE
    drift (``len(encode(decode(tokens[:k]))) != k``).

    The delimiter tokens are included at the **start** of the following step,
    matching the behaviour of the character-level
    ``split_response_into_steps()`` helper.

    Args:
        token_ids: Flat list/sequence of token IDs (e.g. from ``.tolist()``).
        tokenizer: HuggingFace tokenizer used to encode the delimiter and
            decode step text.
        delimiter: The text delimiter whose token encoding is searched for in
            *token_ids*.  Defaults to ``"\\n\\n"``.

    Returns:
        List of ``(token_start, token_end, step_text)`` tuples.
    """
    token_ids = list(token_ids)
    if not token_ids:
        return [(0, 0, "")]

    delim_tokens = tokenizer.encode(delimiter, add_special_tokens=False)
    delim_len = len(delim_tokens)

    if delim_len == 0:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return [(0, len(token_ids), text)]

    # Scan for delimiter token sequence
    split_points: list[int] = []
    i = 0
    while i <= len(token_ids) - delim_len:
        if token_ids[i : i + delim_len] == delim_tokens:
            split_points.append(i)
            i += delim_len
        else:
            i += 1

    # Build step ranges – delimiter tokens go with the *following* step
    ranges: list[tuple[int, int]] = []
    start = 0
    for pos in split_points:
        if pos > start:
            ranges.append((start, pos))
        start = pos
    if start < len(token_ids):
        ranges.append((start, len(token_ids)))

    if not ranges:
        ranges = [(0, len(token_ids))]

    # Decode each range to obtain step_text
    result: list[tuple[int, int, str]] = []
    for tok_start, tok_end in ranges:
        step_text = tokenizer.decode(
            token_ids[tok_start:tok_end], skip_special_tokens=True
        )
        result.append((tok_start, tok_end, step_text))

    return result


def get_split_fn(
    use_xml: bool = False,
) -> Callable[[str], list[str]]:
    """Return a text-level split function controlled by an explicit flag.

    When *use_xml* is ``True``, returns a splitter that tries ``<step>`` XML
    tags first and falls back to ``\\n\\n``.  When ``False`` (default), returns
    ``default_split_fn`` (``\\n\\n`` only).

    This is useful when callers need only the *text segments* (not token
    positions) — e.g. ``TreeManager`` which manages its own token bookkeeping.
    """
    if use_xml:

        def _xml_or_default(response_text: str) -> list[str]:
            steps = split_by_xml_step_tags(response_text)
            if steps:
                return [s[0] for s in steps]
            return default_split_fn(response_text)

        return _xml_or_default
    return default_split_fn


def get_split_fn_for_reward_type(
    step_reward_types: Optional[list[str]] = None,
) -> Callable[[str], list[str]]:
    """Deprecated. Use :func:`get_split_fn` with an explicit ``use_xml`` flag.

    Infers ``use_xml`` from reward type names for backward compatibility.
    """
    use_xml = step_reward_types is not None and any(
        rt in ("fol", "format") for rt in step_reward_types
    )
    return get_split_fn(use_xml=use_xml)
