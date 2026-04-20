from __future__ import annotations

import re
from typing import Callable


def format_step_reward(step_text: str) -> float:
    """Returns 1.0 if step_text contains ≥1 <premise> and exactly 1 <conclusion>."""
    premises = re.findall(r"<premise>.*?</premise>", step_text, re.DOTALL)
    conclusions = re.findall(r"<conclusion>.*?</conclusion>", step_text, re.DOTALL)
    return 1.0 if (len(premises) >= 1 and len(conclusions) == 1) else 0.0


def fol_step_reward(step_text: str) -> float:
    """FOL/Z3-based step verification reward.

    Interface for future implementation. To implement:
      1. Parse premises and conclusion from step_text.
      2. Encode them as FOL formulas.
      3. Use Z3 to verify whether the conclusion follows from the premises.
      4. Return 1.0 if valid, 0.0 otherwise.
    """
    raise NotImplementedError(
        "FOL reward is not yet implemented. "
        "Implement fol_step_reward() in verl/trainer/ppo/sampling/mcts_prm.py."
    )


def get_prm_fn(prm_type: str) -> Callable[[str], float]:
    """Return the PRM scoring function for the given type.

    Supported:
        'format': checks <step>/<premise>/<conclusion> tag structure.
        'fol':    FOL/Z3 verification (not yet implemented).
    """
    if prm_type == "format":
        return format_step_reward
    elif prm_type == "fol":
        return fol_step_reward
    else:
        raise ValueError(f"Unknown PRM type: {prm_type!r}. Supported: 'format', 'fol'")
