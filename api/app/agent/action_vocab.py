"""Metadata-driven postflop action vocabulary (plan §5 encoding facts).

The legal action-set COUNT at a postflop node is determined by the blueprint's
TRAINED bet-size vocab (postflop_bet_sizes / subsequent_bet_sizes /
add_explicit_allin from .bps metadata), NOT a hardcoded constant. rollout.py's
classify_postflop_actions is wired to v3's (0.5, 1.0); the 8.05B trained with
[0.75], so its no-bet node has 2 actions, not 3.

This module mirrors classify_postflop_actions' BRANCH STRUCTURE but
parameterizes the sizes — consistent with production (session_pipeline takes
`bet_sizes` explicitly). It governs the action COUNT only; the seat /
who-acts-next logic (what broke last session) stays in GameState, unchanged
and anchor-validated.
"""

from __future__ import annotations

from typing import List, Tuple

# Reuse the category enum + max-raises cap from the solver so the index
# SEMANTICS (fold/call/raise categories) stay bit-identical to training.
try:
    from rollout import (  # type: ignore[import-untyped]
        POSTFLOP_MAX_RAISES,
        ActionCategory,
    )
except ImportError:  # pragma: no cover
    from python.rollout import (  # type: ignore[no-redef]
        POSTFLOP_MAX_RAISES,
        ActionCategory,
    )

# The 8.05B-trained vocab (capture-verified). v1 targets this blueprint.
DEFAULT_POSTFLOP_BET_SIZES: Tuple[float, ...] = (0.75,)
DEFAULT_SUBSEQUENT_BET_SIZES: Tuple[float, ...] = (1.0,)
DEFAULT_ADD_EXPLICIT_ALLIN: bool = False


def legal_categories(
    facing_bet: bool,
    raises_so_far: int,
    postflop_bet_sizes: Tuple[float, ...],
    subsequent_bet_sizes: Tuple[float, ...],
    add_explicit_allin: bool,
) -> List[ActionCategory]:
    """Legal action categories for a postflop node, per the blueprint vocab.

    Mirrors rollout.classify_postflop_actions' structure:
      tree closed (raises >= max):  facing -> [fold, call]; else [check]
      facing a bet: [fold, call, raise×len(sizes), (allin)?]  where sizes =
        subsequent_bet_sizes if raises_so_far>0 else postflop_bet_sizes
      no bet:       [check, bet×len(postflop_bet_sizes), (allin)?]
    """
    if raises_so_far >= POSTFLOP_MAX_RAISES:
        return (
            [ActionCategory.FOLD, ActionCategory.CALL]
            if facing_bet else [ActionCategory.CALL]
        )

    if facing_bet:
        sizes = subsequent_bet_sizes if raises_so_far > 0 else postflop_bet_sizes
        cats = [ActionCategory.FOLD, ActionCategory.CALL]
        cats.extend([ActionCategory.RAISE] * len(sizes))
        if add_explicit_allin:
            cats.append(ActionCategory.RAISE)
        return cats

    cats = [ActionCategory.CALL]  # check
    cats.extend([ActionCategory.RAISE] * len(postflop_bet_sizes))
    if add_explicit_allin:
        cats.append(ActionCategory.RAISE)
    return cats


def num_legal(
    facing_bet: bool,
    raises_so_far: int,
    postflop_bet_sizes: Tuple[float, ...],
    subsequent_bet_sizes: Tuple[float, ...],
    add_explicit_allin: bool,
) -> int:
    return len(
        legal_categories(
            facing_bet, raises_so_far, postflop_bet_sizes,
            subsequent_bet_sizes, add_explicit_allin,
        )
    )
