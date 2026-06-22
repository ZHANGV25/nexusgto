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
from engine.rollout import (  # type: ignore[import-untyped]
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


# ── Chip-aware (stack-commitment) vocab — the 3a fix ─────────────────────
#
# The legacy legal_categories above is the DEEP-STACK special case: it always
# emits every sized raise. The TRAINER (mccfr_blueprint.c generate_actions,
# lines 938-973) prunes a raise when the player is committed — exactly:
#   * FOLD only when facing a bet (to_call > 0).
#   * CHECK/CALL always.
#   * For each bet size, the raise-to amount `ba` is computed; the slot is
#     DROPPED if `ba <= to_call` (the raise can't legally exceed the call =
#     the player is committed), and collapsed to a SINGLE all-in when
#     `ba >= stack` (deduped).
# Reproducing this here makes the agent's legal-action COUNT equal the blueprint's
# stored σ̄ length on committed lines, so LookupSource's exact-length invariant
# no longer fires mid-hand (the §6 boundary). Integer `(int)` truncation is
# load-bearing — port it, do not round (a boundary `ba` that rounds the other
# way flips the prune and serves the wrong-length node).


def raise_to_amount(frac: float, pot: int, to_call: int, stack: int) -> int:
    """The raise-to chip amount for one bet size, bit-exact to the trainer.

    Trainer (generate_actions): if to_call==0 `ba=(int)(frac*pot)` else
    `ba = to_call + (int)(frac*(pot+to_call))`; then clamp `if ba>=stack ba=stack`.
    Returns `ba` (the chip increment added to the pot for this raise/bet).
    """
    if to_call == 0:
        ba = int(frac * pot)
    else:
        ba = to_call + int(frac * (pot + to_call))
    if ba >= stack:
        ba = stack
    return ba


def legal_categories_chips(
    facing_bet: bool,
    raises_so_far: int,
    pot: int,
    stack: int,
    to_call: int,
    postflop_bet_sizes: Tuple[float, ...],
    subsequent_bet_sizes: Tuple[float, ...],
    add_explicit_allin: bool,
) -> List[ActionCategory]:
    """Legal action categories with the trainer's stack-commitment prune.

    A drop-in, chip-aware replacement for legal_categories. On deep stacks
    (no raise pruned) it returns the IDENTICAL set/order as legal_categories,
    so existing indices and the node-key encoding never shift.
    """
    if raises_so_far >= POSTFLOP_MAX_RAISES:
        return (
            [ActionCategory.FOLD, ActionCategory.CALL]
            if facing_bet else [ActionCategory.CALL]
        )

    cats: List[ActionCategory] = []
    if to_call > 0:
        cats.append(ActionCategory.FOLD)
    cats.append(ActionCategory.CALL)  # check when to_call==0, else call

    sizes = subsequent_bet_sizes if raises_so_far > 0 else postflop_bet_sizes
    added_allin = False
    for frac in sizes:
        ba = raise_to_amount(frac, pot, to_call, stack)
        if ba <= to_call:
            continue  # committed: the raise can't exceed the call -> dropped
        if ba >= stack:
            if added_allin:
                continue  # dedup multiple sizes that all clamp to all-in
            added_allin = True
        cats.append(ActionCategory.RAISE)

    # Explicit all-in slot (OFF for 8.05B). Mirrors the trainer's gated block:
    # only when no sized raise already reached all-in and a raise is possible.
    if add_explicit_allin and not added_allin and stack > to_call:
        cats.append(ActionCategory.RAISE)

    return cats


def num_legal_chips(
    facing_bet: bool,
    raises_so_far: int,
    pot: int,
    stack: int,
    to_call: int,
    postflop_bet_sizes: Tuple[float, ...],
    subsequent_bet_sizes: Tuple[float, ...],
    add_explicit_allin: bool,
) -> int:
    return len(
        legal_categories_chips(
            facing_bet, raises_so_far, pot, stack, to_call,
            postflop_bet_sizes, subsequent_bet_sizes, add_explicit_allin,
        )
    )
