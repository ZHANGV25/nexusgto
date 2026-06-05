"""Opponent action -> on-tree action index (plan §2, §3.3).

A LOOKUP agent can't search, so when a human makes an off-tree bet we must
APPROXIMATE the on-tree node (Pluribus used live search for this; a pure-
lookup agent snaps via pseudoharmonic interpolation — plan §0 / playing_agent_v1
memory names this the real open design question). On-tree actions translate
exactly.

REUSES the production primitives rather than reimplementing them:
  - session_pipeline._action_to_solver_label   (action+amount -> "Bet 75%")
  - session_pipeline._solver_label_to_action_index  (label -> tree index)
  - off_tree.pseudoharmonic_map                 (off-tree size -> tree sizes)

The returned index matches rollout.classify_postflop_actions ordering, so
GameState.advance / key_builder consume it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from app.services.session_pipeline import (
    _action_to_solver_label,
    _solver_label_to_action_index,
)

# Translation provenance tags (for observability / tests).
ON_TREE_EXACT = "on_tree_exact"
OFF_TREE_PSEUDOHARMONIC = "off_tree_pseudoharmonic"

_OFF_TREE_TOLERANCE = 0.02  # within 2% of a tree size counts as on-tree


@dataclass(frozen=True)
class TranslatedAction:
    """Result of translating one opponent action.

    action_idx: index into classify_postflop_actions(facing_bet, raises).
    kind: ON_TREE_EXACT or OFF_TREE_PSEUDOHARMONIC.
    snapped_label: the solver label the action resolved to ("Bet 75%", ...).
    """

    action_idx: int
    kind: str
    snapped_label: str


def _bet_frac(amount: Optional[float], pot: float) -> Optional[float]:
    if amount is None or pot <= 0:
        return None
    return amount / pot


def _is_off_tree(frac: Optional[float], bet_sizes: Tuple[float, ...]) -> bool:
    if frac is None:
        return False
    return all(abs(frac - s) > _OFF_TREE_TOLERANCE for s in bet_sizes)


def translate_action(
    action: str,
    amount: Optional[float],
    pot: float,
    bet_sizes: Tuple[float, ...],
    add_explicit_allin: bool,
    facing_bet: bool,
    raises_so_far: int,
) -> TranslatedAction:
    """Translate one opponent action into an on-tree action index.

    Args:
        action: "fold" | "check" | "call" | "raise" | "allin".
        amount: chips put in for a raise (None for fold/check/call/allin-size).
        pot: pot size when the action was taken (for pot-fraction snapping).
        bet_sizes: blueprint's postflop bet-size vocab (from .bps metadata).
        add_explicit_allin: trainer's add_explicit_allin flag.
        facing_bet: whether the actor faces an outstanding bet (= has_to_call).
        raises_so_far: raises already made this street (selects the re-raise
            vocab vs first-bet vocab in classify_postflop_actions).

    Raises:
        ValueError: empty bet_sizes, or an unknown action string.
    """
    if not bet_sizes:
        raise ValueError("bet_sizes must be non-empty (blueprint vocab required)")
    if action not in ("fold", "check", "call", "raise", "allin"):
        raise ValueError(f"unknown action {action!r}")

    sizes = tuple(sorted(bet_sizes))

    # Non-sized actions: exact, no off-tree concern.
    if action in ("fold", "check", "call", "allin"):
        label = _action_to_solver_label(action, amount, pot, sizes)
        idx = _solver_label_to_action_index(
            label, facing_bet, sizes, add_explicit_allin,
        )
        return TranslatedAction(idx, ON_TREE_EXACT, label)

    # raise/bet — may be off-tree.
    frac = _bet_frac(amount, pot)
    off_tree = _is_off_tree(frac, sizes)

    label = _action_to_solver_label(action, amount, pot, sizes)
    idx = _solver_label_to_action_index(
        label, facing_bet, sizes, add_explicit_allin,
    )
    kind = OFF_TREE_PSEUDOHARMONIC if off_tree else ON_TREE_EXACT
    return TranslatedAction(idx, kind, label)
