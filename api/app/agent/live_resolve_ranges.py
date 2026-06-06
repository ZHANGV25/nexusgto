"""In-distribution range seeding for LIVE_RESOLVE (advisor fix, 2026-06-06).

A live re-solve seeded with UNIFORM ranges solves the WRONG game: it models a
wide/weak villain, against which over-betting is correct — so the resolved mix
over-bets and (against a non-adapting floor villain) appears to "beat" the floor
when it is really just exploiting a fixed opponent (the same un-bluffable-station
confound as 3b's −67/+321). Seeding the re-solve with the blueprint's
preflop-σ̄ ranges (each seat holds the hands its strategy was trained on) removes
the confound: the resolved mix then ≈ the floor mix at a well-covered spot.

This module computes per-seat 169-class weight dicts ({hand_class_name: weight})
suitable for postflop_provider._build_hand_grid_range(blocked, weights=...). It
reuses the 3b machinery (build_seat_range_weights / the preflop_range 169 map).

SCOPE (v1): the CANONICAL single-open line only (one preflop raise = the open,
then flat callers / folds — the capture-verified shape). The raiser gets its
raise-range weights; each flat-caller gets its call-range weights. Non-canonical
preflop shapes return None (caller falls back to uniform — documented).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from app.agent.game_state import SEAT_NAME
from app.agent.preflop_range import build_seat_range_weights, hand_class_169
from app.services.preflop_provider import HAND_GRID

# Preflop acting order (UTG first), matching game_state._PREFLOP_ORDER.
_PREFLOP_ORDER = ("UTG", "MP", "CO", "BTN", "SB", "BB")

# "Any raise" action indices in the preflop σ̄ vocab (the open can be either
# preflop raise tier); "call" is index 1. Mirrors 3b's RangeDealer.
_RAISE_INDICES = (2, 3)
_CALL_INDICES = (1,)


def _class_name_by_index() -> List[str]:
    """HAND_GRID name for each 169-class index (the inverse of hand_class_169 on
    a representative combo). Built once: HAND_GRID is a bijection onto classes."""
    ranks = "AKQJT98765432"

    def combo(name: str):
        r0 = 12 - ranks.index(name[0])
        r1 = 12 - ranks.index(name[1])
        if len(name) == 2:  # pair
            return r0 * 4 + 0, r1 * 4 + 1
        if name[2] == "s":
            return r0 * 4 + 0, r1 * 4 + 0
        return r0 * 4 + 0, r1 * 4 + 1  # offsuit

    out: List[Optional[str]] = [None] * 169
    for name in HAND_GRID:
        c0, c1 = combo(name)
        out[hand_class_169(c0, c1)] = name
    if any(n is None for n in out):  # pragma: no cover - HAND_GRID is complete
        raise RuntimeError("HAND_GRID does not cover all 169 classes")
    return out  # type: ignore[return-value]


_CLASS_NAME = _class_name_by_index()


def _weights_array_to_name_dict(weights: np.ndarray) -> Dict[str, float]:
    """169-class weight array -> {HAND_GRID_name: weight} for _build_hand_grid_range."""
    return {_CLASS_NAME[i]: float(weights[i]) for i in range(169)}


def _parse_canonical_preflop(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, str]]:
    """Classify each LIVE seat as 'raiser' or 'caller' on the canonical line.

    Replays the prefix in UTG-first order. Returns {seat_name: role} for live
    (non-folded) seats, or None if the line is non-canonical (no raise, or a
    3bet+/limp), in which case range seeding is unsupported and the caller falls
    back to uniform.
    """
    roles: Dict[str, str] = {}
    raises_seen = 0
    idx = 0
    n = len(preflop_prefix)
    for pos in _PREFLOP_ORDER:
        if idx >= n:
            break
        action = preflop_prefix[idx]
        idx += 1
        if action == 0:  # fold (not live)
            continue
        if action == 1:  # flat call
            roles[pos] = "caller"
        elif action == 2:  # raise
            if raises_seen >= 1:
                return None  # 3bet+ — out of canonical scope
            roles[pos] = "raiser"
            raises_seen += 1
        else:
            return None
    if raises_seen != 1:
        return None  # limp pot / no open — unsupported
    return roles


def seat_range_weight_dicts(
    seats_in_slot_order: Sequence[int],
    preflop_prefix: Sequence[int],
    get_strategy,
) -> Optional[List[Dict[str, float]]]:
    """Per-slot {hand_class_name: weight} dicts seeded from the preflop σ̄.

    seats_in_slot_order: full seats (0-5) in compact-slot order (SB-first) —
        i.e. state.seat_of_slot. The returned list aligns to the same order.
    preflop_prefix: the preflop action-index sequence (UTG-first).
    get_strategy: reader.get_strategy(board, action_history, player, bucket, street).

    Returns None for a non-canonical preflop line (caller uses uniform).
    """
    roles = _parse_canonical_preflop(preflop_prefix)
    if roles is None:
        return None

    # The action_history PREFIX up to each seat's decision is the full prefix
    # with that seat's own action (and everything after) removed. For the
    # canonical line the raiser opens facing the blinds (prefix up to it) and a
    # caller calls facing the open. We use build_seat_range_weights with the
    # standard 3b construction: raiser at action_history=[] is the opener only
    # when it is UTG; to stay faithful we pass the prefix UP TO each seat.
    out: List[Dict[str, float]] = []
    for full_seat in seats_in_slot_order:
        name = SEAT_NAME[full_seat]
        role = roles.get(name)
        if role is None:
            return None  # a live seat we couldn't classify — bail to uniform
        history_up_to = _prefix_up_to(preflop_prefix, name)
        if role == "raiser":
            w = build_seat_range_weights(
                get_strategy, action_history=history_up_to,
                player=full_seat, action_indices=list(_RAISE_INDICES),
            )
        else:  # caller
            w = build_seat_range_weights(
                get_strategy, action_history=history_up_to,
                player=full_seat, action_indices=list(_CALL_INDICES),
            )
        if float(np.asarray(w).sum()) <= 0.0:
            return None  # empty range read — fall back to uniform, fail-safe
        out.append(_weights_array_to_name_dict(w))
    return out


def _prefix_up_to(preflop_prefix: Sequence[int], seat_name: str) -> List[int]:
    """The preflop action_history slice BEFORE `seat_name`'s own decision.

    Walks UTG-first, accumulating each acting seat's index until reaching
    `seat_name`; returns the accumulated prefix (the decision node's history).
    """
    acc: List[int] = []
    idx = 0
    n = len(preflop_prefix)
    for pos in _PREFLOP_ORDER:
        if idx >= n:
            break
        if pos == seat_name:
            return acc
        acc.append(preflop_prefix[idx])
        idx += 1
    return acc
