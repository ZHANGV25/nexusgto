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

SCOPE (2026-06-11, review fix — was single-open only): every preflop shape the
engine SERVES is now range-parsed:
  * CANONICAL SINGLE-OPEN (one raise + flat callers): raiser gets its
    raise-range weights, each flat-caller its call-range weights (the original
    validated v1 behavior, unchanged — including the "any raise" [2,3]
    convention mirrored from 3b's RangeDealer).
  * 3-BET HEADS-UP (open -> 3-bet idx 2 or 3 -> folds -> opener calls): the
    3-bettor gets P(its EXACT 3-bet index | class) at the vs-open node; the
    opener gets the chain-rule PRODUCT P(open idx 2 | class) x P(call | class)
    at its two decision nodes (the blueprint HAS sigma-bar at both — the
    vs-open and 3-bet-response nodes are capture-verified present + trained,
    preflop_shapes_captured.json 2026-06-10).
  * HU LIMP-CHECK (one limper -> folds -> BB checks): the limper gets
    P(call | class); the BB gets P(check | class) — the check is INDEX 0 at
    the BB-option node (trainer layout [check, 0.7x, 1.0x], capture-verified
    limp_shapes_captured.json 2026-06-11).
Anything else returns None (caller falls back to uniform — and the engine
provider DISCLOSES uniform seeding in the response quality note).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.engine.agent.game_state import SEAT_NAME
from app.engine.agent.preflop_range import build_seat_range_weights, hand_class_169
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
    """Classify each LIVE seat as 'raiser' or 'caller' on the SINGLE-OPEN line.

    Replays the prefix in UTG-first order. Returns {seat_name: role} for live
    (non-folded) seats, or None if the line is not a single open (no raise, or
    a 3bet+/limp shape — those are handled by _parse_preflop_decisions, which
    this feeds for the single-open case).
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


# One range-defining decision: (action_history up to the node, the action
# index/indices the seat actually took there). A seat with several decisions
# (the 3-bet opener: open, then call) gets the chain-rule PRODUCT of its
# per-decision P(action | class) weights.
SeatDecision = Tuple[List[int], Tuple[int, ...]]


def _single_open_decisions(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, List[SeatDecision]]]:
    """Single-open decisions — EXACTLY the original v1 behavior (raiser =
    'any raise' [2,3], callers = [1], history = prefix up to the seat)."""
    roles = _parse_canonical_preflop(preflop_prefix)
    if roles is None:
        return None
    out: Dict[str, List[SeatDecision]] = {}
    for name, role in roles.items():
        history = _prefix_up_to(preflop_prefix, name)
        idxs = _RAISE_INDICES if role == "raiser" else _CALL_INDICES
        out[name] = [(history, tuple(idxs))]
    return out


def _threebet_hu_decisions(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, List[SeatDecision]]]:
    """3-bet HU decisions (open -> 3-bet -> folds -> opener calls, length 7).

    Node histories are pinned by the trainer capture (BTN-open/BB-3bet sz1
    walk, preflop_shapes_captured.json): the opener decides at
    prefix[:opener] (open, index 2) and again at prefix[:6] (the closing
    call, index 1, facing the 3-bet); the 3-bettor decides at prefix[:tb]
    with its EXACT index (2 = 0.7x, 3 = 1.0x — size-specific, unlike the
    single-open 'any raise' convention, because the prefix records which
    size was served).
    """
    prefix = list(preflop_prefix)
    if len(prefix) != 7 or prefix[6] != 1:
        return None
    raise_is = [i for i, a in enumerate(prefix[:6]) if a in (2, 3)]
    if len(raise_is) != 2:
        return None
    opener_i, tb_i = raise_is
    if prefix[opener_i] != 2:
        return None  # a 1.0x OPEN is not a served/verified shape
    if any(prefix[i] != 0 for i in range(6) if i not in raise_is):
        return None  # a caller in a 3-bet pot = squeeze — not served
    opener = _PREFLOP_ORDER[opener_i]
    threebettor = _PREFLOP_ORDER[tb_i]
    return {
        opener: [
            (prefix[:opener_i], (2,)),
            (prefix[:6], (1,)),  # closing call, facing the 3-bet
        ],
        threebettor: [(prefix[:tb_i], (int(prefix[tb_i]),))],
    }


def _limp_check_decisions(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, List[SeatDecision]]]:
    """HU limp-check decisions (one limper -> folds -> BB checks, length 6).

    The BB's check is INDEX 0 at the BB-option node — the trainer layout
    there is [check, 0.7x, 1.0x] (capture-verified limp_shapes_captured.json
    2026-06-11), so P(check | class) is sigma[0], NOT the facing-bet call
    index.
    """
    prefix = list(preflop_prefix)
    if len(prefix) != 6 or any(a not in (0, 1) for a in prefix):
        return None
    if prefix[5] != 0:
        return None  # the BB slot must be the option CHECK
    limper_is = [i for i, a in enumerate(prefix[:5]) if a == 1]
    if len(limper_is) != 1:
        return None  # multi-limper pots are not served (root-captured only)
    i = limper_is[0]
    return {
        _PREFLOP_ORDER[i]: [(prefix[:i], (1,))],
        "BB": [(prefix[:5], (0,))],
    }


def _parse_preflop_decisions(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, List[SeatDecision]]]:
    """Per-LIVE-seat range-defining decisions for every SERVED preflop shape.

    Returns {seat_name: [(history, action_indices), ...]} or None when the
    shape is not range-parsed (caller falls back to uniform, disclosed).
    """
    prefix = list(preflop_prefix)
    if not prefix or any(a not in (0, 1, 2, 3) for a in prefix):
        return None
    n_raises = sum(1 for a in prefix[:6] if a in (2, 3))
    if n_raises == 0:
        return _limp_check_decisions(prefix)
    if n_raises == 1:
        return _single_open_decisions(prefix)
    if n_raises == 2:
        return _threebet_hu_decisions(prefix)
    return None


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

    A seat's weight is the chain-rule product of P(taken action | class) over
    its decision nodes (one node for openers/callers/limpers; two for the
    3-bet opener: open x closing call).

    Returns None for a preflop line outside the parsed shapes (caller uses
    uniform — disclosed in the provider's quality note).
    """
    decisions = _parse_preflop_decisions(preflop_prefix)
    if decisions is None:
        return None

    out: List[Dict[str, float]] = []
    for full_seat in seats_in_slot_order:
        name = SEAT_NAME[full_seat]
        seat_decisions = decisions.get(name)
        if seat_decisions is None:
            return None  # a live seat we couldn't classify — bail to uniform
        w = np.ones(169, dtype=np.float64)
        for history, action_indices in seat_decisions:
            w = w * build_seat_range_weights(
                get_strategy, action_history=list(history),
                player=full_seat, action_indices=list(action_indices),
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
