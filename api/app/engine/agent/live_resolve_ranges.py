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

SCOPE (2026-07-11, generic rewrite — was three hard-coded shapes): range
seeding is now GENERIC over the ENTIRE shipped preflop vocabulary. Instead of a
bespoke per-shape parser, the prefix is walked through the SHARED trainer-exact
transition (app.engine.agent.preflop_play). At each step we record, for the
acting seat, its decision node = (trainer-encoded action_history so far, the
EXACT taken action index). A seat's 169-class weights are then the chain-rule
PRODUCT of build_seat_range_weights over its decisions — exactly as before.

This covers ANY prefix the shared replay accepts:
  * canonical single-open (one raise + flat callers),
  * multi-limp pots (2+ limpers -> BB option),
  * limp-raise lines INCLUDING the limper's closing call of the raise,
  * squeezes (caller(s) then a re-raise),
  * multiway 3-bets, and capped 4-bet lines.
Anything the shared transition rejects (an illegal/out-of-tree prefix) returns
None, as does any seat whose read range has zero mass or that cannot be
classified. The caller then falls back to uniform ranges — and the engine
provider DISCLOSES uniform seeding in the response quality note.

LEGACY-COMPATIBILITY NOTE (byte-identical weights on the three original
shapes): the original v1 single-open path summed the "any raise" indices
(2, 3) for the opener's weights (mirrored from 3b's RangeDealer), whereas the
3-bet-HU and limp-check paths used the EXACT taken index. The generic walk uses
the exact index everywhere and then applies ONE compatibility widening: on a
CANONICAL single-open shape (exactly one raise in the whole prefix, that raise
is the 0.7x open index 2, and every other seat only folded or flat-called), the
opener's lone raise decision is widened from (2,) to (2, 3). Every other shape —
including any pot with a re-raise — keeps exact indices. This reproduces the
three legacy shapes bit-for-bit (see test_live_resolve_translation.py golden
tests) while extending seeding to the rest of the vocabulary.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.engine.agent.game_state import SEAT_NAME
from app.engine.agent.preflop_play import (
    PreflopError,
    new_preflop_state,
)
from app.engine.agent.preflop_range import build_seat_range_weights, hand_class_169
from app.services.preflop_provider import HAND_GRID

# Preflop acting order (UTG first), matching game_state._PREFLOP_ORDER. Kept for
# the legacy golden helpers below; the generic walk uses the shared transition's
# own PREFLOP_ORDER (full-seat ints) instead of these names.
_PREFLOP_ORDER = ("UTG", "MP", "CO", "BTN", "SB", "BB")

# "Any raise" action indices in the preflop σ̄ vocab (the open can be either
# preflop raise tier); "call" is index 1. Mirrors 3b's RangeDealer.
_RAISE_INDICES = (2, 3)
_CALL_INDICES = (1,)

# The single 0.7x open index the legacy canonical single-open path recognized.
_OPEN_INDEX = 2


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


# One range-defining decision: (action_history up to the node, the action
# index/indices the seat actually took there). A seat with several decisions
# (a limp-raiser: open-limp, then closing call; a 3-bet opener: open, then
# call) gets the chain-rule PRODUCT of its per-decision P(action | class)
# weights.
SeatDecision = Tuple[List[int], Tuple[int, ...]]


def _generic_decisions(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, List[SeatDecision]]]:
    """Per-LIVE-seat range-defining decisions, extracted GENERICALLY by walking
    the SHARED trainer-exact transition.

    Walks ``preflop_prefix`` from the initial six-max state via the shared
    ``advance``. At each step the acting seat and the exact taken index are
    known; we record (action_history_before_this_step, (taken_index,)) for that
    seat and advance. Folds are NOT recorded (a folded seat is not live and has
    no seeded range). Returns None if the prefix is not a legal replay through
    the trainer tree (out-of-range index, action after the round closed, etc.).

    The returned map keys are seat NAMES (SB/BB/UTG/MP/CO/BTN); a seat that
    acted more than once (limp-then-call, open-then-call) accumulates a
    decision per node, in order, for the chain-rule product.

    Callers that need the legacy "any raise" (2, 3) widening on a canonical
    single-open apply it via :func:`_apply_single_open_compat`.
    """
    prefix = list(preflop_prefix)
    if not prefix:
        return None
    try:
        state = new_preflop_state()
        decisions: Dict[str, List[SeatDecision]] = {}
        for action_idx in prefix:
            seat = state.to_act_seat()
            if seat is None:
                # An action after the hand/round already closed — not a legal
                # replay. Bail to uniform.
                return None
            history_before = list(state.action_history)
            # advance validates the index against the node's legal actions and
            # raises PreflopReplayError if it is out of range / illegal.
            next_state = state.advance(action_idx)
            # A fold removes the seat from the live set; folded seats hold no
            # seeded range, so we do not record their decision.
            if action_idx != 0:
                name = SEAT_NAME[seat]
                decisions.setdefault(name, []).append(
                    (history_before, (int(action_idx),))
                )
            elif not next_state.active[seat]:
                # index 0 that was a FOLD (facing a bet): drop it. When index 0
                # is a CHECK/limp-option instead (nothing to call), the seat is
                # still active and we DO record it below.
                pass
            else:  # index 0 == check (the BB option / SB-nothing-to-call node)
                name = SEAT_NAME[seat]
                decisions.setdefault(name, []).append(
                    (history_before, (int(action_idx),))
                )
            state = next_state
        # Only a CLOSED multiway pot (round done, 2+ live) is ever seeded — that
        # is the exact surface a postflop re-solve consumes. An unclosed prefix
        # (round still open) or a fold-around (1 live) never reaches postflop
        # seeding; reject both so we don't seed a mid-orbit or dead line.
        if not state.flop_ready:
            return None
        return decisions
    except (PreflopError, KeyError, ValueError):
        # PreflopReplayError (a PreflopError) for an illegal prefix; any other
        # replay/index error is likewise treated as unparseable -> uniform.
        return None


def _is_canonical_single_open(preflop_prefix: Sequence[int]) -> bool:
    """True iff the prefix is the legacy canonical single-open shape.

    That is EXACTLY what the retired ``_parse_canonical_preflop`` accepted:
    ONE orbit (each of the six seats acts exactly once, so exactly six
    actions), exactly one raise, that raise is the 0.7x open (index 2), and
    every other action is a fold (0) or a flat call (1) — no re-raise, no
    3-bet+, no limp-only pot. This is the ONLY shape on which the opener's
    raise weights are widened to the "any raise" (2, 3) convention.

    The length check is load-bearing: a limp-raise line ([1,2,0,0,0,0,1] —
    the limper acts TWICE) has one raise in its first six entries and would
    otherwise masquerade as a single-open, wrongly widening the raiser's
    exact (2,) to (2, 3). Limp-raise was never a legacy-served shape, so the
    widening quirk must not extend to it.
    """
    if len(preflop_prefix) != len(_PREFLOP_ORDER):
        return False
    raises_seen = 0
    idx = 0
    n = len(preflop_prefix)
    for _pos in _PREFLOP_ORDER:
        if idx >= n:
            break
        action = preflop_prefix[idx]
        idx += 1
        if action == 0:  # fold
            continue
        if action == 1:  # flat call
            continue
        if action == _OPEN_INDEX:  # 0.7x open
            if raises_seen >= 1:
                return False
            raises_seen += 1
        else:
            # a 1.0x open (index 3) or any other raise index -> not the legacy
            # canonical single-open shape.
            return False
    return raises_seen == 1


def _apply_single_open_compat(
    preflop_prefix: Sequence[int],
    decisions: Dict[str, List[SeatDecision]],
) -> Dict[str, List[SeatDecision]]:
    """Reproduce the legacy single-open "any raise" (2, 3) opener widening.

    On a canonical single-open shape, the opener has exactly ONE decision whose
    taken index is the 0.7x open (2); the legacy path seeded its weights from
    P(sum of indices 2, 3 | class). We widen that lone decision's index tuple
    from (2,) to (2, 3). Every other seat and every non-single-open shape is
    returned unchanged (exact indices).
    """
    if not _is_canonical_single_open(preflop_prefix):
        return decisions
    out: Dict[str, List[SeatDecision]] = {}
    for name, seat_decisions in decisions.items():
        new_list: List[SeatDecision] = []
        for history, action_indices in seat_decisions:
            if action_indices == (_OPEN_INDEX,):
                new_list.append((history, _RAISE_INDICES))
            else:
                new_list.append((history, action_indices))
        out[name] = new_list
    return out


def _parse_preflop_decisions(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, List[SeatDecision]]]:
    """Per-LIVE-seat range-defining decisions for EVERY replayable preflop shape.

    Returns {seat_name: [(history, action_indices), ...]} or None when the
    prefix is not a legal replay through the shared trainer transition (caller
    falls back to uniform, disclosed).

    Generic path: walk the shared transition to collect exact-index decisions,
    then apply the single legacy compatibility widening (opener -> "any raise"
    (2, 3)) on a canonical single-open shape so those weights stay byte-identical
    to the retired bespoke parser.
    """
    decisions = _generic_decisions(preflop_prefix)
    if decisions is None:
        return None
    if not decisions:
        # A legal replay with no live-seat decisions (e.g. an all-fold walk that
        # somehow closed) — nothing to seed, fall back to uniform.
        return None
    return _apply_single_open_compat(preflop_prefix, decisions)


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
    its decision nodes (one node per action; multiple for a limp-raiser or a
    3-bet opener that acts twice).

    Returns None for a preflop line the shared transition rejects, or when any
    live slot cannot be classified or reads an empty (zero-mass) range (caller
    uses uniform — disclosed in the provider's quality note).
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


# ── Retired bespoke parsers (kept PRIVATE for golden equality tests only) ────
#
# These are the three hard-coded shape parsers the generic walk replaced. They
# are NOT called by any serving path — they exist solely so the test-suite can
# assert the generic extractor reproduces their weight dicts byte-for-byte on
# the canonical single-open, 3-bet-HU, and HU limp-check shapes. Do not wire
# them back into seat_range_weight_dicts.


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


def _parse_canonical_preflop(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, str]]:
    """LEGACY (retired): classify each LIVE seat as 'raiser'/'caller' on the
    single-open line. Kept private for golden tests."""
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


def _legacy_single_open_decisions(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, List[SeatDecision]]]:
    """LEGACY (retired): single-open decisions (raiser = 'any raise' (2, 3),
    callers = (1,)). Kept private for golden tests."""
    roles = _parse_canonical_preflop(preflop_prefix)
    if roles is None:
        return None
    out: Dict[str, List[SeatDecision]] = {}
    for name, role in roles.items():
        history = _prefix_up_to(preflop_prefix, name)
        idxs = _RAISE_INDICES if role == "raiser" else _CALL_INDICES
        out[name] = [(history, tuple(idxs))]
    return out


def _legacy_threebet_hu_decisions(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, List[SeatDecision]]]:
    """LEGACY (retired): 3-bet HU decisions (open -> 3-bet -> folds -> opener
    calls, length 7). Kept private for golden tests."""
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


def _legacy_limp_check_decisions(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, List[SeatDecision]]]:
    """LEGACY (retired): HU limp-check decisions (one limper -> folds -> BB
    checks, length 6). Kept private for golden tests."""
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


def _legacy_parse_preflop_decisions(
    preflop_prefix: Sequence[int],
) -> Optional[Dict[str, List[SeatDecision]]]:
    """LEGACY (retired): the retired raise-count dispatcher. Kept private so the
    golden tests can assert the generic extractor matches it bit-for-bit on the
    three original shapes."""
    prefix = list(preflop_prefix)
    if not prefix or any(a not in (0, 1, 2, 3) for a in prefix):
        return None
    n_raises = sum(1 for a in prefix[:6] if a in (2, 3))
    if n_raises == 0:
        return _legacy_limp_check_decisions(prefix)
    if n_raises == 1:
        return _legacy_single_open_decisions(prefix)
    if n_raises == 2:
        return _legacy_threebet_hu_decisions(prefix)
    return None
