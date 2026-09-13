"""Self-play AIVAT scoring for offline six-max matches.

``run_match``'s duplicate decks, seat rotation and common policy RNGs still
leave a deal-group standard deviation of ~2-4.5 bb/hand: a blueprint playing a
random action 5% of the time was indistinguishable from the blueprint at
3,000 groups (2026-09-12). Offline, every hole card and every seat's sampling
distribution is known, so AIVAT (Burch et al., AAAI 2018) can remove the luck
of each sampled action and of each revealed turn and river card::

    v_hat = u(z) + sum_decisions [ sum_a p(a) V(h.a) - V(h.a_taken) ]
                 + sum_turn_river [ V_k(h) - V_k+1(h.c) ]

Each term has zero mean given the history, so ``v_hat`` is unbiased for ANY
value function ``V``. ``V`` here is the omniscient frozen-pot value: every
seat's expected net chips if betting stopped now and the unrevealed board ran
out uniformly over the cards no seat holds, with exact side-pot layering and
equal splits. Flop, turn and river values enumerate every runout, so
``E_c V_k+1(h.c) == V_k(h)`` and the turn/river chance terms are exact.
Preflop values use a fixed board sample seeded by the hole cards only; the
flop reveal gets no chance term (omitting a term keeps the estimate unbiased).

Terms sum to zero across seats, so corrected deltas keep ``run_match``'s
zero-sum and exact AA-null invariants and ride its ``adjusted_deltas`` path.
"""
from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from scripts import run_match  # noqa: E402

N_SEATS = run_match.N_SEATS
PREFLOP_BOARD_SAMPLES = 512
ESTIMATOR = "aivat_frozen_pot_omniscient_v1"
_BOARD_COUNT_BY_STREET = {0: 0, 1: 3, 2: 4, 3: 5}
_C_LIBRARY_NAMES = ("hand_eval_batch.so", "hand_eval_batch.dll")

Chips = Tuple[Tuple[int, ...], Tuple[bool, ...]]


class AivatUnpricedDecision(ValueError):
    """A decision's sampling distribution is unknown, so AIVAT cannot score it."""


# ── 7-card evaluation ─────────────────────────────────────────────────────


def _python_eval7_batch(cards: np.ndarray) -> np.ndarray:
    """Dense strength ranks for a batch of 7-card hands (slow fallback).

    Ranks are comparable only within one call, which is all ``runout_ranks``
    needs: it evaluates every seat of every runout together.
    """

    from engine.leaf_values import _eval7_py  # type: ignore[import-untyped]

    raw = [_eval7_py([int(card) for card in row]) for row in cards]
    order = {value: rank for rank, value in enumerate(sorted(set(raw)))}
    return np.asarray([order[value] for value in raw], dtype=np.int64)


def _load_c_eval7() -> Optional[Callable[[np.ndarray], np.ndarray]]:
    root = os.environ.get("POKER_SOLVER_PATH")
    if not root:
        return None
    for name in _C_LIBRARY_NAMES:
        path = Path(root) / "build" / name
        if not path.is_file():
            continue
        batch = ctypes.CDLL(str(path)).poker_eval7_batch
        batch.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        batch.restype = ctypes.c_int

        def evaluate(cards: np.ndarray) -> np.ndarray:
            flat = np.ascontiguousarray(cards, dtype=np.int32)
            out = np.empty(len(flat), dtype=np.uint32)
            rc = batch(
                flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                len(flat),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            )
            if rc != 0:
                raise RuntimeError(f"poker_eval7_batch failed with rc={rc}")
            return out.astype(np.int64)

        return evaluate
    return None


_evaluator: Optional[Callable[[np.ndarray], np.ndarray]] = None


def evaluate_7card_batch(cards: np.ndarray) -> np.ndarray:
    """Strength per 7-card row (higher wins); C batch when built, else Python."""

    global _evaluator
    if _evaluator is None:
        _evaluator = _load_c_eval7() or _python_eval7_batch
    return _evaluator(cards)


# ── Frozen-pot values ─────────────────────────────────────────────────────


def frozen_pot_values(
    ranks: np.ndarray, contributions: Sequence[int], folded: Sequence[bool]
) -> np.ndarray:
    """Mean net chips per seat over runouts, settling the pot as it stands.

    ``ranks`` is ``(runouts, seats)``. Live contribution levels delimit the
    main and side pots exactly as ``settlement.settle_hand`` does; an
    uncalled excess forms a layer only its bettor is eligible for, so it
    returns to them. Ties split equally (no odd-chip rule).
    """

    contrib = np.asarray(contributions, dtype=np.float64)
    live = ~np.asarray(folded, dtype=bool)
    if not live.any():
        raise ValueError("frozen-pot value needs at least one live seat")
    awards = np.zeros(len(contrib), dtype=np.float64)
    if live.sum() == 1:
        awards[np.flatnonzero(live)[0]] = contrib.sum()
        return awards - contrib
    previous = 0.0
    for level in sorted({float(c) for c in contrib[live] if c > 0}):
        layer = float(np.clip(np.minimum(contrib, level) - previous, 0.0, None).sum())
        eligible = live & (contrib >= level)
        masked = np.where(eligible[None, :], ranks, -1)
        best = masked.max(axis=1, keepdims=True)
        winners = (masked == best) & eligible[None, :]
        shares = winners / winners.sum(axis=1, keepdims=True)
        awards += shares.mean(axis=0) * layer
        previous = level
    if not np.isclose(awards.sum(), contrib.sum()):
        raise ValueError(
            "folded contributions exceed every live level; frozen-pot value "
            f"cannot conserve chips (awards={awards.sum()}, pot={contrib.sum()})"
        )
    return awards - contrib


def _unseen_cards(deal: run_match.Deal) -> Tuple[int, ...]:
    held = {int(card) for hand in deal.hole_cards for card in hand}
    return tuple(card for card in range(52) if card not in held)


def runout_boards(
    deal: run_match.Deal,
    board_count: int,
    samples: int = PREFLOP_BOARD_SAMPLES,
) -> np.ndarray:
    """All completions of the revealed board, or a hole-seeded preflop sample."""

    unseen = _unseen_cards(deal)
    if board_count == 0:
        seed = np.random.SeedSequence(
            [int(card) for hand in deal.hole_cards for card in hand]
        )
        rng = np.random.default_rng(seed)
        pool = np.asarray(unseen, dtype=np.int32)
        return np.stack([rng.choice(pool, 5, replace=False) for _ in range(samples)])
    if board_count not in (3, 4, 5):
        raise ValueError(f"board_count must be 0, 3, 4 or 5, got {board_count}")
    known = tuple(int(card) for card in deal.board[:board_count])
    remaining = [card for card in unseen if card not in known]
    fills = list(combinations(remaining, 5 - board_count))
    return np.asarray([known + fill for fill in fills], dtype=np.int32).reshape(
        len(fills), 5
    )


def runout_ranks(deal: run_match.Deal, board_count: int) -> np.ndarray:
    """Hand strength ``(runouts, seats)`` for every seat on every runout."""

    boards = runout_boards(deal, board_count)
    holes = np.asarray(deal.hole_cards, dtype=np.int32)
    n_runouts = len(boards)
    cards = np.concatenate(
        [
            np.repeat(boards[:, None, :], N_SEATS, axis=1),
            np.broadcast_to(holes, (n_runouts, N_SEATS, 2)),
        ],
        axis=2,
    )
    strengths = evaluate_7card_batch(cards.reshape(n_runouts * N_SEATS, 7))
    return np.asarray(strengths, dtype=np.int64).reshape(n_runouts, N_SEATS)


class DealValues:
    """Memoized frozen-pot values for one deal, shared by its six arms."""

    def __init__(self, deal: run_match.Deal) -> None:
        self.deal = deal
        self._ranks: Dict[int, np.ndarray] = {}
        self._values: Dict[Tuple[Tuple[int, ...], Tuple[bool, ...], int], np.ndarray] = {}

    def value(
        self, contributions: Sequence[int], folded: Sequence[bool], board_count: int
    ) -> np.ndarray:
        key = (
            tuple(int(chips) for chips in contributions),
            tuple(bool(flag) for flag in folded),
            int(board_count),
        )
        cached = self._values.get(key)
        if cached is None:
            ranks = self._ranks.get(key[2])
            if ranks is None:
                ranks = runout_ranks(self.deal, key[2])
                self._ranks[key[2]] = ranks
            cached = frozen_pot_values(ranks, key[0], key[1])
            cached.setflags(write=False)
            self._values[key] = cached
        return cached


_deal_cache: Dict[run_match.Deal, DealValues] = {}


def reset_cache() -> None:
    _deal_cache.clear()


def _deal_values(deal: run_match.Deal) -> DealValues:
    cached = _deal_cache.get(deal)
    if cached is None:
        _deal_cache.clear()  # one deal group at a time
        cached = DealValues(deal)
        _deal_cache[deal] = cached
    return cached


# ── Decision recording and corrections ───────────────────────────────────


@dataclass(frozen=True)
class DecisionRecord:
    state: object
    taken: int
    probs: Optional[Tuple[float, ...]]


class RecordingPolicy:
    """Log each decision's state and true sampling distribution; play unchanged."""

    def __init__(self, inner: object, log: List[DecisionRecord]) -> None:
        self._inner = inner
        self._log = log

    def act(self, state: object, hero_cards: run_match.HoleCards) -> object:
        result = self._inner.act(state, hero_cards)
        n_legal = int(state.num_legal_actions())
        self._log.append(
            DecisionRecord(
                state=state,
                taken=run_match._action_index(result),
                probs=run_match.sampling_probs_for(result, n_legal),
            )
        )
        return result

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def _is_preflop(state: object) -> bool:
    return int(getattr(state, "street", -1)) == 0


def _board_count(state: object) -> int:
    return _BOARD_COUNT_BY_STREET[min(int(state.street), 3)]


def _chips(state: object, preflop_end: Optional[object]) -> Chips:
    """Full-seat contributions and fold flags for a preflop or postflop state."""

    if _is_preflop(state):
        return tuple(state.contributions), tuple(state.folded)
    if preflop_end is None:
        raise ValueError("postflop decision recorded before the preflop terminal")
    contributions = [int(chips) for chips in preflop_end.contributions]
    folded = [bool(flag) for flag in preflop_end.folded]
    for slot, seat in enumerate(state.seat_of_slot):
        contributions[seat] += int(preflop_end.stacks[seat]) - int(state.stacks[slot])
        folded[seat] = not bool(state.active[slot])
    return tuple(contributions), tuple(folded)


def _chance_terms(
    state: object, next_state: object, chips: Chips, values: DealValues
) -> np.ndarray:
    total = np.zeros(N_SEATS)
    for street in range(int(state.street), min(int(next_state.street), 3)):
        before = values.value(*chips, _BOARD_COUNT_BY_STREET[street])
        after = values.value(*chips, _BOARD_COUNT_BY_STREET[street + 1])
        total = total + before - after
    return total


def aivat_corrections(
    deal: run_match.Deal, records: Sequence[DecisionRecord]
) -> np.ndarray:
    """Sum of AIVAT decision and turn/river chance terms for all six seats."""

    values = _deal_values(deal)
    total = np.zeros(N_SEATS)
    preflop_end: Optional[object] = None
    for record in records:
        state = record.state
        probs = record.probs
        if probs is None:
            raise AivatUnpricedDecision(
                "decision has no known sampling distribution; AIVAT needs every "
                "seat's policy mix"
            )
        if not 0 <= record.taken < len(probs) or probs[record.taken] <= 0.0:
            raise ValueError(
                f"taken action {record.taken} has no probability in {probs}"
            )
        preflop = _is_preflop(state)
        context = None if preflop else preflop_end
        board_count = _board_count(state)
        children = {
            action: state.advance(action)
            for action, p in enumerate(probs)
            if p > 0.0
        }
        child_values = {
            action: values.value(*_chips(child, context), board_count)
            for action, child in children.items()
        }
        expected = sum(probs[action] * value for action, value in child_values.items())
        total = total + expected - child_values[record.taken]
        taken_state = children[record.taken]
        if preflop:
            preflop_end = taken_state
        else:
            total = total + _chance_terms(
                state, taken_state, _chips(taken_state, context), values
            )
    return total - total.sum() / N_SEATS


def play_full_hand_aivat(
    deal: run_match.Deal,
    policies: Sequence[object],
    *,
    allin_equity_adjust: bool = False,
) -> run_match.PlayedHand:
    """``run_match.play_full_hand`` with AIVAT-corrected ``adjusted_deltas``.

    AIVAT's exact turn/river terms subsume the all-in equity adjustment, so the
    flag is accepted for hand-player compatibility and otherwise ignored.
    """

    del allin_equity_adjust
    log: List[DecisionRecord] = []
    recorders = [RecordingPolicy(policy, log) for policy in policies]
    hand = run_match.play_full_hand(deal, recorders)
    corrections = aivat_corrections(deal, log)
    adjusted = tuple(
        float(delta) + float(correction)
        for delta, correction in zip(hand.deltas, corrections)
    )
    return replace(hand, adjusted_deltas=adjusted)
