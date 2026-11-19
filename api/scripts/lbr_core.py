"""Local best response (LBR) primitives: combo ranges, equities, action values.

LBR (Lisy and Bowling, 2017) plays against a fixed policy whose strategy for
every hand is known at each node.  At its own decision it scores each legal
action under a one-step model: the policy responds according to its strategy
at the next node, and afterwards both players check/call to showdown.  Its
realised winnings over many hands are a lower bound on the policy's
exploitability.  This module knows nothing about the engine's state classes;
``scripts/lbr_game.py`` adapts them.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

N_CARDS = 52
COMBOS: Tuple[Tuple[int, int], ...] = tuple(
    (low, high) for low in range(N_CARDS) for high in range(low + 1, N_CARDS)
)
N_COMBOS = len(COMBOS)
COMBO_INDEX: Dict[Tuple[int, int], int] = {combo: i for i, combo in enumerate(COMBOS)}
COMBO_BITS = np.array([(1 << a) | (1 << b) for a, b in COMBOS], dtype=np.uint64)
COMBO_CARDS = np.array(COMBOS, dtype=np.int32)

Evaluator = Callable[[np.ndarray], np.ndarray]


def combo_index(c0: int, c1: int) -> int:
    return COMBO_INDEX[(int(c0), int(c1)) if c0 < c1 else (int(c1), int(c0))]


def cards_bits(cards: Sequence[int]) -> np.uint64:
    return np.uint64(sum(1 << int(card) for card in cards))


def live_mask(blocked: Sequence[int]) -> np.ndarray:
    """Combos sharing no card with ``blocked``."""
    return (COMBO_BITS & cards_bits(blocked)) == 0


@dataclass(frozen=True)
class ComboRange:
    """Unnormalised weights over the 1326 hole-card combos of one player."""

    weights: np.ndarray

    @classmethod
    def uniform(cls, blocked: Sequence[int]) -> "ComboRange":
        return cls(live_mask(blocked).astype(np.float64))

    def conditioned(self, probs: np.ndarray) -> "ComboRange":
        """Multiply every combo by the probability of the action it just took."""
        return ComboRange(self.weights * np.asarray(probs, dtype=np.float64))

    def normalized(self, blocked: Sequence[int] = ()) -> np.ndarray:
        weights = np.where(live_mask(blocked), self.weights, 0.0)
        total = float(weights.sum())
        return weights / total if total > 0.0 else weights

    def mass(self) -> float:
        return float(self.weights.sum())

    def effective_rows(self, blocked: Sequence[int] = ()) -> float:
        probs = self.normalized(blocked)
        total = float((probs * probs).sum())
        return 1.0 / total if total > 0.0 else 0.0


def sample_runouts(
    rng: np.random.Generator, dead: Sequence[int], count: int, max_samples: int,
) -> np.ndarray:
    """``[n, count]`` runout cards drawn from the deck minus ``dead``.

    Exhaustive when the number of runouts is at most ``max_samples`` (always
    for zero or one card), otherwise ``max_samples`` uniform draws without
    replacement inside each runout.
    """
    dead_set = {int(card) for card in dead}
    deck = np.array([card for card in range(N_CARDS) if card not in dead_set], dtype=np.int32)
    if count == 0:
        return np.zeros((1, 0), dtype=np.int32)
    if count == 1:
        return deck.reshape(-1, 1)
    if math.comb(len(deck), count) <= max_samples:
        return np.array(list(itertools.combinations(deck.tolist(), count)), dtype=np.int32)
    draws = [rng.choice(deck, size=count, replace=False) for _ in range(max_samples)]
    return np.array(draws, dtype=np.int32)


def equity_vs_combos(
    hero: Tuple[int, int], board: Sequence[int], runouts: np.ndarray, evaluate: Evaluator,
) -> np.ndarray:
    """P(hero wins) + P(tie)/2 against every combo, averaged over its valid runouts.

    A runout that uses one of a combo's cards is skipped for that combo; combos
    blocked by the hero's cards or the board get zero.
    """
    board_list = [int(card) for card in board]
    n_run = int(runouts.shape[0])
    full = np.hstack([np.tile(np.array(board_list, dtype=np.int32), (n_run, 1)), runouts.astype(np.int32)])
    if full.shape[1] != 5:
        raise ValueError(f"board plus runout must be five cards, got {full.shape[1]}")
    hero_cards = np.array(hero, dtype=np.int32)
    hero_strength = evaluate(np.hstack([full, np.tile(hero_cards, (n_run, 1))]))
    live = live_mask(list(hero) + board_list)
    idx = np.nonzero(live)[0]
    combos = COMBO_CARDS[idx]
    run_bits = np.array([cards_bits(runout) for runout in runouts], dtype=np.uint64)
    valid = (COMBO_BITS[idx][None, :] & run_bits[:, None]) == 0
    # The C evaluator must never see a duplicated card: a combo that shares a
    # card with the runout is replaced by the hero's own cards and masked out.
    hole = np.where(valid[:, :, None], np.broadcast_to(combos[None, :, :], (n_run, len(idx), 2)), hero_cards[None, None, :])
    rows = np.concatenate([np.repeat(full[:, None, :], len(idx), axis=1), hole], axis=2).reshape(-1, 7)
    strength = evaluate(np.ascontiguousarray(rows, dtype=np.int32)).reshape(n_run, len(idx))
    score = np.where(
        hero_strength[:, None] > strength, 1.0, np.where(hero_strength[:, None] == strength, 0.5, 0.0)
    )
    counts = valid.sum(axis=0)
    equity = np.zeros(N_COMBOS, dtype=np.float64)
    safe = np.where(counts > 0, counts, 1)
    equity[idx] = np.where(counts > 0, (score * valid).sum(axis=0) / safe, 0.0)
    return equity


@dataclass(frozen=True)
class Outcome:
    """What the exploiter collects along one modelled path from its decision.

    ``kind``: "fold" (exploiter folds), "p_fold" (policy folds), "showdown".
    ``l_cost``: chips the exploiter adds from the decision point on.
    ``pot``: the pot at the end of the path, every chip included.
    """

    kind: str
    l_cost: int
    pot: int


def outcome_values(outcome: Outcome, equity: np.ndarray) -> np.ndarray:
    """Net chips for the exploiter against every combo along ``outcome``."""
    if outcome.kind == "fold":
        return np.zeros(N_COMBOS, dtype=np.float64)
    if outcome.kind == "p_fold":
        return np.full(N_COMBOS, float(outcome.pot - outcome.l_cost))
    if outcome.kind != "showdown":
        raise ValueError(f"unknown outcome kind {outcome.kind!r}")
    return equity * float(outcome.pot) - float(outcome.l_cost)


@dataclass(frozen=True)
class Candidate:
    """One legal exploiter action with its modelled continuation."""

    action: int
    kind: str
    direct: Optional[Outcome] = None
    responses: Optional[np.ndarray] = None
    outcomes: Tuple[Outcome, ...] = ()


def candidate_values(candidate: Candidate, equity: np.ndarray) -> np.ndarray:
    if candidate.direct is not None:
        return outcome_values(candidate.direct, equity)
    if candidate.responses is None or candidate.responses.shape[1] != len(candidate.outcomes):
        raise ValueError("candidate responses do not match its outcomes")
    values = np.zeros(N_COMBOS, dtype=np.float64)
    for k, outcome in enumerate(candidate.outcomes):
        values = values + candidate.responses[:, k] * outcome_values(outcome, equity)
    return values


def candidate_ev(candidate: Candidate, range_weights: np.ndarray, equity: np.ndarray) -> float:
    return float(np.dot(range_weights, candidate_values(candidate, equity)))


_KIND_PREFERENCE = {"call": 0, "fold": 1, "raise": 2}


def choose_action(
    candidates: Sequence[Candidate], evs: Sequence[float], tolerance: float = 1e-9,
) -> Candidate:
    """The highest-EV candidate; ties go to check/call, then fold, then the smallest raise."""
    if not candidates:
        raise ValueError("no candidates")
    best = max(evs)
    tied = [c for c, ev in zip(candidates, evs) if ev >= best - tolerance]
    return min(tied, key=lambda c: (_KIND_PREFERENCE.get(c.kind, 9), c.action))


def mean_se(values: Sequence[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "se": None, "n": 0}
    se = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else None
    return {"mean": float(array.mean()), "se": se, "n": int(array.size)}
