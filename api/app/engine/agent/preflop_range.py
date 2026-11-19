"""Preflop 169-class mapping + range-correct hole-card sampling.

The strength measurement feeds each seat IN-DISTRIBUTION hole cards sampled from
the blueprint's preflop σ̄ (the BB-caller / UTG-raiser ranges), not uniform —
feeding uniform hands to range-specific strategies is the confound that makes
EV-vs-fixed-opponent uninterpretable as quality.

The 169 lossless preflop classes (13 pairs + 78 suited + 78 offsuit) and their
enumeration are a bit-exact port of the trainer (the preflop class map in
bp_init_unified, solver/src/blueprint/mccfr_blueprint.c):
iterate rank_hi 12..0, rank_lo hi..0; a pair is one class, a non-pair emits
SUITED then OFFSUIT. Card encoding matches the agent's (rank = card//4, suit =
card%4, rank 12 = Ace).
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Set, Tuple

import numpy as np

NUM_PREFLOP_CLASSES = 169


def _build_class_map() -> Dict[Tuple[int, int, int], int]:
    """(rank_hi, rank_lo, suited) -> class index, in the trainer's order."""
    m: Dict[Tuple[int, int, int], int] = {}
    n = 0
    for r0 in range(12, -1, -1):
        for r1 in range(r0, -1, -1):
            if r0 == r1:
                m[(r0, r1, 0)] = n
                m[(r0, r1, 1)] = n
                n += 1
            else:
                m[(r0, r1, 1)] = n  # suited first
                n += 1
                m[(r0, r1, 0)] = n  # offsuit
                n += 1
    return m


_CLASS_MAP = _build_class_map()


def hand_class_169(c0: int, c1: int) -> int:
    """Map a 2-card hand (card ints) to its 169 preflop class."""
    r0, r1 = c0 // 4, c1 // 4
    suited = 1 if (c0 % 4) == (c1 % 4) else 0
    if r0 < r1:
        r0, r1 = r1, r0
    return _CLASS_MAP[(r0, r1, suited)]


def _build_class_combos() -> Tuple[int, ...]:
    counts = [0] * NUM_PREFLOP_CLASSES
    for a in range(52):
        for b in range(a + 1, 52):
            counts[hand_class_169(a, b)] += 1
    return tuple(counts)


_CLASS_COMBOS = _build_class_combos()


def class_combos(class_idx: int) -> int:
    """Number of concrete 2-card combos in a class (pair 6, suited 4, offsuit 12)."""
    return _CLASS_COMBOS[class_idx]


# Precompute the concrete combos per class for fast, board-aware sampling.
def _build_class_to_combos() -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    buckets: list = [[] for _ in range(NUM_PREFLOP_CLASSES)]
    for a in range(52):
        for b in range(a + 1, 52):
            buckets[hand_class_169(a, b)].append((a, b))
    return tuple(tuple(x) for x in buckets)


_CLASS_TO_COMBOS = _build_class_to_combos()


def sample_range_hand(
    class_weights: np.ndarray,
    board: Set[int],
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Sample one concrete 2-card hand from a range over the 169 classes.

    class_weights[i] = relative weight of class i (need not be normalized; e.g.
    P(action | class) × combos, or any nonnegative range weighting). Combos that
    use a board card are excluded; a class with no available combos is skipped.
    The final class is drawn ∝ (weight × #available combos) so blocking is
    handled correctly.

    Raises:
        ValueError: if no combo in any positively-weighted class is available
            (e.g. the whole range is blocked by the board).
    """
    if class_weights.shape[0] != NUM_PREFLOP_CLASSES:
        raise ValueError(
            f"class_weights must have {NUM_PREFLOP_CLASSES} entries, got "
            f"{class_weights.shape[0]}"
        )
    # Available combos per class (excluding board blockers).
    avail: list = []
    eff_weights = np.zeros(NUM_PREFLOP_CLASSES, dtype=np.float64)
    for c in range(NUM_PREFLOP_CLASSES):
        w = float(class_weights[c])
        if w <= 0.0:
            avail.append(())
            continue
        combos = tuple(
            (a, b) for (a, b) in _CLASS_TO_COMBOS[c]
            if a not in board and b not in board
        )
        avail.append(combos)
        eff_weights[c] = w * len(combos)

    total = eff_weights.sum()
    if total <= 0.0:
        raise ValueError("no available combo in the weighted range (all blocked)")
    eff_weights /= total
    cls = int(rng.choice(NUM_PREFLOP_CLASSES, p=eff_weights))
    combos = avail[cls]
    pick = int(rng.integers(0, len(combos)))
    return combos[pick]


# Type of reader.get_strategy(board, action_history, player, bucket, street).
StrategyReader = Callable[[list, Sequence[int], int, int, int], object]


def build_seat_range_weights(
    get_strategy: StrategyReader,
    action_history: Sequence[int],
    player: int,
    action_indices: Sequence[int],
) -> np.ndarray:
    """Per-class range weights = P(the seat's taken action | class), read from
    the blueprint's preflop σ̄ at this decision node.

    get_strategy: reader.get_strategy (board=[], street=0 preflop).
    action_history: the preflop prefix UP TO this seat's decision.
    player: the seat (0-5) making the decision.
    action_indices: the action(s) that define "this seat acted this way" — e.g.
        [2, 3] for "any raise" (the 0.7/1.0 raise slots) for the opener, or [1]
        for "call" for a flat-caller. Weights are P(sum of those indices).

    A class the blueprint never reaches / stores zero for gets weight 0 (it is
    not in the range). The result feeds sample_range_hand.
    """
    weights = np.zeros(NUM_PREFLOP_CLASSES, dtype=np.float64)
    for c in range(NUM_PREFLOP_CLASSES):
        strat = get_strategy([], list(action_history), player, c, 0)
        if strat is None:
            continue
        arr = np.asarray(strat, dtype=np.float64)
        s = arr.sum()
        if s <= 0.0:
            continue
        arr = arr / s
        weights[c] = float(sum(arr[i] for i in action_indices if i < len(arr)))
    return weights
