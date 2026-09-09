"""Board-relative suit isomorphism for the live re-solve hand grid (dose ISO).

WHY (2026-09-09). ``NEXUS_GRID_K=ALL`` holds every physical combo (1128 rows
per player on a turn) so the re-solve can see suits; the cost is a 2p turn
solve of 8-10 s on the serving box against a 6 s decision window, and both
the leaf fill and the CFR scale with rows. Two combos that differ only by a
relabelling of suits that can never make a flush are strategically
identical, so one row can stand for all of them with the summed weight.

THE RULE. A suit is *flush-relevant* on a board when a flush in that suit is
still reachable by the river: a 5-card board needs at least 3 cards of the
suit, a 4-card board at least 2, a 3-card board at least 1. Flush-relevant
suits are never relabelled. Every other suit is *pooled*; pooled suits are
interchangeable. Two live combos are ISO-equivalent when a permutation of the
pooled suits maps one onto the other with a live image (no board card). The
canonical form of a combo is the lexicographically smallest live image over
all pooled-suit permutations.

EXACTNESS. Showdown strength never depends on a pooled suit (it cannot
complete a flush by the river), and card removal is symmetric under the
class-weight seeding (a class's weight is split evenly over its live combos),
so a row's value is the value of every combo it stands for. The one stated
approximation: a pooled suit with one board card on the turn differs from
another pooled suit only in WHICH specific card is dead, which the symmetric
seeding makes immaterial.

A strict board-preserving permutation rule was considered first and rejected:
on a typical turn (9d 2d 7s Jh) every suit has a distinct board signature and
nothing collapses.

Card ints are ``rank * 4 + suit`` with the suit order c, d, h, s
(``app.services.postflop_provider._SUIT_ORDER``).
"""
from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple

Combo = Tuple[int, int]

# Board cards of a suit needed for a flush to remain reachable by the river,
# by board length. Shorter boards (never used for a postflop grid) fall back
# to "any board card of the suit keeps it relevant".
_FLUSH_RELEVANT_MIN_BY_BOARD_LEN: Dict[int, int] = {5: 3, 4: 2, 3: 1}


def flush_relevant_suits(board: Sequence[int]) -> FrozenSet[int]:
    """Suits in which a flush is still reachable by the river."""
    counts = [0, 0, 0, 0]
    for card in board:
        counts[int(card) & 3] += 1
    need = _FLUSH_RELEVANT_MIN_BY_BOARD_LEN.get(len(board), 1)
    return frozenset(suit for suit in range(4) if counts[suit] >= need)


def pooled_suits(board: Sequence[int]) -> Tuple[int, ...]:
    """The interchangeable (flush-irrelevant) suits, ascending."""
    relevant = flush_relevant_suits(board)
    return tuple(suit for suit in range(4) if suit not in relevant)


def _pooled_permutations(
    pooled: Tuple[int, ...],
) -> Iterable[Dict[int, int]]:
    for image in itertools.permutations(pooled):
        yield dict(zip(pooled, image))


def _apply(combo: Combo, suit_map: Dict[int, int]) -> Combo:
    imaged = tuple(
        ((card >> 2) << 2) | suit_map.get(card & 3, card & 3)
        for card in combo
    )
    a, b = sorted(imaged)
    return (int(a), int(b))


def canonical_combo(c0: int, c1: int, board: Sequence[int]) -> Combo:
    """Canonical ``(c0, c1)`` with ``c0 < c1`` under the ISO rule.

    Lexicographically smallest live image of the combo over every permutation
    of the pooled suits; flush-relevant suits are never touched. Idempotent,
    and equal for exactly the combos that are ISO-equivalent.
    """
    blocked = frozenset(int(card) for card in board)
    combo: Combo = (int(c0), int(c1)) if c0 < c1 else (int(c1), int(c0))
    if combo[0] in blocked or combo[1] in blocked:
        raise ValueError(f"combo {combo} collides with the board")
    pooled = pooled_suits(board)
    best = combo
    for suit_map in _pooled_permutations(pooled):
        candidate = _apply(combo, suit_map)
        if candidate[0] in blocked or candidate[1] in blocked:
            continue
        if candidate < best:
            best = candidate
    return best


def group_by_canonical(
    combos: Sequence[Combo], board: Sequence[int],
) -> List[Tuple[Combo, List[Combo]]]:
    """Group live combos by canonical form, in ascending canonical order.

    Returns ``[(canonical, [members...]), ...]``; members keep their input
    order. The canonical combo is always one of its own members.
    """
    groups: Dict[Combo, List[Combo]] = {}
    for combo in combos:
        key = canonical_combo(combo[0], combo[1], board)
        groups.setdefault(key, []).append((int(combo[0]), int(combo[1])))
    return [(key, groups[key]) for key in sorted(groups)]
