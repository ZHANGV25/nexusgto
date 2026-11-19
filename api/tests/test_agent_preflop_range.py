"""Preflop 169-class mapping and range-correct hole-card sampling.

The strength measurement must feed each seat IN-DISTRIBUTION hole cards (the
blueprint's BB-caller and UTG-raiser ranges), not uniform random: uniform
hands fed to range-specific strategies make an EV-vs-fixed number
uninterpretable. This pins the 169-class mapping (bit-exact to the trainer's
preflop classes in solver/src/blueprint/mccfr_blueprint.c) and the range
weighting.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.engine.agent.preflop_range import (
    NUM_PREFLOP_CLASSES,
    class_combos,
    hand_class_169,
    sample_range_hand,
)


def _ci(card: str) -> int:
    ranks = "AKQJT98765432"
    so = ("c", "d", "h", "s")
    return (12 - ranks.index(card[0].upper())) * 4 + so.index(card[1].lower())


# ── 169-class mapping (bit-exact to the trainer) ─────────────────────────


def test_there_are_169_classes():
    assert NUM_PREFLOP_CLASSES == 169


def test_class_enumeration_matches_trainer_order():
    # Trainer: r0 12..0, r1 r0..0; pair=one class, else suited THEN offsuit.
    # AA is class 0; AKs is class 1; AKo is class 2.
    assert hand_class_169(_ci("Ah"), _ci("Ad")) == 0          # AA
    assert hand_class_169(_ci("Ah"), _ci("Kh")) == 1          # AKs
    assert hand_class_169(_ci("Ah"), _ci("Kd")) == 2          # AKo


def test_class_is_order_independent():
    assert hand_class_169(_ci("Kh"), _ci("Ah")) == hand_class_169(_ci("Ah"), _ci("Kh"))


def test_72o_is_a_valid_offsuit_class():
    c = hand_class_169(_ci("7h"), _ci("2d"))
    assert 0 <= c < 169
    # 72 suited and offsuit are distinct classes.
    assert hand_class_169(_ci("7h"), _ci("2h")) != c


def test_all_1326_combos_map_into_169_classes():
    seen = set()
    for a in range(52):
        for b in range(a + 1, 52):
            seen.add(hand_class_169(a, b))
    assert seen == set(range(169))


def test_class_combos_counts():
    # A pair has 6 combos; suited 4; offsuit 12.
    assert class_combos(hand_class_169(_ci("Ah"), _ci("Ad"))) == 6   # AA
    assert class_combos(hand_class_169(_ci("Ah"), _ci("Kh"))) == 4   # AKs
    assert class_combos(hand_class_169(_ci("Ah"), _ci("Kd"))) == 12  # AKo
    # Total combos across all classes = 1326.
    assert sum(class_combos(c) for c in range(169)) == 1326


# ── Range sampling ───────────────────────────────────────────────────────


def test_sample_range_hand_respects_weights_and_board():
    # A degenerate range: all weight on AA. Sampling must return an AA combo,
    # and never use a board card.
    weights = np.zeros(169)
    weights[hand_class_169(_ci("Ah"), _ci("Ad"))] = 1.0
    board = {_ci("Ah")}  # block one ace
    rng = np.random.default_rng(0)
    for _ in range(50):
        h = sample_range_hand(weights, board, rng)
        assert h[0] != h[1]
        assert h[0] not in board and h[1] not in board
        # both cards are aces
        assert h[0] // 4 == 12 and h[1] // 4 == 12


def test_sample_range_hand_zero_weight_class_never_drawn():
    # 72o has zero weight -> never sampled.
    weights = np.zeros(169)
    aa = hand_class_169(_ci("Ah"), _ci("Ad"))
    weights[aa] = 1.0
    o72 = hand_class_169(_ci("7h"), _ci("2d"))
    rng = np.random.default_rng(1)
    draws = [hand_class_169(*sample_range_hand(weights, set(), rng)) for _ in range(100)]
    assert o72 not in draws
    assert all(d == aa for d in draws)


def test_sample_range_hand_raises_if_all_blocked():
    # All weight on AA but both relevant aces blocked enough that no combo fits.
    weights = np.zeros(169)
    weights[hand_class_169(_ci("Ah"), _ci("Ad"))] = 1.0
    board = {_ci("Ah"), _ci("Ad"), _ci("Ac"), _ci("As")}  # all aces gone
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        sample_range_hand(weights, board, rng)
