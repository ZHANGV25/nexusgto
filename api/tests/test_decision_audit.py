"""Pure helpers of the decision audit (scripts/decision_audit_core.py)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.decision_audit_core import (
    _bb100,
    _combine,
    parse_shard,
    sample_joint_combos,
    shard_selected,
)
from scripts.lbr_core import COMBOS, N_COMBOS, combo_index


def _row(p: float) -> dict:
    return {"inclusion_probability": p}


def test_bb100_scales_cell_mean_by_frequency_per_hand():
    # 4 sampled spots out of n_c = 40 eligible decisions in 10 deals (60 hero hands):
    # contribution = mean(adv) * 40 / 60 chips per hand = bb/100 at 100 chips per bb.
    rows = [_row(0.1)] * 4
    out = _bb100([30.0, 10.0, 20.0, 20.0], rows, deals=10)
    assert out["mean"] == pytest.approx(20.0 * 40 / 60)
    expected_se = np.std([30.0, 10.0, 20.0, 20.0], ddof=1) / 2.0 * 40 / 60
    assert out["se"] == pytest.approx(expected_se)
    assert out["n"] == 4


def test_bb100_empty_cell_is_zero():
    assert _bb100([], [], deals=10) == {"mean": 0.0, "se": 0.0, "n": 0}


def test_combine_sums_means_and_adds_variances():
    out = _combine([{"mean": 1.0, "se": 3.0, "n": 5}, {"mean": -2.0, "se": 4.0, "n": 7}])
    assert out["mean"] == pytest.approx(-1.0)
    assert out["se"] == pytest.approx(5.0)
    assert out["n"] == 12


def test_sample_joint_combos_never_shares_cards():
    rng = np.random.default_rng(3)
    board = [0, 5, 10, 15]
    # Two players whose ranges overlap heavily: both hold only hands containing card 20 or 21.
    weights = np.zeros(N_COMBOS)
    for index, (c0, c1) in enumerate(COMBOS):
        if 20 in (c0, c1) or 21 in (c0, c1):
            weights[index] = 1.0
    for _ in range(200):
        draw = sample_joint_combos(rng, {0: weights, 1: weights}, [0, 1], board)
        cards = [c for combo in draw.values() for c in COMBOS[combo]]
        assert len(set(cards)) == 4
        assert not set(cards) & set(board)


def test_sample_joint_combos_rejects_empty_reach():
    with pytest.raises(ValueError):
        sample_joint_combos(np.random.default_rng(0), {0: np.zeros(N_COMBOS)}, [0], [0, 1, 2])


def test_sample_joint_combos_single_combo_range():
    target = combo_index(40, 41)
    weights = np.zeros(N_COMBOS)
    weights[target] = 1.0
    draw = sample_joint_combos(np.random.default_rng(1), {2: weights}, [2], [0, 1, 2])
    assert draw == {2: target}


def test_shards_partition_indices():
    shards = [parse_shard(f"{i}/3") for i in range(3)]
    for index in range(30):
        assert sum(shard_selected(index, shard) for shard in shards) == 1
    assert shard_selected(7, None)
    with pytest.raises(ValueError):
        parse_shard("3/3")
