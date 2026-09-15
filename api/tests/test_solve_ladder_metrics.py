"""Pure tests for the solve-ladder metrics (no blueprint needed)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import solve_ladder_metrics as m


def _tree():
    return {
        "num_players": 2,
        "board": (0, 1, 2, 3, 4),
        "starting_pot": 100,
        "player_hands": [
            [(10, 11, 1.0), (12, 13, 0.0), (14, 15, 2.0)],
            [(20, 21, 1.0), (22, 23, 1.0)],
        ],
        "nodes": [
            {"type": 0, "player": 0, "children": (1, 2), "pot": 100, "bets": (0, 0), "active": (True, True)},
            {"type": 0, "player": 1, "children": (3, 4), "pot": 100, "bets": (0, 0), "active": (True, True)},
            {"type": 2, "player": 0, "children": (), "pot": 100, "bets": (0, 0), "active": (True, True)},
            {"type": 1, "player": 1, "children": (), "pot": 150, "bets": (50, 0), "active": (True, False)},
            {"type": 2, "player": 0, "children": (), "pot": 200, "bets": (50, 50), "active": (True, True)},
        ],
    }


def test_sigma_distance_excludes_zero_weight_rows_and_averages_nodes():
    tree = _tree()
    a = {0: np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]), 1: np.array([[0.2, 0.8], [0.6, 0.4]])}
    b = {0: np.array([[0.6, 0.4], [1.0, 0.0], [0.5, 0.5]]), 1: np.array([[0.2, 0.8], [0.6, 0.4]])}
    out = m.sigma_distance(a, b, tree)
    # node 0: rows 0 and 2 count (row 1 has zero weight): TV 0.4 and 0.0 -> 0.2
    assert out["root_tv"] == pytest.approx(0.2)
    assert out["mean_tv"] == pytest.approx(0.1)  # node 1 identical -> 0.0
    assert out["nodes"] == 2


def test_sigma_distance_rejects_shape_mismatch():
    tree = _tree()
    with pytest.raises(ValueError):
        m.sigma_distance({0: np.zeros((3, 2))}, {0: np.zeros((3, 3))}, tree)


def test_mix_distance_over_label_union():
    assert m.mix_distance({"fold": 0.5, "call": 0.5}, {"call": 0.5, "raise": 0.5}) == pytest.approx(0.5)
    assert m.mix_distance({"call": 1.0}, {"call": 1.0}) == 0.0


def test_cost_helpers():
    assert m.ms_per_kiter(3.0, 60_000) == pytest.approx(50.0)
    assert m.ms_per_kiter(3.0, 0) is None
    assert m.ms_per_kiter(None, 10) is None
    assert m.iterations_in(3000.0, 50.0) == pytest.approx(60_000)
    assert m.iterations_in(3000.0, None) is None


def test_plateau_rung_requires_every_larger_rung_to_stay_inside():
    floor = 0.01
    assert m.plateau_rung([(10, 0.10), (25, 0.02), (50, 0.012), (100, 0.011)], floor) == 50
    # a lucky dip at 25k that rises again at 50k does not count
    assert m.plateau_rung([(10, 0.10), (25, 0.012), (50, 0.03), (100, 0.011)], floor) == 100
    assert m.plateau_rung([(10, 0.10), (25, 0.05)], floor) is None
    assert m.plateau_rung([], floor) is None


def _row(spot, rung, iterations, run, solve_s, pair_tv, tv, mix_tv=None, gap=None, gap_avg=None,
         street="flop", n=2):
    return {
        "spot": spot, "rung": rung, "street": street, "n_players": n,
        "iterations": iterations, "iterations_run": run, "iterations_run_b": run,
        "solve_s": solve_s, "solve_s_b": solve_s, "construct_s": 0.1,
        "pair_mean_tv": pair_tv, "pair_root_tv": pair_tv, "pair_hero_mix_tv": pair_tv,
        "mean_tv": tv, "root_tv": tv, "hero_mix_tv": mix_tv,
        "river_brgap_pct_pot": gap, "river_brgap_avg_pct_pot": gap_avg,
    }


def test_aggregate_cell_uses_the_top_rung_pair_as_floor():
    rows = []
    for spot in ("s1", "s2"):
        rows.append(_row(spot, "it_100000", 100_000, 100_000, 5.0, 0.01, None))
        rows.append(_row(spot, "production", 100_000, 60_000, 3.0, 0.02, 0.05, 0.05))
        rows.append(_row(spot, "it_1000", 1_000, 1_000, 0.05, 0.30, 0.10, 0.1))
        rows.append(_row(spot, "it_5000", 5_000, 5_000, 0.25, 0.05, 0.014, 0.02))
        rows.append(_row(spot, "it_25000", 25_000, 25_000, 1.25, 0.015, 0.011, 0.012))
        rows.append(_row(spot, "avg_5000", 5_000, 5_000, 0.3, None, 0.02, None, 12.0, 6.0, street="flop"))
    cell = m.aggregate_cell(rows)
    assert cell["spots"] == 2
    assert cell["top_rung"] == "it_100000"
    assert cell["noise_floor"]["mean_tv"] == pytest.approx(0.01)
    assert cell["ms_per_kiter"] == pytest.approx(50.0)
    assert cell["iterations_in_3s"] == pytest.approx(60_000)
    assert cell["plateau_iterations"] == 5_000
    assert cell["seconds_to_plateau"] == pytest.approx(0.25)
    assert cell["rungs"]["production"]["iterations_run"] == pytest.approx(60_000)
    assert cell["rungs"]["it_5000"]["mean_tv_se"] == pytest.approx(0.0)
    assert cell["rungs"]["avg_5000"]["river_brgap_avg_pct_pot"] == pytest.approx(6.0)
    assert cell["rungs"]["it_100000"]["mean_tv"] is None


def test_cost_fit_separates_fixed_and_marginal_cost():
    rows = [
        {"iterations_run": 1_000, "solve_s": 1.1, "iterations_run_b": 1_000, "solve_s_b": 1.1},
        {"iterations_run": 10_000, "solve_s": 2.0, "iterations_run_b": 10_000, "solve_s_b": 2.0},
        {"iterations_run": 100_000, "solve_s": 11.0, "iterations_run_b": 100_000, "solve_s_b": 11.0},
    ]
    fit = m.cost_fit(rows)
    assert fit["fixed_s"] == pytest.approx(1.0)
    assert fit["marginal_ms_per_kiter"] == pytest.approx(100.0)
    assert fit["iterations_in_3s"] == pytest.approx(20_000)
    assert fit["points"] == 6
    assert m.cost_fit(rows[:1])["fixed_s"] is None


def test_aggregate_groups_by_street_and_players():
    rows = [
        _row("a", "it_10000", 10_000, 10_000, 1.0, 0.1, None, street="flop", n=2),
        _row("b", "it_10000", 10_000, 10_000, 4.0, 0.1, None, street="turn", n=3),
    ]
    cells = m.aggregate(rows)
    assert set(cells) == {"flop_2p", "turn_3p"}
    assert cells["turn_3p"]["ms_per_kiter"] == pytest.approx(400.0)
    assert cells["turn_3p"]["plateau_iterations"] is None
