"""Pure tests for the river range audit helpers (no blueprint needed)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import river_range_audit as audit


def _tree():
    # Board 0..4; hero rows use cards 10-15, opponent rows cards 20-23.
    return {
        "num_players": 2,
        "board": (0, 1, 2, 3, 4),
        "starting_pot": 100,
        "player_hands": [
            [(10, 11, 1.0), (12, 13, 1.0), (14, 15, 2.0)],
            [(20, 21, 1.0), (22, 23, 3.0)],
        ],
        "nodes": [
            {"type": 0, "player": 0, "children": (1, 2), "pot": 100, "bets": (0, 0), "active": (True, True)},
            {"type": 2, "player": 0, "children": (), "pot": 100, "bets": (0, 0), "active": (True, True)},
            {"type": 0, "player": 1, "children": (3, 4), "pot": 150, "bets": (50, 0), "active": (True, True)},
            {"type": 1, "player": 0, "children": (), "pot": 150, "bets": (50, 0), "active": (True, False)},  # fold: player = winner
            {"type": 2, "player": 0, "children": (), "pot": 200, "bets": (50, 50), "active": (True, True)},
        ],
    }


def test_range_stats_reports_tv_mass_and_strength_shift():
    assumed = np.array([1.0, 1.0, 2.0])
    true = np.array([0.0, 0.5, 2.0])
    strengths = np.array([1, 2, 3])
    out = audit.range_stats(assumed, true, strengths)
    assert out["mass_kept"] == pytest.approx(2.5 / 4.0)
    assert out["tv"] == pytest.approx(0.5 * (0.25 + abs(0.25 - 0.2) + abs(0.5 - 0.8)))
    assert out["mean_percentile_true"] > out["mean_percentile_assumed"]
    assert out["effective_rows_true"] < out["effective_rows_assumed"]


def test_hero_equity_weights_opponent_rows_and_ties():
    tree = _tree()
    strengths = [np.array([5, 1, 3]), np.array([3, 2])]
    opp = np.array([1.0, 1.0])
    equity = audit.hero_equity(tree, 0, opp, strengths)
    assert equity[0] == pytest.approx(1.0)      # beats both
    assert equity[1] == pytest.approx(0.0)      # loses to both
    assert equity[2] == pytest.approx(0.75)     # ties one, beats one


def test_with_weights_and_joint_profile():
    tree = _tree()
    tree_true = audit.with_weights(tree, [np.array([0.0, 1.0, 1.0]), np.array([2.0, 2.0])])
    assert [h[2] for h in tree_true["player_hands"][0]] == [0.0, 1.0, 1.0]
    assert tree["player_hands"][0][0][2] == 1.0  # original untouched
    hero_sigma = {0: np.array([[1.0, 0.0]] * 3)}
    opp_sigma = {0: np.array([[0.0, 1.0]] * 3), 2: np.array([[0.5, 0.5]] * 2)}
    joint = audit.joint_profile(tree, 0, hero_sigma, opp_sigma)
    assert set(joint) == {0, 2}
    assert joint[0][0, 0] == 1.0 and joint[2][0, 1] == 0.5


def test_score_strategy_prefers_betting_strong_hands_into_a_folding_opponent():
    tree = _tree()
    strengths = [np.array([5, 1, 3]), np.array([2, 4])]
    hero_w = np.array([1.0, 1.0, 1.0])
    strong = np.array([1.0, 0.0, 0.0])
    blueprint = {2: np.array([[1.0, 0.0], [1.0, 0.0]])}  # opponent always folds to a bet
    always_bet = {0: np.array([[0.0, 1.0]] * 3)}
    always_check = {0: np.array([[1.0, 0.0]] * 3)}
    bet = audit.score_strategy(tree, 0, always_bet, blueprint, strengths, strong, hero_w)
    check = audit.score_strategy(tree, 0, always_check, blueprint, strengths, strong, hero_w)
    assert bet["strong_bet"] == pytest.approx(1.0) and check["strong_bet"] == pytest.approx(0.0)
    assert bet["ev_pct_pot"] > check["ev_pct_pot"]
    assert bet["br_pct_pot"] == pytest.approx(bet["ev_pct_pot"])


def test_summarize_pairs_resolves():
    rows = [
        {"opp_range": {"tv": 0.4}, "hero_range": {"tv": 0.2},
         "strategies": {"served": {"ev_pct_pot": 50.0, "br_pct_pot": 70.0, "strong_bet": 0.4},
                        "blueprint": {"ev_pct_pot": 55.0, "br_pct_pot": 70.0, "strong_bet": 0.5},
                        "resolve_assumed": {"ev_pct_pot": 50.0, "br_pct_pot": 70.0, "strong_bet": 0.4},
                        "resolve_true": {"ev_pct_pot": 56.0, "br_pct_pot": 70.0, "strong_bet": 0.8}}},
    ]
    out = audit.summarize(rows)
    assert out["spots"] == 1
    assert out["resolve_true_minus_assumed_ev"]["mean"] == pytest.approx(6.0)
    assert out["blueprint_minus_resolve_true_ev"]["mean"] == pytest.approx(-1.0)
