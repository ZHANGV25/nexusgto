"""Range prior hooks: combo multiplicity (NEXUS_RANGE_COMBO_PRIOR) and the own-reach hero range."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from app.engine.agent import live_resolve_adapter as adapter
from app.engine.agent.hero_range_override import N_COMBOS, apply_hero_range


def card(rank: str, suit: str) -> int:
    return "23456789TJQKA".index(rank) * 4 + "cdhs".index(suit)


BOARD = [card("2", "s"), card("3", "s"), card("4", "s"), card("K", "d"), card("9", "h")]


def _river_state():
    from app.engine.agent.game_state import new_postflop_state

    return new_postflop_state(
        seats=["SB", "BB"], preflop_prefix=[0, 0, 0, 0, 1, 0], board=BOARD,
        pot=1000, stacks=[9500, 9500], street=3,
    )


def _class_totals(inputs, player: int):
    totals = defaultdict(float)
    live = {}
    for (_c0, _c1, weight), (name, _index, n_live) in zip(
        inputs["player_ranges"][player], inputs["player_range_metadata"][player],
    ):
        totals[name] += weight
        live[name] = n_live
    return totals, live


def test_default_grid_conserves_class_weight_not_combo_count(monkeypatch):
    monkeypatch.delenv(adapter.COMBO_PRIOR_ENV, raising=False)
    inputs = adapter.build_solve_inputs(_river_state(), range_weight_dicts=None)
    totals, live = _class_totals(inputs, 0)
    assert inputs["combo_prior"] is False
    assert totals["AA"] == pytest.approx(1.0) and totals["72o"] == pytest.approx(1.0)
    assert live["AA"] == 6 and live["72o"] == 9  # the 2s on the board blocks three 72o combos


def test_combo_prior_scales_each_class_by_its_live_combo_count(monkeypatch):
    monkeypatch.setenv(adapter.COMBO_PRIOR_ENV, "1")
    inputs = adapter.build_solve_inputs(_river_state(), range_weight_dicts=None)
    totals, live = _class_totals(inputs, 0)
    assert inputs["combo_prior"] is True
    for name, total in totals.items():
        assert total == pytest.approx(float(live[name]))
    assert totals["AA"] == pytest.approx(6.0) and totals["72o"] == pytest.approx(9.0)
    assert totals["AKs"] == pytest.approx(3.0)  # AdKd is blocked by the Kd on the board
    total_mass = sum(totals.values())
    assert total_mass == pytest.approx(sum(live.values()))


def test_run_match_accepts_the_combo_prior_override():
    from scripts import run_match_cli as cli

    assert cli._env_override("NEXUS_RANGE_COMBO_PRIOR=1") == ("NEXUS_RANGE_COMBO_PRIOR", "1")


def test_hero_range_override_sums_combo_weights_into_canonical_rows():
    inputs = adapter.build_solve_inputs(_river_state(), range_weight_dicts=None)
    hero = inputs["hero_player_idx"]
    weights = np.zeros(N_COMBOS)
    from scripts.lbr_core import combo_index

    ace_high = (card("A", "c"), card("Q", "d"))
    weights[combo_index(*ace_high)] = 2.0
    weights[combo_index(card("A", "h"), card("Q", "c"))] = 3.0  # same canonical row on this board
    weights[combo_index(card("2", "s"), card("A", "c"))] = 9.0  # blocked by the board: ignored
    ranges, stats = apply_hero_range(inputs["player_ranges"], hero, inputs["board_ints"], weights)
    assert stats["applied"] is True and stats["mass"] == pytest.approx(5.0)
    assert stats["unmapped_mass"] == 0.0 and stats["rows_positive"] == 1
    hero_weights = [w for _c0, _c1, w in ranges[hero]]
    assert sum(hero_weights) == pytest.approx(5.0)
    other = 1 - hero
    assert ranges[other] == list(inputs["player_ranges"][other])


def test_hero_range_override_with_zero_mass_is_not_applied():
    inputs = adapter.build_solve_inputs(_river_state(), range_weight_dicts=None)
    ranges, stats = apply_hero_range(inputs["player_ranges"], 0, inputs["board_ints"], np.zeros(N_COMBOS))
    assert stats["applied"] is False and ranges[0] == list(inputs["player_ranges"][0])
