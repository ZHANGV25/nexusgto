"""Per-seat range override hook for the decision audit."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from app.engine.agent import live_resolve_adapter as adapter
from app.engine.agent.game_state import new_postflop_state
from app.engine.agent.hero_range_override import (
    N_COMBOS,
    apply_range_overrides,
)
from app.engine.agent.modes import ModeConfig
from app.engine.agent.strategy_source import LiveResolveSource
from scripts.lbr_core import combo_index


def _ci(card: str) -> int:
    ranks = "23456789TJQKA"
    return ranks.index(card[0].upper()) * 4 + "cdhs".index(card[1].lower())


def _state():
    return new_postflop_state(
        seats=["BB", "UTG", "BTN"],
        preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td")],
        pot=720,
        stacks=[9660, 9780, 9780],
    )


def _weights(*combos: tuple[int, int, float]) -> np.ndarray:
    weights = np.zeros(N_COMBOS, dtype=np.float64)
    for c0, c1, weight in combos:
        weights[combo_index(c0, c1)] = float(weight)
    return weights


class _AnyRowDict(dict):
    def __init__(self, labels: Dict[str, float]) -> None:
        super().__init__()
        self._labels = labels

    def get(self, key: object, default: object = None) -> object:
        return self._labels


class _CaptureEngine:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def solve(self, **kwargs: object) -> Dict[int, object]:
        self.calls.append(dict(kwargs))
        hero = int(kwargs["hero_actual_hand"]["player_idx"])  # type: ignore[index]
        return {hero: _AnyRowDict({"Check": 0.7, "Bet 75%": 0.3})}


def test_apply_range_overrides_maps_by_positions_sorted_seat_name():
    inputs = adapter.build_solve_inputs(_state(), range_weight_dicts=None)
    original = [list(rows) for rows in inputs["player_ranges"]]
    weights = _weights((_ci("Qh"), _ci("Jh"), 3.0))

    ranges, stats = apply_range_overrides(
        inputs["player_ranges"], inputs["positions_sorted"],
        inputs["board_ints"], {"UTG": weights},
    )

    assert inputs["positions_sorted"] == ["BB", "UTG", "BTN"]
    assert stats["players_applied"] == 1
    assert stats["players"]["UTG"]["applied"] is True
    assert sum(row[2] for row in ranges[1]) == pytest.approx(3.0)
    assert ranges[0] == original[0]
    assert ranges[2] == original[2]


def test_apply_range_overrides_accepts_numeric_full_seat_id():
    inputs = adapter.build_solve_inputs(_state(), range_weight_dicts=None)
    weights = _weights((_ci("Qh"), _ci("Jh"), 2.0))

    ranges, stats = apply_range_overrides(
        inputs["player_ranges"], inputs["positions_sorted"],
        inputs["board_ints"], {5: weights},
    )

    assert stats["players"]["BTN"]["applied"] is True
    assert sum(row[2] for row in ranges[2]) == pytest.approx(2.0)


def test_apply_range_overrides_zero_live_mass_keeps_seeded_rows():
    inputs = adapter.build_solve_inputs(_state(), range_weight_dicts=None)
    original = [list(rows) for rows in inputs["player_ranges"]]

    ranges, stats = apply_range_overrides(
        inputs["player_ranges"], inputs["positions_sorted"],
        inputs["board_ints"], {"BB": np.zeros(N_COMBOS, dtype=np.float64)},
    )

    assert stats["applied"] is False
    assert stats["players_unapplied"] == 1
    assert stats["players"]["BB"]["applied"] is False
    assert ranges == original


def test_live_resolve_none_range_overrides_leaves_inputs_unchanged():
    state = _state()
    engine = _CaptureEngine()
    source = LiveResolveSource(
        ModeConfig(mode="LIVE_RESOLVE"),
        cpu_es_solver=engine,
        seed_ranges=False,
    )

    source.get_mix(state, (_ci("Qh"), _ci("Jh")))

    expected = adapter.build_solve_inputs(state, range_weight_dicts=None)
    assert source.last_range_overrides is None
    assert source.last_range_seeding == "uniform"
    assert engine.calls[0]["positions_sorted"] == expected["positions_sorted"]
    assert engine.calls[0]["player_ranges"] == expected["player_ranges"]


def test_live_resolve_range_overrides_win_after_hero_override():
    state = _state()
    engine = _CaptureEngine()
    source = LiveResolveSource(
        ModeConfig(mode="LIVE_RESOLVE"),
        cpu_es_solver=engine,
        seed_ranges=False,
    )
    source.hero_range_override = _weights((_ci("Qh"), _ci("Jh"), 2.0))
    source.range_overrides = {
        "BB": _weights((_ci("Qh"), _ci("Jh"), 5.0)),
        "UTG": _weights((_ci("Qc"), _ci("Jc"), 7.0)),
    }

    source.get_mix(state, (_ci("Qh"), _ci("Jh")))

    ranges = engine.calls[0]["player_ranges"]
    assert source.last_hero_range_override["applied"] is True
    assert source.last_range_overrides["players_applied"] == 2
    assert source.last_range_seeding.endswith("+hero_own_reach+range_override")
    assert sum(row[2] for row in ranges[0]) == pytest.approx(5.0)
    assert sum(row[2] for row in ranges[1]) == pytest.approx(7.0)
