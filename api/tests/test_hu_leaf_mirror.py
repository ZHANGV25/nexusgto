"""Pure GameState-to-solve-input contracts for HU leaf-key mirroring."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from app.engine.agent.game_state import GameState, new_postflop_state
from app.engine.agent.live_resolve_adapter import build_solve_inputs


def _state(*, true_hu: bool) -> GameState:
    return new_postflop_state(
        seats=["SB", "BB"],
        preflop_prefix=[0, 0, 0, 0, 2, 1],
        board=[0, 5, 10],
        pot=300,
        stacks=[9700, 9700],
        true_hu=true_hu,
    )


def test_unset_and_zero_preserve_current_true_hu_inputs(monkeypatch) -> None:
    state = _state(true_hu=True)
    monkeypatch.delenv("NEXUS_HU_LEAF_MIRROR", raising=False)
    unset_inputs = build_solve_inputs(state)
    monkeypatch.setenv("NEXUS_HU_LEAF_MIRROR", "0")
    zero_inputs = build_solve_inputs(state)

    assert unset_inputs["player_seats"] == [1, 0]
    assert json.dumps(unset_inputs, sort_keys=True).encode() == json.dumps(
        zero_inputs, sort_keys=True,
    ).encode()


def test_enabled_mirror_swaps_only_player_seats(monkeypatch) -> None:
    state = _state(true_hu=True)
    monkeypatch.setenv("NEXUS_HU_LEAF_MIRROR", "0")
    off_inputs = build_solve_inputs(state)
    monkeypatch.setenv("NEXUS_HU_LEAF_MIRROR", "1")
    on_inputs = build_solve_inputs(state)

    assert off_inputs["player_seats"] == [1, 0]
    assert on_inputs["player_seats"] == [0, 1]
    assert {
        key: value for key, value in off_inputs.items()
        if key != "player_seats"
    } == {
        key: value for key, value in on_inputs.items()
        if key != "player_seats"
    }


def test_enabled_mirror_never_changes_six_max_model_inputs(monkeypatch) -> None:
    state = _state(true_hu=False)
    monkeypatch.setenv("NEXUS_HU_LEAF_MIRROR", "0")
    off_inputs = build_solve_inputs(state)
    monkeypatch.setenv("NEXUS_HU_LEAF_MIRROR", "1")
    on_inputs = build_solve_inputs(state)

    assert off_inputs["player_seats"] == [0, 1]
    assert json.dumps(off_inputs, sort_keys=True).encode() == json.dumps(
        on_inputs, sort_keys=True,
    ).encode()


@pytest.mark.parametrize("value", ["yes", "2", " 1 "])
def test_invalid_values_fail_loud(monkeypatch, value: str) -> None:
    monkeypatch.setenv("NEXUS_HU_LEAF_MIRROR", value)
    with pytest.raises(ValueError, match="NEXUS_HU_LEAF_MIRROR"):
        build_solve_inputs(_state(true_hu=True))


def test_enabled_mirror_rejects_inconsistent_true_hu_seats(monkeypatch) -> None:
    state = replace(_state(true_hu=True), seat_of_slot=(1, 2))
    monkeypatch.setenv("NEXUS_HU_LEAF_MIRROR", "1")

    with pytest.raises(ValueError, match="requires only full seats"):
        build_solve_inputs(state)
