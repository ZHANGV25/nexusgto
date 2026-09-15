"""Tests for the production solve-request recorder (no blueprint needed)."""
from __future__ import annotations

import pytest

from app.engine.agent.live_resolve_adapter import LiveResolveUnsupported
from scripts import solve_request_capture as cap


class _Engine:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self._last_solve_stats = {"iterations_run": 123, "solve_s": 2.5, "num_nodes": 7, "extra": 1}
        self.other = "passthrough"

    def solve(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {0: {5: {"check": 1.0}}}


def _kwargs(n_players=2, board_len=3, hero_hand=5):
    return {
        "board_ints": list(range(board_len)),
        "player_ranges": [[(10, 11, 1.0)] for _ in range(n_players)],
        "hero_replay": {"player_idx": 0, "street_actions": [1], "expected_actions": 3},
        "hero_actual_hand": {"player_idx": 0, "hand_idx": hero_hand, "traverser_fraction": 0.5},
        "iterations": 100_000,
        "deadline_ms": 3000,
    }


def test_describe_request_reads_street_players_and_hero():
    info = cap.describe_request(_kwargs(n_players=3, board_len=4))
    assert info == {
        "street": "turn", "n_players": 3, "hero_player_idx": 0, "hero_hand_idx": 5,
        "street_actions": (1,), "offtree": False,
    }


def test_recorder_records_solved_requests_with_stats(tmp_path):
    engine = _Engine()
    recorder = cap.SolveRequestRecorder(engine, str(tmp_path), limit=10, clock=iter([1.0, 3.0]).__next__)
    result = recorder.solve(**_kwargs())
    assert result == {0: {5: {"check": 1.0}}}
    assert recorder.captured == 1 and engine.calls
    records = cap.load_requests([str(tmp_path)])
    assert len(records) == 1
    record = records[0]
    assert record["outcome"] == "solved" and record["wall_s"] == pytest.approx(2.0)
    assert record["street"] == "flop" and record["n_players"] == 2
    assert record["solve_stats"]["iterations_run"] == 123 and "extra" not in record["solve_stats"]
    assert record["kwargs"]["iterations"] == 100_000
    assert recorder.other == "passthrough"


def test_capture_only_records_then_refuses_without_solving(tmp_path):
    engine = _Engine()
    recorder = cap.SolveRequestRecorder(engine, str(tmp_path), limit=10, capture_only=True)
    with pytest.raises(LiveResolveUnsupported):
        recorder.solve(**_kwargs())
    assert engine.calls == []
    assert cap.load_requests([str(tmp_path)])[0]["outcome"] == "capture_only"


def test_min_players_and_limit_skip_without_blocking_the_solve(tmp_path):
    engine = _Engine()
    recorder = cap.SolveRequestRecorder(engine, str(tmp_path), limit=1, min_players=3, capture_only=True)
    assert recorder.solve(**_kwargs(n_players=2)) == {0: {5: {"check": 1.0}}}
    assert recorder.skipped == {"below_min_players": 1}
    with pytest.raises(LiveResolveUnsupported):
        recorder.solve(**_kwargs(n_players=3))
    assert recorder.solve(**_kwargs(n_players=4)) == {0: {5: {"check": 1.0}}}
    assert recorder.skipped == {"below_min_players": 1, "limit_reached": 1}
    assert recorder.captured == 1


def test_errors_are_recorded_and_propagated(tmp_path):
    engine = _Engine(error=RuntimeError("boom"))
    recorder = cap.SolveRequestRecorder(engine, str(tmp_path), limit=10)
    with pytest.raises(RuntimeError):
        recorder.solve(**_kwargs())
    assert cap.load_requests([str(tmp_path)])[0]["outcome"] == "error:RuntimeError"


def test_positional_solve_arguments_are_rejected(tmp_path):
    recorder = cap.SolveRequestRecorder(_Engine(), str(tmp_path), limit=10)
    with pytest.raises(ValueError):
        recorder.solve(1, 2)
