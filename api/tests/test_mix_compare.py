"""Pure tests for served-vs-blueprint mix comparison (no blueprint)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mix_compare as mc
from scripts import run_match
from scripts import run_match_cli as cli


def _state(street=1, n_live=2, facing=True, n_legal=3, pot=500):
    return SimpleNamespace(
        street=street, n_live=n_live, facing_bet=facing, pot=pot,
        num_legal_actions=lambda: n_legal, board=(1, 2, 3), postflop_history=(0, 2),
    )


def _mix(freqs):
    return SimpleNamespace(freqs=freqs)


def test_group_masses_follow_action_order():
    assert mc.group_masses(np.array([0.2, 0.5, 0.3]), True) == pytest.approx((0.2, 0.5, 0.3))
    assert mc.group_masses(np.array([0.6, 0.3, 0.1]), False) == pytest.approx((0.0, 0.6, 0.4))
    assert mc.group_masses(np.array([1.0]), False) == pytest.approx((0.0, 1.0, 0.0))


def test_mix_vector_normalizes_and_fills_missing_actions():
    np.testing.assert_allclose(mc.mix_vector({0: 1.0, 2: 3.0}, 3), [0.25, 0.0, 0.75])


def test_comparison_aggregates_by_cell():
    comparison = mc.MixComparison()
    comparison.add(_state(), (40, 41), _mix({0: 0.6, 1: 0.4}), _mix({0: 0.2, 1: 0.5, 2: 0.3}))
    comparison.add(_state(pot=700), (4, 5), _mix({1: 1.0}), _mix({1: 1.0}))
    comparison.add(_state(facing=False, n_legal=2), (8, 9), _mix({1: 1.0}), _mix({0: 1.0}))
    summary = comparison.summary()
    facing = summary["flop|2p|facing"]
    assert facing["count"] == 2
    assert facing["mean_served_fold"] == pytest.approx(0.3)
    assert facing["mean_blueprint_fold"] == pytest.approx(0.1)
    assert facing["mean_blueprint_raise"] == pytest.approx(0.15)
    assert facing["mean_l1"] == pytest.approx((0.4 + 0.1 + 0.3) / 2)
    assert facing["mean_top_differs"] == pytest.approx(0.5)
    assert facing["mean_pot"] == pytest.approx(600)
    unopened = summary["flop|2p|unopened"]
    assert unopened["mean_served_raise"] == pytest.approx(1.0)
    assert unopened["mean_blueprint_call"] == pytest.approx(1.0)


def test_rows_are_capped_and_written(tmp_path):
    comparison = mc.MixComparison(max_rows=1)
    for _ in range(3):
        comparison.add(_state(), (40, 41), _mix({1: 1.0}), _mix({1: 1.0}))
    assert comparison.summary()["flop|2p|facing"]["count"] == 3
    path = tmp_path / "rows.jsonl"
    comparison.write_rows(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["served"] == [0.0, 1.0, 0.0]


class _Refused(RuntimeError):
    pass


class _Live:
    def __init__(self):
        self.error = None

    def get_mix(self, state, hero_cards):
        if self.error is not None:
            raise self.error
        return _mix({0: 1.0})


class _Lookup:
    def get_mix(self, state, hero_cards):
        return _mix({1: 1.0})


class _Recorder:
    def __init__(self):
        self.calls = []

    def add(self, state, hero_cards, served, blueprint):
        self.calls.append((served.freqs, blueprint.freqs))


def test_fallback_source_compares_only_served_live_mixes():
    recorder = _Recorder()
    live = _Live()
    source = cli.ProductionFallbackSource(
        live, _Lookup(), (_Refused,), live_streets=frozenset({1}), mix_compare=recorder
    )
    source.get_mix(_state(street=1), (0, 1))
    source.get_mix(_state(street=2), (0, 1))  # gated: lookup acts, no comparison
    live.error = _Refused("no")
    source.get_mix(_state(street=1), (0, 1))  # fallback: no comparison
    assert recorder.calls == [({0: 1.0}, {1: 1.0})]


def test_parse_args_record_mix_compare_requires_production(tmp_path):
    base = ["--mode", "ab", "--out", str(tmp_path), "--field-source", "lookup"]
    args = cli.parse_args(run_match, base + ["--hero-source", "production", "--record-mix-compare"])
    assert args.record_mix_compare is True
    with pytest.raises(SystemExit):
        cli.parse_args(run_match, base + ["--record-mix-compare"])
