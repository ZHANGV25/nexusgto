"""Pure tests for postflop range conditioning (no blueprint needed)."""
from __future__ import annotations

import pytest

from app.engine.agent import postflop_conditioning as pc
from scripts import run_match_cli as cli


class _Mix:
    def __init__(self, freqs, uniform=False):
        self.freqs = freqs
        self.is_uniform_fallback = uniform


@pytest.mark.parametrize("raw,expected", [(None, False), ("0", False), ("1", True), ("yes", True)])
def test_enabled_parses_flag(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(pc.ENV, raising=False)
    else:
        monkeypatch.setenv(pc.ENV, raw)
    assert pc.enabled() is expected


def _get_mix(state, cards):
    # state is a label; combo (10, 11) checks 80%, (12, 13) checks 20%, (14, 15) never checks
    table = {(10, 11): 0.8, (12, 13): 0.2, (14, 15): 0.0}
    check = table[cards]
    return _Mix({0: check, 1: 1.0 - check}, uniform=(state == "uniform"))


def test_conditioning_multiplies_rows_by_the_taken_action_probability():
    ranges = [[(10, 11, 1.0), (12, 13, 1.0), (14, 15, 2.0)], [(10, 11, 1.0), (12, 13, 1.0)]]
    decisions = [(0, "flop", 0), (1, "flop", 1), (0, "turn", 0)]  # slot 0 checked twice, slot 1 bet once
    out, stats = pc.condition_player_ranges(decisions, ranges, [0, 1], _get_mix)
    assert [round(w, 4) for _c0, _c1, w in out[0]] == [0.64, 0.04, 0.0]
    assert [round(w, 4) for _c0, _c1, w in out[1]] == [0.2, 0.8]
    assert stats["players_conditioned"] == 2 and stats["players_zero_mass"] == 0
    assert stats["lookups"] == 3 + 2 + 2  # zero-factor rows are skipped on the second pass
    assert stats["decisions"] == 3
    assert [round(m, 4) for m in stats["mass_kept"]] == [round(0.68 / 4.0, 4), 0.5]


def test_zero_mass_player_keeps_its_unconditioned_range():
    ranges = [[(14, 15, 1.0)], [(10, 11, 1.0)]]
    decisions = [(0, "flop", 0)]  # slot 0's only combo never checks
    out, stats = pc.condition_player_ranges(decisions, ranges, [0, 1], _get_mix)
    assert out[0] == [(14, 15, 1.0)] and out[1] == [(10, 11, 1.0)]
    assert stats["players_zero_mass"] == 1 and stats["players_conditioned"] == 0


def test_no_earlier_decisions_is_a_no_op():
    ranges = [[(10, 11, 0.5)], [(12, 13, 0.5)]]
    out, stats = pc.condition_player_ranges([], ranges, [0, 1], _get_mix)
    assert out == [[(10, 11, 0.5)], [(12, 13, 0.5)]]
    assert stats["lookups"] == 0 and stats["decisions"] == 0


def test_uniform_fallback_rows_are_counted():
    ranges = [[(10, 11, 1.0)]]
    out, stats = pc.condition_player_ranges([(0, "uniform", 0)], ranges, [0], _get_mix)
    assert stats["uniform_rows"] == 1 and out[0][0][2] == pytest.approx(0.8)


def test_run_match_accepts_the_conditioning_override():
    assert cli._env_override("NEXUS_RANGE_POSTFLOP_CONDITIONING=1") == (
        "NEXUS_RANGE_POSTFLOP_CONDITIONING", "1"
    )


def test_fallback_source_aggregates_conditioning_stats():
    class _Live:
        last_range_conditioning = {
            "decisions": 2, "lookups": 300, "uniform_rows": 4,
            "players_conditioned": 2, "players_zero_mass": 0, "mass_kept": [0.3, 0.4],
        }

        def get_mix(self, state, hero_cards):
            return "live"

    source = cli.ProductionFallbackSource(_Live(), object(), (RuntimeError,))
    assert source.get_mix(None, (0, 1)) == "live"
    assert source.range_conditioning == {
        "solves": 1, "with_earlier_streets": 1, "players_conditioned": 2,
        "players_zero_mass": 0, "lookups": 300, "uniform_rows": 4,
    }
