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


def _pooled_group(board):
    """A canonical row on ``board`` that pools several physical combos, and its members."""
    from app.services.suit_isomorphism import group_by_canonical

    blocked = set(board)
    live = [(a, b) for a in range(52) for b in range(a + 1, 52) if a not in blocked and b not in blocked]
    for canonical, members in group_by_canonical(live, board):
        if len(members) >= 3:
            return canonical, members
    raise AssertionError("board pools no suits")


RAINBOW_RIVER = [0, 5, 10, 15, 20]  # at most two cards of any suit: no flush, suits pooled


def test_row_members_lists_the_pooled_combos_of_an_iso_row():
    canonical, members = _pooled_group(RAINBOW_RIVER)
    rows = [(canonical[0], canonical[1], 1.0)]
    assert sorted(pc.row_members(rows, RAINBOW_RIVER)[0]) == sorted(members)
    assert pc.row_members(rows, None) == [[canonical]]


def test_row_members_does_not_pool_a_grid_of_physical_combos():
    canonical, members = _pooled_group(RAINBOW_RIVER)
    other = next(m for m in members if m != canonical)
    rows = [(canonical[0], canonical[1], 1.0), (other[0], other[1], 1.0)]  # ALL grid: non-canonical row present
    assert pc.row_members(rows, RAINBOW_RIVER) == [[canonical], [other]]


def test_conditioning_averages_the_members_of_a_pooled_row():
    # A flop decision where only ONE suit variant of the pooled row bets (a flush draw that
    # bricked by the river): the row keeps 1/len(members) of its weight, not 0 or 1.
    canonical, members = _pooled_group(RAINBOW_RIVER)
    drawer = members[-1]

    def get_mix(_state, cards):
        bet = 1.0 if tuple(cards) == tuple(drawer) else 0.0
        return _Mix({0: 1.0 - bet, 1: bet})

    ranges = [[(canonical[0], canonical[1], 2.0)]]
    out, stats = pc.condition_player_ranges([(0, "flop", 1)], ranges, [0], get_mix, board=RAINBOW_RIVER)
    assert out[0][0][2] == pytest.approx(2.0 / len(members))
    assert stats["lookups"] == len(members)
    legacy, _ = pc.condition_player_ranges([(0, "flop", 1)], ranges, [0], get_mix)
    assert legacy[0][0][2] in (0.0, 2.0)  # the representative-only form is all-or-nothing


def test_row_members_tolerates_a_board_colliding_row():
    # A zero-weight row holding a board card (2c2d on a board with the 2c) must not
    # abort the grouping: it stands for itself and the live rows still pool.
    canonical, members = _pooled_group(RAINBOW_RIVER)
    rows = [(canonical[0], canonical[1], 1.0), (0, 1, 0.0)]  # card 0 (2c) is on RAINBOW_RIVER
    out = pc.row_members(rows, RAINBOW_RIVER)
    assert sorted(out[0]) == sorted(members)
    assert out[1] == [(0, 1)]
