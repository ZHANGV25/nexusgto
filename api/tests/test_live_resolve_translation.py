"""Unit tests for the LIVE_RESOLVE pure translation layer (GPU-free, fast).

The solve path is exercised by test_live_resolve_adapter.py (heavy, blueprint-
gated). Here we cover the PURE pieces — label↔index mapping, the 169-class
hero-row match, the street/river/terminal gating, and the canonical-preflop
range-seeding classification — with no solver and no blueprint.
"""

from __future__ import annotations

import pytest

from app.agent.game_state import new_postflop_state
from app.agent.live_resolve_adapter import (
    LiveResolveUnsupported,
    build_solve_inputs,
    hero_grid_row,
    labels_to_index_freqs,
)
from app.agent.live_resolve_ranges import (
    _parse_canonical_preflop,
    _prefix_up_to,
)


def _ci(card: str) -> int:
    ranks = "AKQJT98765432"
    so = ("c", "d", "h", "s")
    return (12 - ranks.index(card[0].upper())) * 4 + so.index(card[1].lower())


# ── label -> index mapping (the agent vocab is [0.75]) ───────────────────


def test_labels_to_index_no_bet_node():
    # not facing a bet: vocab [check=0, bet 0.75=1].
    freqs = labels_to_index_freqs(
        {"Check": 0.7, "Bet 75%": 0.3}, has_to_call=False, n_legal=2,
        bet_sizes=(0.75,), add_explicit_allin=False,
    )
    assert freqs == {0: 0.7, 1: 0.3}


def test_labels_to_index_faced_bet_node():
    # facing a bet: vocab [fold=0, call=1, raise 0.75=2].
    freqs = labels_to_index_freqs(
        {"Fold": 0.2, "Call": 0.5, "Raise 75%": 0.3}, has_to_call=True,
        n_legal=3, bet_sizes=(0.75,), add_explicit_allin=False,
    )
    assert freqs == {0: 0.2, 1: 0.5, 2: 0.3}


def test_labels_to_index_rounding_snaps_to_same_slot():
    # "Bet 74%" / "Bet 76%" both snap to the single 0.75 slot and SUM.
    freqs = labels_to_index_freqs(
        {"Bet 74%": 0.4, "Bet 76%": 0.2}, has_to_call=False, n_legal=2,
        bet_sizes=(0.75,), add_explicit_allin=False,
    )
    assert set(freqs) == {1}
    assert abs(freqs[1] - 0.6) < 1e-9


def test_labels_to_index_zero_freq_dropped():
    freqs = labels_to_index_freqs(
        {"Check": 1.0, "Bet 75%": 0.0}, has_to_call=False, n_legal=2,
        bet_sizes=(0.75,), add_explicit_allin=False,
    )
    assert freqs == {0: 1.0}


def test_labels_to_index_out_of_range_raises():
    # An "All-in" with no explicit-allin slot maps to len(sizes)+1 (=2 here when
    # facing) which is in-range; force an out-of-range by giving n_legal=1.
    with pytest.raises(LiveResolveUnsupported):
        labels_to_index_freqs(
            {"Bet 75%": 1.0}, has_to_call=False, n_legal=1,
            bet_sizes=(0.75,), add_explicit_allin=False,
        )


# ── hero grid row (169-class match, not exact combo) ─────────────────────


def test_hero_grid_row_matches_by_class():
    # A minimal grid where AKs is one representative combo; the hero holds a
    # DIFFERENT AKs combo and must still match the class row.
    grid = [
        [(_ci("As"), _ci("Ks"), 1.0)],   # slot 0: AKs representative (spades)
    ]
    grid[0].append((_ci("Qd"), _ci("Qc"), 1.0))  # QQ (pair)
    # Hero holds AhKh (hearts) — same AKs class as the spade representative.
    row = hero_grid_row(grid, 0, (_ci("Ah"), _ci("Kh")))
    assert row == 0


def test_hero_grid_row_no_class_raises():
    grid = [[(_ci("As"), _ci("Ks"), 1.0)]]  # only AKs in the grid
    with pytest.raises(LiveResolveUnsupported):
        hero_grid_row(grid, 0, (_ci("2c"), _ci("7d")))  # 72o not present


# ── build_solve_inputs gating: terminal / river / mid-street re-entry ────


def test_build_inputs_rejects_river():
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    for idx in [0, 0, 0, 0]:  # flop check-through -> turn, turn checks -> ...
        st = st.advance(idx)
    st = st.advance(0)
    st = st.advance(0)  # turn check-through -> river root
    assert st.street == 3
    with pytest.raises(LiveResolveUnsupported):
        build_solve_inputs(st)


def test_build_inputs_rejects_terminal():
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    # BB bets, UTG folds -> terminal.
    st = st.advance(1)  # BB bet
    st = st.advance(0)  # UTG fold -> one player left, terminal
    assert st.terminal
    with pytest.raises(LiveResolveUnsupported):
        build_solve_inputs(st)


def test_build_inputs_rejects_faced_bet_first_action():
    # The bug the acted[hero] guard MISSED (advisor 2026-06-06): UTG faces BB's
    # bet as its FIRST action this street — acted[UTG]=False but facing_bet=True.
    # The first-decision-node read would serve the check-line node remapped as a
    # no-fold {call,raise} mix. Must fail loud, NOT solve.
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    st = st.advance(1)  # BB bets -> UTG faces a bet as its first action
    assert st.facing_bet is True
    assert st.acted[st.to_act_slot] is False  # the guard the old check missed
    with pytest.raises(LiveResolveUnsupported):
        build_solve_inputs(st)


def test_build_inputs_rejects_mid_street_reentry():
    # After BB bets and UTG raises, BB faces the raise — post-raise re-entry,
    # not the street-root first-decision node: reject.
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    st = st.advance(1)  # BB bet (BB acted)
    st = st.advance(2)  # UTG raise -> back to BB, who already acted
    assert st.to_act_slot == 0  # BB
    assert st.acted[0] is True
    with pytest.raises(LiveResolveUnsupported):
        build_solve_inputs(st)


def test_build_inputs_allows_check_through_to_hero():
    # A legit non-root node that IS the hero's first-decision node: BB checks,
    # UTG to act with NO bet yet this street. Not facing a bet, no raises -> the
    # street-root check-line node == the current node. Must be ALLOWED.
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    st = st.advance(0)  # BB checks -> UTG to act, no bet faced
    assert st.facing_bet is False
    assert st.raises_this_street == 0
    inp = build_solve_inputs(st)  # must NOT raise
    assert inp["hero_player_idx"] == 1  # UTG
    assert inp["has_to_call"] is False


def test_build_inputs_flop_root_ok():
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    inp = build_solve_inputs(st)
    assert inp["positions_sorted"] == ["BB", "UTG"]
    assert inp["hero_player_idx"] == 0
    assert inp["board_strs"] == ["Qs", "Js", "2d"]
    assert inp["n_legal"] == 2
    assert inp["has_to_call"] is False
    # chips -> bb: pot 600 -> 6.0, stack 9725 -> 97.25.
    assert abs(inp["pot_bb"] - 6.0) < 1e-9
    assert abs(inp["stack_bb"] - 97.25) < 1e-9
    assert len(inp["player_ranges"]) == 2
    assert len(inp["player_ranges"][0]) == 169


# ── canonical preflop classification for range seeding ───────────────────


def test_parse_canonical_2p_line():
    # [2,0,0,0,0,1] = UTG raise, MP/CO/BTN/SB fold, BB call.
    roles = _parse_canonical_preflop([2, 0, 0, 0, 0, 1])
    assert roles == {"UTG": "raiser", "BB": "caller"}


def test_parse_canonical_3p_line():
    # [2,0,0,1,0,1] = UTG raise, MP/CO fold, BTN call, SB fold, BB call.
    roles = _parse_canonical_preflop([2, 0, 0, 1, 0, 1])
    assert roles == {"UTG": "raiser", "BTN": "caller", "BB": "caller"}


def test_parse_non_canonical_returns_none():
    assert _parse_canonical_preflop([2, 0, 0, 0, 0, 2]) is None  # 3bet
    assert _parse_canonical_preflop([1, 1, 1, 1, 1, 1]) is None  # limp pot


def test_prefix_up_to():
    # The BB caller's decision history in the 2p line is the prefix BEFORE BB:
    # UTG raise + MP/CO/BTN/SB folds = [2,0,0,0,0].
    assert _prefix_up_to([2, 0, 0, 0, 0, 1], "BB") == [2, 0, 0, 0, 0]
    # The UTG raiser acts first: empty history.
    assert _prefix_up_to([2, 0, 0, 0, 0, 1], "UTG") == []
