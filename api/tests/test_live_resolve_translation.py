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
    _parse_preflop_decisions,
    _prefix_up_to,
    seat_range_weight_dicts,
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
    # A label outside the legal set is a vocab BUG -> ValueError (fail loud,
    # never the lookup-fallback type — review 2026-06-10).
    with pytest.raises(ValueError):
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
    # A missing class row is a range-construction BUG (the hero's combo can
    # never be board-blocked) -> ValueError, never the lookup-fallback type.
    grid = [[(_ci("As"), _ci("Ks"), 1.0)]]  # only AKs in the grid
    with pytest.raises(ValueError):
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


def test_build_inputs_faced_bet_first_action_builds_replay():
    # FORMERLY rejected (the v1 fail-loud guard). Now a mid-street entry is
    # SERVED: the solve is posed from the street ROOT (chips rolled back) and
    # this street's actions are returned for node replay inside the tree.
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    st = st.advance(1)  # BB bets -> UTG faces a bet as its first action
    assert st.facing_bet is True
    assert st.acted[st.to_act_slot] is False
    inp = build_solve_inputs(st)
    assert inp["street_actions"] == [1]      # the BB bet, to be replayed
    assert inp["has_to_call"] is True
    assert inp["n_legal"] == 3               # fold / call / raise
    assert inp["hero_player_idx"] == 1       # UTG
    # Chips are STREET-ENTRY chips: BB's bet rolled back out of pot/stacks.
    assert abs(inp["pot_bb"] - 6.0) < 1e-9
    assert abs(inp["stack_bb"] - 97.25) < 1e-9
    # The leaf-rollout history must NOT include this street's actions (the
    # solved tree adds them): preflop prefix only here.
    assert inp["game_action_indices"] == [2, 0, 0, 0, 0, 1]


def test_build_inputs_mid_street_reentry_builds_replay():
    # After BB bets and UTG raises, BB faces the raise — post-raise re-entry.
    # FORMERLY rejected; now returns both actions for replay.
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    st = st.advance(1)  # BB bet (BB acted)
    st = st.advance(2)  # UTG raise -> back to BB, who already acted
    assert st.to_act_slot == 0  # BB
    assert st.acted[0] is True
    inp = build_solve_inputs(st)
    assert inp["street_actions"] == [1, 2]
    assert inp["has_to_call"] is True
    assert inp["hero_player_idx"] == 0
    # Street-entry chips again (both the bet and the raise rolled back).
    assert abs(inp["pot_bb"] - 6.0) < 1e-9
    assert abs(inp["stack_bb"] - 97.25) < 1e-9


def test_build_inputs_excludes_prior_street_folder_includes_this_street_folder():
    # 3p: BTN folds to a flop bet (this-street folder -> INCLUDED in the flop
    # tree at the time, but by the TURN he's a prior-street folder -> EXCLUDED
    # from the turn solve). BB + UTG continue. Board carries the turn card so
    # the turn-entry solve passes the board-length guard.
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d"), _ci("7h")],
    )
    # Flop: BB bets, UTG calls, BTN folds -> round closes -> turn root.
    st_flop_faced = st.advance(1)         # BB bet; UTG to act
    inp_flop = build_solve_inputs(st_flop_faced)
    assert len(inp_flop["player_ranges"]) == 3   # all 3 in the flop tree
    st_turn = st_flop_faced.advance(1).advance(0)  # UTG call, BTN fold -> turn
    assert st_turn.street == 2
    inp_turn = build_solve_inputs(st_turn)
    # BTN (prior-street folder) is OUT of the turn solve.
    assert inp_turn["positions_sorted"] == ["BB", "UTG"]
    assert len(inp_turn["player_ranges"]) == 2
    assert inp_turn["street_actions"] == []      # turn root
    # Turn-entry pot includes the flop bet+call chips.
    assert inp_turn["pot_bb"] > 6.0


def test_build_inputs_this_street_folder_does_not_cap_effective_stack():
    # A SHORT stack that folds this street stays IN the tree (fold line is
    # replayed) but must NOT set the solve's effective stack (review
    # 2026-06-10: min over entry players let a dead 20bb stack shrink a
    # ~97bb confrontation, silently).
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
        pot=875, stacks=[9725, 9725, 2000],   # BTN short
    )
    st = st.advance(0)                  # BB checks
    st = st.advance(1)                  # UTG bets
    st = st.advance_observed("fold").state  # short BTN folds -> BB to act
    assert st.to_act_slot == 0 and st.facing_bet
    inp = build_solve_inputs(st)
    assert len(inp["player_ranges"]) == 3       # folder IS in the tree
    assert abs(inp["stack_bb"] - 97.25) < 1e-9  # ... but doesn't cap stacks


def test_build_inputs_rejects_street_board_mismatch():
    # A turn-street state whose board still has only 3 cards would silently
    # solve the WRONG street — must fail loud (the driver forgot to inject
    # the dealt card).
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    st = st.advance(0).advance(0)  # flop checks through -> turn root
    assert st.street == 2 and len(st.board) == 3
    with pytest.raises(ValueError):
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


# ── range-seeding decision parsing for the SERVED shapes (2026-06-11) ─────
# Node histories below are pinned to the trainer captures
# (preflop_shapes_captured.json sz1 walk node prefix_before values;
# limp_shapes_captured.json BB-option node check_idx=0).


def test_parse_decisions_single_open_unchanged():
    # The original v1 behavior: raiser = 'any raise' (2,3), callers = (1,).
    d = _parse_preflop_decisions([2, 0, 0, 1, 0, 1])
    assert d == {
        "UTG": [([], (2, 3))],
        "BTN": [([2, 0, 0], (1,))],
        "BB": [([2, 0, 0, 1, 0], (1,))],
    }


def test_parse_decisions_threebet_hu_sz1():
    # BTN-open / BB-3-bet 1.0x: [0,0,0,2,0,3,1]. Capture-pinned histories:
    # open at [0,0,0]; 3-bet (EXACT index 3) at [0,0,0,2,0]; the opener's
    # closing call at the full first orbit [0,0,0,2,0,3].
    d = _parse_preflop_decisions([0, 0, 0, 2, 0, 3, 1])
    assert d == {
        "BTN": [([0, 0, 0], (2,)), ([0, 0, 0, 2, 0, 3], (1,))],
        "BB": [([0, 0, 0, 2, 0], (3,))],
    }


def test_parse_decisions_threebet_hu_sz0_with_folds_after():
    # UTG-open / BTN-3-bet 0.7x (folds between AND after the 3-bettor).
    d = _parse_preflop_decisions([2, 0, 0, 2, 0, 0, 1])
    assert d == {
        "UTG": [([], (2,)), ([2, 0, 0, 2, 0, 0], (1,))],
        "BTN": [([2, 0, 0], (2,))],
    }


def test_parse_decisions_limp_lines():
    # UTG-limp: the BB's check is INDEX 0 at the BB-option node (layout
    # [check, 0.7x, 1.0x] — NOT the facing-bet call index 1).
    assert _parse_preflop_decisions([1, 0, 0, 0, 0, 0]) == {
        "UTG": [([], (1,))],
        "BB": [([1, 0, 0, 0, 0], (0,))],
    }
    # SB-complete.
    assert _parse_preflop_decisions([0, 0, 0, 0, 1, 0]) == {
        "SB": [([0, 0, 0, 0], (1,))],
        "BB": [([0, 0, 0, 0, 1], (0,))],
    }


def test_parse_decisions_unparsed_shapes_return_none():
    assert _parse_preflop_decisions([2, 0, 0, 2, 0, 0, 2, 1]) is None  # 4-bet
    assert _parse_preflop_decisions([2, 1, 0, 0, 0, 2, 1, 1]) is None  # squeeze
    assert _parse_preflop_decisions([1, 0, 0, 1, 0, 0]) is None  # multi-limp
    assert _parse_preflop_decisions([3, 0, 0, 0, 0, 1]) is None  # 1.0x open
    assert _parse_preflop_decisions([0, 0, 0, 3, 0, 2, 1]) is None  # 1.0x open
    assert _parse_preflop_decisions([]) is None
    assert _parse_preflop_decisions([0, 0, 0, 2, 0, 3]) is None  # unclosed


def test_seat_range_weights_threebet_chain_rule_product():
    """The 3-bet opener's range = P(open | class) x P(call | class) — the
    chain rule over its TWO decision nodes; the 3-bettor uses its EXACT
    3-bet index. Fake sigma-bar pins the queried (history, player) nodes."""

    def get_strategy(board, history, player, bucket, street):
        assert board == [] and street == 0
        if history == [0, 0, 0] and player == 5:        # BTN open node
            return [0.2, 0.3, 0.5, 0.0]                 # P(idx2) = 0.5
        if history == [0, 0, 0, 2, 0] and player == 1:  # BB vs-open node
            return [0.5, 0.3, 0.0, 0.2]                 # P(idx3) = 0.2
        if history == [0, 0, 0, 2, 0, 3] and player == 5:  # BTN vs-3-bet
            return [0.4, 0.3, 0.3]                      # P(call=idx1) = 0.3
        raise AssertionError(f"unexpected node {history} p{player}")

    # Slot order SB-first for live ["BB", "BTN"]: BB (seat 1), BTN (seat 5).
    out = seat_range_weight_dicts([1, 5], [0, 0, 0, 2, 0, 3, 1], get_strategy)
    assert out is not None
    bb, btn = out
    assert abs(bb["AA"] - 0.2) < 1e-12
    assert abs(btn["AA"] - 0.5 * 0.3) < 1e-12


def test_seat_range_weights_limp_check_uses_bb_check_index_0():
    def get_strategy(board, history, player, bucket, street):
        if history == [] and player == 2:               # UTG limp node
            return [0.3, 0.6, 0.1, 0.0]                 # P(call=idx1) = 0.6
        if history == [1, 0, 0, 0, 0] and player == 1:  # BB-option node
            return [0.7, 0.2, 0.1]                      # P(check=idx0) = 0.7
        raise AssertionError(f"unexpected node {history} p{player}")

    out = seat_range_weight_dicts([1, 2], [1, 0, 0, 0, 0, 0], get_strategy)
    assert out is not None
    bb, utg = out
    assert abs(bb["AA"] - 0.7) < 1e-12
    assert abs(utg["AA"] - 0.6) < 1e-12


def test_seat_range_weights_unparsed_shape_returns_none():
    def get_strategy(*a):  # pragma: no cover - must not be queried
        raise AssertionError("no sigma read for an unparsed shape")

    assert seat_range_weight_dicts(
        [1, 5], [0, 0, 0, 2, 0, 2, 2, 1], get_strategy,  # 4-bet
    ) is None


# ── LiveResolveSource RECORDS its range-seeding mode (disclosure) ──────────


class _AnyRowDict(dict):
    """all_by_player[player].get(row) -> the same labels for ANY row."""

    def __init__(self, labels):
        super().__init__()
        self._labels = labels

    def get(self, key, default=None):
        return self._labels


class _FakeEsEngine:
    def solve(self, **kwargs):
        labels = {"Check": 0.7, "Bet 75%": 0.3}
        return {0: _AnyRowDict(labels), 1: _AnyRowDict(labels)}


def test_live_resolve_source_records_uniform_seeding_without_reader():
    from app.agent.strategy_source import LiveResolveSource
    from app.agent.modes import ModeConfig

    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    src = LiveResolveSource(
        ModeConfig(mode="LIVE_RESOLVE"), cpu_es_solver=_FakeEsEngine(),
    )
    assert src.last_range_seeding is None
    mix = src.get_mix(st, (_ci("Ah"), _ci("Kh")))
    # No strategy reader -> seeding impossible -> uniform, RECORDED.
    assert src.last_range_seeding == "uniform"
    assert abs(sum(mix.freqs.values()) - 1.0) < 1e-9


def test_live_resolve_source_records_blueprint_seeding_with_reader():
    from app.agent.strategy_source import LiveResolveSource
    from app.agent.modes import ModeConfig

    class _Reader:
        def get_strategy(self, board, history, player, bucket, street):
            return [0.25, 0.25, 0.25, 0.25]

    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    src = LiveResolveSource(
        ModeConfig(mode="LIVE_RESOLVE"), cpu_es_solver=_FakeEsEngine(),
        strategy_reader=_Reader(),
    )
    src.get_mix(st, (_ci("Ah"), _ci("Kh")))
    assert src.last_range_seeding == "blueprint_preflop_sigma"
