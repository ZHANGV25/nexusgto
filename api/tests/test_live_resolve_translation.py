"""Unit tests for the LIVE_RESOLVE pure translation layer (GPU-free, fast).

The solve path is exercised by test_live_resolve_adapter.py (heavy, blueprint-
gated). Here we cover the PURE pieces — label↔index mapping, the 169-class
hero-row match, the street/river/terminal gating, and the canonical-preflop
range-seeding classification — with no solver and no blueprint.
"""

from __future__ import annotations

import pytest

from app.engine.agent.game_state import new_postflop_state
from app.engine.agent.live_resolve_adapter import (
    LiveResolveUnsupported,
    build_solve_inputs,
    hero_grid_row,
    labels_to_index_freqs,
)
from app.engine.agent.live_resolve_ranges import (
    _legacy_parse_preflop_decisions,
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


def test_build_inputs_rejects_wrong_length_range_weights():
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )

    with pytest.raises(ValueError, match="range_weight_dicts length"):
        build_solve_inputs(st, range_weight_dicts=[{}])


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


def test_parse_decisions_illegal_or_unclosed_returns_none():
    # Empty prefix — nothing to seed.
    assert _parse_preflop_decisions([]) is None
    # Out-of-tree action index (UTG has 4 legal actions; 9 is illegal) — the
    # shared transition rejects it, so seeding bails to uniform.
    assert _parse_preflop_decisions([9, 0, 0, 0, 0, 0]) is None
    # An action AFTER the round already closed (single-open closes at the BB
    # call; a 7th action is not a legal replay).
    assert _parse_preflop_decisions([2, 0, 0, 0, 0, 1, 0]) is None
    # Unclosed line (BTN opens, BB 3-bets, but the opener has not responded) —
    # the pot is not flop_ready, so it is not a postflop-seeding surface.
    assert _parse_preflop_decisions([0, 0, 0, 2, 0, 3]) is None
    # Fold-around (uncontested, 1 live) — never reaches a postflop solve.
    assert _parse_preflop_decisions([2, 0, 0, 0, 0, 0]) is None


# ── GOLDEN equality: the generic extractor reproduces the retired bespoke
# parser BYTE-FOR-BYTE on the three original shapes (single-open at several
# caller counts, 3-bet-HU, HU limp-check). The retired parser is kept private
# as _legacy_parse_preflop_decisions for exactly this comparison.


@pytest.mark.parametrize(
    "prefix",
    [
        # canonical single-open, varying caller counts:
        [2, 0, 0, 0, 0, 1],           # open + BB call (1 caller)
        [2, 0, 0, 1, 0, 1],           # open + BTN call + BB call (2 callers)
        [2, 1, 1, 1, 0, 1],           # open + 3 flat callers + BB call
        [0, 0, 2, 0, 0, 1],           # CO open + BB call
        [0, 0, 0, 0, 2, 1],           # SB open (completes to raise) + BB call
        # 3-bet heads-up (open -> 3-bet -> folds -> opener calls):
        [0, 0, 0, 2, 0, 3, 1],        # BTN open / BB 3-bet 1.0x
        [2, 0, 0, 2, 0, 0, 1],        # UTG open / BTN 3-bet 0.7x, folds after
        [0, 0, 2, 0, 0, 3, 1],        # CO open / BB 3-bet 1.0x
        # HU limp-check (single limper -> folds -> BB checks):
        [1, 0, 0, 0, 0, 0],           # UTG limp
        [0, 0, 0, 0, 1, 0],           # SB complete
        [0, 0, 1, 0, 0, 0],           # CO limp
    ],
)
def test_generic_matches_legacy_on_original_shapes(prefix):
    legacy = _legacy_parse_preflop_decisions(prefix)
    assert legacy is not None, "test prefix must be a legacy-parsed shape"
    assert _parse_preflop_decisions(prefix) == legacy


def test_generic_matches_legacy_weight_dicts_byte_identical():
    """Beyond decision-node equality, the WEIGHT DICTS themselves must be
    byte-identical between the legacy nodes and the generic public API under
    the same mock get_strategy (single-open opener widening (2, 3) included)."""
    import numpy as np

    from app.engine.agent.game_state import SEAT_NAME
    from app.engine.agent.live_resolve_ranges import _weights_array_to_name_dict
    from app.engine.agent.preflop_range import build_seat_range_weights

    def get_strategy(board, history, player, bucket, street):
        # A deterministic, node-dependent distribution over 4 slots. bucket is
        # ignored (same vector for every class) so both paths read identically.
        seed = (len(history) * 7 + player * 3 + sum(history)) % 4
        base = [0.1, 0.2, 0.3, 0.4]
        return [base[(seed + i) % 4] for i in range(4)]

    def legacy_weight_dict(nodes, slot):
        w = np.ones(169, dtype=np.float64)
        for hist, idxs in nodes[SEAT_NAME[slot]]:
            w = w * build_seat_range_weights(
                get_strategy, action_history=list(hist),
                player=slot, action_indices=list(idxs),
            )
        return _weights_array_to_name_dict(w)

    cases = [
        ([2, 0, 0, 1, 0, 1], [1, 5, 2]),        # single-open: BB, BTN, UTG
        ([0, 0, 0, 2, 0, 3, 1], [1, 5]),        # 3-bet HU: BB, BTN
        ([1, 0, 0, 0, 0, 0], [1, 2]),           # limp-check: BB, UTG
    ]
    for prefix, live_slots in cases:
        legacy_nodes = _legacy_parse_preflop_decisions(prefix)
        assert legacy_nodes is not None
        generic = seat_range_weight_dicts(live_slots, prefix, get_strategy)
        assert generic is not None
        for slot, gd in zip(live_slots, generic):
            assert gd == legacy_weight_dict(legacy_nodes, slot)


# ── NEW shapes now SEEDED (hand-derived nodes from the shared transition) ──


def test_parse_decisions_multi_limp():
    # UTG limp, MP limp, CO limp, BTN/SB fold, BB check. Each limper records a
    # call (idx 1) at its own history; BB checks (idx 0) at the option node.
    assert _parse_preflop_decisions([1, 1, 1, 0, 0, 0]) == {
        "UTG": [([], (1,))],
        "MP": [([1], (1,))],
        "CO": [([1, 1], (1,))],
        "BB": [([1, 1, 1, 0, 0], (0,))],
    }


def test_parse_decisions_squeeze_limper_call_chained():
    # UTG open, MP call, CO/BTN/SB fold, BB squeeze (0.7x, idx 2), UTG fold,
    # MP CALLS the squeeze. MP has TWO decisions chained: the flat call, then
    # the closing call of the squeeze.
    assert _parse_preflop_decisions([2, 1, 0, 0, 0, 2, 0, 1]) == {
        "UTG": [([], (2,))],
        "MP": [([2], (1,)), ([2, 1, 0, 0, 0, 2, 0], (1,))],
        "BB": [([2, 1, 0, 0, 0], (2,))],
    }


def test_parse_decisions_multiway_threebet():
    # UTG open, MP call, CO 3-bet (0.7x, idx 2), BTN/SB/BB fold, UTG call,
    # MP call. Three live: opener and flat-caller each act twice.
    assert _parse_preflop_decisions([2, 1, 2, 0, 0, 0, 1, 1]) == {
        "UTG": [([], (2,)), ([2, 1, 2, 0, 0, 0], (1,))],
        "MP": [([2], (1,)), ([2, 1, 2, 0, 0, 0, 1], (1,))],
        "CO": [([2, 1], (2,))],
    }


def test_parse_decisions_capped_4bet():
    # UTG open, folds, BB 3-bet (idx 2), UTG 4-bet (idx 2 at the one-size tier),
    # BB call. num_raises hits the cap (3); the last node offers only
    # [fold, call], so BB's closing call is idx 1.
    assert _parse_preflop_decisions([2, 0, 0, 0, 0, 2, 2, 1]) == {
        "UTG": [([], (2,)), ([2, 0, 0, 0, 0, 2], (2,))],
        "BB": [([2, 0, 0, 0, 0], (2,)), ([2, 0, 0, 0, 0, 2, 2], (1,))],
    }


def test_parse_decisions_limp_raise_includes_limper_closing_call():
    # UTG limp, MP raise (isolate), CO/BTN/SB fold, BB fold, UTG CALLS the
    # raise. The limp-raiser confound: UTG's range chains its open-limp (idx 1)
    # with its closing call of the raise (idx 1).
    assert _parse_preflop_decisions([1, 2, 0, 0, 0, 0, 1]) == {
        "UTG": [([], (1,)), ([1, 2, 0, 0, 0, 0], (1,))],
        "MP": [([1], (2,))],
    }


def test_parse_decisions_one_dot_zero_open_seeded_exact():
    # A 1.0x open (idx 3) is NOT the canonical single-open shape (that requires
    # the 0.7x open idx 2), so the opener is seeded with its EXACT index (3),
    # NOT widened to the 'any raise' (2, 3) convention.
    assert _parse_preflop_decisions([3, 0, 0, 0, 0, 1]) == {
        "UTG": [([], (3,))],
        "BB": [([3, 0, 0, 0, 0], (1,))],
    }


def test_seat_range_weights_multi_limp_hand_derived():
    # Deterministic mock: P(taken | class) is the same for every class, so the
    # per-seat weight = the product of the queried slot probabilities.
    def get_strategy(board, history, player, bucket, street):
        assert board == [] and street == 0
        table = {
            ((), 2): [0.3, 0.5, 0.2, 0.0],              # UTG limp: P(idx1)=0.5
            ((1,), 3): [0.4, 0.4, 0.2, 0.0],            # MP limp: P(idx1)=0.4
            ((1, 1), 4): [0.1, 0.6, 0.3, 0.0],          # CO limp: P(idx1)=0.6
            ((1, 1, 1, 0, 0), 1): [0.8, 0.1, 0.1],      # BB check: P(idx0)=0.8
        }
        key = (tuple(history), player)
        if key not in table:
            raise AssertionError(f"unexpected node {history} p{player}")
        return table[key]

    # slot order SB-first for live [UTG, MP, CO, BB] -> seats 2,3,4,1.
    out = seat_range_weight_dicts([2, 3, 4, 1], [1, 1, 1, 0, 0, 0], get_strategy)
    assert out is not None
    utg, mp, co, bb = out
    assert abs(utg["AA"] - 0.5) < 1e-12
    assert abs(mp["AA"] - 0.4) < 1e-12
    assert abs(co["AA"] - 0.6) < 1e-12
    assert abs(bb["AA"] - 0.8) < 1e-12


def test_seat_range_weights_limp_raise_chains_limper_closing_call():
    # UTG limps then CALLS MP's isolation raise: UTG range = P(limp) x P(call).
    def get_strategy(board, history, player, bucket, street):
        table = {
            ((), 2): [0.2, 0.5, 0.3, 0.0],              # UTG limp: P(idx1)=0.5
            ((1,), 3): [0.1, 0.2, 0.7, 0.0],            # MP raise: P(idx2)=0.7
            ((1, 2, 0, 0, 0, 0), 2): [0.3, 0.4, 0.3],   # UTG call: P(idx1)=0.4
        }
        key = (tuple(history), player)
        if key not in table:
            raise AssertionError(f"unexpected node {history} p{player}")
        return table[key]

    # live [UTG, MP] -> seats 2, 3 (SB-first order is [2,3] since neither is SB).
    out = seat_range_weight_dicts([2, 3], [1, 2, 0, 0, 0, 0, 1], get_strategy)
    assert out is not None
    utg, mp = out
    assert abs(utg["AA"] - 0.5 * 0.4) < 1e-12   # chained limp x closing call
    assert abs(mp["AA"] - 0.7) < 1e-12


def test_seat_range_weights_capped_4bet_hand_derived():
    def get_strategy(board, history, player, bucket, street):
        table = {
            ((), 2): [0.2, 0.3, 0.5, 0.0],                 # UTG open: P(idx2)=0.5
            ((2, 0, 0, 0, 0), 1): [0.4, 0.2, 0.4, 0.0],    # BB 3-bet: P(idx2)=0.4
            ((2, 0, 0, 0, 0, 2), 2): [0.3, 0.3, 0.4],      # UTG 4-bet: P(idx2)=0.4
            ((2, 0, 0, 0, 0, 2, 2), 1): [0.1, 0.9],        # BB call: P(idx1)=0.9
        }
        key = (tuple(history), player)
        if key not in table:
            raise AssertionError(f"unexpected node {history} p{player}")
        return table[key]

    # live [UTG, BB] -> seats 2, 1 -> SB-first order [1, 2] (BB=1 before UTG=2).
    out = seat_range_weight_dicts([1, 2], [2, 0, 0, 0, 0, 2, 2, 1], get_strategy)
    assert out is not None
    bb, utg = out
    assert abs(utg["AA"] - 0.5 * 0.4) < 1e-12   # open x 4-bet
    assert abs(bb["AA"] - 0.4 * 0.9) < 1e-12    # 3-bet x closing call


def test_seat_range_weights_zero_mass_seat_returns_none():
    # A live seat whose read range is entirely zero -> None (fail-safe): every
    # class reads P(taken)=0, so the seat's weight sums to 0.
    def get_strategy(board, history, player, bucket, street):
        if history == [] and player == 2:               # UTG limp: ZERO on idx1
            return [1.0, 0.0, 0.0, 0.0]
        if history == [1, 0, 0, 0, 0] and player == 1:  # BB check
            return [0.7, 0.2, 0.1]
        raise AssertionError(f"unexpected node {history} p{player}")

    assert seat_range_weight_dicts(
        [1, 2], [1, 0, 0, 0, 0, 0], get_strategy,
    ) is None


def test_seat_range_weights_unclassifiable_live_seat_returns_none():
    # A live slot with no recorded decision cannot be classified -> None. Here
    # we pass a slot (BTN=5) that is NOT live in an UTG-limp pot, so
    # decisions.get("BTN") is None and the whole seeding bails to uniform.
    def get_strategy(board, history, player, bucket, street):
        return [0.25, 0.25, 0.25, 0.25]

    assert seat_range_weight_dicts(
        [1, 2, 5], [1, 0, 0, 0, 0, 0], get_strategy,
    ) is None


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


def test_seat_range_weights_illegal_prefix_returns_none():
    # An illegal / out-of-tree prefix (the shared transition rejects the index)
    # bails to uniform WITHOUT ever reading sigma-bar.
    def get_strategy(*a):  # pragma: no cover - must not be queried
        raise AssertionError("no sigma read for an illegal prefix")

    assert seat_range_weight_dicts(
        [1, 5], [7, 0, 0, 0, 0, 0], get_strategy,  # index 7 illegal at UTG
    ) is None
    # An unclosed (not flop_ready) line likewise seeds nothing.
    assert seat_range_weight_dicts(
        [1, 5], [0, 0, 0, 2, 0, 3], get_strategy,
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
    from app.engine.agent.strategy_source import LiveResolveSource
    from app.engine.agent.modes import ModeConfig

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
    from app.engine.agent.strategy_source import LiveResolveSource
    from app.engine.agent.modes import ModeConfig

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
