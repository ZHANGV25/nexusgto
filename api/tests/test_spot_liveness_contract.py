"""Spot-liveness + game_action_indices contract (2026-07-03).

Python half of the history→hash contract whose C half is
poker-solver/tests/test_cont_ev_history_hash_contract.c. The C test pins that
the trainer's token emission equals the engine's documented token constants
for the REAL canonical 3-way line; THIS file pins that the engine actually
produces those constants end-to-end (GameState -> build_solve_inputs), and
that the game-impossible construction behind the Stage-2 hash disjunction is
now rejected loudly.

Background (docs/STAGE2_HASH_DISJUNCTION_RESOLVED_2026-07-03.md): every
cont_ev harness since 2026-06-12 built its spot as all-six-seats-live over the
heads-up prefix [2,0,0,0,0,1] (UTG opens, four FOLD, BB calls). The canonical
chip replay validated liveness for limped and 3-bet shapes but NOT the
single-raise shape, so the fiction built silently; the producer's multiway-only
boundary dispatch means no walk table can contain keys for any continuation of
that prefix — hence 0 hits / 25,684,620 gate queries.
"""
from __future__ import annotations

from dataclasses import replace as dc_replace

import pytest

from app.engine.agent.game_state import new_postflop_state
from app.engine.agent.live_resolve_adapter import build_solve_inputs

SIX_SEATS = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
THREE_SEATS = ["UTG", "BTN", "BB"]
HU_PREFIX = [2, 0, 0, 0, 0, 1]          # UTG opens, 4 folds, BB calls
THREEWAY_PREFIX = [2, 0, 0, 1, 0, 1]    # UTG opens, BTN calls, BB calls
FLOP = [0 * 4 + 2, 11 * 4 + 3, 8 * 4 + 1]   # any 3 distinct cards; ints only
TURN = FLOP + [5 * 4 + 0]


def test_fictional_six_live_over_hu_prefix_rejected():
    """The exact pre-fix harness construction must now fail loudly."""
    with pytest.raises(ValueError, match="do not match the live players"):
        new_postflop_state(
            seats=list(SIX_SEATS), preflop_prefix=list(HU_PREFIX),
            board=list(FLOP), street=1,
        )


def test_real_hu_line_still_accepted():
    """[2,0,0,0,0,1] with the TRUE live set {UTG,BB} is a real line and must
    keep working (the shipped HU live-solve path)."""
    state = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=list(HU_PREFIX),
        board=list(FLOP), street=1,
    )
    inputs = build_solve_inputs(state, range_weight_dicts=None)
    assert inputs["game_action_indices"] == HU_PREFIX


def test_threeway_flop_spot_tokens_and_seats():
    """The re-gate spot: derived canonical chips + trainer-exact tokens.

    pot=875 and per-player stacks 9725 cross-check the C contract test's
    hand-derived trainer arithmetic (3 x 275 + SB's dead 50)."""
    state = new_postflop_state(
        seats=list(THREE_SEATS), preflop_prefix=list(THREEWAY_PREFIX),
        board=list(FLOP), street=1,
    )
    assert state.pot == 875
    assert all(s == 9725 for s in state.stacks)
    inputs = build_solve_inputs(state, range_weight_dicts=None)
    assert inputs["game_action_indices"] == THREEWAY_PREFIX
    assert inputs["positions_sorted"] == ["BB", "UTG", "BTN"]  # SB-first compact
    assert inputs["player_seats"] == [1, 2, 5]                 # FULL seats
    assert inputs["street_actions"] == []


def test_threeway_turn_spot_includes_flop_tokens():
    """Turn entry built by advancing a checked-through flop: the turn
    history MUST carry the three flop check tokens (the pre-fix turn harness
    dropped them by seeding a fresh street=2 state with flop-entry chips)."""
    state = new_postflop_state(
        seats=list(THREE_SEATS), preflop_prefix=list(THREEWAY_PREFIX),
        board=list(FLOP), street=1,
    )
    for _ in range(len(THREE_SEATS)):
        state = state.advance(0)  # check
    assert state.street == 2, "checked-through flop must close the round"
    state = dc_replace(state, board=tuple(TURN))
    inputs = build_solve_inputs(state, range_weight_dicts=None)
    assert inputs["game_action_indices"] == THREEWAY_PREFIX + [0, 0, 0]
    assert inputs["street_actions"] == []
    # Chips carry: three checks add nothing.
    assert inputs["pot_bb"] == pytest.approx(875 / 100.0)
