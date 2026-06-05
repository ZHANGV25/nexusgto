"""TDD: immutable GameState + seat/turn correctness (plan §3.1, §4a canary).

The CANARY (the bug hit last session): who-acts-next after a raise/bet, with
correct SEAT identity. The root cause was compact-slot vs full-seat confusion;
GameState carries the slot->seat map explicitly and these tests pin it.
"""

from __future__ import annotations

import pytest

from app.agent.game_state import (
    FULL_SEAT,
    new_postflop_state,
)

# ── Construction + the slot<->seat seam ──────────────────────────────────


def test_full_seat_map_matches_blueprint_encoding():
    # SB0, BB1, UTG2, MP3, CO4, BTN5 (capture-verified encoding).
    assert FULL_SEAT == {"SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5}


def test_new_postflop_state_3p_seats_and_first_to_act():
    # 3p line UTG raise / BTN call / BB call. Postflop SB-first order over
    # live seats => BB acts first (SB folded preflop / not in hand).
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"],
        preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[],
        pot=720,
        stacks=[9660, 9780, 9780],
    )
    # Compact slots are ordered SB-first by full seat: BB(1) < UTG(2) < BTN(5)
    assert st.seat_of_slot == (1, 2, 5)
    # First to act postflop is the lowest-seat live player = BB (slot 0).
    assert st.to_act_slot == 0
    assert st.seat_of_slot[st.to_act_slot] == 1  # BB


def test_new_postflop_state_2p_seats():
    # 2p line UTG raise, fold to BB, BB call. BB(1) acts first postflop.
    st = new_postflop_state(
        seats=["BB", "UTG"],
        preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[],
        pot=440,
        stacks=[9780, 9780],
    )
    assert st.seat_of_slot == (1, 2)
    assert st.to_act_slot == 0  # BB
    assert st.street == 1  # flop


# ── Immutability ──────────────────────────────────────────────────────────


def test_state_is_frozen():
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[], pot=440, stacks=[9780, 9780],
    )
    with pytest.raises(Exception):
        st.pot = 999  # type: ignore[misc]


def test_advance_returns_new_object_does_not_mutate():
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[], pot=440, stacks=[9780, 9780],
    )
    before_hist = st.postflop_history
    st2 = st.advance(action_idx=0)  # BB checks
    assert st2 is not st
    assert st.postflop_history == before_hist  # original untouched
    assert st.postflop_history == ()
    assert st2.postflop_history == (0,)


# ── CANARY: who-acts-next after a check vs after a bet (3-way) ────────────


def test_canary_3way_check_passes_to_next_seat_utg_not_btn():
    # BB checks the flop. Next to act is UTG (seat 2), NOT BTN (seat 5).
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[], pot=720, stacks=[9660, 9780, 9780],
    )
    assert st.seat_of_slot[st.to_act_slot] == 1  # BB first
    st = st.advance(action_idx=0)  # BB check
    assert st.seat_of_slot[st.to_act_slot] == 2  # UTG next, NOT BTN(5)
    st = st.advance(action_idx=0)  # UTG check
    assert st.seat_of_slot[st.to_act_slot] == 5  # BTN last


def test_canary_3way_bet_reopens_action_to_utg_then_btn():
    # BB BETS the flop. Action goes to UTG (seat 2) facing a bet, then BTN.
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[], pot=720, stacks=[9660, 9780, 9780],
    )
    st = st.advance(action_idx=1)  # BB bet (no-bet vocab: check0, bet1)
    assert st.facing_bet is True
    assert st.seat_of_slot[st.to_act_slot] == 2  # UTG faces the bet
    st = st.advance(action_idx=1)  # UTG call (facing-bet vocab: fold0,call1,raise2)
    assert st.seat_of_slot[st.to_act_slot] == 5  # BTN faces it next


def test_canary_raise_reopens_to_already_acted_players():
    # BB check, UTG bet, BTN call, action back to BB (who must respond).
    # Order is forward-from-bettor (real-poker position order), NOT
    # rollout's scan-from-slot0: after UTG bets, BTN(5) acts BEFORE the
    # action wraps back to BB. Capture-verified: [2,0,0,1,0,1,0,1] HITs at
    # p=5(BTN), MISSES at p=1(BB); and [...,0,1,1] wraps to p=1(BB).
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[], pot=720, stacks=[9660, 9780, 9780],
    )
    st = st.advance(0)   # BB check
    st = st.advance(1)   # UTG bet -> reopens action
    assert st.seat_of_slot[st.to_act_slot] == 5  # BTN faces bet
    st = st.advance(1)   # BTN call
    # BB checked then a bet came in behind -> BB must act again (facing bet).
    assert st.seat_of_slot[st.to_act_slot] == 1  # back to BB
    assert st.facing_bet is True


# ── Street advance on round closure ──────────────────────────────────────


def test_round_closes_on_check_around_advances_street():
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[], pot=720, stacks=[9660, 9780, 9780],
    )
    assert st.street == 1  # flop
    st = st.advance(0)  # BB check
    st = st.advance(0)  # UTG check
    st = st.advance(0)  # BTN check -> flop round closes
    assert st.street == 2  # turn
    assert st.betting_round_closed_count == 1
    # Turn: BB first to act again, no bet outstanding.
    assert st.seat_of_slot[st.to_act_slot] == 1
    assert st.facing_bet is False


def test_round_closes_on_bet_call_call_advances_street():
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[], pot=720, stacks=[9660, 9780, 9780],
    )
    st = st.advance(1)  # BB bet
    st = st.advance(1)  # UTG call
    st = st.advance(1)  # BTN call -> flop round closes
    assert st.street == 2  # turn
    assert st.facing_bet is False
    assert st.seat_of_slot[st.to_act_slot] == 1  # BB first on turn
