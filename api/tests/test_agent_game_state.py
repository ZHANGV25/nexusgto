"""Immutable GameState and seat/turn correctness.

The CANARY: who acts next after a raise/bet, with the correct SEAT identity.
The failure it guards against is compact-slot vs full-seat confusion;
GameState carries the slot->seat map explicitly and these tests pin it.
"""

from __future__ import annotations

import pytest

from app.engine.agent.game_state import (
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


# ── Canonical flop chips (VERIFIED convention) ───────────────────────────
#
# The flop-entry pot/stacks are NOT free placeholders once the vocab prunes by
# commitment — they must equal the trainer's canonical postflop chips for the
# preflop line. VERIFIED against the blueprint σ̄ length along the full raise-war
# line on both prefixes (the 0.7 smallest-preflop-open convention): 2p
# pot=600/stacks 9725; 3p pot=875/stacks 9725. A hand-picked 440/720 is
# chip-INCONSISTENT and gives a wrong prune depth at the deepest node.

from app.engine.agent.game_state import canonical_flop_chips  # noqa: E402


def test_canonical_flop_chips_2p():
    pot, stacks = canonical_flop_chips([2, 0, 0, 0, 0, 1], ["BB", "UTG"])
    assert pot == 600
    assert stacks == (9725, 9725)  # slot order SB-first: BB, UTG


def test_canonical_flop_chips_3p():
    pot, stacks = canonical_flop_chips([2, 0, 0, 1, 0, 1], ["BB", "UTG", "BTN"])
    assert pot == 875
    assert stacks == (9725, 9725, 9725)  # BB, UTG, BTN


def test_new_postflop_state_computes_canonical_chips_when_not_given():
    # With pot/stacks omitted, the factory derives the verified convention.
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[],
    )
    assert st.pot == 875
    assert st.stacks == (9725, 9725, 9725)


# ── Real chip accounting in advance() ────────────────────────────────────


def test_advance_bet_uses_real_trainer_sizing():
    # 3p flop root: pot=875, BB stack=9725, no-bet vocab [check, bet0.75].
    # BB bets: ba = int(0.75*875) = 656. pot 875+656=1531, BB stack 9069,
    # current_bet=656.
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[],
    )
    st = st.advance(1)  # BB bet
    assert st.pot == 1531
    assert st.stacks[0] == 9069  # BB (slot 0) stack reduced by the bet
    assert st.current_bet == 656
    assert st.street_bets[0] == 656


def test_advance_call_decrements_stack_and_adds_to_pot():
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[],
    )
    st = st.advance(1)  # BB bet 656 -> pot 1531
    # UTG (slot 1) faces to_call=656; calls (idx 1).
    st = st.advance(1)
    assert st.pot == 1531 + 656
    assert st.stacks[1] == 9725 - 656  # UTG stack reduced by the call
    assert st.street_bets[1] == 656


def test_advance_pot_carries_stacks_persist_bets_reset_on_street_close():
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[10, 20, 30],
    )
    st = st.advance(1)  # BB bet 656
    st = st.advance(1)  # UTG call 656
    st = st.advance(1)  # BTN call 656 -> flop closes
    assert st.street == 2  # turn
    # pot carried forward; current-street bets reset; stacks persist.
    assert st.pot == 875 + 656 * 3
    assert st.current_bet == 0
    assert st.street_bets == (0, 0, 0)
    assert st.stacks == (9725 - 656, 9725 - 656, 9725 - 656)


def test_num_legal_actions_chip_aware_on_committed_node():
    # Drive a 2p line into a committed faced-bet node and assert the raise slot
    # is pruned (num_legal_actions == 2), not the deep-stack 3. It uses a raise
    # war that exhausts the stack: BB bet, UTG raise, BB raise(cap), then UTG
    # faces a near-all-in -> committed.
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[],
    )
    # 2p flop: pot=600, stacks 9725. BB bet, UTG raise, BB raise (cap=3 raises),
    # UTG now faces a re-raise; by the cap raises_so_far==3 -> [fold,call] (2).
    st = st.advance(1)  # BB bet
    st = st.advance(2)  # UTG raise
    st = st.advance(2)  # BB raise (3rd raise -> at cap)
    # UTG faces, raises_this_street == 3 -> capped to [fold, call].
    assert st.facing_bet is True
    assert st.num_legal_actions() == 2


# -- street_history: the per-street action tail for live-resolve node replay --


def test_street_history_tracks_and_resets_on_street_close():
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[20, 30, 40],
    )
    assert st.street_history == ()
    st = st.advance(1)  # BB bets
    assert st.street_history == (1,)
    assert st.postflop_history == (1,)
    st = st.advance(1)  # UTG calls -> flop round closes -> turn root
    assert st.street == 2
    assert st.street_history == ()           # reset at street close
    assert st.postflop_history == (1, 1)     # full tail preserved
    st = st.advance(0)  # BB checks the turn
    assert st.street_history == (0,)
    assert st.postflop_history == (1, 1, 0)


def test_street_history_is_postflop_history_suffix_mid_street():
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[20, 30, 40],
    )
    st = st.advance(0)  # BB check
    st = st.advance(1)  # UTG bet
    st = st.advance(2)  # BTN raise
    assert st.street_history == (0, 1, 2)
    n = len(st.street_history)
    assert st.postflop_history[-n:] == st.street_history
