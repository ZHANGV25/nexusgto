"""advance_observed + HybridSource — the off-tree / full-hand layer.

advance_observed is the split that makes the agent playable against humans:
REAL chips into the state (pot odds, prunes, street-entry reconstruction),
SNAPPED on-tree index into the histories (blueprint keys + street replay).
HybridSource is the composition that lets a full hand finish (river falls
back to LOOKUP — CpuEsSolver's documented side-pot gap).
"""

from __future__ import annotations

import pytest

from app.agent.game_state import ObservedAdvance, new_postflop_state
from app.agent.strategy_source import (
    ActionMix,
    HybridSource,
    LiveResolveUnsupported,
    StrategySource,
)


def _ci(card: str) -> int:
    ranks = "AKQJT98765432"
    so = ("c", "d", "h", "s")
    return (12 - ranks.index(card[0].upper())) * 4 + so.index(card[1].lower())


def _st_2p():
    # Canonical 2p flop entry: pot 600, stacks 9725 each. BB (slot 0) to act.
    return new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )


# ── on-tree observed action == vocab advance, bit for bit ────────────────


def test_observed_on_tree_bet_matches_vocab_advance():
    st = _st_2p()
    canonical = st._surviving_raise_amounts(0)[0]  # the 0.75-pot bet, 450
    obs = st.advance_observed("raise", amount=canonical)
    assert isinstance(obs, ObservedAdvance)
    assert obs.action_idx == 1
    assert obs.off_tree is False
    assert obs.applied_amount == canonical
    assert obs.state == st.advance(1)  # frozen dataclass equality: all fields


def test_observed_check_and_call_paths():
    st = _st_2p()
    obs = st.advance_observed("check")
    assert (obs.action_idx, obs.off_tree, obs.applied_amount) == (0, False, 0)
    assert obs.state == st.advance(0)
    # "call" with nothing outstanding is a check (some feeds say "call").
    obs2 = st.advance_observed("call")
    assert obs2.action_idx == 0
    assert obs2.state == st.advance(0)


# ── off-tree bet: REAL chips in the state, SNAPPED index in history ──────


def test_observed_off_tree_bet_real_chips_snapped_index():
    st = _st_2p()
    obs = st.advance_observed("raise", amount=500)  # 83% pot; canonical is 450
    assert obs.action_idx == 1                       # snaps to the 0.75 slot
    assert obs.off_tree is True
    assert obs.applied_amount == 500
    st2 = obs.state
    assert st2.pot == 600 + 500                      # REAL chips
    assert st2.stacks[0] == 9725 - 500
    assert st2.current_bet == 500
    assert st2.street_bets[0] == 500
    assert st2.postflop_history == (1,)              # SNAPPED index
    assert st2.street_history == (1,)
    # The opponent's pot odds are the REAL ones: to_call 500, not 450.
    assert st2.to_act_slot == 1
    assert st2.facing_bet is True
    assert st2.current_bet - st2.street_bets[1] == 500


def test_observed_off_tree_reraise_tracks_real_chips():
    st = _st_2p().advance_observed("raise", amount=500).state  # BB bets 500
    # UTG raises BY 1300 (off-tree; canonical subsequent-1.0 raise is
    # to_call + 1.0*(pot+to_call) = 500 + (1100+500) = 2100).
    obs = st.advance_observed("raise", amount=1300)
    assert obs.action_idx == 2                       # the single raise slot
    assert obs.off_tree is True
    st2 = obs.state
    assert st2.pot == 1100 + 1300
    assert st2.stacks[1] == 9725 - 1300
    assert st2.current_bet == 1300
    assert st2.raises_this_street == 2
    assert st2.postflop_history == (1, 2)
    # BB now owes the REAL difference: 1300 - 500 = 800.
    assert st2.to_act_slot == 0
    assert st2.current_bet - st2.street_bets[0] == 800


def test_observed_under_raise_is_applied_as_call():
    st = _st_2p().advance_observed("raise", amount=500).state  # BB bets 500
    obs = st.advance_observed("raise", amount=300)   # under-raise (< to_call)
    assert obs.action_idx == 1                       # recorded as CALL
    # normalized to a full call (UTG is not all-in; malformed-feed handling)
    assert obs.applied_amount == 500
    # the call CLOSED the flop round -> street advanced, real chips carried.
    assert obs.state.street == 2
    assert obs.state.pot == 600 + 500 + 500
    assert obs.state.stacks == (9225, 9225)


def test_observed_allin_snaps_to_raise_slot():
    st = _st_2p()
    obs = st.advance_observed("allin")               # 9725 into a 600 pot
    assert obs.off_tree is True                      # canonical is 450
    assert obs.action_idx == 1                       # nearest (only) raise slot
    assert obs.applied_amount == 9725
    assert obs.state.stacks[0] == 0
    assert obs.state.pot == 600 + 9725


def test_observed_illegal_inputs_fail_loud():
    st = _st_2p()
    with pytest.raises(ValueError):
        st.advance_observed("fold")                  # nothing to call
    with pytest.raises(ValueError):
        st.advance_observed("raise")                 # no amount
    faced = st.advance_observed("raise", amount=450).state
    with pytest.raises(ValueError):
        faced.advance_observed("check")              # check facing a bet
    terminal = faced.advance_observed("fold").state
    assert terminal.terminal
    with pytest.raises(ValueError):
        terminal.advance_observed("check")


def test_observed_round_closure_matches_vocab_path():
    # BB bets (off-tree 500), UTG calls -> flop closes -> turn root with
    # REAL chips carried.
    st = _st_2p()
    st = st.advance_observed("raise", amount=500).state
    st = st.advance_observed("call").state
    assert st.street == 2
    assert st.pot == 600 + 500 + 500
    assert st.street_history == ()
    assert st.postflop_history == (1, 1)
    assert st.stacks == (9225, 9225)


# ── HybridSource: live where supported, lookup at the documented gap ──────


class _FakeLive(StrategySource):
    def __init__(self):
        self.calls = 0

    def get_mix(self, state, hero_cards):
        self.calls += 1
        if state.street == 3:
            raise LiveResolveUnsupported("river — side-pot gap")
        return ActionMix({0: 1.0})


class _FakeLookup(StrategySource):
    def __init__(self):
        self.calls = 0

    def get_mix(self, state, hero_cards):
        self.calls += 1
        return ActionMix({0: 0.5, 1: 0.5})


class _BrokenLive(StrategySource):
    def get_mix(self, state, hero_cards):
        raise ValueError("replay mismatch — must NOT be swallowed")


def test_hybrid_uses_live_on_flop_and_lookup_on_river():
    live, lookup = _FakeLive(), _FakeLookup()
    src = HybridSource(live, lookup)
    st = _st_2p()
    mix = src.get_mix(st, (_ci("Ah"), _ci("Kh")))
    assert mix.freqs == {0: 1.0}
    assert (live.calls, lookup.calls) == (1, 0)
    # Walk to the river: flop checks, turn checks.
    for idx in [0, 0, 0, 0]:
        st = st.advance(idx)
    assert st.street == 3
    mix_r = src.get_mix(st, (_ci("Ah"), _ci("Kh")))
    assert set(mix_r.freqs) == {0, 1}
    assert (live.calls, lookup.calls) == (2, 1)


def test_hybrid_does_not_swallow_real_errors():
    src = HybridSource(_BrokenLive(), _FakeLookup())
    with pytest.raises(ValueError):
        src.get_mix(_st_2p(), (_ci("Ah"), _ci("Kh")))
