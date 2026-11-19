"""advance_observed + HybridSource — the off-tree / full-hand layer.

advance_observed applies an observed (possibly off-menu) opponent action:
actual chips into the state (pot odds, prunes, street-entry reconstruction),
the snapped on-tree index into the histories (blueprint keys + street replay).
HybridSource is the composition that lets a full hand finish: wherever the
live source refuses (LiveResolveUnsupported), the lookup acts.
"""

from __future__ import annotations

import pytest

from app.engine.agent.game_state import ObservedAdvance, new_postflop_state
from app.engine.agent.strategy_source import (
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
    # "call" with nothing outstanding is a check (some callers pass "call").
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


# ── HybridSource: live where supported, lookup where live refuses ─────────


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


# -- Regression tests: round closure, raise cap, pruned vocab --------------


def test_short_allin_call_closes_the_round():
    # A short all-in CALL must close the betting round (otherwise the all-in
    # player would forever "owe" the unmatched bet) -> no state desync.
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d"), _ci("7h")],
        pot=600, stacks=[9725, 300],
    )
    st = st.advance_observed("raise", amount=450).state  # BB bets 450
    obs = st.advance_observed("call")                    # UTG all-in for 300
    st2 = obs.state
    assert obs.applied_amount == 300
    assert st2.street == 2                # the round CLOSED
    assert st2.pot == 600 + 450 + 300
    assert st2.stacks == (9275, 0)


def test_allin_player_still_gets_forced_turn_action():
    # The trainer KEEPS all-in players in the rotation (forced check nodes
    # are trained blueprint keys) -- the round-closure rule must not drop them.
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d"), _ci("7h")],
        pot=600, stacks=[9725, 300],
    )
    st = st.advance_observed("raise", amount=450).state
    st = st.advance_observed("call").state               # UTG all-in, turn
    assert st.street == 2 and st.to_act_slot == 0
    st = st.advance(0)                                   # BB checks the turn
    assert st.to_act_slot == 1                           # all-in UTG: forced
    assert st.num_legal_actions() == 1                   # [check] only
    st = st.advance(0)
    assert st.street == 3                                # round closed


def test_observed_raise_at_capped_node_records_call_with_real_chips():
    # A real 4th raise at a raise-capped node must not be recorded as a
    # phantom raise index that exists in no trainer vocab. It must record
    # the CALL (the only on-tree continuation) with REAL chips, and NOT
    # advance raises_this_street past the cap.
    st = _st_2p()
    st = st.advance(1)   # BB bet 450        (raise 1)
    st = st.advance(2)   # UTG raise to 1950 (raise 2)
    st = st.advance(2)   # BB re-raise       (raise 3 = cap)
    assert st.raises_this_street == 3
    assert st.num_legal_actions() == 2       # [fold, call] -- capped
    pot0, cb0 = st.pot, st.current_bet
    obs = st.advance_observed("raise", amount=7000)  # real 4-bet
    assert obs.action_idx == 1               # recorded as CALL
    assert obs.off_tree is True
    assert obs.applied_amount == 7000        # REAL chips applied
    st2 = obs.state
    assert st2.pot == pot0 + 7000
    assert st2.current_bet > cb0             # reality: the bet DID grow
    assert st2.raises_this_street == 3       # vocab stays capped
    assert st2.postflop_history[-1] == 1     # on-tree history
    # The next player legally responds fold/call at the capped node.
    assert st2.num_legal_actions() == 2


def test_live_resolve_converts_replay_unsupported_to_fallback_type():
    # The engine signals known abstraction limits (tree raise cap, action-
    # count divergence) as ReplayUnsupported; LiveResolveSource must convert
    # it to LiveResolveUnsupported (the hybrid-fallback type) -- and ONLY it.
    from app.engine.agent.modes import ModeConfig
    from app.engine.agent.strategy_source import LiveResolveSource
    from app.engine.cpu_es_solver import ReplayUnsupported

    class _CapLimitedEngine:
        def solve(self, **_kw):
            raise ReplayUnsupported("tree raise cap below agent vocab")

    src = LiveResolveSource(
        ModeConfig(mode="LIVE_RESOLVE"), cpu_es_solver=_CapLimitedEngine(),
        seed_ranges=False,
    )
    with pytest.raises(LiveResolveUnsupported):
        src.get_mix(_st_2p(), (_ci("Ah"), _ci("Kh")))


def test_lookup_projects_stored_sigma_onto_real_pruned_vocab():
    # A covering all-in on the hero prunes the REAL vocab to
    # [fold, call] while the blueprint stored 3 actions (canonical chips).
    # The stored sigma is prefix-aligned (raise slots are the tail) -> the
    # lookup must project + renormalize, not crash.
    from app.engine.agent.modes import ModeConfig
    from app.engine.agent.strategy_source import LookupSource

    class _FakeReader:
        def get_metadata(self):
            return {}

        def hand_to_bucket(self, board, c0, c1):
            return 7

        def get_strategy(self, board, ah, player, bucket, street=None):
            return [0.2, 0.5, 0.3]   # fold / call / raise (canonical node)

    st = _st_2p().advance_observed("allin").state  # BB jams 9725 (snapped idx 1)
    assert st.facing_bet and st.num_legal_actions() == 2  # raise REAL-pruned
    src = LookupSource(ModeConfig(mode="LOOKUP"), reader=_FakeReader())
    mix = src.get_mix(st, (_ci("Ah"), _ci("Kh")))
    assert set(mix.freqs) == {0, 1}
    assert abs(mix.freqs[0] - 0.2 / 0.7) < 1e-9
    assert abs(mix.freqs[1] - 0.5 / 0.7) < 1e-9
    assert mix.is_uniform_fallback is False
