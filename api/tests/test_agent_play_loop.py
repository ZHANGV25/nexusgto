"""TDD: PokerAgent LOOKUP play loop (plan §3.5, §4d) — BEHAVIORAL.

These are behavioral ("plays whole hands correctly"), not just unit:
  - plays a full 2p hand and a full 3p hand to a terminal state, every
    decision sampled from a blueprint HIT (no uniform-fallback on trained
    lines);
  - sampled action frequencies ≈ the σ̄ mix over many samples (§4d);
  - sampling is deterministic under a fixed seed;
  - LOOKUP never imports/constructs the GPU solver.

A fake source drives the seed/determinism/loop-structure tests without the
blueprint; the full-hand HIT tests use the real 8.05B (skip if absent).
"""

from __future__ import annotations

import pytest

from app.agent.agent import PokerAgent, SampledAction
from app.agent.game_state import new_postflop_state
from app.agent.modes import ModeConfig
from app.agent.strategy_source import ActionMix, LookupSource, StrategySource


def _ci(card: str) -> int:
    ranks = "AKQJT98765432"
    so = ("c", "d", "h", "s")
    return (12 - ranks.index(card[0].upper())) * 4 + so.index(card[1].lower())


# ── Fake source: structural / determinism tests (no blueprint) ───────────


class _FakeSource(StrategySource):
    def __init__(self, mix: ActionMix):
        self._mix = mix
        self.calls = 0

    def get_mix(self, state, hero_cards):
        self.calls += 1
        return self._mix


def test_act_returns_sampled_action_within_legal_set():
    src = _FakeSource(ActionMix({0: 0.3, 1: 0.7}))
    agent = PokerAgent(source=src, config=ModeConfig(mode="LOOKUP", seed=7))
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td")], pot=440, stacks=[9780, 9780],
    )
    res = agent.act(st, hero_cards=(_ci("Qh"), _ci("Jh")))
    assert isinstance(res, SampledAction)
    assert res.action_idx in (0, 1)
    assert res.mix is src._mix


def test_seed_makes_sampling_deterministic():
    mix = ActionMix({0: 0.5, 1: 0.5})
    a1 = PokerAgent(_FakeSource(mix), ModeConfig(mode="LOOKUP", seed=123))
    a2 = PokerAgent(_FakeSource(mix), ModeConfig(mode="LOOKUP", seed=123))
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[], pot=440, stacks=[9780, 9780],
    )
    seq1 = [a1.act(st, (0, 1)).action_idx for _ in range(50)]
    seq2 = [a2.act(st, (0, 1)).action_idx for _ in range(50)]
    assert seq1 == seq2


def test_sampled_frequencies_track_the_mix():
    # §4d: over many samples, action freq ≈ the σ̄ mix.
    mix = ActionMix({0: 0.2, 1: 0.8})
    agent = PokerAgent(_FakeSource(mix), ModeConfig(mode="LOOKUP", seed=0))
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[], pot=440, stacks=[9780, 9780],
    )
    n = 5000
    samples = [agent.act(st, (0, 1)).action_idx for _ in range(n)]
    frac1 = sum(1 for s in samples if s == 1) / n
    assert abs(frac1 - 0.8) < 0.03


# ── Full-hand play (blueprint oracle) ─────────────────────────────────────
#
# We drive the agent along SCRIPTED realistic lines (check-down + single-bet)
# that stay on v1's modeled surface (one bet size, no raise war → no all-in /
# stack-exhaustion). Free self-play with a fixed seed can wander into a maxed
# raise war that reaches a reduced-vocab all-in-ish node; that boundary is
# DEFERRED (plan §6) and tested separately as a loud-fail below. At each
# scripted decision the agent still SAMPLES from the blueprint mix — the
# script only bounds the OPPONENTS' line to the modeled surface.


def _play_scripted(agent, st, hero, opp_when_forced):
    """Walk a hand: the agent samples for HERO's seat; for other seats we take
    a scripted, modeled-surface action. Returns (hero_actions, final_state).

    opp_when_forced: action_idx to use for non-hero seats (0 = check/fold-
    avoiding-... here always 0 = check, which keeps the line on the
    check-down / single-bet modeled surface).
    """
    hero_actions = []
    guard = 0
    hero_seat = None
    # hero is the BB (seat 1, slot 0) in these lines.
    while not st.terminal and guard < 40:
        guard += 1
        seat = st.to_act_seat()
        if hero_seat is None and seat == 1:
            hero_seat = 1
        if seat == 1:  # hero (BB)
            res = agent.act(st, hero)
            hero_actions.append(res)
            st = st.advance(res.action_idx)
        else:  # opponent: scripted modeled-surface action (check)
            n = st.num_legal_actions()
            idx = opp_when_forced if opp_when_forced < n else 0
            st = st.advance(idx)
    return hero_actions, st


def test_plays_full_2p_hand_all_hits(blueprint_path):
    agent = PokerAgent(
        LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path)),
        ModeConfig(mode="LOOKUP", seed=1),
    )
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s")],
        pot=440, stacks=[9780, 9780],
    )
    actions, final = _play_scripted(agent, st, (_ci("Qh"), _ci("Jh")),
                                    opp_when_forced=0)
    assert len(actions) >= 2  # hero acts on flop, turn, river (check-down)
    assert all(not a.mix.is_uniform_fallback for a in actions)
    assert final.terminal


def test_plays_full_3p_hand_all_hits(blueprint_path):
    agent = PokerAgent(
        LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path)),
        ModeConfig(mode="LOOKUP", seed=2),
    )
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s")],
        pot=720, stacks=[9660, 9780, 9780],
    )
    actions, final = _play_scripted(agent, st, (_ci("Qh"), _ci("Jh")),
                                    opp_when_forced=0)
    assert len(actions) >= 3
    assert all(not a.mix.is_uniform_fallback for a in actions)
    assert final.terminal


def test_3p_hand_reaches_terminal_state(blueprint_path):
    agent = PokerAgent(
        LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path)),
        ModeConfig(mode="LOOKUP", seed=5),
    )
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s")],
        pot=720, stacks=[9660, 9780, 9780],
    )
    _, final = _play_scripted(agent, st, (_ci("Qh"), _ci("Jh")),
                              opp_when_forced=0)
    assert final.terminal


def test_lookup_play_loop_never_touches_gpu(blueprint_path):
    import sys
    sys.modules.pop("street_solver_gpu", None)
    agent = PokerAgent(
        LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path)),
        ModeConfig(mode="LOOKUP", seed=3),
    )
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s")],
        pot=720, stacks=[9660, 9780, 9780],
    )
    _play_scripted(agent, st, (_ci("Qh"), _ci("Jh")), opp_when_forced=0)
    assert "street_solver_gpu" not in sys.modules


def test_deep_raise_war_node_now_serves_correct_length_mix(blueprint_path):
    # 3a FIX (was test_deep_raise_war_node_fails_loud_at_deferred_boundary):
    # the SAME maxed flop raise war that used to CRASH (ValueError: action-count
    # mismatch) now plays cleanly. GameState tracks stack commitment, so its
    # vocab prunes exactly like the trainer's generate_actions and num_legal
    # matches the blueprint's stored σ̄ length at the committed turn root.
    #
    # CRITICAL: chips are DERIVED canonically (no fixture pot/stacks) — the old
    # fixture 720/9660 was chip-inconsistent and mis-fired the prune. With the
    # verified convention this node serves a correct-length mix, not a crash.
    src = LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path))
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s")],
    )  # pot/stacks derived: pot=875, stacks 9725 each (verified convention)
    # Drive the raise war: BB chk, UTG chk, BTN bet, BB raise, UTG raise (cap),
    # BTN call, then BB call -> flop closes -> TURN root (formerly the crash).
    for idx in [0, 0, 1, 2, 2, 1, 1]:
        st = st.advance(idx)
    assert st.street == 2  # turn root after the warred flop closed
    # No crash: the mix is served with the blueprint's exact length.
    mix = src.get_mix(st, hero_cards=(_ci("Qh"), _ci("Jh")))
    assert len(mix.freqs) == st.num_legal_actions()
    # The committed all-in turn root: blueprint stored a single forced action
    # ([1.0]); the chip-aware vocab agrees (1 legal action), so it is a HIT,
    # not a uniform fallback.
    assert st.num_legal_actions() == 1
    assert mix.is_uniform_fallback is False


def test_live_resolve_requires_engine_or_blueprint():
    # LIVE_RESOLVE must fail LOUD (not silently fall back to LOOKUP) when it has
    # neither an injected CpuEsSolver nor a blueprint_path to build one.
    from app.agent.strategy_source import LiveResolveSource
    src = LiveResolveSource(ModeConfig(mode="LIVE_RESOLVE", blueprint_path=None))
    with pytest.raises(ValueError):
        src._ensure_engine()


def test_live_resolve_river_without_dealt_cards_fails_loud():
    # SMOKE: river is SERVED since 2026-06-11, but a river state whose board
    # was never extended past the flop must fail LOUD (ValueError at the
    # board-length guard) BEFORE constructing/calling any engine — never a
    # silent wrong-street solve and never a lookup-fallback (this is a driver
    # bug, not a documented unsupported surface). A fake engine that would
    # explode if called proves get_mix never reaches it.
    from app.agent.strategy_source import LiveResolveSource

    class _ExplodingEngine:
        def solve(self, **_kw):  # pragma: no cover - must never be called
            raise AssertionError("engine.solve called on a short-board river")

    src = LiveResolveSource(
        ModeConfig(mode="LIVE_RESOLVE"), cpu_es_solver=_ExplodingEngine(),
    )
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    for idx in [0, 0, 0, 0, 0, 0]:  # flop + turn check-throughs -> river root
        st = st.advance(idx)
    assert st.street == 3
    with pytest.raises(ValueError, match="board cards"):
        src.get_mix(st, (_ci("Ah"), _ci("Kh")))


# -- 6p LIVE_RESOLVE play along a real line (Lane A validation, 2026-06-10) --
#
# Drives PokerAgent through a REAL 6p flop line with the vec-Linear-FINAL
# re-solve as the strategy source: scripted opponents bring the action to the
# hero (BTN) FACING A BET; the agent samples its decision from a per-decision
# 6p re-solve (street-entry reconstruction + in-tree replay of this street's
# actions), the sampled action advances the state, and the scripted line
# closes the street. This is play-validation of the live game loop, not an
# exploitability number.


def test_6p_live_resolve_plays_faced_bet_decision(blueprint_path):
    pytest.importorskip("numpy")
    from app.agent.strategy_source import LiveResolveSource

    try:
        from app.services.cpu_es_solver import CpuEsSolver
        engine = CpuEsSolver(blueprint_path)
    except Exception as e:
        pytest.skip(f"CpuEsSolver unavailable: {e}")
    if not engine.ready:
        pytest.skip("CpuEsSolver not ready")

    src = LiveResolveSource(
        ModeConfig(mode="LIVE_RESOLVE", blueprint_path=blueprint_path),
        cpu_es_solver=engine, iterations=2000,
        vec_mode=True, readout="final", num_threads=12,
    )
    agent = PokerAgent(source=src, config=ModeConfig(mode="LIVE_RESOLVE", seed=11))

    st = new_postflop_state(
        seats=["SB", "BB", "UTG", "MP", "CO", "BTN"],
        preflop_prefix=[2, 1, 1, 1, 1, 1],
        board=[_ci("Ah"), _ci("Kh"), _ci("2c")],
    )
    # Scripted line to the hero: SB chk, BB chk, UTG bets, MP folds, CO calls.
    for idx in [0, 0, 1, 0, 1]:
        st = st.advance(idx)
    assert st.to_act_slot == 5  # BTN = hero, facing UTG's bet
    n_legal = st.num_legal_actions()
    assert n_legal == 3  # fold / call / raise

    res = agent.act(st, hero_cards=(_ci("Ad"), _ci("Kd")))  # AKdd on AhKh2c
    assert isinstance(res, SampledAction)
    assert 0 <= res.action_idx < n_legal
    assert res.mix.is_uniform_fallback is False  # a real re-solve, not a miss

    # The sampled action ADVANCES the real game state (the loop closes).
    st2 = st.advance(res.action_idx)
    assert st2.postflop_history == st.postflop_history + (res.action_idx,)
    # SB and BB checked BEFORE the bet, so they still owe action whatever the
    # hero did -- forward-from-actor order wraps to SB next in all 3 cases.
    assert st2.street == 1 and not st2.terminal
    assert st2.to_act_slot == 0  # SB owes action against the outstanding bet
    # Script the hand to street close: SB folds, BB folds; any remaining
    # callers (only if the hero raised) flat-call.
    st3 = st2.advance(0).advance(0)
    guard = 0
    while st3.street == 1 and not st3.terminal and guard < 6:
        st3 = st3.advance(1)
        guard += 1
    assert st3.terminal or st3.street == 2  # flop closed -> turn (or hand over)
    print(f"\n6p play-loop decision: mix={res.mix.freqs} sampled={res.action_idx}")


# -- FULL 6p hand: off-tree opponent bet + hybrid live/lookup (2026-06-10) --
#
# The closest-to-real-play test in the suite: a 6p hand played to TERMINAL
# where an opponent bets an OFF-TREE size (applied with real chips via
# advance_observed, snapped on-tree in the histories), the hero decides via
# HybridSource -- vec-Linear-final live re-solves on flop/turn, blueprint
# LOOKUP at river (the side-pot gap) -- and every decision advances the
# real state. Asserts the hand completes and every hero mix is legal.


def test_6p_full_hand_off_tree_bet_hybrid(blueprint_path):
    pytest.importorskip("numpy")
    from app.agent.strategy_source import (
        HybridSource, LiveResolveSource, LookupSource,
    )

    try:
        from app.services.cpu_es_solver import CpuEsSolver
        engine = CpuEsSolver(blueprint_path)
    except Exception as e:
        pytest.skip(f"CpuEsSolver unavailable: {e}")
    if not engine.ready:
        pytest.skip("CpuEsSolver not ready")

    live = LiveResolveSource(
        ModeConfig(mode="LIVE_RESOLVE", blueprint_path=blueprint_path),
        cpu_es_solver=engine, iterations=2000,
        vec_mode=True, readout="final", num_threads=12,
    )
    lookup = LookupSource(
        ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path))
    agent = PokerAgent(source=HybridSource(live, lookup),
                       config=ModeConfig(mode="LIVE_RESOLVE", seed=17))

    hero = (_ci("Ad"), _ci("Kd"))
    st = new_postflop_state(
        seats=["SB", "BB", "UTG", "MP", "CO", "BTN"],
        preflop_prefix=[2, 1, 1, 1, 1, 1],
        board=[_ci("Ah"), _ci("Kh"), _ci("2c"), _ci("Qc"), _ci("2s")],
    )  # canonical 6p single-open entry: pot 1650, all six live; hero = BTN
    entry_pot = st.pot
    assert entry_pot == 1650  # 6×275 open/calls incl. blinds (derived)

    hero_mixes = []

    def hero_act():
        nonlocal st
        res = agent.act(st, hero)
        hero_mixes.append(res)
        st = st.advance(res.action_idx)
        return res

    # FLOP: SB chk, BB chk, UTG bets OFF-TREE 600 into 875 (canonical 656),
    # MP folds, CO calls (real chips). Then the hero faces the bet.
    st = st.advance_observed("check").state
    st = st.advance_observed("check").state
    obs = st.advance_observed("raise", amount=600)
    assert obs.off_tree is True and obs.action_idx == 1
    st = obs.state
    st = st.advance_observed("fold").state
    st = st.advance_observed("call").state
    assert st.to_act_slot == 5 and st.facing_bet
    assert st.pot == entry_pot + 600 + 600            # REAL chips in the pot
    res = hero_act()
    # AdKd = top two on AhKh2c: a sane final-iterate never folds here. This
    # doubles as a play-quality canary on the off-tree faced-bet node.
    assert res.action_idx in (1, 2), f"hero folded top-two?! mix={res.mix.freqs}"
    # Close the flop: SB/BB fold; any remaining callers flat-call.
    st = st.advance_observed("fold").state
    st = st.advance_observed("fold").state
    guard = 0
    while st.street == 1 and not st.terminal and guard < 4:
        st = st.advance_observed("call").state
        guard += 1
    assert st.street == 2 and not st.terminal

    # TURN (UTG, CO, hero left): checks to the hero, hero decides (live).
    while st.to_act_slot != 5 and not st.terminal and st.street == 2:
        st = st.advance_observed("check").state
    if st.street == 2 and not st.terminal:
        hero_act()
        guard = 0
        while st.street == 2 and not st.terminal and guard < 4:
            st = st.advance_observed("call" if st.facing_bet else "check").state
            guard += 1
    assert st.terminal or st.street == 3

    # RIVER: checks to the hero; the hero''s mix comes from LOOKUP via the
    # hybrid fallback (live raises LiveResolveUnsupported on river).
    while not st.terminal and st.to_act_slot != 5 and st.street == 3:
        st = st.advance_observed("check").state
    if not st.terminal and st.street == 3:
        hero_act()
        guard = 0
        while not st.terminal and guard < 4:
            st = st.advance_observed("call" if st.facing_bet else "check").state
            guard += 1

    assert st.terminal or st.betting_round_closed_count >= 3
    assert len(hero_mixes) >= 2                       # flop + turn at minimum
    for r in hero_mixes:
        assert abs(sum(r.mix.freqs.values()) - 1.0) < 1e-9
    # Flop + turn decisions are LIVE re-solves -- never the uniform-miss
    # placeholder (river lookup MAY be a flagged uniform fallback: 6p river
    # blueprint coverage is sparse, and that honesty flag is the contract).
    assert hero_mixes[0].mix.is_uniform_fallback is False
    assert hero_mixes[1].mix.is_uniform_fallback is False
    print(f"\n6p full hand: {len(hero_mixes)} hero decisions, "
          f"mixes={[r.mix.freqs for r in hero_mixes]}, terminal={st.terminal}")
