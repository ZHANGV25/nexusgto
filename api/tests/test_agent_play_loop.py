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


def test_deep_raise_war_node_fails_loud_at_deferred_boundary(blueprint_path):
    # BOUNDARY (plan §6): a maxed flop raise war reaches a deep node where the
    # blueprint stores a REDUCED action set (v1 models one bet size + does not
    # track stack commitment, so its vocab can disagree there). v1 detects
    # this and FAILS LOUD (the exact-length invariant) rather than mis-play.
    # We do NOT claim a causal story (all-in/side-pot) — just that the
    # boundary is detected, not silently mis-played.
    src = LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path))
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s")],
        pot=720, stacks=[9660, 9780, 9780],
    )
    # Drive the raise war: BB chk, UTG chk, BTN bet, BB raise, UTG raise (cap),
    # BTN call, then BB call -> flop closes -> TURN root with reduced vocab.
    for idx in [0, 0, 1, 2, 2, 1, 1]:
        st = st.advance(idx)
    assert st.street == 2  # turn root after the warred flop closed
    with pytest.raises(ValueError, match="action-count mismatch"):
        src.get_mix(st, hero_cards=(_ci("Qh"), _ci("Jh")))


def test_live_resolve_mode_rejected_for_act_without_solver():
    # An agent in LIVE_RESOLVE without a primed solver should not silently
    # produce LOOKUP play; get_mix raises (NotImplementedError surfaces).
    from app.agent.strategy_source import LiveResolveSource
    agent = PokerAgent(src, ModeConfig(mode="LIVE_RESOLVE", seed=0))
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[], pot=440, stacks=[9780, 9780],
    )
    with pytest.raises(NotImplementedError):
        agent.act(st, (_ci("Qh"), _ci("Jh")))


    # SMOKE (plan §3.6 "+ smoke"): the LiveResolveSource adapter must actually
    # (new_hand / on_villain_action / on_hero_action / get_strategy). We do NOT
    # run a full GPU solve here (that path is GPU-gated and stays UNVERIFIED in
    # CI); this proves the adapter constructs the real solver, the part that
    # was otherwise only exercised in production.
    from app.agent.strategy_source import LiveResolveSource

    try:
    except Exception:  # pragma: no cover

    # blueprint_path=None -> solver-only construction (no 12GB BlueprintV2 load).
    src = LiveResolveSource(ModeConfig(mode="LIVE_RESOLVE", blueprint_path=None))
    solver = src._ensure_solver()
    assert solver is not None
    for api in ("new_hand", "on_villain_action", "on_hero_action", "get_strategy"):
    # Idempotent: second call returns the same instance.
    assert src._ensure_solver() is solver
