"""TDD anchor + verify for LiveResolveSource (LIVE_RESOLVE ceiling, plan §3.6).

LIVE_RESOLVE is the non-real-time ceiling: a per-decision CPU re-solve. This
file is the §3a-style anchor — on a known 2p flop spot AND a 3p turn spot, the
resolved strategy must be:
  - non-degenerate (a real mix or a justified pure action, not empty/garbage),
  - legal-length (len(mix) == GameState.num_legal_actions()),
  - mapped back to the agent's action-index vocab (Check/Bet 75% -> indices).
and the per-decision latency is MEASURED (expected seconds, NOT sub-second).

EMPIRICAL BASIS (recon this session, docs/BENCHMARK_RESULTS.md §3c): the wired
own CPU fallback (solver_v2) crashes on multi-street trees no-GPU; CpuEsSolver
builds real blueprint-grounded leaf values (2p-flop table 99.9% nonzero, 3p-turn
98.6% nonzero, 0 coverage substitutions). RIVER (2026-06-11) is served via
exact showdown terminals (side-pot validated) — no leaf table at all.

These tests load a ~12GB blueprint through es_resolver_real.dll (session-scoped,
once) and SKIP cleanly if it's absent — like the LOOKUP oracle tests.
"""

from __future__ import annotations

import time

import pytest

from app.agent.game_state import new_postflop_state
from app.agent.modes import ModeConfig


def _ci(card: str) -> int:
    ranks = "AKQJT98765432"
    so = ("c", "d", "h", "s")
    return (12 - ranks.index(card[0].upper())) * 4 + so.index(card[1].lower())


# ── Session-scoped CpuEsSolver (the heavy 12GB load happens ONCE) ─────────


@pytest.fixture(scope="session")
def cpu_es_engine(blueprint_path):
    """Construct the CpuEsSolver once for the whole test session.

    Skips if the engine can't be built (no es_resolver_real.dll / blueprint).
    """
    try:
        from app.services.cpu_es_solver import CpuEsSolver
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CpuEsSolver not importable: {e}")
    try:
        engine = CpuEsSolver(blueprint_path)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CpuEsSolver construct failed (no GPU-free ES?): {e}")
    if not engine.ready:  # pragma: no cover
        pytest.skip("CpuEsSolver not ready")
    return engine


@pytest.fixture(scope="session")
def live_source(cpu_es_engine, blueprint_path):
    from app.agent.strategy_source import LiveResolveSource
    cfg = ModeConfig(mode="LIVE_RESOLVE", blueprint_path=blueprint_path, seed=0)
    # A low iteration count keeps the anchor fast; quality is verified by the
    # EV-beats-floor test, not by iteration count here.
    return LiveResolveSource(cfg, cpu_es_solver=cpu_es_engine, iterations=200)


# ── ANCHOR: 2p flop ──────────────────────────────────────────────────────


def _is_uniform(freqs, n_legal, eps=0.02):
    """True if the mix is within eps of uniform-over-legal (the placeholder a
    near-zero-weight / unresolved row collapses to). The anchor must REJECT this
    so the test distinguishes a real solve from a no-op (advisor 2026-06-06)."""
    u = 1.0 / n_legal
    return all(abs(freqs.get(i, 0.0) - u) <= eps for i in range(n_legal))


def test_live_resolve_2p_flop_gated_to_lookup(live_source):
    """2p flop (UTG raise, BB call -> flop): the live re-solve is GATED
    (2026-06-14, supersedes the old 'anchor' that asserted a live mix).

    A 2p flop/turn tree extends through an in-tree CHANCE node (the OP3
    turn-extension; ss_tree_builder.c should_extend_via_chance ->
    orig_num_players==2). The CPU-ES traversal handles chance nodes only with a
    defensive stub (es_resolver.c:554) with no reach/strat-sum accounting, so
    the resolved sigma-bar is DEGENERATE — pure (0/1) per-hand strategies
    uncorrelated with hand strength (EC2-measured 2026-06-14: pure-checks two
    pair, pure-bets air). NOTE the OLD anchor's 'not uniform' assertion would
    have PASSED on that degenerate output (a pure strategy is not uniform) —
    which is exactly why it gave false confidence. CpuEsSolver.solve now raises
    ReplayUnsupported on num_chance_nodes>0, and LiveResolveSource converts that
    to LiveResolveUnsupported -> the hybrid serves the (correct) blueprint
    lookup. Pin that contract here; 3p+ (no chance node) still resolves live."""
    from app.agent.live_resolve_adapter import LiveResolveUnsupported

    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )  # pot/stacks derived canonically (2p: 600 / 9725 each)
    assert st.num_legal_actions() == 2  # flop root, no bet faced: [check, bet]

    with pytest.raises(LiveResolveUnsupported) as ei:
        live_source.get_mix(st, hero_cards=(_ci("9h"), _ci("9c")))
    assert "chance node" in str(ei.value).lower()


# ── ANCHOR: 3p turn ──────────────────────────────────────────────────────


def test_live_resolve_3p_turn_anchor(cpu_es_engine, blueprint_path):
    """3p turn (UTG raise, BTN call, BB call -> flop checks through -> turn):
    hero=BB resolves a legal-length, real (non-fallback) mix (STRUCTURAL anchor;
    3p has NO 2p-only EV claim — ev_eval has no side pots). Hero = 99 (in BB's
    flat-call range).

    NOTE on 3p variance: the CPU-ES solve is multi-thread Hogwild — at low iters
    the 3p-turn solve for a near-mixed hand has real run-to-run variance (a hand
    can land ~0.98/0.02 one run and ~0.55/0.45 another). So we do NOT assert
    strict non-uniformity here (that is a flaky claim for a genuinely
    close-to-mixed 3p spot); the unconfounded structural facts — runs GPU-free,
    legal-length, NOT the uniform-on-miss fallback, sums to 1, measured latency —
    are the 3p headline. Strict non-uniformity is asserted on the lower-variance
    2p anchor. We bump iters to reduce (not eliminate) the variance."""
    from app.agent.strategy_source import LiveResolveSource
    src = LiveResolveSource(
        ModeConfig(mode="LIVE_RESOLVE", blueprint_path=blueprint_path, seed=0),
        cpu_es_solver=cpu_es_engine, iterations=600,
    )
    # NOTE (review 2026-06-10): the board must carry the TURN card — with a
    # 3-card board this anchor was silently solving a FLOP (the exact
    # wrong-street bug the new board-length guard now rejects).
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d"), _ci("7h")],
    )
    # Walk the flop check-through to the turn root: BB chk, UTG chk, BTN chk.
    for idx in [0, 0, 0]:
        st = st.advance(idx)
    assert st.street == 2  # turn root
    n_legal = st.num_legal_actions()
    assert n_legal == 2  # turn root, no bet faced

    t0 = time.time()
    mix = src.get_mix(st, hero_cards=(_ci("9h"), _ci("9c")))
    dt = time.time() - t0

    assert set(mix.freqs.keys()) <= set(range(n_legal))
    assert len(mix.freqs) >= 1
    assert abs(sum(mix.freqs.values()) - 1.0) < 1e-9
    assert mix.is_uniform_fallback is False  # a REAL solve, not the miss fallback
    assert dt < 120.0
    print(f"\n3p-turn live-resolve: dt={dt:.2f}s mix={mix.freqs}")


# ── Check-through: hero NOT first-to-act extracts the RIGHT (no-bet) node ─


def test_live_resolve_check_through_2p_flop_gated(live_source):
    """A 2p flop check-through node (BB checks -> UTG to act, no bet faced).
    Like the 2p flop anchor, this tree extends via an in-tree chance node, so it
    is now GATED (2026-06-14): CpuEsSolver.solve raises ReplayUnsupported ->
    LiveResolveUnsupported -> the hybrid serves the blueprint lookup.

    (The node-extraction wiring this used to assert — that the all-check-line
    node, not a faced-bet branch, is extracted — is unit-covered by the replay
    tests below on the chance-free river path; for 2p flop the engine no longer
    runs, so there is no extracted node to check.)"""
    from app.agent.live_resolve_adapter import LiveResolveUnsupported

    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d")],
    )
    st = st.advance(0)  # BB checks -> UTG (the raiser) to act, no bet faced
    assert st.to_act_slot == 1 and st.facing_bet is False
    with pytest.raises(LiveResolveUnsupported):
        live_source.get_mix(st, hero_cards=(_ci("Ah"), _ci("Qh")))


# ── RIVER is SERVED (2026-06-11): exact showdown terminals, no leaf table ─


def test_live_resolve_3p_river_serves_mix(live_source):
    """3p RIVER — the old §6 side-pot gap — is now served: the river tree
    terminates in showdown/fold terminals settled natively by the C ES
    (es_showdown_value, side-pot validated incl. the nh>1 vec gate), and
    CpuEsSolver bypasses the leaf-table construct fail-loud. The mix must be
    a legal-shape REAL solve (structural anchor, like the 3p turn anchor)."""
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d"), _ci("7h"), _ci("3c")],
    )
    # Walk flop check-through -> turn, turn check-through -> river root.
    for idx in [0, 0, 0, 0, 0, 0]:
        st = st.advance(idx)
    assert st.street == 3  # river root
    n_legal = st.num_legal_actions()
    assert n_legal == 2  # river root, no bet faced

    t0 = time.time()
    mix = live_source.get_mix(st, hero_cards=(_ci("9h"), _ci("9c")))
    dt = time.time() - t0

    assert set(mix.freqs.keys()) <= set(range(n_legal))
    assert len(mix.freqs) >= 1
    assert abs(sum(mix.freqs.values()) - 1.0) < 1e-9
    assert mix.is_uniform_fallback is False  # a REAL solve, not the miss fallback
    # River skips the leaf-table construct -> should be FASTER than flop/turn.
    assert dt < 120.0
    print(f"\n3p-river live-resolve: dt={dt:.2f}s mix={mix.freqs}")


def test_live_resolve_2p_river_faced_bet_replay(live_source):
    """2p river FACED BET: BB bets the river, UTG faces it — the street
    replay walks the bet inside the solved RIVER tree to UTG's 3-action
    {fold, call, raise} node. Proves hero_replay works on the new street."""
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Qs"), _ci("Js"), _ci("2d"), _ci("7h"), _ci("3c")],
    )
    # flop check-check -> turn check-check -> river: BB bets (idx 1).
    for idx in [0, 0, 0, 0, 1]:
        st = st.advance(idx)
    assert st.street == 3
    assert st.to_act_slot == 1  # UTG
    assert st.facing_bet is True
    n_legal = st.num_legal_actions()
    assert n_legal == 3  # [fold, call, raise]

    mix = live_source.get_mix(st, hero_cards=(_ci("Ah"), _ci("Qh")))
    assert set(mix.freqs.keys()) <= set(range(n_legal))
    assert abs(sum(mix.freqs.values()) - 1.0) < 1e-9
    assert mix.is_uniform_fallback is False
    print(f"\n2p-river faced-bet live-resolve: mix={mix.freqs}")


# -- 6p vec-Linear-FINAL: the playing-agent re-solve path (Lane A, 2026-06-10) --
#
# These exercise the NEW 6p path end-to-end through the live loop surface:
# GameState -> build_solve_inputs (street-entry reconstruction + street
# replay actions) -> CpuEsSolver.solve(vec_mode=1, readout="final",
# hero_replay) -> label->index mix. vec-Linear-FINAL is the Gate-1-validated
# EC2-measured config (6p ~0.15%/pot floor in ~22s solve @96 cores --
# measured_logs/VEC_SPEEDUP_2026-06-10.md). Iteration counts here are SMALL:
# these validate the PATH (correct node, legal mix), not solve quality.


@pytest.fixture(scope="session")
def live_source_6p(cpu_es_engine, blueprint_path):
    from app.agent.strategy_source import LiveResolveSource
    cfg = ModeConfig(mode="LIVE_RESOLVE", blueprint_path=blueprint_path, seed=0)
    return LiveResolveSource(
        cfg, cpu_es_solver=cpu_es_engine, iterations=2000,
        vec_mode=True, readout="final", num_threads=12,
    )


def _state_6p_flop():
    # 6p single-open line: UTG opens 0.7x, everyone (incl. blinds) flat-calls.
    # Canonical chips derived (single open + calls). Flop order is SB-first.
    return new_postflop_state(
        seats=["SB", "BB", "UTG", "MP", "CO", "BTN"],
        preflop_prefix=[2, 1, 1, 1, 1, 1],
        board=[_ci("Ah"), _ci("Kh"), _ci("2c")],
    )


def test_live_resolve_6p_checkline_vec_final(live_source_6p):
    """6p flop, checked to CO (4 checks replayed): the hero's node is reached
    by street replay; the mix is over the no-bet vocab and is a REAL solve."""
    st = _state_6p_flop()
    for idx in [0, 0, 0, 0]:  # SB, BB, UTG, MP check
        st = st.advance(idx)
    assert st.to_act_slot == 4  # CO
    assert st.facing_bet is False
    n_legal = st.num_legal_actions()
    assert n_legal == 2  # [check, bet 0.75]

    t0 = time.time()
    mix = live_source_6p.get_mix(st, hero_cards=(_ci("Qh"), _ci("Jh")))
    dt = time.time() - t0

    assert set(mix.freqs.keys()) <= set(range(n_legal))
    assert abs(sum(mix.freqs.values()) - 1.0) < 1e-9
    assert mix.is_uniform_fallback is False
    print(f"\n6p check-line vec-final: dt={dt:.1f}s mix={mix.freqs}")


def test_live_resolve_6p_faced_bet_vec_final(live_source_6p):
    """6p flop FACED BET: SB/BB check, UTG bets, MP folds, CO calls, BTN to
    act facing the bet. The replay walks check/check/bet/fold/call inside the
    solved tree to BTN's faced-bet node -- the mix must be over the 3-action
    {fold, call, raise} vocab (the node the old guard refused to serve)."""
    st = _state_6p_flop()
    for idx in [0, 0, 1, 0, 1]:  # SB chk, BB chk, UTG bet, MP fold, CO call
        st = st.advance(idx)
    assert st.to_act_slot == 5  # BTN
    assert st.facing_bet is True
    n_legal = st.num_legal_actions()
    assert n_legal == 3  # [fold, call, raise]

    t0 = time.time()
    mix = live_source_6p.get_mix(st, hero_cards=(_ci("Ad"), _ci("Kd")))
    dt = time.time() - t0

    assert set(mix.freqs.keys()) <= set(range(n_legal))
    assert abs(sum(mix.freqs.values()) - 1.0) < 1e-9
    assert mix.is_uniform_fallback is False
    print(f"\n6p faced-bet vec-final: dt={dt:.1f}s mix={mix.freqs}")
