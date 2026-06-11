"""Side-pot showdown validation gate — es_resolver.c multiway layering.

Validates the C ES re-solver's SHOWDOWN terminal (both the scalar
es_showdown_value path and the vectorized traverser's inline showdown block)
against an INDEPENDENT exact Python side-pot reference, per the bit-exact-gate
pattern of docs/CPU_ES_RESOLVER_BENCHMARK.md (nexusgto-api).

Four sub-gates:
  1. UNITS    — hand-crafted cases (3p-6p all-in-for-less chains, layer ties,
                dead money, folder-excess convention, 2p regression, check-down)
                with LITERAL hand-computed expecteds. Validates the reference
                itself first, then C-scalar and C-vec against the literals.
  2. FUZZ     — SIDEPOT_CASES random cases (default 100,000) across 2-6p:
                random street bets incl. capped all-in-for-less chains, fold
                patterns, folder-excess, builder pot-accounting slack, forced
                board-plays ties. Asserts |C − ref| < 1e-9 for EVERY player as
                traverser, scalar AND vec, plus the reference's own zero-sum
                invariant (Σ awards == pot).
  3. REGRESS  — on every degenerate case (all active street bets equal, no
                folded bet above — the ENTIRE currently-reachable node set,
                see the scope note above es_layered_award in es_resolver.c),
                C must equal the OLD single-pot closed form BIT-EXACTLY
                (pot/nwin − my_bet, IEEE-identical ops).
  4. VEC-NH   — nh>1 vectorized gate (added 2026-06-11). Gates 1-3 drive the
                vec path through es_test_showdown_value_vec, which builds its
                context with num_hands[p] = 1 for EVERY player
                (es_resolver.c es_test_build_ctx) and dead[1]/out_vals[1] —
                so the per-hand strength loop inside es_traverse_vec's inline
                showdown block (the river-critical path: per-hand eval7,
                opp_strength reuse across hands, per-hand single-pot vs
                layered routing, dead-hand skip) was NEVER exercised with
                nh > 1. This gate drives the REAL es_create/es_set_vec_mode/
                es_run/es_final_strategy surface on a 4-node tree (decision
                node for the traverser with 3 showdown children) with the
                traverser holding up to ~20 hands (overlapping each other,
                board-colliding, weight-0) over random side-pot cases, and
                recovers each hand's showdown value EXACTLY from the
                single-iteration vanilla-CFR regret-matched readout (closed
                form below), comparing per hand against the same exact
                Python reference. Opponent-card-colliding LIVE traverser
                hands are excluded by construction (they could block the
                opponent's deterministic single-hand sample and pollute the
                one-shot regrets); that dead-mask branch is covered by
                Gate-1 vec validation on overlapping ranges
                (es_resolver.c:95-97) and by weight-0 overlap hands here.

Reference formulation: iterative pot-STRIPPING (min-remaining-contribution
rounds) — deliberately a DIFFERENT algorithm from the C's level-interval loop,
computing the same function. Conventions shared with es_layered_award:
  - contributions capped at the top ACTIVE street-bet level;
  - base layer = pot − Σ capped contributions (entry pot + folder excess +
    builder accounting slack), awarded with the bottom layer;
  - layer eligibility = active players with street bet ≥ the layer's top;
  - ties chop by double division.

Run:  PYTHONUTF8=1 [ES_DLL=es_resolver] [SIDEPOT_CASES=100000] \
      [VECNH_CASES=2000] python -m python.es_resolver.validate_sidepot
Exit: 0 on PASS, 1 on any failure (fail-loud, prints the offending case).
"""
from __future__ import annotations

import ctypes
import math
import os
import random
import sys
import time
from typing import List, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(_HERE))   # python/
sys.path.insert(0, _HERE)

from leaf_values import _eval7_py  # noqa: E402  (validated vs C eval7)

# Canonical SSNode/SSTreeData ctypes mirrors (must match street_solve.cuh) —
# importing street_solver_gpu is DLL-free at module level (its CDLL loads are
# lazy inside StreetSolverGPU.__init__ / _get_lib).
from street_solver_gpu import (  # noqa: E402
    SS_MAX_HANDS,
    SS_MAX_PLAYERS,
    SSNode,
    SSTreeData,
)

TOL = 1e-9
TOL_NH = 1e-6   # vec-nh closed-form recovery (double algebra, see gate 4)
DEFAULT_CASES = 100_000
DEFAULT_VECNH_CASES = 2_000


# ── C library ────────────────────────────────────────────────────────────────

def _load_lib() -> ctypes.CDLL:
    ext = ".dll" if sys.platform == "win32" else ".so"
    name = os.environ.get("ES_DLL", "es_resolver")
    if not name.endswith((".dll", ".so")):
        name += ext
    path = os.path.join(_REPO, "build", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — build es_resolver first")
    lib = ctypes.CDLL(path)
    for fn_name in ("es_test_showdown_value", "es_test_showdown_value_vec"):
        if not hasattr(lib, fn_name):
            raise RuntimeError(f"{fn_name} missing from {path} — stale build")
        fn = getattr(lib, fn_name)
        fn.restype = ctypes.c_double
        fn.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ]
    # Full-solver surface for the VEC-NH gate (gate 4): drive es_traverse_vec
    # with num_hands[traverser] > 1 through the production entry points.
    for fn_name in ("es_create", "es_free", "es_set_vec_mode", "es_run",
                    "es_final_strategy", "es_get_last_error"):
        if not hasattr(lib, fn_name):
            raise RuntimeError(f"{fn_name} missing from {path} — stale build")
    lib.es_create.restype = ctypes.c_void_p
    lib.es_create.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    lib.es_free.argtypes = [ctypes.c_void_p]
    lib.es_set_vec_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.es_run.restype = ctypes.c_int
    lib.es_run.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
    lib.es_final_strategy.restype = ctypes.c_int
    lib.es_final_strategy.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.es_get_last_error.restype = ctypes.c_char_p
    return lib


_LIB = _load_lib()


def c_net(vec: bool, np_: int, board: Sequence[int],
          hands: Sequence[Tuple[int, int]], active: Sequence[int],
          bets: Sequence[int], pot: int, traverser: int) -> float:
    arr = lambda xs: (ctypes.c_int * len(xs))(*xs)  # noqa: E731
    fn = (_LIB.es_test_showdown_value_vec if vec
          else _LIB.es_test_showdown_value)
    out = fn(np_, arr(list(board)),
             arr([h[0] for h in hands]), arr([h[1] for h in hands]),
             arr(list(active)), arr(list(bets)), pot, traverser)
    if math.isnan(out):
        raise RuntimeError("C export returned NAN (invalid args)")
    return float(out)


# ── Independent exact reference (pot-stripping formulation) ─────────────────

def ref_awards(np_: int, board: Sequence[int],
               hands: Sequence[Tuple[int, int]], active: Sequence[int],
               bets: Sequence[int], pot: int) -> List[float]:
    """GROSS award per player (before subtracting own street bet)."""
    actives = [p for p in range(np_) if active[p]]
    awards = [0.0] * np_
    if not actives:
        return awards
    strengths = {p: _eval7_py(list(board) + [hands[p][0], hands[p][1]])
                 for p in actives}
    top = max(bets[p] for p in actives)
    rem = [min(max(bets[p], 0), top) for p in range(np_)]
    base = float(pot) - float(sum(rem))

    def award_layer(layer: float, elig: List[int]) -> None:
        best = max(strengths[p] for p in elig)
        wins = [p for p in elig if strengths[p] == best]
        for w in wins:
            awards[w] += layer / len(wins)

    # Base layer (entry pot + folder excess + accounting slack): EVERY
    # showdown-active player contests it — including a player whose THIS-street
    # bet is 0 (e.g. all-in on a prior street): they matched all prior streets,
    # so the lump entry money is common. Matches the C convention (the bottom
    # level's eligibility, bets >= min active bet, always spans all actives).
    award_layer(base, actives)
    while True:
        live = [p for p in actives if rem[p] > 0]
        if not live:
            break
        m = min(rem[p] for p in live)
        layer = 0.0
        for p in range(np_):
            take = min(rem[p], m)
            rem[p] -= take
            layer += float(take)
        award_layer(layer, live)
    return awards


def ref_net(np_, board, hands, active, bets, pot, traverser) -> float:
    if not active[traverser]:
        return -float(bets[traverser])
    awards = ref_awards(np_, board, hands, active, bets, pot)
    return awards[traverser] - float(bets[traverser])


def old_single_pot_net(np_, board, hands, active, bets, pot, traverser) -> float:
    """The HISTORICAL single-pot closed form (pre-side-pot es_showdown_value),
    IEEE-identical operations — the bit-exact regression oracle."""
    if not active[traverser]:
        return -float(bets[traverser])
    actives = [p for p in range(np_) if active[p]]
    strengths = {p: _eval7_py(list(board) + [hands[p][0], hands[p][1]])
                 for p in actives}
    best = max(strengths.values())
    winners = [p for p in actives if strengths[p] == best]
    if traverser in winners:
        return float(pot) / len(winners) - float(bets[traverser])
    return -float(bets[traverser])


def is_single_pot(np_, active, bets) -> bool:
    ab = [bets[p] for p in range(np_) if active[p]]
    fb = [bets[p] for p in range(np_) if not active[p]]
    if not ab:
        return True
    return min(ab) == max(ab) and (not fb or max(fb) <= max(ab))


# ── Unit tests (literal hand-computed expecteds) ────────────────────────────

def card(rank: int, suit: int) -> int:
    """rank 0=2 .. 12=A; suit 0..3 — matches _eval5_py (c>>2 / c&3) and eval7."""
    return rank * 4 + suit


def _units() -> List[dict]:
    A, K, Q, J, T = 12, 11, 10, 9, 8
    u = []
    # U1: 3p all-in-for-less. entry 300; bets 50/200/200; pot 750.
    # AA (allin 50) best overall, KK second, 33 worst.
    # bottom layer 450+base300−wait: base=300, bottom=50×3+300=450; side=300.
    # AA wins bottom-with-base (50*3+300=450), KK wins side 300.
    u.append(dict(
        name="U1_3p_allin_chain",
        np_=3,
        board=[card(0, 0), card(5, 1), card(7, 2), card(9, 0), card(10, 3)],
        hands=[(card(A, 0), card(A, 1)), (card(K, 0), card(K, 1)),
               (card(1, 0), card(1, 1))],
        active=[1, 1, 1], bets=[50, 200, 200], pot=750,
        expect=[450.0 - 50.0, 300.0 - 200.0, -200.0]))
    # U2: side-pot layer tie. entry 0; bets 50/200/200; pot 450.
    # board As Kc Qh 7d 8c; A=22 (pair 2), B=KsQs, C=KdQd (identical two pair).
    # bottom 150 split B/C (75 each); side 300 split (150 each).
    u.append(dict(
        name="U2_layer_tie",
        np_=3,
        board=[card(A, 0), card(K, 1), card(Q, 2), card(5, 3), card(6, 1)],
        hands=[(card(0, 0), card(0, 1)), (card(K, 0), card(Q, 0)),
               (card(K, 3), card(Q, 3))],
        active=[1, 1, 1], bets=[50, 200, 200], pot=450,
        expect=[-50.0, 225.0 - 200.0, 225.0 - 200.0]))
    # U3: dead money from a folder. 4p, D folded bets=40; A allin 100; B=C=300.
    # entry 200 → pot 940. base = 940−(100+300+300+40) = 200.
    # bottom (0,100]: 100×3+40+200 = 540 → A (AA best). side (100,300]:
    # 200×2=400 → B (KK > QQ).
    u.append(dict(
        name="U3_dead_money",
        np_=4,
        board=[card(0, 0), card(3, 1), card(5, 2), card(7, 3), card(9, 0)],
        hands=[(card(A, 0), card(A, 1)), (card(K, 0), card(K, 1)),
               (card(Q, 0), card(Q, 1)), (card(1, 2), card(2, 3))],
        active=[1, 1, 1, 0], bets=[100, 300, 300, 40], pot=940,
        expect=[540.0 - 100.0, 400.0 - 300.0, -300.0, -40.0]))
    # U4: folder excess ABOVE the top active level (convention: excess → base).
    # A=100, B=100 active; F folded 150. entry 0 → pot 350. contributions
    # capped at 100 ⇒ base = 350−300 = 50; single layer 300+50=350 → A (AA).
    u.append(dict(
        name="U4_folder_excess",
        np_=3,
        board=[card(0, 0), card(3, 1), card(5, 2), card(7, 3), card(9, 0)],
        hands=[(card(A, 0), card(A, 1)), (card(K, 0), card(K, 1)),
               (card(1, 2), card(2, 3))],
        active=[1, 1, 0], bets=[100, 100, 150], pot=350,
        expect=[350.0 - 100.0, -100.0, -150.0]))
    # U5: 6p all-in chain 10/20/40/80/160/160, entry 60 → pot 530. base 60.
    # layers: (0,10] 60+60=120 → p0 (AA). (10,20] 50, (20,40] 80, (40,80] 120,
    # (80,160] 160 → all to p4 (KK, best among eligibles) = 410.
    u.append(dict(
        name="U5_6p_chain",
        np_=6,
        board=[card(1, 0), card(5, 1), card(7, 2), card(9, 1), card(2, 3)],
        hands=[(card(A, 0), card(A, 1)), (card(0, 0), card(0, 1)),
               (card(3, 0), card(3, 1)), (card(6, 0), card(6, 1)),
               (card(K, 0), card(K, 1)), (card(Q, 0), card(Q, 1))],
        active=[1, 1, 1, 1, 1, 1], bets=[10, 20, 40, 80, 160, 160], pot=530,
        expect=[120.0 - 10.0, -20.0, -40.0, -80.0, 410.0 - 160.0, -160.0]))
    # U6: 2p equal-bet regression (degenerate). entry 100, bets 200/200 → 500.
    u.append(dict(
        name="U6_2p_regression",
        np_=2,
        board=[card(0, 0), card(3, 1), card(5, 2), card(7, 3), card(9, 0)],
        hands=[(card(A, 0), card(A, 1)), (card(K, 0), card(K, 1))],
        active=[1, 1], bets=[200, 200], pot=500,
        expect=[500.0 - 200.0, -200.0]))
    # U7: check-down (all street bets 0): pot = entry 400 → best hand takes it.
    u.append(dict(
        name="U7_checkdown",
        np_=3,
        board=[card(0, 0), card(3, 1), card(5, 2), card(7, 3), card(9, 0)],
        hands=[(card(A, 0), card(A, 1)), (card(K, 0), card(K, 1)),
               (card(Q, 0), card(Q, 1))],
        active=[1, 1, 1], bets=[0, 0, 0], pot=400,
        expect=[400.0, 0.0, 0.0]))
    return u


def run_units() -> bool:
    ok = True
    for u in _units():
        np_, board, hands = u["np_"], u["board"], u["hands"]
        active, bets, pot = u["active"], u["bets"], u["pot"]
        for trav in range(np_):
            exp = u["expect"][trav]
            rv = ref_net(np_, board, hands, active, bets, pot, trav)
            cs = c_net(False, np_, board, hands, active, bets, pot, trav)
            cv = c_net(True, np_, board, hands, active, bets, pot, trav)
            for label, got in (("ref", rv), ("C-scalar", cs), ("C-vec", cv)):
                if abs(got - exp) > 1e-12:
                    print(f"  FAIL {u['name']} trav={trav} {label}: "
                          f"got {got!r} expected {exp!r}")
                    ok = False
        # zero-sum invariant on the reference
        aw = ref_awards(np_, board, hands, active, bets, pot)
        if abs(sum(aw) - pot) > 1e-9:
            print(f"  FAIL {u['name']}: ref awards sum {sum(aw)} != pot {pot}")
            ok = False
    print(f"UNITS: {'PASS' if ok else 'FAIL'} ({len(_units())} cases × all traversers)")
    return ok


# ── Fuzz ─────────────────────────────────────────────────────────────────────

def _gen_case(rng: random.Random) -> dict:
    np_ = rng.randint(2, 6)
    force_tie = rng.random() < 0.15
    if force_tie:
        # board plays for everyone holding low cards: quads + ace kicker.
        qrank = rng.randint(8, 11)
        board = [card(qrank, s) for s in range(4)] + [card(12, rng.randint(0, 3))]
        pool = [c for c in range(32) if c not in board]  # ranks 0..7 only
    else:
        deck = list(range(52))
        rng.shuffle(deck)
        board, pool = deck[:5], deck[5:]
    hole = rng.sample(pool, 2 * np_)
    hands = [(hole[2 * p], hole[2 * p + 1]) for p in range(np_)]
    while True:
        active = [1 if rng.random() < 0.75 else 0 for _ in range(np_)]
        if sum(active) >= 2:
            break
    M = rng.randint(0, 3000)
    degenerate_batch = rng.random() < 0.3
    bets = []
    for p in range(np_):
        if active[p]:
            if degenerate_batch:
                bets.append(M)
            else:
                bets.append(M if rng.random() < 0.5 else rng.randint(0, M))
        else:
            if not degenerate_batch and rng.random() < 0.05:
                bets.append(rng.randint(0, M + M // 2 + 1))  # folder excess
            else:
                bets.append(rng.randint(0, M))
    entry = rng.randint(0, 4000)
    pot = entry + sum(bets)
    if rng.random() < 0.2:
        pot -= rng.randint(0, min(entry, 300))  # builder accounting slack
    return dict(np_=np_, board=board, hands=hands, active=active,
                bets=bets, pot=pot)


def run_fuzz(cases: int) -> bool:
    rng = random.Random(20260610)
    max_d_scalar = 0.0
    max_d_vec = 0.0
    n_layered = n_degenerate = 0
    regress_fail = 0
    t0 = time.time()
    for i in range(cases):
        c = _gen_case(rng)
        np_, board, hands = c["np_"], c["board"], c["hands"]
        active, bets, pot = c["active"], c["bets"], c["pot"]
        awards = ref_awards(np_, board, hands, active, bets, pot)
        if abs(sum(awards) - pot) > TOL:
            print(f"FAIL case {i}: ref zero-sum broke: "
                  f"sum(awards)={sum(awards)} pot={pot} case={c}")
            return False
        degen = is_single_pot(np_, active, bets)
        if degen:
            n_degenerate += 1
        else:
            n_layered += 1
        for trav in range(np_):
            rv = (awards[trav] - float(bets[trav])) if active[trav] \
                 else -float(bets[trav])
            cs = c_net(False, np_, board, hands, active, bets, pot, trav)
            cv = c_net(True, np_, board, hands, active, bets, pot, trav)
            ds, dv = abs(cs - rv), abs(cv - rv)
            max_d_scalar = max(max_d_scalar, ds)
            max_d_vec = max(max_d_vec, dv)
            if ds > TOL or dv > TOL:
                print(f"FAIL case {i} trav={trav}: ref={rv!r} "
                      f"C-scalar={cs!r} C-vec={cv!r} case={c}")
                return False
            if degen:
                old = old_single_pot_net(np_, board, hands, active, bets,
                                         pot, trav)
                if cs != old or cv != old:
                    regress_fail += 1
                    print(f"REGRESS-FAIL case {i} trav={trav}: old={old!r} "
                          f"C-scalar={cs!r} C-vec={cv!r} case={c}")
                    if regress_fail > 5:
                        return False
        if (i + 1) % 20000 == 0:
            print(f"  ... {i + 1:,}/{cases:,} cases "
                  f"({time.time() - t0:.0f}s, max|d| scalar={max_d_scalar:.3g} "
                  f"vec={max_d_vec:.3g})", flush=True)
    ok = regress_fail == 0
    print(f"FUZZ: {'PASS' if ok else 'FAIL'} — {cases:,} cases "
          f"({n_degenerate:,} single-pot / {n_layered:,} layered), "
          f"max|C−ref| scalar={max_d_scalar:.3g} vec={max_d_vec:.3g}, "
          f"bit-exact regressions on single-pot: "
          f"{'0 failures' if ok else f'{regress_fail} FAILURES'} "
          f"[{time.time() - t0:.0f}s]")
    return ok


# ── Gate 4: VEC-NH — nh>1 vectorized showdown (the river-critical loop) ─────
#
# MECHANISM (exact one-shot value recovery, no statistics):
#   Tree: node 0 = DECISION owned by the traverser T=0 with 3 children, all
#   SHOWDOWN terminals: child a0 = the side-pot CASE under test; child a1 =
#   a node where T is INACTIVE with bets[T]=0   -> constant value K1 = 0;
#   child a2 = T INACTIVE with bets[T]=theta    -> constant value K2 = -theta.
#   Every opponent has EXACTLY ONE hand (deterministic sample); T has nh>1
#   hands. One single-thread vanilla es_run iteration performs ONE vec pass
#   for T: zero regrets -> uniform sigma -> per hand h,
#       reg[a] = v_a(h) - m(h),   m = (v + K1 + K2)/3.
#   With K1 = 0 and theta = 2*max_bet + max(pot,0) + 1000 it holds STRICTLY
#   for every reachable v in [-max_bet, pot] that v > m and K1 > m and
#   K2 < m, so es_final_strategy (regret matching) returns
#       s0/s1 = (v - m)/(-m)  ->  m = K2/(2 + s0/s1),  v = K2(1-r)/(2+r).
#   Recovery is closed-form double algebra -> per-hand |v_C - v_ref| checked
#   at TOL_NH. DEAD hands (board-collision or weight<=0) receive no update
#   -> exactly uniform rows (impossible for a live hand: reg0, reg1 > 0
#   strictly) — an exact dead/live discriminator.
#
# SAFETY: live T hands never share cards with opponents' fixed hands, so the
# default sampler (es_sample_hands, which also reserves a T hand) can never
# block an opponent and pollute the one-shot regrets. Board-colliding and
# weight-0 T hands are sampler-invisible (skipped) — safe to include.

_T = 0  # traverser player index in the vec-nh tree


def _vecnh_build_tree(np_: int, board: Sequence[int],
                      t_hands: Sequence[Tuple[int, int]],
                      t_weights: Sequence[float],
                      opp_hands: Sequence[Tuple[int, int]],
                      active: Sequence[int], bets: Sequence[int],
                      pot: int, theta: int):
    """4-node SSTreeData: decision(T) -> [case showdown, K1=0, K2=-theta].
    Returns (tree, keepalive) — keepalive holds the ctypes arrays."""
    nodes = (SSNode * 4)()
    children = (ctypes.c_int * 3)(1, 2, 3)

    root = nodes[0]
    root.type = 0
    root.player = _T
    root.num_children = 3
    root.first_child = 0
    root.pot = pot
    root.num_board = 5

    def fill_showdown(nd, nd_bets, nd_active, nd_pot):
        nd.type = 2          # ES_T_SHOWDOWN
        nd.player = -1
        nd.fold_player = -1
        nd.leaf_idx = -1
        nd.pot = nd_pot
        nd.num_board = 5
        for b in range(5):
            nd.board_cards[b] = board[b]
        for p in range(np_):
            nd.bets[p] = nd_bets[p]
            nd.active_players[p] = 1 if nd_active[p] else 0

    # child 1: the side-pot CASE under test.
    fill_showdown(nodes[1], list(bets), list(active), pot)
    # child 2 (K1 = 0): T inactive, bets[T] = 0 — value to T is exactly 0.
    k1_bets = [0] * np_
    k1_active = [0 if p == _T else 1 for p in range(np_)]
    fill_showdown(nodes[2], k1_bets, k1_active, pot)
    # child 3 (K2 = -theta): T inactive, bets[T] = theta.
    k2_bets = [theta if p == _T else 0 for p in range(np_)]
    fill_showdown(nodes[3], k2_bets, k1_active, pot)

    td = SSTreeData()
    td.nodes = ctypes.cast(nodes, ctypes.POINTER(SSNode))
    td.children = ctypes.cast(children, ctypes.POINTER(ctypes.c_int))
    td.num_nodes = 4
    td.num_children_total = 3
    td.num_players_field = np_
    td.num_board = 5
    for b in range(5):
        td.board[b] = board[b]
    td.num_hands[_T] = len(t_hands)
    for h, (c0, c1) in enumerate(t_hands):
        td.hands[_T][h][0] = c0
        td.hands[_T][h][1] = c1
        td.weights[_T][h] = float(t_weights[h])
    oi = 0
    for p in range(np_):
        if p == _T:
            continue
        td.num_hands[p] = 1
        td.hands[p][0][0] = opp_hands[oi][0]
        td.hands[p][0][1] = opp_hands[oi][1]
        td.weights[p][0] = 1.0
        oi += 1
    return td, (nodes, children)


def _vecnh_solve_readout(td, nh: int, seed: int):
    """es_create -> vec mode -> ONE single-thread iteration -> final sigma.
    Returns an [nh][3] list of regret-matched rows at the root."""
    handle = _LIB.es_create(ctypes.byref(td), None, None,
                            ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF))
    if not handle:
        err = _LIB.es_get_last_error()
        raise RuntimeError(f"es_create failed: {err.decode() if err else '?'}")
    try:
        _LIB.es_set_vec_mode(handle, 1)
        rc = _LIB.es_run(handle, ctypes.c_int64(1), 1)
        if rc != 0:
            err = _LIB.es_get_last_error()
            raise RuntimeError(f"es_run failed: {err.decode() if err else '?'}")
        buf = (ctypes.c_double * (nh * 3))()
        n = _LIB.es_final_strategy(handle, 0, buf, nh * 3)
        if n != nh * 3:
            err = _LIB.es_get_last_error()
            raise RuntimeError(
                f"es_final_strategy returned {n} (expected {nh * 3}): "
                f"{err.decode() if err else '?'}")
        return [[buf[h * 3 + a] for a in range(3)] for h in range(nh)]
    finally:
        _LIB.es_free(handle)


def _vecnh_recover(sigma_row: Sequence[float], theta: int) -> float:
    """Closed-form per-hand showdown value from the one-shot readout."""
    s0, s1 = float(sigma_row[0]), float(sigma_row[1])
    if s1 <= 0.0:
        raise ZeroDivisionError("s1 == 0 — recovery preconditions violated")
    r = s0 / s1
    k2 = -float(theta)
    return k2 * (1.0 - r) / (2.0 + r)


def _vecnh_gen_case(rng: random.Random) -> dict:
    """Random side-pot case with an nh>1 traverser range (live + board-dead +
    weight-0 hands), single-hand opponents, all-in-for-less chains included."""
    np_ = rng.randint(2, 6)
    deck = list(range(52))
    rng.shuffle(deck)
    board = deck[:5]
    pool = deck[5:]
    # opponents: one fixed, mutually-disjoint, board-disjoint hand each.
    n_opp = np_ - 1
    opp_cards = pool[:2 * n_opp]
    opp_hands = [(opp_cards[2 * i], opp_cards[2 * i + 1]) for i in range(n_opp)]
    free = pool[2 * n_opp:]   # never collides with board or opponents

    nh = rng.randint(2, 20)
    t_hands: List[Tuple[int, int]] = []
    t_weights: List[float] = []
    kinds: List[str] = []
    for h in range(nh):
        roll = rng.random()
        if roll < 0.15:
            # board-dead: one card from the board (sampler-invisible).
            c0 = rng.choice(board)
            c1 = rng.choice([c for c in free if c != c0])
            kind = "dead_board"
            w = 1.0
        elif roll < 0.25:
            # weight-0 dead: MAY overlap opponent cards (never sampled).
            c0 = rng.choice(opp_cards) if (opp_cards and rng.random() < 0.5) \
                else rng.choice(free)
            c1 = rng.choice([c for c in free if c != c0])
            kind = "dead_weight"
            w = 0.0
        else:
            # live: free-pool cards only; may overlap OTHER t_hands (vec
            # enumerates each hand independently — realistic 169-grid shape).
            c0, c1 = rng.sample(free, 2)
            kind = "live"
            w = 1.0
        t_hands.append((c0, c1))
        t_weights.append(w)
        kinds.append(kind)
    if "live" not in kinds:
        c0, c1 = rng.sample(free, 2)
        t_hands[0] = (c0, c1)
        t_weights[0] = 1.0
        kinds[0] = "live"

    # active/bets/pot — same families as _gen_case (chains, folders, slack).
    while True:
        active = [1 if rng.random() < 0.75 else 0 for _ in range(np_)]
        if sum(active) >= 2:
            break
    M = rng.randint(0, 3000)
    degenerate = rng.random() < 0.3
    bets = []
    for p in range(np_):
        if active[p]:
            bets.append(M if (degenerate or rng.random() < 0.5)
                        else rng.randint(0, M))
        else:
            if not degenerate and rng.random() < 0.05:
                bets.append(rng.randint(0, M + M // 2 + 1))  # folder excess
            else:
                bets.append(rng.randint(0, M))
    entry = rng.randint(0, 4000)
    pot = entry + sum(bets)
    if rng.random() < 0.2:
        pot -= rng.randint(0, min(entry, 300))  # builder accounting slack
    theta = 2 * max(bets + [0]) + max(pot, 0) + 1000
    return dict(np_=np_, board=board, t_hands=t_hands, t_weights=t_weights,
                kinds=kinds, opp_hands=opp_hands, active=active, bets=bets,
                pot=pot, theta=theta)


def _vecnh_check_case(c: dict, seed: int) -> Tuple[bool, float]:
    """Run one vec-nh case; returns (ok, max_abs_dev_over_live_hands)."""
    np_, board = c["np_"], c["board"]
    t_hands, t_weights, kinds = c["t_hands"], c["t_weights"], c["kinds"]
    opp_hands, active, bets, pot = (c["opp_hands"], c["active"], c["bets"],
                                    c["pot"])
    theta = c["theta"]
    nh = len(t_hands)
    td, _keep = _vecnh_build_tree(np_, board, t_hands, t_weights, opp_hands,
                                  active, bets, pot, theta)
    sigma = _vecnh_solve_readout(td, nh, seed)

    opp_card_set = {cd for hand in opp_hands for cd in hand}
    board_set = set(board)
    max_dev = 0.0
    ok = True
    for h in range(nh):
        c0, c1 = t_hands[h]
        dead = (t_weights[h] <= 0.0 or c0 in board_set or c1 in board_set
                or c0 in opp_card_set or c1 in opp_card_set)
        row = sigma[h]
        if dead:
            if any(abs(x - 1.0 / 3.0) > 1e-12 for x in row):
                print(f"  VEC-NH FAIL: dead hand {h} ({kinds[h]}) row not "
                      f"uniform: {row} case={c}")
                ok = False
            continue
        # live: exactly two positive regrets by construction -> sigma[2]==0.
        if row[2] != 0.0:
            print(f"  VEC-NH FAIL: live hand {h} has sigma[K2]={row[2]!r} "
                  f"(expected 0) case={c}")
            ok = False
            continue
        # per-hand exact reference: T holds hand h, opponents hold their
        # single fixed hands.
        ref_hands = [t_hands[h]] + list(opp_hands)
        rv = ref_net(np_, board, ref_hands, active, bets, pot, _T)
        try:
            got = _vecnh_recover(row, theta)
        except ZeroDivisionError:
            print(f"  VEC-NH FAIL: hand {h} recovery degenerate row={row} "
                  f"case={c}")
            ok = False
            continue
        dev = abs(got - rv)
        max_dev = max(max_dev, dev)
        if dev > TOL_NH:
            print(f"  VEC-NH FAIL: hand {h} ({kinds[h]}) recovered {got!r} "
                  f"ref {rv!r} |d|={dev:.3g} case={c}")
            ok = False
    return ok, max_dev


def run_vec_nh_units() -> bool:
    """Literal nh>1 case: U1's 3p all-in-for-less chain with T holding the
    best hand, the worst hand, a board-colliding hand and a weight-0 hand."""
    A, K = 12, 11
    board = [card(0, 0), card(5, 1), card(7, 2), card(9, 0), card(10, 3)]
    # opponents (players 1,2): KK (allin 200), 33 (200) — as in U1.
    opp_hands = [(card(K, 0), card(K, 1)), (card(1, 0), card(1, 1))]
    # T (player 0, allin 50): AA wins bottom 450 layer; 22 loses everything;
    # one board-colliding hand (2c is board[0]); one weight-0 hand.
    t_hands = [
        (card(A, 0), card(A, 1)),    # AA: 450 - 50 = +400
        (card(2, 0), card(2, 1)),    # 44: worst -> -50
        (card(0, 1), card(0, 2)),    # 2d2h: 2d ok, but pair with... live
        (card(0, 0), card(3, 3)),    # 2c5s: 2c IS board[0] -> dead
        (card(6, 0), card(6, 1)),    # 88 weight-0 -> dead
    ]
    t_weights = [1.0, 1.0, 1.0, 1.0, 0.0]
    kinds = ["live", "live", "live", "dead_board", "dead_weight"]
    case = dict(np_=3, board=board, t_hands=t_hands, t_weights=t_weights,
                kinds=kinds, opp_hands=opp_hands, active=[1, 1, 1],
                bets=[50, 200, 200], pot=750,
                theta=2 * 200 + 750 + 1000)
    ok, max_dev = _vecnh_check_case(case, seed=0xBEEF)
    # cross-check the literal: AA recovers +400 (asserted via ref in
    # _vecnh_check_case; assert the ref itself matches U1's literal here).
    rv_aa = ref_net(3, board, [t_hands[0]] + opp_hands, [1, 1, 1],
                    [50, 200, 200], 750, _T)
    if abs(rv_aa - 400.0) > 1e-12:
        print(f"  VEC-NH FAIL: U1 literal drifted: ref(AA)={rv_aa!r} != 400.0")
        ok = False
    print(f"VEC-NH UNITS: {'PASS' if ok else 'FAIL'} "
          f"(nh=5 incl. board-dead + weight-0; max|d|={max_dev:.3g})")
    return ok


def run_vec_nh_fuzz(cases: int) -> bool:
    rng = random.Random(20260611)
    t0 = time.time()
    max_dev = 0.0
    n_hands = n_dead = 0
    for i in range(cases):
        c = _vecnh_gen_case(rng)
        ok, dev = _vecnh_check_case(c, seed=0x5EED0000 + i)
        if not ok:
            print(f"VEC-NH FUZZ: FAIL at case {i}")
            return False
        max_dev = max(max_dev, dev)
        n_hands += len(c["t_hands"])
        n_dead += sum(1 for k in c["kinds"] if k != "live")
        if (i + 1) % 500 == 0:
            print(f"  ... vec-nh {i + 1:,}/{cases:,} "
                  f"({time.time() - t0:.0f}s, max|d|={max_dev:.3g})",
                  flush=True)
    print(f"VEC-NH FUZZ: PASS — {cases:,} cases, {n_hands:,} traverser hands "
          f"({n_dead:,} dead), max|recovered − ref| = {max_dev:.3g} "
          f"[{time.time() - t0:.0f}s]")
    return True


def main() -> int:
    cases = int(os.environ.get("SIDEPOT_CASES", str(DEFAULT_CASES)))
    vecnh_cases = int(os.environ.get("VECNH_CASES", str(DEFAULT_VECNH_CASES)))
    print("=" * 78)
    print(f"SIDE-POT SHOWDOWN GATE — es_resolver scalar+vec vs exact reference "
          f"({cases:,} fuzz cases; {vecnh_cases:,} vec-nh cases)")
    print("=" * 78, flush=True)
    ok = run_units()
    ok = run_vec_nh_units() and ok
    ok = run_vec_nh_fuzz(vecnh_cases) and ok
    ok = run_fuzz(cases) and ok
    print(f"\nGATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
