"""Side-pot showdown validation gate — es_resolver.c multiway layering.

Validates the C ES re-solver's SHOWDOWN terminal (both the scalar
es_showdown_value path and the vectorized traverser's inline showdown block)
against an INDEPENDENT exact Python side-pot reference, per the bit-exact-gate
pattern of docs/CPU_ES_RESOLVER_BENCHMARK.md (nexusgto-api).

Three sub-gates:
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

Reference formulation: iterative pot-STRIPPING (min-remaining-contribution
rounds) — deliberately a DIFFERENT algorithm from the C's level-interval loop,
computing the same function. Conventions shared with es_layered_award:
  - contributions capped at the top ACTIVE street-bet level;
  - base layer = pot − Σ capped contributions (entry pot + folder excess +
    builder accounting slack), awarded with the bottom layer;
  - layer eligibility = active players with street bet ≥ the layer's top;
  - ties chop by double division.

Run:  PYTHONUTF8=1 [ES_DLL=es_resolver] [SIDEPOT_CASES=100000] \
      python -m python.es_resolver.validate_sidepot
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

TOL = 1e-9
DEFAULT_CASES = 100_000


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


def main() -> int:
    cases = int(os.environ.get("SIDEPOT_CASES", str(DEFAULT_CASES)))
    print("=" * 78)
    print(f"SIDE-POT SHOWDOWN GATE — es_resolver scalar+vec vs exact reference "
          f"({cases:,} fuzz cases)")
    print("=" * 78, flush=True)
    ok = run_units()
    ok = run_fuzz(cases) and ok
    print(f"\nGATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
