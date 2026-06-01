"""Gate 1b (variant-independent multiway mechanics): C ES forward_ev_uniform vs
EXACT enumeration. Complements gate 2's L1 test — this tests the WALK/TERMINAL
arithmetic (≥3p _fold_value, betting walk, leaf indexing) under a FIXED uniform
profile (no solving), immune to equilibrium multiplicity. The enumeration stays
in Python (validate_fixed_profile.enumerate_root_ev); only the ES forward
estimate is the C solver's es_forward_ev_uniform.

Objective test (advisor): |Δ|/SE < 3 = MC noise, not a bug. Matches the Python
reference's 3p+4p all-within-3·SE result.

Run: PYTHONIOENCODING=utf-8 py -3.9 -m es_resolver.validate_fixed_profile_c
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from street_solver_gpu import StreetSolverGPU  # noqa: E402
from solver import card_to_int  # noqa: E402
from es_resolver.es_solver import snapshot_tree, T_LEAF  # noqa: E402
from es_resolver.validate_vs_gpu import _disjoint_ranges  # noqa: E402
from es_resolver.validate_fixed_profile import enumerate_root_ev  # noqa: E402
from es_resolver.c_solver import CSolver  # noqa: E402


def main() -> int:
    print("=" * 76)
    print("GATE 1b: C ES forward-uniform EV vs EXACT enumeration (multiway mechanics)")
    print("(variant-independent; isolates the new multiway walk/terminal code)")
    print("=" * 76)
    board = ["Ah", "Kh", "2c"]
    bi = [card_to_int(c) for c in board]
    all_ok = True
    for n_players in (3, 4):
        ranges = _disjoint_ranges(bi, n_players, 6)
        gpu = StreetSolverGPU(
            board=board, pot_bb=20.0, stack_bb=95.0, player_ranges=ranges,
            bet_sizes=[0.75], subsequent_bet_sizes=[1.0],
            use_cont_strats=True, leaf_value_fn=None, add_explicit_allin=False,
        )
        snap = snapshot_tree(gpu._tree, ranges)
        nL = max(nd.leaf_idx for nd in snap.nodes if nd.type == T_LEAF) + 1
        maxh = snap.max_hands
        # SAME random leaf table fed to enumeration (Python) and the C oracle.
        table = np.random.default_rng(123).normal(
            0, 50, size=(nL, n_players, maxh)).astype(np.float32)

        # Build the C solver bound to the live tree + the table oracle.
        cs = CSolver.from_table(gpu._tree, table, n_players, maxh, seed=0)

        # MULTI-SEED mean test (advisor-rigor 2026-06-01): a single seed at the
        # 3·SE boundary is a coin-flip (with 7 players, one hitting z≈3 by chance
        # is expected). The UNBIASEDNESS gate is: the mean over S seeds must lie
        # within 3·SE_of_the_mean of the exact enumeration. SE_of_mean shrinks
        # ~1/√(S·iters), so a true bias surfaces and pure noise averages out.
        S = 8
        ITERS = 1_000_000
        print(f"\n {n_players}p flop (uniform σ): per-player root EV "
              f"enum vs C-ES mean over {S} seeds × {ITERS:,} iters "
              f"(|Δ| < 3·SE_mean = unbiased)")
        ok = True
        for p in range(n_players):
            enum_ev = enumerate_root_ev(snap, table, p)
            ests = []
            for seed in range(S):
                cs_s = CSolver.from_table(gpu._tree, table, n_players, maxh, seed=seed)
                m, _ = cs_s.forward_ev_uniform(p, ITERS)
                ests.append(m)
            ests = np.asarray(ests)
            mean = ests.mean()
            se_mean = ests.std(ddof=1) / np.sqrt(S) if S > 1 else float("inf")
            d = abs(enum_ev - mean)
            z = d / se_mean if se_mean > 0 else float("inf")
            flag = "OK (unbiased)" if z < 3.0 else "MISMATCH (biased = BUG)"
            if z >= 3.0:
                ok = False
            print(f"   p{p}: enum={enum_ev:+8.3f}  C-ES={mean:+8.3f}±{se_mean:.3f}(mean)  "
                  f"|Δ|={d:6.3f}  |Δ|/SE_mean={z:4.1f}  {flag}")
        gpu._freed = True
        _ = cs  # keep the single-shot solver referenced for the live tree
        print(f"   => {n_players}p multiway mechanics: "
              f"{'VALIDATED' if ok else 'BUG (>3·SE)'}")
        all_ok = all_ok and ok

    print(f"\n  >>> GATE 1b {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
