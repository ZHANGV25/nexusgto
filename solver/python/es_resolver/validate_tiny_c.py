"""Gate 1a: 2p BR-gap -> 0 for the PRODUCTION C ES solver (CSolver), on the
identical tiny zero-sum tree validate_tiny.py uses for the Python reference.

Reuses validate_tiny's metric code verbatim (exploitability, _value_under_profile,
_build_real_tree) — only the solver is swapped ESSolver -> CSolver.joint_zero_sum.
The metric stays in Python so a buggy C metric can't bless a buggy C solver.

A buggy ES gives a wrong T; prove correctness FIRST (guardrail #4). PASS if the
C solver's BR-gap falls toward ~0 as iters grow, matching the reference.

Run: PYTHONIOENCODING=utf-8 py -3.9 -m es_resolver.validate_tiny_c
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from street_solver_gpu import StreetSolverGPU  # noqa: E402
from solver import card_to_int  # noqa: E402
from es_resolver.es_solver import snapshot_tree  # noqa: E402
from es_resolver.validate_tiny import exploitability  # noqa: E402
from es_resolver.c_solver import CSolver  # noqa: E402


def _build_tree_and_strength(n_players, board_strs, leaf_seed=7, nh=4):
    """Build the live GPU SSTreeData AND its PlainTree snapshot + random
    integer strengths, exactly as validate_tiny._build_real_tree but KEEPING
    the live solver so its SSTreeData can drive the C solver."""
    board_ints = [card_to_int(c) for c in board_strs]
    used = set(board_ints)
    deck = [x for x in range(52) if x not in used]
    ranges = []
    cur = 0
    for _p in range(n_players):
        hs = []
        for _ in range(nh):
            a, b = deck[cur], deck[cur + 1]
            cur += 2
            hs.append((a, b, 1.0))
        ranges.append(hs)
    solver = StreetSolverGPU(
        board=board_strs, pot_bb=20.0, stack_bb=95.0,
        player_ranges=ranges, bet_sizes=[0.75], subsequent_bet_sizes=[1.0],
        use_cont_strats=True, leaf_value_fn=None, add_explicit_allin=False,
    )
    snap = snapshot_tree(solver._tree, ranges)
    rng = np.random.default_rng(leaf_seed)
    strength = [rng.integers(0, 10, size=snap.num_hands[p]).astype(float)
                for p in range(n_players)]
    return solver, snap, strength


def main() -> int:
    print("=" * 72)
    print("GATE 1a: C ES solver best-response gap -> 0 (tiny 2p ZERO-SUM tree)")
    print("=" * 72)
    solver, tree, strength = _build_tree_and_strength(
        2, ["Ah", "Kh", "2c"], leaf_seed=7, nh=4)

    print(f"\n tree: nodes={len(tree.nodes)} "
          f"decisions={len(tree.decision_node_indices)} nh=4")
    print(f" strengths p0={strength[0].tolist()} p1={strength[1].tolist()}")

    # baseline uniform-σ gap (no solve) — same metric as the reference.
    uni_gap, _, _ = exploitability(tree, {}, strength)
    print(f" baseline (uniform sigma) BR-gap = {uni_gap:.3f}")

    print(f"\n {'iters':>8} {'traversals':>11} {'rollouts':>10} {'r/t':>5} "
          f"{'BR-gap':>10} {'(gap0':>8} {'gap1)':>8}")

    gpu_tree = solver._tree  # live SSTreeData ctypes struct
    strength0 = strength[0].astype(np.float64)
    strength1 = strength[1].astype(np.float64)

    last_gap = None
    for iters in (2_000, 10_000, 40_000, 120_000, 300_000):
        cs = CSolver.joint_zero_sum(gpu_tree, strength0, strength1, seed=1)
        cs.run(iters, num_threads=1)
        sigma = {ni: cs.avg_strategy(ni) for ni in cs.info.strat_sum
                 if cs.avg_strategy(ni) is not None}
        gap, g0, g1 = exploitability(tree, sigma, strength)
        st = cs.stats
        rpt = st.leaf_rollouts / max(st.traversals, 1)
        print(f" {iters:>8,} {st.traversals:>11,} {st.leaf_rollouts:>10,} "
              f"{rpt:>5.2f} {gap:>10.4f} {g0:>8.3f} {g1:>8.3f}")
        last_gap = gap

    solver._freed = True  # native ss_free_tree() hangs on this box; skip it

    print("\n VERDICT: PASS if BR-gap falls toward ~0 as iters grow "
          "(matches the Python reference 2781->24).")
    ok = last_gap is not None and last_gap < 100.0
    print(f"  -> final gap = {last_gap:.4f}  {'PASS' if ok else 'INVESTIGATE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
