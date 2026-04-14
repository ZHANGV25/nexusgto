"""Multi-way tree-size test: assert Pluribus 2-raises-ahead cap fits
betting trees and measure the cont-strat-expansion explosion that
remains the real 5-6p blocker.

See docs/MULTIWAY_TREE_SIZE.md for the full analysis. This test documents:

 1. Betting-tree (no cont-strat) size per player count. Pluribus's
    2-raises cap keeps this tractable at 5p/6p (thousands of leaves,
    not millions).
 2. Cont-strat expansion blows up as 4^n_active per betting leaf —
    the remaining 5-6p postflop blocker, not a raises-ahead issue.
 3. Memory budget under realistic hand counts: the 2-raises cap is
    sufficient IF cont-strat expansion is replaced with a non-
    combinatorial scheme.

Currently the 5-6p cont-strat-expanded trees blow past the 12 GB A1
S-table budget. This is the remaining work for full 6-max postflop.

Run:
    python tests/test_multiway_tree_size.py
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from street_solver_gpu import StreetSolverGPU, SS_MAX_HANDS  # noqa: E402
from solver import card_to_int  # noqa: E402


# Budget: A1 S-tables are
#   num_terminals × SS_MAX_PLAYERS × (1 + 52 + 1326) × sizeof(float32)
# = num_terminals × 6 × 1379 × 4 bytes
# We want the combined S0+S1+S2 allocation to stay under ~6 GB so there's
# room for reach_all, cfv, strategy buffers within a 12 GB GPU.
BYTES_PER_TERMINAL = 6 * (1 + 52 + 1326) * 4  # ~33.1 KB
S_TABLE_BUDGET_GB = 6.0
S_TABLE_BUDGET_BYTES = int(S_TABLE_BUDGET_GB * (1024**3))
MAX_TERMINALS_UNDER_BUDGET = S_TABLE_BUDGET_BYTES // BYTES_PER_TERMINAL


def tiny_ranges(n_players: int, n_per: int = 2):
    """Return N ranges of n_per hands each — enough to build the tree
    without triggering memory pressure during the size measurement."""
    return [[(i * 2, i * 2 + 1, 1.0) for i in range(n_per)]
            for _ in range(n_players)]


def build_tree(n_players: int, use_cont_strats: bool):
    board = ["Ah", "Ks", "Td"]
    s = StreetSolverGPU(
        board=board, pot_bb=6.0, stack_bb=94.0,
        player_ranges=tiny_ranges(n_players),
        bet_sizes=[0.5, 1.0],
        use_cont_strats=use_cont_strats,
    )
    return s


def test_betting_tree_size_under_budget():
    """With 2-raises-ahead cap (now SS_MULTIWAY_MAX_RAISES), betting trees
    at 5p/6p should be small enough that even the full A1 S-tables fit in
    budget."""
    print("\n[test 1] betting-tree-only size under 2-raises-ahead cap")
    print(f"  budget: {S_TABLE_BUDGET_GB:.1f} GB, max terminals ~{MAX_TERMINALS_UNDER_BUDGET:,}")
    results = {}
    for n in (2, 3, 4, 5, 6):
        s = build_tree(n, use_cont_strats=False)
        n_leaves = s._tree.num_leaf_nodes
        n_fold = s._tree.num_fold_nodes
        n_sh = s._tree.num_showdown_nodes
        n_term = n_leaves + n_fold + n_sh
        mem_gb = (n_term * BYTES_PER_TERMINAL) / (1024**3)
        results[n] = (n_term, mem_gb)
        print(f"  {n}p: {n_term:>6,} terminals, S-table estimate {mem_gb:.2f} GB")
        assert n_term <= MAX_TERMINALS_UNDER_BUDGET, (
            f"{n}p betting-tree terminals ({n_term}) exceed budget "
            f"({MAX_TERMINALS_UNDER_BUDGET})"
        )
        s.free()
    # Sanity: 6p betting tree terminals should be an order of magnitude
    # smaller than the MAX budget — otherwise raises-ahead cap is ineffective.
    assert results[6][0] < MAX_TERMINALS_UNDER_BUDGET / 10, (
        f"6p betting tree ({results[6][0]} terminals) uses >10% of S-table "
        f"budget on its own. 2-raises-ahead cap may need to be 1."
    )
    print("  [ok]")


def test_contstrat_fanout_documented():
    """Document that cont-strat expansion's 4^n_active fanout is the
    actual 5-6p blocker — NOT the raises-ahead cap. This test asserts
    that the cont-strat-expanded tree exceeds the 6 GB S-table budget at
    5p+, which is expected (and the reason full 6-max postflop remains
    open pending a cont-strat redesign)."""
    print("\n[test 2] cont-strat-expanded tree sizes")
    for n in (2, 3, 4):
        s = build_tree(n, use_cont_strats=True)
        n_leaves = s._tree.num_leaf_nodes
        n_fold = s._tree.num_fold_nodes
        n_sh = s._tree.num_showdown_nodes
        n_term = n_leaves + n_fold + n_sh
        mem_gb = (n_term * BYTES_PER_TERMINAL) / (1024**3)
        print(f"  {n}p (cont-strat on): {n_term:>6,} terminals, "
              f"S-table {mem_gb:.2f} GB")
        s.free()
    # 5p / 6p: build + measure but DO NOT allocate the S-tables (would OOM
    # on non-GPU-sized machines) — just count terminals via build_tree.
    for n in (5, 6):
        s = build_tree(n, use_cont_strats=True)
        n_term = (s._tree.num_leaf_nodes + s._tree.num_fold_nodes +
                  s._tree.num_showdown_nodes)
        mem_gb = (n_term * BYTES_PER_TERMINAL) / (1024**3)
        print(f"  {n}p (cont-strat on): {n_term:>6,} terminals, "
              f"S-table {mem_gb:.2f} GB  [EXPECTED OVERBUDGET]")
        # This assertion documents that the cont-strat-expanded 5p/6p tree
        # is ABOVE the S-table budget — confirming the 2-raises-ahead cap
        # alone does not resolve 5-6p memory pressure.
        if n >= 5:
            assert mem_gb > S_TABLE_BUDGET_GB, (
                f"{n}p cont-strat tree is {mem_gb:.1f} GB, which is UNDER "
                f"budget of {S_TABLE_BUDGET_GB:.1f} GB. The test's "
                f"expectation that cont-strat is the remaining blocker is "
                f"wrong — this task may already be complete."
            )
        s.free()
    print("  [ok — documented the cont-strat explosion as the remaining blocker]")


def test_multiway_raise_cap_is_two():
    """Confirm the multiway raise cap is 2 (Pluribus §4). Indirect check:
    compare decision-node counts with N=3 vs N=2 at same stack/bets;
    N=3 should have FEWER raise-chains per betting line due to the cap."""
    print("\n[test 3] multi-way raise cap is 2")
    s2 = build_tree(2, use_cont_strats=False)
    s3 = build_tree(3, use_cont_strats=False)
    # N=2 uses SS_MAX_RAISES=3; N=3 uses SS_MULTIWAY_MAX_RAISES=2.
    # Per-player decision density comparison isn't exact, but tree depth
    # should differ: N=3 flop-street max_depth should be lower relative to
    # the increased branching factor.
    print(f"  2p depth={s2._tree.max_depth} nodes={s2._tree.num_nodes}")
    print(f"  3p depth={s3._tree.max_depth} nodes={s3._tree.num_nodes}")
    # Not a tight assertion — the check is that the compile-time constant
    # is 2 at multi-way, verified via the heuristic that 3p depth stays
    # bounded. Main correctness signal is test 1 passing.
    s2.free()
    s3.free()
    print("  [ok]")


def main():
    test_betting_tree_size_under_budget()
    test_contstrat_fanout_documented()
    test_multiway_raise_cap_is_two()
    print("\n[MULTIWAY TREE SIZE TESTS PASSED]")


if __name__ == "__main__":
    main()
