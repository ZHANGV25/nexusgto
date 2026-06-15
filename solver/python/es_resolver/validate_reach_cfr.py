"""Local correctness gate for the full-per-hand-REACH vector CFR path (2026-06-14).

The CPU ES re-solver gained a `reach_mode` (es_set_reach_mode / CSolver.set_reach_mode):
full-per-hand opponent REACH, integrate ALL opponent hands each iteration with NO
sampling, ENUMERATE chance — a pure-C port of street_solve.cu's vector CFR math.
This is the INTERACTIVE 2p in-tree-chance solve (the sampling vec path was correct-
but-slow AND, this gate shows, actually still degenerate at the chance-tree fixed
point — reach mode fixes both).

These checks run LOCALLY (CPU-only, sub-minute on the small turn/river trees) with
NO blueprint and NO GPU — they prove the MATH. GPU-vs-CPU bit-agreement and the
QJs/blueprint check need EC2 (a GPU box + the 12GB blueprint) and live elsewhere.

The gates (a green test here CANNOT be faked by a degenerate strategy — see the
prior-session footgun "a green test can be blind to the bug"):
  G1 DETERMINISM   reach mode is seed-invariant AND thread-count-invariant
                   (bit-identical), because it samples nothing.
  G2 NO-CHANCE     on a RIVER tree (0 chance nodes, pure showdown) reach mode
                   AGREES with the heavily-validated SCALAR sampling reference at
                   the shallow/root nodes — so the reach math (regret/showdown/
                   fold/BR) is correct independent of the chance machinery.
  G3 EXPLOIT-DROP  on a TURN tree (real in-tree chance nodes) the reach strategy's
                   2p best-response exploitability DROPS monotonically toward 0 as
                   iters increase, measured by es_exploitability_2p — a VALIDATED
                   computer (G2 validates it against the scalar reference on river;
                   it reuses the same showdown evaluator so it grades chance trees,
                   unlike the Python exploitability_c.py which zeros showdowns).
  G4 BEATS-SAMPLE  on the SAME turn tree the reach strategy is FAR less exploitable
                   than the prior sampling-vec strategy (the degenerate over-bet) —
                   the decisive correctness discriminator.

Run:
  ES_DLL=es_resolver POKER_SEARCH_DEPTH=full POKER_SEARCH_SEED=1 \
  PYTHONIOENCODING=utf-8 python3 -m es_resolver.validate_reach_cfr
Exit 0 = all gates pass.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ES_DLL", "es_resolver")
os.environ.setdefault("POKER_SEARCH_DEPTH", "full")
os.environ.setdefault("POKER_SEARCH_SEED", "1")

POT_BB = 6.5
POT_CHIPS = POT_BB * 100.0


def _build(nh, board):
    """A 2p subgame on `board` with `nh` arbitrary off-deck hands per player."""
    from street_solver_gpu import StreetSolverGPU
    from solver import card_to_int

    bset = {card_to_int(c) for c in board}
    deck = [x for x in range(52) if x not in bset]
    ranges = []
    cur = 0
    for _p in range(2):
        ranges.append([(deck[cur + 2 * i], deck[cur + 2 * i + 1], 1.0)
                       for i in range(nh)])
        cur += 2 * nh
    gpu = StreetSolverGPU(
        board=list(board), pot_bb=POT_BB, stack_bb=97.5, player_ranges=ranges,
        bet_sizes=[0.75], subsequent_bet_sizes=[1.0], use_cont_strats=True,
        leaf_value_fn=None, add_explicit_allin=False)
    return gpu, ranges


def _root_node(tree):
    for ni in range(tree.num_nodes):
        nd = tree.nodes[ni]
        if nd.type == 0 and nd.num_children > 0:
            return ni
    return None


def _all_decision_sigs(cs, tree):
    out = {}
    for ni in range(tree.num_nodes):
        nd = tree.nodes[ni]
        if nd.type == 0 and nd.num_children > 0:
            s = cs.avg_strategy(ni)
            if s is not None:
                out[ni] = s
    return out


def gate1_determinism():
    """Seed- and thread-count-invariance (reach mode samples nothing)."""
    from es_resolver.c_solver import CSolver
    gpu, _ = _build(nh=6, board=("Ah", "Ks", "Td", "2c"))  # turn, 7 chance
    t = gpu._tree
    max_h = gpu._max_hands
    table = np.zeros((0, 2, max_h), dtype=np.float32)
    try:
        def run(seed, threads):
            cs = CSolver.from_table(t, table, 2, max_h, seed=seed)
            cs.set_reach_mode(1)
            cs.run(200, num_threads=threads)
            return _all_decision_sigs(cs, t)
        a = run(1, 1)
        b = run(999, 1)     # different seed -> must be identical (deterministic)
        c = run(1, 4)       # 4 threads     -> must be bit-identical to 1 thread
        seed_diff = max(float(np.max(np.abs(a[k] - b[k]))) for k in a)
        thread_diff = max(float(np.max(np.abs(a[k] - c[k]))) for k in a)
        ok = (seed_diff == 0.0) and (thread_diff == 0.0)
        print(f"  G1 determinism: seed1-vs-seed999 maxdiff={seed_diff:.2e}, "
              f"1-vs-4-thread maxdiff={thread_diff:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        gpu._freed = True


def gate2_no_chance_matches_scalar():
    """On a RIVER tree (0 chance) reach mode matches the validated scalar sampler
    at the root/shallow nodes — the reach math is correct independent of chance."""
    from es_resolver.c_solver import CSolver
    gpu, _ = _build(nh=6, board=("Ah", "Ks", "Td", "2c", "7h"))  # river, 0 chance
    t = gpu._tree
    assert int(getattr(t, "num_chance_nodes", 0)) == 0, "river must have 0 chance"
    max_h = gpu._max_hands
    table = np.zeros((0, 2, max_h), dtype=np.float32)
    try:
        cs_r = CSolver.from_table(t, table, 2, max_h, seed=1)
        cs_r.set_reach_mode(1)
        cs_r.run(2000, num_threads=1)
        sr = _all_decision_sigs(cs_r, t)

        cs_s = CSolver.from_table(t, table, 2, max_h, seed=1)
        cs_s.set_cfr_mode(1)
        cs_s.run(200000, num_threads=1)   # validated scalar sampling reference
        ss = _all_decision_sigs(cs_s, t)

        root = _root_node(t)
        root_diff = float(np.max(np.abs(sr[root] - ss[root])))
        # The root (and shallow) nodes carry the reach and converge; deep
        # near-zero-reach betting lines are CFR-noisy on BOTH and not graded.
        ok = root_diff < 0.02
        print(f"  G2 no-chance (river): root reach-vs-scalar maxdiff="
              f"{root_diff:.4f} (< 0.02)  {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        gpu._freed = True


def gate3_exploitability_drops():
    """On a TURN tree (real chance) reach exploitability drops toward 0."""
    from es_resolver.c_solver import CSolver
    gpu, _ = _build(nh=6, board=("Ah", "Ks", "Td", "2c"))
    t = gpu._tree
    assert int(getattr(t, "num_chance_nodes", 0)) > 0, "turn must have chance nodes"
    max_h = gpu._max_hands
    table = np.zeros((0, 2, max_h), dtype=np.float32)
    try:
        ladder = (50, 200, 800)
        expl = []
        for iters in ladder:
            cs = CSolver.from_table(t, table, 2, max_h, seed=1)
            cs.set_reach_mode(1)
            cs.run(iters, num_threads=1)
            tot, g0, g1 = cs.exploitability_2p()
            expl.append(tot)
            assert g0 >= -1e-6 and g1 >= -1e-6, f"NEGATIVE BR-gap g0={g0} g1={g1}"
            print(f"  G3 turn reach: iters={iters:>4} exploit={tot:9.3f} "
                  f"({100 * tot / POT_CHIPS:6.3f}% pot)")
        # monotone non-increasing AND a real drop (final < 20% of first)
        mono = all(expl[i + 1] <= expl[i] + 1e-9 for i in range(len(expl) - 1))
        dropped = expl[-1] < 0.2 * expl[0]
        ok = mono and dropped
        print(f"  G3 exploitability monotone-drop {expl[0]:.1f}->{expl[-1]:.1f}  "
              f"{'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        gpu._freed = True


def gate4_beats_sampling():
    """Reach strategy is FAR less exploitable than the prior sampling-vec one."""
    from es_resolver.c_solver import CSolver
    gpu, _ = _build(nh=6, board=("Ah", "Ks", "Td", "2c"))
    t = gpu._tree
    max_h = gpu._max_hands
    table = np.zeros((0, 2, max_h), dtype=np.float32)
    try:
        cs_r = CSolver.from_table(t, table, 2, max_h, seed=1)
        cs_r.set_reach_mode(1)
        cs_r.run(1600, num_threads=1)
        er, _, _ = cs_r.exploitability_2p()

        cs_s = CSolver.from_table(t, table, 2, max_h, seed=1)
        cs_s.set_cfr_mode(1)
        cs_s.set_chance_mode(1)
        cs_s.set_opp_mode(1)
        cs_s.set_vec_mode(1)
        cs_s.run(8000, num_threads=1)
        es, _, _ = cs_s.exploitability_2p()   # same validated BR computer

        ok = er < 0.25 * es   # reach at least 4x less exploitable
        print(f"  G4 reach exploit={er:.2f} ({100*er/POT_CHIPS:.3f}%) vs "
              f"sampling exploit={es:.2f} ({100*es/POT_CHIPS:.3f}%)  "
              f"{'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        gpu._freed = True


def main():
    print("Validating full-per-hand-reach vector CFR (LOCAL, no GPU/blueprint) ...")
    t0 = time.time()
    results = {
        "G1 determinism": gate1_determinism(),
        "G2 no-chance==scalar": gate2_no_chance_matches_scalar(),
        "G3 exploitability-drops": gate3_exploitability_drops(),
        "G4 beats-sampling": gate4_beats_sampling(),
    }
    print(f"\nelapsed {time.time() - t0:.1f}s")
    allok = all(results.values())
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("ALL GATES PASS" if allok else "GATES FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
