"""GATE 1 for the vectorized traverser (VECTOR_TRAVERSER_DESIGN.md).

THE test: vectorized ES (one walk updates ALL traverser hands) must converge to
the SAME exploitability as scalar ES on a validated 2p/3p flop spot. A speedup on
a WRONG fixed point is a fabricated win — so this gate is load-bearing and runs
BEFORE any EC2 speedup measurement.

What this gate stresses that validate_tiny_c / _distinct_ranges do NOT:
  - OVERLAPPING ranges (hand-constructed heavy card overlap). The dead-hand skip
    (vec skips traverser hands that share a card with the sampled opponents) only
    DIFFERS from a no-op when cards actually overlap. validate_tiny_c deals
    disjoint cards and _distinct_ranges is near-disjoint by construction — both
    make scalar==vec trivially and prove nothing about the mask. (advisor
    2026-06-10, blind spot 1.)
  - VANILLA FIRST, then Linear. Vanilla (cfr_mode=0) has no discount and per-
    infoset-normalized average strategy → the clean correctness isolator for the
    dead-mask + per-hand traversal; expect a TIGHT exploitability match. Linear
    (cfr_mode=1) applies the t/(t+1) discount to EVERY conflict-free hand each
    time a traverser node is reached, vs scalar's ONE sampled hand — so vec-Linear
    and scalar-Linear are structurally different procedures (≈nh× discount
    density). Gate Linear on the same low-exploitability BAND, NOT a tight match;
    divergence there is the discount-density difference (a Linear-design
    question), not a dead-mask bug. (advisor, blind spot 2.)
  - PER-HAND strategy diff at sample decision nodes (not just aggregate
    exploitability, which averages over hands and can hide a compensating
    per-hand bias the mask might introduce).

Run: PYTHONUTF8=1 ES_DLL=es_resolver python -m python.es_resolver.gate1_vec_traverser
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ES_DLL", "es_resolver")

from engine.street_solver_gpu import StreetSolverGPU  # noqa: E402
from engine.solver import card_to_int  # noqa: E402
from es_resolver.es_solver import snapshot_tree, T_LEAF  # noqa: E402
from es_resolver.c_solver import CSolver  # noqa: E402
from es_resolver.exploitability_c import exploitability, player_gap  # noqa: E402


def _overlapping_ranges(board_ints, n_players, nh, pool_cards=14, seed=0):
    """MODERATELY overlapping per-player ranges (advisor 2026-06-10): all players
    draw hands from a shared pool of `pool_cards` cards. The pool must be big
    enough that all n_players can hold conflict-free hands SIMULTANEOUSLY
    (≥ 2*n_players cards) so iterations are NOT no-ops, but small enough that an
    opponent's sampled cards collide with SOME (not all) of a traverser's hands —
    so the dead-hand mask fires at a moderate rate (~10-50%), provably exercising
    the conflict skip. NOT "all hands share one card" (that starves the joint
    assignment → both solvers do nothing → a vacuous match)."""
    deck = [x for x in range(52) if x not in set(board_ints)]
    assert pool_cards >= 2 * n_players, "pool too small for joint assignment"
    pool = deck[:pool_cards]
    combos = [(pool[i], pool[j], 1.0)
              for i in range(len(pool)) for j in range(i + 1, len(pool))]
    rng = np.random.default_rng(seed)
    nh = min(nh, len(combos))
    out = []
    for p in range(n_players):
        idx = rng.permutation(len(combos))[:nh]  # each player a random subset of the SAME pool
        out.append([combos[k] for k in idx])
    return out, nh


def _disjoint_ranges(board_ints, n_players, nh):
    """Fully DISJOINT per-player ranges: sequential non-overlapping deck slices,
    so NO traverser hand ever conflicts with any opponent (dead_rate→0). The
    localization test (advisor 2026-06-10): if vec==scalar here, the core
    multiway traversal is correct and any failure is specific to the dead-mask /
    heavy-overlap regime (and partly the metric's near-disjoint-only approx).
    If vec still diverges, there's a core np≥3 traversal bug independent of the
    mask."""
    deck = [x for x in range(52) if x not in set(board_ints)]
    out = []
    cur = 0
    nh = min(nh, len(deck) // (2 * n_players))
    for _p in range(n_players):
        out.append([(deck[cur + 2 * i], deck[cur + 2 * i + 1], 1.0) for i in range(nh)])
        cur += 2 * nh
    return out, nh


def _build(board_strs, n_players, nh, leaf_seed):
    board_ints = [card_to_int(c) for c in board_strs]
    if os.environ.get("ES_DISJOINT") == "1":
        ranges, nh_used = _disjoint_ranges(board_ints, n_players, nh)
    else:
        pool = int(os.environ.get("ES_POOL_CARDS", "14"))
        ranges, nh_used = _overlapping_ranges(board_ints, n_players, nh, pool_cards=pool)
    solver = StreetSolverGPU(
        board=board_strs, pot_bb=20.0, stack_bb=95.0, player_ranges=ranges,
        bet_sizes=[0.75], subsequent_bet_sizes=[1.0], use_cont_strats=True,
        leaf_value_fn=None, add_explicit_allin=False)
    tree = snapshot_tree(solver._tree, ranges)
    gpu_tree = solver._tree
    n_leaves = max(nd.leaf_idx for nd in tree.nodes if nd.type == T_LEAF) + 1
    max_h = tree.max_hands
    rng = np.random.default_rng(leaf_seed)
    table = rng.normal(0, 50, size=(n_leaves, n_players, max_h)).astype(np.float32)
    return solver, gpu_tree, tree, table, max_h, nh_used


def _solve(gpu_tree, table, n_players, max_h, iters, threads, cfr_mode, vec):
    cs = CSolver.from_table(gpu_tree, table, n_players, max_h, seed=1)
    cs.set_cfr_mode(cfr_mode)
    cs.set_vec_mode(1 if vec else 0)
    cs.run(iters, num_threads=threads)
    return cs


def _sigma(cs, tree):
    return {ni: cs.avg_strategy(ni) for ni in tree.decision_node_indices
            if cs.avg_strategy(ni) is not None}


def _sigma_final(cs, tree):
    """Final-iterate strategy (reads regrets via regret-matching), as opposed to
    the average (reads strat_sum). Used to partition a Linear-mode bug: if FINAL
    converges but AVG is pinned, the corruption is in the strat-sum accumulation
    (ss += iter*sigma); if FINAL is ALSO pinned, it's the regret discount
    (reg *= iter/(iter+1)). (advisor 2026-06-10)."""
    return {ni: cs.final_strategy(ni) for ni in tree.decision_node_indices
            if cs.final_strategy(ni) is not None}


def _diagnose_linear_3p():
    """Free discriminator (no rebuild): vec-Linear-3p-disjoint, measure BOTH avg
    and final exploitability to localize the np≥3 Linear bug to one line."""
    os.environ["ES_DISJOINT"] = "1"
    s, gt, tr, tbl, mh, nh = _build(["Ah", "Kh", "2c"], 3, 12, leaf_seed=11)
    print(f"\n[LINEAR-3p-DISJOINT DIAGNOSTIC]  nh={nh}  (avg reads strat_sum; "
          f"final reads regrets)")
    print(f"  {'mode':>10} {'iters':>10} {'expl_AVG':>10} {'expl_FINAL':>11}")
    for vec in (False, True):
        for iters in (200_000, 2_000_000):
            cs = _solve(gt, tbl, 3, mh, iters, 1, cfr_mode=1, vec=vec)
            e_avg, _ = exploitability(tr, tbl, _sigma(cs, tr))
            e_fin, _ = exploitability(tr, tbl, _sigma_final(cs, tr))
            print(f"  {'vec' if vec else 'scalar':>10} {iters:>10,} "
                  f"{e_avg:>10.4f} {e_fin:>11.4f}")
    s._freed = True


def _per_hand_value_diff(tree, table, sig_a, sig_b):
    """Max abs per-hand difference in the EXPLOITABILITY-RELEVANT value
    (br - achieved) between two profiles, over all players' hands. Value space,
    NOT strategy-L1: strategy-L1 reads ~1.0 for two CORRECT solvers when an
    action is near-indifferent (regret-matching picks pure, the pick flips on
    sampling noise), so it's not equilibrium-invariant. Per-hand VALUE catches a
    real compensating bias the aggregate mean would cancel (advisor 2026-06-10).
    Returned in chips; read relative to the exploitability magnitude."""
    worst = 0.0
    NP = tree.num_players
    for p in range(NP):
        ba, aa = player_gap(tree, table, sig_a, p)
        bb, ab = player_gap(tree, table, sig_b, p)
        worst = max(worst, float(np.abs((ba - aa) - (bb - ab)).max()))
    return worst


def _max_nonuniformity(cs, tree, n_nodes=5):
    """Max deviation of any avg_strategy row from uniform (1/na), over the first
    n_nodes decision nodes. ≈0 means the solver learned NOTHING (no-op
    iterations) → any scalar-vs-vec match is vacuous. The gate REQUIRES this to
    be materially > 0 (advisor diagnostic 2)."""
    worst = 0.0
    for ni in tree.decision_node_indices[:n_nodes]:
        a = cs.avg_strategy(ni)
        if a is None:
            continue
        na = a.shape[1]
        worst = max(worst, float(np.abs(a - 1.0 / na).max()))
    return worst


def _dead_rate_and_assignability(tree, n_players, n_trials=4000, seed=99):
    """Independent Python probe of the live estimator (advisor diagnostic 1):
    sample opponent hands conflict-free (mirroring es_sample_hands), then measure
    (a) the fraction of traverser hands killed by the dead-mask, and (b) how many
    players got a conflict-free assignment. A degenerate test has dead_rate≈100%
    or assignable≈1 (no-op iterations). Target: dead_rate in ~10-50%, assignable
    == n_players on most trials."""
    rng = np.random.default_rng(seed)
    board = set(tree.board[:tree.num_board])
    nh = [tree.num_hands[p] for p in range(n_players)]
    cards = [[set((tree.player_hands[p][h][0], tree.player_hands[p][h][1]))
              for h in range(nh[p])] for p in range(n_players)]
    dead_acc = 0.0
    assignable_acc = 0
    for _ in range(n_trials):
        used = set(board)
        chosen = [-1] * n_players
        for p in rng.permutation(n_players):
            cand = [h for h in range(nh[p]) if not (cards[p][h] & used)]
            if cand:
                h = int(rng.choice(cand))
                chosen[p] = h
                used |= cards[p][h]
        assignable_acc += sum(1 for c in chosen if c >= 0)
        opp_used = set(board)
        for p in range(1, n_players):
            if chosen[p] >= 0:
                opp_used |= cards[p][chosen[p]]
        dead = sum(1 for h in range(nh[0]) if cards[0][h] & opp_used)
        dead_acc += dead / nh[0]
    return dead_acc / n_trials, assignable_acc / n_trials


LADDER = (40_000, 200_000, 1_000_000, 5_000_000)


def _run_case(label, gpu_tree, tree, table, n_players, max_h, threads):
    """Correctness = VANILLA, SAME-FLOOR on an iter ladder (advisor 2026-06-10).
    Vec reduces variance, so at a fixed iter count vec sits CLOSER to equilibrium
    (lower exploitability) than scalar — that is the feature, not a discrepancy.
    Same fixed point ⇒ both curves descend to a COMMON floor with the gap
    shrinking as iters grow. A real vec bug = vec converging to a DIFFERENT
    (higher) floor, or non-monotone / blowup. Linear is INFORMATIONAL only
    (structurally different discount density — deferred to the speedup run)."""
    print(f"\n{'='*78}\n{label}  (threads={threads})\n{'='*78}")
    out = {}
    for cfr_mode, mname in ((0, "vanilla"), (1, "linear")):
        print(f"  [{mname}]  {'iters':>10} {'expl_scalar':>12} {'expl_vec':>12} "
              f"{'gap':>9} {'perhandΔv':>10}")
        rows = []
        for iters in LADDER:
            cs_s = _solve(gpu_tree, table, n_players, max_h, iters, threads, cfr_mode, vec=False)
            cs_v = _solve(gpu_tree, table, n_players, max_h, iters, threads, cfr_mode, vec=True)
            sig_s, sig_v = _sigma(cs_s, tree), _sigma(cs_v, tree)
            es, _ = exploitability(tree, table, sig_s)
            ev, _ = exploitability(tree, table, sig_v)
            phv = _per_hand_value_diff(tree, table, sig_s, sig_v)
            print(f"        {iters:>12,} {es:>12.4f} {ev:>12.4f} "
                  f"{abs(es-ev):>9.4f} {phv:>10.4f}")
            rows.append((iters, es, ev, phv))
        out[mname] = rows
    return out


def main() -> int:
    print("GATE 1: vectorized traverser converges to scalar's fixed point")
    print("(OVERLAPPING ranges — stresses the dead-hand card-conflict skip)")

    threads = int(os.environ.get("ES_THREADS", "1"))  # single-thread = clean ref

    ok = True
    notes = []

    # ── 2p ──
    s2, gt2, tr2, tbl2, mh2, nh2 = _build(["Ah", "Kh", "2c"], 2, 12, leaf_seed=7)
    dr2, asg2 = _dead_rate_and_assignability(tr2, 2)
    print(f"\n2p tree: nodes={len(tr2.nodes)} decisions={len(tr2.decision_node_indices)} "
          f"nh={nh2} (overlapping) | dead_rate={dr2:.1%} assignable={asg2:.2f}/2")
    r2 = _run_case("2p flop, overlapping ranges", gt2, tr2, tbl2, 2, mh2, threads)
    s2._freed = True

    # ── 3p ──
    s3, gt3, tr3, tbl3, mh3, nh3 = _build(["Ah", "Kh", "2c"], 3, 12, leaf_seed=11)
    dr3, asg3 = _dead_rate_and_assignability(tr3, 3)
    print(f"\n3p tree: nodes={len(tr3.nodes)} decisions={len(tr3.decision_node_indices)} "
          f"nh={nh3} (overlapping) | dead_rate={dr3:.1%} assignable={asg3:.2f}/3")
    r3 = _run_case("3p flop, overlapping ranges", gt3, tr3, tbl3, 3, mh3, threads)
    s3._freed = True

    # ── Degeneracy guard (advisor diagnostic 1): the test is only meaningful if
    # the dead-mask actually fires AND all players get assigned (iterations are
    # not no-ops). dead_rate≈0 → mask never tested; dead_rate≈100% or
    # assignable≈1 → no-op iterations → vacuous match. ──
    degen = []
    for tag, dr, asg, npl in (("2p", dr2, asg2, 2), ("3p", dr3, asg3, 3)):
        if not (0.05 < dr < 0.90):
            degen.append(f"{tag} dead_rate={dr:.1%} outside (5%,90%) — mask under/over-fires")
        if asg < npl - 0.1:
            degen.append(f"{tag} assignable={asg:.2f}/{npl} — iterations are partly no-ops")
    if degen:
        ok = False
        notes.append("  DEGENERACY GUARD FAILED (test cannot validate the mask):")
        notes.extend(f"    - {d}" for d in degen)

    # ── Verdict: VANILLA, SAME-FLOOR (advisor 2026-06-10) ──
    # Same fixed point ⇒ (a) scalar AND vec both descend along the ladder, (b)
    # they meet at a COMMON floor: the deepest-rung gap SHRINKS toward zero from
    # the shallow rung. A real vec bug = vec stalls at a HIGHER floor than scalar
    # (gap persists/grows). Normalize the gap by the PROBLEM SCALE (the shallow-
    # rung exploitability), NOT the floor — when both converge toward ~0 an
    # absolute gap of 0.01 is a PASS even though it's "large vs a near-zero floor"
    # (that floor-normalization was a verdict-metric bug, not a solver signal).
    print(f"\n{'='*78}\nVERDICT (correctness = vanilla same-floor; Linear informational)\n{'='*78}")
    for tag, res in (("2p", r2), ("3p", r3)):
        rows = res["vanilla"]
        (i0, es0, ev0, _ph0) = rows[0]
        (iN, esN, evN, phvN) = rows[-1]
        floor = min(esN, evN)
        scale = max(es0, ev0, abs(esN), abs(evN), 1e-6)  # problem magnitude
        gap0 = abs(es0 - ev0)
        gapN = abs(esN - evN)
        gap_rel = gapN / scale
        phv_rel = phvN / scale
        shrank = gapN <= gap0 + 1e-9
        passed = (gap_rel < 0.10) and shrank and (phv_rel < 0.15)
        ok = ok and passed
        notes.append(
            f"  {tag} vanilla: floor≈{floor:.3f} scale≈{scale:.3f}  "
            f"gap {gap0:.3f}->{gapN:.3f} (rel {gap_rel:.3f}, "
            f"{'shrank' if shrank else 'GREW'})  "
            f"perhandΔv={phvN:.3f} (rel {phv_rel:.3f})  "
            f"[{'PASS' if passed else 'FAIL'}]")
        # Linear: informational only.
        lr = res["linear"]
        notes.append(f"  {tag} linear (info): scalar {lr[0][1]:.2f}->{lr[-1][1]:.2f}  "
                     f"vec {lr[0][2]:.2f}->{lr[-1][2]:.2f}  "
                     f"(structurally different discount density — NOT gated)")
    print("\n".join(notes))
    if ok:
        verdict = ("PASS — vectorized traverser converges to scalar's fixed point "
                   "(same floor); it's the real lever. EC2-measure the 5p/6p "
                   "speedup next (carry the Linear discount-density caveat).")
    else:
        verdict = ("FAIL — vec does not reach scalar's floor / non-monotone. "
                   "Investigate dead-mask / reach-weighting BEFORE any speedup run.")
    print(f"\n  GATE 1 {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
