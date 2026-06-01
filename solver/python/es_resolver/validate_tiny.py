"""Validate the ES solver by BEST-RESPONSE GAP → 0 on a tiny ZERO-SUM tree
(guardrail #4). A buggy solver gives a wrong T — prove correctness first.

DESIGN (advisor-corrected twice, 2026-06-01):
  - The game must be GENUINELY zero-sum. A *separable* leaf table (value depends
    only on each player's own hand) is NOT zero-sum even after negating one
    column: payoff(i,j)=t0[i]-t0[j] ≠ 0 for i≠j. So we use a JOINT oracle: each
    hand has a scalar strength; leaf payoff to p0 = +pot if s(h0)>s(h1), -pot if
    <, 0 if tie. Zero-sum by construction, deterministic in (h0,h1), no eval
    noise. The ES interface already passes `sampled_hands`, so the oracle reads
    both hands.
  - Correctness criterion = BEST-RESPONSE GAP, not strategy-match (Nash
    strategies aren't unique; only value is). exploitability =
    Σ_p (BR_p - achieved_p) → 0 within MC noise at convergence.
  - BR tracks the OPPONENT-HAND REACH as a VECTOR (the joint leaf depends on the
    opponent's hand, so a scalar mean-action weight is insufficient).
  - Deterministic oracle → residual gap reflects convergence + any bug, not
    leaf-sampling variance.

Run: python -m es_resolver.validate_tiny
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from street_solver_gpu import StreetSolverGPU  # noqa: E402
from solver import card_to_int  # noqa: E402
from es_resolver.es_solver import (  # noqa: E402
    ESSolver, snapshot_tree, PlainTree, PlainNode,
    T_DECISION, T_FOLD, T_SHOWDOWN, T_LEAF,
)


# ─── Joint zero-sum oracle ───────────────────────────────────────────────────
# strength[player][hand_idx] -> scalar. Leaf payoff to p0 = sign(s0 - s1) * pot.
# Card conflicts between the two sampled hands are impossible here because the
# ES samples conflict-free hands; if a conflict ever slipped in we'd return 0.

def make_joint_oracle(tree: PlainTree, strength):
    def fn(leaf_idx, traverser, hand, sampled_hands, leaf_node):
        h0 = sampled_hands[0]
        h1 = sampled_hands[1]
        if h0 < 0 or h1 < 0:
            return 0.0
        s0 = strength[0][h0]
        s1 = strength[1][h1]
        pot = float(leaf_node.pot)
        my_bet = leaf_node.bets[traverser]
        # winner gets pot minus own bet; loser loses own bet; tie splits.
        if s0 == s1:
            share = pot / 2.0
            return share - float(my_bet)
        p0_wins = s0 > s1
        traverser_wins = (traverser == 0 and p0_wins) or (traverser == 1 and not p0_wins)
        if traverser_wins:
            return pot - float(my_bet)
        return -float(my_bet)
    return fn


# ─── Reach-aware best response + achieved value (vector over opponent hands) ──

def _hand_cards(tree, player, hidx):
    c0, c1, _ = tree.player_hands[player][hidx]
    return c0, c1


def _terminal_value_matrix(tree, node, strength):
    """[nh0, nh1] payoff-to-p0 matrix at a terminal, conflict-aware.

    Conflicting (i,j) hand pairs are marked NaN (excluded from reach sums)."""
    nh0, nh1 = tree.num_hands[0], tree.num_hands[1]
    M = np.zeros((nh0, nh1))
    conflict = np.zeros((nh0, nh1), dtype=bool)
    pot = float(node.pot)
    bet0 = node.bets[0]
    if node.type == T_FOLD:
        # last remaining player wins regardless of strength
        winner = node.player
        for i in range(nh0):
            for j in range(nh1):
                ci = _hand_cards(tree, 0, i)
                cj = _hand_cards(tree, 1, j)
                if set(ci) & set(cj):
                    conflict[i, j] = True
                    continue
                # value to p0
                M[i, j] = (pot - bet0) if winner == 0 else -float(bet0)
        return M, conflict
    # LEAF: strength showdown
    for i in range(nh0):
        si = strength[0][i]
        ci = _hand_cards(tree, 0, i)
        for j in range(nh1):
            cj = _hand_cards(tree, 1, j)
            if set(ci) & set(cj):
                conflict[i, j] = True
                continue
            sj = strength[1][j]
            if si == sj:
                M[i, j] = pot / 2.0 - bet0
            elif si > sj:
                M[i, j] = pot - bet0
            else:
                M[i, j] = -float(bet0)
    return M, conflict


def _value_under_profile(tree, sigma, strength, br):
    """BR value vector for `br` AND achieved value vector for `br` under σ̄,
    both as per-(br hand) vectors, integrating the opponent's hand reach.

    Returns (br_value[nh_br], achieved_value[nh_br]).
    """
    opp = 1 - br
    nh_br = tree.num_hands[br]
    nh_opp = tree.num_hands[opp]

    # Precompute, per br hand i, the conflict-adjusted opponent prior mass:
    #   norm[i] = Σ_{j : j not conflicting with i} prior[j]
    # Reach is carried UNNORMALIZED through the tree (advisor 2026-06-01); we
    # divide ONCE at the top by norm[i]. The conflict set for hand i is
    # path-independent, so this single normalizer is exact and handles
    # card-blocking. Returns UNNORMALIZED accumulators; caller divides by norm.
    def walk(ni, opp_reach):
        """opp_reach: [nh_opp] UNNORMALIZED reach over opponent hands at this
        node (uniform prior × Π opponent σ along the path; never renormalized).
        Returns (br_acc[nh_br], ach_acc[nh_br]) — unnormalized reach-weighted
        value sums; the caller divides each by the conflict-adjusted norm."""
        node = tree.nodes[ni]
        if node.type in (T_LEAF, T_FOLD):
            M, conflict = _terminal_value_matrix(tree, node, strength)
            Mbr = M if br == 0 else -M.T          # [nh_br, nh_opp] payoff to br
            cf = conflict if br == 0 else conflict.T
            val = np.zeros(nh_br)
            for i in range(nh_br):
                wi = opp_reach.copy()
                wi[cf[i]] = 0.0
                val[i] = (Mbr[i] * wi).sum()       # NO /tot — divide once at top
            return val, val.copy()
        if node.type == T_SHOWDOWN:
            return np.zeros(nh_br), np.zeros(nh_br)
        acting = node.player
        na = node.num_children
        if acting == br:
            # BR: per hand pick the max-value action. achieved: play σ̄.
            # Both are over UNNORMALIZED reach-weighted sums, so the argmax is
            # invariant to the (positive) per-hand normalizer applied at the top.
            childs = [walk(node.children[a], opp_reach) for a in range(na)]
            br_children = np.stack([c[0] for c in childs], axis=0)   # [na, nh_br]
            ach_children = np.stack([c[1] for c in childs], axis=0)
            br_val = br_children.max(axis=0)
            sig = sigma.get(ni)
            if sig is None:
                sig = np.full((nh_br, na), 1.0 / na)
            ach_val = (sig * ach_children.T).sum(axis=1)
            return br_val, ach_val
        else:
            # opponent node: partition opp_reach by opp's σ̄ PER OPP HAND.
            sig = sigma.get(ni)
            if sig is None:
                sig = np.full((nh_opp, na), 1.0 / na)
            br_acc = np.zeros(nh_br)
            ach_acc = np.zeros(nh_br)
            for a in range(na):
                child_reach = opp_reach * sig[:, a]   # unnormalized partition
                bv, av = walk(node.children[a], child_reach)
                br_acc += bv
                ach_acc += av
            return br_acc, ach_acc

    opp_reach0 = np.ones(nh_opp)  # uniform UNNORMALIZED prior over opp hands
    br_acc, ach_acc = walk(0, opp_reach0)
    # conflict-adjusted per-(br hand) normalizer
    norm = np.zeros(nh_br)
    for i in range(nh_br):
        ci = set(_hand_cards(tree, br, i))
        tot = 0.0
        for j in range(nh_opp):
            cj = set(_hand_cards(tree, opp, j))
            if not (ci & cj):
                tot += 1.0  # uniform prior weight
        norm[i] = tot if tot > 0 else 1.0
    return br_acc / norm, ach_acc / norm


def exploitability(tree, sigma, strength):
    """Σ_p (BR_p - achieved_p), averaged over each player's uniform hand prior.
    → 0 within MC noise at Nash."""
    br0, ach0 = _value_under_profile(tree, sigma, strength, 0)
    br1, ach1 = _value_under_profile(tree, sigma, strength, 1)
    gap0 = (br0 - ach0).mean()
    gap1 = (br1 - ach1).mean()
    return float(gap0 + gap1), float(gap0), float(gap1)


def _build_real_tree(n_players, board_strs, leaf_seed=0, nh=4):
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
    tree = snapshot_tree(solver._tree, ranges)
    rng = np.random.default_rng(leaf_seed)
    # distinct integer strengths per hand (no ties unless across players equal);
    # allow some cross-player equal strengths to exercise the tie branch.
    strength = [rng.integers(0, 10, size=tree.num_hands[p]).astype(float)
                for p in range(n_players)]
    solver._freed = True  # native ss_free_tree() hangs on this box; skip it
    return tree, strength


def main():
    print("=" * 72)
    print("VALIDATION: ES best-response gap -> 0 on a tiny 2p ZERO-SUM tree")
    print("=" * 72)
    tree, strength = _build_real_tree(2, ["Ah", "Kh", "2c"], leaf_seed=7, nh=4)
    oracle = make_joint_oracle(tree, strength)

    # ── METRIC INVARIANT CHECK (advisor 2026-06-01): on ANY fixed profile the
    # zero-sum game's achieved values must cancel: ach0 + ach1 ≈ 0. We check it
    # on uniform σ̄ (no ES run needed). If this fails, the METRIC is broken and
    # any plateau is a metric artifact. Also assert values lie in [-stack,stack].
    br0u, ach0u = _value_under_profile(tree, {}, strength, 0)
    br1u, ach1u = _value_under_profile(tree, {}, strength, 1)
    ach_sum = float(ach0u.mean() + ach1u.mean())
    stack = tree.effective_stack
    print(f"\n tree: nodes={len(tree.nodes)} decisions={len(tree.decision_node_indices)} nh=4")
    print(f" strengths p0={strength[0].tolist()} p1={strength[1].tolist()}")
    print(f" [INVARIANT] uniform-σ̄ achieved-sum (must be ~0): {ach_sum:.4f}")
    in_bounds = (np.all(np.abs(ach0u) <= stack + 1) and np.all(np.abs(ach1u) <= stack + 1))
    print(f" [INVARIANT] achieved values within [-stack,stack]={stack}: {in_bounds}")
    if abs(ach_sum) > 1.0 or not in_bounds:
        print(" >>> METRIC INVARIANT FAILED — fix the BR metric before trusting any gap.")
    else:
        print(" >>> METRIC INVARIANT OK — gaps below are trustworthy.")
    uni_gap, _, _ = exploitability(tree, {}, strength)
    print(f" baseline (uniform sigma) BR-gap = {uni_gap:.3f}")

    print(f"\n {'iters':>8} {'traversals':>11} {'rollouts':>10} {'r/t':>5} "
          f"{'BR-gap':>10} {'(gap0':>8} {'gap1)':>8}")
    for iters in (2_000, 10_000, 40_000, 120_000, 300_000):
        es = ESSolver(tree, oracle, seed=1, linear_cfr=True)
        es.run(iters)
        sigma = {ni: es.avg_strategy(ni) for ni in es.info.strat_sum
                 if es.avg_strategy(ni) is not None}
        gap, g0, g1 = exploitability(tree, sigma, strength)
        rpt = es.stats.leaf_rollouts / max(es.stats.traversals, 1)
        print(f" {iters:>8,} {es.stats.traversals:>11,} {es.stats.leaf_rollouts:>10,} "
              f"{rpt:>5.2f} {gap:>10.4f} {g0:>8.3f} {g1:>8.3f}")

    print("\n VERDICT: PASS if BR-gap falls toward ~0 as iters grow.")
    print(" A high plateau = ES bug (check strat_sum reach-weighting first).")


if __name__ == "__main__":
    main()
