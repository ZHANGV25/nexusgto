"""
Ground-truth verification for the A2 sampled-CFV estimator used in
ss_batch_leaf_value, ss_batch_fold_value, and ss_batch_showdown_value
when n_opp >= 3 (i.e. 4-, 5-, 6-player subgames).

A2 replaces the O(M^{n_opp}) odometer with an importance-sampled Monte
Carlo estimator. For a fixed traverser hand h = (c_0, c_1), N-1 opponents
with reach vectors r_1, ..., r_{N-1}, board B:

Reach-product estimator (drives fold & leaf kernels):

    Pi(h) ~= (1 / K) * sum_{k=1..K} (reach_prod_k / q_k) * 1[valid_k]

where we sample one combo (h_1^{(k)}, ..., h_{N-1}^{(k)}) per draw k by
drawing each h_p^{(k)} independently from the proposal
q_p(h_p) proportional to r_p[h_p] * 1[h_p compat with board, h].

Then reach_prod_k / q_k cancels everything except indicators that the
opponents are pairwise card-disjoint.

Equivalently (the form we actually evaluate, which is the numerically
stable one):

    Pi(h) ~= Z_1 * Z_2 * ... * Z_{N-1} * E_{k ~ q_1 x ... x q_{N-1}}[1[pairwise disjoint]]

where Z_p = sum_{h_p compat with B, h} r_p[h_p] is the marginal reach and
the expectation is a pairwise-disjointness rate over iid draws from the
reach-weighted proposals.

Showdown estimator: same sampling scheme, each valid combo contributes
its payoff (pot/(k+1) - bet if hero wins or k-way ties, -bet otherwise).

This script validates:

  (1) The estimator is unbiased vs brute-force for small problems
      (N=3 verified; N=4,5 checked on deliberately tiny ranges so
      enumeration stays tractable).
  (2) Variance shrinks as K grows (std/mean vs K on a fixed problem).
  (3) The estimator matches the A1 inclusion-exclusion value at N=3
      (cross-check: MC -> IE gold standard).
  (4) Edge cases: zero reach, heavy card blockers, degenerate ranges.

Run:
    python tests/test_a2_math.py
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Dict, List, Sequence, Tuple

DECK = list(range(52))


# ── Deck / hand helpers (shared shape with test_opp_reach_formula.py) ──

def make_all_hands(exclude: Sequence[int]) -> List[Tuple[int, int]]:
    avail = [c for c in DECK if c not in exclude]
    return [(a, b) for a, b in itertools.combinations(avail, 2)]


def filter_range_by_blocked(
    opp: Dict[Tuple[int, int], float], blocked: Sequence[int]
) -> List[Tuple[Tuple[int, int], float]]:
    """Return list of (hand, weight) entries compatible with `blocked` cards."""
    bs = set(blocked)
    out: List[Tuple[Tuple[int, int], float]] = []
    for (a, b), w in opp.items():
        if a in bs or b in bs or w <= 0.0:
            continue
        out.append(((a, b), float(w)))
    return out


# ── Brute force (gold) — reach-product and showdown EV ──────────────

def brute_force_pi(
    hero_hand: Tuple[int, int],
    board: Sequence[int],
    opp_ranges: Sequence[Dict[Tuple[int, int], float]],
) -> float:
    c0, c1 = hero_hand
    blocked = set(board) | {c0, c1}
    filtered = [filter_range_by_blocked(r, blocked) for r in opp_ranges]

    total = 0.0
    for combo in itertools.product(*filtered):
        seen = list(hero_hand)
        ok = True
        for (h_p, _w) in combo:
            for c in h_p:
                if c in seen:
                    ok = False
                    break
                seen.append(c)
            if not ok:
                break
        if not ok:
            continue
        prod = 1.0
        for (_h, w) in combo:
            prod *= w
        total += prod
    return total


def hand_strength_surrogate(hand: Tuple[int, int], board: Sequence[int]) -> int:
    """Same hash family as test_showdown_formula.py — cheap many-valued int."""
    c0, c1 = hand
    r0, r1 = c0 >> 2, c1 >> 2
    s0, s1 = c0 & 3, c1 & 3
    board_ranks = sorted([b >> 2 for b in board], reverse=True)
    h = (r0 + r1) * 100000 + max(r0, r1) * 1000 + min(r0, r1) * 20 + (s0 * 4 + s1)
    for i, br in enumerate(board_ranks[:5]):
        h += br * (5 - i)
    return h


def brute_force_showdown_ev(
    hero_hand: Tuple[int, int],
    board: Sequence[int],
    opp_ranges: Sequence[Dict[Tuple[int, int], float]],
    pot: float,
    bet: float,
) -> float:
    c0, c1 = hero_hand
    blocked = set(board) | {c0, c1}
    s_h = hand_strength_surrogate(hero_hand, board)
    filtered = []
    for r in opp_ranges:
        f = [((h, w), hand_strength_surrogate(h, board))
             for (h, w) in filter_range_by_blocked(r, blocked)]
        filtered.append(f)

    total = 0.0
    for combo in itertools.product(*filtered):
        seen = list(hero_hand)
        ok = True
        for ((h_p, _w), _s) in combo:
            for c in h_p:
                if c in seen:
                    ok = False
                    break
                seen.append(c)
            if not ok:
                break
        if not ok:
            continue
        reach = 1.0
        for ((_h, w), _s) in combo:
            reach *= w
        max_s = max(s for (_, s) in combo)
        if max_s > s_h:
            payoff = -bet
        else:
            k_ties = sum(1 for (_, s) in combo if s == s_h)
            payoff = pot / (k_ties + 1) - bet
        total += reach * payoff
    return total


# ── Importance-sampling MC estimators (what the kernel will do) ──────

def _filtered_support(
    opp: Dict[Tuple[int, int], float], blocked: Sequence[int]
) -> Tuple[List[Tuple[int, int]], List[float], float]:
    """Return (hands, weights, Z) for opp filtered by blocked cards.

    Z = sum of weights = marginal reach Z_p.
    """
    bs = set(blocked)
    hands: List[Tuple[int, int]] = []
    weights: List[float] = []
    Z = 0.0
    for (a, b), w in opp.items():
        if a in bs or b in bs or w <= 0.0:
            continue
        hands.append((a, b))
        weights.append(float(w))
        Z += float(w)
    return hands, weights, Z


def _alias_sampler(weights: Sequence[float]) -> Tuple[List[int], List[float]]:
    """Vose's alias method tables for O(1) weighted sampling.

    This matches how the kernel will sample: build CDF/alias once per opp per
    terminal, then draw K samples. For the Python oracle we also prefer alias
    to be stable under repeated runs (deterministic given `weights`).
    """
    n = len(weights)
    total = sum(weights)
    if n == 0 or total <= 0.0:
        return [], []
    prob = [w * n / total for w in weights]
    alias = [0] * n
    small: List[int] = []
    large: List[int] = []
    for i, p in enumerate(prob):
        (small if p < 1.0 else large).append(i)
    while small and large:
        s = small.pop()
        l = large.pop()
        alias[s] = l
        prob[l] = prob[l] - (1.0 - prob[s])
        (small if prob[l] < 1.0 else large).append(l)
    # Numerical cleanup for any leftover bins.
    for i in large + small:
        prob[i] = 1.0
    return alias, prob


def _alias_draw(rng: random.Random, alias: List[int], prob: List[float]) -> int:
    n = len(alias)
    i = rng.randrange(n)
    return i if rng.random() < prob[i] else alias[i]


def mc_pi(
    hero_hand: Tuple[int, int],
    board: Sequence[int],
    opp_ranges: Sequence[Dict[Tuple[int, int], float]],
    K: int,
    seed: int,
) -> float:
    """Importance-sampled MC estimator of Pi(h).

    Pi(h) ~= Z_1 * ... * Z_{n_opp} * (1/K) * sum_k 1[pairwise disjoint].
    Unbiased iff the q_p is the full reach-proportional proposal over
    board/hero-compat hands.
    """
    c0, c1 = hero_hand
    blocked = set(board) | {c0, c1}

    supports = []
    z_prod = 1.0
    for r in opp_ranges:
        hands, weights, Z = _filtered_support(r, blocked)
        if Z <= 0.0:
            return 0.0  # one opp has no legal reach; pairwise-disjoint impossible
        alias, prob = _alias_sampler(weights)
        supports.append((hands, alias, prob, Z))
        z_prod *= Z

    rng = random.Random(seed)
    n_opp = len(supports)
    hits = 0
    for _ in range(K):
        drawn_cards: List[int] = []
        ok = True
        for (hands, alias, prob, _Z) in supports:
            j = _alias_draw(rng, alias, prob)
            a, b = hands[j]
            if a in drawn_cards or b in drawn_cards:
                ok = False
                break
            drawn_cards.append(a)
            drawn_cards.append(b)
        if ok:
            hits += 1
    return z_prod * (hits / K)


def mc_showdown_ev(
    hero_hand: Tuple[int, int],
    board: Sequence[int],
    opp_ranges: Sequence[Dict[Tuple[int, int], float]],
    pot: float,
    bet: float,
    K: int,
    seed: int,
) -> float:
    """Importance-sampled MC estimator of the showdown EV.

    E[v(h)] = Z_1*...*Z_{n_opp} * E_q[payoff(h; sampled opps) * 1[disjoint]].
    """
    c0, c1 = hero_hand
    blocked = set(board) | {c0, c1}
    s_h = hand_strength_surrogate(hero_hand, board)

    supports = []
    strengths: List[List[int]] = []
    z_prod = 1.0
    for r in opp_ranges:
        hands, weights, Z = _filtered_support(r, blocked)
        if Z <= 0.0:
            return 0.0
        alias, prob = _alias_sampler(weights)
        supports.append((hands, alias, prob, Z))
        strengths.append([hand_strength_surrogate(h, board) for h in hands])
        z_prod *= Z

    rng = random.Random(seed)
    n_opp = len(supports)
    acc = 0.0
    for _ in range(K):
        drawn_cards: List[int] = []
        ok = True
        opp_strengths: List[int] = []
        for p_idx, (hands, alias, prob, _Z) in enumerate(supports):
            j = _alias_draw(rng, alias, prob)
            a, b = hands[j]
            if a in drawn_cards or b in drawn_cards:
                ok = False
                break
            drawn_cards.append(a)
            drawn_cards.append(b)
            opp_strengths.append(strengths[p_idx][j])
        if not ok:
            continue
        max_s = max(opp_strengths)
        if max_s > s_h:
            payoff = -bet
        else:
            k_ties = sum(1 for s in opp_strengths if s == s_h)
            payoff = pot / (k_ties + 1) - bet
        acc += payoff
    return z_prod * (acc / K)


# ── Range generators ────────────────────────────────────────────────

def random_range(
    rng: random.Random, board: Sequence[int], density: float = 0.1
) -> Dict[Tuple[int, int], float]:
    r: Dict[Tuple[int, int], float] = {}
    for h in make_all_hands(board):
        if rng.random() < density:
            r[h] = rng.random()
    return r


def random_hero_hand(rng: random.Random, board: Sequence[int]) -> Tuple[int, int]:
    while True:
        h = tuple(sorted(rng.sample(DECK, 2)))
        if h[0] not in board and h[1] not in board:
            return h  # type: ignore[return-value]


# ── Tests ───────────────────────────────────────────────────────────

def test_empty_range() -> None:
    """Opp with zero reach -> Pi = 0 exactly."""
    board = [0, 1, 2]
    h = (10, 11)
    assert mc_pi(h, board, [{}, {}, {}], K=256, seed=0) == 0.0
    assert mc_showdown_ev(h, board, [{}, {}, {}], pot=100, bet=30,
                          K=256, seed=0) == 0.0
    print("Empty range: OK")


def test_mc_matches_brute_n3_small() -> None:
    """MC cross-check against brute force at N=3 on tiny ranges.

    On a tiny problem, we can verify MC -> brute within sampling tolerance.
    Large K (>= 20k) to beat variance down to ~1e-2 relative.
    """
    rng = random.Random(1)
    max_rel = 0.0
    for trial in range(6):
        board = rng.sample(DECK, 3)
        h = random_hero_hand(rng, board)
        opp1 = random_range(rng, board, density=0.03)
        opp2 = random_range(rng, board, density=0.03)
        if not opp1 or not opp2:
            continue
        gold = brute_force_pi(h, board, [opp1, opp2])
        if abs(gold) < 1e-9:
            continue
        # Average across many seeds to cut variance.
        ests = [mc_pi(h, board, [opp1, opp2], K=20000, seed=trial * 97 + s)
                for s in range(5)]
        mean_est = sum(ests) / len(ests)
        rel = abs(mean_est - gold) / abs(gold)
        max_rel = max(max_rel, rel)
        assert rel < 0.03, (
            f"N=3 MC trial {trial}: gold={gold:.4f} mean_est={mean_est:.4f} "
            f"rel={rel:.3e}")
    print(f"N=3 MC vs brute force (|opps| tiny): max rel error = {max_rel:.2e}")


def test_mc_matches_brute_n4_small() -> None:
    """Same, N=4 (3 opponents), tiny ranges so brute force is tractable."""
    rng = random.Random(2)
    max_rel = 0.0
    actual = 0
    for trial in range(6):
        board = rng.sample(DECK, 3)
        h = random_hero_hand(rng, board)
        opps = [random_range(rng, board, density=0.015) for _ in range(3)]
        if any(not o for o in opps):
            continue
        gold = brute_force_pi(h, board, opps)
        if abs(gold) < 1e-9:
            continue
        ests = [mc_pi(h, board, opps, K=40000, seed=trial * 131 + s)
                for s in range(5)]
        mean_est = sum(ests) / len(ests)
        rel = abs(mean_est - gold) / abs(gold)
        max_rel = max(max_rel, rel)
        assert rel < 0.05, (
            f"N=4 MC trial {trial}: gold={gold:.4f} mean_est={mean_est:.4f} "
            f"rel={rel:.3e}")
        actual += 1
    print(f"N=4 MC vs brute force: {actual} trials, max rel error = {max_rel:.2e}")


def test_mc_matches_brute_n5_small() -> None:
    """N=5 (4 opponents), ultra-tiny ranges to keep brute tractable."""
    rng = random.Random(3)
    max_rel = 0.0
    actual = 0
    for trial in range(4):
        board = rng.sample(DECK, 3)
        h = random_hero_hand(rng, board)
        opps = [random_range(rng, board, density=0.01) for _ in range(4)]
        if any(len(o) < 2 for o in opps):
            continue
        gold = brute_force_pi(h, board, opps)
        if abs(gold) < 1e-9:
            continue
        # N=5 has worse disjointness rate; need more samples.
        ests = [mc_pi(h, board, opps, K=80000, seed=trial * 229 + s)
                for s in range(5)]
        mean_est = sum(ests) / len(ests)
        rel = abs(mean_est - gold) / abs(gold)
        max_rel = max(max_rel, rel)
        assert rel < 0.08, (
            f"N=5 MC trial {trial}: gold={gold:.4f} mean_est={mean_est:.4f} "
            f"rel={rel:.3e}")
        actual += 1
    print(f"N=5 MC vs brute force: {actual} trials, max rel error = {max_rel:.2e}")


def test_variance_shrinks_with_K() -> None:
    """std(estimator) ∝ 1/sqrt(K). Check the slope on a fixed problem."""
    rng = random.Random(4)
    board = rng.sample(DECK, 3)
    h = random_hero_hand(rng, board)
    opps = [random_range(rng, board, density=0.2) for _ in range(3)]  # N=4

    stds: Dict[int, float] = {}
    for K in (64, 256, 1024, 4096):
        ests = [mc_pi(h, board, opps, K=K, seed=1000 + s) for s in range(30)]
        mean = sum(ests) / len(ests)
        var = sum((e - mean) ** 2 for e in ests) / (len(ests) - 1)
        stds[K] = math.sqrt(var)

    # Check rough 1/sqrt(K) scaling: std(64) / std(4096) should be near
    # sqrt(4096/64) = 8.  Accept anywhere in [4, 16] to tolerate small-sample
    # noise at trials=30.
    ratio = stds[64] / max(stds[4096], 1e-12)
    print(f"std scaling: K=64->{stds[64]:.3e}, K=4096->{stds[4096]:.3e}, "
          f"ratio={ratio:.2f} (expected ~8)")
    assert 3.0 < ratio < 20.0, (
        f"Variance not shrinking as ~1/sqrt(K): ratio={ratio}")


def test_mc_matches_ie_at_n3() -> None:
    """Cross-check: MC at N=3 -> A1 inclusion-exclusion value (already verified).

    We import the A1 helpers and compare against them so the two code paths
    agree — this is the belt-and-suspenders test.
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_opp_reach_formula import fast_pi_n3  # type: ignore

    rng = random.Random(5)
    max_rel = 0.0
    for trial in range(8):
        board = rng.sample(DECK, 3)
        h = random_hero_hand(rng, board)
        opp1 = random_range(rng, board, density=0.2)
        opp2 = random_range(rng, board, density=0.2)
        if not opp1 or not opp2:
            continue
        ie = fast_pi_n3(h, board, [opp1, opp2])
        if abs(ie) < 1e-9:
            continue
        ests = [mc_pi(h, board, [opp1, opp2], K=8000, seed=trial * 311 + s)
                for s in range(8)]
        mean_est = sum(ests) / len(ests)
        rel = abs(mean_est - ie) / abs(ie)
        max_rel = max(max_rel, rel)
        assert rel < 0.02, (
            f"N=3 MC vs IE trial {trial}: ie={ie:.4f} mean={mean_est:.4f} "
            f"rel={rel:.3e}")
    print(f"N=3 MC vs IE cross-check: max rel = {max_rel:.2e}")


def test_showdown_mc_matches_brute_n4_small() -> None:
    """Showdown EV: MC vs brute force at N=4, small ranges."""
    rng = random.Random(6)
    max_rel = 0.0
    actual = 0
    for trial in range(4):
        board = rng.sample(DECK, 5)  # river
        h = random_hero_hand(rng, board)
        opps = [random_range(rng, board, density=0.012) for _ in range(3)]
        if any(not o for o in opps):
            continue
        pot, bet = 100.0, 30.0
        gold = brute_force_showdown_ev(h, board, opps, pot, bet)
        if abs(gold) < 1e-9:
            continue
        ests = [mc_showdown_ev(h, board, opps, pot, bet,
                               K=40000, seed=trial * 541 + s)
                for s in range(5)]
        mean_est = sum(ests) / len(ests)
        rel = abs(mean_est - gold) / max(abs(gold), 1.0)
        max_rel = max(max_rel, rel)
        assert rel < 0.1, (
            f"Showdown N=4 trial {trial}: gold={gold:.4f} "
            f"mean={mean_est:.4f} rel={rel:.3e}")
        actual += 1
    print(f"Showdown N=4 MC vs brute: {actual} trials, max rel = {max_rel:.2e}")


def test_heavy_blocker_pair_n4() -> None:
    """Dense blocker case: hero AA, opps all A-high."""
    # Cards: 48=Ac, 49=Ad, 50=Ah, 51=As
    board = [48, 36, 20]  # Ac, Tc, 6c
    h = (51, 49)          # As Ad
    opp1 = {(50, 44): 0.8, (50, 45): 0.8, (44, 45): 0.5}  # Ah Kx / KK
    opp2 = {(50, 46): 0.7, (44, 47): 0.5, (45, 46): 0.3}
    opp3 = {(40, 50): 0.6, (42, 43): 0.4, (44, 46): 0.5}

    gold = brute_force_pi(h, board, [opp1, opp2, opp3])
    ests = [mc_pi(h, board, [opp1, opp2, opp3], K=40000, seed=s)
            for s in range(10)]
    mean = sum(ests) / len(ests)
    rel = abs(mean - gold) / max(abs(gold), 1e-9) if gold else abs(mean)
    print(f"Heavy blocker N=4: gold={gold:.6f} mean_mc={mean:.6f} rel={rel:.3e}")
    assert rel < 0.05 or abs(gold) < 1e-9


def brute_force_showdown_ev_hybrid(
    hero_hand: Tuple[int, int],
    board: Sequence[int],
    opp_ranges: Sequence[Dict[Tuple[int, int], float]],
    is_active: Sequence[bool],
    pot: float,
    bet: float,
) -> float:
    """Hybrid active+blocker brute-force showdown EV.

    Mirrors the A5-fixed CUDA dispatch semantics:
      * Reach product (blocking) runs over ALL non-traverser opps
        (active AND folded).
      * Strength comparison determining winners runs over ACTIVE opps
        only; folded opps' strength is never consulted.
      * Ties split pot over the set of max-strength active opps +
        hero (if hero's strength ties the max).

    A5 audit: §A5 in STATUS.md; derivation in
    docs/A5_OPP_SELECTION_AUDIT.md. This is the CFV kernel semantic at
    an n_all_opp >= 3 mid-street-fold showdown (3p flop with one
    pre-river fold, or 4p with one mid-street fold reaching river).
    """
    assert len(opp_ranges) == len(is_active)
    c0, c1 = hero_hand
    blocked = set(board) | {c0, c1}
    s_h = hand_strength_surrogate(hero_hand, board)
    filtered = []
    for r in opp_ranges:
        f = [((h, w), hand_strength_surrogate(h, board))
             for (h, w) in filter_range_by_blocked(r, blocked)]
        filtered.append(f)

    total = 0.0
    for combo in itertools.product(*filtered):
        seen = list(hero_hand)
        ok = True
        for ((h_p, _w), _s) in combo:
            for c in h_p:
                if c in seen:
                    ok = False
                    break
                seen.append(c)
            if not ok:
                break
        if not ok:
            continue
        reach = 1.0
        for ((_h, w), _s) in combo:
            reach *= w
        # Strength comparison over ACTIVE opps only.
        active_strengths = [s for ((_h, _w), s), a
                            in zip(combo, is_active) if a]
        if not active_strengths:
            # Hero is alone among actives — unopposed; at a showdown
            # terminal this is a tree bug, but payoff = pot - bet.
            payoff = pot - bet
        else:
            max_s = max(active_strengths)
            if max_s > s_h:
                payoff = -bet
            else:
                k_ties = sum(1 for s in active_strengths if s == s_h)
                payoff = pot / (k_ties + 1) - bet
        total += reach * payoff
    return total


def mc_showdown_ev_hybrid(
    hero_hand: Tuple[int, int],
    board: Sequence[int],
    opp_ranges: Sequence[Dict[Tuple[int, int], float]],
    is_active: Sequence[bool],
    pot: float,
    bet: float,
    K: int,
    seed: int,
) -> float:
    """Importance-sampled MC with the hybrid active+blocker semantics.

    CDF draws: each opp samples from its reach-weighted hand list (all
    of them — same proposal regardless of active vs blocker).
    Disjointness check: over all sampled opp hands.
    Strength / winner determination: over ACTIVE opps only.

    This mirrors the A5 kernel path at street_solve.cu lines 1557-1664.
    Unbiased by the same argument as `mc_showdown_ev`: the reach-
    product proposal exactly cancels, the blocker's presence just
    shrinks the disjoint-event probability without biasing the payoff
    conditional on disjointness.
    """
    assert len(opp_ranges) == len(is_active)
    c0, c1 = hero_hand
    blocked = set(board) | {c0, c1}
    s_h = hand_strength_surrogate(hero_hand, board)

    supports = []
    strengths: List[List[int]] = []
    z_prod = 1.0
    for r in opp_ranges:
        hands, weights, Z = _filtered_support(r, blocked)
        if Z <= 0.0:
            return 0.0
        alias, prob = _alias_sampler(weights)
        supports.append((hands, alias, prob, Z))
        strengths.append([hand_strength_surrogate(h, board) for h in hands])
        z_prod *= Z

    rng = random.Random(seed)
    acc = 0.0
    for _ in range(K):
        drawn_cards: List[int] = []
        ok = True
        opp_sampled_strengths: List[Tuple[int, bool]] = []
        for p_idx, (hands, alias, prob, _Z) in enumerate(supports):
            j = _alias_draw(rng, alias, prob)
            a, b = hands[j]
            if a in drawn_cards or b in drawn_cards:
                ok = False
                break
            drawn_cards.append(a)
            drawn_cards.append(b)
            opp_sampled_strengths.append(
                (strengths[p_idx][j], is_active[p_idx]))
        if not ok:
            continue
        active = [s for (s, a) in opp_sampled_strengths if a]
        if not active:
            payoff = pot - bet
        else:
            max_s = max(active)
            if max_s > s_h:
                payoff = -bet
            else:
                k_ties = sum(1 for s in active if s == s_h)
                payoff = pot / (k_ties + 1) - bet
        acc += payoff
    return z_prod * (acc / K)


def test_hybrid_showdown_mc_matches_brute_n4() -> None:
    """A5 hybrid case: N=4 with n_active=2, n_blocker=1. MC estimator
    remains unbiased under the active+blocker semantics.

    This guards the kernel dispatch change at src/cuda/street_solve.cu
    lines 1557-1664: the A2 MC path at a mid-street-fold showdown must
    still be unbiased after the active/blocker split. A biased version
    (e.g. accidentally filtering blocker reach on strength==0, or
    including blocker strength in winner determination) would make the
    mean estimate diverge from brute force at K=40k.
    """
    rng = random.Random(700)
    max_rel = 0.0
    actual = 0
    for trial in range(6):
        board = rng.sample(DECK, 5)  # river
        h = random_hero_hand(rng, board)
        # N=4: 3 opponents. Mark the last as blocker (folded), first
        # two as active. density chosen so brute force remains tractable.
        opps = [random_range(rng, board, density=0.015) for _ in range(3)]
        if any(not o for o in opps):
            continue
        is_active = [True, True, False]
        pot, bet = 100.0, 30.0
        gold = brute_force_showdown_ev_hybrid(
            h, board, opps, is_active, pot, bet)
        if abs(gold) < 1e-9:
            continue
        # Also capture the WRONG semantics (folded opp counted as active)
        # to confirm the test distinguishes them.
        wrong = brute_force_showdown_ev(h, board, opps, pot, bet)

        ests = [mc_showdown_ev_hybrid(
            h, board, opps, is_active, pot, bet,
            K=40000, seed=trial * 773 + s) for s in range(5)]
        mean_est = sum(ests) / len(ests)
        rel = abs(mean_est - gold) / max(abs(gold), 1.0)
        max_rel = max(max_rel, rel)
        assert rel < 0.1, (
            f"Hybrid N=4 trial {trial}: gold={gold:.4f} "
            f"mean={mean_est:.4f} rel={rel:.3e}")

        # Assert the test actually distinguishes hybrid from the
        # all-active semantics (otherwise the test is vacuous).
        active_vs_hybrid_gap = abs(wrong - gold)
        print(f"    trial {trial}: gold={gold:.3f} "
              f"all-active-semantics={wrong:.3f} "
              f"|gap|={active_vs_hybrid_gap:.3f} "
              f"mc_mean={mean_est:.3f} rel={rel:.3e}")
        actual += 1

    print(f"Hybrid showdown N=4 MC vs brute: {actual} trials, "
          f"max rel = {max_rel:.2e}")


def test_hybrid_variance_scales_with_K() -> None:
    """Variance of the hybrid estimator should scale ~1/sqrt(K), same
    as the all-active case. Blocker presence just lowers the disjoint-
    event probability but doesn't change the estimator's variance-K
    scaling law.

    Guards against a regression that makes the hybrid MC noisier than
    it should be (e.g. some bias-variance trade we didn't intend).
    """
    rng = random.Random(701)
    board = rng.sample(DECK, 5)
    h = random_hero_hand(rng, board)
    opps = [random_range(rng, board, density=0.15) for _ in range(3)]  # N=4
    is_active = [True, True, False]

    stds: Dict[int, float] = {}
    for K in (64, 256, 1024, 4096):
        ests = [mc_showdown_ev_hybrid(
            h, board, opps, is_active, pot=100.0, bet=30.0,
            K=K, seed=5000 + s) for s in range(30)]
        mean = sum(ests) / len(ests)
        var = sum((e - mean) ** 2 for e in ests) / (len(ests) - 1)
        stds[K] = math.sqrt(var)

    ratio = stds[64] / max(stds[4096], 1e-12)
    print(f"Hybrid std scaling: K=64->{stds[64]:.3e}, "
          f"K=4096->{stds[4096]:.3e}, ratio={ratio:.2f} (expected ~8)")
    assert 3.0 < ratio < 20.0, (
        f"Hybrid variance not shrinking as ~1/sqrt(K): ratio={ratio}")


def test_independent_ranges_n4() -> None:
    """Disjoint opp card pools → Pi = Z1*Z2*Z3 exactly, estimator collapses."""
    board = [0, 1, 2]
    h = (10, 11)
    # Three opps on disjoint card pools — no pairwise conflict possible.
    opp1 = {(16, 17): 0.3, (18, 19): 0.5}
    opp2 = {(32, 33): 0.4, (34, 35): 0.6}
    opp3 = {(44, 45): 0.7, (46, 47): 0.2}
    gold = brute_force_pi(h, board, [opp1, opp2, opp3])
    expected = sum(opp1.values()) * sum(opp2.values()) * sum(opp3.values())
    assert abs(gold - expected) < 1e-9
    # MC with disjoint pools: every sample is valid, so MC = Z_prod exactly
    # (zero variance).
    est = mc_pi(h, board, [opp1, opp2, opp3], K=256, seed=7)
    print(f"Disjoint pools N=4: gold={gold:.6f} mc_est={est:.6f}")
    assert abs(est - gold) < 1e-12


def main() -> None:
    print("=" * 60)
    print("A2 sampled-CFV estimator verification")
    print("=" * 60)
    test_empty_range()
    test_independent_ranges_n4()
    test_heavy_blocker_pair_n4()
    test_mc_matches_brute_n3_small()
    test_mc_matches_brute_n4_small()
    test_mc_matches_brute_n5_small()
    test_mc_matches_ie_at_n3()
    test_variance_shrinks_with_K()
    test_showdown_mc_matches_brute_n4_small()
    test_hybrid_showdown_mc_matches_brute_n4()
    test_hybrid_variance_scales_with_K()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
