"""
Ground-truth verification for the N-opponent showdown value decomposition
used in the fast ss_batch_showdown_value kernel.

The traverser's expected value at a showdown node, given their hand h with
strength s_h, total pot P, and their bet b, is:

  v(h) = E_{opps} [ payoff(s_h; s_opps) * prod r_opp ]

where opponents are pairwise card-disjoint AND disjoint from h and board.

Payoff decomposition (cleaner than the raw if/else):

  payoff = -b * 1[any opp combo present]                 # traverser pays b in all cases
         + (P/(k+1)) * 1[hero wins or k-way ties]        # traverser collects pot / (k+1)

Equivalently:
  v(h) = -b * Pi(h)   +   P * W(h)

where:
  Pi(h) = sum over valid opp tuples of prod(r_opp)                 (reach S-table formula)
  W(h)  = sum over valid opp tuples of (1/(k+1)) * prod(r_opp)     (needs the decomposition below)
        with k = # of opps whose strength == s_h, among opps with strength <= s_h (i.e. none strictly greater)

For n_opp=1 the existing kernel already does O(M) — no change needed.
For n_opp=2 (3p river), W(h) decomposes as:

  W(h) = Pi_{<,<}(h) + (1/2) * [Pi_{=,<}(h) + Pi_{<,=}(h)] + (1/3) * Pi_{=,=}(h)

where Pi_{A,B}(h) enumerates opp1-hand at strength-relation A and opp2-hand at
strength-relation B (one of <, =, >; only < and = are needed because > is a hero
loss and contributes 0 to W, only to -b*Pi).

Each Pi_{A,B}(h) has the same inclusion-exclusion structure as Pi(h):
  Pi_{A,B}(h) = sum r_1[h1] r_2[h2] [h1 in set_A, h2 in set_B, all disjoint h,board,pairwise]
             = A'(h) - B1'(h) + B2'(h)

where the S_0, S_1, S_2 tables are computed on the SUBSET of each opp's range
filtered by the strength relation (<, =).

This script verifies the decomposition against brute-force enumeration on
random 3p river spots (with all 5 board cards dealt).

Run:
  python tests/test_showdown_formula.py
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, List, Tuple

DECK = list(range(52))


def make_hands(exclude: List[int]) -> List[Tuple[int, int]]:
    avail = [c for c in DECK if c not in exclude]
    return [(a, b) for a, b in itertools.combinations(avail, 2)]


def hand_strength_surrogate(hand: Tuple[int, int], board: List[int]) -> int:
    """Cheap surrogate strength for testing: hash rank-sum + tiebreakers.

    Any deterministic function that yields many values will do: the test targets
    the combinatorial decomposition, not the evaluator.
    """
    c0, c1 = hand
    r0, r1 = c0 >> 2, c1 >> 2
    s0, s1 = c0 & 3, c1 & 3
    board_ranks = sorted([b >> 2 for b in board], reverse=True)
    # Hash that depends on all cards; produces ~all distinct values
    h = (r0 + r1) * 100000 + max(r0, r1) * 1000 + min(r0, r1) * 20 + (s0 * 4 + s1)
    for i, br in enumerate(board_ranks[:5]):
        h += br * (5 - i)
    return h


def brute_force_showdown_ev(
    hero_hand: Tuple[int, int],
    board: List[int],
    opp_ranges: List[Dict[Tuple[int, int], float]],
    strength_fn,
    pot: float,
    bet: float,
) -> float:
    """Gold standard: enumerate all opp tuples, compute exact EV."""
    c0, c1 = hero_hand
    blocked_base = set(board) | {c0, c1}
    s_h = strength_fn(hero_hand, board)

    filtered = []
    for opp in opp_ranges:
        f = []
        for (a, b), w in opp.items():
            if a in blocked_base or b in blocked_base:
                continue
            f.append(((a, b), w, strength_fn((a, b), board)))
        filtered.append(f)

    total = 0.0
    for combo in itertools.product(*filtered):
        all_cards = list(hero_hand) + list(board)
        ok = True
        for ((a, b), _w, _s) in combo:
            if a in all_cards or b in all_cards:
                ok = False
                break
            all_cards.extend([a, b])
        if not ok:
            continue
        reach = 1.0
        for (_h, w, _s) in combo:
            reach *= w
        # Payoff
        max_opp_s = max(s for (_h, _w, s) in combo)
        if max_opp_s > s_h:
            payoff = -bet
        else:
            k_ties = sum(1 for (_h, _w, s) in combo if s == s_h)
            payoff = pot / (k_ties + 1) - bet
        total += reach * payoff
    return total


# --- Fast formula building blocks (mirrors the kernel design) ---

def precompute_tables(
    hands_weights: List[Tuple[Tuple[int, int], float]],
    board: List[int],
) -> Tuple[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """S0, S1[c], S2[(a,b)] for a filtered hand list (board-compat only)."""
    board_set = set(board)
    S2: Dict[Tuple[int, int], float] = {}
    S0 = 0.0
    S1: Dict[int, float] = {}
    for (a, b), w in hands_weights:
        if a in board_set or b in board_set:
            continue
        S0 += w
        S1[a] = S1.get(a, 0.0) + w
        S1[b] = S1.get(b, 0.0) + w
        key = (a, b) if a < b else (b, a)
        S2[key] = S2.get(key, 0.0) + w
    return S0, S1, S2


def single_opp_reach_ie(h: Tuple[int, int], S0: float, S1: Dict[int, float],
                        S2: Dict[Tuple[int, int], float]) -> float:
    c0, c1 = h
    key = (c0, c1) if c0 < c1 else (c1, c0)
    return S0 - S1.get(c0, 0.0) - S1.get(c1, 0.0) + S2.get(key, 0.0)


def pi_n3_given_tables(h: Tuple[int, int], board: List[int],
                       tables_1, tables_2,
                       hands_1: List[Tuple[Tuple[int, int], float]]) -> float:
    """Evaluate Pi(h) = A - B1 + B2 given precomputed tables for each opp,
    where tables_1 / tables_2 = (S0, S1, S2).

    hands_1 is a list of opp1's (hand, weight) filtered to its stratum
    (used to enumerate B2, which sums over opp1's support). The B2 formula
    is: sum over (a,b) compat with hero & board of S2_1[(a,b)] * S2_2[(a,b)].
    """
    c0, c1 = h
    blocked = set(board) | {c0, c1}

    S0_1, S1_1, S2_1 = tables_1
    S0_2, S1_2, S2_2 = tables_2

    r1 = single_opp_reach_ie(h, S0_1, S1_1, S2_1)
    r2 = single_opp_reach_ie(h, S0_2, S1_2, S2_2)
    A = r1 * r2

    B1 = 0.0
    for c in DECK:
        if c in blocked:
            continue
        T1 = S1_1.get(c, 0.0) - S2_1.get((min(c, c0), max(c, c0)), 0.0) \
             - S2_1.get((min(c, c1), max(c, c1)), 0.0)
        T2 = S1_2.get(c, 0.0) - S2_2.get((min(c, c0), max(c, c0)), 0.0) \
             - S2_2.get((min(c, c1), max(c, c1)), 0.0)
        B1 += T1 * T2

    B2 = 0.0
    seen = set()
    for (a, b), _w in hands_1:
        if a in blocked or b in blocked:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        B2 += S2_1.get(key, 0.0) * S2_2.get(key, 0.0)

    return A - B1 + B2


def fast_showdown_n3_ev(
    hero_hand: Tuple[int, int],
    board: List[int],
    opp_ranges: List[Dict[Tuple[int, int], float]],
    strength_fn,
    pot: float,
    bet: float,
) -> float:
    """3p showdown EV via the W(h) decomposition.

    v(h) = -b * Pi(h) + P * W(h)
    W(h) = Pi_{<,<}(h) + (1/2) * (Pi_{=,<} + Pi_{<,=}) + (1/3) * Pi_{=,=}
    """
    assert len(opp_ranges) == 2
    c0, c1 = hero_hand
    blocked = set(board) | {c0, c1}
    s_h = strength_fn(hero_hand, board)

    # Partition each opp's range by strength relative to s_h: '<', '=', '>'
    partitions = []
    for opp in opp_ranges:
        lt = []
        eq = []
        gt = []
        for (a, b), w in opp.items():
            if a in blocked or b in blocked:
                continue
            s = strength_fn((a, b), board)
            if s < s_h:
                lt.append(((a, b), w))
            elif s == s_h:
                eq.append(((a, b), w))
            else:
                gt.append(((a, b), w))
        partitions.append({"<": lt, "=": eq, ">": gt, "all": lt + eq + gt})

    # Precompute tables per (opp, relation)
    def tables_for(rel_hands):
        return precompute_tables(rel_hands, board)

    tabs = []
    for p in range(2):
        tabs.append({
            "<": tables_for(partitions[p]["<"]),
            "=": tables_for(partitions[p]["="]),
            "all": tables_for(partitions[p]["all"]),
        })

    # Pi(h) = opp-combo reach with no strength filter (the plain reach formula, relation "all")
    Pi_all = pi_n3_given_tables(hero_hand, board, tabs[0]["all"], tabs[1]["all"],
                                 partitions[0]["all"])

    # Pi_{<,<}(h): both opps at strength < s_h
    Pi_ll = pi_n3_given_tables(hero_hand, board, tabs[0]["<"], tabs[1]["<"],
                                partitions[0]["<"])
    # Pi_{=,<}(h): opp1 ties, opp2 < (hero + opp1 split pot)
    Pi_el = pi_n3_given_tables(hero_hand, board, tabs[0]["="], tabs[1]["<"],
                                partitions[0]["="])
    # Pi_{<,=}(h)
    Pi_le = pi_n3_given_tables(hero_hand, board, tabs[0]["<"], tabs[1]["="],
                                partitions[0]["<"])
    # Pi_{=,=}(h): three-way tie
    Pi_ee = pi_n3_given_tables(hero_hand, board, tabs[0]["="], tabs[1]["="],
                                partitions[0]["="])

    W = Pi_ll + 0.5 * (Pi_el + Pi_le) + (1.0 / 3.0) * Pi_ee
    v = -bet * Pi_all + pot * W
    return v


# --- Tests ---

def random_range(rng, board, density=0.25):
    r = {}
    for (a, b) in make_hands(board):
        if rng.random() < density:
            r[(a, b)] = rng.random()
    return r


def test_n3_random(trials: int = 30, seed: int = 123) -> None:
    rng = random.Random(seed)
    max_err = 0.0
    actual = 0
    for t in range(trials):
        # River = 5-card board
        board = rng.sample(DECK, 5)
        while True:
            h = tuple(sorted(rng.sample(DECK, 2)))
            if h[0] not in board and h[1] not in board:
                break
        opp1 = random_range(rng, board, density=0.04)
        opp2 = random_range(rng, board, density=0.04)
        if not opp1 or not opp2:
            continue
        pot = 100.0
        bet = 30.0
        gold = brute_force_showdown_ev(h, board, [opp1, opp2],
                                        hand_strength_surrogate, pot, bet)
        fast = fast_showdown_n3_ev(h, board, [opp1, opp2],
                                    hand_strength_surrogate, pot, bet)
        err = abs(gold - fast)
        max_err = max(max_err, err)
        rel = err / max(abs(gold), 1e-12)
        if rel > 1e-9 and err > 1e-9:
            print(f"  FAIL trial {t}: gold={gold:.6f} fast={fast:.6f} "
                  f"err={err:.2e} rel={rel:.2e}")
        actual += 1
    print(f"N=3 showdown random: {actual} trials, max abs err = {max_err:.2e}")
    assert max_err < 1e-6, f"showdown formula mismatch: {max_err}"


def test_n3_tie_heavy() -> None:
    """All opps have exactly the same hand (3-way tie scenarios)."""
    random.seed(7)
    board = [0, 1, 2, 3, 4]  # all suit 0, not ordered like a real deal but fine
    h = (10, 11)
    # opp1 and opp2 both hold the same single hand
    opp1 = {(20, 21): 1.0}
    opp2 = {(20, 21): 1.0}
    pot, bet = 90.0, 30.0
    gold = brute_force_showdown_ev(h, board, [opp1, opp2],
                                    hand_strength_surrogate, pot, bet)
    fast = fast_showdown_n3_ev(h, board, [opp1, opp2],
                                hand_strength_surrogate, pot, bet)
    # Both opps hold same hand (20,21) -> pairwise card conflict, Pi = 0
    print(f"Tie-heavy (card-conflict): gold={gold:.6f} fast={fast:.6f}")
    assert abs(gold - fast) < 1e-9
    assert abs(gold) < 1e-9


def test_n3_dense() -> None:
    rng = random.Random(999)
    board = rng.sample(DECK, 5)
    while True:
        h = tuple(sorted(rng.sample(DECK, 2)))
        if h[0] not in board and h[1] not in board:
            break
    opp1 = random_range(rng, board, density=0.6)
    opp2 = random_range(rng, board, density=0.6)
    pot, bet = 100.0, 30.0
    gold = brute_force_showdown_ev(h, board, [opp1, opp2],
                                    hand_strength_surrogate, pot, bet)
    fast = fast_showdown_n3_ev(h, board, [opp1, opp2],
                                hand_strength_surrogate, pot, bet)
    rel = abs(gold - fast) / max(abs(gold), 1e-12)
    print(f"Dense (|opp1|={len(opp1)} |opp2|={len(opp2)}): "
          f"gold={gold:.4f} fast={fast:.4f} rel={rel:.2e}")
    assert rel < 1e-9


def main() -> None:
    print("=" * 60)
    print("Showdown inclusion-exclusion decomposition")
    print("=" * 60)
    test_n3_tie_heavy()
    test_n3_random(30)
    test_n3_dense()
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
