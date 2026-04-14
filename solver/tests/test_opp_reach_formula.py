"""
Ground-truth verification for the opponent-reach inclusion-exclusion formula
used in the fast leaf/fold value kernels.

Validates that:
  Pi(h) = sum_{h_1, ..., h_{N-1}} 1[all disjoint from h, B, each other] * prod_p r_p[h_p]

can be computed via inclusion-exclusion in O(M) per hand (N=2) or O(52) per
hand (N=3) instead of O(M^{N-1}).

Run with: python tests/test_opp_reach_formula.py
"""
import itertools
import random
from typing import Dict, List, Tuple

# Cards: 0..51. rank = c >> 2, suit = c & 3.
DECK = list(range(52))


def make_all_hands(exclude: List[int]) -> List[Tuple[int, int]]:
    """All 2-card combos from deck \\ exclude."""
    avail = [c for c in DECK if c not in exclude]
    return [(a, b) for a, b in itertools.combinations(avail, 2)]


def random_range(rng: random.Random, board: List[int],
                 density: float = 0.3) -> Dict[Tuple[int, int], float]:
    """Build a random sparse opponent range over hands compatible with board."""
    r: Dict[Tuple[int, int], float] = {}
    for h in make_all_hands(board):
        if rng.random() < density:
            r[h] = rng.random()
    return r


def brute_force_pi(traverser_hand: Tuple[int, int],
                   board: List[int],
                   opp_ranges: List[Dict[Tuple[int, int], float]]) -> float:
    """Gold standard: enumerate all opponent N-tuples with full conflict checks."""
    c0, c1 = traverser_hand
    blocked = set(board) | {c0, c1}
    # Filter each opponent's hands by traverser+board compat
    filtered = []
    for opp in opp_ranges:
        f = {h: w for h, w in opp.items()
             if h[0] not in blocked and h[1] not in blocked}
        filtered.append(f)

    total = 0.0
    # Enumerate N-tuples of (hand_p, weight_p) across opponents
    for combo in itertools.product(*[list(f.items()) for f in filtered]):
        # Check pairwise card disjointness across opponents
        all_cards = []
        ok = True
        for (h_p, _w) in combo:
            for c in h_p:
                if c in all_cards:
                    ok = False
                    break
                all_cards.append(c)
            if not ok:
                break
        if not ok:
            continue
        prod = 1.0
        for (_h, w) in combo:
            prod *= w
        total += prod
    return total


def single_opp_reach(traverser_hand: Tuple[int, int],
                     board: List[int],
                     opp_range: Dict[Tuple[int, int], float]) -> float:
    """tilde_r_p(h) = sum over h_p in range compatible with board + h."""
    c0, c1 = traverser_hand
    blocked = set(board) | {c0, c1}
    s = 0.0
    for (a, b), w in opp_range.items():
        if a in blocked or b in blocked:
            continue
        s += w
    return s


def precompute_card_tables(board: List[int],
                           opp_range: Dict[Tuple[int, int], float]) -> Tuple[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """S_0, S_1[c], S_2[(c0,c1)] filtered by board compat only."""
    board_set = set(board)
    # Board-compat filtered range
    S2: Dict[Tuple[int, int], float] = {}
    for (a, b), w in opp_range.items():
        if a in board_set or b in board_set:
            continue
        # Canonicalize order
        key = (a, b) if a < b else (b, a)
        S2[key] = w
    # S_0
    S0 = sum(S2.values())
    # S_1[c]
    S1: Dict[int, float] = {c: 0.0 for c in DECK if c not in board_set}
    for (a, b), w in S2.items():
        S1[a] = S1.get(a, 0.0) + w
        S1[b] = S1.get(b, 0.0) + w
    return S0, S1, S2


def single_opp_reach_via_ie(traverser_hand: Tuple[int, int],
                            S0: float, S1: Dict[int, float],
                            S2: Dict[Tuple[int, int], float]) -> float:
    """tilde_r_p(h) via inclusion-exclusion: S0 - S1[c0] - S1[c1] + S2[(c0,c1)]."""
    c0, c1 = traverser_hand
    key = (c0, c1) if c0 < c1 else (c1, c0)
    return S0 - S1.get(c0, 0.0) - S1.get(c1, 0.0) + S2.get(key, 0.0)


def fast_pi_n2(traverser_hand: Tuple[int, int],
               board: List[int],
               opp_range: Dict[Tuple[int, int], float]) -> float:
    """N=2: exact via inclusion-exclusion."""
    S0, S1, S2 = precompute_card_tables(board, opp_range)
    return single_opp_reach_via_ie(traverser_hand, S0, S1, S2)


def fast_pi_n3(traverser_hand: Tuple[int, int],
               board: List[int],
               opp_ranges: List[Dict[Tuple[int, int], float]]) -> float:
    """N=3 (2 opponents): Pi = A(h) - B(h) exactly.

    Inclusion-exclusion on "opps share a card":
      1[disjoint] = 1 - 1[share >= 1 card]
      1[share >= 1 card] = (sum_c 1[share card c]) - 1[share both cards, i.e. h_1 == h_2]
    (because for 2-card hands, sharing >= 2 cards means the hands are equal)

    So:
      Pi(h) = sum r_1 r_2 [compat with h,B, pairwise disjoint]
            = A(h) - B1(h) + B2(h)
    where
      A(h)  = (sum_{h_1 compat h,B} r_1[h_1]) * (sum_{h_2 compat h,B} r_2[h_2])
      B1(h) = sum_c T_1(c; h) * T_2(c; h)    # "share at least card c"
      B2(h) = sum_{(a,b) compat h,B} r_1[(a,b)] * r_2[(a,b)]    # "h_1 == h_2"
    """
    assert len(opp_ranges) == 2
    c0, c1 = traverser_hand
    blocked_cards = set(board) | {c0, c1}

    s_tables = [precompute_card_tables(board, r) for r in opp_ranges]

    # A(h)
    r_tildes = [single_opp_reach_via_ie(traverser_hand, *s_tables[p]) for p in range(2)]
    A = r_tildes[0] * r_tildes[1]

    # B1(h) = sum_c T_1(c;h) * T_2(c;h)
    # T_p(c; h) = reach of opp p on hands containing c, compat with board ∪ h
    #           = S_1^p[c] - r_p[(c, c0)]_compat_board - r_p[(c, c1)]_compat_board
    # For c in blocked_cards: skip (c can't appear in any compat opp hand).
    B1 = 0.0
    for c in DECK:
        if c in blocked_cards:
            continue
        T_vals = []
        for p in range(2):
            _S0_p, S1_p, S2_p = s_tables[p]
            T_pB_c = S1_p.get(c, 0.0)
            key_c0 = (c, c0) if c < c0 else (c0, c)
            key_c1 = (c, c1) if c < c1 else (c1, c)
            T_p_h = T_pB_c - S2_p.get(key_c0, 0.0) - S2_p.get(key_c1, 0.0)
            T_vals.append(T_p_h)
        B1 += T_vals[0] * T_vals[1]

    # B2(h) = sum over hands (a,b) compat with h,B of r_1[(a,b)] * r_2[(a,b)]
    # Use opp1's board-compat-filtered S_2 as the enumeration set.
    _, _, S2_1 = s_tables[0]
    _, _, S2_2 = s_tables[1]
    B2 = 0.0
    for key, w1 in S2_1.items():
        a, b = key
        if a in blocked_cards or b in blocked_cards:
            continue  # not compat with h
        w2 = S2_2.get(key, 0.0)
        B2 += w1 * w2

    return A - B1 + B2


def test_n2_random(num_trials: int = 50, seed: int = 42) -> None:
    """Exhaustively verify N=2 fast formula matches brute force on random inputs."""
    rng = random.Random(seed)
    max_err = 0.0
    for trial in range(num_trials):
        # Random board: 3, 4, or 5 cards
        num_board = rng.choice([3, 4, 5])
        board = rng.sample(DECK, num_board)
        # Random traverser hand (not blocked by board)
        while True:
            h = tuple(sorted(rng.sample(DECK, 2)))
            if h[0] not in board and h[1] not in board:
                break
        # Random opponent range
        opp = random_range(rng, board, density=0.15)
        if not opp:
            continue

        gold = brute_force_pi(h, board, [opp])
        fast = fast_pi_n2(h, board, opp)
        err = abs(gold - fast)
        max_err = max(max_err, err)
        if err > 1e-9:
            print(f"  N=2 FAIL trial {trial}: gold={gold:.9f} fast={fast:.9f} err={err:.2e}")
    print(f"N=2 random trials: {num_trials}, max error = {max_err:.2e}")
    assert max_err < 1e-9, f"N=2 formula mismatch: {max_err}"


def test_n3_random(num_trials: int = 30, seed: int = 123) -> None:
    """Verify N=3 exact formula matches brute force."""
    rng = random.Random(seed)
    max_err = 0.0
    for trial in range(num_trials):
        num_board = rng.choice([3, 4, 5])
        board = rng.sample(DECK, num_board)
        while True:
            h = tuple(sorted(rng.sample(DECK, 2)))
            if h[0] not in board and h[1] not in board:
                break
        # Two opponent ranges (keep sparse so brute force stays tractable)
        opp1 = random_range(rng, board, density=0.05)
        opp2 = random_range(rng, board, density=0.05)
        if not opp1 or not opp2:
            continue

        gold = brute_force_pi(h, board, [opp1, opp2])
        fast = fast_pi_n3(h, board, [opp1, opp2])
        err = abs(gold - fast)
        max_err = max(max_err, err)
        if err > 1e-9:
            print(f"  N=3 FAIL trial {trial}: gold={gold:.9f} fast={fast:.9f} err={err:.2e}")
    print(f"N=3 random trials: {num_trials}, max error = {max_err:.2e}")
    assert max_err < 1e-6, f"N=3 formula mismatch: {max_err}"


def test_n3_blocker_edge_cases() -> None:
    """Adversarial cases: heavy pair blockers, full board, overlapping ranges."""
    # Board with Ax — makes AA blocker effects strong
    # Cards: 0=2c,1=2d,2=2h,3=2s, ..., 48=Ac,49=Ad,50=Ah,51=As
    board = [48, 36, 20]  # Ac, Tc, 6c
    h = (51, 49)  # As, Ad — strong hand blocking other aces

    # Opponents with AA-heavy ranges
    opp1 = {
        (50, 51): 0.0,  # Ah As — blocked by hero
        (50, 49): 0.0,  # Ah Ad — blocked by hero (Ad)
        (50, 48): 0.0,  # Ah Ac — blocked by board
        # Other AA combos all blocked by hero or board
        (44, 45): 1.0,  # KcKd
        (46, 47): 0.5,  # KhKs
        (44, 47): 0.8,  # KcKs
    }
    opp2 = {
        (44, 45): 0.6,  # KcKd — conflict with opp1
        (44, 47): 0.4,  # KcKs — conflict with opp1
        (46, 47): 0.7,  # KhKs — conflict with opp1
        (40, 41): 0.9,  # QcQd — no conflict
        (42, 43): 0.3,  # QhQs — no conflict
    }
    gold = brute_force_pi(h, board, [opp1, opp2])
    fast = fast_pi_n3(h, board, [opp1, opp2])
    err = abs(gold - fast)
    print(f"Blocker edge case: gold={gold:.6f} fast={fast:.6f} err={err:.2e}")
    assert err < 1e-9, f"Edge case mismatch: {err}"


def test_empty_range() -> None:
    """Degenerate: opp has zero reach → Pi should be 0."""
    board = [0, 1, 2]
    h = (10, 11)
    result_n2 = fast_pi_n2(h, board, {})
    result_n3 = fast_pi_n3(h, board, [{}, {}])
    assert result_n2 == 0.0
    assert result_n3 == 0.0
    print("Empty range: OK")


def test_full_overlap_ranges() -> None:
    """Both opponents have the same single hand — they must conflict, Pi should be 0."""
    board = [0, 1, 2]
    h = (10, 11)
    single_hand = {(20, 21): 1.0}
    fast = fast_pi_n3(h, board, [single_hand, single_hand])
    gold = brute_force_pi(h, board, [single_hand, single_hand])
    # Both opps hold (20, 21) → they share both cards → always conflict
    print(f"Full overlap: gold={gold:.6f} fast={fast:.6f}")
    assert abs(gold) < 1e-9
    assert abs(fast) < 1e-9


def test_independent_ranges() -> None:
    """Opponents with disjoint card pools — no conflicts possible → Pi = r1 * r2 exactly."""
    board = [0, 1, 2]  # low cards
    h = (10, 11)  # different low cards
    # Opp1 only has hands in {16..23}, opp2 only in {32..39} — no shared cards possible
    opp1 = {(16, 17): 0.3, (18, 19): 0.5, (20, 21): 0.2}
    opp2 = {(32, 33): 0.4, (34, 35): 0.6}
    gold = brute_force_pi(h, board, [opp1, opp2])
    fast = fast_pi_n3(h, board, [opp1, opp2])
    # Since disjoint, Pi should equal A(h) with no B correction
    r1 = sum(opp1.values())
    r2 = sum(opp2.values())
    expected = r1 * r2
    print(f"Disjoint pools: gold={gold:.6f} fast={fast:.6f} expected={expected:.6f}")
    assert abs(gold - expected) < 1e-9
    assert abs(fast - gold) < 1e-9


def test_realistic_density() -> None:
    """Dense opponent ranges approximating real subgames.

    With density=0.6 on a 3-card flop we get ~700 hands per opp. Enumerating
    all (700 x 700) N-tuples in brute force is manageable (~500K ops per call).
    """
    rng = random.Random(999)
    board = rng.sample(DECK, 3)
    while True:
        h = tuple(sorted(rng.sample(DECK, 2)))
        if h[0] not in board and h[1] not in board:
            break
    opp1 = random_range(rng, board, density=0.6)
    opp2 = random_range(rng, board, density=0.6)

    gold = brute_force_pi(h, board, [opp1, opp2])
    fast = fast_pi_n3(h, board, [opp1, opp2])
    rel_err = abs(gold - fast) / max(abs(gold), 1e-12)
    print(f"Dense (|opp1|={len(opp1)}, |opp2|={len(opp2)}): "
          f"gold={gold:.6f} fast={fast:.6f} rel_err={rel_err:.2e}")
    assert rel_err < 1e-9, f"Dense case relative error too large: {rel_err}"


def test_timing_crossover() -> None:
    """Sanity-check the speedup we'd see on a realistic 200-hand subgame."""
    import time
    rng = random.Random(7)
    board = rng.sample(DECK, 3)
    while True:
        h = tuple(sorted(rng.sample(DECK, 2)))
        if h[0] not in board and h[1] not in board:
            break
    opp1 = random_range(rng, board, density=0.2)  # ~235 hands
    opp2 = random_range(rng, board, density=0.2)

    # Time brute force (one call)
    t0 = time.perf_counter()
    gold = brute_force_pi(h, board, [opp1, opp2])
    t_brute = time.perf_counter() - t0

    # Time fast formula (precompute + one query)
    t0 = time.perf_counter()
    fast = fast_pi_n3(h, board, [opp1, opp2])
    t_fast = time.perf_counter() - t0

    # Fast with precompute amortized across 200 hand queries
    t0 = time.perf_counter()
    s_tables = [precompute_card_tables(board, r) for r in [opp1, opp2]]
    all_hands = make_all_hands(board)[:200]
    for test_h in all_hands:
        if test_h[0] in board or test_h[1] in board:
            continue
        # Inline fast computation using cached s_tables
        c0, c1 = test_h
        blocked = set(board) | {c0, c1}
        r_tildes = [single_opp_reach_via_ie(test_h, *s_tables[p]) for p in range(2)]
        A = r_tildes[0] * r_tildes[1]
        B1 = 0.0
        for c in DECK:
            if c in blocked:
                continue
            T_vals = []
            for p in range(2):
                _, S1_p, S2_p = s_tables[p]
                T_pB = S1_p.get(c, 0.0)
                kc0 = (c, c0) if c < c0 else (c0, c)
                kc1 = (c, c1) if c < c1 else (c1, c)
                T_vals.append(T_pB - S2_p.get(kc0, 0.0) - S2_p.get(kc1, 0.0))
            B1 += T_vals[0] * T_vals[1]
        B2 = 0.0
        _, _, S2_1 = s_tables[0]
        _, _, S2_2 = s_tables[1]
        for key, w1 in S2_1.items():
            a, b = key
            if a in blocked or b in blocked:
                continue
            B2 += w1 * S2_2.get(key, 0.0)
        _ = A - B1 + B2
    t_fast_amortized = time.perf_counter() - t0

    rel_err = abs(gold - fast) / max(abs(gold), 1e-12)
    print(f"Timing (|opp1|={len(opp1)}, |opp2|={len(opp2)}):")
    print(f"  brute force 1 call:      {t_brute*1e6:.1f} us")
    print(f"  fast formula 1 call:     {t_fast*1e6:.1f} us   (speedup {t_brute/t_fast:.1f}x)")
    print(f"  fast 200 hands (one-time precompute amortized): {t_fast_amortized*1e6:.1f} us")
    print(f"  effective per-hand cost: {t_fast_amortized/200*1e6:.2f} us")
    print(f"  rel_err: {rel_err:.2e}")
    assert rel_err < 1e-9


def main() -> None:
    print("=" * 60)
    print("Opponent-reach inclusion-exclusion verification")
    print("=" * 60)
    test_empty_range()
    test_full_overlap_ranges()
    test_independent_ranges()
    test_n3_blocker_edge_cases()
    test_n2_random(num_trials=50)
    test_n3_random(num_trials=30)
    test_realistic_density()
    test_timing_crossover()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
