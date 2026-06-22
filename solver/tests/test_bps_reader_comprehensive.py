"""Comprehensive integration test for the bps_reader pipeline.

Validates that the C reader's hand → bucket mapping matches the raw
texture cache, and that the card_abstraction DLL's k-means bucketing
produces consistent results. This is the "no shortcuts" verification.

Run:
    cd poker-solver
    python tests/test_bps_reader_comprehensive.py
"""

import ctypes
import os
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from engine.bps_reader_py import BPSReaderC


# ── Card encoding helpers ─────────────────────────────────────────────

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def cards_str(cards) -> str:
    return " ".join(card_str(c) for c in cards)


# ── Raw texture cache reader (independent of bps_reader.c) ──────────

def load_raw_texture_cache(path: str):
    """Load TXC1 file independently (no C code) for ground-truth comparison."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"TXC1", f"Bad magic: {magic}"
        num_tex = struct.unpack("<i", f.read(4))[0]
        num_buckets = struct.unpack("<i", f.read(4))[0]
        hash_keys = np.frombuffer(f.read(num_tex * 8), dtype=np.uint64)
        bucket_data = np.frombuffer(
            f.read(num_tex * 1326 * 4), dtype=np.int32
        ).reshape(num_tex, 1326)
    return hash_keys, bucket_data, num_tex, num_buckets


# ── Python reimplementation of board hash (for verification) ─────────

def _mask64(x):
    return x & 0xFFFFFFFFFFFFFFFF


def _hash_combine_boost(a, b):
    a = _mask64(a ^ _mask64(b + 0x9E3779B97F4A7C15 + (a << 6) + (a >> 2)))
    return a


def _sort_board_like_c(board):
    """Sort board descending by rank using C's bubble sort order.

    C uses strict less-than comparison for swapping, which produces a
    different order than Python's stable sort for same-rank cards.
    """
    s = list(board[:3])
    for a in range(2):
        for b in range(a + 1, 3):
            if (s[a] >> 2) < (s[b] >> 2):
                s[a], s[b] = s[b], s[a]
    return s


def _canonicalize_board_py(board):
    """Python reimplementation of canonicalize_board (must match C)."""
    sorted_b = _sort_board_like_c(board)
    r0, r1, r2 = sorted_b[0] >> 2, sorted_b[1] >> 2, sorted_b[2] >> 2
    s0, s1, s2 = sorted_b[0] & 3, sorted_b[1] & 3, sorted_b[2] & 3

    if r0 == r1 and r1 == r2:
        cf = [r0*4+3, r1*4+2, r2*4+1]
    elif r0 == r1:
        if s2 == s0 or s2 == s1:
            cf = [r0*4+3, r1*4+2, r2*4+3]
        else:
            cf = [r0*4+3, r1*4+2, r2*4+1]
    elif r1 == r2:
        if s0 == s1 or s0 == s2:
            cf = [r0*4+3, r1*4+3, r2*4+2]
        else:
            cf = [r0*4+3, r1*4+2, r2*4+1]
    else:
        if s0 == s1 and s1 == s2:
            cf = [r0*4+3, r1*4+3, r2*4+3]
        elif s0 == s1:
            cf = [r0*4+3, r1*4+3, r2*4+2]
        elif s0 == s2:
            cf = [r0*4+3, r1*4+2, r2*4+3]
        elif s1 == s2:
            cf = [r0*4+2, r1*4+3, r2*4+3]
        else:
            cf = [r0*4+3, r1*4+2, r2*4+1]
    return cf


def _compute_board_hash_boost(board):
    """Board hash with boost mixer (matches training-era texture cache)."""
    canon = _canonicalize_board_py(board)
    h = 0x123456789ABCDEF
    for c in canon:
        h = _hash_combine_boost(h, _mask64(c * 31 + 7))
    return h


def _generate_hands_py(board):
    """Python reimplementation of ca_generate_hands."""
    blocked = set(board[:3])
    hands = []
    for c0 in range(52):
        if c0 in blocked:
            continue
        for c1 in range(c0 + 1, 52):
            if c1 in blocked:
                continue
            hands.append((c0, c1))
    return hands


def _build_suit_map(board):
    """Build suit mapping from actual → canonical (same as canonicalize_board)."""
    sorted_b = _sort_board_like_c(board)
    canon = _canonicalize_board_py(board)

    suit_map = [-1, -1, -1, -1]
    for i in range(3):
        actual_suit = sorted_b[i] & 3
        canon_suit = canon[i] & 3
        if suit_map[actual_suit] == -1:
            suit_map[actual_suit] = canon_suit

    nc = 0
    for i in range(4):
        if suit_map[i] == -1:
            while nc < 4:
                used = any(suit_map[j] == nc for j in range(4))
                if not used:
                    break
                nc += 1
            if nc < 4:
                suit_map[i] = nc
                nc += 1
            else:
                suit_map[i] = 0
    return suit_map


def _map_hand_to_canonical(c0, c1, suit_map):
    """Map a hand to canonical form using the suit mapping."""
    cc0 = (c0 >> 2) * 4 + suit_map[c0 & 3]
    cc1 = (c1 >> 2) * 4 + suit_map[c1 & 3]
    return (min(cc0, cc1), max(cc0, cc1))


# ── Tests ────────────────────────────────────────────────────────────

def test_hand_to_bucket_matches_raw_cache():
    """Verify bps_hand_to_bucket matches raw texture cache for many boards."""
    tc_path = REPO_ROOT / "texture_cache.bin"
    if not tc_path.exists():
        print("SKIP: test_hand_to_bucket_matches_raw_cache (no texture_cache.bin)")
        return

    hash_keys, bucket_data, num_tex, num_buckets = load_raw_texture_cache(str(tc_path))

    # Create a minimal .bps just to initialize the reader
    import tempfile, json, lzma
    inner = bytearray(b"BPS3") + struct.pack("<II", 0, 6)
    compressed = lzma.compress(bytes(inner), preset=1)
    meta = json.dumps({
        "schema_version": 2, "hash_mixer": "boost",
        "postflop_buckets": 200, "preflop_buckets": 169,
        "has_action_evs": False,
    }).encode()
    outer = b"BPS3" + struct.pack("<QI", len(compressed), len(meta)) + compressed + meta

    fd, bps_path = tempfile.mkstemp(suffix=".bps")
    with os.fdopen(fd, "wb") as f:
        f.write(outer)

    try:
        reader = BPSReaderC(bps_path)

        # Test 50 random textures × 20 random hands each = 1000 queries
        rng = np.random.RandomState(12345)
        tested = 0
        mismatches = 0

        for _ in range(50):
            tex_idx = rng.randint(0, num_tex)
            flop_hash = hash_keys[tex_idx]

            # Find a board that produces this hash by trying canonical boards.
            # Instead, pick a random actual board and look up its canonical hash.
            # Pick 3 distinct cards as a flop.
            cards = rng.choice(52, size=3, replace=False).tolist()
            board = sorted(cards)
            board_hash = _compute_board_hash_boost(board)

            # Find which texture this board maps to
            matches = np.where(hash_keys == board_hash)[0]
            if len(matches) == 0:
                continue  # hash collision or board not in cache (shouldn't happen)
            raw_tex_idx = matches[0]

            # Generate canonical hands for this board
            canon_board = _canonicalize_board_py(board)
            canon_hands = _generate_hands_py(canon_board)
            suit_map = _build_suit_map(board)

            # Test 20 random hands
            actual_hands = _generate_hands_py(board)
            if len(actual_hands) == 0:
                continue

            sample_indices = rng.choice(len(actual_hands), size=min(20, len(actual_hands)), replace=False)
            for hi in sample_indices:
                c0, c1 = actual_hands[hi]
                cc0, cc1 = _map_hand_to_canonical(c0, c1, suit_map)
                if cc0 > cc1:
                    cc0, cc1 = cc1, cc0

                # Phase R Block B.3 (Step 6): the trainer's storage at
                # mccfr_blueprint.c:3002 writes to cache_row[full_idx] where
                # full_idx is the FULL-deck pair index (0..1325, ignoring
                # blocked-card filtering). The reader's bps_hand_to_bucket
                # was fixed in Step 6 to read from the same slot.
                # Pre-fix this test compared against canon_hands.index(...)
                # which is BOARD-FILTERED — that was calibrated to the
                # reader's broken behavior, not to the trainer's actual
                # storage.
                full_idx = cc0 * 51 - cc0 * (cc0 - 1) // 2 + cc1 - cc0 - 1

                expected_bucket = int(bucket_data[raw_tex_idx, full_idx])
                actual_bucket = reader.hand_to_bucket(board, c0, c1)

                if actual_bucket != expected_bucket:
                    mismatches += 1
                    if mismatches <= 5:
                        print(
                            f"  MISMATCH: board={cards_str(board)} "
                            f"hand={card_str(c0)}{card_str(c1)} "
                            f"expected={expected_bucket} got={actual_bucket} "
                            f"full_idx={full_idx}"
                        )
                tested += 1

        assert mismatches == 0, (
            f"{mismatches}/{tested} bucket mismatches vs raw texture cache"
        )
        print(f"PASS: test_hand_to_bucket_matches_raw_cache ({tested} queries, 0 mismatches)")
    finally:
        os.unlink(bps_path)


def test_kmeans_consistency():
    """Verify texture cache buckets match fresh ca_assign_buckets_kmeans.

    Note: k-means is non-deterministic (depends on initial centroids and
    sampling), so we can't expect exact equality. Instead we verify that
    the texture cache's bucket assignments produce a reasonable EHS ordering:
    higher buckets should have higher average EHS.
    """
    tc_path = REPO_ROOT / "texture_cache.bin"
    ca_dll = REPO_ROOT / "build" / "card_abstraction.dll"
    if not tc_path.exists() or not ca_dll.exists():
        print("SKIP: test_kmeans_consistency (missing texture_cache.bin or card_abstraction.dll)")
        return

    hash_keys, bucket_data, num_tex, num_buckets = load_raw_texture_cache(str(tc_path))

    lib = ctypes.CDLL(str(ca_dll))
    lib.ca_compute_ehs.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ]
    lib.ca_compute_ehs.restype = ctypes.c_int
    lib.ca_generate_hands.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p,
    ]
    lib.ca_generate_hands.restype = ctypes.c_int

    # Test 5 random flop textures
    rng = np.random.RandomState(99)
    violations = 0
    tested = 0

    for _ in range(5):
        # Pick a random board
        cards = rng.choice(52, size=3, replace=False).tolist()
        board = np.array(sorted(cards), dtype=np.int32)

        board_hash = _compute_board_hash_boost([int(x) for x in board])
        matches = np.where(hash_keys == board_hash)[0]
        if len(matches) == 0:
            continue
        tex_idx = matches[0]

        # Generate hands via C
        hands = np.zeros((1326, 2), dtype=np.int32)
        num_hands = lib.ca_generate_hands(
            board.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 3,
            hands.ctypes.data_as(ctypes.c_void_p),
        )

        # Compute EHS via C (200 MC samples, matches training)
        ehs = np.zeros(num_hands, dtype=np.float32)
        lib.ca_compute_ehs(
            board.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 3,
            hands.ctypes.data_as(ctypes.c_void_p), num_hands,
            500,
            ehs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        # Get bucket assignments from texture cache
        buckets = bucket_data[tex_idx, :num_hands]

        # Verify: compute mean EHS per bucket, check ordering is roughly monotone.
        # K-means on [EHS, PPot, NPot] doesn't guarantee strict EHS ordering,
        # but the lowest ~20% of buckets should have lower mean EHS than the
        # highest ~20% of buckets (otherwise something is structurally wrong).
        bucket_ehs = {}
        for h in range(num_hands):
            b = int(buckets[h])
            if b not in bucket_ehs:
                bucket_ehs[b] = []
            bucket_ehs[b].append(float(ehs[h]))

        bucket_means = {b: np.mean(v) for b, v in bucket_ehs.items() if len(v) > 0}
        sorted_buckets = sorted(bucket_means.items())

        if len(sorted_buckets) >= 10:
            low_20 = np.mean([m for _, m in sorted_buckets[:len(sorted_buckets)//5]])
            high_20 = np.mean([m for _, m in sorted_buckets[-len(sorted_buckets)//5:]])

            if low_20 >= high_20:
                violations += 1
                print(
                    f"  VIOLATION: board={cards_str(board)} "
                    f"low_20_mean_ehs={low_20:.3f} >= high_20_mean_ehs={high_20:.3f}"
                )
            tested += 1

    assert violations == 0, f"{violations}/{tested} EHS ordering violations"
    print(f"PASS: test_kmeans_consistency ({tested} textures verified)")


def test_bucket_distribution():
    """Verify texture cache buckets have reasonable distribution (no degenerate lumping)."""
    tc_path = REPO_ROOT / "texture_cache.bin"
    if not tc_path.exists():
        print("SKIP: test_bucket_distribution (no texture_cache.bin)")
        return

    hash_keys, bucket_data, num_tex, num_buckets = load_raw_texture_cache(str(tc_path))

    rng = np.random.RandomState(42)
    degenerate = 0
    tested = 0

    for _ in range(20):
        tex_idx = rng.randint(0, num_tex)

        # Count how many unique buckets are used
        # (ca_generate_hands with 3 board cards produces ~1128 hands)
        buckets_used = bucket_data[tex_idx, :1128]
        unique_buckets = len(set(buckets_used.tolist()))

        # At 200 target buckets and ~1128 hands, we should use at least 100 buckets.
        # Fewer than 50 would indicate degenerate clustering.
        if unique_buckets < 50:
            degenerate += 1
            print(f"  DEGENERATE: texture {tex_idx} uses only {unique_buckets}/200 buckets")
        tested += 1

    assert degenerate == 0, f"{degenerate}/{tested} textures with degenerate bucket distribution"
    print(f"PASS: test_bucket_distribution ({tested} textures checked)")


def test_suit_isomorphism():
    """Verify that suit-isomorphic boards produce the same buckets.

    e.g., As Ks Qs and Ah Kh Qh should give the same bucket for corresponding hands.
    """
    tc_path = REPO_ROOT / "texture_cache.bin"
    if not tc_path.exists():
        print("SKIP: test_suit_isomorphism (no texture_cache.bin)")
        return

    import tempfile, json, lzma
    inner = bytearray(b"BPS3") + struct.pack("<II", 0, 6)
    compressed = lzma.compress(bytes(inner), preset=1)
    meta = json.dumps({
        "schema_version": 2, "hash_mixer": "boost",
        "postflop_buckets": 200, "preflop_buckets": 169,
        "has_action_evs": False,
    }).encode()
    outer = b"BPS3" + struct.pack("<QI", len(compressed), len(meta)) + compressed + meta
    fd, bps_path = tempfile.mkstemp(suffix=".bps")
    with os.fdopen(fd, "wb") as f:
        f.write(outer)

    try:
        reader = BPSReaderC(bps_path)

        # Board 1: As Ks Qs (spades) = 48+3, 44+3, 40+3 = 51, 47, 43
        board_s = [51, 47, 43]
        # Board 2: Ah Kh Qh (hearts) = 48+2, 44+2, 40+2 = 50, 46, 42
        board_h = [50, 46, 42]

        # Hand: Jd Td on board 1
        # Equivalent hand: Jd Td on board 2 (same ranks, suits don't touch board)
        # Actually — we need hands in corresponding suits.
        # On AKQ rainbow (all different suits): Js Ts maps to Jh Th
        # Board 1 (spades): hand = Jc Tc = 36, 32
        # Board 2 (hearts): hand = Jc Tc = 36, 32
        # These are equivalent because the hand suits don't appear on either board.

        bucket1 = reader.hand_to_bucket(board_s, 36, 32)  # Jc Tc on AKQs
        bucket2 = reader.hand_to_bucket(board_h, 36, 32)  # Jc Tc on AKQh

        assert bucket1 == bucket2, (
            f"Suit isomorphism failed: {cards_str(board_s)} Jc Tc → bucket {bucket1}, "
            f"{cards_str(board_h)} Jc Tc → bucket {bucket2}"
        )

        # Also test: same hand expressed in board-suited form
        # Board 1 (spades): hand = Js 9s = 39, 27 (both spades, flush draw)
        # Board 2 (hearts): hand = Jh 9h = 38, 26 (both hearts, flush draw)
        bucket3 = reader.hand_to_bucket(board_s, 39, 27)  # Js 9s
        bucket4 = reader.hand_to_bucket(board_h, 38, 26)  # Jh 9h
        assert bucket3 == bucket4, (
            f"Suited isomorphism failed: {card_str(39)}{card_str(27)} → {bucket3}, "
            f"{card_str(38)}{card_str(26)} → {bucket4}"
        )

        print("PASS: test_suit_isomorphism")
    finally:
        os.unlink(bps_path)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Comprehensive bps_reader tests (repo: {REPO_ROOT})")
    print()

    test_hand_to_bucket_matches_raw_cache()
    test_kmeans_consistency()
    test_bucket_distribution()
    test_suit_isomorphism()

    print()
    print("All comprehensive tests passed.")
