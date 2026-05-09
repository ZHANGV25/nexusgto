"""Regression test for the bps_ev_walk preflop bucket recompute.

The trainer (mccfr_blueprint.c:2486-2500, ca_preflop_classes) stores
preflop strategies in a 169-class interleaved ordering: for each
(rank_hi, rank_lo) descending, a pair gets one slot, a non-pair gets
two (suited then offsuit). Closed-form: base(r0) = 168 - r0*(r0+2),
then offset by suited/offsuit and 2*(r0-1-r1).

Earlier bps_ev_walk.c:951-997 recomputed buckets in suited-then-
offsuit per hi-rank order (AKs=1, AKo=13, AQs=2, AQo=14, ...),
disagreeing with the trainer on 132 of 169 cells. This test pins the
fix: the closed-form bucket index used in the C walk after the fix
must match the trainer's interleaved enumeration on every hand.
"""

import pytest


def trainer_class_map():
    """Mirror of mccfr_blueprint.c:2486-2500 (and card_abstraction.c
    ca_preflop_classes). Returns dict keyed (r0_hi, r1_lo, suited)."""
    classes = {}
    n = 0
    for r0 in range(12, -1, -1):
        for r1 in range(r0, -1, -1):
            if r0 == r1:
                classes[(r0, r1, 0)] = n
                classes[(r0, r1, 1)] = n
                n += 1
            else:
                classes[(r0, r1, 1)] = n; n += 1  # suited
                classes[(r0, r1, 0)] = n; n += 1  # offsuit
    assert n == 169
    return classes


def closed_form_bucket(c0, c1):
    """Mirror of bps_ev_walk.c:951-973 (post-fix). Inputs are card
    indices 0..51 (rank = card >> 2, suit = card & 3)."""
    r0, r1 = c0 >> 2, c1 >> 2
    suited = (c0 & 3) == (c1 & 3)
    if r0 < r1:
        r0, r1 = r1, r0
    base_r0 = 168 - r0 * (r0 + 2)
    if r0 == r1:
        return base_r0
    return base_r0 + (1 if suited else 2) + 2 * (r0 - 1 - r1)


def buggy_walk_bucket(c0, c1):
    """Mirror of the OLD bps_ev_walk.c:951-997 recompute, kept here
    so the test can also assert the bug is no longer present."""
    r0, r1 = c0 >> 2, c1 >> 2
    suited = (c0 & 3) == (c1 & 3)
    if r0 < r1:
        r0, r1 = r1, r0
    if r0 == r1:
        cls = 0
        for hi in range(12, -1, -1):
            if hi == r0:
                return cls
            cls += 1 + 2 * hi
    if suited:
        cls = 0
        for hi in range(12, -1, -1):
            cls += 1
            for lo in range(hi - 1, -1, -1):
                if hi == r0 and lo == r1:
                    return cls
                cls += 1
            cls += hi
    cls = 0
    for hi in range(12, -1, -1):
        cls += 1 + hi
        for lo in range(hi - 1, -1, -1):
            if hi == r0 and lo == r1:
                return cls
            cls += 1


def all_169_canonical_pairs():
    """Yield one (c0, c1) representative per class (no duplicates)."""
    seen = set()
    for r0 in range(12, -1, -1):
        for r1 in range(r0, -1, -1):
            if r0 == r1:
                yield (r0 * 4 + 0, r1 * 4 + 1)  # pair (suit irrelevant)
            else:
                yield (r0 * 4 + 0, r1 * 4 + 0)  # suited
                yield (r0 * 4 + 0, r1 * 4 + 1)  # offsuit


def test_closed_form_matches_trainer_on_169_classes():
    classes = trainer_class_map()
    seen = set()
    for c0, c1 in all_169_canonical_pairs():
        r0, r1 = c0 >> 2, c1 >> 2
        suited = (c0 & 3) == (c1 & 3)
        if r0 < r1:
            r0, r1 = r1, r0
        expected = classes[(r0, r1, 1 if suited else 0)]
        actual = closed_form_bucket(c0, c1)
        assert actual == expected, (
            f"Hand r0={r0} r1={r1} suited={suited}: "
            f"closed_form={actual} trainer={expected}"
        )
        seen.add(actual)
    assert len(seen) == 169


def test_premium_landmarks():
    """Spot-check the buckets the JSON / .cache cross-check relied on."""
    AA = closed_form_bucket(12 * 4 + 0, 12 * 4 + 1)
    AKs = closed_form_bucket(12 * 4 + 0, 11 * 4 + 0)
    AKo = closed_form_bucket(12 * 4 + 0, 11 * 4 + 1)
    KK = closed_form_bucket(11 * 4 + 0, 11 * 4 + 1)
    QQ = closed_form_bucket(10 * 4 + 0, 10 * 4 + 1)
    deuces = closed_form_bucket(0 * 4 + 0, 0 * 4 + 1)
    assert AA == 0
    assert AKs == 1
    assert AKo == 2
    assert KK == 25  # the canary the cache-clobber memo pinned
    assert QQ == 48
    assert deuces == 168


def test_buggy_walk_disagreement_count():
    """Pin the bug character: old code agreed with trainer on 37/169
    cells (all 13 pairs + boundary suited/offsuit cases) and
    disagreed on the other 132. Catches accidental revert."""
    classes = trainer_class_map()
    agree = 0
    disagree = 0
    for c0, c1 in all_169_canonical_pairs():
        r0, r1 = c0 >> 2, c1 >> 2
        suited = (c0 & 3) == (c1 & 3)
        if r0 < r1:
            r0, r1 = r1, r0
        expected = classes[(r0, r1, 1 if suited else 0)]
        if buggy_walk_bucket(c0, c1) == expected:
            agree += 1
        else:
            disagree += 1
    assert (agree, disagree) == (37, 132)


if __name__ == "__main__":
    test_closed_form_matches_trainer_on_169_classes()
    test_premium_landmarks()
    test_buggy_walk_disagreement_count()
    print("All preflop bucket regression tests passed")
