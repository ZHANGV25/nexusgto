"""Cross-artifact contract: leaf encoder indices == trainer enumeration.

This test guards against a bet-vocab desync between the leaf encoder and
the trainer. The published 8.05B blueprint was trained with
postflop_bet_sizes=[0.75] / subsequent=[1.0]; an encoder that hardcodes
[0.5,1.0] / [1.0] desyncs from it. Because the blueprint node key is a
hash of the action-INDEX sequence, an encoder that enumerates a different
action set assigns a different index to the same action → a different
action_hash → a blueprint MISS → uniform fallback. The divergence bites at
the ALL-IN index: under [0.75] the trainer puts all-in at idx 2 (non-facing)
/ 3 (facing); a [0.5,1.0] encoder puts it at 3 / 4.

What this test locks: for BOTH vocabs, at BOTH facing and non-facing nodes,
for first and subsequent raises, the encoder
(python/engine/leaf_values.py::_train_action_idx) emits EXACTLY the index
the trainer's generate_actions (src/blueprint/mccfr_blueprint.c:954-991)
assigns to the same action — fold / check / call / each sized bet / all-in.

PURE PYTHON. No ctypes, no .so, no blueprint load. Importable with only the
repo's python/ on sys.path.
The trainer rule is reimplemented as a small reference function below rather
than linked against the C .so, so the test is a genuine independent oracle
(a bug that changes BOTH the encoder and the C generator would still be
caught here, because the reference is derived from a reading of the C
source, not from the C binary).
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

import pytest  # noqa: E402

from engine.leaf_values import _train_action_idx  # noqa: E402


# ── Reference reimplementation of the trainer's enumeration ──────────────
#
# Derived from src/blueprint/mccfr_blueprint.c:954-991 (generate_actions),
# read directly (not from any doc summary). The emission order there is:
#
#   1. if to_call > 0:            emit FOLD
#   2.                           emit (to_call>0 ? CALL : CHECK)
#   3. if num_raises < max_raises:
#        for i in 0..num_bet_sizes-1:  emit sized BET  (bet_idx = i)
#        [each clamps to stack; a size that reaches stack becomes the all-in
#         and suppresses the explicit all-in below]
#        if add_explicit_allin and no sized bet already reached stack:
#             emit ALL-IN (bet_idx = num_bet_sizes)   <-- index = num_bet_sizes
#
# So, indexing the emitted list, with n = num_bet_sizes at this node:
#   NON-FACING (to_call == 0):
#       [check@0, bet_0@1, ..., bet_{n-1}@n, allin@(n+1)]
#   FACING (to_call > 0):
#       [fold@0, call@1, bet_0@2, ..., bet_{n-1}@(n+1), allin@(n+2)]
#   TREE CLOSED (num_raises >= max_raises): no raises emitted
#       facing:      [fold@0, call@1]
#       non-facing:  [check@0]
#
# num_bet_sizes at a node = the SIZE LIST in force at that node. The two
# artifacts select that list by DIFFERENT predicates:
#
#   (i)  TRAINER: selects the subsequent list on num_raises > 0 ALONE — no
#        facing check (mccfr_blueprint.c:2028-2031 and :2610-2612, both
#        postflop call sites: `ts->num_raises > 0 && num_subsequent_bet_sizes
#        > 0` -> subsequent_bet_sizes, else bet_sizes).
#   (ii) ENCODER: leaf_values._train_action_idx uses first_fracs for ALL
#        non-facing nodes (its !facing branch never consults raises_so_far
#        for list selection); only a FACING node with raises_so_far >= 1
#        uses subseq_fracs.
#
# These predicates differ only on nodes with to_call == 0 AND num_raises > 0
# on the SAME street — which is UNREACHABLE in NLHE betting: once a raise
# exists on a street, every subsequent actor faces to_call > 0 until the
# street closes (the round ends when all callers match; nobody acts again
# at to_call == 0 after a raise). So encoder and trainer agree on every
# REACHABLE node. The reference below deliberately mirrors the ENCODER's
# predicate (use_subseq = facing and raises_so_far >= 1) so the contract is
# exact on reachable nodes and still well-defined on the unreachable probe
# cases this test sweeps.

_POSTFLOP_MAX_RAISES = 3  # matches leaf_values._POSTFLOP_MAX_RAISES / trainer


def _reference_action_index(facing, raises_so_far, kind, size_frac,
                            first_fracs, subseq_fracs):
    """Return the index the trainer's generate_actions assigns to `kind`.

    Independent oracle for _train_action_idx. `kind` in
    {fold, check, call, bet, raise, allin}. `size_frac` is the pot fraction
    for bet/raise (fuzzy-matched to the nearest trained slot, mirroring the
    trainer consuming its discrete size list); ignored otherwise.
    """
    tree_closed = raises_so_far >= _POSTFLOP_MAX_RAISES
    # Size list in force at this node (see module note above): subsequent
    # list only at a FACING subsequent raise; opening list everywhere else.
    use_subseq = facing and raises_so_far >= 1
    fracs = subseq_fracs if use_subseq else first_fracs
    n = len(fracs)

    # Build the emitted-action index map explicitly, in emission order.
    order = []
    if facing:
        order.append("fold")           # step 1
        order.append("call")           # step 2 (call, since to_call>0)
    else:
        order.append("check")          # step 2 (check, since to_call==0)
    if not tree_closed:
        for i in range(n):             # step 3: sized bets/raises
            order.append(("size", i))
        order.append("allin")          # explicit all-in @ num_bet_sizes

    if kind == "fold":
        return order.index("fold")
    if kind in ("check", "call"):
        return order.index("check" if not facing else "call")
    if kind == "allin":
        return order.index("allin")
    if kind in ("bet", "raise"):
        best_i = min(range(n), key=lambda i: abs(size_frac - fracs[i]))
        return order.index(("size", best_i))
    raise ValueError(f"unknown kind {kind!r}")


# ── Vocabs under contract ────────────────────────────────────────────────

SERVED = ((0.75,), (1.0,))          # 8.05B manifest footer (config_hash bc8ac105)
LEGACY = ((0.5, 1.0), (1.0,))       # older compiled default

VOCABS = [
    pytest.param(*SERVED, id="served_0.75"),
    pytest.param(*LEGACY, id="legacy_0.5_1.0"),
]


def _all_actions(first_fracs, subseq_fracs):
    """Enumerate every (facing, raises_so_far, kind, size_frac) case the
    encoder must key, for a given vocab. raises_so_far spans a first raise
    (0), a subsequent raise (1), and a tree-closed node (MAX_RAISES)."""
    cases = []
    # Non-facing, first raise (opening bet): check, each trained bet, all-in.
    cases.append((False, 0, "check", 0.0))
    for f in first_fracs:
        cases.append((False, 0, "bet", f))
    cases.append((False, 0, "allin", 1.0))
    # Facing, first raise: fold, call, each trained raise, all-in.
    cases.append((True, 0, "fold", 0.0))
    cases.append((True, 0, "call", 0.0))
    for f in first_fracs:
        cases.append((True, 0, "raise", f))
    cases.append((True, 0, "allin", 1.0))
    # Facing, subsequent raise: fold, call, each trained subseq raise, all-in.
    cases.append((True, 1, "fold", 0.0))
    cases.append((True, 1, "call", 0.0))
    for f in subseq_fracs:
        cases.append((True, 1, "raise", f))
    cases.append((True, 1, "allin", 1.0))
    # Non-facing, subsequent (to_call==0, raises_so_far>0): UNREACHABLE in
    # NLHE betting (see module note) — kept as a defensive probe; both sides
    # key it with first_fracs, so the contract stays well-defined.
    cases.append((False, 1, "check", 0.0))
    for f in subseq_fracs:
        cases.append((False, 1, "bet", f))
    cases.append((False, 1, "allin", 1.0))
    # Tree closed (raises_so_far == MAX_RAISES): only fold/call or check.
    cases.append((True, _POSTFLOP_MAX_RAISES, "fold", 0.0))
    cases.append((True, _POSTFLOP_MAX_RAISES, "call", 0.0))
    cases.append((False, _POSTFLOP_MAX_RAISES, "check", 0.0))
    return cases


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_encoder_matches_trainer_enumeration(first_fracs, subseq_fracs):
    """Encoder index == trainer reference index for every action, both
    facing and non-facing, first and subsequent raises, for this vocab."""
    for facing, rsf, kind, frac in _all_actions(first_fracs, subseq_fracs):
        got = _train_action_idx(
            facing, rsf, kind, frac,
            first_fracs=first_fracs, subseq_fracs=subseq_fracs,
        )
        want = _reference_action_index(
            facing, rsf, kind, frac,
            first_fracs=first_fracs, subseq_fracs=subseq_fracs,
        )
        assert got == want, (
            f"encoder/trainer index mismatch for vocab "
            f"first={first_fracs} subseq={subseq_fracs}: "
            f"(facing={facing}, raises_so_far={rsf}, kind={kind}, "
            f"size_frac={frac}) encoder={got} trainer={want}"
        )


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_encoder_matches_trainer_off_tree_sizes(first_fracs, subseq_fracs):
    """Off-tree (subgame) bet/raise sizes must fuzzy-map to the SAME slot in
    both the encoder and the reference, so a nested re-solve size still keys
    to a trained action index. Probe a spread of off-tree fractions."""
    off_tree = (0.33, 0.55, 0.6, 0.8, 0.9, 1.25, 2.0)
    for facing in (False, True):
        for rsf in (0, 1):
            for frac in off_tree:
                kind = "raise" if facing else "bet"
                got = _train_action_idx(
                    facing, rsf, kind, frac,
                    first_fracs=first_fracs, subseq_fracs=subseq_fracs,
                )
                want = _reference_action_index(
                    facing, rsf, kind, frac,
                    first_fracs=first_fracs, subseq_fracs=subseq_fracs,
                )
                assert got == want, (
                    f"off-tree size {frac} mis-mapped: encoder={got} "
                    f"trainer={want} (facing={facing}, rsf={rsf}, "
                    f"first={first_fracs} subseq={subseq_fracs})"
                )


# ── The two all-in regression cases ──────────────────────────────────────


def test_regression_nonfacing_allin_is_index_2_under_0_75():
    """Under the published [0.75] vocab the NON-FACING all-in must be
    index 2 (NOT 3, the [0.5,1.0] value).
    Both the encoder and the independent trainer reference must say 2."""
    first, subseq = SERVED
    assert _train_action_idx(False, 0, "allin", 1.0,
                             first_fracs=first, subseq_fracs=subseq) == 2
    assert _reference_action_index(False, 0, "allin", 1.0,
                                   first_fracs=first, subseq_fracs=subseq) == 2
    # The desynced value (3) is what the LEGACY [0.5,1.0] vocab gives —
    # correct there, wrong when applied to a [0.75]-trained artifact.
    lf, ls = LEGACY
    assert _train_action_idx(False, 0, "allin", 1.0,
                             first_fracs=lf, subseq_fracs=ls) == 3


def test_regression_facing_allin_is_index_3_under_0_75():
    """Under the published [0.75] vocab the FACING all-in must be index 3
    (NOT 4, the [0.5,1.0] value)."""
    first, subseq = SERVED
    assert _train_action_idx(True, 0, "allin", 1.0,
                             first_fracs=first, subseq_fracs=subseq) == 3
    assert _reference_action_index(True, 0, "allin", 1.0,
                                   first_fracs=first, subseq_fracs=subseq) == 3
    lf, ls = LEGACY
    assert _train_action_idx(True, 0, "allin", 1.0,
                             first_fracs=lf, subseq_fracs=ls) == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
