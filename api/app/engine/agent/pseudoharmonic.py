"""Pseudo-harmonic off-tree action mapping (Gap 4(1), Pluribus conformance).

Pluribus translates a slightly off-tree bet onto the blueprint's nearest
sizes with the PSEUDO-HARMONIC mapping (Ganzfried & Sandholm 2013):

    f_lo(x) = ((B - x) * (1 + A)) / ((B - A) * (1 + x))

with A = lower tree size, B = upper tree size, x = the observed bet, all as
FRACTIONS OF THE POT (the "+1" terms are the pot itself). f_lo is the weight
on the LOWER size; the deterministic decision boundary (f_lo = 0.5) sits at
x* = (A + B + 2AB) / (A + B + 2). The live path previously snapped by abs
distance (action_translation via _snap_amount_to_tree) and log distance
(game_state observed raises); this module supplies the conformant boundary
for both (PLURIBUS_EXACT_CONFORMANCE_SPEC_2026-06-26.md §Gap 4, item 1).

⚠️ NOT the reciprocal "harmonic" weighting. The 2026-07-13 merge review
caught that poker-solver/python/off_tree.py:pseudoharmonic_map — and the
first draft of this module, vendored from it — computes the plain harmonic
mapping w_hi=(1/A-1/x)/(1/A-1/B) (boundary at the harmonic mean 2AB/(A+B)),
which is NOT the G&S formula and flips every bet between the two boundaries
(e.g. (0.667, 0.714)×pot on a {50%,100%} vocab). The upstream research
module still carries the harmonic form (used only by research/legacy paths);
this module is the conformant implementation, no longer a vendored copy.

⚠️ POT-FRACTION INPUTS REQUIRED. The G&S "(1+x)" terms are pot-relative, so
the mapping is NOT scale-invariant — callers must divide chip amounts by the
pot at the decision (game_state passes increment/pot and survivors/pot).

DETERMINISM: Pluribus randomizes between the two brackets with these
weights; this module selects the MAP bracket (max weight, tie -> smaller
size). GameState.advance_observed must stay deterministic (replay + cache
keys), so the randomized mapping is deferred until a per-hand seeded rng
exists — the f_lo=0.5 pseudo-harmonic threshold is still the conformant
decision boundary.

Kill-switch: NEXUS_OFFTREE_SNAP=legacy restores the pre-Gap-4(1) abs/log
snapping at both call sites (read per call — no import-order dependence).
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple


def use_pseudoharmonic_snap() -> bool:
    """Whether the pseudo-harmonic boundary is active (default ON).

    NEXUS_OFFTREE_SNAP=legacy reverts both call sites to the historical
    abs/log-distance snap.
    """
    return os.environ.get("NEXUS_OFFTREE_SNAP", "pseudoharmonic") != "legacy"


def pseudoharmonic_map(
    actual_bet_frac: float,
    tree_bet_fracs: Sequence[float],
) -> List[Tuple[int, float]]:
    """Map an observed bet onto the bracketing tree sizes with G&S weights.

    All inputs are POT FRACTIONS (see module docstring — not scale-
    invariant). Returns [(index, weight)] pairs summing to 1.0; index is the
    position in tree_bet_fracs. Exact match (|x - tf| < 0.001) or an
    out-of-range size returns a single pair with weight 1.0.
    """
    if not tree_bet_fracs:
        return []

    x = actual_bet_frac

    for i, tf in enumerate(tree_bet_fracs):
        if abs(x - tf) < 0.001:
            return [(i, 1.0)]

    smaller = [(i, tf) for i, tf in enumerate(tree_bet_fracs) if tf < x]
    larger = [(i, tf) for i, tf in enumerate(tree_bet_fracs) if tf > x]

    if not smaller:
        # Below all tree sizes — map entirely to the smallest size's index.
        idx = min(range(len(tree_bet_fracs)), key=lambda i: tree_bet_fracs[i])
        return [(idx, 1.0)]

    if not larger:
        # Above all tree sizes — map entirely to the largest size's index.
        idx = max(range(len(tree_bet_fracs)), key=lambda i: tree_bet_fracs[i])
        return [(idx, 1.0)]

    lo_idx, a = max(smaller, key=lambda t: t[1])
    hi_idx, b = min(larger, key=lambda t: t[1])

    denom = (b - a) * (1.0 + x)
    if denom <= 1e-12:
        w_lo = 0.5  # degenerate bracket; split evenly
    else:
        # Ganzfried & Sandholm 2013, eq. for the lower-size weight.
        w_lo = ((b - x) * (1.0 + a)) / denom

    w_lo = max(0.0, min(1.0, w_lo))
    w_hi = 1.0 - w_lo

    return [(lo_idx, w_lo), (hi_idx, w_hi)]


def pseudoharmonic_pick(
    actual: float,
    tree_values: Sequence[float],
) -> int:
    """MAP index of the pseudo-harmonic mapping (tie -> the SMALLER size).

    `actual` and `tree_values` must share POT-FRACTION units (callers with
    chip amounts divide by the pot first). tree_values need not be sorted
    (the map brackets by value and returns positions in the input sequence).

    Raises:
        ValueError: empty tree_values.
    """
    mapping = pseudoharmonic_map(actual, tree_values)
    if not mapping:
        raise ValueError("pseudoharmonic_pick requires non-empty tree_values")
    if len(mapping) == 1:
        return mapping[0][0]
    (lo_idx, w_lo), (hi_idx, w_hi) = mapping
    # Strict '>' keeps the smaller size on the exact f_lo=0.5 boundary — the
    # conservative call at the pseudo-harmonic threshold.
    return hi_idx if w_hi > w_lo else lo_idx
