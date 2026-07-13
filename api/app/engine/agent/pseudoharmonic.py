"""Pseudo-harmonic off-tree action mapping (Gap 4(1), Pluribus conformance).

Pluribus translates a slightly off-tree bet onto the blueprint's nearest
sizes with PSEUDO-HARMONIC weighting (Ganzfried & Sandholm 2013), not
absolute/log distance — the harmonic (reciprocal) transform captures that a
50% pot bet is strategically closer to 33% than to 100%. The live path
previously snapped by abs distance (action_translation via
_snap_amount_to_tree) and log distance (game_state observed raises); this
module supplies the conformant boundary for both
(PLURIBUS_EXACT_CONFORMANCE_SPEC_2026-06-26.md §Gap 4, "exact change" item 1).

`pseudoharmonic_map` is a VENDORED pure-math copy of
poker-solver/python/off_tree.py:pseudoharmonic_map (keep in sync). Vendored
rather than imported because agent modules must not depend on the engine's
poker-solver sys.path bootstrap (engine_paths) having run — a silent
ImportError fallback here would recreate the exact "tag says pseudoharmonic
but the code snaps" bug this closes.

SCALE INVARIANCE (why game_state can pass raw CHIP amounts): the weight
    w_hi = (1/b_lo - 1/b) / (1/b_lo - 1/b_hi)
uses only reciprocals, so a common scale factor (the pot) cancels —
pseudoharmonic_pick(720, [500, 1000]) == pseudoharmonic_pick(0.72, [0.5, 1.0]).

DETERMINISM: Pluribus randomizes between the two brackets with these weights;
this module selects the MAP bracket (max weight, tie -> smaller size).
GameState.advance_observed must stay deterministic (replay + cache keys), so
the randomized mapping is deferred until a per-hand seeded rng exists — the
w=0.5 pseudo-harmonic threshold is still the conformant decision boundary.

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
    """Map an actual bet size onto the nearest tree sizes with weights.

    Vendored verbatim (math) from poker-solver/python/off_tree.py. Returns
    [(index, weight)] pairs summing to 1.0; index is the position in
    tree_bet_fracs. Exact match (|b - tf| < 0.001) or an out-of-range size
    returns a single pair with weight 1.0. Inputs may be pot fractions or
    chip amounts (scale-invariant; see module docstring).
    """
    if not tree_bet_fracs:
        return []

    b = actual_bet_frac

    for i, tf in enumerate(tree_bet_fracs):
        if abs(b - tf) < 0.001:
            return [(i, 1.0)]

    smaller = [(i, tf) for i, tf in enumerate(tree_bet_fracs) if tf < b]
    larger = [(i, tf) for i, tf in enumerate(tree_bet_fracs) if tf > b]

    if not smaller:
        # Below all tree sizes — map entirely to the smallest size's index.
        idx = min(range(len(tree_bet_fracs)), key=lambda i: tree_bet_fracs[i])
        return [(idx, 1.0)]

    if not larger:
        # Above all tree sizes — map entirely to the largest size's index.
        idx = max(range(len(tree_bet_fracs)), key=lambda i: tree_bet_fracs[i])
        return [(idx, 1.0)]

    lo_idx, b_lo = max(smaller, key=lambda x: x[1])
    hi_idx, b_hi = min(larger, key=lambda x: x[1])

    if b_lo <= 0 or b <= 0 or b_hi <= 0:
        # Degenerate case: linear interpolation.
        w_hi = (b - b_lo) / (b_hi - b_lo)
    else:
        inv_lo = 1.0 / b_lo
        inv_b = 1.0 / b
        inv_hi = 1.0 / b_hi
        denom = inv_lo - inv_hi
        if abs(denom) < 1e-10:
            w_hi = 0.5
        else:
            w_hi = (inv_lo - inv_b) / denom

    w_hi = max(0.0, min(1.0, w_hi))
    w_lo = 1.0 - w_hi

    return [(lo_idx, w_lo), (hi_idx, w_hi)]


def pseudoharmonic_pick(
    actual: float,
    tree_values: Sequence[float],
) -> int:
    """MAP index of the pseudo-harmonic mapping (tie -> the SMALLER size).

    The deterministic selection both conformance call sites use. tree_values
    need not be sorted (the map brackets by value and returns positions in
    the input sequence).

    Raises:
        ValueError: empty tree_values.
    """
    mapping = pseudoharmonic_map(actual, tree_values)
    if not mapping:
        raise ValueError("pseudoharmonic_pick requires non-empty tree_values")
    if len(mapping) == 1:
        return mapping[0][0]
    (lo_idx, w_lo), (hi_idx, w_hi) = mapping
    # Strict '>' keeps the smaller size on the exact w=0.5 boundary — the
    # conservative call at the harmonic midpoint.
    return hi_idx if w_hi > w_lo else lo_idx
