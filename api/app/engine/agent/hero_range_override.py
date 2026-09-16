"""Replace the hero's solve range with weights over the 1326 physical combos.

Research hook (SCOREBOARD_RESULTS section 34): a driver that tracks the
hero's own reach (its earlier-street search strategies) supplies per-combo
weights; each grid row receives the summed weight of the live combos it
stands for under the board's suit isomorphism.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

Row = Tuple[int, int, float]
N_COMBOS = 1326


def _combos() -> List[Tuple[int, int]]:
    return [(low, high) for low in range(52) for high in range(low + 1, 52)]


def apply_hero_range(
    player_ranges: Sequence[Sequence[Row]],
    hero_idx: int,
    board: Sequence[int],
    combo_weights: np.ndarray,
) -> Tuple[List[List[Row]], Dict[str, object]]:
    """Hero rows re-weighted from ``combo_weights`` (1326 entries, combo order).

    Rows keep their cards; a row's weight becomes the total weight of the live
    combos whose canonical form is that row.  Zero total mass leaves the
    ranges unchanged and reports ``applied=False``.
    """
    from app.services.suit_isomorphism import canonical_combo

    weights = np.asarray(combo_weights, dtype=np.float64)
    if weights.shape != (N_COMBOS,):
        raise ValueError(f"combo_weights must have {N_COMBOS} entries, got {weights.shape}")
    blocked = {int(card) for card in board}
    hero_rows = list(player_ranges[hero_idx])
    index = {(int(c0), int(c1)): i for i, (c0, c1, _w) in enumerate(hero_rows)}
    accumulated = np.zeros(len(hero_rows), dtype=np.float64)
    unmapped = 0.0
    for j, (c0, c1) in enumerate(_combos()):
        weight = float(weights[j])
        if weight <= 0.0 or c0 in blocked or c1 in blocked:
            continue
        row = index.get(canonical_combo(c0, c1, list(board)))
        if row is None:
            unmapped += weight
            continue
        accumulated[row] += weight
    total = float(accumulated.sum())
    stats: Dict[str, object] = {
        "applied": total > 0.0, "mass": total, "unmapped_mass": unmapped,
        "rows_positive": int((accumulated > 0.0).sum()), "rows": len(hero_rows),
    }
    if total <= 0.0:
        return [list(rows) for rows in player_ranges], stats
    replaced = [(c0, c1, float(accumulated[i])) for i, (c0, c1, _w) in enumerate(hero_rows)]
    return [replaced if p == hero_idx else list(rows) for p, rows in enumerate(player_ranges)], stats
