"""Replace the hero's solve range with weights over the 1326 physical combos.

Research hook (SCOREBOARD_RESULTS section 34): a driver that tracks the
hero's own reach (its earlier-street search strategies) supplies per-combo
weights; each grid row receives the summed weight of the live combos it
stands for under the board's suit isomorphism.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from app.engine.agent.game_state import FULL_SEAT

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


def _override_for_seat(
    range_overrides: Mapping[object, object],
    seat_name: str,
) -> object:
    if seat_name in range_overrides:
        return range_overrides[seat_name]
    seat_id = FULL_SEAT.get(seat_name)
    if seat_id is not None and seat_id in range_overrides:
        return range_overrides[seat_id]
    if seat_id is not None and str(seat_id) in range_overrides:
        return range_overrides[str(seat_id)]
    return None


def apply_range_overrides(
    player_ranges: Sequence[Sequence[Row]],
    positions_sorted: Sequence[str],
    board: Sequence[int],
    range_overrides: Mapping[object, object],
) -> Tuple[List[List[Row]], Dict[str, object]]:
    """Apply per-seat 1326-combo overrides to the matching solver-player rows.

    ``positions_sorted`` is the solver-player -> full-seat-name mapping from
    ``build_solve_inputs``.  It is intentionally used instead of
    ``player_seats`` because true-HU leaf-key mirroring may alter
    ``player_seats`` without changing the player/range order.
    """
    if len(positions_sorted) != len(player_ranges):
        raise ValueError(
            "positions_sorted length must match player_ranges: "
            f"{len(positions_sorted)} vs {len(player_ranges)}"
        )

    ranges = [list(rows) for rows in player_ranges]
    players: Dict[str, object] = {}
    configured = 0
    applied = 0
    for player_idx, seat_name in enumerate(positions_sorted):
        override = _override_for_seat(range_overrides, str(seat_name))
        if override is None:
            players[str(seat_name)] = {
                "configured": False,
                "applied": False,
                "player_idx": int(player_idx),
            }
            continue
        configured += 1
        ranges, stats = apply_hero_range(ranges, player_idx, board, override)
        stats = {
            **stats,
            "configured": True,
            "player_idx": int(player_idx),
        }
        if stats.get("applied"):
            applied += 1
        players[str(seat_name)] = stats

    return ranges, {
        "players": players,
        "players_configured": configured,
        "players_applied": applied,
        "players_unapplied": configured - applied,
        "applied": applied > 0,
    }
