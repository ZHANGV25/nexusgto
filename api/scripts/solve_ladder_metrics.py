"""Pure convergence and timing metrics for the solve-iteration ladder.

A ladder solves one captured spot at rising iteration budgets and compares
each rung's strategy with a high-budget reference solve of the same spot.
The strategy of a solve is ``sigma``: ``{decision_node: rows}`` where
``rows[hand_row, action]`` is that player's action probability, in the tree
snapshot's node order and hand order (``scripts/river_brgap_capture.plain_tree``).
Two reference solves with different seeds give the noise floor: the distance
a converged solve keeps from itself under sampled leaves and Hogwild threads.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

T_DECISION = 0


def _weighted_rows(tree: Mapping[str, Any], player: int) -> np.ndarray:
    hands = tree["player_hands"][player]
    return np.array([float(h[2]) > 0.0 for h in hands], dtype=bool)


def sigma_distance(
    sigma_a: Mapping[int, np.ndarray],
    sigma_b: Mapping[int, np.ndarray],
    tree: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    """Total-variation distance between two strategies on one tree.

    ``mean_tv`` averages the per-row TV (half the L1 distance) over every
    weighted hand row of every decision node present in both strategies;
    ``root_tv`` is the same average over the root node only. Rows the range
    holds at zero weight are never played and are excluded.
    """
    nodes = tree["nodes"]
    per_node: List[float] = []
    root_tv: Optional[float] = None
    for node_idx, node in enumerate(nodes):
        if node["type"] != T_DECISION:
            continue
        rows_a = sigma_a.get(node_idx)
        rows_b = sigma_b.get(node_idx)
        if rows_a is None or rows_b is None:
            continue
        a = np.asarray(rows_a, dtype=np.float64)
        b = np.asarray(rows_b, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(
                f"node {node_idx}: strategy shapes differ {a.shape} vs {b.shape}"
            )
        mask = _weighted_rows(tree, node["player"])
        if mask.shape[0] != a.shape[0]:
            raise ValueError(
                f"node {node_idx}: {a.shape[0]} rows for {mask.shape[0]} hands"
            )
        if not mask.any():
            continue
        tv = 0.5 * np.abs(a[mask] - b[mask]).sum(axis=1)
        value = float(tv.mean())
        per_node.append(value)
        if node_idx == 0:
            root_tv = value
    return {
        "mean_tv": float(np.mean(per_node)) if per_node else None,
        "root_tv": root_tv,
        "nodes": len(per_node),
    }


def mix_distance(mix_a: Mapping[str, float], mix_b: Mapping[str, float]) -> float:
    """Total-variation distance between two action mixes keyed by label."""
    labels = set(mix_a) | set(mix_b)
    return 0.5 * sum(abs(float(mix_a.get(k, 0.0)) - float(mix_b.get(k, 0.0))) for k in labels)


def ms_per_kiter(solve_s: float, iterations_run: int) -> Optional[float]:
    """Milliseconds of solve wall per thousand iterations."""
    if iterations_run <= 0:
        return None
    return 1000.0 * float(solve_s) / (float(iterations_run) / 1000.0)


def iterations_in(budget_ms: float, ms_per_kiter_value: Optional[float]) -> Optional[float]:
    """Iterations a budget buys at a measured cost per thousand iterations."""
    if ms_per_kiter_value is None or ms_per_kiter_value <= 0.0:
        return None
    return 1000.0 * float(budget_ms) / float(ms_per_kiter_value)


def plateau_rung(
    rung_values: Sequence[tuple],
    floor: float,
    tolerance: float = 1.5,
) -> Optional[int]:
    """Smallest iteration count whose distance to the reference is within
    ``tolerance`` times the noise floor, provided every larger rung stays
    there too (a single lucky rung does not count as converged).

    ``rung_values`` is ``[(iterations, distance), ...]``; None distances are
    skipped. Returns None when no rung qualifies.
    """
    ordered = sorted(
        (int(it), float(d)) for it, d in rung_values if d is not None and not math.isnan(d)
    )
    if not ordered:
        return None
    threshold = tolerance * float(floor)
    result: Optional[int] = None
    for idx, (iterations, distance) in enumerate(ordered):
        if distance <= threshold and all(d <= threshold for _, d in ordered[idx:]):
            result = iterations
            break
    return result


def _mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else None


def _se(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(clean) < 2:
        return None
    return float(np.std(clean, ddof=1) / math.sqrt(len(clean)))


def aggregate_cell(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize ladder rows of one (street, n_players) cell.

    Rows carry ``rung`` (a label), ``iterations`` (requested), ``iterations_run``,
    ``solve_s``, ``construct_s``, ``mean_tv``, ``root_tv``, ``hero_mix_tv`` and
    optionally ``river_brgap_pct_pot`` and ``spot``. Reference-B rows carry the
    noise floor in the same distance fields (measured against reference A).
    """
    by_rung: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_rung.setdefault(str(row["rung"]), []).append(row)
    floor_rows = by_rung.get("ref_b", [])
    floor = {
        "mean_tv": _mean([r.get("mean_tv") for r in floor_rows]),
        "root_tv": _mean([r.get("root_tv") for r in floor_rows]),
        "hero_mix_tv": _mean([r.get("hero_mix_tv") for r in floor_rows]),
    }
    rungs: Dict[str, Any] = {}
    for label, group in by_rung.items():
        costs = [ms_per_kiter(r["solve_s"], r["iterations_run"]) for r in group]
        rungs[label] = {
            "spots": len(group),
            "iterations": _mean([r.get("iterations") for r in group]),
            "iterations_run": _mean([r.get("iterations_run") for r in group]),
            "solve_s": _mean([r.get("solve_s") for r in group]),
            "construct_s": _mean([r.get("construct_s") for r in group]),
            "ms_per_kiter": _mean(costs),
            "mean_tv": _mean([r.get("mean_tv") for r in group]),
            "mean_tv_se": _se([r.get("mean_tv") for r in group]),
            "root_tv": _mean([r.get("root_tv") for r in group]),
            "hero_mix_tv": _mean([r.get("hero_mix_tv") for r in group]),
            "hero_mix_tv_se": _se([r.get("hero_mix_tv") for r in group]),
            "river_brgap_pct_pot": _mean([r.get("river_brgap_pct_pot") for r in group]),
        }
    ladder = [
        (info["iterations"], info["mean_tv"])
        for label, info in rungs.items()
        if label.startswith("it_") and info["iterations"] is not None
    ]
    plateau = (
        plateau_rung(ladder, floor["mean_tv"]) if floor["mean_tv"] is not None else None
    )
    cost_rows = [
        ms_per_kiter(r["solve_s"], r["iterations_run"])
        for r in rows
        if str(r["rung"]).startswith("it_") and r.get("iterations_run", 0) >= 10_000
    ]
    cost = _mean(cost_rows)
    return {
        "spots": len({r.get("spot") for r in rows}),
        "noise_floor": floor,
        "rungs": rungs,
        "plateau_iterations": plateau,
        "ms_per_kiter": cost,
        "iterations_in_3s": iterations_in(3000.0, cost),
        "seconds_to_plateau": (
            (plateau / 1000.0) * cost / 1000.0
            if plateau is not None and cost is not None
            else None
        ),
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate ladder rows by (street, n_players)."""
    cells: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        key = f"{row['street']}_{int(row['n_players'])}p"
        cells.setdefault(key, []).append(row)
    return {key: aggregate_cell(group) for key, group in sorted(cells.items())}
