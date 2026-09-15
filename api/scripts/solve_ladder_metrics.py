"""Pure convergence and timing metrics for the solve-iteration ladder.

A ladder solves one captured spot at rising iteration budgets, twice per rung
with two seeds. The strategy of a solve is ``sigma``: ``{decision_node: rows}``
where ``rows[hand_row, action]`` is that player's action probability, in the
tree snapshot's node order and hand order (``scripts/river_brgap_capture.plain_tree``).
Two distances are reported per rung: seed A versus seed B at the same budget
(self-consistency: the noise floor of a sampled-leaf, Hogwild solve at that
budget) and seed A versus the top rung's seed A (convergence toward the
largest budget).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

T_DECISION = 0
PAIR_FIELDS = ("pair_mean_tv", "pair_root_tv", "pair_hero_mix_tv")
TOP_FIELDS = ("mean_tv", "root_tv", "hero_mix_tv")
MIN_COST_ITERATIONS = 1_000


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


def ms_per_kiter(solve_s: Optional[float], iterations_run: Optional[int]) -> Optional[float]:
    """Milliseconds of solve wall per thousand iterations."""
    if solve_s is None or iterations_run is None or iterations_run <= 0:
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
    """Smallest iteration count whose distance to the top rung is within
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


def _clean(values: Sequence[Any]) -> List[float]:
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def _mean(values: Sequence[Any]) -> Optional[float]:
    clean = _clean(values)
    return float(np.mean(clean)) if clean else None


def _se(values: Sequence[Any]) -> Optional[float]:
    clean = _clean(values)
    if len(clean) < 2:
        return None
    return float(np.std(clean, ddof=1) / math.sqrt(len(clean)))


def cost_fit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """Least-squares ``solve_s = fixed_s + marginal_s * iterations`` over both seeds.

    Every solve pays a fixed cost (thread start, per-solve memo warm-up) that
    dominates small budgets, so the marginal cost is the slope, not the ratio.
    Returns the fixed overhead in seconds, the marginal ms per 1k iterations,
    and the iterations a 3,000 ms budget buys after the overhead.
    """
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        for solve_key, iter_key in (("solve_s", "iterations_run"), ("solve_s_b", "iterations_run_b")):
            run, solve_s = row.get(iter_key), row.get(solve_key)
            if run is not None and solve_s is not None and int(run) > 0:
                xs.append(float(run))
                ys.append(float(solve_s))
    empty = {"fixed_s": None, "marginal_ms_per_kiter": None, "iterations_in_3s": None, "points": len(xs)}
    if len(xs) < 3 or len(set(xs)) < 2:
        return empty
    slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    if slope <= 0.0:
        return empty
    budget = 3.0 - max(float(intercept), 0.0)
    return {
        "fixed_s": float(intercept),
        "marginal_ms_per_kiter": float(slope) * 1e6,
        "iterations_in_3s": float(budget / slope) if budget > 0.0 else 0.0,
        "points": len(xs),
    }


def _rung_costs(rows: Sequence[Mapping[str, Any]]) -> List[float]:
    costs: List[float] = []
    for row in rows:
        for solve_key, iter_key in (("solve_s", "iterations_run"), ("solve_s_b", "iterations_run_b")):
            run = row.get(iter_key)
            if run is not None and int(run) >= MIN_COST_ITERATIONS:
                cost = ms_per_kiter(row.get(solve_key), run)
                if cost is not None:
                    costs.append(cost)
    return costs


def summarize_rung(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "spots": len(rows),
        "iterations": _mean([r.get("iterations") for r in rows]),
        "iterations_run": _mean([r.get("iterations_run") for r in rows]),
        "solve_s": _mean([r.get("solve_s") for r in rows]),
        "construct_s": _mean([r.get("construct_s") for r in rows]),
        "num_nodes": _mean([r.get("num_nodes") for r in rows]),
        "num_leaf_nodes": _mean([r.get("num_leaf_nodes") for r in rows]),
        "ms_per_kiter": _mean(_rung_costs(rows)),
        "river_brgap_pct_pot": _mean([r.get("river_brgap_pct_pot") for r in rows]),
        "river_brgap_pct_pot_se": _se([r.get("river_brgap_pct_pot") for r in rows]),
        "river_brgap_avg_pct_pot": _mean([r.get("river_brgap_avg_pct_pot") for r in rows]),
        "river_brgap_avg_pct_pot_se": _se([r.get("river_brgap_avg_pct_pot") for r in rows]),
    }
    for field in PAIR_FIELDS + TOP_FIELDS:
        summary = {
            **summary,
            field: _mean([r.get(field) for r in rows]),
            f"{field}_se": _se([r.get(field) for r in rows]),
        }
    return summary


def top_label(rungs: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    """The ladder rung with the largest requested iteration count."""
    candidates = [
        (info["iterations"], label)
        for label, info in rungs.items()
        if label.startswith("it_") and info.get("iterations") is not None
    ]
    return max(candidates)[1] if candidates else None


_NUMERIC_MEAN_FIELDS = (
    "iterations", "iterations_run", "iterations_run_b", "solve_s", "solve_s_b",
    "construct_s", "num_nodes", "num_leaf_nodes", "river_brgap_pct_pot",
    "river_brgap_avg_pct_pot",
) + PAIR_FIELDS + TOP_FIELDS


def collapse_duplicates(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Average rows that share a hand (``solve_key``) and rung into one row.

    Rows without a key are kept as they are. Non-numeric fields come from the
    first row of the group.
    """
    groups: Dict[tuple, List[Mapping[str, Any]]] = {}
    order: List[tuple] = []
    for index, row in enumerate(rows):
        key = (row.get("solve_key") or f"row{index}", str(row.get("rung")))
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for key in order:
        group = groups[key]
        merged: Dict[str, Any] = {**group[0], "spot": group[0].get("solve_key") or group[0].get("spot")}
        if len(group) > 1:
            for field in _NUMERIC_MEAN_FIELDS:
                merged[field] = _mean([r.get(field) for r in group])
            merged["arms"] = len(group)
        out.append(merged)
    return out


def aggregate_cell(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize ladder rows of one (street, n_players) cell.

    Rows carry ``rung`` (``it_N``, ``production`` or ``avg_N``), requested
    ``iterations``, per-seed ``iterations_run``/``solve_s`` (``_b`` for seed B),
    the pair distances (``pair_mean_tv``, ``pair_root_tv``,
    ``pair_hero_mix_tv``), the distances to the top rung (``mean_tv``,
    ``root_tv``, ``hero_mix_tv``) and optional river best-response gaps. Rows
    of one hand's seat arms are averaged first so the spread is across hands.
    """
    rows = collapse_duplicates(rows)
    by_rung: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_rung.setdefault(str(row["rung"]), []).append(row)
    rungs = {label: summarize_rung(group) for label, group in by_rung.items()}
    top = top_label(rungs)
    floor = {
        field: (rungs[top][f"pair_{field}"] if top is not None else None)
        for field in TOP_FIELDS
    }
    ladder = [
        (info["iterations"], info["mean_tv"])
        for label, info in rungs.items()
        if label.startswith("it_") and label != top and info["iterations"] is not None
    ]
    plateau = (
        plateau_rung(ladder, floor["mean_tv"]) if floor["mean_tv"] is not None else None
    )
    ladder_rows = [r for r in rows if str(r["rung"]).startswith("it_")]
    # Median: the first solves of a process can still be paying page faults.
    costs = _rung_costs(ladder_rows)
    cost = float(np.median(costs)) if costs else None
    fit = cost_fit(ladder_rows)
    marginal = fit["marginal_ms_per_kiter"]
    return {
        "spots": len({r.get("spot") for r in rows}),
        "top_rung": top,
        "noise_floor": floor,
        "rungs": rungs,
        "plateau_iterations": plateau,
        "ms_per_kiter": cost,
        "iterations_in_3s": iterations_in(3000.0, cost),
        "cost_fit": fit,
        "seconds_to_plateau": (
            float(fit["fixed_s"]) + (plateau / 1000.0) * marginal / 1000.0
            if plateau is not None and marginal is not None
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
