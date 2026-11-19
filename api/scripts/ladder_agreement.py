"""Convergence ladder report for root value audits run on the same deals.

Each run directory holds ``spots.jsonl`` and ``tables.npz`` from
``root_value_audit.py --save-tables``. Per run: iterations, tree size, the
search-minus-blueprint root value and the reach-weighted distance between the
search's root strategy and the blueprint's. Per pair: the same distance between
two runs' root strategies (seed agreement) and the paired value difference.

  python scripts/ladder_agreement.py RUN_DIR [RUN_DIR ...] --pairs a:b [a:b ...] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.lbr_core import mean_se


def strategy_distance(table_a: np.ndarray, table_b: np.ndarray, reach: np.ndarray) -> Optional[float]:
    """Reach-weighted total variation distance between two [hands, actions] strategy tables.

    Hands either table leaves empty (blocked or outside the solved range) are skipped.
    """
    a, b, w = np.asarray(table_a, dtype=float), np.asarray(table_b, dtype=float), np.asarray(reach, dtype=float)
    if a.shape != b.shape or a.shape[0] != w.shape[0]:
        raise ValueError(f"shape mismatch: {a.shape} {b.shape} {w.shape}")
    live = (a.sum(axis=1) > 0) & (b.sum(axis=1) > 0) & (w > 0)
    if not live.any():
        return None
    tv = 0.5 * np.abs(a[live] - b[live]).sum(axis=1)
    return float((tv * w[live]).sum() / w[live].sum())


def load_run(run_dir: Path) -> Dict[str, Any]:
    rows = [json.loads(line) for line in (run_dir / "spots.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tables_path = run_dir / "tables.npz"
    tables = dict(np.load(tables_path)) if tables_path.exists() else {}
    return {"name": run_dir.name, "rows": rows, "tables": tables}


def _spot_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (row["deal"], tuple(row["board"]), row["pot"], row["hero_seat"])


def run_report(run: Dict[str, Any]) -> Dict[str, Any]:
    rows, tables = run["rows"], run["tables"]
    stats = [r.get("solve_stats") or {} for r in rows]
    distances = [
        strategy_distance(tables[f"search_{i}"], tables[f"blueprint_{i}"], tables[f"true_hero_{i}"])
        for i in range(len(rows)) if f"search_{i}" in tables
    ]
    return {
        "spots": len(rows),
        "iterations_run": mean_se([s["iterations_run"] for s in stats if s.get("iterations_run") is not None]),
        "solve_s": mean_se([s["solve_s"] for s in stats if s.get("solve_s") is not None]),
        "num_nodes": mean_se([s["num_nodes"] for s in stats if s.get("num_nodes") is not None]),
        "search_minus_blueprint_pct": mean_se([r["true"]["search_minus_blueprint_pct"] for r in rows]),
        "bet_freq_search": mean_se([r["true"]["bet_freq_search"] for r in rows]),
        "bet_freq_blueprint": mean_se([r["true"]["bet_freq_blueprint"] for r in rows]),
        "distance_to_blueprint": mean_se([d for d in distances if d is not None]),
    }


def pair_report(run_a: Dict[str, Any], run_b: Dict[str, Any]) -> Dict[str, Any]:
    index_b = {_spot_key(row): i for i, row in enumerate(run_b["rows"])}
    distances: List[float] = []
    value_diffs: List[float] = []
    for i, row in enumerate(run_a["rows"]):
        j = index_b.get(_spot_key(row))
        if j is None:
            continue
        value_diffs.append(row["true"]["search_minus_blueprint_pct"] - run_b["rows"][j]["true"]["search_minus_blueprint_pct"])
        if f"search_{i}" in run_a["tables"] and f"search_{j}" in run_b["tables"]:
            d = strategy_distance(run_a["tables"][f"search_{i}"], run_b["tables"][f"search_{j}"], run_a["tables"][f"true_hero_{i}"])
            if d is not None:
                distances.append(d)
    return {
        "matched_spots": len(value_diffs),
        "strategy_distance": mean_se(distances),
        "value_a_minus_b_pct": mean_se(value_diffs),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pairs", nargs="*", default=[], help="name_a:name_b run directory names to compare")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    runs = {Path(p).name: load_run(Path(p)) for p in args.runs}
    report: Dict[str, Any] = {"runs": {name: run_report(run) for name, run in runs.items()}, "pairs": {}}
    for pair in args.pairs:
        name_a, _, name_b = pair.partition(":")
        report["pairs"][pair] = pair_report(runs[name_a], runs[name_b])
    text = json.dumps(report, indent=2, default=float)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
