"""Aggregate solve-ladder rows into per-cell tables.

usage: python scripts/solve_ladder_report.py OUT_DIR [OUT_DIR ...] [--json FILE]

Reads ``ladder.jsonl`` under each directory (scripts/solve_ladder_bench.py),
drops ``done`` markers and failed rows, and prints one block per
(street, n_players) cell: cost per thousand iterations, iterations the 3 s
budget buys, the production rung, the noise floor, every rung's distance to the
reference, and the plateau rung with its wall time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.solve_ladder_metrics import aggregate  # noqa: E402


def load_rows(dirs: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for directory in dirs:
        path = Path(directory) / "ladder.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def split_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    """Solved rows versus failures (refusals and errors), markers dropped."""
    solved = [r for r in rows if r.get("rung") != "done" and r.get("outcome") == "solved"]
    failed = [r for r in rows if r.get("rung") != "done" and r.get("outcome") != "solved"]
    return {"solved": solved, "failed": failed}


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _rung_order(label: str) -> tuple:
    if label == "ref_a":
        return (3, 0)
    if label == "ref_b":
        return (2, 0)
    if label == "production":
        return (0, 0)
    return (1, int(label.split("_")[1]) if label.startswith("it_") else 0)


def render(cells: Mapping[str, Any], failed: Sequence[Mapping[str, Any]]) -> str:
    lines: List[str] = []
    for key, cell in cells.items():
        lines.append(f"## {key}: {cell['spots']} spots")
        lines.append(
            f"cost {_fmt(cell['ms_per_kiter'], 1)} ms per 1k iterations; "
            f"3 s buys {_fmt(cell['iterations_in_3s'], 0)} iterations; "
            f"plateau {cell['plateau_iterations']} iterations "
            f"({_fmt(cell['seconds_to_plateau'], 1)} s)"
        )
        floor = cell["noise_floor"]
        lines.append(
            f"noise floor (ref B vs ref A): mean TV {_fmt(floor['mean_tv'])}, "
            f"root TV {_fmt(floor['root_tv'])}, hero mix TV {_fmt(floor['hero_mix_tv'])}"
        )
        lines.append("")
        lines.append("| rung | spots | iterations run | solve s | ms/1k it | mean TV vs ref | hero mix TV | river BR-gap % pot |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for label in sorted(cell["rungs"], key=_rung_order):
            info = cell["rungs"][label]
            lines.append(
                f"| {label} | {info['spots']} | {_fmt(info['iterations_run'], 0)} | "
                f"{_fmt(info['solve_s'], 2)} | {_fmt(info['ms_per_kiter'], 1)} | "
                f"{_fmt(info['mean_tv'])} +/- {_fmt(info['mean_tv_se'])} | "
                f"{_fmt(info['hero_mix_tv'])} +/- {_fmt(info['hero_mix_tv_se'])} | "
                f"{_fmt(info['river_brgap_pct_pot'], 1)} |"
            )
        lines.append("")
    if failed:
        counts: Dict[str, int] = {}
        for row in failed:
            key = f"{row.get('street')}_{row.get('n_players')}p {row.get('rung')} {row.get('outcome')}"
            counts = {**counts, key: counts.get(key, 0) + 1}
        lines.append("## failed or refused rows")
        for key, count in sorted(counts.items()):
            lines.append(f"- {key}: {count}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    parts = split_rows(load_rows(args.dirs))
    if not parts["solved"]:
        print("no solved ladder rows found")
        return 1
    cells = aggregate(parts["solved"])
    print(render(cells, parts["failed"]))
    if args.json:
        Path(args.json).write_text(
            json.dumps({"cells": cells, "failed": len(parts["failed"])}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
