"""Pool independent run_match output directories into one estimate.

Each directory's ``deal_groups.jsonl`` holds independent deal groups (distinct
master seeds give distinct decks). Pooling concatenates groups, so the
standard error comes from the pooled between-group spread rather than an
average of per-run intervals.

    python scripts/pool_match_results.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

BIG_BLIND = 100


def _estimate(values: Sequence[float]) -> Dict[str, float]:
    n = len(values)
    mean = math.fsum(values) / n
    variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else float("nan")
    stderr = math.sqrt(variance / n) if n > 1 else float("nan")
    scale = 100.0 / BIG_BLIND
    return {
        "n_groups": n,
        "mean_bb_per_100": round(mean * scale, 4),
        "stderr_bb_per_100": round(stderr * scale, 4),
        "ci95_bb_per_100": [
            round((mean - 1.96 * stderr) * scale, 4),
            round((mean + 1.96 * stderr) * scale, 4),
        ],
        "group_sd_bb_per_hand": round(math.sqrt(variance) / BIG_BLIND, 4),
    }


def load_groups(run_dirs: Sequence[Path]) -> List[dict]:
    groups: List[dict] = []
    keys = set()
    for run_dir in run_dirs:
        path = run_dir / "deal_groups.jsonl"
        if not path.is_file():
            path = run_dir / "deal_groups.jsonl.partial"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        # Shards of one seed (--start-group) are independent groups; only the
        # same seed AND group index is a repeat.
        run_keys = {(row["master_seed"], row.get("group_index")) for row in rows}
        if run_keys & keys:
            raise ValueError(
                f"{run_dir} repeats master seed groups {sorted(run_keys & keys)[:5]}"
            )
        keys |= run_keys
        groups.extend(rows)
    return groups


def pool(run_dirs: Sequence[Path]) -> Dict[str, object]:
    groups = load_groups(run_dirs)
    raw = [row["hero_net_chips_per_hand"] for row in groups]
    result: Dict[str, object] = {"runs": [str(d) for d in run_dirs], "raw": _estimate(raw)}
    adjusted = [row.get("hero_net_chips_per_hand_adjusted") for row in groups]
    if all(value is not None for value in adjusted):
        result["adjusted"] = _estimate(adjusted)
        result["corrections"] = _estimate([a - r for a, r in zip(adjusted, raw)])
    return result


def _group_key(row: dict) -> tuple:
    return int(row["master_seed"]), int(row["group_index"])


def paired(
    candidate_dirs: Sequence[Path], baseline_dirs: Sequence[Path]
) -> Dict[str, object]:
    """Per-group candidate minus baseline on identical decks and field RNGs.

    Groups pair by (master_seed, group_index); the recorded deal must match.
    Shared deal luck cancels, so the difference is far tighter than the gap
    between two independent estimates.
    """

    baseline = {_group_key(row): row for row in load_groups(baseline_dirs)}
    candidate = load_groups(candidate_dirs)
    pairs = []
    for row in candidate:
        key = _group_key(row)
        base = baseline.get(key)
        if base is None:
            raise ValueError(f"candidate group {key} has no baseline pair")
        if row.get("deal") != base.get("deal"):
            raise ValueError(f"group {key} was dealt differently in the two runs")
        pairs.append((row, base))
    result: Dict[str, object] = {
        "candidate_runs": [str(d) for d in candidate_dirs],
        "baseline_runs": [str(d) for d in baseline_dirs],
        "raw_difference": _estimate(
            [c["hero_net_chips_per_hand"] - b["hero_net_chips_per_hand"] for c, b in pairs]
        ),
    }
    adjusted = [
        (c.get("hero_net_chips_per_hand_adjusted"), b.get("hero_net_chips_per_hand_adjusted"))
        for c, b in pairs
    ]
    if all(c is not None and b is not None for c, b in adjusted):
        result["adjusted_difference"] = _estimate([c - b for c, b in adjusted])
    return result


def main(argv: Sequence[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument(
        "--baseline", nargs="+", type=Path, default=None,
        help="pair each run group with these runs' same-seed, same-index group",
    )
    args = ap.parse_args(argv)
    result = (
        paired(args.runs, args.baseline) if args.baseline else pool(args.runs)
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
