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
    seeds = set()
    for run_dir in run_dirs:
        path = run_dir / "deal_groups.jsonl"
        if not path.is_file():
            path = run_dir / "deal_groups.jsonl.partial"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        run_seeds = {row["master_seed"] for row in rows}
        if run_seeds & seeds:
            raise ValueError(f"{run_dir} repeats master seeds {sorted(run_seeds & seeds)}")
        seeds |= run_seeds
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


def main(argv: Sequence[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    print(json.dumps(pool([Path(arg) for arg in argv]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
