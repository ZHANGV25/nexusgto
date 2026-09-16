"""Run local best response against the blueprint or the production search.

usage (Git Bash, serve environment sourced for --policy production):
  python scripts/lbr_run.py --policy blueprint --hands 2000 --seed 1 --out DIR
  python scripts/lbr_run.py --policy production --hands 150 --seed 1 --out DIR --live-threads 8

Each deal is played twice with the seats swapped (duplicate); the deal is
the independent unit for the standard error.  Two estimators are reported:
the realised chips, and the expectation over the policy's tracked range at
showdown (same mean, lower variance).  Results are bb/100 for the exploiter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lbr_core import mean_se  # noqa: E402
from scripts.lbr_game import ExploiterConfig, deal_from_rng, play_hand  # noqa: E402

SEAT_INDEX = {"SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5}
SEAT_NAME = {index: name for name, index in SEAT_INDEX.items()}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=("blueprint", "production"), required=True)
    ap.add_argument("--hands", type=int, required=True, help="deals; each is played in two seat arms")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seats", default="SB,BB", help="the two participating seats, e.g. SB,BB or BTN,BB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blueprint", default=None)
    ap.add_argument("--live-threads", type=int, default=8)
    ap.add_argument("--production-env", action="append", default=[], help="NAME=VALUE override for the search")
    ap.add_argument("--hero-range", choices=("blueprint", "own"), default="blueprint",
                    help="search hero range: blueprint stand-in (production) or the tracked own reach")
    ap.add_argument("--runouts-preflop", type=int, default=200)
    ap.add_argument("--runouts-flop", type=int, default=100)
    ap.add_argument("--terminal-samples", type=int, default=400)
    ap.add_argument("--no-duplicate", action="store_true", default=False)
    ap.add_argument("--max-wall-s", type=float, default=None)
    ap.add_argument("--progress", type=int, default=10)
    return ap.parse_args(argv)


def _seats(raw: str) -> List[int]:
    names = [part.strip().upper() for part in raw.split(",")]
    if len(names) != 2 or any(name not in SEAT_INDEX for name in names) or names[0] == names[1]:
        raise SystemExit(f"--seats needs two distinct seat names, got {raw!r}")
    return [SEAT_INDEX[name] for name in names]


def _env_pairs(raw: Sequence[str]) -> List[tuple]:
    pairs = []
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"--production-env expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        pairs.append((name.strip(), value))
    return pairs


def summarize(rows: Sequence[Dict[str, Any]], policy_stats: Dict[str, Any], wall_s: float) -> Dict[str, Any]:
    by_deal: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_deal.setdefault(int(row["deal"]), []).append(row)
    per_deal_expected = [float(np.mean([r["l_net_expected"] for r in arms])) for arms in by_deal.values()]
    per_deal_realized = [float(np.mean([r["l_net_realized"] for r in arms])) for arms in by_deal.values()]
    by_seat = {}
    for seat in sorted({int(r["p_seat"]) for r in rows}):
        seat_rows = [r for r in rows if int(r["p_seat"]) == seat]
        by_seat[SEAT_NAME[seat]] = {
            "expected_bb100": mean_se([r["l_net_expected"] for r in seat_rows]),
            "realized_bb100": mean_se([r["l_net_realized"] for r in seat_rows]),
        }
    exploiter = Counter()
    for row in rows:
        for decision in row["decisions"]:
            if decision.get("actor") == "exploiter":
                exploiter[f"street{decision['street']}_{decision['chosen']}"] += 1
    terminals = Counter(row["terminal"] for row in rows)
    return {
        "deals": len(by_deal), "arms": len(rows),
        "exploiter_expected_bb100": mean_se(per_deal_expected),
        "exploiter_realized_bb100": mean_se(per_deal_realized),
        "by_policy_seat": by_seat,
        "terminals": dict(terminals), "exploiter_actions": dict(sorted(exploiter.items())),
        "mean_hand_wall_s": float(np.mean([r["wall_s"] for r in rows])) if rows else None,
        "policy_stats": policy_stats, "wall_s": wall_s,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    blueprint = args.blueprint or os.environ.get("AGENT_BLUEPRINT_PATH")
    if not blueprint:
        raise SystemExit("set --blueprint or AGENT_BLUEPRINT_PATH")
    seats = _seats(args.seats)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from scripts.lbr_policies import build_policy
    from scripts.match_aivat import evaluate_7card_batch

    rng_deal = np.random.default_rng(args.seed)
    policy = build_policy(
        args.policy, blueprint, np.random.default_rng(args.seed + 1),
        live_threads=args.live_threads, env_overrides=_env_pairs(args.production_env),
        hero_range=args.hero_range,
    )
    rng_lbr = np.random.default_rng(args.seed + 2)
    config = ExploiterConfig(
        runouts_preflop=args.runouts_preflop, runouts_flop=args.runouts_flop, terminal_samples=args.terminal_samples,
    )
    arms = [(seats[0], seats[1])] if args.no_duplicate else [(seats[0], seats[1]), (seats[1], seats[0])]
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "blueprint": blueprint}, indent=2), encoding="utf-8")
    rows: List[Dict[str, Any]] = []
    errors: Counter = Counter()
    started = time.perf_counter()
    with (out_dir / "hands.jsonl").open("w", encoding="utf-8") as fh:
        for deal_index in range(int(args.hands)):
            deal = deal_from_rng(rng_deal)
            for p_seat, l_seat in arms:
                try:
                    row = play_hand(deal, p_seat, l_seat, policy, evaluate_7card_batch, rng_lbr, config)
                except Exception as exc:  # keep the run alive; report the failure classes
                    errors[f"{type(exc).__name__}: {str(exc)[:80]}"] += 1
                    print(f"[lbr] deal {deal_index} P={SEAT_NAME[p_seat]} failed: {type(exc).__name__}: {exc}", flush=True)
                    continue
                row = {"deal": deal_index, **row}
                rows.append(row)
                fh.write(json.dumps(row, default=float) + "\n")
                fh.flush()
            elapsed = time.perf_counter() - started
            if (deal_index + 1) % int(args.progress) == 0 or deal_index + 1 == int(args.hands):
                summary = summarize(rows, dict(policy.stats), elapsed)
                summary["errors"] = dict(errors)
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
                exp = summary["exploiter_expected_bb100"]
                print(
                    f"[lbr] {deal_index + 1}/{args.hands} deals {elapsed / 60:.1f} min  "
                    f"exploiter {exp['mean']:+.1f} +/- {exp['se'] or 0:.1f} bb/100 (expected)  "
                    f"realized {summary['exploiter_realized_bb100']['mean']:+.1f}  "
                    f"solves={policy.stats.get('solves', 0)}",
                    flush=True,
                )
            if args.max_wall_s is not None and elapsed > args.max_wall_s:
                print("[lbr] wall budget reached; stopping", flush=True)
                break
    summary = summarize(rows, dict(policy.stats), time.perf_counter() - started)
    summary["errors"] = dict(errors)
    if hasattr(policy, "fallbacks"):
        summary["policy_fallbacks"] = policy.fallbacks()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "exploiter_actions"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
