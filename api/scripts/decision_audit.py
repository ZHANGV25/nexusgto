"""Phase A decision audit.

usage:
  python scripts/decision_audit.py spots --seed 1 --deals 200 --per-cell 3 --out DIR
  python scripts/decision_audit.py solve --spots DIR --arm NAME --production-env K=V \
      [--oracle-ranges] --live-threads 8 [--shard i/n] --out DIR
  python scripts/decision_audit.py score --spots DIR --tables ARM=DIR \
      [--tables ARM=DIR ...] --samples 2000 [--shard i/n] --out DIR
  python scripts/decision_audit.py report --spots DIR --scores DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decision_audit_core import (  # noqa: E402
    collect_sampled_spots,
    json_default,
    load_jsonl,
    oracle_range_overrides,
    parse_shard,
    replay_spot,
    score_tables_for_spot,
    shard_selected,
    summarize_report,
    render_markdown,
)


def _blueprint_arg(value: Optional[str]) -> str:
    path = value or os.environ.get("AGENT_BLUEPRINT_PATH")
    if not path:
        raise SystemExit("set --blueprint or AGENT_BLUEPRINT_PATH")
    return path


def _lookup(blueprint: str) -> Any:
    from app.engine.agent.modes import ModeConfig
    from app.engine.agent.strategy_source import LookupSource

    return LookupSource(
        ModeConfig(mode="LOOKUP", blueprint_path=blueprint, uniform_on_miss=True)
    )


def _shard_label(shard: Optional[Tuple[int, int]]) -> str:
    return "all" if shard is None else f"{shard[0]}of{shard[1]}"


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def _solve_stats(policy: Any) -> Dict[str, Any]:
    stats = getattr(getattr(policy, "_engine", None), "_last_solve_stats", None) or {}
    keys = (
        "iterations_run", "iterations_run_observed", "solve_s", "num_nodes",
        "flop_chance", "master_seed", "readout", "deadline_ms",
    )
    return {key: stats.get(key) for key in keys if key in stats}


def _range_override_stats(policy: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source in getattr(policy, "_sources", ()):
        live = getattr(source, "_live", None)
        stats = getattr(live, "last_range_overrides", None)
        if isinstance(stats, dict):
            rows.append(stats)
    return rows


def _set_range_overrides(policy: Any, overrides: Optional[Dict[str, np.ndarray]]) -> None:
    setter = getattr(policy, "set_range_overrides", None)
    if not callable(setter):
        raise RuntimeError("policy does not support range_overrides")
    setter(overrides)
    # Only the source that solves next rewrites its stats; clear the others so a
    # stale "applied" from an earlier spot cannot pass for this one.
    for source in getattr(policy, "_sources", ()):
        live = getattr(source, "_live", None)
        if live is not None:
            live.last_range_overrides = None


def _oracle_applied(stats_rows: Sequence[Mapping[str, Any]]) -> bool:
    """True when the solving source applied every configured seat's override."""
    return any(
        not row.get("error") and int(row.get("players_configured", 0)) > 0
        and int(row.get("players_applied", 0)) == int(row.get("players_configured", 0))
        for row in stats_rows
    )


MAX_CONSECUTIVE_ORACLE_FAILURES = 10


def _warm_up(policy: Any, spots: Sequence[Mapping[str, Any]], seconds: float) -> Dict[str, Any]:
    """Solve and discard spots until ``seconds`` pass.

    A fresh process pages in the blueprint on its first solves (SCOREBOARD_RESULTS
    section 30: 35 iterations in the first 3 s); depth-limited trees (3-player
    turns) query it on every leaf visit and would be scored under-converged.
    Leafed spots go first because they touch the blueprint.
    """
    if seconds <= 0 or not spots:
        return {"seconds": 0.0, "solves": 0}
    order = sorted(spots, key=lambda spot: 0 if int(spot["cell"]["entry_players"]) == 3
                   and int(spot["cell"]["street_index"]) == 2 else 1)
    started = time.perf_counter()
    solves = 0
    while time.perf_counter() - started < seconds:
        spot = order[solves % len(order)]
        state, _node, _start = replay_spot(spot)
        hero_cards = tuple(int(c) for c in spot["deal"]["hole_cards"][int(spot["hero_seat"])])
        _set_range_overrides(policy, None)
        policy.table(state, hero_cards)
        stats = _solve_stats(policy)
        print(f"[warmup] {solves + 1} {spot['cell']['key']} iterations={stats.get('iterations_run')}", flush=True)
        solves += 1
    return {"seconds": time.perf_counter() - started, "solves": solves}


def cmd_spots(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = _lookup(_blueprint_arg(args.blueprint))
    spots, meta = collect_sampled_spots(
        deals=int(args.deals),
        per_cell=int(args.per_cell),
        seed=int(args.seed),
        lookup=lookup,
    )
    _write_json(out_dir / "meta.json", meta)
    with (out_dir / "spots.jsonl").open("w", encoding="utf-8") as fh:
        for spot in spots:
            fh.write(json.dumps(spot, default=json_default) + "\n")
    print(json.dumps(meta, indent=2, default=json_default))
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    from scripts.lbr_policies import build_policy
    from scripts.lbr_run import _env_pairs
    from scripts.run_match_cli import apply_env_overrides

    blueprint = _blueprint_arg(args.blueprint)
    spots_dir = Path(args.spots)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = parse_shard(args.shard)
    spots = load_jsonl(spots_dir / "spots.jsonl")
    if args.limit is not None:
        spots = spots[: int(args.limit)]
    env_pairs = _env_pairs(args.production_env)
    apply_env_overrides(env_pairs, os.environ)
    policy = build_policy(
        "production",
        blueprint,
        np.random.default_rng(int(args.seed)),
        live_threads=int(args.live_threads),
        env_overrides=env_pairs,
    )
    lookup = getattr(policy, "_lookup")
    shard_spots = [spot for index, spot in enumerate(spots) if shard_selected(index, shard)]
    warmup = _warm_up(policy, shard_spots, float(args.warmup_seconds))
    table_arrays: Dict[str, np.ndarray] = {}
    metadata: List[Dict[str, Any]] = []
    log_path = out_dir / f"solve_log_{args.arm}_{_shard_label(shard)}.jsonl"
    started = time.perf_counter()
    oracle_failures = 0
    with log_path.open("w", encoding="utf-8") as log:
        for index, spot in enumerate(spots):
            if not shard_selected(index, shard):
                continue
            state, _node_decisions, start_decisions = replay_spot(spot)
            hero_seat = int(spot["hero_seat"])
            hero_cards = tuple(int(c) for c in spot["deal"]["hole_cards"][hero_seat])
            if args.oracle_ranges:
                _set_range_overrides(
                    policy, oracle_range_overrides(state, start_decisions, lookup)
                )
            else:
                _set_range_overrides(policy, None)
            t0 = time.perf_counter()
            table = np.asarray(policy.table(state, hero_cards), dtype=np.float32)
            key = f"table_{len(metadata):06d}"
            source_reason = getattr(policy, "last_table_source_reason", None)
            override_stats = _range_override_stats(policy)
            oracle_ok = _oracle_applied(override_stats) if args.oracle_ranges else None
            if args.oracle_ranges and source_reason == "stash" and not oracle_ok:
                oracle_failures += 1
                if oracle_failures >= MAX_CONSECUTIVE_ORACLE_FAILURES:
                    raise SystemExit(
                        f"oracle ranges not applied on {oracle_failures} consecutive solves; "
                        f"last stats {override_stats}"
                    )
            else:
                oracle_failures = 0
            row = {
                "schema": "decision_audit_solve_log_v1",
                "spot_id": spot["spot_id"],
                "array_key": key,
                "arm": args.arm,
                "table_source": source_reason,
                "fallback": source_reason != "stash",
                "solve_s": time.perf_counter() - t0,
                "solve_stats": _solve_stats(policy),
                "range_overrides": override_stats,
                "oracle_applied": oracle_ok,
                "cell": spot["cell"],
            }
            table_arrays[key] = table
            metadata.append(row)
            log.write(json.dumps(row, default=json_default) + "\n")
            log.flush()
            print(
                f"[solve] {args.arm} {len(metadata)} spot={spot['spot_id']} "
                f"source={source_reason} {row['solve_s']:.1f}s",
                flush=True,
            )
    np.savez_compressed(
        out_dir / f"tables_{args.arm}_{_shard_label(shard)}.npz",
        metadata_json=np.array(json.dumps(metadata, default=json_default)),
        **table_arrays,
    )
    _write_json(out_dir / f"solve_summary_{args.arm}_{_shard_label(shard)}.json", {
        "arm": args.arm,
        "shard": _shard_label(shard),
        "spots": len(metadata),
        "wall_s": time.perf_counter() - started,
        "warmup": warmup,
        "policy_stats": getattr(policy, "stats", {}),
    })
    return 0


def _parse_table_arg(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--tables expects ARM=DIR, got {raw!r}")
    arm, directory = raw.split("=", 1)
    arm = arm.strip()
    if not arm:
        raise argparse.ArgumentTypeError(f"empty arm in {raw!r}")
    return arm, Path(directory)


def _load_arm_tables(arm: str, directory: Path) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    out: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
    for path in sorted(directory.glob(f"tables_{arm}_*.npz")):
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"].item()))
        for row in metadata:
            key = str(row["array_key"])
            out[str(row["spot_id"])] = (
                np.asarray(data[key], dtype=np.float32),
                dict(row),
            )
    return out


def cmd_score(args: argparse.Namespace) -> int:
    from scripts.match_aivat import evaluate_7card_batch

    spots_dir = Path(args.spots)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = parse_shard(args.shard)
    spots = load_jsonl(spots_dir / "spots.jsonl")
    if args.limit is not None:
        spots = spots[: int(args.limit)]
    arms = [_parse_table_arg(raw) for raw in args.tables]
    tables = {arm: _load_arm_tables(arm, directory) for arm, directory in arms}
    lookup = _lookup(_blueprint_arg(args.blueprint))
    rng = np.random.default_rng(int(args.seed) + (0 if shard is None else 1_000 * (shard[0] + 1)))
    out_path = out_dir / f"scores_{_shard_label(shard)}.jsonl"
    rows: List[Dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for index, spot in enumerate(spots):
            if not shard_selected(index, shard):
                continue
            per_arm = {}
            table_meta: Dict[str, Any] = {}
            for arm, _directory in arms:
                if str(spot["spot_id"]) not in tables[arm]:
                    raise RuntimeError(f"missing {arm} table for {spot['spot_id']}")
                per_arm[arm], table_meta[arm] = tables[arm][str(spot["spot_id"])]
            t0 = time.perf_counter()
            row = score_tables_for_spot(
                spot, per_arm, lookup, evaluate_7card_batch, rng, int(args.samples)
            )
            row["table_sources"] = {
                arm: {
                    "table_source": meta.get("table_source"),
                    "fallback": meta.get("fallback"),
                    "oracle_applied": meta.get("oracle_applied"),
                    "solve_stats": meta.get("solve_stats"),
                }
                for arm, meta in table_meta.items()
            }
            rows.append(row)
            fh.write(json.dumps(row, default=json_default) + "\n")
            fh.flush()
            print(
                f"[score] {len(rows)} spot={spot['spot_id']} "
                f"{time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    _write_json(out_dir / f"score_summary_{_shard_label(shard)}.json", {
        "spots": len(rows),
        "samples": int(args.samples),
        "arms": [arm for arm, _directory in arms],
    })
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    spots_dir = Path(args.spots)
    scores_dir = Path(args.scores)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((spots_dir / "meta.json").read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for path in sorted(scores_dir.glob("scores_*.jsonl")):
        rows.extend(load_jsonl(path))
    report = summarize_report(rows, deals=int(meta["deals"]))
    _write_json(out_dir / "report.json", report)
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "spots": report["spots"],
        "arms": sorted(report["arms"]),
        "out": str(out_dir),
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    spots = sub.add_parser("spots")
    spots.add_argument("--seed", type=int, required=True)
    spots.add_argument("--deals", type=int, required=True)
    spots.add_argument("--per-cell", type=int, required=True)
    spots.add_argument("--out", required=True)
    spots.add_argument("--blueprint", default=None)
    spots.set_defaults(func=cmd_spots)

    solve = sub.add_parser("solve")
    solve.add_argument("--spots", required=True)
    solve.add_argument("--arm", required=True)
    solve.add_argument("--production-env", action="append", default=[])
    solve.add_argument("--oracle-ranges", action="store_true")
    solve.add_argument("--live-threads", type=int, default=8)
    solve.add_argument("--shard", default=None)
    solve.add_argument("--out", required=True)
    solve.add_argument("--blueprint", default=None)
    solve.add_argument("--seed", type=int, default=17)
    solve.add_argument("--limit", type=int, default=None)
    solve.add_argument("--warmup-seconds", type=float, default=0.0,
                       help="solve and discard spots this long before the timed pass")
    solve.set_defaults(func=cmd_solve)

    score = sub.add_parser("score")
    score.add_argument("--spots", required=True)
    score.add_argument("--tables", action="append", required=True)
    score.add_argument("--samples", type=int, default=2000)
    score.add_argument("--shard", default=None)
    score.add_argument("--out", required=True)
    score.add_argument("--blueprint", default=None)
    score.add_argument("--seed", type=int, default=23)
    score.add_argument("--limit", type=int, default=None)
    score.set_defaults(func=cmd_score)

    report = sub.add_parser("report")
    report.add_argument("--spots", required=True)
    report.add_argument("--scores", required=True)
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
