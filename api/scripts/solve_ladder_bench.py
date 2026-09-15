"""Solve captured production spots at an iteration ladder: cost and convergence.

usage: python scripts/solve_ladder_bench.py SPOT_DIR [SPOT_DIR ...] --out OUT_DIR
         [--blueprint PATH] [--rungs 10000,25000,50000,100000,250000,500000]
         [--reference 1000000] [--threads 8] [--limit N] [--streets flop,turn,river]
         [--players 2,3,4,5,6] [--seed-a 229] [--seed-b 4242]
         [--production-env NAME=VALUE ...] [--no-production-rung]

Spots come from ``run_match --capture-solve-requests`` (scripts/solve_request_capture.py):
the exact keyword inputs of production re-solves. The bench runs under the serve
environment (scripts/run_match_cli.PRODUCTION_SERVE_ENV) with NEXUS_BR_GAP_EXPOSE=1,
which stashes the solved tree and the final-readout strategy, and the live player cap
raised to 6 so 4+ player spots solve.

Per spot, in order: reference A (``--reference`` iterations, seed A, no deadline);
reference B (seed B; its distance to A is the noise floor); the production rung
(inputs exactly as captured: 100k ceiling, 3,000 ms deadline, served fill guard);
then every ladder rung (seed A, no deadline). Each row is appended to
OUT_DIR/ladder.jsonl as it finishes and a ``done`` marker closes the spot, so a
rerun skips finished spots. scripts/solve_ladder_report.py aggregates the rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.solve_ladder_metrics import mix_distance, sigma_distance  # noqa: E402
from scripts.solve_request_capture import load_requests  # noqa: E402

DEFAULT_RUNGS = (10_000, 25_000, 50_000, 100_000, 250_000, 500_000)
FILL_GUARD_OFF = 10**12
_ROW_STATS = (
    "iterations_run", "construct_s", "solve_s", "readout_s", "num_nodes",
    "num_leaf_nodes", "eff_threads", "deadline_ms", "leaf_mode",
)


def hero_mix(result: Any, hero_idx: Optional[int], hand_idx: Optional[int]) -> Optional[Dict[str, float]]:
    """The mix production would act on: ``result[hero][hand]`` as label -> freq."""
    if hero_idx is None or hand_idx is None or not isinstance(result, Mapping):
        return None
    by_hand = result.get(hero_idx)
    if not isinstance(by_hand, Mapping):
        return None
    mix = by_hand.get(hand_idx)
    if not isinstance(mix, Mapping):
        return None
    return {str(label): float(freq) for label, freq in mix.items()}


def _parse_ints(raw: str) -> Tuple[int, ...]:
    values = tuple(int(part) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"expected positive integers, got {raw!r}")
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    from scripts.run_match_cli import _env_override

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blueprint", default=None, help="8.05B .bps (default AGENT_BLUEPRINT_PATH)")
    ap.add_argument("--rungs", type=_parse_ints, default=DEFAULT_RUNGS)
    ap.add_argument("--reference", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=8, help="solve threads (production box: 8)")
    ap.add_argument("--limit", type=int, default=None, help="spots per (street, players) cell")
    ap.add_argument("--streets", default="flop,turn,river")
    ap.add_argument("--players", type=_parse_ints, default=(2, 3, 4, 5, 6))
    ap.add_argument("--seed-a", type=int, default=0xE5, help="production master seed")
    ap.add_argument("--seed-b", type=int, default=4242, help="noise-floor replicate seed")
    ap.add_argument("--production-env", type=_env_override, action="append", default=None)
    ap.add_argument("--no-production-rung", action="store_true", default=False)
    ap.add_argument(
        "--fill-guard-bytes", type=int, default=FILL_GUARD_OFF,
        help="projected-fill guard for ladder and reference rungs "
        "(the production rung always keeps the served guard)",
    )
    return ap.parse_args(argv)


def select_spots(
    records: Sequence[Mapping[str, Any]],
    streets: Sequence[str],
    players: Sequence[int],
    limit: Optional[int],
) -> List[Mapping[str, Any]]:
    """Filter by street and player count; cap each cell at ``limit`` spots."""
    counts: Dict[Tuple[str, int], int] = {}
    chosen: List[Mapping[str, Any]] = []
    for record in records:
        key = (str(record["street"]), int(record["n_players"]))
        if key[0] not in streets or key[1] not in players:
            continue
        if limit is not None and counts.get(key, 0) >= limit:
            continue
        counts = {**counts, key: counts.get(key, 0) + 1}
        chosen.append(record)
    return chosen


def finished_spots(path: Path) -> set:
    done = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("rung") == "done":
                done.add(row["spot"])
    return done


class LadderRunner:
    """Run one captured spot through the reference, production and ladder rungs."""

    def __init__(
        self,
        engine: Any,
        threads: int,
        seed_a: int,
        seed_b: int,
        fill_guard_bytes: int,
        evaluate: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._engine = engine
        self._threads = int(threads)
        self._seed_a = int(seed_a)
        self._seed_b = int(seed_b)
        self._fill_guard_bytes = int(fill_guard_bytes)
        self._evaluate = evaluate
        self._clock = clock

    def _solve(
        self, record: Mapping[str, Any], label: str, iterations: int,
        deadline_ms: Optional[int], seed: int, served_guard: bool,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Any]:
        kwargs = {
            **record["kwargs"],
            "iterations": int(iterations),
            "deadline_ms": deadline_ms,
            "num_threads": self._threads,
            "master_seed": int(seed),
            "readout": "final",
        }
        if not served_guard:
            kwargs["max_projected_fill_bytes"] = self._fill_guard_bytes
        before = getattr(self._engine, "_last_br_gap_objs", None)
        row: Dict[str, Any] = {
            "spot": record.get("path", str(record.get("index"))),
            "street": record["street"],
            "n_players": int(record["n_players"]),
            "rung": label,
            "iterations": int(iterations),
            "deadline_ms_requested": deadline_ms,
            "seed": int(seed),
            "threads": self._threads,
        }
        started = self._clock()
        try:
            result = self._engine.solve(**kwargs)
        except Exception as exc:  # a refused or failed solve is itself a result
            return (
                {**row, "outcome": f"error:{type(exc).__name__}", "error": str(exc)[:300],
                 "wall_s": self._clock() - started},
                None, None,
            )
        wall_s = self._clock() - started
        stats = getattr(self._engine, "_last_solve_stats", None) or {}
        objs = getattr(self._engine, "_last_br_gap_objs", None)
        row = {**row, "outcome": "solved", "wall_s": wall_s,
               **{key: stats.get(key) for key in _ROW_STATS}}
        if objs is None or objs is before or not objs.get("sigma"):
            return {**row, "outcome": "solved_no_stash"}, None, result
        from scripts.river_brgap_capture import plain_tree

        solved = {"tree": plain_tree(objs["tree"]), "sigma": objs["sigma"],
                  "mix": hero_mix(result, record.get("hero_player_idx"), record.get("hero_hand_idx"))}
        return row, solved, result

    def _river_gap(self, solved: Mapping[str, Any]) -> Optional[float]:
        tree = solved["tree"]
        if self._evaluate is None or int(tree["num_players"]) != 2 or len(tree["board"]) != 5:
            return None
        from scripts.river_brgap_score import exploitability, showdown_strengths

        strengths = showdown_strengths(tree, self._evaluate)
        total, _ = exploitability(tree, solved["sigma"], strengths)
        return 100.0 * total / float(tree["starting_pot"])

    def _compare(self, row: Dict[str, Any], solved: Optional[Mapping[str, Any]],
                 reference: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if solved is None:
            return row
        out = {**row, "hero_mix": solved["mix"], "river_brgap_pct_pot": self._river_gap(solved)}
        if reference is None:
            return out
        distance = sigma_distance(solved["sigma"], reference["sigma"], reference["tree"])
        mix_tv = (
            mix_distance(solved["mix"], reference["mix"])
            if solved["mix"] is not None and reference["mix"] is not None else None
        )
        return {**out, **distance, "hero_mix_tv": mix_tv}

    def run_spot(
        self, record: Mapping[str, Any], rungs: Sequence[int], reference: int,
        production_rung: bool, emit: Callable[[Dict[str, Any]], None],
    ) -> None:
        row, ref_a, _ = self._solve(record, "ref_a", reference, None, self._seed_a, False)
        emit(self._compare(row, ref_a, None))
        if ref_a is None:
            emit({"spot": row["spot"], "rung": "done", "reference_failed": True})
            return
        row, ref_b, _ = self._solve(record, "ref_b", reference, None, self._seed_b, False)
        emit(self._compare(row, ref_b, ref_a))
        if production_rung:
            captured = record["kwargs"]
            row, solved, _ = self._solve(
                record, "production", int(captured.get("iterations", 100_000)),
                captured.get("deadline_ms", 3_000), self._seed_a, True,
            )
            emit(self._compare(row, solved, ref_a))
        for iterations in rungs:
            if int(iterations) >= int(reference):
                continue
            row, solved, _ = self._solve(record, f"it_{int(iterations)}", int(iterations), None, self._seed_a, False)
            emit(self._compare(row, solved, ref_a))
        emit({"spot": row["spot"], "rung": "done"})


def _prepare_environment(pairs: Sequence[Tuple[str, str]]) -> None:
    from scripts.run_match_cli import apply_env_overrides, production_env_problems

    problems = production_env_problems(os.environ)
    if problems:
        raise SystemExit("serve environment needed: " + "; ".join(problems))
    apply_env_overrides(pairs, os.environ)
    os.environ["NEXUS_BR_GAP_EXPOSE"] = "1"
    os.environ["NEXUS_MAX_LIVE_RESOLVE_PLAYERS"] = "6"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _prepare_environment(args.production_env or ())
    blueprint = args.blueprint or os.environ.get("AGENT_BLUEPRINT_PATH")
    if not blueprint or not Path(blueprint).is_file():
        raise SystemExit("set --blueprint or AGENT_BLUEPRINT_PATH to the 8.05B .bps")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ladder_path = out_dir / "ladder.jsonl"
    streets = [part.strip() for part in args.streets.split(",") if part.strip()]
    spots = select_spots(load_requests(args.dirs), streets, args.players, args.limit)
    done = finished_spots(ladder_path)
    pending = [record for record in spots if record["path"] not in done]
    print(f"[ladder] {len(spots)} spots selected, {len(pending)} pending", flush=True)

    from app.engine.cpu_es_solver import CpuEsSolver
    from scripts.match_aivat import evaluate_7card_batch

    t0 = time.time()
    engine = CpuEsSolver(blueprint)
    print(f"[ladder] engine ready in {time.time() - t0:.0f} s", flush=True)
    runner = LadderRunner(
        engine, args.threads, args.seed_a, args.seed_b, args.fill_guard_bytes,
        evaluate=evaluate_7card_batch,
    )
    config = {
        "rungs": list(args.rungs), "reference": int(args.reference), "threads": int(args.threads),
        "seed_a": int(args.seed_a), "seed_b": int(args.seed_b),
        "production_env": dict(args.production_env or ()),
        "env": {name: os.environ.get(name) for name in (
            "NEXUS_LEAF_VEC_SAMPLED", "NEXUS_HU_LEAF_MIRROR", "NEXUS_OFFTREE_ADD",
            "NEXUS_MAX_PROJECTED_FILL_BYTES", "NEXUS_VEC_SAMPLED_OPP_HAND",
        )},
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with ladder_path.open("a", encoding="utf-8") as fh:
        def emit(row: Dict[str, Any]) -> None:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            if row.get("rung") != "done":
                print(
                    f"[ladder] {Path(row['spot']).name} {row.get('street')} {row.get('n_players')}p "
                    f"{row['rung']} iters={row.get('iterations_run')} solve={row.get('solve_s')} "
                    f"tv={row.get('mean_tv')} mix_tv={row.get('hero_mix_tv')} "
                    f"brgap={row.get('river_brgap_pct_pot')} {row.get('outcome')}",
                    flush=True,
                )

        for index, record in enumerate(pending, 1):
            print(f"[ladder] spot {index}/{len(pending)}: {record['path']}", flush=True)
            runner.run_spot(record, args.rungs, args.reference, not args.no_production_rung, emit)
    print("[ladder] finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
