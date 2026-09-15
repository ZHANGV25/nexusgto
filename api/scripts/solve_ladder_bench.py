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

Every rung is solved twice, with seed A and seed B: their distance is the rung's
self-consistency (the noise floor of a sampled-leaf, Hogwild solve at that
budget), and seed A's distance to the top rung's seed-A solve is its convergence
toward the largest budget. Per spot, in order: the top rung (``--reference``
iterations, no deadline), the production rung (100k ceiling, 3,000 ms deadline,
the served fill guard, for every player count), then each smaller ladder rung.
On heads-up river spots ``--river-avg-arm`` adds a scalar solve with both
readouts at every rung, scoring the exact best-response gap of the average and
the final iterate. The first solve of a process is a warm-up that is recorded as
``warmup`` and ignored by the report. Each row is appended to OUT_DIR/ladder.jsonl
as it finishes and a ``done`` marker closes the spot, so a rerun skips finished
spots. scripts/solve_ladder_report.py aggregates the rows.
"""
from __future__ import annotations

import argparse
import hashlib
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

DEFAULT_RUNGS = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
# Scalar external sampling updates one hand per iteration, so its rungs run
# about two orders of magnitude higher than the vector traverser's for the
# same wall time.
DEFAULT_AVG_RUNGS = (20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000)
FILL_GUARD_OFF = 10**12
PRODUCTION_ITERATIONS = 100_000
PRODUCTION_DEADLINE_MS = 3_000
SCALAR_UNSET_ENVS = ("NEXUS_LEAF_VEC_SAMPLED", "NEXUS_HU_LEAF_MIRROR")
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
    ap.add_argument("--reference", type=int, default=100_000, help="top rung iterations")
    ap.add_argument(
        "--river-avg-arm", action="store_true", default=False,
        help="heads-up river spots: add a scalar solve with both readouts at each --avg-rungs budget",
    )
    ap.add_argument("--avg-rungs", type=_parse_ints, default=DEFAULT_AVG_RUNGS)
    ap.add_argument("--threads", type=int, default=8, help="solve threads (production box: 8)")
    ap.add_argument("--limit", type=int, default=None, help="spots per (street, players) cell")
    ap.add_argument(
        "--all-arms", action="store_true", default=False,
        help="keep every seat arm of a hand (default: one record per distinct solve problem)",
    )
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


def solve_key(record: Mapping[str, Any]) -> str:
    """Identity of the hand behind a spot: the match replays each deal in six
    seat arms and, with every seat on the blueprint, the six arms are the same
    hand with the hero label moved (each arm records the hero at its own
    decision node of the same street-root solve). The board and player count
    identify the independent unit."""
    k = record["kwargs"]
    blob = json.dumps(
        [k.get("board_ints"), len(k.get("player_ranges") or ())], default=list,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def select_spots(
    records: Sequence[Mapping[str, Any]],
    streets: Sequence[str],
    players: Sequence[int],
    limit: Optional[int],
    distinct: bool = True,
) -> List[Mapping[str, Any]]:
    """Filter by street and player count; cap each cell at ``limit`` spots.

    With ``distinct`` (default) only the first record of each solve problem is
    kept, so a cell's spots are different hands rather than seat arms of one.
    """
    counts: Dict[Tuple[str, int], int] = {}
    seen: set = set()
    chosen: List[Mapping[str, Any]] = []
    for record in records:
        key = (str(record["street"]), int(record["n_players"]))
        if key[0] not in streets or key[1] not in players:
            continue
        identity = solve_key(record)
        if distinct and identity in seen:
            continue
        if limit is not None and counts.get(key, 0) >= limit:
            continue
        seen.add(identity)
        counts = {**counts, key: counts.get(key, 0) + 1}
        chosen.append({**record, "solve_key": identity})
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
        scalar_both: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Any]:
        kwargs = {
            **record["kwargs"],
            "iterations": int(iterations),
            "deadline_ms": deadline_ms,
            "num_threads": self._threads,
            "master_seed": int(seed),
            "readout": "both" if scalar_both else "final",
            "vec_mode": 0 if scalar_both else record["kwargs"].get("vec_mode", 1),
        }
        if not served_guard:
            kwargs["max_projected_fill_bytes"] = self._fill_guard_bytes
        before = getattr(self._engine, "_last_br_gap_objs", None)
        # The vec-sampled leaf oracle insists on the vector traverser. A river
        # tree has no leaves, so the scalar both-readout solve runs with the
        # leaf flags unset: the same exact showdown game, scalar external
        # sampling, average and final iterate read from one solve.
        leaf_flags = {name: os.environ.pop(name, None) for name in SCALAR_UNSET_ENVS} if scalar_both else {}
        try:
            return self._solve_row(record, label, iterations, deadline_ms, seed, kwargs, before, scalar_both)
        finally:
            for name, value in leaf_flags.items():
                if value is not None:
                    os.environ[name] = value

    def _solve_row(
        self, record: Mapping[str, Any], label: str, iterations: int,
        deadline_ms: Optional[int], seed: int, kwargs: Dict[str, Any], before: Any,
        scalar_both: bool,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Any]:
        row: Dict[str, Any] = {
            "spot": record.get("path", str(record.get("index"))),
            "solve_key": record.get("solve_key"),
            "street": record["street"],
            "n_players": int(record["n_players"]),
            "rung": label,
            "variant": "scalar_both" if scalar_both else "vec_final",
            "iterations": int(iterations),
            "deadline_ms_requested": deadline_ms,
            "seed": int(seed),
            "threads": self._threads,
            "captured_iterations": record["kwargs"].get("iterations"),
            "captured_deadline_ms": record["kwargs"].get("deadline_ms"),
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

        hero_idx, hand_idx = record.get("hero_player_idx"), record.get("hero_hand_idx")
        sigmas = objs.get("sigmas") or {}
        readout_result = result.get("final", result) if scalar_both and isinstance(result, Mapping) else result
        solved = {
            "tree": plain_tree(objs["tree"]),
            "sigma": sigmas.get("final", objs["sigma"]),
            "sigma_avg": sigmas.get("avg"),
            "mix": hero_mix(readout_result, hero_idx, hand_idx),
            "mix_avg": hero_mix(result.get("avg"), hero_idx, hand_idx)
            if scalar_both and isinstance(result, Mapping) else None,
        }
        return row, solved, result

    def _river_gap(self, tree: Mapping[str, Any], sigma: Optional[Mapping[int, Any]]) -> Optional[float]:
        if sigma is None or self._evaluate is None:
            return None
        if int(tree["num_players"]) != 2 or len(tree["board"]) != 5:
            return None
        from scripts.river_brgap_score import exploitability, showdown_strengths

        strengths = showdown_strengths(tree, self._evaluate)
        total, _ = exploitability(tree, sigma, strengths)
        return 100.0 * total / float(tree["starting_pot"])

    @staticmethod
    def _distance(a: Optional[Mapping[str, Any]], b: Optional[Mapping[str, Any]], prefix: str) -> Dict[str, Any]:
        if a is None or b is None:
            return {}
        mix_tv = (
            mix_distance(a["mix"], b["mix"])
            if a["mix"] is not None and b["mix"] is not None else None
        )
        try:
            distance = sigma_distance(a["sigma"], b["sigma"], b["tree"])
        except ValueError as exc:  # readout rows and snapshot hands can disagree
            return {f"{prefix}hero_mix_tv": mix_tv, f"{prefix}distance_error": str(exc)[:200]}
        return {
            f"{prefix}mean_tv": distance["mean_tv"],
            f"{prefix}root_tv": distance["root_tv"],
            f"{prefix}hero_mix_tv": mix_tv,
        }

    def _pair(
        self, record: Mapping[str, Any], label: str, iterations: int,
        deadline_ms: Optional[int], served_guard: bool,
        top: Optional[Mapping[str, Any]], emit: Callable[[Dict[str, Any]], None],
    ) -> Optional[Dict[str, Any]]:
        """Solve one rung with both seeds; emit one row; return seed A's solve."""
        row_a, solved_a, _ = self._solve(record, label, iterations, deadline_ms, self._seed_a, served_guard)
        row_b, solved_b, _ = self._solve(record, label, iterations, deadline_ms, self._seed_b, served_guard)
        row = {
            **row_a,
            "outcome_b": row_b.get("outcome"),
            "iterations_run_b": row_b.get("iterations_run"),
            "solve_s_b": row_b.get("solve_s"),
            "wall_s_b": row_b.get("wall_s"),
        }
        if solved_a is not None:
            row = {
                **row,
                "hero_mix": solved_a["mix"],
                "river_brgap_pct_pot": self._river_gap(solved_a["tree"], solved_a["sigma"]),
            }
        if solved_b is not None:
            row = {**row, "river_brgap_pct_pot_b": self._river_gap(solved_b["tree"], solved_b["sigma"])}
        row = {**row, **self._distance(solved_a, solved_b, "pair_"), **self._distance(solved_a, top, "")}
        emit(row)
        return solved_a

    def _river_avg(
        self, record: Mapping[str, Any], iterations: int,
        top: Optional[Mapping[str, Any]], emit: Callable[[Dict[str, Any]], None],
    ) -> None:
        row, solved, _ = self._solve(
            record, f"avg_{int(iterations)}", iterations, None, self._seed_a, False, scalar_both=True,
        )
        if solved is not None:
            tree = solved["tree"]
            row = {
                **row,
                "hero_mix": solved["mix"],
                "hero_mix_avg": solved["mix_avg"],
                "river_brgap_pct_pot": self._river_gap(tree, solved["sigma"]),
                "river_brgap_avg_pct_pot": self._river_gap(tree, solved["sigma_avg"]),
                **self._distance(solved, top, ""),
            }
        emit(row)

    def warmup(
        self, record: Mapping[str, Any], emit: Callable[[Dict[str, Any]], None],
        deadline_ms: int = 20_000, max_runs: int = 15, settle: float = 0.9,
    ) -> None:
        """Solve until the iteration rate stops rising.

        A fresh process pays first-touch page faults on the blueprint cache
        for its first minutes, so its early solves run far below the warm
        rate. Repeat a deadline-bound solve until two consecutive runs agree
        within ``1 - settle`` of each other.
        """
        previous: Optional[int] = None
        for _ in range(max_runs):
            row, _, _ = self._solve(record, "warmup", 10**7, deadline_ms, self._seed_a, True)
            emit(row)
            run = row.get("iterations_run")
            if run is None:
                return
            if previous is not None and abs(int(run) - previous) <= (1.0 - settle) * previous:
                return
            previous = int(run)

    def run_spot(
        self, record: Mapping[str, Any], rungs: Sequence[int], reference: int,
        production_rung: bool, emit: Callable[[Dict[str, Any]], None],
        river_avg_arm: bool = False, avg_rungs: Sequence[int] = DEFAULT_AVG_RUNGS,
    ) -> None:
        top = self._pair(record, f"it_{int(reference)}", int(reference), None, False, None, emit)
        spot = record.get("path", str(record.get("index")))
        if top is None:
            emit({"spot": spot, "rung": "done", "reference_failed": True})
            return
        if production_rung:
            self._pair(record, "production", PRODUCTION_ITERATIONS, PRODUCTION_DEADLINE_MS, True, top, emit)
        for iterations in rungs:
            if int(iterations) >= int(reference):
                continue
            self._pair(record, f"it_{int(iterations)}", int(iterations), None, False, top, emit)
        is_hu_river = int(record["n_players"]) == 2 and record["street"] == "river"
        if river_avg_arm and is_hu_river:
            for iterations in sorted({int(r) for r in avg_rungs}):
                self._river_avg(record, iterations, top, emit)
        emit({"spot": spot, "rung": "done"})


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
    spots = select_spots(
        load_requests(args.dirs), streets, args.players, args.limit, distinct=not args.all_arms,
    )
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
        "river_avg_arm": bool(args.river_avg_arm), "avg_rungs": list(args.avg_rungs),
        "production_rung": {"iterations": PRODUCTION_ITERATIONS, "deadline_ms": PRODUCTION_DEADLINE_MS},
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
                    f"pair_tv={row.get('pair_mean_tv')} tv={row.get('mean_tv')} "
                    f"mix_tv={row.get('hero_mix_tv')} brgap={row.get('river_brgap_pct_pot')} "
                    f"brgap_avg={row.get('river_brgap_avg_pct_pot')} {row.get('outcome')}",
                    flush=True,
                )

        if pending:
            runner.warmup(pending[0], emit)
        for index, record in enumerate(pending, 1):
            print(f"[ladder] spot {index}/{len(pending)}: {record['path']}", flush=True)
            try:
                runner.run_spot(
                    record, args.rungs, args.reference, not args.no_production_rung, emit,
                    river_avg_arm=bool(args.river_avg_arm), avg_rungs=args.avg_rungs,
                )
            except Exception as exc:  # one bad spot must not end the ladder
                print(f"[ladder] spot failed: {type(exc).__name__}: {exc}", flush=True)
                emit({"spot": record["path"], "rung": "done", "spot_error": f"{type(exc).__name__}: {exc}"[:300]})
    print("[ladder] finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
