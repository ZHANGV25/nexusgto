"""CLI, source construction, and output helpers for :mod:`scripts.run_match`.

The pure match machinery stays in ``run_match.py`` so its no-blueprint tests
remain cheap.  Imports that construct a blueprint reader or CPU solver are
deliberately local to :func:`build_factories`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


SOURCE_CHOICES = ("lookup", "live_eager", "live_sampled")

# These flags select mutually exclusive leaf-value implementations or arm a
# construct-time continuation table.  Every live solve gets a clean snapshot
# and exact restore so alternating eager/sampled roles cannot leak state.
_CONTROLLED_LEAF_ENVS = (
    "NEXUS_LEAF_SAMPLED",
    "NEXUS_LEAF_SAMPLED_POLICY",
    "NEXUS_SAMPLED_DEADLINE_MS",
    "NEXUS_LEAF_ROLLOUT",
    "NEXUS_LEAF_CACHE_K",
    "NEXUS_LEAF_RANGE_CONDITIONING",
    "NEXUS_LEAF_BIAS_CFR",
    "NEXUS_BIAS_CFR_ROUNDS",
    "NEXUS_BIAS_CFR_NR",
    "NEXUS_A3_CONT_EV",
    "NEXUS_A3_CONT_EV_TURN",
    "NEXUS_A3_CEV_PATH",
    "NEXUS_CONT_EV_BIAS_ACTION",
    "NEXUS_CONT_EV_MISS_FALLBACK",
    "CONT_EV_CATALOG_MANIFEST",
)


def percentile(values: Sequence[float], q: float) -> float:
    """NumPy-compatible linear percentile for small wall-time vectors."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def solve_wall_summaries(
    samples: Mapping[str, Sequence[float]],
    sources: Mapping[str, str],
) -> Dict[str, Dict[str, object]]:
    """Count/mean/p50/p95/max solve walls for each configured role."""

    summaries: Dict[str, Dict[str, object]] = {}
    for role in sorted(set(sources) | set(samples)):
        values = tuple(float(value) for value in samples.get(role, ()))
        summaries[role] = {
            "source": sources.get(role),
            "count": len(values),
            "mean": (sum(values) / len(values)) if values else None,
            "p50": percentile(values, 0.50) if values else None,
            "p95": percentile(values, 0.95) if values else None,
            "max": max(values) if values else None,
        }
    return summaries


def resolved_leaf_mode_env(source: str) -> Dict[str, Optional[str]]:
    """Return the leaf-mode environment enforced for one source."""

    if source == "lookup":
        return {}
    if source not in ("live_eager", "live_sampled"):
        raise ValueError(f"unknown strategy source {source!r}")
    resolved = {name: None for name in _CONTROLLED_LEAF_ENVS}
    if source == "live_sampled":
        resolved.update(
            {
                "NEXUS_LEAF_SAMPLED": "1",
                "NEXUS_LEAF_SAMPLED_POLICY": "1",
                "NEXUS_LEAF_RANGE_CONDITIONING": "0",
            }
        )
    return resolved


def effective_leaf_mode_env(source: str) -> Dict[str, str]:
    """Human-readable values after solver defaults resolve unset flags."""

    if source == "lookup":
        return {}
    if source == "live_sampled":
        return {
            "NEXUS_LEAF_SAMPLED": "1",
            "NEXUS_LEAF_SAMPLED_POLICY": "1",
            "NEXUS_LEAF_RANGE_CONDITIONING": "0",
        }
    if source == "live_eager":
        return {
            "NEXUS_LEAF_SAMPLED": "0 (unset)",
            "NEXUS_LEAF_SAMPLED_POLICY": "1 (inert default)",
            "NEXUS_LEAF_RANGE_CONDITIONING": "1 (implicit default)",
        }
    raise ValueError(f"unknown strategy source {source!r}")


@contextmanager
def scoped_leaf_mode(source: str) -> Iterator[None]:
    """Apply one live leaf mode and restore exact prior process state."""

    resolved = resolved_leaf_mode_env(source)
    if source == "lookup":
        yield
        return
    previous = {name: os.environ.get(name) for name in _CONTROLLED_LEAF_ENVS}
    for name in _CONTROLLED_LEAF_ENVS:
        os.environ.pop(name, None)
    for name, value in resolved.items():
        if value is not None:
            os.environ[name] = value
    try:
        yield
    finally:
        for name in _CONTROLLED_LEAF_ENVS:
            value = previous[name]
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class SolveWallTracker:
    """Collect actual ``CpuEsSolver.solve`` walls by match role."""

    def __init__(self, sources: Mapping[str, str]) -> None:
        self._sources = dict(sources)
        self._samples: Dict[str, List[float]] = {
            role: [] for role in self._sources
        }
        self._group_offsets: Optional[Dict[str, int]] = None

    @property
    def sources(self) -> Mapping[str, str]:
        return self._sources

    def record(self, role: str, wall_s: float) -> None:
        if role not in self._samples:
            raise ValueError(f"unknown solve-wall role {role!r}")
        self._samples[role].append(float(wall_s))

    def begin_group(self) -> None:
        self._group_offsets = {
            role: len(values) for role, values in self._samples.items()
        }

    def finish_group(self) -> Dict[str, Tuple[float, ...]]:
        if self._group_offsets is None:
            raise RuntimeError("finish_group called before begin_group")
        result = {
            role: tuple(values[self._group_offsets[role]:])
            for role, values in self._samples.items()
        }
        self._group_offsets = None
        return result


class _ScopedTimedSolver:
    """Role-tagged proxy around one held ``CpuEsSolver``."""

    def __init__(
        self,
        inner: Any,
        *,
        role: str,
        source: str,
        tracker: SolveWallTracker,
        clock: Any = time.perf_counter,
    ) -> None:
        self._inner = inner
        self._role = role
        self._source = source
        self._tracker = tracker
        self._clock = clock

    def solve(self, *args: object, **kwargs: object) -> object:
        with scoped_leaf_mode(self._source):
            started = self._clock()
            try:
                return self._inner.solve(*args, **kwargs)
            finally:
                self._tracker.record(
                    self._role, float(self._clock() - started)
                )

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


class _LiveByPlayerCountSource:
    """Lookup preflop; select a play-config live source by street-entry N."""

    def __init__(
        self, lookup: object, live_by_count: Mapping[int, object]
    ) -> None:
        self._lookup = lookup
        self._live_by_count = dict(live_by_count)

    def get_mix(self, state: object, hero_cards: object) -> object:
        if int(getattr(state, "street", -1)) == 0:
            return self._lookup.get_mix(state, hero_cards)
        active = tuple(getattr(state, "active", ()))
        acted = tuple(getattr(state, "acted", ()))
        if len(active) != len(acted):
            raise ValueError(
                "postflop state active/acted vectors have different lengths"
            )
        n_players = sum(
            bool(is_active) or bool(has_acted)
            for is_active, has_acted in zip(active, acted)
        )
        source = self._live_by_count.get(n_players)
        if source is None:
            raise ValueError(
                f"{n_players}"
            )
        return source.get_mix(state, hero_cards)


def _positive_finite(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite number, got {raw!r}"
        ) from None
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite number, got {raw!r}"
        )
    return value


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {raw!r}"
        ) from None
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {raw!r}"
        )
    return value


def parse_args(
    runner: Any, argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=runner.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=runner.VALID_MODES, required=True)
    ap.add_argument(
        "--deals",
        type=_positive_int,
        default=1000,
        help="independent deal groups (six full duplicate hands per group)",
    )
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--blueprint-a", default=None)
    ap.add_argument("--blueprint-b", default=None)
    ap.add_argument("--seats", type=int, choices=(2, 6), default=6)
    ap.add_argument(
        "--hero-source", choices=SOURCE_CHOICES, default="lookup"
    )
    ap.add_argument(
        "--field-source", choices=SOURCE_CHOICES, default="lookup"
    )
    ap.add_argument(
        "--live-iterations", type=_positive_int, default=250_000
    )
    ap.add_argument(
        "--live-num-rollouts", type=_positive_int, default=None
    )
    ap.add_argument(
        "--max-wall-s",
        type=_positive_finite,
        default=None,
        help="stop cleanly after the first completed group beyond this wall",
    )
    ap.add_argument(
        "--allin-equity-adjust",
        action="store_true",
        default=False,
        help=(
            "score postflop all-ins by exact mean settlement over remaining "
            "board completions"
        ),
    )
    args = ap.parse_args(argv)
    if args.seats == 2:
        ap.error(
            "--seats 2 is unavailable: the full-hand preflop transition, "
            "button order, and policy vectors are structurally six-seat; "
            "the repo's existing heads-up paths are six-max-with-folds or "
            "postflop-only, so this driver refuses to invent HU rules"
        )
    if (
        args.mode in ("aa", "epsilon")
        and args.hero_source != args.field_source
    ):
        ap.error(
            f"mode {args.mode!r} requires identical hero/field sources; "
            "use mode 'ab' for different source architectures"
        )
    if args.mode == "aa" and args.hero_source != "lookup":
        ap.error(
            "mode 'aa' is the exact deterministic lookup plumbing null; "
            "all-core live solves are not scheduling-deterministic, so use "
            "mode 'ab' for live architecture matches"
        )
    return args


def _absolute_blueprint_paths(
    args: argparse.Namespace,
) -> Tuple[str, str, Dict[str, Optional[str]]]:
    a_path = args.blueprint_a or os.environ.get("AGENT_BLUEPRINT_PATH")
    if not a_path or not Path(a_path).is_file():
        raise ValueError(
            "set --blueprint-a or AGENT_BLUEPRINT_PATH to the 8.05B .bps"
        )
    if args.mode == "ab":
        b_path = args.blueprint_b or os.environ.get(
            "AGENT_BLUEPRINT_B_PATH"
        )
        if not b_path:
            raise ValueError(
                "mode 'ab' requires --blueprint-b or "
                "AGENT_BLUEPRINT_B_PATH; refusing to silently fall back to "
                "blueprint A (that would run A-vs-A labeled as A-vs-B)"
            )
        if not Path(b_path).is_file():
            raise ValueError(
                "--blueprint-b / AGENT_BLUEPRINT_B_PATH is not a file"
            )
    else:
        b_path = a_path
    a_abs = os.path.abspath(str(a_path))
    b_abs = os.path.abspath(str(b_path))
    return a_abs, b_abs, {
        "blueprint_a": a_abs,
        "blueprint_b": b_abs if args.mode == "ab" else None,
    }


def build_factories(
    runner: Any, args: argparse.Namespace,
) -> Tuple[Any, Any, Dict[str, object]]:
    """Build lookup/live role factories after fail-fast path validation."""

    if int(args.seats) != 6:
        raise ValueError(
            "full-hand match factories currently support exactly six seats"
        )
    if args.hero_source not in SOURCE_CHOICES:
        raise ValueError(f"unknown hero source {args.hero_source!r}")
    if args.field_source not in SOURCE_CHOICES:
        raise ValueError(f"unknown field source {args.field_source!r}")
    if (
        args.mode in ("aa", "epsilon")
        and args.hero_source != args.field_source
    ):
        raise ValueError(
            f"mode {args.mode!r} requires identical hero/field sources"
        )
    if args.mode == "aa" and args.hero_source != "lookup":
        raise ValueError(
            "mode 'aa' supports lookup only; use mode 'ab' for live "
            "architecture matches"
        )
    a_path, b_path, metadata = _absolute_blueprint_paths(args)

    from app.engine.agent.agent import PokerAgent
    from app.engine.agent.modes import ModeConfig
    from app.engine.agent.strategy_source import (
        HybridSource,
        LiveResolveSource,
        LookupSource,
    )

    lookup_by_path: Dict[str, object] = {}

    def lookup_source(path: str) -> object:
        source = lookup_by_path.get(path)
        if source is None:
            source = LookupSource(
                ModeConfig(
                    mode="LOOKUP",
                    blueprint_path=path,
                    uniform_on_miss=True,
                )
            )
            lookup_by_path[path] = source
        return source

    role_paths = {"hero": a_path, "field": b_path}
    role_sources = {
        "hero": args.hero_source,
        "field": args.field_source,
    }
    has_live_source = any(
        source != "lookup" for source in role_sources.values()
    )
    tracker = SolveWallTracker(role_sources) if has_live_source else None
    live_config_for_fn = None
    if has_live_source:
        from app.engine.engine_constants import live_config_for

        live_config_for_fn = live_config_for

    engine_by_path: Dict[str, object] = {}

    def cpu_engine(path: str) -> object:
        engine = engine_by_path.get(path)
        if engine is None:
            from app.engine.cpu_es_solver import CpuEsSolver

            # Construction-time continuation tables are disabled here too;
            # solve-time role scopes below repeat the guard per decision.
            with scoped_leaf_mode("live_eager"):
                engine = CpuEsSolver(path)
            engine_by_path[path] = engine
        return engine

    strategy_by_role: Dict[str, object] = {}
    for role in ("hero", "field"):
        path = role_paths[role]
        source_name = role_sources[role]
        lookup = lookup_source(path)
        if source_name == "lookup":
            strategy_by_role[role] = lookup
            continue
        if live_config_for_fn is None:
            raise RuntimeError("live source selected without a live config")
        if tracker is None:
            raise RuntimeError("live source selected without wall tracking")
        timed_engine = _ScopedTimedSolver(
            cpu_engine(path),
            role=role,
            source=source_name,
            tracker=tracker,
        )
        live_by_count: Dict[int, object] = {}
        for n_players in range(2, 7):
            config_kwargs = {
                "iterations_override": int(args.live_iterations),
                "purpose": "play",
            }
            if source_name != "live_sampled":
                config_kwargs["num_rollouts_override"] = (
                    args.live_num_rollouts
                )
            live_config = live_config_for_fn(n_players, **config_kwargs)
            if live_config.readout != "final":
                raise RuntimeError(
                    "live_config_for(purpose='play') must return final "
                    f"readout at {n_players} players"
                )
            # The wall verdict measures per-visit sampled leaves down to nr=1;
            # eager config bounds do not apply (docs/PLURIBUS_WALL_VERDICT_2026-07-28.md).
            num_rollouts = (
                args.live_num_rollouts
                if (
                    source_name == "live_sampled"
                    and args.live_num_rollouts is not None
                )
                else live_config.num_rollouts
            )
            live = LiveResolveSource(
                ModeConfig(
                    mode="LIVE_RESOLVE",
                    blueprint_path=path,
                    uniform_on_miss=True,
                ),
                cpu_es_solver=timed_engine,
                iterations=int(live_config.iterations),
                seed_ranges=True,
                vec_mode=bool(live_config.vec_mode),
                readout=str(live_config.readout),
                num_threads=0,
                num_rollouts=int(num_rollouts),
                cache_k=1,
                belief_carry=False,
                offtree_add=False,
            )
            live_by_count[n_players] = HybridSource(live, lookup)
        strategy_by_role[role] = _LiveByPlayerCountSource(
            lookup, live_by_count
        )

    def make_factory(role: str) -> Any:
        path = role_paths[role]
        source_name = role_sources[role]
        strategy = strategy_by_role[role]
        mode = "LOOKUP" if source_name == "lookup" else "LIVE_RESOLVE"

        def factory(seed: int) -> object:
            return PokerAgent(
                source=strategy,
                config=ModeConfig(
                    mode=mode,
                    seed=seed,
                    blueprint_path=path,
                    uniform_on_miss=True,
                ),
            )

        return factory

    hero_factory = make_factory("hero")
    field_factory = make_factory("field")
    if tracker is not None:
        setattr(hero_factory, "_solve_wall_tracker", tracker)
        setattr(field_factory, "_solve_wall_tracker", tracker)
    if has_live_source or args.max_wall_s is not None:
        metadata.update({
            "seats": int(args.seats),
            "hero_source": args.hero_source,
            "field_source": args.field_source,
            "live_iterations": int(args.live_iterations),
            "live_num_rollouts": args.live_num_rollouts,
            "leaf_mode_envs": {
                role: {
                    "source": role_sources[role],
                    **(
                        {
                            "live_num_rollouts_bounds": (
                                "sampled_direct"
                                if role_sources[role] == "live_sampled"
                                else "eager_config"
                            )
                        }
                        if role_sources[role] != "lookup"
                        else {}
                    ),
                    "env_overrides": resolved_leaf_mode_env(
                        role_sources[role]
                    ),
                    "effective": effective_leaf_mode_env(
                        role_sources[role]
                    ),
                }
                for role in ("hero", "field")
            },
        })
    use_field_factory = args.mode == "ab" or has_live_source
    return (
        hero_factory,
        field_factory if use_field_factory else None,
        metadata,
    )


def write_match(run: Any, out_dir: object) -> Tuple[Path, Path]:
    """Write deterministic JSONL group records plus one summary JSON."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups_path = out / "deal_groups.jsonl"
    groups_tmp = out / "deal_groups.jsonl.tmp"
    mode = str(run.summary["mode"])
    master_seed = int(run.summary["master_seed"])
    epsilon_value = run.summary.get("epsilon")
    record_epsilon = (
        float(epsilon_value) if epsilon_value is not None else None
    )
    with groups_tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for group in run.groups:
            json.dump(
                group.as_dict(
                    mode=mode,
                    master_seed=master_seed,
                    epsilon=record_epsilon,
                ),
                fh,
                sort_keys=True,
                separators=(",", ":"),
            )
            fh.write("\n")
    os.replace(str(groups_tmp), str(groups_path))

    summary_path = out / "summary.json"
    summary_tmp = out / "summary.json.tmp"
    with summary_tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(run.summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(str(summary_tmp), str(summary_path))
    return groups_path, summary_path


def main(runner: Any, argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI against the passed pure match module."""

    args = runner._parse_args(argv)
    try:
        hero_factory, field_factory, metadata = runner._lookup_factories(
            args
        )
        tracker = getattr(
            hero_factory, "_solve_wall_tracker", None
        ) or getattr(field_factory, "_solve_wall_tracker", None)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        partial_path = out / "deal_groups.jsonl.partial"
        record_epsilon = args.epsilon if args.mode == "epsilon" else None
        progress: Dict[str, object] = {"n": 0, "fh": None}

        def on_group(group: object) -> None:
            if progress["fh"] is None:
                progress["fh"] = partial_path.open(
                    "w", encoding="utf-8", newline="\n"
                )
            fh = progress["fh"]
            json.dump(
                group.as_dict(
                    mode=args.mode,
                    master_seed=args.master_seed,
                    epsilon=record_epsilon,
                ),
                fh,
                sort_keys=True,
                separators=(",", ":"),
            )
            fh.write("\n")
            fh.flush()
            progress["n"] = int(progress["n"]) + 1
            print(
                f"[match] group {progress['n']}/{args.deals} complete",
                file=sys.stderr,
                flush=True,
            )

        try:
            run = runner.run_match(
                deals=args.deals,
                mode=args.mode,
                master_seed=args.master_seed,
                epsilon=args.epsilon,
                a_factory=hero_factory,
                b_factory=field_factory,
                on_group=on_group,
                solve_wall_tracker=tracker,
                max_wall_s=args.max_wall_s,
                allin_equity_adjust=args.allin_equity_adjust,
            )
        finally:
            if progress["fh"] is not None:
                progress["fh"].close()
        run = runner.MatchRun(
            groups=run.groups, summary={**run.summary, **metadata}
        )
        groups_path, summary_path = runner.write_match(run, args.out)
        partial_path.unlink(missing_ok=True)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"[match] wrote {groups_path} and {summary_path}")
    print(
        "MATCH_SUMMARY "
        + json.dumps(
            run.summary, sort_keys=True, separators=(",", ":")
        )
    )
    return 0
