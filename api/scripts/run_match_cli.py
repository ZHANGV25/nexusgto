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
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


SOURCE_CHOICES = ("lookup", "live_eager", "live_sampled", "production")

# The server's solve-shaping environment (stage-userdata.sh, staged
# 2026-09-10: API 36a1680 / solver 9f108f1). A "production" source refuses to
# run without it, so a match can never be labeled production while solving a
# different arm. Source the uvicorn /proc/<pid>/environ before running.
PRODUCTION_SERVE_ENV = {
    "NEXUS_LEAF_VEC_SAMPLED": "1",
    "NEXUS_HU_LEAF_MIRROR": "1",
    "NEXUS_OFFTREE_ADD": "1",
    "NEXUS_MAX_PROJECTED_FILL_BYTES": "7000000",
}
# (cdp_ui.SOLVE_MAX_LIVE_PLAYERS) and the API refuses them; the lookup floor acts.
PRODUCTION_MAX_LIVE_PLAYERS = 3

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


def production_env_problems(environ: Mapping[str, str]) -> List[str]:
    """Differences between ``environ`` and the server's solve environment."""

    problems = []
    for name, expected in PRODUCTION_SERVE_ENV.items():
        actual = environ.get(name)
        if actual != expected:
            problems.append(f"{name}={actual!r} (server: {expected!r})")
    for name in _CONTROLLED_LEAF_ENVS:
        if environ.get(name):
            problems.append(f"{name}={environ[name]!r} is not set on the server")
    return problems


def resolved_leaf_mode_env(source: str) -> Dict[str, Optional[str]]:
    """Return the leaf-mode environment enforced for one source."""

    if source in ("lookup", "production"):
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
    if source == "production":
        return {name: str(os.environ.get(name)) for name in PRODUCTION_SERVE_ENV}
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
    if source in ("lookup", "production"):
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


def _production_fallback_errors() -> Tuple[type, ...]:
    from app.engine.agent.live_resolve_adapter import (
        LiveResolveConvergenceError,
        LiveResolveUnsupported,
    )
    from app.engine.cpu_es_solver import (
        LivePlayerCountExceeded,
        ProjectedFillMemoryExceeded,
    )

    return (
        LiveResolveUnsupported,
        LiveResolveConvergenceError,
        LivePlayerCountExceeded,
        ProjectedFillMemoryExceeded,
    )


class ProductionFallbackSource:
    """A live re-solve whose refusals act on the lookup floor, as in production.

    The API turns these error classes into a lookup tier or a coded error, and
    """

    def __init__(
        self,
        live: object,
        lookup: object,
        fallback_errors: Optional[Tuple[type, ...]] = None,
        live_streets: Optional[FrozenSet[int]] = None,
        mix_compare: Optional[Any] = None,
    ) -> None:
        self._live = live
        self._lookup = lookup
        self._fallback_errors = (
            tuple(fallback_errors)
            if fallback_errors is not None
            else _production_fallback_errors()
        )
        # A/B: streets outside this set act on the lookup floor without a solve.
        self._live_streets = live_streets
        self.fallbacks: Dict[str, int] = {}
        self.spots: Tuple[Dict[str, object], ...] = ()
        self.gated = 0
        # Belief-carry engagement read from LiveResolveSource.last_belief_carry.
        self.belief_carry: Dict[str, int] = {}
        # Postflop range conditioning read from LiveResolveSource.last_range_conditioning.
        self.range_conditioning: Dict[str, int] = {}
        self.belief_drop_reasons: Dict[str, int] = {}
        # Research: served mix vs blueprint mix at decisions live search served.
        self._mix_compare = mix_compare

    def get_mix(self, state: object, hero_cards: object) -> object:
        if (
            self._live_streets is not None
            and int(getattr(state, "street", -1)) not in self._live_streets
        ):
            self.gated = self.gated + 1
            return self._lookup.get_mix(state, hero_cards)
        try:
            mix = self._live.get_mix(state, hero_cards)
        except self._fallback_errors as exc:
            name = type(exc).__name__
            self.fallbacks = {**self.fallbacks, name: self.fallbacks.get(name, 0) + 1}
            if len(self.spots) < MAX_RECORDED_FALLBACK_SPOTS:
                self.spots = self.spots + (_fallback_spot(state, hero_cards, exc),)
            return self._lookup.get_mix(state, hero_cards)
        finally:
            self._record_belief_carry()
            self._record_range_conditioning()
        if self._mix_compare is not None:
            self._mix_compare.add(
                state, hero_cards, mix, self._lookup.get_mix(state, hero_cards)
            )
        return mix

    def _record_range_conditioning(self) -> None:
        stats = getattr(self._live, "last_range_conditioning", None)
        if not isinstance(stats, dict):
            return
        events = {"solves": 1}
        if stats.get("error"):
            events["errors"] = 1
        if stats.get("decisions"):
            events["with_earlier_streets"] = 1
        events["players_conditioned"] = int(stats.get("players_conditioned", 0))
        events["players_zero_mass"] = int(stats.get("players_zero_mass", 0))
        events["lookups"] = int(stats.get("lookups", 0))
        events["uniform_rows"] = int(stats.get("uniform_rows", 0))
        self.range_conditioning = _sum_counts([self.range_conditioning, events])

    def _record_belief_carry(self) -> None:
        summary = getattr(self._live, "last_belief_carry", None)
        if not isinstance(summary, dict):
            return
        events = ["solves"]
        events.extend(key for key in ("applied", "dropped") if summary.get(key))
        if summary.get("folded_streets"):
            events.append("folded")
        self.belief_carry = _sum_counts([self.belief_carry, {key: 1 for key in events}])
        reason = summary.get("drop_reason")
        if reason:
            self.belief_drop_reasons = _sum_counts(
                [self.belief_drop_reasons, {str(reason)[:160]: 1}]
            )


MAX_RECORDED_FALLBACK_SPOTS = 2000
PRODUCTION_STREETS = {"flop": 1, "turn": 2, "river": 3}


def _sum_counts(counts: Sequence[Mapping[str, int]]) -> Dict[str, int]:
    total: Dict[str, int] = {}
    for mapping in counts:
        for name, count in mapping.items():
            total = {**total, name: total.get(name, 0) + int(count)}
    return total


def _street_set(raw: str) -> FrozenSet[int]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in PRODUCTION_STREETS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated streets from {sorted(PRODUCTION_STREETS)}, "
            f"got {raw!r}"
        )
    return frozenset(PRODUCTION_STREETS[name] for name in names)


def street_names(streets: Optional[FrozenSet[int]]) -> Optional[List[str]]:
    if streets is None:
        return None
    return [name for name, index in PRODUCTION_STREETS.items() if index in streets]


# A/B: leaf-model flags a production arm may change after the serve environment
# check passes. An empty value unsets the flag.
PRODUCTION_ENV_OVERRIDABLE = (
    "NEXUS_LEAF_VEC_SAMPLED",
    "NEXUS_LEAF_ROLLOUT",
    "NEXUS_LEAF_CACHE_K",
    "NEXUS_LEAF_RANGE_CONDITIONING",
    "NEXUS_VEC_SAMPLED_HERO_BIAS",
    "NEXUS_VEC_SAMPLED_EPS",
    "NEXUS_VEC_SAMPLED_RUNOUT",
    "NEXUS_VEC_SAMPLED_NR",
    "NEXUS_VEC_SAMPLED_OPP_BIAS",
    "NEXUS_VEC_SAMPLED_HERO_FIXED",
    "NEXUS_VEC_SAMPLED_OPP_HAND",
    "NEXUS_RIVER_AVG_READOUT",
    "NEXUS_RIVER_AVG_ITERATIONS",
    "NEXUS_RANGE_POSTFLOP_CONDITIONING",
    "NEXUS_RANGE_COMBO_PRIOR",
    "NEXUS_FLOP_CHANCE_SOLVE",
    "NEXUS_FLOP_CHANCE_TURN_CARDS",
    "NEXUS_FLOP_CHANCE_RIVER_CARDS",
    "NEXUS_FLOP_CHANCE_SUBSET_SEED",
    "NEXUS_MASTER_SEED_OVERRIDE",
)


def _env_override(raw: str) -> Tuple[str, str]:
    name, sep, value = raw.partition("=")
    if not sep or name not in PRODUCTION_ENV_OVERRIDABLE:
        raise argparse.ArgumentTypeError(
            f"expected NAME=VALUE with NAME in {list(PRODUCTION_ENV_OVERRIDABLE)}, got {raw!r}"
        )
    return name, value


def apply_env_overrides(
    pairs: Sequence[Tuple[str, str]], environ: Any
) -> Dict[str, Optional[str]]:
    """Set (or unset, for an empty value) each flag; return what was applied."""
    applied: Dict[str, Optional[str]] = {}
    for name, value in pairs:
        if value == "":
            environ.pop(name, None)
            applied = {**applied, name: None}
        else:
            environ[name] = value
            applied = {**applied, name: value}
    return applied


def _fallback_spot(state: object, hero_cards: object, exc: Exception) -> Dict[str, object]:
    """A replayable description of one production live-solve fallback."""

    def _tuple(name: str) -> Optional[List[object]]:
        value = getattr(state, name, None)
        return list(value) if value is not None else None

    return {
        "error": type(exc).__name__,
        "message": str(exc)[:400],
        "street": getattr(state, "street", None),
        "n_live": sum(1 for flag in (getattr(state, "active", ()) or ()) if flag),
        "board": _tuple("board"),
        "preflop_prefix": _tuple("preflop_prefix"),
        "postflop_history": _tuple("postflop_history"),
        "seat_of_slot": _tuple("seat_of_slot"),
        "stacks": _tuple("stacks"),
        "pot": getattr(state, "pot", None),
        "to_act_slot": getattr(state, "to_act_slot", None),
        "hero_cards": list(hero_cards) if hero_cards is not None else None,
    }


def _production_strategy(
    *,
    path: str,
    lookup: object,
    engine: object,
    live_config_for_fn: Any,
    num_threads: int,
    iterations_override: Optional[int] = None,
    deadline_ms_override: Optional[int] = None,
    belief_carry: bool = False,
    live_streets: Optional[FrozenSet[int]] = None,
    mix_compare: Optional[Any] = None,
    max_live_players: int = PRODUCTION_MAX_LIVE_PLAYERS,
) -> Tuple[object, List[ProductionFallbackSource]]:

    Player counts whose configs are identical share one LiveResolveSource, so
    a belief-carry session survives a street-root count change within a hand
    (a three-way flop becoming a heads-up turn). Sources hold no other
    cross-call state, so sharing is neutral with carry off.

    ``max_live_players`` above the served cap is a research A/B: the engine's
    own cap (NEXUS_MAX_LIVE_RESOLVE_PLAYERS) must be raised alongside it.
    """

    from app.engine.agent.modes import ModeConfig
    from app.engine.agent.strategy_source import LiveResolveSource

    if not (2 <= int(max_live_players) <= 6):
        raise ValueError(
            f"max_live_players must be within [2, 6], got {max_live_players}"
        )
    live_by_count: Dict[int, object] = {}
    by_config: Dict[Tuple[object, ...], ProductionFallbackSource] = {}
    for n_players in range(2, 7):
        if n_players > int(max_live_players):
            live_by_count[n_players] = lookup
            continue
        cfg = live_config_for_fn(
            n_players,
            iterations_override=iterations_override,
            deadline_ms_override=deadline_ms_override,
            purpose="play",
        )
        key = (
            cfg.iterations, cfg.vec_mode, cfg.readout, cfg.num_rollouts,
            cfg.deadline_ms, cfg.hero_traverser_actual_fraction,
            cfg.hero_row_min_strategy_sum_mass,
        )
        source = by_config.get(key)
        if source is None:
            live = LiveResolveSource(
                ModeConfig(mode="LIVE_RESOLVE", blueprint_path=path),
                cpu_es_solver=engine,
                iterations=cfg.iterations,
                vec_mode=cfg.vec_mode,
                readout=cfg.readout,
                num_rollouts=cfg.num_rollouts,
                deadline_ms=cfg.deadline_ms,
                hero_traverser_actual_fraction=cfg.hero_traverser_actual_fraction,
                hero_row_min_strategy_sum_mass=cfg.hero_row_min_strategy_sum_mass,
                num_threads=num_threads,
                # None follows NEXUS_BELIEF_CARRY (unset on the server).
                belief_carry=True if belief_carry else None,
            )
            source = ProductionFallbackSource(
                live, lookup, live_streets=live_streets, mix_compare=mix_compare
            )
            by_config[key] = source
        live_by_count[n_players] = source
    return _LiveByPlayerCountSource(lookup, live_by_count), list(by_config.values())


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
    ap.add_argument(
        "--start-group",
        type=int,
        default=0,
        help="skip this many decks: shard a seed's groups across processes",
    )
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
        "--live-threads",
        type=int,
        default=0,
        help="native solve threads per decision (0 = all cores, production)",
    )
    ap.add_argument(
        "--production-deadline-ms",
        type=_positive_int,
        default=None,
        help="A/B override of the production CFR deadline (served: 3000)",
    )
    ap.add_argument(
        "--production-iterations",
        type=_positive_int,
        default=None,
        help="A/B override of the production iteration ceiling (served: 100000)",
    )
    ap.add_argument(
        "--belief-carry",
        action="store_true",
        default=False,
        help="A/B: carry cross-street beliefs in production re-solves (off on the box)",
    )
    ap.add_argument(
        "--production-live-streets",
        type=_street_set,
        default=None,
        help="A/B: re-solve only these streets (comma list of flop,turn,river); "
        "the rest act on lookup (served: all three)",
    )
    ap.add_argument(
        "--capture-river-brgap",
        default=None,
        help="research: pickle heads-up river street-root solves with the "
        "blueprint projected onto the same tree into this directory "
        "(scripts/river_brgap_score.py); sets NEXUS_BR_GAP_EXPOSE=1",
    )
    ap.add_argument(
        "--capture-limit",
        type=_positive_int,
        default=500,
        help="maximum spots captured per process (river BR-gap or solve requests)",
    )
    ap.add_argument(
        "--capture-solve-requests",
        default=None,
        help="research: pickle the exact solve inputs of every hero production "
        "re-solve (any street or player count) into this directory "
        "(scripts/solve_ladder_bench.py replays them at other budgets)",
    )
    ap.add_argument(
        "--capture-only",
        action="store_true",
        default=False,
        help="with --capture-solve-requests: record each request and refuse the "
        "solve so the lookup floor acts (a fast spot generator, not a result)",
    )
    ap.add_argument(
        "--capture-min-players",
        type=int,
        default=2,
        help="with --capture-solve-requests: record only street-entry player "
        "counts at or above this",
    )
    ap.add_argument(
        "--production-max-live-players",
        type=int,
        default=None,
        "street roots re-solve too; sets NEXUS_MAX_LIVE_RESOLVE_PLAYERS",
    )
    ap.add_argument(
        "--field-style",
        choices=("station", "maniac", "nit"),
        default=None,
        help="mode ab: field seats play the blueprint with this style's leaks "
        "(scripts/field_styles.py)",
    )
    ap.add_argument(
        "--record-mix-compare",
        action="store_true",
        default=False,
        help="research: at every served live decision also query the blueprint and "
        "record both mixes (mix_compare.jsonl + summary; scripts/mix_compare.py)",
    )
    ap.add_argument(
        "--production-env",
        type=_env_override,
        action="append",
        default=None,
        help="A/B: NAME=VALUE leaf-model flag applied after the serve environment "
        "check (repeatable; empty VALUE unsets)",
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
    ap.add_argument(
        "--aivat",
        action="store_true",
        default=False,
        help=(
            "score adjusted deltas with self-play AIVAT (scripts/match_aivat.py): "
            "decision and exact turn/river corrections from known policies and "
            "cards; subsumes --allin-equity-adjust"
        ),
    )
    ap.add_argument(
        "--hand-log",
        action="store_true",
        default=False,
        help="write chronological nexusgto_handlog_v1 events",
    )
    args = ap.parse_args(argv)
    if args.field_style is not None and args.mode != "ab":
        ap.error("--field-style needs --mode ab (the field differs from the hero)")
    if args.aivat and args.hand_log:
        ap.error("--aivat and --hand-log both replace the hand player; pick one")
    production_overrides = (
        args.production_deadline_ms is not None
        or args.production_iterations is not None
        or args.belief_carry
        or args.production_live_streets is not None
        or args.capture_river_brgap is not None
        or args.capture_solve_requests is not None
        or args.production_max_live_players is not None
        or args.record_mix_compare
        or args.production_env is not None
    )
    if production_overrides and "production" not in (
        args.hero_source, args.field_source
    ):
        ap.error(
            "--production-deadline-ms/--production-iterations/--belief-carry/"
            "--production-live-streets/--capture-river-brgap/--capture-solve-requests/"
            "--production-max-live-players/--record-mix-compare/"
            "--production-env need a production source"
        )
    if (args.capture_only or args.capture_min_players != 2) and (
        args.capture_solve_requests is None
    ):
        ap.error("--capture-only/--capture-min-players need --capture-solve-requests")
    if not (2 <= args.capture_min_players <= 6):
        ap.error("--capture-min-players must be within [2, 6]")
    if args.production_max_live_players is not None and not (
        2 <= args.production_max_live_players <= 6
    ):
        ap.error("--production-max-live-players must be within [2, 6]")
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
    requested = {args.hero_source, args.field_source}
    if "production" in requested:
        if requested & {"live_eager", "live_sampled"}:
            raise ValueError(
                "production cannot share a process with live_eager/live_sampled: "
                "their leaf environments conflict"
            )
        problems = production_env_problems(os.environ)
        if problems:
            raise ValueError(
                "production source needs the server environment: "
                + "; ".join(problems)
            )
        apply_env_overrides(getattr(args, "production_env", None) or (), os.environ)
    max_live_players = int(
        getattr(args, "production_max_live_players", None)
        or PRODUCTION_MAX_LIVE_PLAYERS
    )
    if max_live_players != PRODUCTION_MAX_LIVE_PLAYERS:
        # The engine refuses above its own cap; raise both for the A/B.
        os.environ["NEXUS_MAX_LIVE_RESOLVE_PLAYERS"] = str(max_live_players)
    live_threads = int(getattr(args, "live_threads", 0))
    if live_threads < 0:
        raise ValueError("--live-threads must be >= 0")
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
            # Production constructs under the serve environment as-is.
            with scoped_leaf_mode(
                "production" if "production" in requested else "live_eager"
            ):
                engine = CpuEsSolver(path)
            engine_by_path[path] = engine
        return engine

    strategy_by_role: Dict[str, object] = {}
    production_sources: List[ProductionFallbackSource] = []
    river_captures: List[object] = []
    request_recorders: List[object] = []
    mix_compare = None
    if getattr(args, "record_mix_compare", False):
        from scripts.mix_compare import MixComparison

        mix_compare = MixComparison()
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
        if source_name == "production":
            production_engine: object = timed_engine
            request_dir = getattr(args, "capture_solve_requests", None)
            if request_dir and role == "hero":
                from scripts.solve_request_capture import SolveRequestRecorder

                production_engine = SolveRequestRecorder(
                    timed_engine, request_dir,
                    limit=int(getattr(args, "capture_limit", 500)),
                    capture_only=bool(getattr(args, "capture_only", False)),
                    min_players=int(getattr(args, "capture_min_players", 2)),
                )
                request_recorders.append(production_engine)
            strategy, sources = _production_strategy(
                path=path,
                lookup=lookup,
                engine=production_engine,
                live_config_for_fn=live_config_for_fn,
                num_threads=live_threads,
                iterations_override=getattr(args, "production_iterations", None),
                deadline_ms_override=getattr(args, "production_deadline_ms", None),
                belief_carry=bool(getattr(args, "belief_carry", False)),
                live_streets=getattr(args, "production_live_streets", None),
                mix_compare=mix_compare if role == "hero" else None,
                max_live_players=max_live_players,
            )
            capture_dir = getattr(args, "capture_river_brgap", None)
            if capture_dir and role == "hero":
                from scripts.river_brgap_capture import RiverCaptureSource

                os.environ["NEXUS_BR_GAP_EXPOSE"] = "1"
                strategy = RiverCaptureSource(
                    strategy, lookup, production_engine, capture_dir,
                    int(getattr(args, "capture_limit", 500)),
                )
                river_captures.append(strategy)
            strategy_by_role[role] = strategy
            production_sources.extend(sources)
            continue
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

    field_style = getattr(args, "field_style", None)
    if field_style:
        from scripts.field_styles import StyleSource

        styled = StyleSource(strategy_by_role["field"], field_style)
        strategy_by_role["field"] = styled
        metadata.update({
            "field_style": {"name": field_style, "params": styled.style.as_dict()},
        })

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
    if production_sources:
        setattr(hero_factory, "_production_sources", tuple(production_sources))
        metadata.update({
            "production_serve_env": effective_leaf_mode_env("production"),
            "production_max_live_players": max_live_players,
            "live_threads": live_threads,
            "production_ab_overrides": {
                "deadline_ms": getattr(args, "production_deadline_ms", None),
                "iterations": getattr(args, "production_iterations", None),
                "belief_carry": bool(getattr(args, "belief_carry", False)),
                "live_streets": street_names(
                    getattr(args, "production_live_streets", None)
                ),
                "env": {
                    name: (value or None)
                    for name, value in (getattr(args, "production_env", None) or ())
                },
            },
        })
    if mix_compare is not None:
        setattr(hero_factory, "_mix_compare", mix_compare)
    if river_captures:
        setattr(hero_factory, "_river_captures", tuple(river_captures))
        metadata.update({
            "river_brgap_capture": {
                "dir": str(args.capture_river_brgap),
                "limit": int(args.capture_limit),
                "nexus_br_gap_expose": os.environ.get("NEXUS_BR_GAP_EXPOSE"),
            },
        })
    if request_recorders:
        setattr(hero_factory, "_request_recorders", tuple(request_recorders))
        metadata.update({
            "solve_request_capture": {
                "dir": str(args.capture_solve_requests),
                "limit": int(args.capture_limit),
                "capture_only": bool(getattr(args, "capture_only", False)),
                "min_players": int(getattr(args, "capture_min_players", 2)),
            },
        })
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
        use_aivat = bool(getattr(args, "aivat", False))
        hand_log_emitter = None
        if args.hand_log:
            from scripts.hand_log_emitter import HandLogEmitter

            hand_log_emitter = HandLogEmitter(
                out,
                hand_player=runner.play_full_hand,
                mode=args.mode,
                master_seed=args.master_seed,
                epsilon=args.epsilon,
                hero_source=args.hero_source,
                field_source=args.field_source,
                allin_equity_adjust=args.allin_equity_adjust,
            )
        partial_path = out / "deal_groups.jsonl.partial"
        record_epsilon = args.epsilon if args.mode == "epsilon" else None
        progress: Dict[str, object] = {"n": 0, "fh": None}

        def progress_on_group(group: object) -> None:
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

        group_callback = progress_on_group
        if hand_log_emitter is not None:
            active_emitter = hand_log_emitter

            def logged_on_group(group: object) -> None:
                active_emitter.on_group(group)
                progress_on_group(group)

            group_callback = logged_on_group

        try:
            match_kwargs = {
                "deals": args.deals,
                "mode": args.mode,
                "master_seed": args.master_seed,
                "epsilon": args.epsilon,
                "a_factory": hero_factory,
                "b_factory": field_factory,
                "on_group": group_callback,
                "solve_wall_tracker": tracker,
                "max_wall_s": args.max_wall_s,
                "allin_equity_adjust": args.allin_equity_adjust or use_aivat,
                "start_group": int(getattr(args, "start_group", 0)),
            }
            metadata = {**metadata, "start_group": match_kwargs["start_group"]}
            if use_aivat:
                from scripts.match_aivat import ESTIMATOR, play_full_hand_aivat

                match_kwargs["hand_player"] = play_full_hand_aivat
                metadata = {**metadata, "adjusted_estimator": ESTIMATOR}
            if hand_log_emitter is not None:
                match_kwargs["hand_player"] = hand_log_emitter.hand_player
            run = runner.run_match(**match_kwargs)
        finally:
            if progress["fh"] is not None:
                progress["fh"].close()
        production_sources = getattr(hero_factory, "_production_sources", ())
        if production_sources:
            fallbacks: Dict[str, int] = {}
            spots: List[Dict[str, object]] = []
            for source in production_sources:
                spots.extend(source.spots)
                for name, count in source.fallbacks.items():
                    fallbacks = {**fallbacks, name: fallbacks.get(name, 0) + count}
            spots_path = out / "production_fallback_spots.jsonl"
            with spots_path.open("w", encoding="utf-8", newline="\n") as fh:
                for spot in spots:
                    fh.write(json.dumps(spot, sort_keys=True) + "\n")
            metadata = {
                **metadata,
                "production_live_fallbacks": fallbacks,
                "production_fallback_spots_recorded": len(spots),
                "production_street_gated": sum(s.gated for s in production_sources),
                "production_belief_carry": _sum_counts(
                    [s.belief_carry for s in production_sources]
                ),
                "production_range_conditioning": _sum_counts(
                    [s.range_conditioning for s in production_sources]
                ),
                "production_belief_drop_reasons": dict(sorted(
                    _sum_counts(
                        [s.belief_drop_reasons for s in production_sources]
                    ).items(),
                    key=lambda item: -item[1],
                )[:10]),
            }
        river_captures = getattr(hero_factory, "_river_captures", ())
        if river_captures:
            metadata = {
                **metadata,
                "river_brgap_captured": sum(c.captured for c in river_captures),
                "river_brgap_skipped": _sum_counts([c.skipped for c in river_captures]),
            }
        request_recorders = getattr(hero_factory, "_request_recorders", ())
        if request_recorders:
            metadata = {
                **metadata,
                "solve_requests_captured": sum(r.captured for r in request_recorders),
                "solve_requests_skipped": _sum_counts(
                    [r.skipped for r in request_recorders]
                ),
            }
        mix_compare = getattr(hero_factory, "_mix_compare", None)
        if mix_compare is not None:
            mix_compare.write_rows(out / "mix_compare.jsonl")
            metadata = {**metadata, "mix_compare": mix_compare.summary()}
        run = runner.MatchRun(
            groups=run.groups, summary={**run.summary, **metadata}
        )
        groups_path, summary_path = runner.write_match(run, args.out)
        if hand_log_emitter is not None:
            hand_log_emitter.finalize()
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
