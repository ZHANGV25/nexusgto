"""Measure blueprint strategy-row coverage along blueprint self-play.

Phase 1 of the 2026-09-12 Pluribus-quality plan. The research audits found that
97.8-99.2% of later-node lookups in their (untrained-policy) evaluations had no
exported blueprint record. That population was not the blueprint's own play, so
this census asks the question that decides "history translation vs retrain":
how often does a blueprint-only six-max agent hit real trained rows on the
trajectories it actually plays?

Every lookup is classified as:

``absent``
    The reader has no node for the history (explicit uniform substitution).
``uniform``
    The stored row is exactly uniform. ``bps_reader.c`` fills every new node
    block with ``255 / na`` before copying exported buckets, so an unexported
    bucket reads as exactly uniform. A genuinely trained uniform row reads the
    same way, so this rate is an UPPER bound on padded support.
``trained``
    A non-uniform stored row: exported support.

Lookups come from ``run_match`` ``aa`` mode, so they are weighted by the
blueprint's own reach. The six duplicate arms of a deal group replay the same
trajectory, which scales every count by six and leaves rates unchanged.

Run from the api root::

    $env:POKER_SOLVER_PATH="C:/.../poker-solver"
    python scripts/blueprint_coverage_census.py --deals 200 --blueprint <8.05B.bps> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from scripts import run_match  # noqa: E402

STATUSES = ("absent", "uniform", "trained")
STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
UNIFORM_TOLERANCE = 1e-6
SCHEMA = "nexusgto_blueprint_coverage_v1"

CensusKey = Tuple[int, int, str]  # (street, n_active, status)


def classify_row(
    strategy: Optional[Sequence[float]], tolerance: float = UNIFORM_TOLERANCE
) -> str:
    """Classify one reader result as absent, uniform or trained."""

    if strategy is None or len(strategy) == 0:
        return "absent"
    values = [float(value) for value in strategy]
    if max(values) - min(values) <= tolerance:
        return "uniform"
    return "trained"


def _rates(counts: Mapping[str, int]) -> Dict[str, object]:
    total = sum(counts.get(status, 0) for status in STATUSES)
    return {
        "lookups": total,
        **{status: counts.get(status, 0) for status in STATUSES},
        **{
            f"{status}_rate": (
                round(counts.get(status, 0) / total, 6) if total else None
            )
            for status in STATUSES
        },
    }


def summarize(counts: Mapping[CensusKey, int]) -> Dict[str, object]:
    """Aggregate raw (street, n_active, status) counts into rate tables."""

    overall: Counter = Counter()
    by_street: Dict[str, Counter] = {}
    by_street_players: Dict[str, Counter] = {}
    for (street, n_active, status), count in counts.items():
        street_name = STREET_NAMES.get(street, f"street{street}")
        overall[status] += count
        by_street.setdefault(street_name, Counter())[status] += count
        cell = f"{street_name}/{n_active}p"
        by_street_players.setdefault(cell, Counter())[status] += count
    return {
        "schema": SCHEMA,
        "overall": _rates(overall),
        "by_street": {name: _rates(c) for name, c in sorted(by_street.items())},
        "by_street_players": {
            name: _rates(c) for name, c in sorted(by_street_players.items())
        },
        "uniform_is_upper_bound": (
            "exactly-uniform rows include reader-padded unexported buckets and "
            "any genuinely uniform trained rows"
        ),
    }


def _active_count(state: object) -> int:
    active = getattr(state, "active", None)
    if active is None:
        return -1
    return sum(1 for is_active in active if is_active)


class CensusSource:
    """Classify each lookup, then delegate play to the wrapped LookupSource.

    The extra reader query is side-effect free, so the wrapped source's action
    choice, RNG use and fallback behaviour are unchanged.
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.counts: Counter = Counter()

    def get_mix(self, state: object, hero_cards: Tuple[int, int]) -> object:
        from app.engine.agent.strategy_source import build_key

        key = build_key(state)
        bucket = self._inner._bucket(state, hero_cards)
        strategy = None
        if bucket >= 0:
            strategy = self._inner._reader.get_strategy(
                list(state.board), key.action_history, key.player, bucket,
                street=key.street,
            )
        status = classify_row(strategy)
        self.counts[(int(key.street), _active_count(state), status)] += 1
        return self._inner.get_mix(state, hero_cards)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.deals <= 0:
        ap.error("--deals must be positive")
    if not Path(args.blueprint).is_file():
        ap.error(f"blueprint not found: {args.blueprint}")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    from app.engine.agent.agent import PokerAgent
    from app.engine.agent.modes import ModeConfig
    from app.engine.agent.strategy_source import LookupSource

    path = str(Path(args.blueprint).resolve())
    census = CensusSource(
        LookupSource(
            ModeConfig(mode="LOOKUP", blueprint_path=path, uniform_on_miss=True)
        )
    )

    def factory(seed: int) -> object:
        return PokerAgent(
            source=census,
            config=ModeConfig(
                mode="LOOKUP", seed=seed, blueprint_path=path, uniform_on_miss=True
            ),
        )

    run = run_match.run_match(
        deals=args.deals, mode="aa", master_seed=args.master_seed, a_factory=factory
    )
    summary = {
        **summarize(census.counts),
        "deal_groups": len(run.groups),
        "master_seed": args.master_seed,
        "blueprint": path,
        "raw_counts": [
            {"street": s, "n_active": n, "status": st, "count": c}
            for (s, n, st), c in sorted(census.counts.items())
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("COVERAGE_SUMMARY " + json.dumps(summary["overall"], sort_keys=True))
    for name, row in summary["by_street"].items():
        print(f"  {name}: {json.dumps(row, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
