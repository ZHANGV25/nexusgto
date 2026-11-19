"""Record the exact CpuEsSolver.solve inputs of a production source's live re-solves.

Every such re-solve (any street, any street-entry player count) reaches
the engine as plain keyword arguments: board, seeded ranges, positions, pot
geometry, bet sizes, hero replay and the hero's actual-hand row. Recording
those inputs lets ``scripts/solve_ladder_bench.py`` re-run the identical solve
offline at any iteration budget. In ``capture_only`` mode the recorder refuses
the solve after recording it (``LiveResolveUnsupported``), so the lookup floor
acts and the match runs at lookup speed: a spot generator, not a strength
measurement.
"""
from __future__ import annotations

import copy
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

SCHEMA = "solve_request_v1"
STREET_BY_BOARD_LEN = {3: "flop", 4: "turn", 5: "river"}
_STATS_KEYS = (
    "iterations", "iterations_run", "solve_s", "construct_s", "readout_s",
    "num_nodes", "num_leaf_nodes", "eff_threads", "leaf_mode", "deadline_ms",
)


class CaptureOnlyRefusal(Exception):
    """Placeholder; the real refusal class is resolved at capture time."""


def _refusal_class() -> type:
    from app.engine.agent.live_resolve_adapter import LiveResolveUnsupported

    return LiveResolveUnsupported


def describe_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Street, player count and hero identity of one solve request."""
    board = kwargs.get("board_ints") or ()
    ranges = kwargs.get("player_ranges") or ()
    hero = kwargs.get("hero_actual_hand") or {}
    replay = kwargs.get("hero_replay") or {}
    return {
        "street": STREET_BY_BOARD_LEN.get(len(board), f"board{len(board)}"),
        "n_players": len(ranges),
        "hero_player_idx": hero.get("player_idx", replay.get("player_idx")),
        "hero_hand_idx": hero.get("hand_idx"),
        "street_actions": tuple(replay.get("street_actions", ())),
        "offtree": "offtree" in replay,
    }


def _stats_subset(stats: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(stats, dict):
        return None
    return {key: stats.get(key) for key in _STATS_KEYS}


class SolveRequestRecorder:
    """Proxy around the production source's engine that pickles each solve request."""

    def __init__(
        self,
        inner: Any,
        out_dir: str,
        limit: int,
        capture_only: bool = False,
        min_players: int = 2,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._inner = inner
        self._out_dir = Path(out_dir)
        self._limit = int(limit)
        self._capture_only = bool(capture_only)
        self._min_players = int(min_players)
        self._clock = clock
        self.captured = 0
        self.skipped: Dict[str, int] = {}

    def _skip(self, reason: str) -> None:
        self.skipped = {**self.skipped, reason: self.skipped.get(reason, 0) + 1}

    def _eligible(self, description: Dict[str, Any]) -> bool:
        if self.captured >= self._limit:
            self._skip("limit_reached")
            return False
        if int(description["n_players"]) < self._min_players:
            self._skip("below_min_players")
            return False
        return True

    def solve(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise ValueError("solve request capture needs keyword-only solve calls")
        description = describe_request(kwargs)
        eligible = self._eligible(description)
        if eligible and self._capture_only:
            self._write(kwargs, description, outcome="capture_only")
            raise _refusal_class()(
                "solve request captured; capture-only mode refuses the solve"
            )
        started = self._clock()
        try:
            result = self._inner.solve(**kwargs)
        except Exception as exc:
            if eligible:
                self._write(
                    kwargs, description,
                    outcome=f"error:{type(exc).__name__}",
                    wall_s=self._clock() - started,
                )
            raise
        if eligible:
            self._write(
                kwargs, description, outcome="solved",
                wall_s=self._clock() - started,
                solve_stats=_stats_subset(getattr(self._inner, "_last_solve_stats", None)),
            )
        return result

    def _write(
        self,
        kwargs: Dict[str, Any],
        description: Dict[str, Any],
        outcome: str,
        wall_s: Optional[float] = None,
        solve_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "schema": SCHEMA,
            "index": self.captured,
            **description,
            "kwargs": copy.deepcopy(kwargs),
            "outcome": outcome,
            "wall_s": wall_s,
            "solve_stats": solve_stats,
            "captured_at": time.time(),
        }
        self._out_dir.mkdir(parents=True, exist_ok=True)
        with (self._out_dir / f"spot_{self.captured:05d}.pkl").open("wb") as fh:
            pickle.dump(record, fh, protocol=4)
        self.captured += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def load_requests(dirs: Any) -> list:
    """Load every ``solve_request_v1`` record under the given directories."""
    records = []
    for directory in dirs:
        for path in sorted(Path(directory).rglob("spot_*.pkl")):
            with path.open("rb") as fh:
                record = pickle.load(fh)
            if record.get("schema") != SCHEMA:
                raise ValueError(f"{path}: not a {SCHEMA} record")
            records.append({**record, "path": str(path)})
    return records
