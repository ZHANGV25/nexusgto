"""Search mix vs blueprint mix at the same decisions of a production source.

At every live decision where the search chose the action (the "served" mix),
the blueprint lookup is queried for the same state and hand. Aggregates by
street, live player count and whether the hero faces a bet show what search
does differently:
fold/call/raise mass for each source, mean L1 distance and how often the top
action differs. Capped per-decision rows (cards, board, pot, both mixes) allow
offline breakdowns such as hand strength. Action groups use the shared order
[fold (only when facing a bet)][check/call][raises ascending].
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

STREETS = {1: "flop", 2: "turn", 3: "river"}
MAX_ROWS = 50_000
_SUM_FIELDS = (
    "served_fold", "served_call", "served_raise",
    "blueprint_fold", "blueprint_call", "blueprint_raise",
    "l1", "top_differs", "pot",
)


def mix_vector(freqs: Mapping[int, float], n_legal: int) -> np.ndarray:
    probs = np.zeros(n_legal, dtype=np.float64)
    for idx, freq in freqs.items():
        probs[int(idx)] += float(freq)
    total = probs.sum()
    return probs / total if total > 0.0 else probs


def group_masses(probs: np.ndarray, facing_bet: bool) -> Tuple[float, float, float]:
    """(fold, check/call, raise) mass of one decision's action distribution."""
    n_nonraise = 2 if facing_bet else 1
    fold = float(probs[0]) if facing_bet and len(probs) else 0.0
    call = float(probs[n_nonraise - 1]) if len(probs) >= n_nonraise else 0.0
    return fold, call, float(probs[n_nonraise:].sum())


def cell_key(state: Any) -> str:
    street = STREETS.get(int(getattr(state, "street", -1)), "other")
    facing = "facing" if bool(getattr(state, "facing_bet", False)) else "unopened"
    return f"{street}|{int(getattr(state, 'n_live', 0))}p|{facing}"


def _tuple(state: Any, name: str) -> List[int]:
    return [int(x) for x in (getattr(state, name, None) or ())]


class MixComparison:
    """Accumulate served-vs-blueprint action frequencies across decisions."""

    def __init__(self, max_rows: int = MAX_ROWS) -> None:
        self._max_rows = int(max_rows)
        self.cells: Dict[str, Dict[str, float]] = {}
        self.rows: List[Dict[str, Any]] = []

    def add(self, state: Any, hero_cards: Any, served: Any, blueprint: Any) -> None:
        n_legal = int(state.num_legal_actions())
        facing = bool(state.facing_bet)
        served_probs = mix_vector(served.freqs, n_legal)
        blueprint_probs = mix_vector(blueprint.freqs, n_legal)
        s_fold, s_call, s_raise = group_masses(served_probs, facing)
        b_fold, b_call, b_raise = group_masses(blueprint_probs, facing)
        values = {
            "served_fold": s_fold, "served_call": s_call, "served_raise": s_raise,
            "blueprint_fold": b_fold, "blueprint_call": b_call, "blueprint_raise": b_raise,
            "l1": float(np.abs(served_probs - blueprint_probs).sum()),
            "top_differs": float(int(served_probs.argmax()) != int(blueprint_probs.argmax())),
            "pot": float(getattr(state, "pot", 0)),
        }
        key = cell_key(state)
        cell = self.cells.get(key, {"count": 0.0, **{name: 0.0 for name in _SUM_FIELDS}})
        updated = {name: cell[name] + values[name] for name in _SUM_FIELDS}
        self.cells = {**self.cells, key: {"count": cell["count"] + 1.0, **updated}}
        if len(self.rows) < self._max_rows:
            self.rows.append({
                "cell": key,
                "street": int(getattr(state, "street", -1)),
                "n_live": int(getattr(state, "n_live", 0)),
                "facing_bet": facing,
                "pot": int(getattr(state, "pot", 0)),
                "hero_cards": [int(c) for c in hero_cards],
                "board": _tuple(state, "board"),
                "postflop_history": _tuple(state, "postflop_history"),
                "served": [round(float(p), 6) for p in served_probs],
                "blueprint": [round(float(p), 6) for p in blueprint_probs],
            })

    def summary(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for key in sorted(self.cells):
            cell = self.cells[key]
            count = cell["count"]
            result[key] = {"count": int(count), **{
                f"mean_{name}": round(cell[name] / count, 6) for name in _SUM_FIELDS
            }}
        return result

    def write_rows(self, path: Path) -> None:
        with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
            for row in self.rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
