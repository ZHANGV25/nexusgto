"""Policies under local best response: the blueprint and the production search.

Both expose ``decide(state, cards) -> (action, table)`` for the policy's own
turn and ``table(state, cards)`` for a hypothetical node, where a table is a
``[1326, n_actions]`` array of the policy's action distribution per combo.
The search policy reads its table from the solver's BR-gap stash
(``NEXUS_BR_GAP_EXPOSE=1``), so the same re-solve that serves the actual
hand also supplies the strategy for every other hand at that node.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.lbr_core import COMBOS, N_COMBOS, live_mask
from scripts.lbr_game import STREET_BOARD_LEN, is_preflop

T_DECISION = 0


def _visible(state: Any) -> List[int]:
    board = list(getattr(state, "board", ()) or ())
    return [int(card) for card in board[: STREET_BOARD_LEN[int(state.street)]]]


def lookup_table(lookup: Any, state: Any) -> np.ndarray:
    """The blueprint's action distribution for every combo live on the board."""
    n_legal = int(state.num_legal_actions())
    table = np.zeros((N_COMBOS, n_legal), dtype=np.float64)
    for index in np.nonzero(live_mask(_visible(state)))[0]:
        mix = lookup.get_mix(state, COMBOS[int(index)])
        for action, freq in mix.freqs.items():
            table[index, int(action)] = float(freq)
    return table


class BlueprintPolicy:
    name = "blueprint"

    def __init__(self, lookup: Any, rng: np.random.Generator) -> None:
        self._lookup = lookup
        self._rng = rng
        self.stats: Dict[str, Any] = {"tables": 0}

    def table(self, state: Any, cards: Optional[Tuple[int, int]] = None) -> np.ndarray:
        self.stats = {**self.stats, "tables": self.stats["tables"] + 1}
        return lookup_table(self._lookup, state)

    def decide(self, state: Any, cards: Tuple[int, int]) -> Tuple[int, np.ndarray]:
        table = self.table(state, cards)
        mix = self._lookup.get_mix(state, cards)
        return int(mix.sample(self._rng)), table


def _entry_hero_index(state: Any) -> int:
    slots = [s for s in range(state.n_slots) if state.active[s] or state.acted[s]]
    return slots.index(state.to_act_slot)


def stash_table(state: Any, objs: Any) -> Tuple[Optional[np.ndarray], str]:
    """The solved strategy at the state's node for every live combo, or a reason."""
    from app.services.suit_isomorphism import canonical_combo
    from scripts.river_brgap_capture import plain_tree

    tree = plain_tree(objs["tree"])
    hero = _entry_hero_index(state)
    node = 0
    for action in state.street_history:
        children = tree["nodes"][node]["children"]
        if int(action) >= len(children):
            return None, "history_off_tree"
        node = children[int(action)]
    entry = tree["nodes"][node]
    if entry["type"] != T_DECISION or entry["player"] != hero:
        return None, "node_mismatch"
    sigma = objs["sigma"].get(node)
    n_legal = int(state.num_legal_actions())
    if sigma is None:
        return None, "no_sigma"
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape[1] != n_legal:
        return None, "n_actions"
    board = list(tree["board"])
    hands = tree["player_hands"][hero]
    if sigma.shape[0] != len(hands):
        # The solver's readout occasionally drops a row (343 sigma rows for
        # 344 hands); the node then falls back to the lookup table.
        return None, "row_mismatch"
    rows = {(int(c0), int(c1)): i for i, (c0, c1, _w) in enumerate(hands)}
    table = np.zeros((N_COMBOS, n_legal), dtype=np.float64)
    missing = 0
    for index in np.nonzero(live_mask(board))[0]:
        c0, c1 = COMBOS[int(index)]
        row = rows.get(canonical_combo(c0, c1, board))
        if row is None:
            missing += 1
            continue
        table[index] = sigma[row]
    if missing:
        return None, f"rows_missing_{missing}"
    return table, "stash"


class SearchPolicy:
    name = "production"

    def __init__(self, strategy: Any, engine: Any, lookup: Any, rng: np.random.Generator,
                 sources: Sequence[Any] = (), hero_range: str = "blueprint") -> None:
        if hero_range not in ("blueprint", "own"):
            raise ValueError(f"hero_range must be 'blueprint' or 'own', got {hero_range!r}")
        self._strategy = strategy
        self._engine = engine
        self._lookup = lookup
        self._rng = rng
        self._sources = tuple(sources)
        self._hero_range = hero_range
        self.stats: Dict[str, Any] = {
            "solves": 0, "solve_wall_s": 0.0, "tables": 0, "table_sources": {},
            "served_row_max_diff": 0.0, "lookup_tables": 0, "hero_range": hero_range,
            "own_range_applied": 0, "own_range_skipped": 0,
        }

    def set_own_range(self, combo_weights: np.ndarray) -> None:
        """Hand the driver's tracked reach to every live source (own-reach mode)."""
        if self._hero_range != "own":
            return
        for source in self._sources:
            live = getattr(source, "_live", None)
            if live is not None:
                live.hero_range_override = np.asarray(combo_weights, dtype=np.float64)

    def _record_own_range(self) -> None:
        for source in self._sources:
            live = getattr(source, "_live", None)
            stats = getattr(live, "last_hero_range_override", None)
            if isinstance(stats, dict):
                key = "own_range_applied" if stats.get("applied") else "own_range_skipped"
                self._bump(key)

    def _bump(self, key: str, amount: float = 1) -> None:
        self.stats = {**self.stats, key: self.stats[key] + amount}

    def _count_source(self, reason: str) -> None:
        sources = dict(self.stats["table_sources"])
        sources[reason] = sources.get(reason, 0) + 1
        self.stats = {**self.stats, "table_sources": sources}

    def _solve(self, state: Any, cards: Tuple[int, int]) -> Tuple[Any, np.ndarray]:
        before = getattr(self._engine, "_last_br_gap_objs", None)
        started = time.perf_counter()
        mix = self._strategy.get_mix(state, cards)
        self._bump("solve_wall_s", time.perf_counter() - started)
        self._bump("solves")
        self._record_own_range()
        objs = getattr(self._engine, "_last_br_gap_objs", None)
        table, reason = (None, "no_fresh_solve") if objs is None or objs is before else stash_table(state, objs)
        if table is None:
            table = lookup_table(self._lookup, state)
            self._bump("lookup_tables")
        else:
            from scripts.lbr_core import combo_index

            row = table[combo_index(*cards)]
            diff = max(abs(float(mix.freqs.get(i, 0.0)) - float(row[i])) for i in range(row.shape[0]))
            self.stats = {**self.stats, "served_row_max_diff": max(self.stats["served_row_max_diff"], diff)}
        self._count_source(reason)
        return mix, table

    def table(self, state: Any, cards: Tuple[int, int]) -> np.ndarray:
        self._bump("tables")
        if is_preflop(state):
            return lookup_table(self._lookup, state)
        return self._solve(state, cards)[1]

    def decide(self, state: Any, cards: Tuple[int, int]) -> Tuple[int, np.ndarray]:
        if is_preflop(state):
            table = lookup_table(self._lookup, state)
            mix = self._lookup.get_mix(state, cards)
        else:
            mix, table = self._solve(state, cards)
        return int(mix.sample(self._rng)), table

    def fallbacks(self) -> Dict[str, int]:
        merged: Dict[str, int] = {}
        for source in self._sources:
            for name, count in getattr(source, "fallbacks", {}).items():
                merged[name] = merged.get(name, 0) + int(count)
        return merged


def build_policy(
    name: str, blueprint_path: str, rng: np.random.Generator, *,
    live_threads: int = 8, env_overrides: Sequence[Tuple[str, str]] = (),
    hero_range: str = "blueprint",
) -> Any:
    from app.engine.agent.modes import ModeConfig
    from app.engine.agent.strategy_source import LookupSource

    lookup = LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path, uniform_on_miss=True))
    if name == "blueprint":
        return BlueprintPolicy(lookup, rng)
    if name != "production":
        raise ValueError(f"unknown policy {name!r}")
    from app.engine.cpu_es_solver import CpuEsSolver
    from app.engine.engine_constants import live_config_for
    from scripts.run_match_cli import (
        _production_strategy, apply_env_overrides, production_env_problems, scoped_leaf_mode,
    )

    problems = production_env_problems(os.environ)
    if problems:
        raise SystemExit("production policy needs the serve environment: " + "; ".join(problems))
    apply_env_overrides(tuple(env_overrides), os.environ)
    os.environ["NEXUS_BR_GAP_EXPOSE"] = "1"
    with scoped_leaf_mode("production"):
        engine = CpuEsSolver(blueprint_path)
    strategy, sources = _production_strategy(
        path=blueprint_path, lookup=lookup, engine=engine, live_config_for_fn=live_config_for,
        num_threads=int(live_threads),
    )
    return SearchPolicy(strategy, engine, lookup, rng, sources, hero_range=hero_range)
