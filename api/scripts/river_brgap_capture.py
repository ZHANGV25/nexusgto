"""Capture heads-up river live solves with the blueprint strategy on the same tree.

For each production river solve where exactly two players remain and the hero
acts first on the street, the tree snapshot and served strategy stashed by
``CpuEsSolver`` (``NEXUS_BR_GAP_EXPOSE=1``) are saved with the blueprint's
strategy for every weighted hand row at every decision node of that tree. The
blueprint rows come from ``LookupSource.get_mix`` on the real ``GameState``
advanced along each node's path, so both profiles share one tree, one pair of
ranges and one action order ([fold][check/call][raises ascending]).
``scripts/river_brgap_score.py`` grades both with an exact river best response.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA = "river_brgap_v1"
RIVER = 3
T_DECISION = 0


class ProjectionMismatch(ValueError):
    """The solved tree and the advanced game state disagree at a node."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


def is_eligible(state: Any) -> bool:
    """A two-player river street root with a player to act."""
    return (
        int(getattr(state, "street", -1)) == RIVER
        and int(getattr(state, "n_live", 0)) == 2
        and tuple(getattr(state, "street_history", ()) or ()) == ()
        and getattr(state, "to_act_slot", None) is not None
    )


def entry_slots(state: Any) -> List[int]:
    """Compact slots in solver player order (live_resolve_adapter convention)."""
    return [s for s in range(state.n_slots) if state.active[s] or state.acted[s]]


def plain_tree(tree: Any) -> Dict[str, Any]:
    """A module-free copy of a PlainTree snapshot (unpickles without poker-solver)."""
    return {
        "num_players": int(tree.num_players),
        "board": tuple(int(card) for card in tree.board),
        "starting_pot": int(tree.starting_pot),
        "player_hands": [
            [(int(c0), int(c1), float(weight)) for c0, c1, weight in hands]
            for hands in tree.player_hands
        ],
        "nodes": [
            {
                "type": int(node.type),
                "player": int(node.player),
                "children": tuple(int(child) for child in node.children),
                "pot": int(node.pot),
                "bets": tuple(int(bet) for bet in node.bets),
                "active": tuple(bool(flag) for flag in node.active),
            }
            for node in tree.nodes
        ],
    }


def _row_mix(
    lookup: Any, state: Any, cards: Tuple[int, int], n_legal: int
) -> Tuple[np.ndarray, bool]:
    mix = lookup.get_mix(state, cards)
    probs = np.zeros(n_legal, dtype=np.float64)
    for idx, freq in mix.freqs.items():
        probs[int(idx)] += float(freq)
    return probs, bool(getattr(mix, "is_uniform_fallback", False))


def _checked_decision(node_idx: int, node: Dict[str, Any], state: Any, slots: List[int]) -> int:
    if state.terminal:
        raise ProjectionMismatch("terminal", f"state is terminal at decision node {node_idx}")
    if state.to_act_slot != slots[node["player"]]:
        raise ProjectionMismatch(
            "to_act",
            f"node {node_idx} solver player {node['player']} vs slot {state.to_act_slot}",
        )
    n_legal = int(state.num_legal_actions())
    if n_legal != len(node["children"]):
        raise ProjectionMismatch(
            "n_legal",
            f"node {node_idx} has {len(node['children'])} children, state has {n_legal}",
        )
    return n_legal


def project_blueprint_sigma(
    tree: Dict[str, Any], root_state: Any, lookup: Any
) -> Tuple[Dict[int, np.ndarray], Dict[str, int]]:
    """Blueprint strategy per decision node and hand row of ``tree``.

    Zero-weight rows stay uniform; the range-prior best response never reads them.
    """
    nodes = tree["nodes"]
    slots = entry_slots(root_state)
    if len(slots) != tree["num_players"]:
        raise ProjectionMismatch(
            "players", f"{len(slots)} entry slots for {tree['num_players']} solver players"
        )
    sigma: Dict[int, np.ndarray] = {}
    stats = {"nodes": 0, "rows": 0, "misses": 0}
    pending = [(0, root_state)]
    while pending:
        node_idx, state = pending.pop()
        node = nodes[node_idx]
        if node["type"] != T_DECISION:
            if not state.terminal:
                raise ProjectionMismatch(
                    "terminal", f"tree node {node_idx} is terminal but the state is not"
                )
            continue
        n_legal = _checked_decision(node_idx, node, state, slots)
        hands = tree["player_hands"][node["player"]]
        rows = np.full((len(hands), n_legal), 1.0 / n_legal)
        for row, (c0, c1, weight) in enumerate(hands):
            if weight <= 0.0:
                continue
            rows[row], missed = _row_mix(lookup, state, (c0, c1), n_legal)
            stats = {**stats, "rows": stats["rows"] + 1, "misses": stats["misses"] + int(missed)}
        sigma = {**sigma, node_idx: rows}
        stats = {**stats, "nodes": stats["nodes"] + 1}
        pending.extend(
            (child, state.advance(action)) for action, child in enumerate(node["children"])
        )
    return sigma, stats


class RiverCaptureSource:
    """Strategy-source wrapper recording eligible heads-up river solves (research only)."""

    def __init__(
        self, inner: Any, lookup: Any, engine: Any, out_dir: str, limit: int
    ) -> None:
        self._inner = inner
        self._lookup = lookup
        self._engine = engine
        self._out_dir = Path(out_dir)
        self._limit = int(limit)
        self.captured = 0
        self.skipped: Dict[str, int] = {}

    def get_mix(self, state: Any, hero_cards: Any) -> Any:
        before = getattr(self._engine, "_last_br_gap_objs", None)
        mix = self._inner.get_mix(state, hero_cards)
        if self.captured < self._limit and is_eligible(state):
            reason = self._capture(state, hero_cards, before)
            if reason is not None:
                self.skipped = {**self.skipped, reason: self.skipped.get(reason, 0) + 1}
        return mix

    def _capture(self, state: Any, hero_cards: Any, before: Any) -> Optional[str]:
        objs = getattr(self._engine, "_last_br_gap_objs", None)
        if objs is None or objs is before:
            return "no_fresh_solve"
        if objs.get("street") != "river" or int(objs.get("n_players", 0)) != 2:
            return "not_hu_river_solve"
        if not objs.get("sigma"):
            return "empty_sigma"
        tree = plain_tree(objs["tree"])
        if tree["board"] != tuple(int(card) for card in state.board[:5]):
            return "board_mismatch"
        try:
            blueprint_sigma, stats = project_blueprint_sigma(tree, state, self._lookup)
        except ProjectionMismatch as exc:
            return f"projection_{exc.kind}"
        except ValueError:
            return "projection_advance"
        self._write(state, hero_cards, objs, tree, blueprint_sigma, stats)
        return None

    def _write(
        self,
        state: Any,
        hero_cards: Any,
        objs: Dict[str, Any],
        tree: Dict[str, Any],
        blueprint_sigma: Dict[int, np.ndarray],
        stats: Dict[str, int],
    ) -> None:
        solve_stats = getattr(self._engine, "_last_solve_stats", None)
        record = {
            "schema": SCHEMA,
            "tree": tree,
            "search_sigma": {
                int(node): np.asarray(rows, dtype=np.float64)
                for node, rows in objs["sigma"].items()
            },
            "blueprint_sigma": blueprint_sigma,
            "projection_stats": stats,
            "hero_player": entry_slots(state).index(state.to_act_slot),
            "hero_cards": tuple(int(card) for card in hero_cards),
            "state_pot": int(state.pot),
            "pot_chips": objs.get("pot_chips"),
            "seat_of_slot": tuple(state.seat_of_slot),
            "preflop_prefix": tuple(state.preflop_prefix),
            "postflop_history": tuple(state.postflop_history),
            "true_hu": bool(getattr(state, "true_hu", False)),
            "iterations_run": (
                solve_stats.get("iterations_run") if isinstance(solve_stats, dict) else None
            ),
        }
        self._out_dir.mkdir(parents=True, exist_ok=True)
        with (self._out_dir / f"spot_{self.captured:05d}.pkl").open("wb") as fh:
            pickle.dump(record, fh, protocol=4)
        self.captured += 1
