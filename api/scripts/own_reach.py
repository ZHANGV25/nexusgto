"""The hero's own range from the strategies it actually played (Pluribus sigma).

A live re-solve models the hero's range at the street root. By default that is
the blueprint's reach (preflop seeding, and with postflop conditioning the
blueprint's probabilities of the hero's earlier actions). After the hero has
played SEARCH on an earlier street, that stand-in is wrong: Pluribus instead
tracks every player's range from the strategy that player actually used,
including its own (SCOREBOARD_RESULTS section 39.4; EVERY_STREET_PLAN B1).

``OwnReachPolicy`` wraps the hero's acting policy in the match harness. At every
hero decision it multiplies a per-combo reach (1326 entries, uniform deal prior)
by the column of the action taken in the table the hero played from: the
solver's served table at a live re-solve (the BR-gap stash, the same rows the
hero sampled from), the blueprint lookup table otherwise (preflop, gated
streets, fallbacks). Before each postflop decision it hands the reach at the
street root to every live source as ``hero_range_override``; this street's own
actions are replayed in-tree by the solve, so they are not in the override.
Needs ``NEXUS_BR_GAP_EXPOSE=1`` for the stash.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from scripts.lbr_core import N_COMBOS
from scripts.lbr_policies import lookup_table, stash_table


def _is_preflop(state: Any) -> bool:
    return int(state.street) == 0


def _preflop_actions(state: Any) -> int:
    return len(getattr(state, "action_history", ()) or ())


def _action_index(result: Any) -> int:
    if isinstance(result, (int, np.integer)):
        return int(result)
    idx = getattr(result, "action_idx", None)
    if idx is None:
        raise TypeError("policy act() must return int or an object with action_idx")
    return int(idx)


class OwnReachStats:
    """Counters shared by every wrapped hero policy of one match."""

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {
            "decisions": 0, "stash_tables": 0, "lookup_tables": 0,
            "overrides_set": 0, "overrides_applied": 0, "overrides_unapplied": 0,
            "zero_mass_resets": 0,
        }
        self.stash_misses: Dict[str, int] = {}

    def bump(self, key: str, amount: int = 1) -> None:
        self.counts = {**self.counts, key: self.counts.get(key, 0) + amount}

    def miss(self, reason: str) -> None:
        self.stash_misses = {**self.stash_misses, reason: self.stash_misses.get(reason, 0) + 1}

    def as_dict(self) -> Dict[str, Any]:
        return {**self.counts, "stash_misses": dict(self.stash_misses)}


class OwnReachPolicy:
    """Hero policy wrapper that feeds the solve its own-play reach."""

    def __init__(self, inner: Any, lookup: Any, engine: Any, sources: Sequence[Any],
                 stats: Optional[OwnReachStats] = None) -> None:
        self._inner = inner
        self._lookup = lookup
        self._engine = engine
        self._sources = tuple(sources)
        self.own_reach_stats = stats if stats is not None else OwnReachStats()
        self._reset(None)

    def _reset(self, cards: Optional[Tuple[int, int]]) -> None:
        self._reach = np.ones(N_COMBOS, dtype=np.float64)
        self._street_start = self._reach
        self._street: Optional[int] = None
        self._cards = cards
        self._last_preflop_actions = -1
        self._seen_postflop = False

    def _new_hand(self, state: Any, cards: Tuple[int, int]) -> bool:
        if self._cards is None or tuple(cards) != tuple(self._cards):
            return True
        if _is_preflop(state):
            return self._seen_postflop or _preflop_actions(state) <= self._last_preflop_actions
        return False

    def _set_override(self) -> None:
        for source in self._sources:
            live = getattr(source, "_live", None)
            if live is not None:
                live.hero_range_override = np.array(self._street_start, dtype=np.float64)
                # Only the source that solves rewrites this; clear stale stats.
                live.last_hero_range_override = None
        self.own_reach_stats.bump("overrides_set")

    def _record_applied(self) -> None:
        for source in self._sources:
            live = getattr(source, "_live", None)
            stats = getattr(live, "last_hero_range_override", None)
            if isinstance(stats, dict):
                self.own_reach_stats.bump("overrides_applied" if stats.get("applied") else "overrides_unapplied")
                return

    def _played_table(self, state: Any, before: Any) -> np.ndarray:
        if not _is_preflop(state):
            objs = getattr(self._engine, "_last_br_gap_objs", None)
            if objs is not None and objs is not before:
                table, reason = stash_table(state, objs)
                if table is not None:
                    self.own_reach_stats.bump("stash_tables")
                    return table
                self.own_reach_stats.miss(reason)
        self.own_reach_stats.bump("lookup_tables")
        return lookup_table(self._lookup, state)

    def act(self, state: Any, hero_cards: Tuple[int, int]) -> Any:
        cards = (int(hero_cards[0]), int(hero_cards[1]))
        if self._new_hand(state, cards):
            self._reset(cards)
        if _is_preflop(state):
            self._last_preflop_actions = _preflop_actions(state)
        else:
            self._seen_postflop = True
            if self._street != int(state.street):
                self._street = int(state.street)
                self._street_start = self._reach
            self._set_override()
        before = getattr(self._engine, "_last_br_gap_objs", None)
        result = self._inner.act(state, hero_cards)
        action = _action_index(result)
        table = self._played_table(state, before)
        if not _is_preflop(state):
            self._record_applied()
        reach = self._reach * table[:, action]
        if float(reach.sum()) <= 0.0:
            # Cannot happen for a sampled action (the hero's own combo has positive
            # probability); keep the previous reach rather than zero the range.
            self.own_reach_stats.bump("zero_mass_resets")
        else:
            self._reach = reach
        self.own_reach_stats.bump("decisions")
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
