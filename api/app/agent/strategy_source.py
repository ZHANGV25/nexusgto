"""Strategy sources — the per-mode seam (plan §2, §3.4/§3.6).

A StrategySource maps a GameState (+ hero hand) to an action MIX
({action_idx: freq}). The three modes differ ONLY here:

  LookupSource        — blueprint σ̄ lookup, sub-second, NO GPU. v1 PRIMARY.
                        = Pluribus's blueprint FLOOR.
  OfflineLiftedSource — serves a precomputed lifted-strategy cache (the
                        genuinely-new bridge). v1 = source + cache-format stub.
                        Analysis only; flagged not-real-time.

All three feed the SAME sampling loop in agent.py.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from app.agent.game_state import GameState
from app.agent.key_builder import build_key
from app.agent.modes import ModeConfig

# Hero cards = (c0, c1) ints 0-51.
HeroCards = Tuple[int, int]


@dataclass(frozen=True)
class ActionMix:
    """A normalized distribution over legal action indices.

    freqs: {action_idx: probability}; normalized to sum 1.0 on construction.
    is_uniform_fallback: True when this is the Pluribus Algorithm-1
        uniform-on-miss substitution (the blueprint node was absent), so the
        caller can flag honest "no blueprint data here" play.
    """

    freqs: Dict[int, float]
    is_uniform_fallback: bool = False

    def __post_init__(self) -> None:
        if not self.freqs:
            raise ValueError("ActionMix requires at least one action")
        total = float(sum(self.freqs.values()))
        if total <= 0.0:
            raise ValueError("ActionMix frequencies sum to <= 0")
        if abs(total - 1.0) > 1e-9:
            object.__setattr__(
                self, "freqs", {k: v / total for k, v in self.freqs.items()}
            )

    @classmethod
    def uniform(cls, n_actions: int, is_fallback: bool = False) -> "ActionMix":
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        return cls(
            {i: 1.0 / n_actions for i in range(n_actions)},
            is_uniform_fallback=is_fallback,
        )

    def sample(self, rng: np.random.Generator) -> int:
        """Sample one action index from the mix using a seeded rng."""
        idxs = list(self.freqs.keys())
        probs = np.array([self.freqs[i] for i in idxs], dtype=np.float64)
        probs = probs / probs.sum()
        return int(rng.choice(idxs, p=probs))


class StrategySource(abc.ABC):
    """Abstract per-decision strategy source."""

    @abc.abstractmethod
    def get_mix(self, state: GameState, hero_cards: HeroCards) -> ActionMix:
        """Return the action mix for the player to act in `state`."""
        raise NotImplementedError


# ── LOOKUP (floor, sub-second, NO GPU) ───────────────────────────────────


class LookupSource(StrategySource):
    """Blueprint σ̄ lookup + sample. v1 PRIMARY mode.

    Loads a BPSReaderC once and reads the stored average strategy at the node
    key_builder constructs. NEVER constructs or imports the GPU solver — this
    is the sub-second blueprint floor, not the live-search ceiling.
    """

    def __init__(self, config: ModeConfig, reader=None) -> None:
        self._config = config
        if reader is not None:
            self._reader = reader
        else:
            if not config.blueprint_path:
                raise ValueError(
                    "LookupSource needs a blueprint_path (or an injected reader)"
                )
            try:
                from bps_reader_py import BPSReaderC  # type: ignore[import-untyped]
            except ImportError:  # pragma: no cover
                from python.bps_reader_py import BPSReaderC  # type: ignore[no-redef]
            self._reader = BPSReaderC(config.blueprint_path)

        # Read the trained vocab from the blueprint metadata (the single
        # source of truth for the action COUNT). Used to fail-loud if a caller
        # built a GameState against a different blueprint's vocab.
        meta = self._reader.get_metadata() or {}
        self._meta_postflop = tuple(meta.get("postflop_bet_sizes", ()) or ())
        self._meta_subsequent = tuple(meta.get("subsequent_bet_sizes", ()) or ())
        self._meta_add_allin = bool(int(meta.get("add_explicit_allin", 0) or 0))

    def _bucket(self, state: GameState, hero_cards: HeroCards) -> int:
        c0, c1 = hero_cards
        board = list(state.board)
        if state.street == 1:
            return self._reader.hand_to_bucket(board[:3], c0, c1)
        if state.street == 2:
            return self._reader.hand_to_bucket_turn(board[:4], c0, c1)
        return self._reader.hand_to_bucket_river(board[:5], c0, c1)

    def get_mix(self, state: GameState, hero_cards: HeroCards) -> ActionMix:
        self._assert_vocab_matches(state)
        key = build_key(state)
        n_legal = state.num_legal_actions()
        bucket = self._bucket(state, hero_cards)

        strat = None
        if bucket >= 0:
            strat = self._reader.get_strategy(
                list(state.board), key.action_history, key.player, bucket,
                street=key.street,
            )

        if strat is None:
            return self._on_miss(n_legal)

        arr = np.asarray(strat, dtype=np.float64)
        if arr.sum() <= 0.0:
            return self._on_miss(n_legal)

        # Exact-length invariant: the GameState's vocab must agree with the
        # blueprint's stored σ̄ length. A mismatch is a vocab bug (wrong
        # blueprint / wrong action-size config), NOT something to silently
        # truncate or pad — surface it. v1's validated lines never reach this.
        if len(arr) != n_legal:
            raise ValueError(
                f"action-count mismatch at {key.as_tuple()}: blueprint stores "
                f"{len(arr)} actions but GameState vocab says {n_legal}. The "
                f"GameState's postflop_bet_sizes/subsequent_bet_sizes/"
                f"add_explicit_allin must match the loaded blueprint metadata."
            )

        arr = arr / arr.sum()
        return ActionMix({i: float(arr[i]) for i in range(n_legal)})

    def _assert_vocab_matches(self, state: GameState) -> None:
        """Fail loud if the GameState's action vocab disagrees with the loaded
        blueprint's metadata — that means the state was built for a different
        blueprint and its legal-action counts would be wrong.

        Only enforced when the blueprint actually carries the metadata fields
        (some test readers don't); a missing field is skipped, not failed.
        """
        if self._meta_postflop and (
            tuple(state.postflop_bet_sizes) != self._meta_postflop
            or tuple(state.subsequent_bet_sizes) != self._meta_subsequent
            or bool(state.add_explicit_allin) != self._meta_add_allin
        ):
            raise ValueError(
                "GameState vocab does not match the loaded blueprint: state "
                f"postflop={state.postflop_bet_sizes} subsequent="
                f"{state.subsequent_bet_sizes} allin={state.add_explicit_allin}"
                f" vs blueprint postflop={self._meta_postflop} subsequent="
                f"{self._meta_subsequent} allin={self._meta_add_allin}."
            )

    def _on_miss(self, n_legal: int) -> ActionMix:
        if self._config.uniform_on_miss:
            # Pluribus Algorithm 1: substitute uniform-over-legal-actions.
            return ActionMix.uniform(n_legal, is_fallback=True)
        raise LookupError(
            "blueprint node absent (MISS) and uniform_on_miss=False"
        )


# ── OFFLINE_LIFTED (bridge, sub-second) — v1 scaffold + cache format ─────


@dataclass(frozen=True)
class LiftedCacheEntry:
    """One precomputed lifted-strategy cache entry (the v1 cache FORMAT).

    The offline pipeline (out of scope for v1 — plan §6) runs LIVE_RESOLVE
    over the cacheable spot surface and stores results keyed by the node key.
    OfflineLiftedSource looks these up as an instant serve.

    key: (tuple(action_history), player, street, bucket) — the lifted node id.
    freqs: {action_idx: freq}.
    """

    action_history: Tuple[int, ...]
    player: int
    street: int
    bucket: int
    freqs: Dict[int, float]


class OfflineLiftedSource(StrategySource):
    """Serve a precomputed lifted-strategy cache (v1 = source + format stub).

    v1 scaffolds the source interface and the LiftedCacheEntry format only;
    the precompute farm that POPULATES the cache is deferred (plan §6). With
    no cache loaded, get_mix raises NotImplementedError (fail-loud, not silent
    garbage).
    """

    def __init__(self, config: ModeConfig, cache=None) -> None:
        self._config = config
        # cache: optional {(ah, player, street, bucket): LiftedCacheEntry}.
        self._cache: Optional[Dict] = cache

    def get_mix(self, state: GameState, hero_cards: HeroCards) -> ActionMix:
        if not self._cache:
            raise NotImplementedError(
                "OFFLINE_LIFTED cache is empty — the offline precompute farm "
                "is out of scope for v1 (plan §6). Load a cache or use LOOKUP."
            )
        key = build_key(state)
        # Bucket would come from the same bucketizer LookupSource uses; the
        # cache lookup shape is fixed here so the precompute farm can target it.
        entry_key = (tuple(key.action_history), key.player, key.street)
        for (ah, player, street, _bucket), entry in self._cache.items():
            if (ah, player, street) == entry_key:
                return ActionMix(dict(entry.freqs))
        raise NotImplementedError(
            f"no lifted cache entry for {entry_key} (sparse v1 cache)"
        )




class LiveResolveSource(StrategySource):

    Build = WIRE + VERIFY, not write-from-scratch (plan §3.6). This is the
    1-33s live-search ceiling; it is NOT sub-second and must be flagged
    not-real-time in any UI. v1 provides the adapter + a smoke path; full

    single get_mix into a get_strategy call. Range narrowing across a full
    hand is driven by the caller via on_villain_action/on_hero_action on the
    underlying solver (exposed as `.solver`).

    VERIFICATION STATUS (v1): construction + API-surface are smoke-tested
    new_hand -> get_strategy SOLVE path is GPU-gated and remains UNVERIFIED in
    loop is deferred (see get_mix below). LOOKUP is the verified v1 mode.
    """

        self._config = config

    def _ensure_solver(self):
        if self.solver is not None:
            return self.solver
        try:
        except ImportError:  # pragma: no cover
        # its own BlueprintV2. Pass None to run solver-only (caller wires the
        # range state). Full construction is the caller's responsibility.
        return self.solver

    def get_mix(self, state: GameState, hero_cards: HeroCards) -> ActionMix:
        # LIVE_RESOLVE is non-sub-second and stateful; v1 wires the adapter
        # and verifies construction. A full single-call get_mix needs the
        raise NotImplementedError(
            "LiveResolveSource is the non-real-time ceiling: drive the wrapped "
            "on_hero_action / get_strategy. A single stateless get_mix is not "
        )
