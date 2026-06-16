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
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from app.agent.game_state import GameState
from app.agent.key_builder import build_key
from app.agent.live_resolve_adapter import LiveResolveUnsupported
from app.agent.modes import ModeConfig

# Re-export so callers can `from app.agent.strategy_source import
# LiveResolveUnsupported` alongside LiveResolveSource (the public seam).
__all__ = [
    "ActionMix",
    "StrategySource",
    "LookupSource",
    "OfflineLiftedSource",
    "LiveResolveSource",
    "HybridSource",
    "LiveResolveUnsupported",
]

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

        # Length reconciliation. The blueprint stored the node's action set
        # under CANONICAL trainer chips; the state's n_legal uses REAL chips
        # (advance_observed applies real amounts). A real bet that
        # commitment-prunes the vocab drops RAISE slots only — and the
        # trainer orders actions [fold][check/call][raises asc], so the
        # stored σ̄ is PREFIX-aligned with the pruned legal set: project by
        # truncating the tail and renormalizing (review 2026-06-10; without
        # this, a covering bet/all-in on the hero crashed the hand). A
        # SHORTER stored σ̄ remains a vocab BUG — fail loud.
        if len(arr) > n_legal:
            n_nonraise = (1 if state.facing_bet else 0) + 1
            if n_legal < n_nonraise:
                raise ValueError(
                    f"action-count mismatch at {key.as_tuple()}: blueprint "
                    f"stores {len(arr)} actions but n_legal={n_legal} < "
                    f"non-raise slots {n_nonraise} — not a raise-tail prune."
                )
            arr = arr[:n_legal]
            if arr.sum() <= 0.0:
                # All stored mass was on the pruned raise slots.
                return self._on_miss(n_legal)
        elif len(arr) < n_legal:
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
    """Live per-decision CPU re-solve behind the StrategySource ABC (plan §3.6).

    This is the NON-real-time ceiling — a real-time CFR re-solve, seconds per
    decision (2p-flop ~3s, 3p-turn ~18s at a few hundred iters), NOT sub-second.
    It must be flagged not-real-time in any UI. It is the ceiling MODE, not the
    blueprint FLOOR (that is LookupSource).

    ENGINE (empirically validated GPU-free this session — see
    docs/BENCHMARK_RESULTS.md §3c and app/agent/live_resolve_adapter.py): wraps
    poker-solver's CpuEsSolver (CPU external-sampling ES) for flop, turn AND
    river (2026-06-11: river showdown terminals settle natively in the
    named engine — has no GPU-free multi-street solve path (its solver_v2 CPU
    fallback crashes building flop/turn trees), so LIVE_RESOLVE wraps CpuEsSolver
    instead; that is the honest WIRE+VERIFY of the realtime pipeline GPU-free.

    STATELESS per-call: each get_mix is an independent re-solve with ranges
    rebuilt from the board (no cross-street narrowing — a documented v1
    simplification). The 12GB blueprint load is one-time in the held engine.
    """

    def __init__(
        self, config: ModeConfig, cpu_es_solver=None, iterations: int = 200,
        seed_ranges: bool = True, strategy_reader=None,
        vec_mode: bool = False, readout: str = "avg", num_threads: int = 0,
        num_rollouts: int = 3, cache_k: int = 1,
    ) -> None:
        self._config = config
        self._engine = cpu_es_solver  # held across calls (12GB load is one-time)
        self._iterations = int(iterations)
        self._num_threads = int(num_threads)  # 0 = all cores
        # cache_k = bias-profile samples per leaf (each leaf value averages over
        # cache_k random continuation-strategy draws). cache_k=1 is one random
        # joint-bias draw -> high variance AND a directional bias when that single
        # draw favors aggression (EC2 2026-06-15: the 2p flop QJs leaf at cache_k=1
        # converged stably to bet-heavy vs the blueprint's check 0.957). Averaging
        # over more draws approximates the exact 4^np-profile mean. Small trees
        # (2p flop, 16 joint profiles) afford a high cache_k cheaply.
        self._cache_k = int(cache_k)
        # Leaf-table rollouts per (leaf, bias). The 6p floor is leaf-sample-
        # noise-dominated: nr=3 floors ~0.15%/pot and the per-spot strategy is
        # noise-realization-driven (mean |dSigma| 0.40 across noise seeds);
        # nr=10 floors ~0.10% (measured_logs/DISCRIMINATOR_2026-06-10.md).
        # 6p playing-agent config should use >=10; default 3 preserves the
        # validated 2p/3p analysis surface.
        self._num_rollouts = int(num_rollouts)
        # vec_mode + readout="final" = the PLAYING-AGENT configuration
        # (vec-Linear-FINAL — the Gate-1-validated, EC2-measured config that
        # makes a 6p re-solve viable in tens of seconds at ~0.15%/pot:
        # measured_logs/VEC_SPEEDUP_2026-06-10.md). The default (scalar, σ̄)
        # is the historically validated 2p/3p analysis surface. CpuEsSolver
        # rejects vec_mode without readout="final" (vec-Linear-AVERAGE is
        # seed-unstable at 3+ players).
        self._vec_mode = bool(vec_mode)
        self._readout = str(readout)
        # In-distribution range seeding (advisor fix): seed the re-solve with the
        # blueprint's preflop-σ̄ ranges so it doesn't over-bet vs a wide/weak
        # uniform villain. Needs a blueprint strategy reader; when absent we fall
        # back to uniform ranges (the over-bet caveat then applies — documented).
        self._seed_ranges = bool(seed_ranges)
        self._reader = strategy_reader  # reader.get_strategy(...) or None
        # RECORDED per call (call-local, like RecordingHybridSource): how the
        # last get_mix seeded its ranges — "blueprint_preflop_sigma" or
        # "uniform". The engine provider reads this to DISCLOSE uniform
        # seeding in the response quality note (review 2026-06-11: silent
        # uniform seeding on unparsed preflop shapes was an unflagged
        # quality compromise).
        self.last_range_seeding: Optional[str] = None

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        if not self._config.blueprint_path:
            raise ValueError(
                "LiveResolveSource needs a blueprint_path (or an injected "
                "cpu_es_solver) to construct the CpuEsSolver engine."
            )
        from app.services.cpu_es_solver import CpuEsSolver
        self._engine = CpuEsSolver(self._config.blueprint_path)
        return self._engine

    def _get_strategy_reader(self, engine):
        """A reader.get_strategy callable for preflop-σ̄ range seeding.

        Priority: an injected strategy_reader, else the held engine's own
        blueprint (CpuEsSolver._blueprint is a BPSReaderC with get_strategy).
        Returns None if neither exposes get_strategy (→ uniform fallback).
        """
        if self._reader is not None and hasattr(self._reader, "get_strategy"):
            return self._reader.get_strategy
        bp = getattr(engine, "_blueprint", None)
        if bp is not None and hasattr(bp, "get_strategy"):
            return bp.get_strategy
        return None

    def _seat_range_dicts(self, state: GameState, engine):
        """Per-slot in-distribution range weight dicts, or None (→ uniform)."""
        get_strategy = self._get_strategy_reader(engine)
        if get_strategy is None:
            return None
        try:
            from app.agent.live_resolve_ranges import seat_range_weight_dicts
            return seat_range_weight_dicts(
                state.seat_of_slot, state.preflop_prefix, get_strategy,
            )
        except Exception:
            # Range seeding is an OPTIONAL refinement; never let it break a
            # solve. A failure here falls back to uniform ranges — but NEVER
            # silently (review 2026-06-11): the fallback is disclosed in the
            # provider quality note and the cause is logged here.
            logger.warning(
                "LiveResolveSource: range seeding FAILED for prefix %s — "
                "falling back to UNIFORM ranges",
                list(state.preflop_prefix), exc_info=True,
            )
            return None

    def get_mix(self, state: GameState, hero_cards: HeroCards) -> ActionMix:
        from app.agent.live_resolve_adapter import (
            build_solve_inputs,
            hero_grid_row,
            labels_to_index_freqs,
        )

        engine = self._ensure_engine()

        # Seed in-distribution ranges (preflop σ̄) when possible — removes the
        # over-bet confound (uniform ranges make the re-solve model a wide/weak
        # villain). Falls back to uniform on an unparsed preflop line or
        # without a strategy reader — RECORDED so the provider can disclose it.
        range_dicts = self._seat_range_dicts(state, engine) if self._seed_ranges else None
        self.last_range_seeding = (
            "blueprint_preflop_sigma" if range_dicts is not None else "uniform"
        )

        inputs = build_solve_inputs(state, range_weight_dicts=range_dicts)
        #   raises LiveResolveUnsupported on terminal / river / mid-street re-entry.

        # bet_sizes == the agent's blueprint vocab so the solver's action count
        # matches state.num_legal_actions() 1:1 (clean label->index mapping).
        bet_sizes = list(state.postflop_bet_sizes)
        subsequent = list(state.subsequent_bet_sizes)
        add_allin = bool(state.add_explicit_allin)

        hero_idx = inputs["hero_player_idx"]
        try:
            all_by_player = engine.solve(
                board_strs=inputs["board_strs"],
                board_ints=inputs["board_ints"],
                player_ranges=inputs["player_ranges"],
                positions_sorted=inputs["positions_sorted"],
                player_seats=inputs["player_seats"],
                pot_bb=inputs["pot_bb"],
                stack_bb=inputs["stack_bb"],
                bet_sizes=bet_sizes,
                subsequent_bet_sizes=subsequent,
                add_explicit_allin=add_allin,
                game_action_indices=inputs["game_action_indices"],
                iterations=self._iterations,
                num_threads=self._num_threads,
                num_rollouts=self._num_rollouts,
                cache_k=self._cache_k,
                vec_mode=1 if self._vec_mode else 0,
                readout=self._readout,
                hero_replay={
                    "player_idx": hero_idx,
                    "street_actions": inputs["street_actions"],
                    "expected_actions": inputs["n_legal"],
                },
            )
        except Exception as e:
            # Known abstraction-surface limits (tree raise cap, real-chips
            # action-set divergence, unvalidated vocab shape) are surfaced
            # by the engine as ReplayUnsupported -> convert to the
            # documented LiveResolveUnsupported so HybridSource falls back
            # to LOOKUP. Anything else (real solver errors, vocab drift)
            # propagates loud. (Name fallback covers injected fake engines
            # that define their own ReplayUnsupported.)
            from app.services.cpu_es_solver import ReplayUnsupported
            if isinstance(e, ReplayUnsupported) or \
                    type(e).__name__ == "ReplayUnsupported":
                raise LiveResolveUnsupported(str(e)) from e
            raise

        row = hero_grid_row(inputs["player_ranges"], hero_idx, hero_cards)
        strat = all_by_player.get(hero_idx, {}).get(row)
        n_legal = int(inputs["n_legal"])

        if not strat:
            # The engine returned no strategy for the hero's hand (e.g. a
            # never-visited / weight-0 row). Fall back to uniform-over-legal,
            # flagged honestly — same Pluribus Algorithm-1 substitution LOOKUP
            # uses on a miss.
            return ActionMix.uniform(n_legal, is_fallback=True)

        # The hero's node-local label set maps with the size vocab in force
        # AT THAT NODE: subsequent sizes once this street has a bet/raise.
        mix_sizes = subsequent if state.raises_this_street > 0 else bet_sizes
        freqs = labels_to_index_freqs(
            strat,
            has_to_call=bool(inputs["has_to_call"]),
            n_legal=n_legal,
            bet_sizes=tuple(mix_sizes),
            add_explicit_allin=add_allin,
        )
        if not freqs:
            return ActionMix.uniform(n_legal, is_fallback=True)
        return ActionMix(freqs)


# ── HYBRID (full-hand playability: live where supported, lookup elsewhere) ─


class HybridSource(StrategySource):
    """LIVE_RESOLVE where it's supported, blueprint LOOKUP where it isn't.

    The composition that makes a FULL HAND playable end-to-end: flop, turn
    and (since 2026-06-11) river decisions go to the live re-solve (the
    quality ceiling — for 6p, the vec-Linear-final path; river settles
    showdowns natively), while the remaining unsupported surface (raise-capped
    trees, action-count divergence, unsupported preflop shapes) falls back to
    the blueprint σ̄ lookup (the floor the 2p/3p full-hand play tests
    validated).

    Only the DOCUMENTED unsupported surface falls back (the
    LiveResolveUnsupported exception type). Real solver errors — vocab
    drift, replay mismatches, engine failures — still raise (fail loud).
    """

    def __init__(self, live: StrategySource, lookup: StrategySource) -> None:
        self._live = live
        self._lookup = lookup

    def get_mix(self, state: GameState, hero_cards: HeroCards) -> ActionMix:
        try:
            return self._live.get_mix(state, hero_cards)
        except LiveResolveUnsupported as e:
            # Observability: the fallback is by design, but a live-resolve
            # surface regression would otherwise be invisible in play.
            logger.info("HybridSource: live unsupported -> LOOKUP (%s)", e)
            return self._lookup.get_mix(state, hero_cards)
