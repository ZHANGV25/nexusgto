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
import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from app.engine.agent.game_state import GameState
from app.engine.agent.key_builder import build_key
from app.engine.agent.live_resolve_adapter import (
    LiveResolveConvergenceError,
    LiveResolveUnsupported,
)
from app.engine.agent.modes import ModeConfig
from app.engine.agent.preflop_range import hand_class_169

# Re-export so callers can `from app.engine.agent.strategy_source import
# LiveResolveUnsupported` alongside LiveResolveSource (the public seam).
__all__ = [
    "ActionMix",
    "StrategySource",
    "LookupSource",
    "OfflineLiftedSource",
    "LiveResolveSource",
    "HybridSource",
    "LiveResolveConvergenceError",
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
            from engine.bps_reader_py import BPSReaderC  # type: ignore[import-untyped]
            self._reader = BPSReaderC(config.blueprint_path)

        # Read the trained vocab from the blueprint metadata (the single
        # source of truth for the action COUNT). Used to fail-loud if a caller
        # built a GameState against a different blueprint's vocab.
        meta = self._reader.get_metadata() or {}
        self._meta_postflop = tuple(meta.get("postflop_bet_sizes", ()) or ())
        self._meta_subsequent = tuple(meta.get("subsequent_bet_sizes", ()) or ())
        self._meta_add_allin = bool(int(meta.get("add_explicit_allin", 0) or 0))
        raw_preflop_tiers = meta.get("preflop_tiers", {}) or {}
        tier_items = tuple(sorted(
            (int(level), tuple(float(size) for size in sizes))
            for level, sizes in raw_preflop_tiers.items()
        ))
        if len({level for level, _sizes in tier_items}) != len(tier_items):
            raise ValueError(
                "blueprint preflop_tiers has duplicate normalized level indices"
            )
        self._meta_preflop_tier_items = tier_items
        self._meta_preflop_max_raises = int(
            meta.get("preflop_max_raises", 0) or 0
        )
        self._meta_has_add_allin = "add_explicit_allin" in meta
        raw_blinds = meta.get("blinds", ()) or ()
        self._meta_blinds = tuple(float(chips) for chips in raw_blinds)
        if self._meta_blinds and len(self._meta_blinds) != 2:
            raise ValueError(
                f"blueprint blinds metadata must be [SB, BB], got {raw_blinds!r}"
            )
        self._meta_initial_stack = (
            float(meta["initial_stack"]) if "initial_stack" in meta else None
        )
        self._meta_num_players = (
            float(meta["num_players"]) if "num_players" in meta else None
        )

    def _bucket(self, state: GameState, hero_cards: HeroCards) -> int:
        c0, c1 = hero_cards
        board = list(state.board)
        if state.street == 0:
            if board:
                raise ValueError("street-0 blueprint lookup requires board=[]")
            return hand_class_169(c0, c1)
        if state.street == 1:
            return self._reader.hand_to_bucket(board[:3], c0, c1)
        if state.street == 2:
            return self._reader.hand_to_bucket_turn(board[:4], c0, c1)
        if state.street == 3:
            return self._reader.hand_to_bucket_river(board[:5], c0, c1)
        raise ValueError(f"unsupported blueprint street {state.street}")

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

        # A canonical preflop state is generated from the exact same tiered
        # vocabulary as the trainer. Unlike observed postflop play below,
        # there is no real-chip abstraction gap that can safely prune only a
        # raise tail: any length difference is a state/metadata bug.
        if key.street == 0 and len(arr) != n_legal:
            raise ValueError(
                f"preflop action-count mismatch at {key.as_tuple()}: "
                f"blueprint stores {len(arr)} actions but state says "
                f"n_legal={n_legal}"
            )
        if arr.sum() <= 0.0:
            return self._on_miss(n_legal)

        # Length reconciliation. The blueprint stored the node's action set
        # under CANONICAL trainer chips; the state's n_legal uses REAL chips
        # (advance_observed applies real amounts). A real bet that
        # commitment-prunes the vocab drops RAISE slots only — and the
        # trainer orders actions [fold][check/call][raises asc], so the
        # stored σ̄ is PREFIX-aligned with the pruned legal set: project by
        # truncating the tail and renormalizing (review 2026-06-10; without
        # this, a covering bet/all-in on the hero crashed the hand). A SHORTER
        # undercount; prefix-pad the phantom raise tail with zero, symmetric to
        # the 2026-06-10 truncation above. Non-raise shortfalls still fail loud.
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
            n_nonraise = (1 if state.facing_bet else 0) + 1
            if len(arr) < n_nonraise:
                raise ValueError(
                    f"action-count mismatch at {key.as_tuple()}: blueprint stores "
                    f"{len(arr)} actions but GameState vocab says {n_legal}. The "
                    f"GameState's postflop_bet_sizes/subsequent_bet_sizes/"
                    f"add_explicit_allin must match the loaded blueprint metadata."
                )
            arr = np.pad(arr, (0, n_legal - len(arr)), constant_values=0.0)

        arr = arr / arr.sum()
        return ActionMix({i: float(arr[i]) for i in range(n_legal)})

    def _assert_vocab_matches(self, state: GameState) -> None:
        """Fail loud if the GameState's action vocab disagrees with the loaded
        blueprint's metadata — that means the state was built for a different
        blueprint and its legal-action counts would be wrong.

        Only enforced when the blueprint actually carries the metadata fields
        (some test readers don't); a missing field is skipped, not failed.
        """
        if state.street == 0:
            state_tiers = tuple(
                tuple(float(size) for size in tier)
                for tier in state.preflop_tiers
            )
            state_tier_items = tuple(enumerate(state_tiers))
            mismatch = bool(
                self._meta_preflop_tier_items
                and state_tier_items != self._meta_preflop_tier_items
            )
            mismatch = mismatch or bool(
                self._meta_preflop_max_raises
                and state.preflop_max_raises
                != self._meta_preflop_max_raises
            )
            mismatch = mismatch or bool(
                self._meta_has_add_allin
                and bool(state.add_explicit_allin) != self._meta_add_allin
            )
            mismatch = mismatch or bool(
                self._meta_blinds
                and (state.small_blind, state.big_blind) != self._meta_blinds
            )
            mismatch = mismatch or bool(
                self._meta_initial_stack is not None
                and state.initial_stack != self._meta_initial_stack
            )
            mismatch = mismatch or bool(
                self._meta_num_players is not None
                and self._meta_num_players != 6
            )
            if mismatch:
                raise ValueError(
                    "PreflopState vocab does not match the loaded blueprint: "
                    f"state tiers={state_tiers} max_raises="
                    f"{state.preflop_max_raises} allin="
                    f"{state.add_explicit_allin} blinds="
                    f"{(state.small_blind, state.big_blind)} initial_stack="
                    f"{state.initial_stack} num_players=6 vs blueprint tiers="
                    f"{self._meta_preflop_tier_items} max_raises="
                    f"{self._meta_preflop_max_raises} allin="
                    f"{self._meta_add_allin} blinds={self._meta_blinds} "
                    f"initial_stack={self._meta_initial_stack} "
                    f"num_players={self._meta_num_players}."
                )
            return

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

    This is the NON-real-time ceiling — a CFR re-solve whose latency depends on
    the configured iteration budget, NOT a sub-second lookup.
    It must be flagged not-real-time in any UI. It is the ceiling MODE, not the
    blueprint FLOOR (that is LookupSource).

    ENGINE (empirically validated GPU-free this session — see
    docs/BENCHMARK_RESULTS.md §3c and app/agent/live_resolve_adapter.py): wraps
    poker-solver's CpuEsSolver (CPU external-sampling ES) for flop, turn AND
    river (2026-06-11: river showdown terminals settle natively in the
    named engine — has no GPU-free multi-street solve path (its solver_v2 CPU
    fallback crashes building flop/turn trees), so LIVE_RESOLVE wraps CpuEsSolver
    instead; that is the honest WIRE+VERIFY of the realtime pipeline GPU-free.

    Each get_mix remains a fresh re-solve. By default its ranges are rebuilt
    independently; NEXUS_BELIEF_CARRY (or the constructor override) enables a
    per-hand session that multiplies completed-street search-sigma likelihoods
    into later sigma-seeded ranges. The 12GB blueprint load is one-time in the
    held engine.
    """

    def __init__(
        self, config: ModeConfig, cpu_es_solver=None, iterations: int = 200,
        seed_ranges: bool = True, strategy_reader=None,
        vec_mode: bool = False, readout: str = "avg", num_threads: int = 0,
        num_rollouts: int = 3, cache_k: int = 1,
        hero_traverser_actual_fraction: float = 0.5,
        hero_row_min_strategy_sum_mass: float = 10.0,
        belief_carry: Optional[bool] = None,
        offtree_add: Optional[bool] = None,
        deadline_ms: Optional[int] = None,
    ) -> None:
        self._config = config
        self._engine = cpu_es_solver  # held across calls (12GB load is one-time)
        self._iterations = int(iterations)
        self._deadline_ms = (
            int(deadline_ms) if deadline_ms is not None else None
        )
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
        self._hero_traverser_actual_fraction = float(
            hero_traverser_actual_fraction
        )
        self._hero_row_min_strategy_sum_mass = float(
            hero_row_min_strategy_sum_mass
        )
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
        # None follows the environment on every get_mix so in-process gate
        # harnesses can switch it without rebuilding the held 12GB engine.
        # The session itself stays lazy: flag OFF never constructs one.
        # NEXUS_BELIEF_CARRY=1 is for held-source in-process drivers; a fresh
        # per-job /v1 source pays stash extraction but dies before consuming it.
        self._belief_carry_override = belief_carry
        self._belief_session = None
        # Same per-call override contract as belief carry. None follows the
        # environment; explicit True/False is used by in-process A/B harnesses.
        self._offtree_add_override = offtree_add
        # RECORDED per call (call-local, like RecordingHybridSource): how the
        # last get_mix seeded its ranges — "blueprint_preflop_sigma" or
        # "uniform". The engine provider reads this to DISCLOSE uniform
        # seeding in the response quality note (review 2026-06-11: silent
        # uniform seeding on unparsed preflop shapes was an unflagged
        # quality compromise).
        self.last_range_seeding: Optional[str] = None
        # Hand-grid disclosure for the provider: the dose used and whether
        # the served row was the hero's physical combo (K=ALL) or a 169-class
        # representative (K=1, suit-blind for the hero).
        self.last_grid_k: Optional[object] = None
        self.last_grid_rows: Optional[int] = None
        self.last_hero_row_exact: Optional[bool] = None
        self.last_hero_row_floored: Optional[bool] = None
        # Structured per-call disclosure. None is the flag-OFF contract.
        self.last_belief_carry: Optional[dict] = None
        self.last_offtree_add: Optional[dict] = None
        self.last_hero_row_strategy_sum_mass: Optional[float] = None
        self.last_deadline_ms: Optional[int] = None
        self.last_iterations_run: Optional[int] = None

    def _belief_carry_enabled(self) -> bool:
        if self._belief_carry_override is not None:
            return bool(self._belief_carry_override)
        value = os.environ.get("NEXUS_BELIEF_CARRY", "")
        return value[:1] in ("1", "t", "T", "y", "Y")

    def _offtree_add_enabled(self) -> bool:
        if self._offtree_add_override is not None:
            return bool(self._offtree_add_override)
        from app.engine.agent.offtree_add import offtree_add_enabled
        return offtree_add_enabled()

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        if not self._config.blueprint_path:
            raise ValueError(
                "LiveResolveSource needs a blueprint_path (or an injected "
                "cpu_es_solver) to construct the CpuEsSolver engine."
            )
        from app.engine.cpu_es_solver import CpuEsSolver
        self._engine = CpuEsSolver(self._config.blueprint_path)
        return self._engine

    @staticmethod
    def _supports_unequal_stack_abi(engine) -> bool:
        """Probe both additive native entry points through the held engine."""
        probe = getattr(engine, "supports_unequal_stack_abi", None)
        if not callable(probe):
            return False
        try:
            return bool(probe())
        except Exception:
            logger.info(
                "unequal-stack solver ABI probe failed",
                exc_info=True,
            )
            return False

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
            from app.engine.agent.live_resolve_ranges import seat_range_weight_dicts
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
        from app.engine.agent.live_resolve_adapter import (
            HERO_ROW_WEIGHT_FLOOR,
            aggregate_grid_row_strategies,
            floor_hero_row,
            build_solve_inputs,
            hero_exact_row,
            hero_grid_row,
            hero_grid_rows,
            labels_to_index_freqs,
        )

        self.last_hero_row_strategy_sum_mass = None
        self.last_deadline_ms = self._deadline_ms
        self.last_iterations_run = None

        # Real geometry is a request-local exactness contract.  It cannot be
        # disabled by the process-level/default off-tree gate: observed street
        # sizes must either be inserted and replayed exactly or fail closed.
        real_geometry = bool(getattr(state, "real_geometry", False))
        offtree_enabled = real_geometry or self._offtree_add_enabled()
        if offtree_enabled:
            self.last_offtree_add = {
                "enabled": True,
                "events": [
                    {
                        "step": int(step),
                        "increment": int(increment),
                        "pot_before": int(pot_before),
                        "to_call_before": int(to_call_before),
                        "stack_before": int(stack_before),
                    }
                    for (step, increment, pot_before, to_call_before,
                         stack_before)
                    in state.street_offtree_raises
                ],
                "added_first": [],
                "added_subsequent": [],
                "skipped": [],
                "replayed_through_added": False,
                "hero_merged_mass": 0.0,
                "dropped": False,
                "drop_reason": None,
            }
        else:
            self.last_offtree_add = None

        belief_enabled = self._belief_carry_enabled()
        belief_session = None
        if belief_enabled:
            from app.engine.agent.belief_carry import BeliefCarrySession

            if self._belief_session is None:
                self._belief_session = BeliefCarrySession()
            belief_session = self._belief_session
            try:
                belief_session.observe(state)
                self.last_belief_carry = belief_session.summary()
            except Exception:
                # Belief carry is an optional refinement. A malformed session
                # must never prevent the underlying live solve.
                logger.warning(
                    "LiveResolveSource: belief session observe FAILED; "
                    "solving this node without carry",
                    exc_info=True,
                )
                self._belief_session = None
                belief_session = None
                self.last_belief_carry = {
                    "continued": False, "reset": False, "reset_reason": None,
                    "applied": False, "dropped": True,
                    "drop_reason": "belief_carry: dropped (observe failed)",
                    "invalidated_post_solve": False, "carry_dead": False,
                    "folded_streets": [], "skipped_streets": [],
                    "slot_factors": [], "stash_status": None,
                }
        else:
            # Do not retain a stale hand across an OFF interval if a driver
            # toggles the environment between calls.
            self._belief_session = None
            self.last_belief_carry = None

        engine = self._ensure_engine()

        # Seed in-distribution ranges (preflop σ̄) when possible — removes the
        # over-bet confound (uniform ranges make the re-solve model a wide/weak
        # villain). Falls back to uniform on an unparsed preflop line or
        # without a strategy reader — RECORDED so the provider can disclose it.
        range_dicts = self._seat_range_dicts(state, engine) if self._seed_ranges else None
        self.last_range_seeding = (
            "blueprint_preflop_sigma" if range_dicts is not None else "uniform"
        )

        if belief_session is not None:
            try:
                if range_dicts is None:
                    belief_session.skip_uniform_preflop()
                else:
                    carried = belief_session.apply_ranges(
                        range_dicts, state.n_slots
                    )
                    if carried is None:
                        # Wrong-length sigma dicts would be silently ignored by
                        # the adapter. Make that uniform fallback explicit.
                        range_dicts = None
                        self.last_range_seeding = "uniform"
                    else:
                        range_dicts = carried
                        if belief_session.summary()["applied"]:
                            self.last_range_seeding = (
                                "blueprint_preflop_sigma+belief_carry"
                            )
                self.last_belief_carry = belief_session.summary()
            except Exception:
                logger.warning(
                    "LiveResolveSource: belief range application FAILED; "
                    "solving with unfactored preflop ranges",
                    exc_info=True,
                )
                belief_session.record_failure("range application failed")
                self.last_belief_carry = belief_session.summary()

        inputs = build_solve_inputs(state, range_weight_dicts=range_dicts)
        #   raises LiveResolveUnsupported on terminal / river / mid-street re-entry.

        unequal_solve_kwargs = {}
        if "root_stacks_bb" in inputs:
            if not self._supports_unequal_stack_abi(engine):
                raise LiveResolveUnsupported(
                    "loaded poker-solver lacks ss_build_tree_stacks and/or "
                    "es_create_with_root_pots"
                )
            unequal_solve_kwargs = {
                "root_stacks_bb": inputs["root_stacks_bb"],
                "root_pot_amounts": inputs["root_pot_amounts"],
                "root_pot_eligible_masks": inputs[
                    "root_pot_eligible_masks"
                ],
            }

        # bet_sizes == the agent's blueprint vocab so the solver's action count
        # matches state.num_legal_actions() 1:1 (clean label->index mapping).
        bet_sizes = list(state.postflop_bet_sizes)
        subsequent = list(state.subsequent_bet_sizes)
        add_allin = bool(state.add_explicit_allin)

        offtree_analysis = None
        offtree_plan = None
        if offtree_enabled:
            from app.engine.agent.offtree_add import (
                OffTreeExactUnsupported,
                build_offtree_extension,
            )

            try:
                offtree_analysis = build_offtree_extension(
                    state,
                    bet_sizes,
                    subsequent,
                    strict=real_geometry,
                )
            except OffTreeExactUnsupported as unsupported:
                if self.last_offtree_add is None:
                    raise RuntimeError(
                        "strict off-tree failure without enabled disclosure"
                    ) from unsupported
                skip = unsupported.skip
                self.last_offtree_add["skipped"] = [{
                    "step": int(skip.step),
                    "fraction": float(skip.fraction),
                    "raise_level": str(skip.raise_level),
                    "reason": str(skip.reason),
                }]
                self.last_offtree_add["dropped"] = True
                self.last_offtree_add["drop_reason"] = str(unsupported)
                raise LiveResolveUnsupported(str(unsupported)) from unsupported
            if offtree_analysis is not None:
                offtree_plan = offtree_analysis.plan
                if self.last_offtree_add is None:
                    raise RuntimeError(
                        "off-tree analysis exists without enabled disclosure"
                    )
                self.last_offtree_add["events"] = [
                    {
                        "step": int(event.step),
                        "increment": int(event.increment),
                        "pot_before": int(event.pot_before),
                        "to_call_before": int(event.to_call_before),
                        "stack_before": int(event.stack_before),
                        "fraction": float(event.fraction),
                        "raise_level": str(event.raise_level),
                    }
                    for event in offtree_analysis.events
                ]
                self.last_offtree_add["skipped"] = [
                    {
                        "step": int(skip.step),
                        "fraction": float(skip.fraction),
                        "raise_level": str(skip.raise_level),
                        "reason": str(skip.reason),
                    }
                    for skip in offtree_analysis.skipped
                ]
                if offtree_plan is not None:
                    self.last_offtree_add["added_first"] = list(
                        offtree_plan.added_first
                    )
                    self.last_offtree_add["added_subsequent"] = list(
                        offtree_plan.added_subsequent
                    )

        hero_idx = inputs["hero_player_idx"]
        hero_rows = hero_grid_rows(
            inputs["player_ranges"], hero_idx, hero_cards,
            inputs.get("player_range_metadata"),
        )
        # K=ALL contains the physical combo; a K=1 grid contains one
        # representative combo per 169 class. In the latter case that class row
        # is the resolver's private-hand identity (and suit-blind for hero).
        exact_row = hero_exact_row(
            inputs["player_ranges"], hero_idx, hero_rows, hero_cards,
            board=(
                inputs["board_ints"] if inputs.get("grid_k") == "ISO"
                else None
            ),
        )
        actual_row = exact_row if exact_row is not None else hero_rows[0][0]
        # HERO-ROW FLOOR (2026-09-09): a hero hand the seeded range holds at
        # zero for this line would be dropped at readout and refuse; serve it
        # from a floored row instead (disclosed in provenance).
        floored_ranges, hero_row_floored = floor_hero_row(
            inputs["player_ranges"], hero_idx, actual_row,
        )
        if hero_row_floored:
            inputs = dict(inputs, player_ranges=floored_ranges)
            hero_rows = [
                (row_idx, HERO_ROW_WEIGHT_FLOOR if row_idx == actual_row
                 else weight)
                for row_idx, weight in hero_rows
            ]
        self.last_hero_row_floored = bool(hero_row_floored)
        self.last_grid_k = inputs.get("grid_k", 1)
        self.last_grid_rows = len(inputs["player_ranges"][hero_idx])
        self.last_hero_row_exact = exact_row is not None
        hero_actual_hand = {
            "player_idx": hero_idx,
            "hand_idx": actual_row,
            "traverser_fraction": self._hero_traverser_actual_fraction,
        }
        belief_stash = None
        belief_solve_kwargs = {}
        if belief_session is not None:
            belief_stash = {
                "entry_slots": list(inputs["entry_slots"]),
                "hist_start": (
                    len(state.postflop_history) - len(state.street_history)
                ),
            }
            # Empty when OFF, so the existing solve call receives no new kwarg.
            belief_solve_kwargs["belief_stash_out"] = belief_stash
        # cont_ev CATALOG routing (multi-spot, 2026-07-08): when the engine has a
        # catalog router attached (engine.cont_ev_router — only when
        # CONT_EV_CATALOG_MANIFEST is configured), route this solve to the right
        # pre-filled table under the router's lock (arm -> solve -> restore as one
        # unit). When no router is attached (default OFF), the nullcontext branch
        # runs the identical engine.solve — byte-identical to the shipped path.
        _router = getattr(engine, "cont_ev_router", None)
        _direct_cev_holder = getattr(engine, "_cev_holder", None)
        if real_geometry and (
            _router is not None or _direct_cev_holder is not None
        ):
            cont_ev_source = (
                "cont_ev catalog"
                if _router is not None
                else "single-table cont_ev"
            )
            reason = (
                "real geometry is incompatible with the canonical-geometry "
                f"{cont_ev_source}"
            )
            if self.last_offtree_add is not None:
                self.last_offtree_add["dropped"] = True
                self.last_offtree_add["drop_reason"] = reason
            raise LiveResolveUnsupported(reason)
        if _router is not None:
            _arm_cm = _router.arm_for_solve(
                engine,
                board_ints=inputs["board_ints"],
                game_action_indices=inputs["game_action_indices"],
                n_players=len(inputs["player_seats"]),
            )
        else:
            _arm_cm = contextlib.nullcontext()
        _offtree_extension_served = False
        try:
            with _arm_cm:
                if offtree_plan is not None:
                    from app.engine.cpu_es_solver import (
                        OffTreeExtensionUnsafe,
                    )

                    try:
                        all_by_player = engine.solve(
                            board_strs=inputs["board_strs"],
                            board_ints=inputs["board_ints"],
                            player_ranges=inputs["player_ranges"],
                            positions_sorted=inputs["positions_sorted"],
                            player_seats=inputs["player_seats"],
                            pot_bb=inputs["pot_bb"],
                            stack_bb=inputs["stack_bb"],
                            bet_sizes=list(offtree_plan.bet_sizes),
                            subsequent_bet_sizes=list(
                                offtree_plan.subsequent_bet_sizes
                            ),
                            add_explicit_allin=add_allin,
                            game_action_indices=inputs["game_action_indices"],
                            iterations=self._iterations,
                            deadline_ms=self._deadline_ms,
                            num_threads=self._num_threads,
                            num_rollouts=self._num_rollouts,
                            cache_k=self._cache_k,
                            vec_mode=1 if self._vec_mode else 0,
                            readout=self._readout,
                            hero_replay={
                                "player_idx": hero_idx,
                                "street_actions": inputs["street_actions"],
                                "expected_actions": inputs["n_legal"],
                                "offtree": {
                                    "base_bet_sizes": list(bet_sizes),
                                    "base_subsequent_bet_sizes": list(
                                        subsequent
                                    ),
                                    "event_fractions": dict(
                                        offtree_plan.event_fractions
                                    ),
                                    **({
                                        "expected_event_amounts": dict(
                                            offtree_plan.event_amounts
                                        ),
                                    } if real_geometry else {}),
                                },
                            },
                            hero_actual_hand=hero_actual_hand,
                            **belief_solve_kwargs,
                            **unequal_solve_kwargs,
                        )
                        _offtree_extension_served = True
                    except OffTreeExtensionUnsafe as unsafe:
                        if self.last_offtree_add is None:
                            raise RuntimeError(
                                "unsafe off-tree extension without disclosure"
                            ) from unsafe
                        self.last_offtree_add["dropped"] = True
                        self.last_offtree_add["drop_reason"] = str(unsafe)
                        if real_geometry:
                            raise LiveResolveUnsupported(str(unsafe)) from unsafe
                        # Fail safe exactly once: rebuild/solve the unextended
                        # base tree through the shipped replay path.
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
                            deadline_ms=self._deadline_ms,
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
                            hero_actual_hand=hero_actual_hand,
                            **belief_solve_kwargs,
                            **unequal_solve_kwargs,
                        )
                else:
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
                        deadline_ms=self._deadline_ms,
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
                        hero_actual_hand=hero_actual_hand,
                        **belief_solve_kwargs,
                        **unequal_solve_kwargs,
                    )
        except Exception as e:
            if belief_session is not None and belief_stash is not None:
                # Never leave an older same-street stash pending after the
                # latest solve attempt failed. A later street must skip this
                # factor source instead of silently using stale search sigma.
                try:
                    belief_session.receive_stash(
                        {
                            "status": "empty",
                            "reason": (
                                "latest live solve failed before belief "
                                "stash intake"
                            ),
                            "entry_slots": list(inputs["entry_slots"]),
                            "hist_start": (
                                len(state.postflop_history)
                                - len(state.street_history)
                            ),
                            "street": int(state.street),
                        }
                    )
                    self.last_belief_carry = belief_session.summary()
                except Exception:
                    logger.warning(
                        "LiveResolveSource: failed to discard stale belief "
                        "stash after solve error",
                        exc_info=True,
                    )
                    belief_session.record_failure(
                        "solve failed and stale stash discard failed",
                        clear=True,
                    )
                    self.last_belief_carry = belief_session.summary()
            # Known abstraction-surface limits (tree raise cap, real-chips
            # action-set divergence, unvalidated vocab shape) are surfaced
            # by the engine as ReplayUnsupported -> convert to the
            # documented LiveResolveUnsupported so HybridSource falls back
            # to LOOKUP. Anything else (real solver errors, vocab drift)
            # propagates loud. (Name fallback covers injected fake engines
            # that define their own ReplayUnsupported.)
            from app.engine.cpu_es_solver import ReplayUnsupported
            if isinstance(e, ReplayUnsupported) or \
                    type(e).__name__ == "ReplayUnsupported":
                if (self.last_offtree_add is not None
                        and offtree_plan is not None):
                    self.last_offtree_add["dropped"] = True
                    self.last_offtree_add["drop_reason"] = str(e)
                raise LiveResolveUnsupported(str(e)) from e
            raise

        if belief_session is not None and belief_stash is not None:
            # The packer preserves these pre-seeded fields, but setdefault
            # makes injected fake engines and an extraction failure equally
            # safe and fully disclosed.
            belief_stash.setdefault("entry_slots", list(inputs["entry_slots"]))
            belief_stash.setdefault(
                "hist_start",
                len(state.postflop_history) - len(state.street_history),
            )
            belief_stash.setdefault("street", int(state.street))
            belief_stash.setdefault("status", "empty")
            if belief_stash.get("status") != "ok":
                belief_stash.setdefault(
                    "reason", "solver did not populate the belief stash"
                )
            try:
                belief_session.receive_stash(belief_stash)
            except Exception:
                logger.warning(
                    "LiveResolveSource: belief stash intake FAILED; "
                    "discarding carry data from this solve",
                    exc_info=True,
                )
                # A later same-street solve must replace the prior stash even
                # when this latest extraction failed: beliefs use the LAST
                # search solve, never a silently stale one.
                try:
                    belief_session.receive_stash(
                        {
                            "status": "empty",
                            "reason": "belief stash intake failed",
                            "entry_slots": list(inputs["entry_slots"]),
                            "hist_start": (
                                len(state.postflop_history)
                                - len(state.street_history)
                            ),
                            "street": int(state.street),
                        }
                    )
                except Exception:
                    belief_session.record_failure(
                        "stash intake failed", clear=True
                    )
            self.last_belief_carry = belief_session.summary()

        # Preserve the exact K=1 lookup path, including hero_grid_row's
        # fail-loud contract. K>1 uses the first row only as an observability
        # anchor; the served strategy below aggregates every class row.
        legacy_k1 = inputs.get("grid_k", 1) == 1
        row = (
            hero_grid_row(inputs["player_ranges"], hero_idx, hero_cards)
            if legacy_k1 else hero_rows[0][0]
        )
        solve_stats = getattr(engine, "_last_solve_stats", None)
        if (
            self._deadline_ms is not None
            and isinstance(solve_stats, dict)
            and solve_stats.get("deadline_ms") is not None
        ):
            self.last_deadline_ms = int(solve_stats["deadline_ms"])
        if (
            isinstance(solve_stats, dict)
            and solve_stats.get("iterations_run_observed") is True
            and solve_stats.get("iterations_run") is not None
        ):
            self.last_iterations_run = int(solve_stats["iterations_run"])
        # CpuEsSolver always supplies this key and fails loud if the native
        # coverage export is missing. The conditional preserves only injected
        # test/third-party solver compatibility; production cannot skip it.
        if (
            isinstance(solve_stats, dict)
            and "hero_row_strategy_sum_mass_by_row" in solve_stats
        ):
            mass_by_row = (
                solve_stats.get("hero_row_strategy_sum_mass_by_row") or {}
            )
            mass = float(
                mass_by_row.get(
                    actual_row,
                    mass_by_row.get(str(actual_row), float("nan")),
                )
            )
            self.last_hero_row_strategy_sum_mass = mass
            if (
                not np.isfinite(mass)
                or mass < self._hero_row_min_strategy_sum_mass
            ):
                raise LiveResolveConvergenceError(
                    "live resolve hero row strategy-sum mass "
                    f"{mass:g} is below the configured minimum "
                    f"{self._hero_row_min_strategy_sum_mass:g}"
                )
        if self.last_offtree_add is not None and offtree_plan is not None:
            stats = getattr(engine, "_last_solve_stats", None) or {}
            self.last_offtree_add["replayed_through_added"] = bool(
                stats.get("offtree_replayed_through", False)
            )
            merged_by_row = stats.get(
                "offtree_hero_merged_mass_by_row", {}
            ) or {}
            dropped_by_row = stats.get(
                "offtree_hero_dropped_mass_by_row", {}
            ) or {}
            reason_by_row = stats.get(
                "offtree_hero_drop_reason_by_row", {}
            ) or {}
            positive_rows = [(r, w) for r, w in hero_rows if w > 0.0]
            if legacy_k1:
                self.last_offtree_add["hero_merged_mass"] = float(
                    merged_by_row.get(row, merged_by_row.get(str(row), 0.0))
                )
                served_dropped_mass = float(
                    dropped_by_row.get(row, dropped_by_row.get(str(row), 0.0))
                )
            else:
                stat_weight = sum(w for _r, w in positive_rows)
                self.last_offtree_add["hero_merged_mass"] = (
                    sum(
                        w * float(merged_by_row.get(r, merged_by_row.get(str(r), 0.0)))
                        for r, w in positive_rows
                    ) / stat_weight if stat_weight > 0.0 else 0.0
                )
                served_dropped_mass = (
                    sum(
                        w * float(dropped_by_row.get(r, dropped_by_row.get(str(r), 0.0)))
                        for r, w in positive_rows
                    ) / stat_weight if stat_weight > 0.0 else 0.0
                )
            if (served_dropped_mass > 0.0
                    and not self.last_offtree_add["dropped"]):
                self.last_offtree_add["dropped"] = True
                if legacy_k1:
                    self.last_offtree_add["drop_reason"] = str(
                        reason_by_row.get(
                            row,
                            reason_by_row.get(str(row), "no_base_raise_child"),
                        )
                    )
                else:
                    reasons = {
                        str(reason_by_row.get(r, reason_by_row.get(str(r))))
                        for r, _w in positive_rows
                        if reason_by_row.get(r, reason_by_row.get(str(r))) is not None
                    }
                    self.last_offtree_add["drop_reason"] = (
                        ",".join(sorted(reasons)) if reasons
                        else "no_base_raise_child"
                    )

        per_hand = all_by_player.get(hero_idx, {})
        positive_class_mass = sum(
            weight for _row_idx, weight in hero_rows if weight > 0.0
        )
        # Serve the hero's PHYSICAL combo row whenever the grid holds it: the
        # class average would blend a flush with three non-flushes. Only a
        # K=1 grid (no physical row) or a zero-mass class falls back to the
        # class row / class aggregate.
        strat = (
            per_hand.get(actual_row)
            if legacy_k1 or positive_class_mass <= 0.0 or exact_row is not None
            else aggregate_grid_row_strategies(per_hand, hero_rows)
        )
        n_legal = int(inputs["n_legal"])

        if not strat:
            # A missing row means the solve never accumulated a usable hero
            # strategy. Live resolve owns no policy fallback.
            raise LiveResolveConvergenceError(
                "live resolve produced no strategy for the requested hero hand"
            )

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
            raise LiveResolveConvergenceError(
                "live resolve produced no positive hero action frequencies"
            )
        if _offtree_extension_served:
            # CpuEsSolver projects all extended children into the base label
            # universe. Preserve literal-zero base actions as keys so the
            # served arity remains exactly the agent's legal arity.
            freqs = {
                action_idx: float(freqs.get(action_idx, 0.0))
                for action_idx in range(n_legal)
            }
        mix = ActionMix(freqs)
        uniform = 1.0 / n_legal
        if (
            len(mix.freqs) == n_legal
            and all(mix.freqs.get(i) == uniform for i in range(n_legal))
        ):
            # CpuEsSolver's scalar average readout is mathematically exact
            # uniform when the per-hand strategy sum is untouched. Its first
            # update also accumulates the initial uniform regret-matched row.
            # Neither is a converged hero policy, so it must never be served or
            # cached as a successful live resolve.
            raise LiveResolveConvergenceError(
                "live resolve produced an exactly uniform hero strategy"
            )
        return mix


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
