"""CPU external-sampling ES re-solver — production engine for postflop streets.

The HYBRID architecture (see docs/MULTIWAY_5_6P_DECISION_MEMO.md): 2-3p (+4p
flop/turn) stay on the GPU; 5p flop/turn move here — a CPU external-sampling
MCCFR re-solver that runs OFF the GPU lock, freeing the single GPU for the common
interactive 2-3p solves. Benchmarked head-to-head (CPU_ES_RESOLVER_BENCHMARK.md):
with Linear CFR + multi-thread Hogwild it reaches the GPU's 1600-iter ship-budget
exploitability ~1.65× faster at 5p AND concurrently with GPU play.

ISOLATION (load-bearing): this path uses its OWN ctypes bindings into
es_resolver_real.dll and its OWN BPSReaderC loaded through that same DLL. It does
NOT touch the GPU provider's `rollout_c` / `bps_reader_py` module globals — the
benchmark's monkeypatch of those would silently repoint the GPU provider's leaf
construction to a different DLL and feed it a cross-DLL BPSReader* handle. A
dedicated `rollout_c` module instance (via importlib) keeps the construct binding
isolated.

SCOPE: FLOP/TURN — cont-strat depth-limited leaf-table terminals, the regime
every quality gate was measured on — and RIVER (2026-06-11). A river tree
terminates in SHOWDOWN/FOLD terminals only (StreetSolverGPU disables cont-strat
leaves at is_river), and the C ES settles showdowns natively via
es_showdown_value — side-pot validated against an exact Python reference
(poker-solver validate_sidepot.py: literal units + 100k-case fuzz < 1e-9 +
bit-exact single-pot regression + the nh>1 vectorized gate). River solves skip
the leaf-table construct entirely (no depth-limit leaves -> num_rollouts has no
effect, no leaf-sampling noise); the bypass is fail-loud (see solve()).
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.engine.offtree_replay import (
    NodeProjection,
    OffTreeExtensionUnsafe,
    ReplayUnsupported,
    build_node_projection,
    replay_offtree_street_node,
    validate_extended_tree,
)

logger = logging.getLogger(__name__)

# Serializes the OPT-IN env-mutation window (NEXUS_ALLOW_CHANCE_SOLVE -> set
# POKER_SEARCH_DEPTH/POKER_TURN_DEPTH_LIMIT around the tree build). solves run in
# asyncio.to_thread worker threads and os.environ is process-global: today the
# tier executor runs solves one-at-a-time so this never contends, but if solve
# concurrency is ever bumped >1 a concurrent default-path build could observe a
# half-set env. The lock is taken ONLY on the opt-in path (the common default
# path never touches env and never acquires it), so it adds zero contention to
# normal solves. See solve().
_ENV_MUTATION_LOCK = threading.Lock()

# Per-leaf flop-history buffer width the C real-oracle (es_real_oracle.c)
# copies — MUST equal ESRO_MAX_FLOP_HIST in poker-solver
# (python/es_resolver/run_real_spot_c.py:35, the validated caller). Only used
# on the lazy and sampled C leaf-oracle construct paths.
_ESRO_MAX_FLOP_HIST = 256


class ProjectedFillMemoryExceeded(MemoryError):
    """An eager-fill arm exceeded its pre-allocation measurement guard."""


class LivePlayerCountExceeded(ValueError):
    """A live re-solve was requested for more players than the served arm.

    2026-09-09: the 7 MB projected-fill guard used to separate 3p from 4p
    fills by size. With the ISO grid the row count depends on the board
    texture (a 4p monotone flop projects ~2.3 MB, a 3p rainbow flop ~4 MB),
    so bytes no longer tell the player counts apart. The policy is stated
    suppresses 4+ live players and the multiway vec path is not a served
    arm. NEXUS_MAX_LIVE_RESOLVE_PLAYERS raises the cap for measurement.
    """


_MAX_LIVE_RESOLVE_PLAYERS_ENV = "NEXUS_MAX_LIVE_RESOLVE_PLAYERS"
_MAX_LIVE_RESOLVE_PLAYERS_DEFAULT = 3


def _max_live_resolve_players() -> int:
    raw = os.environ.get(_MAX_LIVE_RESOLVE_PLAYERS_ENV, "").strip()
    if not raw:
        return _MAX_LIVE_RESOLVE_PLAYERS_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{_MAX_LIVE_RESOLVE_PLAYERS_ENV}={raw!r} is not an integer"
        ) from None
    if not (2 <= value <= 6):
        raise ValueError(
            f"{_MAX_LIVE_RESOLVE_PLAYERS_ENV} must be within [2, 6], got {value}"
        )
    return value


# poker-solver locations (the provider side-loads python/ onto sys.path at init;
# we resolve the DLL + python dir from the same env the provider uses).
_POKER_SOLVER = os.environ.get(
    "POKER_SOLVER_PATH",
)
# Bare name (no extension); c_solver resolves .dll on Windows / .so on Linux.
_ES_DLL_NAME = "es_resolver_real"


def _a3_cont_ev_enabled() -> bool:
    """A3 EXPERIMENT flag — DEFAULT OFF. When unset/0 the cont_ev boundary
    consumer is never loaded or enabled and this engine behaves byte-identically
    to the shipped path (the now-green E2E contract). Enable ONLY on the A3
    measurement box via NEXUS_A3_CONT_EV=1 + NEXUS_A3_CEV_PATH=<...>.cev.
    See docs/A3_CONT_EV_CONSUMER_BRIEF.md."""
    v = os.environ.get("NEXUS_A3_CONT_EV", "")
    return v[:1] in ("1", "t", "T", "y", "Y")


def _a3_cont_ev_turn_enabled() -> bool:
    """Dual-boundary build (2026-06-25) — TURN-boundary arm flag, DEFAULT OFF
    and INDEPENDENT of the flop flag. When set (NEXUS_A3_CONT_EV_TURN=1), the
    engine arms the cont_ev TURN-boundary consumer for multiway TURN solves,
    short-circuiting the live river rollout with the turn-boundary table
    lookup. Requires the cont_ev table to be loaded — i.e. NEXUS_A3_CONT_EV=1
    (+ NEXUS_A3_CEV_PATH) must ALSO be set so _cev_holder is populated; the
    same .cev carries both boundaries (street disambiguation is in the
    key-hash). Enabling the turn arm does NOT change the flop default.

    This stays DEFAULT OFF in the served path: enabling either boundary in
    production is gated behind the 1326-combo equity-validation gate passing
    AND an explicit user decision (docs/CONT_EV_DUAL_BOUNDARY_*)."""
    v = os.environ.get("NEXUS_A3_CONT_EV_TURN", "")
    return v[:1] in ("1", "t", "T", "y", "Y")


def _cont_ev_bias_action_enabled() -> bool:
    """Unit 3 (Stage-1 mean-field) — bias-as-hero-action consumer flag,
    DEFAULT OFF. When set (NEXUS_CONT_EV_BIAS_ACTION=1), an ARMED cont_ev
    boundary (flop and/or turn) regret-matches over the querying player's 4
    stored per-bias EVs PER LEAF INFOSET and returns the regret-matched mixture
    value instead of the uniform mean (docs/PLURIBUS_EXACT_CONFORMANCE_SPEC
    Gap 2 / docs/STAGE1_MEANFIELD_PLAN_2026-07-01.md Unit 3). It only changes
    HOW an armed boundary combines its EVs — it does NOT arm a boundary by
    itself (still needs NEXUS_A3_CONT_EV=1 + the boundary predicate).

    The SEMANTICS the mode computes depend on the .cev the values were walked
    under: a mean_field .cev gives the real thing; a uniform .cev (or one with
    no meta sidecar) gives the hero-best-response-to-uniform "half measure" —
    still legal, but the engine LOGS which one ran (see _load_cont_ev_table's
    bias_semantics read). Stays OFF in the served path until the equity gate
    passes + user opt-in."""
    v = os.environ.get("NEXUS_CONT_EV_BIAS_ACTION", "")
    return v[:1] in ("1", "t", "T", "y", "Y")


def _cont_ev_miss_fallback_enabled() -> bool:
    """Stage-2' pinned-tail MISS fallback — MEASUREMENT ONLY, DEFAULT OFF.
    When set (NEXUS_CONT_EV_MISS_FALLBACK=1), an ARMED cont_ev boundary query
    whose slot is MISS/EMPTY_BIAS sends that ONE rollout down the legacy live
    playout instead of failing loud (INVALID still fails loud). Exists so the
    Stage-2 equity gate / mix-delta can run against tables that cover ~69% of
    the exhaustive key manifest (the power-law tail is unreachable by traffic
    walking — docs/STAGE2_GATE_RESULTS_2026-07-02.md): with the same
    master_seed, a key missing in BOTH A/B tables produces a bit-identical
    fallback value in both arms (per-(leaf,combo) RNG), pinning the tail to
    common ground truth. NEVER enable in the served path — serving keeps the
    fail-loud contract (a serve-time fallback would pay the live rollout per
    uncovered query and destroy the armed latency win)."""
    v = os.environ.get("NEXUS_CONT_EV_MISS_FALLBACK", "")
    return v[:1] in ("1", "t", "T", "y", "Y")


def _max_projected_fill_bytes_default() -> Optional[int]:
    """Env-sourced ceiling on the eager fill's projected buffer, default OFF.

    The eager fill materialises [n_leaves, n_players, max_hands] at 12 bytes
    per cell and is >=99.7% of solve time; on 2026-09-03 one resolve spent
    1353.5s in it and held the box's single solve slot for 22 minutes while
    every other table fell back to the blueprint. The projection is computed
    from the built tree BEFORE the first allocation, so the cost of a spot is
    knowable before it is paid - which is the only predicate that works here,
    because player count does not predict cost (a 4p spot replays at 247s and
    a 6-way limped flop at 8.6s).

    Unset or 0 keeps the shipped behaviour exactly: no ceiling, nothing
    refused. An explicit ``max_projected_fill_bytes`` argument always wins.
    """
    raw = os.environ.get("NEXUS_MAX_PROJECTED_FILL_BYTES", "")
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"NEXUS_MAX_PROJECTED_FILL_BYTES={raw!r} is not an integer"
        ) from None
    if value < 0:
        raise ValueError(
            "NEXUS_MAX_PROJECTED_FILL_BYTES must be >= 0, got "
            f"{value}"
        )
    return value or None


def _leaf_rollout_enabled() -> bool:
    """LAZY per-visited-leaf rollout leaf-value oracle — DEFAULT OFF. When
    unset/0 the solve constructs its leaf values EAGERLY (the shipped path:
    compute_subgame_leaf_values_sampled_c fills a full table, then
    CSolver.from_table) and this engine behaves BYTE-IDENTICALLY to the
    shipped path — not a single behavioral or timing-path change.

    When set (NEXUS_LEAF_ROLLOUT=1), the solve SKIPS the eager fill entirely
    and constructs the solver via CSolver.from_real_oracle (poker-solver
    es_real_oracle.c): per-visited-leaf rollout with a Hogwild-safe cache, the
    blueprint continuation queried live in C, no Python in the hot path. This
    is the LATENCY alternative to the ~27min eager exhaustive fill at 6p (the
    fill is >=99.7% of solve time; the lazy oracle only rolls out the leaves
    the CFR walk actually visits).

    This is the CACHED all-hands variant of the oracle (leaf values are
    strategy-independent, filled once on first hit averaged over cache_k bias
    samples x num_rollouts), so it is safe in BOTH scalar and vec traverser
    modes — this pass does NOT force scalar. (The fresh-bias-per-visit variant
    — the fidelity pass, later — WILL require scalar; see the mode note at the
    construction site.)

    ROLLOUT MODE BYPASSES the eager per-leaf construct, so anything that
    modifies THAT construct is structurally incompatible and MUST fail loud
    rather than half-apply — see the interaction guards at the construction
    site (cont_ev boundary arming NEXUS_A3_CONT_EV / _TURN / bias-action /
    miss-fallback, and 2p NEXUS_LEAF_RANGE_CONDITIONING). DEFAULT OFF keeps the
    served path (byte-golden-tested) untouched until this is EC2-validated."""
    v = os.environ.get("NEXUS_LEAF_ROLLOUT", "")
    return v[:1] in ("1", "t", "T", "y", "Y")


def _leaf_sampled_enabled() -> bool:
    """Fresh per-visit sampled leaf oracle -- DEFAULT OFF.

    Unlike ``NEXUS_LEAF_ROLLOUT`` this mode does not cache an all-hands leaf
    table. Each scalar traversal samples the continuation hands and policy in
    C. The eager serving path is untouched while the flag is absent/off.
    """
    v = os.environ.get("NEXUS_LEAF_SAMPLED", "")
    return v[:1] in ("1", "t", "T", "y", "Y")


def _leaf_sampled_policy() -> int:
    """Parse the sampled-oracle continuation policy (0 or 1)."""
    raw = os.environ.get("NEXUS_LEAF_SAMPLED_POLICY") or "1"
    try:
        policy = int(raw)
    except ValueError:
        raise ValueError(
            f"NEXUS_LEAF_SAMPLED_POLICY={raw!r} is not an integer"
        ) from None
    if policy not in (0, 1):
        raise ValueError(
            "NEXUS_LEAF_SAMPLED_POLICY must be 0 (uniform-fresh) or "
            f"1 (regret-matched), got {policy}"
        )
    return policy


def _leaf_bias_cfr_mode() -> str:
    """Parse the table-backed continuation-bias CFR mode."""
    raw = os.environ.get("NEXUS_LEAF_BIAS_CFR") or "off"
    if raw in ("off", "0"):
        return "off"
    if raw in ("marginal", "joint"):
        return raw
    raise ValueError(
        "NEXUS_LEAF_BIAS_CFR must be off/0, marginal, or joint, "
        f"got {raw!r}"
    )


def _bias_cfr_rounds() -> int:
    """Parse B4 fill rounds; rounds=1 with nr=1/bin is the recommended start.

    The design estimates that starting point at about 1.33x today's eager-fill
    wall clock.
    """
    raw = os.environ.get("NEXUS_BIAS_CFR_ROUNDS") or "1"
    try:
        rounds = int(raw)
    except ValueError:
        raise ValueError(
            f"NEXUS_BIAS_CFR_ROUNDS={raw!r} is not an integer"
        ) from None
    if not (1 <= rounds <= 64):
        raise ValueError(
            f"NEXUS_BIAS_CFR_ROUNDS={rounds} outside [1, 64]"
        )
    return rounds


def _bias_cfr_nr() -> int:
    """Parse the number of rollouts per bias bin/profile."""
    raw = os.environ.get("NEXUS_BIAS_CFR_NR") or "1"
    try:
        nr = int(raw)
    except ValueError:
        raise ValueError(
            f"NEXUS_BIAS_CFR_NR={raw!r} is not an integer"
        ) from None
    if not (1 <= nr <= 100):
        raise ValueError(
            f"NEXUS_BIAS_CFR_NR={nr} outside [1, 100]"
        )
    return nr


def _sampled_deadline_ms() -> int:
    """Parse the optional relative sampled-solve deadline in milliseconds."""
    raw = os.environ.get("NEXUS_SAMPLED_DEADLINE_MS") or "0"
    try:
        deadline_ms = int(raw)
    except ValueError:
        raise ValueError(
            f"NEXUS_SAMPLED_DEADLINE_MS={raw!r} is not an integer"
        ) from None
    if not (0 <= deadline_ms <= 0x7FFFFFFFFFFFFFFF):
        raise ValueError(
            "NEXUS_SAMPLED_DEADLINE_MS must fit a nonnegative int64, "
            f"got {deadline_ms}"
        )
    return deadline_ms


class CpuEsSolver:
    """Isolated CPU-ES engine. Constructed once at provider init (loads a second
    ~12GB blueprint through es_resolver_real.dll); thread-safe per-solve (each
    solve builds its own tree + table + CSolver; the only shared state is the
    read-only blueprint handle)."""

    def __init__(self, blueprint_path: str) -> None:
        self._ready = False
        self._blueprint = None
        self._bp_handle = None
        self._rollout_c = None  # isolated module instance
        self._CSolver = None
        self._snapshot_tree = None
        self._extract_leaf_info = None
        self._build_labels = None
        self._StreetSolverGPU = None
        self._get_street_solver_lib = None
        # Trained postflop bet vocab from the .bps manifest (bet-vocab desync
        # fix). Set in _init from the loaded blueprint's manifest footer.
        self._train_postflop_sizes: Optional[List[float]] = None
        self._train_subsequent_sizes: Optional[List[float]] = None
        self._cev_holder = None  # A3: kept alive iff NEXUS_A3_CONT_EV=1
        self._last_cont_ev_stats = None  # A3: {hits, attempts} from last solve
        self._last_cont_ev_turn_stats = None  # dual-boundary: turn {hits,attempts}
        # Unit 3 (mean-field bias-action): semantics the loaded .cev was walked
        # under ("mean_field"|"uniform"), read from <cev>.cev.meta.json; absent
        # meta => "uniform" (back-compat with pre-Unit-2 artifacts).
        self._cev_bias_semantics = "uniform"
        # {selects, semantics} snapshot from the last bias-action solve.
        self._last_cont_ev_bias_action_stats = None
        # NEXUS_BR_GAP_EXPOSE: {tree, table, sigma}; readout="both" adds
        # sigmas={avg, final} while keeping legacy sigma=avg.
        self._last_br_gap_objs = None
        self._last_solve_stats = None  # per-solve observability, incl. Gap 4
        self._last_preflight_stats = None  # measurement guard, set before fill
        # NEXUS_LEAF_ROLLOUT: {leaf_fills, cache_hits, rollout_sims, oracle_errors}
        # from the last lazy-oracle solve (es_real_oracle_stats2). None on the
        # eager (default) path — _last_solve_stats.leaf_mode disambiguates.
        self._last_leaf_rollout_stats = None
        # NEXUS_LEAF_SAMPLED: fresh per-visit oracle counters from the last
        # sampled solve. None on every other leaf-value path.
        self._last_sampled_stats = None
        # NEXUS_LEAF_BIAS_CFR: table-backed continuation-bias oracle counters
        # from the last marginal/joint solve. None on every other path.
        self._last_bias_oracle_stats = None
        self._init(blueprint_path)

    def _init(self, blueprint_path: str) -> None:
        py_dir = os.path.join(_POKER_SOLVER, "python")
        if py_dir not in sys.path:
            sys.path.insert(0, py_dir)
        # The live solver modules live in python/engine/ (2026-06 repo cleanup).
        # The two private isolated copies below are loaded by explicit FILE PATH
        # (not import), so they must point at the engine/ subdir.
        engine_dir = os.path.join(py_dir, "engine")

        # Force c_solver to load es_resolver_real.dll (rollout + bps + ES in one).
        os.environ["ES_DLL"] = _ES_DLL_NAME
        from es_resolver.c_solver import _lib as es_lib, _HAS_REAL, CSolver  # type: ignore
        if not _HAS_REAL:
            raise RuntimeError(
                "es_resolver_real.dll lacks the real oracle/construct — rebuild it"
            )
        self._CSolver = CSolver
        self._es_lib = es_lib

        # ── Isolated rollout_c bound to es_resolver_real.dll ──
        # Load a SECOND copy of the rollout_c module under a private name so its
        # `_LIB` global points at es_resolver_real.dll WITHOUT mutating the
        # shared rollout_c the GPU provider uses.
        spec = importlib.util.spec_from_file_location(
            "rollout_c_es", os.path.join(engine_dir, "rollout_c.py")
        )
        rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rc)
        rc._LIB = es_lib
        rc._setup_ctypes(es_lib)
        self._rollout_c = rc

        # ── Isolated BPSReaderC loaded THROUGH es_resolver_real.dll ──
        # Load a second copy of bps_reader_py whose _find_lib returns es_lib, so
        # the BPSReader* + the construct + the ES all live in one DLL (the
        # validated config) and the shared bps_reader_py is untouched.
        spec2 = importlib.util.spec_from_file_location(
            "bps_reader_py_es", os.path.join(engine_dir, "bps_reader_py.py")
        )
        bpm = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(bpm)
        bpm._find_lib = lambda: es_lib  # type: ignore
        t0 = time.time()
        self._blueprint = bpm.BPSReaderC(blueprint_path)
        self._bp_handle = self._blueprint._reader
        logger.info("CpuEsSolver: blueprint loaded through %s in %.1fs",
                    _ES_DLL_NAME, time.time() - t0)

        # Tree builder + helpers (shared, read-only use).
        from engine.street_solver_gpu import (  # type: ignore
            StreetSolverGPU,
            _build_labels_at_node,
            _get_lib,
        )
        from es_resolver.es_solver import snapshot_tree  # type: ignore
        from engine.leaf_values import extract_leaf_info_from_tree  # type: ignore
        self._StreetSolverGPU = StreetSolverGPU
        self._build_labels = _build_labels_at_node
        self._get_street_solver_lib = _get_lib
        self._snapshot_tree = snapshot_tree
        self._extract_leaf_info = extract_leaf_info_from_tree

        # ── Bet-vocab desync fix (2026-07-01) ──
        # The leaf-history encoder AND the C rollout must enumerate the SAME
        # postflop bet vocab the served blueprint was TRAINED with, or every
        # flop all-in line mis-keys → uniform fallback → over-bet
        # (docs/BET_VOCAB_DESYNC_FINDING_2026-06-26.md). The single source of
        # truth is the loaded .bps manifest footer. Read it once here (fail
        # loud if absent — mirrors strategy_source.py:199-206) and push it into
        # the isolated es_resolver_real rollout via the C setter BEFORE any
        # rollout runs. extract_leaf_info_from_tree gets the same vocab at each
        # solve (see solve()).
        self._train_postflop_sizes, self._train_subsequent_sizes = \
            self._read_manifest_bet_vocab()
        rc.set_rollout_train_vocab(
            self._train_postflop_sizes, self._train_subsequent_sizes)
        logger.info(
            "CpuEsSolver: rollout train-vocab set from manifest — "
            "postflop=%s subsequent=%s",
            self._train_postflop_sizes, self._train_subsequent_sizes)

        # ── A3 (default OFF): load the cont_ev boundary table THROUGH es_lib ──
        # The table pointer must be valid in the SAME DLL that runs the construct
        # rollouts (es_resolver_real), so we bind bp_cont_ev_load_standalone on
        # self._es_lib — NOT via rollout_c.load_cont_ev_table_standalone, which
        # uses _find_blueprint_lib() (the separate mccfr_blueprint.so → a pointer
        # the es_resolver_real consumer cannot read). es_resolver_real compiles
        # mccfr_cont_ev_table.c, so it exports the loader + shares the struct.
        self._cev_holder = None
        if _a3_cont_ev_enabled():
            self._load_cont_ev_table(rc, es_lib)

        self._ready = True

    def _read_manifest_bet_vocab(self) -> Tuple[List[float], List[float]]:
        """Read the trained postflop bet vocab from the loaded .bps manifest.

        The manifest footer (postflop_bet_sizes / subsequent_bet_sizes) is the
        single source of truth for what the served blueprint was trained with.
        Fail loud if postflop_bet_sizes is absent — never fall back to a
        hardcoded constant (mirrors the strategy_source.py:199-206 vocab
        guard). subsequent_bet_sizes defaults to the C-side [1.0] only when the
        manifest predates that field (older exporters), and only after the
        first-raise vocab came from the manifest.
        """
        meta = self._blueprint.get_metadata() or {}
        postflop = meta.get("postflop_bet_sizes")
        if not postflop:
            raise RuntimeError(
                "CpuEsSolver: loaded blueprint manifest has no "
                "'postflop_bet_sizes' — cannot key the leaf encoder/rollout to "
                "the trained vocab. Refusing to fall back to a hardcoded "
                "[0.5,1.0] (bet-vocab desync guard, "
                "docs/BET_VOCAB_DESYNC_FINDING_2026-06-26.md)."
            )
        subsequent = meta.get("subsequent_bet_sizes") or [1.0]
        return ([float(x) for x in postflop], [float(x) for x in subsequent])

    def _load_cont_ev_table(self, rc, es_lib) -> None:
        """A3: load the standalone .cev artifact through es_resolver_real and
        hand its pointer to the isolated rollout consumer. Fail-loud — if the
        flag is on, a missing/unloadable artifact is an error, never a silent
        skip (the experiment must not run against a half-wired engine)."""
        import ctypes

        cev_path = os.environ.get("NEXUS_A3_CEV_PATH", "")
        if not cev_path or not os.path.exists(cev_path):
            raise RuntimeError(
                f"NEXUS_A3_CONT_EV=1 but NEXUS_A3_CEV_PATH is missing/not found: "
                f"{cev_path!r}"
            )
        # Reuse rollout_c's struct mirror so the layout matches the C side
        # exactly (drift here scrambles every slot — see rollout_c.py).
        table = rc._BPContEVTableStandalone()
        ctypes.memset(ctypes.byref(table), 0, ctypes.sizeof(table))
        es_lib.bp_cont_ev_load_standalone.argtypes = [ctypes.c_void_p,
                                                      ctypes.c_char_p]
        es_lib.bp_cont_ev_load_standalone.restype = ctypes.c_int
        t0 = time.time()
        rcode = es_lib.bp_cont_ev_load_standalone(
            ctypes.byref(table), cev_path.encode("utf-8"))
        if rcode != 0:
            raise RuntimeError(
                f"bp_cont_ev_load_standalone({cev_path}) failed rc={rcode} "
                f"(bad magic/version/checksum/truncation)")
        n_slots = int(table.num_entries)
        if n_slots == 0:
            raise RuntimeError(
                f"cont_ev artifact {cev_path} loaded 0 slots — empty-state save")
        # Keep the struct alive for the solver's lifetime (the pointer the
        # consumer reads lives inside it).
        self._cev_holder = table
        table_ptr = ctypes.addressof(table)
        rc.set_rollout_cont_ev_table(table_ptr)
        logger.info(
            "CpuEsSolver A3: cont_ev table loaded through %s in %.1fs "
            "(%d slots) — boundary consumer ARMED (default-off flag is ON)",
            _ES_DLL_NAME, time.time() - t0, n_slots)

        # ── Unit 3 (mean-field): read the .cev meta sidecar for bias_semantics.
        # The Unit-2 walk writes <cev>.cev.meta.json next to the artifact with
        # bias_semantics="uniform"|"mean_field". Absent meta => "uniform" (the
        # legacy artifacts predate the sidecar). The C consumer only exposes the
        # bias-action MODE; the semantics of the values it selects among live in
        # the artifact, so we surface them here and LOG which one is in play
        # whenever bias-action mode is used (mean_field = the real thing;
        # uniform = the hero-best-response-to-uniform half measure).
        self._cev_bias_semantics = self._read_cev_bias_semantics(cev_path)
        logger.info(
            "CpuEsSolver: cont_ev .cev bias_semantics=%s (from %s)",
            self._cev_bias_semantics, cev_path + ".meta.json")

    @staticmethod
    def _read_cev_bias_semantics(cev_path: str) -> str:
        """Read bias_semantics from <cev_path>.meta.json. Returns "mean_field"
        or "uniform"; ABSENT sidecar => "uniform" (back-compat). A present-but-
        malformed sidecar, or an unexpected value, logs a warning and falls back
        to "uniform" — a mislabelled artifact must not silently masquerade as
        mean_field. The naming matches the Unit-2 walk manifest
        (docs/STAGE1_MEANFIELD_PLAN_2026-07-01.md Unit 2 item 5)."""
        meta_path = cev_path + ".meta.json"
        if not os.path.exists(meta_path):
            return "uniform"
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            sem = str(meta.get("bias_semantics", "uniform")).strip().lower()
        except (OSError, ValueError) as e:
            logger.warning(
                "CpuEsSolver: cont_ev meta sidecar %s unreadable (%s) — "
                "treating bias_semantics as uniform", meta_path, e)
            return "uniform"
        if sem not in ("uniform", "mean_field"):
            logger.warning(
                "CpuEsSolver: cont_ev meta sidecar %s has unexpected "
                "bias_semantics=%r — treating as uniform", meta_path, sem)
            return "uniform"
        return sem

    @property
    def ready(self) -> bool:
        return self._ready

    def supports_unequal_stack_abi(self) -> bool:
        """Return whether the complete additive native/Python ABI is loaded."""
        get_street_lib = getattr(self, "_get_street_solver_lib", None)
        c_solver = getattr(self, "_CSolver", None)
        es_lib = getattr(self, "_es_lib", None)
        street_solver = getattr(self, "_StreetSolverGPU", None)
        label_builder = getattr(self, "_build_labels", None)
        root_pot_factory = getattr(
            c_solver, "from_table_with_root_pots", None
        )
        if (
            not callable(get_street_lib)
            or c_solver is None
            or not callable(root_pot_factory)
            or es_lib is None
            or not hasattr(es_lib, "es_create_with_root_pots")
            or not self._callable_accepts_keyword(
                street_solver, "root_stacks_bb"
            )
            or not self._callable_accepts_keyword(
                label_builder, "root_stacks_chips"
            )
            or not self._callable_accepts_keyword(
                root_pot_factory, "root_pot_amounts"
            )
            or not self._callable_accepts_keyword(
                root_pot_factory, "root_pot_eligible_masks"
            )
        ):
            return False
        try:
            return hasattr(get_street_lib(), "ss_build_tree_stacks")
        except Exception:
            logger.info(
                "street-solver unequal-stack ABI probe failed",
                exc_info=True,
            )
            return False

    @staticmethod
    def _callable_accepts_keyword(candidate, keyword: str) -> bool:
        """Conservatively reject a stale Python wrapper around a new DLL."""
        if not callable(candidate):
            return False
        try:
            parameters = inspect.signature(candidate).parameters
        except (TypeError, ValueError):
            return False
        parameter = parameters.get(keyword)
        if parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
        return any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in parameters.values()
        )

    def solve(
        self,
        *,
        board_strs: List[str],
        board_ints: List[int],
        player_ranges: List[List[Tuple[int, int, float]]],
        positions_sorted: List[str],
        player_seats: List[int],
        pot_bb: float,
        stack_bb: float,
        bet_sizes: List[float],
        subsequent_bet_sizes: List[float],
        add_explicit_allin: bool,
        game_action_indices: Optional[List[int]],
        iterations: int,
        num_threads: int = 0,
        cache_k: int = 1,
        num_rollouts: int = 3,
        master_seed: int = 0xE5,
        vec_mode: int = 0,
        readout: str = "avg",
        hero_replay: Optional[Dict[str, object]] = None,
        hero_actual_hand: Optional[Dict[str, object]] = None,
        belief_stash_out: Optional[Dict[str, object]] = None,
        deadline_ms: Optional[int] = None,
        max_projected_fill_bytes: Optional[int] = None,
        progress_callback: Optional[
            Callable[[str, Dict[str, object]], None]
        ] = None,
        root_stacks_bb: Optional[Sequence[float]] = None,
        root_pot_amounts: Optional[Sequence[int]] = None,
        root_pot_eligible_masks: Optional[Sequence[int]] = None,
    ) -> (
        Dict[int, Dict[int, Dict[str, float]]]
        | Dict[str, Dict[int, Dict[int, Dict[str, float]]]]
    ):
        """Run an N-player flop/turn/river CPU-ES solve. Returns all_by_player:
        {player_idx: {hand_idx: {action_label: freq}}} — the SAME structure the
        GPU provider's get_all_avg_strategies produces, so the provider's
        _build_response_strategies maps it identically.

        Builds the tree (CPU; GPU tree-builder, no GPU solve), constructs the
        leaf table (parallel batch = the GPU's exact construct floor, off-lock),
        runs the C ES (Linear CFR, multi-thread Hogwild), extracts each
        player's strategy with the weight-0 filter the GPU applies.

        RIVER: terminals are exact showdown/fold (no depth-limit leaves), so
        the leaf-table construct is skipped (fail-loud bypass below) and
        num_rollouts / cache_k / master_seed have NO effect on the solve —
        there is no leaf-sampling noise on river.

        vec_mode: 0 = scalar traverser (the validated 5p-display reference);
            1 = vectorized traverser. vec-Linear is Gate-1-validated for the
            FINAL-iterate readout ONLY (vec-Linear-AVERAGE is seed-unstable at
            3+ players — VECTOR_TRAVERSER_DESIGN.md). Enforced below.
        readout: "avg" = σ̄ (the frequency-display contract), "final" =
            final iterate (the playing-agent contract; what the EC2 wall-clock
            gate measured: 6p ~0.15%/pot floor in ~22s solve at 96 cores), or
            "both" = one scalar solve with both mappings returned under their
            readout names.
        hero_replay: optional {"player_idx": int, "street_actions": [int,...]}.
            When given, that player's strategy is read at the node reached by
            REPLAYING this street's action indices from the street root —
            the mid-street / faced-bet entry the live game loop needs — not
            at the first-BFS (no-bet) decision node. Fail-loud on any
            replay mismatch (off-tree action / vocab drift).
        hero_actual_hand: optional ``player_idx`` + ``hand_idx`` resolver-row
            identity and ``traverser_fraction``. In scalar ES, only that
            player's traverser pass mixes the modeled range with this actual
            row. Opponent-as-traverser passes retain modeled hero sampling.
        belief_stash_out: optional in/out dict for cross-street belief carry.
            The caller seeds ``entry_slots`` and ``hist_start``. After CFR,
            the dict is replaced with this street's PlainTree + sigma-bar
            readout. ``None`` (the default) performs no stash work.
        """
        if deadline_ms is not None and int(deadline_ms) < 0:
            raise ValueError(f"deadline_ms must be >= 0, got {deadline_ms}")
        if (max_projected_fill_bytes is not None
                and int(max_projected_fill_bytes) < 0):
            raise ValueError(
                "max_projected_fill_bytes must be >= 0, got "
                f"{max_projected_fill_bytes}"
            )
        if vec_mode and readout == "both":
            raise ValueError(
                "vec_mode=1 requires readout='final' — vec-Linear-AVERAGE is "
                "seed-unstable at 3+ players (Gate-1, VECTOR_TRAVERSER_DESIGN.md)"
            )

        # Sampled leaves are scalar-only. Apply their precedence before the
        # normal vec/readout validation so a caller requesting vec + avg is
        # accepted and run as scalar without changing the requested readout.
        _sampled_requested = _leaf_sampled_enabled()
        _rollout_requested = _leaf_rollout_enabled()
        _bias_mode_requested = _leaf_bias_cfr_mode()
        _leaf_bias_mode = _bias_mode_requested
        _bias_rounds = None
        _bias_nr = None
        _sampled_policy = None
        _sampled_deadline = 0
        _sampled_iterations_run = None
        self._last_sampled_stats = None
        self._last_bias_oracle_stats = None
        if _sampled_requested:
            vec_mode = 0
            _sampled_policy = _leaf_sampled_policy()
            _sampled_deadline = _sampled_deadline_ms()
        if _bias_mode_requested == "marginal":
            _bias_rounds = _bias_cfr_rounds()
        if _bias_mode_requested != "off":
            _bias_nr = _bias_cfr_nr()
        if vec_mode and readout != "final":
            raise ValueError(
                "vec_mode=1 requires readout='final' — vec-Linear-AVERAGE is "
                "seed-unstable at 3+ players (Gate-1, VECTOR_TRAVERSER_DESIGN.md)"
            )
        if readout not in ("avg", "final", "both"):
            raise ValueError(
                "readout must be 'avg', 'final', or 'both', "
                f"got {readout!r}"
            )
        if not self._ready:
            raise RuntimeError("CpuEsSolver not initialized")
        # Belief-narrowed root ranges and offline-filled catalog boundary EVs
        # are different fidelity models. Check only on the opt-in stash path:
        # default-OFF solves add no env read or other hot-path work.
        if (belief_stash_out is not None
                and os.environ.get("CONT_EV_CATALOG_MANIFEST")):
            raise RuntimeError(
                "belief carry is incompatible with CONT_EV_CATALOG_MANIFEST: "
                "belief-narrowed root ranges cannot be combined with offline-"
                "filled catalog boundary values; disable one feature")
        _offtree_replay = None
        if hero_replay is not None and "offtree" in hero_replay:
            candidate = hero_replay["offtree"]
            if not isinstance(candidate, dict):
                raise ValueError("hero_replay.offtree must be a dict")
            _offtree_replay = candidate
            if os.environ.get("CONT_EV_CATALOG_MANIFEST"):
                raise RuntimeError(
                    "off-tree add is incompatible with "
                    "CONT_EV_CATALOG_MANIFEST: extended search trees cannot "
                    "be combined with offline-filled catalog boundary "
                    "values; disable one feature")
        rc = self._rollout_c
        n_players = len(player_ranges)
        # LIVE PLAYER CAP (2026-09-09): refuse 4+ player subgames before any
        # tree or fill work; see LivePlayerCountExceeded. The byte guard
        # below stays as the secondary bound.
        _max_live_players = _max_live_resolve_players()
        if n_players > _max_live_players:
            raise LivePlayerCountExceeded(
                f"live re-solve serves at most {_max_live_players} players, "
                f"got {n_players}; 4+ player subgames are refused by policy "
                "is not a served arm)"
            )
        root_geometry_present = tuple(
            value is not None
            for value in (
                root_stacks_bb,
                root_pot_amounts,
                root_pot_eligible_masks,
            )
        )
        if any(root_geometry_present) and not all(root_geometry_present):
            raise ValueError(
                "root_stacks_bb, root_pot_amounts, and "
                "root_pot_eligible_masks must be provided together"
            )
        use_root_geometry = all(root_geometry_present)
        root_stack_values: Tuple[float, ...] = ()
        root_amount_values: Tuple[int, ...] = ()
        root_mask_values: Tuple[int, ...] = ()
        if use_root_geometry:
            if not self.supports_unequal_stack_abi():
                raise ReplayUnsupported(
                    "loaded poker-solver lacks ss_build_tree_stacks and/or "
                    "es_create_with_root_pots"
                )
            if len(board_strs) != 5:
                raise ReplayUnsupported(
                    "unequal-stack root geometry is supported only on river"
                )
            if n_players not in (2, 3):
                raise ReplayUnsupported(
                    "unequal-stack root geometry requires 2 or 3 players"
                )
            root_stack_values = tuple(float(x) for x in root_stacks_bb)
            root_amount_values = tuple(int(x) for x in root_pot_amounts)
            root_mask_values = tuple(int(x) for x in root_pot_eligible_masks)
            if len(root_stack_values) != n_players:
                raise ValueError(
                    "root_stacks_bb length does not match compact players"
                )
            if any(
                not np.isfinite(value) or value < 0.0
                for value in root_stack_values
            ):
                raise ValueError(
                    "root_stacks_bb entries must be finite and non-negative"
                )
            if (
                not root_amount_values
                or len(root_amount_values) != len(root_mask_values)
                or len(root_amount_values) > n_players
            ):
                raise ValueError(
                    "root-pot arrays must have the same length in [1, players]"
                )
            if any(value <= 0 for value in root_amount_values):
                raise ValueError("root-pot amounts must be positive")
            all_players_mask = (1 << n_players) - 1
            if any(
                value <= 0 or value & ~all_players_mask
                for value in root_mask_values
            ):
                raise ValueError(
                    "root-pot masks must contain compact-player bits only"
                )
            if sum(root_amount_values) != int(round(float(pot_bb) * 100.0)):
                raise ValueError(
                    "root-pot amounts must sum to the street-root pot"
                )
        _actual_player_idx: Optional[int] = None
        _actual_hand_idx: Optional[int] = None
        _actual_fraction = 0.0
        if hero_actual_hand is not None:
            if not isinstance(hero_actual_hand, dict):
                raise ValueError("hero_actual_hand must be a dict")
            try:
                _actual_player_idx = int(hero_actual_hand["player_idx"])
                _actual_hand_idx = int(hero_actual_hand["hand_idx"])
                _actual_fraction = float(
                    hero_actual_hand["traverser_fraction"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "hero_actual_hand requires integer player_idx/hand_idx "
                    "and numeric traverser_fraction"
                ) from exc
            if not (0 <= _actual_player_idx < n_players):
                raise ValueError(
                    "hero_actual_hand.player_idx outside solver players: "
                    f"{_actual_player_idx} for {n_players} players"
                )
            if not (
                0 <= _actual_hand_idx
                < len(player_ranges[_actual_player_idx])
            ):
                raise ValueError(
                    "hero_actual_hand.hand_idx outside player range: "
                    f"{_actual_hand_idx} for player {_actual_player_idx}"
                )
            if (
                not np.isfinite(_actual_fraction)
                or not 0.0 <= _actual_fraction <= 1.0
            ):
                raise ValueError(
                    "hero_actual_hand.traverser_fraction must be finite and "
                    f"within [0, 1], got {_actual_fraction!r}"
                )
        scale = 100.0
        stack_chips = int(round(stack_bb * scale))
        if len(board_strs) not in (3, 4, 5):
            raise ValueError(
                f"board must be 3 (flop), 4 (turn) or 5 (river) cards, got "
                f"{len(board_strs)}"
            )
        street = {3: "flop", 4: "turn", 5: "river"}[len(board_strs)]

        # Sampled mode is a distinct leaf-value strategy even on a native
        # zero-leaf river/chance tree, where no sampled context is constructed.
        # Reject incompatible operator intent before the tree shape can bypass
        # these guards.
        if _sampled_requested:
            if _bias_mode_requested != "off":
                raise RuntimeError(
                    "NEXUS_LEAF_SAMPLED=1 is incompatible with "
                    "NEXUS_LEAF_BIAS_CFR: these are two distinct leaf-value "
                    "strategies; enable exactly one.")
            if _rollout_requested:
                raise RuntimeError(
                    "NEXUS_LEAF_SAMPLED=1 is incompatible with "
                    "NEXUS_LEAF_ROLLOUT=1; enable exactly one leaf oracle")
            if _a3_cont_ev_enabled():
                raise RuntimeError(
                    "NEXUS_LEAF_SAMPLED=1 is incompatible with "
                    "NEXUS_A3_CONT_EV=1")
            if _a3_cont_ev_turn_enabled():
                raise RuntimeError(
                    "NEXUS_LEAF_SAMPLED=1 is incompatible with "
                    "NEXUS_A3_CONT_EV_TURN=1")
            if _cont_ev_bias_action_enabled():
                raise RuntimeError(
                    "NEXUS_LEAF_SAMPLED=1 is incompatible with "
                    "NEXUS_CONT_EV_BIAS_ACTION=1")
            if _cont_ev_miss_fallback_enabled():
                raise RuntimeError(
                    "NEXUS_LEAF_SAMPLED=1 is incompatible with "
                    "NEXUS_CONT_EV_MISS_FALLBACK=1")
            if os.environ.get("CONT_EV_CATALOG_MANIFEST"):
                raise RuntimeError(
                    "NEXUS_LEAF_SAMPLED=1 is incompatible with "
                    "CONT_EV_CATALOG_MANIFEST")
            _sampled_cev_would_arm = (
                self._cev_holder is not None
                and ((street == "flop" and n_players >= 3)
                     or (street == "turn" and n_players >= 3
                         and _a3_cont_ev_turn_enabled()))
            )
            if _sampled_cev_would_arm:
                raise RuntimeError(
                    "NEXUS_LEAF_SAMPLED=1 is incompatible with the "
                    "loaded/armed cont_ev boundary")
            _sampled_cond_on = os.environ.get(
                "NEXUS_LEAF_RANGE_CONDITIONING", "1")[:1] not in (
                    "0", "f", "F", "n", "N")
            if _sampled_cond_on:
                raise RuntimeError(
                    "NEXUS_LEAF_SAMPLED=1 requires "
                    "NEXUS_LEAF_RANGE_CONDITIONING=0: sampled leaves "
                    "bypass the eager per-leaf range-conditioning construct")

        if _bias_mode_requested != "off":
            # ── INTERACTION GUARDS: bias-table CFR is its own eager leaf-value
            # strategy. Combining it with another leaf oracle or an armed
            # cont_ev boundary would silently discard one requested strategy.
            # PRECEDENCE: neither wins — the operator must enable exactly one.
            if _rollout_requested:
                raise RuntimeError(
                    "NEXUS_LEAF_BIAS_CFR is incompatible with "
                    "NEXUS_LEAF_ROLLOUT=1: these are two distinct leaf-value "
                    "strategies; enable exactly one.")
            _bias_cev_would_arm = (
                self._cev_holder is not None
                and ((street == "flop" and n_players >= 3)
                     or (_a3_cont_ev_turn_enabled()
                         and street == "turn" and n_players >= 3))
            )
            if _bias_cev_would_arm:
                raise RuntimeError(
                    "NEXUS_LEAF_BIAS_CFR is incompatible with the armed "
                    "cont_ev boundary (NEXUS_A3_CONT_EV / "
                    "NEXUS_A3_CONT_EV_TURN): these are two distinct "
                    "leaf-value strategies; enable exactly one.")
            if _cont_ev_bias_action_enabled():
                raise RuntimeError(
                    "NEXUS_LEAF_BIAS_CFR is incompatible with "
                    "NEXUS_CONT_EV_BIAS_ACTION: these are two distinct "
                    "leaf-value strategies; enable exactly one.")
            if _cont_ev_miss_fallback_enabled():
                raise RuntimeError(
                    "NEXUS_LEAF_BIAS_CFR is incompatible with "
                    "NEXUS_CONT_EV_MISS_FALLBACK: these are two distinct "
                    "leaf-value strategies; enable exactly one.")
            if _bias_mode_requested == "joint" and n_players != 2:
                raise RuntimeError(
                    "NEXUS_LEAF_BIAS_CFR=joint requires n_players == 2")
            _bias_cond_on = os.environ.get(
                "NEXUS_LEAF_RANGE_CONDITIONING", "1")[:1] not in (
                    "0", "f", "F", "n", "N")
            if (_bias_mode_requested == "joint" and _bias_cond_on
                    and n_players == 2 and street in ("flop", "turn")):
                raise RuntimeError(
                    "NEXUS_LEAF_BIAS_CFR=joint at 2p flop/turn requires "
                    "NEXUS_LEAF_RANGE_CONDITIONING=0: the joint16 enumerator "
                    "has no per-leaf range variant, and v1 will not silently "
                    "drop conditioning.")

        _base_bet_sizes = list(bet_sizes)
        _base_subsequent_sizes = list(subsequent_bet_sizes)
        _event_fractions: Dict[int, float] = {}
        _expected_event_amounts: Optional[Dict[int, int]] = None
        if _offtree_replay is not None:
            try:
                _base_bet_sizes = [
                    float(value) for value in
                    _offtree_replay["base_bet_sizes"]  # type: ignore[index]
                ]
                _base_subsequent_sizes = [
                    float(value) for value in
                    _offtree_replay["base_subsequent_bet_sizes"]  # type: ignore[index]
                ]
                _event_fractions = {
                    int(step): float(fraction)
                    for step, fraction in dict(
                        _offtree_replay["event_fractions"]  # type: ignore[index]
                    ).items()
                }
                _raw_expected_amounts = _offtree_replay.get(
                    "expected_event_amounts"
                )
                if _raw_expected_amounts is not None:
                    _expected_event_amounts = {
                        int(step): int(amount)
                        for step, amount in dict(
                            _raw_expected_amounts
                        ).items()
                    }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid hero_replay.offtree descriptor: {exc}"
                ) from exc
            if not _event_fractions:
                raise ValueError(
                    "hero_replay.offtree requires at least one event fraction"
                )
            if (
                _expected_event_amounts is not None
                and set(_expected_event_amounts) != set(_event_fractions)
            ):
                raise ValueError(
                    "hero_replay.offtree expected_event_amounts steps must "
                    "match event_fractions steps"
                )

        # ── 2p-TURN: Pluribus-DEPTH-LIMITED by default (2026-06-18) ──
        # The 2p turn now terminates at a cont-strat LEAF at the RIVER boundary
        # (POKER_TURN_DEPTH_LIMIT=1, the C default), exactly like the flop — a
        # sub-second tree solved live by the conditioned sampled construct. NO
        # special env is needed for the default path.
        #
        # The legacy in-tree turn->river reach-CFR extension (the iter-hungry
        # research path — EC2 probe 2026-06-17: correct/converging but
        # under-converged at interactive iters, a muddy ~0.65-check strategy worse
        # than lookup) is OPT-IN via NEXUS_ALLOW_CHANCE_SOLVE. It needs BOTH
        # POKER_SEARCH_DEPTH=full (enable the depth rule) AND POKER_TURN_DEPTH_LIMIT
        # =0 (so the turn root extends to the river instead of depth-limiting), set
        # for THIS build only and restored in finally so they never leak.
        # DEFAULT ON since 2026-09-09: the 2p turn extends through the river
        # in-tree (sampled river per iteration, hero line frozen, cross-street
        # contributions charged at terminals). The depth-limited turn's leaf
        # table was 3-rollout noise CFR best-responded to (sets folded to a
        # turn raise at 1.00). NEXUS_ALLOW_CHANCE_SOLVE=0 restores the
        # depth-limited turn for measurement.
        _allow_chance_turn = os.environ.get(
            "NEXUS_ALLOW_CHANCE_SOLVE", "1")[:1] in ("1", "t", "T", "y", "Y")
        _want_turn_reach = (n_players == 2 and street == "turn"
                            and _allow_chance_turn)
        _depth_prev = os.environ.get("POKER_SEARCH_DEPTH")
        _tdl_prev = os.environ.get("POKER_TURN_DEPTH_LIMIT")
        _depth_set = False
        # CONCURRENCY GUARD (2026-06-18): os.environ is process-global and solves
        # run in asyncio.to_thread worker threads. Hold _ENV_MUTATION_LOCK across
        # the mutate -> build -> restore window so a concurrent solve can never
        # observe a half-set env (or have its restore race ours). Taken ONLY on
        # this opt-in path; the default path (the overwhelming majority of solves)
        # never sets env, never acquires, never serializes. Released in finally.
        if _want_turn_reach and _depth_prev is None and _tdl_prev is None:
            _ENV_MUTATION_LOCK.acquire()
            try:
                os.environ["POKER_SEARCH_DEPTH"] = "full"
                os.environ["POKER_TURN_DEPTH_LIMIT"] = "0"
                _depth_set = True
            except BaseException:
                _ENV_MUTATION_LOCK.release()
                raise
        street_geometry_kwargs = {}
        if use_root_geometry:
            street_geometry_kwargs = {
                "root_stacks_bb": root_stack_values,
            }
        try:
            # ── Build the tree (CPU tree-builder; no GPU solve / no leaf_value_fn) ──
            gpu = self._StreetSolverGPU(
                board=board_strs, pot_bb=pot_bb, stack_bb=stack_bb,
                player_ranges=player_ranges, bet_sizes=bet_sizes,
                subsequent_bet_sizes=subsequent_bet_sizes,
                use_cont_strats=(not use_root_geometry), leaf_value_fn=None,
                add_explicit_allin=add_explicit_allin,
                **street_geometry_kwargs,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            if use_root_geometry:
                raise ReplayUnsupported(
                    f"unequal-stack river tree construction failed: {exc}"
                ) from exc
            raise
        finally:
            # Restore immediately after the build (the env only matters during
            # ss_build_tree2); keep the window minimal. Restore THEN release the
            # lock so the next opt-in solve sees the restored env.
            if _depth_set:
                try:
                    if _depth_prev is None:
                        os.environ.pop("POKER_SEARCH_DEPTH", None)
                    else:
                        os.environ["POKER_SEARCH_DEPTH"] = _depth_prev
                    if _tdl_prev is None:
                        os.environ.pop("POKER_TURN_DEPTH_LIMIT", None)
                    else:
                        os.environ["POKER_TURN_DEPTH_LIMIT"] = _tdl_prev
                finally:
                    _ENV_MUTATION_LOCK.release()
        try:
            if len(bet_sizes) > 1 or len(subsequent_bet_sizes) > 1:
                # ss_build_tree2 permits more children than the fixed-width
                # ES solver layout can hold. Validate the native tree before
                # any CSolver is constructed; LiveResolveSource can then drop
                # an unsafe optional extension and re-solve with base sizes.
                validate_extended_tree(gpu._tree)
            tree = self._snapshot_tree(gpu._tree, player_ranges)
            if use_root_geometry:
                # PlainTree is intentionally an open Python sidecar.  Replay
                # and label helpers need the same per-actor cap as the native
                # tree, while every scalar tree remains byte-for-byte shaped
                # as before.
                tree.root_stacks_chips = tuple(
                    int(round(value * 100.0))
                    for value in root_stack_values
                )
                tree.root_pot_amounts = root_amount_values
                tree.root_pot_eligible_masks = root_mask_values
            # ── HERO-LINE FREEZING (Pluribus safe re-solving, 2026-09-09) ──
            # The subgame is rooted at the street start, but the served node
            # can be several actions deep (facing a jam after a check-raise).
            # Without freezing, villain's nodes on hero's real line are only
            # trained when self-play wanders there, so villain still jams too
            # wide after 100k iterations and hero calls with air. Freezing
            # hero's OWN earlier decisions on this street (never the served
            # node) makes every iteration walk the real line. The pairs come
            # from the same replay walk the readout uses; the C resolver
            # consumes SSTreeData.frozen_action. NEXUS_FREEZE_HERO_LINE=0
            # disables it for measurement.
            _frozen_steps: Tuple[Tuple[int, int], ...] = ()
            _freeze_hero_line = os.environ.get(
                "NEXUS_FREEZE_HERO_LINE", "1")[:1] in ("1", "t", "T", "y", "Y")
            if (_freeze_hero_line and _offtree_replay is not None
                    and hero_replay is not None
                    and hero_replay.get("street_actions")
                    and hasattr(gpu, "set_frozen_action_indices")):
                try:
                    _pre_walk = replay_offtree_street_node(
                        tree,
                        list(hero_replay["street_actions"]),  # type: ignore[arg-type]
                        n_players=n_players,
                        base_bet_sizes=_base_bet_sizes,
                        base_subsequent_bet_sizes=_base_subsequent_sizes,
                        bet_sizes=bet_sizes,
                        subsequent_bet_sizes=subsequent_bet_sizes,
                        add_explicit_allin=add_explicit_allin,
                        event_fractions=_event_fractions,
                        expected_event_amounts=_expected_event_amounts,
                    )
                    _frozen_steps = _pre_walk.hero_frozen_steps(
                        int(hero_replay["player_idx"])  # type: ignore[arg-type]
                    )
                except ReplayUnsupported:
                    # The post-solve walk raises the same error with the
                    # served-path context; do not duplicate it here.
                    _frozen_steps = ()
                if _frozen_steps:
                    gpu.set_frozen_action_indices(_frozen_steps)
            _chance_sampled = False
            max_h = gpu._max_hands

            # IN-TREE CHANCE NODE GATE (2026-06-14, narrowed 2026-06-15).
            # Background: the OP3 §4 depth rule once extended a HEADS-UP (2p) flop
            # tree THROUGH a chance node, enumerating the full turn+river runout
            # (873k nodes). The CPU-ES chance handling was first a degenerate stub,
            # then fixed with full-per-hand-REACH vector CFR (correct) — but the
            # 873k-node flop tree is 20-90 min on CPU (not interactive), and that
            # full extension is NOT what Pluribus does.
            #
            # PLURIBUS-FAITHFUL (2026-06-15, paper-verified, Science 2019 supp):
            # the FLOP subgame terminates at a continuation-strategy LEAF at the
            # turn chance boundary — NO in-tree chance node — for ALL player
            # counts. poker-solver ss_tree_builder now depth-limits the 2p flop by
            # DEFAULT (POKER_FLOP_DEPTH_LIMIT=1): the 2p flop is a ~21-node
            # cont-strat-leaf betting tree (sub-ms), IDENTICAL in shape to the
            # already-shipping 3p+ flop path → it solves LIVE here with NO chance
            # node and NO gate. (POKER_FLOP_DEPTH_LIMIT=0 restores the 873k
            # extension for research.)
            #
            # So this gate now ONLY fires for trees that STILL contain a chance
            # node. The ONLY such tree the engine builds by default is the 2p TURN
            # subgame (we set POKER_SEARCH_DEPTH=full for it above; the turn
            # extends to river per the paper's "all other cases"). The 2p turn
            # reach-CFR solve is correct + converging (es_exploitability_2p drops
            # monotonically) BUT iter-hungry: at the interactive budget it is
            # UNDER-converged and gives a muddy non-polarized strategy worse than
            # the blueprint lookup (EC2 probe 2026-06-17). So it is OPT-IN only
            # (NEXUS_ALLOW_CHANCE_SOLVE=1 sets POKER_SEARCH_DEPTH=full above to
            # build the chance node); without the opt-in NO chance node is built
            # for the 2p turn and we route it to the blueprint LOOKUP (below).
            #
            # A chance node here therefore means EITHER the 2p-turn opt-in fired
            # (n_players==2 + reach available -> served by reach-CFR), OR an
            # operator forced POKER_SEARCH_DEPTH=full at a player count with NO
            # validated reach path -> gated to lookup. _chance_is_reach mirrors the
            # use_reach predicate below.
            _has_chance = int(getattr(gpu._tree, "num_chance_nodes", 0)) > 0
            # NEXUS_ES_REACH default OFF since 2026-09-09: full-reach CFR on
            # the river-extended tree costs ~1.3 s per iteration at 1128 rows
            # (270 s for 200 iterations); the sampled vec path is the live one.
            _reach_default = os.environ.get("NEXUS_ES_REACH", "0")[:1] in (
                "1", "t", "T", "y", "Y")
            _chance_is_reach = (_has_chance and n_players == 2 and _reach_default)
            _allow_chance = os.environ.get(
                "NEXUS_ALLOW_CHANCE_SOLVE", "1")[:1] in ("1", "t", "T", "y", "Y")
            if _has_chance and not _chance_is_reach and not _allow_chance:
                raise ReplayUnsupported(
                    "live re-solve tree contains an in-tree chance node at a "
                    "player count with no validated reach-CFR path (reach is "
                    "heads-up only); chance-node solving is gated off for it (set "
                    "NEXUS_ALLOW_CHANCE_SOLVE=1 to force the sampling+enum chance "
                    "path), so this decision is served from the blueprint lookup "
                    "instead"
                )

            # 2p TURN is now Pluribus-DEPTH-LIMITED (POKER_TURN_DEPTH_LIMIT=1,
            # default, 2026-06-18): it builds a cont-strat-LEAF tree at the river
            # boundary with NO chance node, exactly like the flop, and solves LIVE
            # via the same conditioned sampled construct below (sub-second). The
            # old "turn -> lookup" gate is gone. The reach-CFR full river extension
            # remains available behind NEXUS_ALLOW_CHANCE_SOLVE (sets
            # POKER_TURN_DEPTH_LIMIT=0 -> chance node -> the gate above), but it is
            # the iter-hungry research path, not the default.

            n_leaf_nodes = int(gpu._tree.num_leaf_nodes)
            n_leaves = int(gpu._tree.num_leaves)
            # The eager sampled fill materializes a float32 output table with
            # shape [n_leaves, n_players, max_h] and a float64 accumulator of
            # the same shape: 12 bytes per cell at peak.  num_samples_k changes
            # how many samples are accumulated, not either allocation's shape.
            # Expose the tree-derived projection before the first allocation so
            # measurement drivers can abort one unsafe arm without killing the
            # box. The callback/guard are default-off production no-ops.
            projected_fill_bytes = (
                int(n_leaves) * int(n_players) * int(max_h) * 12
            )
            self._last_preflight_stats = {
                "n_leaves": int(n_leaves),
                "n_players": int(n_players),
                "max_hands": int(max_h),
                "bytes_per_leaf_player_hand": 12,
                "projected_fill_bytes": int(projected_fill_bytes),
            }
            if progress_callback is not None:
                progress_callback("pre_fill", dict(self._last_preflight_stats))
            # Log the projection on EVERY solve. The 2026-09-03 runaway could
            # not be attributed afterwards because the box recorded only a job
            # id and no shape; this line makes the next one self-describing.
            logger.info(
                "CpuEsSolver pre-fill: %dp %s n_leaves=%d max_hands=%d "
                "projected_fill_bytes=%d guard=%s",
                n_players, street, n_leaves, max_h, projected_fill_bytes,
                max_projected_fill_bytes
                if max_projected_fill_bytes is not None else "off",
            )
            if max_projected_fill_bytes is None:
                max_projected_fill_bytes = _max_projected_fill_bytes_default()
            if (max_projected_fill_bytes is not None
                    and projected_fill_bytes > int(max_projected_fill_bytes)):
                raise ProjectedFillMemoryExceeded(
                    "projected eager-fill buffer "
                    f"{projected_fill_bytes} bytes exceeds guard "
                    f"{int(max_projected_fill_bytes)} bytes "
                    f"(n_leaves={n_leaves}, n_players={n_players}, "
                    f"max_hands={max_h}, bytes_per_cell=12)"
                )
            # ZERO-LEAF trees terminate in SHOWDOWN/FOLD only and need NO leaf
            # table — the C ES settles those natively. Two cases:
            #   (a) RIVER: the builder disables cont-strat leaves at is_river.
            #   (b) A 2p FLOP/TURN tree EXTENDED through an in-tree CHANCE node to
            #       the next street (the OP3 depth rule): the extension deals the
            #       remaining board and terminates at showdown/fold, so it ALSO
            #       has zero leaves (verified: flop 873k-node + turn 5349-node 2p
            #       chance trees both have 0 leaf nodes). Before reach mode this
            #       path raised "no leaf infos extracted" (the chance tree has no
            #       leaves to extract) — so a 2p chance solve crashed in construct
            #       BEFORE the solver even ran (EC2 2026-06-14). Route any zero-leaf
            #       tree through the empty-table path; the table oracle exists but
            #       is provably unreachable (no leaf node references it).
            tree_has_chance0 = int(getattr(gpu._tree, "num_chance_nodes", 0)) > 0
            if street == "river" or (n_leaf_nodes == 0 and n_leaves == 0
                                     and tree_has_chance0):
                # FAIL-LOUD on the river contract: a depth-limit leaf in a river
                # tree would read garbage from the empty table. (For the chance
                # branch we already gated on n_leaf_nodes==0, so it cannot fire.)
                if street == "river" and (n_leaf_nodes != 0 or n_leaves != 0):
                    raise RuntimeError(
                        f"river tree has {n_leaf_nodes} leaf nodes / "
                        f"{n_leaves} leaves — river terminals must all be "
                        "showdown/fold (builder contract violated; refusing "
                        "to solve with an empty leaf table)"
                    )
                table = np.zeros((0, n_players, max_h), dtype=np.float32)
                construct_s = 0.0
                # Zero-leaf trees (river / in-tree-chance extension) have NO
                # depth-limit leaves to roll out, so NEXUS_LEAF_ROLLOUT does
                # not apply here — the empty table drives the native
                # showdown/fold settle. Record the eager mode + no rollout so
                # the construct site and stats below stay well-defined.
                leaf_mode = "table_eager"
                _leaf_rollout = False
                _leaf_sampled = False
                _leaf_bias_mode = "off"
                _oracle_arrays = None
                self._last_leaf_rollout_stats = None
            else:
                # Encode the per-leaf flop history under the TRAINED manifest
                # vocab (bet-vocab desync fix) — NOT the subgame/tree-build
                # sizes and NOT the module default [0.5,1.0]. The C rollout was
                # already pinned to the same vocab at _init via the setter, so
                # the encoded indices and the rollout's classify agree.
                leaf_infos = self._extract_leaf_info(
                    gpu._tree, num_players=n_players,
                    postflop_bet_sizes=self._train_postflop_sizes,
                    subsequent_bet_sizes=self._train_subsequent_sizes)
                if not leaf_infos:
                    raise RuntimeError("no leaf infos extracted from tree")

                # ── LAZY per-leaf ROLLOUT oracle (NEXUS_LEAF_ROLLOUT, default
                # OFF) ──────────────────────────────────────────────────────
                # When OFF (_leaf_rollout False) NOTHING below changes: the
                # eager arming + compute_subgame_leaf_values_sampled_c fill
                # runs exactly as shipped (byte-identical). When ON we SKIP the
                # eager fill entirely and construct CSolver.from_real_oracle at
                # the solver-construction site — the C oracle rolls out only the
                # leaves the CFR walk visits (the ~27min 6p exhaustive-fill
                # bypass; the fill is >=99.7% of solve time). See
                # _leaf_rollout_enabled() for the full contract.
                _leaf_rollout = _rollout_requested
                _leaf_sampled = _sampled_requested
                if _leaf_sampled:
                    leaf_mode = "sampled_per_visit"
                elif _leaf_rollout:
                    leaf_mode = "rollout_lazy"
                elif _leaf_bias_mode == "marginal":
                    leaf_mode = "table_eager_bias4"
                elif _leaf_bias_mode == "joint":
                    leaf_mode = "table_eager_joint16"
                else:
                    leaf_mode = "table_eager"
                self._last_leaf_rollout_stats = None

                if _leaf_rollout:
                    # ── INTERACTION GUARDS: rollout mode BYPASSES the eager
                    # per-leaf construct, so any flag that modifies THAT
                    # construct would be SILENTLY DROPPED. Fail loud rather
                    # than half-apply (AGENTS.md: explicit raises, no silent
                    # divergence). ──
                    #
                    # (1) cont_ev boundary arming (NEXUS_A3_CONT_EV[_TURN],
                    # bias-action, miss-fallback). These arm C globals that the
                    # EAGER rollout's leaf machinery consumes at the flop/turn
                    # boundary. from_real_oracle drives a DIFFERENT C rollout
                    # path (es_real_oracle.c) that does not consult those
                    # globals — so an armed cont_ev boundary would be a no-op
                    # while the caller believes it is active. The catalog /
                    # cont_ev routing layer and lazy-rollout serving are
                    # mutually exclusive leaf-value strategies; refuse the
                    # ambiguous combination. PRECEDENCE: neither wins — the
                    # operator must pick one.
                    _cev_would_arm = (
                        self._cev_holder is not None
                        and ((street == "flop" and n_players >= 3)
                             or (_a3_cont_ev_turn_enabled()
                                 and street == "turn" and n_players >= 3))
                    )
                    if _cev_would_arm:
                        raise RuntimeError(
                            "NEXUS_LEAF_ROLLOUT=1 is incompatible with the "
                            "armed cont_ev boundary (NEXUS_A3_CONT_EV / "
                            "NEXUS_A3_CONT_EV_TURN): the lazy real-oracle "
                            "rollout bypasses the eager leaf construct the "
                            "cont_ev consumer hooks into, so the boundary "
                            "table would be silently ignored. These are two "
                            "distinct leaf-value strategies — enable exactly "
                            "one.")
                    if _cont_ev_bias_action_enabled():
                        raise RuntimeError(
                            "NEXUS_LEAF_ROLLOUT=1 is incompatible with "
                            "NEXUS_CONT_EV_BIAS_ACTION (it only affects an "
                            "armed cont_ev boundary, which rollout mode "
                            "bypasses).")
                    if _cont_ev_miss_fallback_enabled():
                        raise RuntimeError(
                            "NEXUS_LEAF_ROLLOUT=1 is incompatible with "
                            "NEXUS_CONT_EV_MISS_FALLBACK (cont_ev-only "
                            "measurement flag; rollout mode has no cont_ev "
                            "boundary).")
                    # (2) 2p leaf range-conditioning feeds the EAGER per-leaf
                    # construct ONLY (it builds per_leaf_ranges consumed by
                    # compute_subgame_leaf_values_sampled_perleaf_c). The
                    # real-oracle build takes no per-leaf range argument, so
                    # conditioning cannot be applied. It DEFAULTS ON at 2p
                    # flop/turn — reject the combination loud rather than serve
                    # an un-conditioned (over-bet) 2p answer that the operator
                    # believes is conditioned. Explicit NEXUS_LEAF_RANGE_
                    # CONDITIONING=0 is required to run rollout mode at 2p.
                    _cond_on = os.environ.get(
                        "NEXUS_LEAF_RANGE_CONDITIONING", "1")[:1] not in (
                            "0", "f", "F", "n", "N")
                    if _cond_on and n_players == 2 and street in ("flop", "turn"):
                        raise RuntimeError(
                            "NEXUS_LEAF_ROLLOUT=1 at 2p flop/turn requires "
                            "NEXUS_LEAF_RANGE_CONDITIONING=0: villain-range "
                            "conditioning feeds the eager per-leaf construct "
                            "only and cannot be applied to the real-oracle "
                            "rollout, so leaving it ON (the default) would "
                            "silently serve the un-conditioned over-bet.")

                    # Build the per-leaf int32 state arrays the real oracle
                    # copies (mirrors poker-solver run_real_spot_c
                    # ._extract_leaf_state — the validated caller of this
                    # oracle). pot/bets/flop_history come from the SAME
                    # leaf_infos the eager path uses; ACTIVE is read per-leaf
                    # from the LEAF SSNodes (nd.type==4, active_players).
                    #
                    # FIDELITY NOTE (validate on EC2): the eager fill passes a
                    # GLOBAL active=[True]*n_players to the sampled construct,
                    # whereas the real oracle uses this PER-LEAF active. This is
                    # the documented (poker-solver run_real_spot_c:141-145)
                    # semantic difference between the two oracles — the
                    # per-leaf active is the more correct quantity (it does not
                    # deal folded players into the runout); it is NOT a bug in
                    # this wiring, but it means rollout-mode leaf values are not
                    # bit-identical to the eager table. Flagged for the EC2 A/B.
                    _oracle_arrays = self._build_real_oracle_leaf_state(
                        gpu._tree, leaf_infos, n_players, player_seats)
                    table = None  # no eager table; from_real_oracle at construct
                    # construct_s is measured as ~0 by the empty fill block
                    # below (the real leaf-fill cost is LAZY — folded into
                    # solve_s as the CFR walk visits leaves).

                # ── A3 (default OFF): arm the cont_ev boundary consumer for
                # THIS construct iff the flag is on, the table is loaded, and
                # this is a flop multiway spot (the consumer's own predicate).
                # When off, none of these globals are touched. ──
                if _leaf_sampled:
                    _oracle_arrays = self._build_real_oracle_leaf_state(
                        gpu._tree, leaf_infos, n_players, player_seats)
                    table = None

                a3_armed = (
                    self._cev_holder is not None
                    and street == "flop"
                    and n_players >= 3
                )
                if a3_armed:
                    rc.reset_cont_ev_boundary_hit_count()
                    rc.set_use_cont_ev_at_flop_boundary(True)

                # ── Dual-boundary (2026-06-25, default OFF): arm the cont_ev
                # TURN boundary for THIS construct iff NEXUS_A3_CONT_EV_TURN=1,
                # the table is loaded, and this is a multiway TURN spot. The
                # turn arm is INDEPENDENT of the flop arm (separate flag,
                # separate counters); enabling it does NOT change the flop
                # default. Stays OFF in the served path until the equity gate
                # passes + user opt-in. ──
                a3_turn_armed = (
                    self._cev_holder is not None
                    and _a3_cont_ev_turn_enabled()
                    and street == "turn"
                    and n_players >= 3
                )
                if a3_turn_armed:
                    rc.reset_cont_ev_turn_boundary_hit_count()
                    rc.set_use_cont_ev_at_turn_boundary(True)

                # ── Unit 3 (mean-field, default OFF): arm the bias-as-hero-
                # action leaf semantics iff NEXUS_CONT_EV_BIAS_ACTION=1 AND some
                # boundary is armed for THIS construct. It only changes how an
                # armed boundary combines its 4 stored EVs (regret-matched
                # mixture vs uniform mean). Reset the per-infoset regret store
                # before the construct so it never leaks across solves; disarmed
                # + snapshotted in the finally below. When off, the C global is
                # never touched (byte-identical legacy path). ──
                bias_action_armed = (
                    (a3_armed or a3_turn_armed)
                    and _cont_ev_bias_action_enabled()
                    and hasattr(rc, "set_cont_ev_bias_action")
                )
                if bias_action_armed:
                    rc.reset_cont_ev_bias_regrets()
                    rc.set_cont_ev_bias_action(True)

                # ── Stage-2' pinned-tail MISS fallback (MEASUREMENT ONLY,
                # default OFF): armed iff a boundary is armed AND the env flag
                # is set AND the .so has the symbol. rc.set_cont_ev_miss_
                # fallback raises on a pre-Stage-2' .so (fail-loud: a
                # measurement that asked for fallback must not silently run
                # fail-loud and crash the gate an hour in). When off, the C
                # global is never touched (byte-identical legacy path). ──
                miss_fallback_armed = (
                    (a3_armed or a3_turn_armed)
                    and _cont_ev_miss_fallback_enabled()
                )
                if miss_fallback_armed:
                    rc.reset_cont_ev_fallback_counts()
                    rc.set_cont_ev_miss_fallback(True)

                # DIAG (NEXUS_ES_DIAG=1): reset the rollout blueprint-miss counters
                # before the construct so we can read the miss-rate after. A high
                # blueprint miss-rate -> the turn/river CONTINUATION falls back to
                # UNIFORM random play, which over-rewards BETTING (betting into a
                # randomly-playing villain looks great) -> the QJs over-bet. This
                # is the suspected cause (memory solve_overbet_rootcause_turn_centroids).
                _diag = os.environ.get("NEXUS_ES_DIAG", "")[:1] in ("1","t","T","y","Y")
                if _diag and hasattr(rc, "reset_rollout_observability"):
                    rc.reset_rollout_observability()

                # ── 2p OPPONENT-RANGE CONDITIONING (2026-06-17, gated) ──
                # The depth-limited cont-strat-leaf values the BET leaf against
                # the UN-conditioned full preflop range -> betting marginal hands
                # (QJs) is under-punished -> over-bet vs the blueprint's confident
                # check (root cause: docs/CPU_ES_CHANCE_NODE_FINDINGS). Condition
                # each leaf's villain sampling weights on the blueprint probability
                # that the seat took THIS street's actions on the leaf's line, so
                # the BET leaf is valued vs the CALLING subset. Gated to 2p
                # FLOP/TURN ONLY (both are now Pluribus-depth-limited cont-strat-
                # leaf trees, 2026-06-18); the 3p+ and 5-6p external-sampling paths
                # take the UNCHANGED shared-range construct (provably no regression).
                #
                # DEFAULT ON (2026-06-17, EC2-validated, user flip): QJs flop check
                # OFF=0.002 (over-bet) -> ON=0.97-0.99 (matches blueprint 0.957),
                # clean (nodes_conditioned=1498, zero miss/fallback), faster
                # (1.5s vs 2.7s). Explicit NEXUS_LEAF_RANGE_CONDITIONING=0 opts out.
                _cond_on = os.environ.get(
                    "NEXUS_LEAF_RANGE_CONDITIONING", "1")[:1] not in (
                        "0", "f", "F", "n", "N")
                _use_cond = (_cond_on and n_players == 2
                             and street in ("flop", "turn"))
                per_leaf_ranges = None
                if _use_cond:
                    from engine.leaf_villain_conditioning import (  # type: ignore
                        build_conditioned_per_leaf_ranges)
                    _cond_street = 1 if street == "flop" else 2
                    per_leaf_ranges, _cond_stats = build_conditioned_per_leaf_ranges(
                        leaf_infos=leaf_infos, player_ranges=player_ranges,
                        board=board_ints, player_seats=list(player_seats),
                        game_action_history=list(game_action_indices or []),
                        blueprint=self._blueprint, n_players=n_players,
                        street=_cond_street,
                    )
                    if _diag:
                        print(f"CpuEsSolver DIAG: range-conditioning ({street}) stats="
                              f"{_cond_stats}", file=sys.stderr, flush=True)

                    # CENTROID-ABSENCE GUARD (2026-06-18, review critic finding).
                    # If conditioning silently no-op'd because the bucketizer could
                    # not bucket ANY hand — the dominant failure mode is the TURN
                    # when turn_centroids.bin was dropped from the deploy (memory:
                    # turn_centroids has been omitted from CPU userdata before),
                    # making hand_to_bucket_turn return -1 for every hand -> every
                    # multiplier 1.0 -> the conditioning becomes a no-op and the
                    # answer reverts to the UN-conditioned over-bet, but still ships
                    # with full "live solve" provenance. There is no other reason a
                    # whole solve would condition zero nodes while attempting it, so
                    # surface it LOUD. We log a WARNING rather than hard-fail the
                    # request: a degraded (un-conditioned) answer is still a valid
                    # solve, and crashing a live request on a deploy misconfig is
                    # worse than serving it with a screaming log line for ops.
                    _nc = int(_cond_stats.get("nodes_conditioned", 0))
                    _bm = int(_cond_stats.get("bucket_miss", 0))
                    if _nc == 0 and _bm > 0:
                        logger.warning(
                            "CpuEsSolver: %dp %s range-conditioning conditioned 0 "
                            "nodes with %d bucket misses — the bucketizer returned "
                            "no bucket for any hand (turn_centroids.bin missing for "
                            "a turn solve?). Conditioning is a SILENT NO-OP and this "
                            "answer reverts to the un-conditioned (over-bet) leaf "
                            "values. Check the deploy's blueprint/centroid artifacts.",
                            n_players, street, _bm)

                # ── Construct the leaf table (parallel batch, off-lock) ──
                # In LAZY rollout mode the eager fill is SKIPPED entirely
                # (table stays None, set above); the try/finally below is a
                # no-op (all cont_ev disarm branches are gated on *_armed which
                # are False in rollout mode — the interaction guards above
                # already rejected any armed-cont_ev + rollout combination), so
                # construct_s stays 0.0 and we go straight to construction.
                t0 = time.time()
                try:
                    if _leaf_rollout or _leaf_sampled:
                        pass  # leaf oracle is constructed at the solver site
                    elif _leaf_bias_mode == "marginal":
                        if per_leaf_ranges is not None:
                            table = rc.compute_subgame_leaf_values_bias4_perleaf(
                                leaf_infos=leaf_infos, board=board_ints,
                                street=street, player_hands=player_ranges,
                                per_leaf_player_hands=per_leaf_ranges,
                                active=[True] * n_players,
                                acting_order=list(range(n_players)),
                                player_seats=list(player_seats),
                                game_action_history=list(
                                    game_action_indices or []),
                                blueprint=self._blueprint,
                                starting_pot_chips=int(
                                    round(pot_bb * scale)),
                                stack_chips=stack_chips, max_hands=max_h,
                                num_rollouts=_bias_nr,
                                num_rounds=_bias_rounds,
                                master_seed=master_seed,
                            )
                        else:
                            table = rc.compute_subgame_leaf_values_bias4(
                                leaf_infos=leaf_infos, board=board_ints,
                                street=street, player_hands=player_ranges,
                                active=[True] * n_players,
                                acting_order=list(range(n_players)),
                                player_seats=list(player_seats),
                                game_action_history=list(
                                    game_action_indices or []),
                                blueprint=self._blueprint,
                                starting_pot_chips=int(
                                    round(pot_bb * scale)),
                                stack_chips=stack_chips, max_hands=max_h,
                                num_rollouts=_bias_nr,
                                num_rounds=_bias_rounds,
                                master_seed=master_seed,
                            )
                    elif _leaf_bias_mode == "joint":
                        table = rc.compute_subgame_leaf_values_c(
                            leaf_infos=leaf_infos, board=board_ints,
                            street=street, player_hands=player_ranges,
                            active=[True] * n_players,
                            acting_order=list(range(n_players)),
                            player_seats=list(player_seats),
                            game_action_history=list(
                                game_action_indices or []),
                            blueprint=self._blueprint,
                            starting_pot_chips=int(round(pot_bb * scale)),
                            stack_chips=stack_chips, max_hands=max_h,
                            num_rollouts=_bias_nr,
                            master_seed=master_seed,
                        )
                    elif per_leaf_ranges is not None:
                        table = rc.compute_subgame_leaf_values_sampled_perleaf_c(
                            leaf_infos=leaf_infos, board=board_ints, street=street,
                            player_hands=player_ranges,
                            per_leaf_player_hands=per_leaf_ranges,
                            active=[True] * n_players,
                            acting_order=list(range(n_players)),
                            player_seats=list(player_seats),
                            game_action_history=list(game_action_indices or []),
                            blueprint=self._blueprint,
                            starting_pot_chips=int(round(pot_bb * scale)),
                            stack_chips=stack_chips, max_hands=max_h,
                            num_rollouts=num_rollouts, num_samples_k=cache_k,
                            master_seed=master_seed,
                        )
                    else:
                        table = rc.compute_subgame_leaf_values_sampled_c(
                            leaf_infos=leaf_infos, board=board_ints, street=street,
                            player_hands=player_ranges, active=[True] * n_players,
                            acting_order=list(range(n_players)),
                            player_seats=list(player_seats),
                            game_action_history=list(game_action_indices or []),
                            blueprint=self._blueprint,
                            starting_pot_chips=int(round(pot_bb * scale)),
                            stack_chips=stack_chips, max_hands=max_h,
                            num_rollouts=num_rollouts, num_samples_k=cache_k,
                            master_seed=master_seed,
                        )
                finally:
                    if a3_armed:
                        # Disable the global flag so it can't leak into other
                        # solves, then snapshot the counters for this solve.
                        rc.set_use_cont_ev_at_flop_boundary(False)
                        hits = rc.get_cont_ev_boundary_hit_count()
                        attempts = rc.get_cont_ev_boundary_attempt_count()
                        self._last_cont_ev_stats = {
                            "hits": int(hits),
                            "attempts": int(attempts),
                            "hit_rate": (hits / attempts) if attempts else 0.0,
                        }
                        if miss_fallback_armed:
                            self._last_cont_ev_stats["fallbacks"] = int(
                                rc.get_cont_ev_fallback_count())
                        logger.info(
                            "CpuEsSolver A3: cont_ev boundary hits=%d attempts=%d "
                            "hit_rate=%.4f", hits, attempts,
                            self._last_cont_ev_stats["hit_rate"])
                    if a3_turn_armed:
                        # Symmetric disarm + snapshot for the TURN boundary.
                        # MUST happen in finally so an exception cannot leave
                        # the turn flag enabled for a subsequent solve.
                        rc.set_use_cont_ev_at_turn_boundary(False)
                        t_hits = rc.get_cont_ev_turn_boundary_hit_count()
                        t_att = rc.get_cont_ev_turn_boundary_attempt_count()
                        self._last_cont_ev_turn_stats = {
                            "hits": int(t_hits),
                            "attempts": int(t_att),
                            "hit_rate": (t_hits / t_att) if t_att else 0.0,
                        }
                        if miss_fallback_armed:
                            self._last_cont_ev_turn_stats["fallbacks"] = int(
                                rc.get_cont_ev_turn_fallback_count())
                        logger.info(
                            "CpuEsSolver dual-boundary: cont_ev TURN hits=%d "
                            "attempts=%d hit_rate=%.4f", t_hits, t_att,
                            self._last_cont_ev_turn_stats["hit_rate"])
                    if miss_fallback_armed:
                        # Symmetric disarm — MUST happen in finally so the
                        # measurement-only fallback can never leak into a
                        # subsequent (potentially served) solve.
                        rc.set_cont_ev_miss_fallback(False)
                        logger.info(
                            "CpuEsSolver Stage-2': pinned-tail fallback "
                            "flop=%d turn=%d (of attempts flop=%d turn=%d)",
                            rc.get_cont_ev_fallback_count(),
                            rc.get_cont_ev_turn_fallback_count(),
                            (self._last_cont_ev_stats or {}).get("attempts", 0),
                            (self._last_cont_ev_turn_stats or {}).get("attempts", 0))
                    if bias_action_armed:
                        # Symmetric disarm + snapshot for the bias-action mode.
                        # MUST happen in finally so an exception cannot leave the
                        # mode enabled for a subsequent solve. Reset the regret
                        # store so its memory is freed and no state leaks across
                        # solves. Log which .cev semantics the mode ran under
                        # (mean_field = the real thing; uniform = the hero-best-
                        # response-to-uniform half measure — both legal, must be
                        # visible).
                        rc.set_cont_ev_bias_action(False)
                        selects = rc.get_cont_ev_bias_action_stats()
                        rc.reset_cont_ev_bias_regrets()
                        self._last_cont_ev_bias_action_stats = {
                            "selects": int(selects),
                            "bias_semantics": self._cev_bias_semantics,
                        }
                        logger.info(
                            "CpuEsSolver Unit3: cont_ev bias-action selects=%d "
                            "bias_semantics=%s%s", selects,
                            self._cev_bias_semantics,
                            "" if self._cev_bias_semantics == "mean_field"
                            else " (HERO-BR-TO-UNIFORM half measure — the .cev "
                                 "was NOT walked mean_field)")
                construct_s = time.time() - t0

            if progress_callback is not None:
                progress_callback(
                    "post_fill",
                    {
                        "construct_s": float(construct_s),
                        "projected_fill_bytes": int(projected_fill_bytes),
                    },
                )

            # DIAG (NEXUS_ES_DIAG=1): dump leaf-table stats — isolate the 2p-flop
            # uniform-0.5 mystery (is the construct producing action-blind/zero
            # leaf values?). Cheap; off by default.
            if (os.environ.get("NEXUS_ES_DIAG", "")[:1] in ("1", "t", "T", "y", "Y")
                    and table is not None):
                # table is None in lazy rollout mode (no eager fill) — skip the
                # table dump there; the oracle fill/cache stats are logged from
                # real_oracle_stats() after run() instead.
                try:
                    import numpy as _np
                    _t = _np.asarray(table)
                    print(
                        f"CpuEsSolver DIAG: {n_players}p {street} table shape={_t.shape} "
                        f"min={float(_t.min()) if _t.size else 0.0:.4f} "
                        f"max={float(_t.max()) if _t.size else 0.0:.4f} "
                        f"mean={float(_t.mean()) if _t.size else 0.0:.4f} "
                        f"nonzero={int(_np.count_nonzero(_t)) if _t.size else 0}/{_t.size} "
                        f"iters={iterations}", file=sys.stderr, flush=True)
                    # blueprint MISS-RATE during the construct rollout — the key
                    # over-bet suspect (miss -> uniform continuation -> bet looks
                    # too good).
                    if hasattr(rc, "get_rollout_observability"):
                        _miss, _zs = rc.get_rollout_observability()
                        _bm = rc.get_rollout_bucket_miss_count() if hasattr(
                            rc, "get_rollout_bucket_miss_count") else -1
                        _ps = rc.get_rollout_per_street_miss_counts() if hasattr(
                            rc, "get_rollout_per_street_miss_counts") else None
                        print(f"CpuEsSolver DIAG: rollout blueprint_miss={_miss} "
                              f"zerosum={_zs} bucket_miss={_bm} per_street={_ps}",
                              file=sys.stderr, flush=True)
                except Exception as _e:
                    print(f"CpuEsSolver DIAG: table dump failed: {_e}", file=sys.stderr, flush=True)

            # ── Solve: C ES, Linear CFR, multi-thread Hogwild ──
            if use_root_geometry:
                try:
                    cs = self._CSolver.from_table_with_root_pots(
                        gpu._tree,
                        table,
                        n_players,
                        max_h,
                        root_amount_values,
                        root_mask_values,
                        seed=1,
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise ReplayUnsupported(
                        "unequal-stack river root-pot construction failed: "
                        f"{exc}"
                    ) from exc
            elif _leaf_sampled:
                # Fresh per-visit blueprint rollout. The resolver seed is the
                # sole RNG source for traversal hands, actions, continuation
                # biases, and rollouts, so map the public master_seed here.
                # There is deliberately no cache_k or oracle master_seed.
                _street_code = 1 if street == "flop" else 2
                (_so_n_leaves, _so_pot, _so_bets, _so_active,
                 _so_hist, _so_hist_len, _so_seats) = _oracle_arrays
                cs = self._CSolver.from_sampled_oracle(
                    gpu._tree,
                    self._bp_handle.value,
                    street=_street_code,
                    stack_chips=stack_chips,
                    n_leaves=_so_n_leaves,
                    max_h=max_h,
                    pot=_so_pot, bets=_so_bets, active=_so_active,
                    hist_flat=_so_hist, hist_len=_so_hist_len,
                    player_seats=_so_seats,
                    num_rollouts=num_rollouts,
                    bias_policy=_sampled_policy,
                    seed=master_seed,
                )
            elif _leaf_rollout:
                # LAZY real-blueprint-rollout oracle (NEXUS_LEAF_ROLLOUT=1).
                # Construct the solver over the SAME tree but with the C
                # es_real_oracle instead of the eager table: leaf values are
                # rolled out (blueprint continuation queried live in C) only for
                # the leaves the CFR walk visits, and CACHED on first hit
                # (strategy-independent, averaged over cache_k bias samples x
                # num_rollouts). Parameters are mapped faithfully from what the
                # eager fill uses today: same blueprint handle, street, stack,
                # num_rollouts, cache_k (num_samples_k), master_seed. street as
                # the C oracle's int code (1=flop, 2=turn; rollout does not
                # apply to river / zero-leaf trees — handled above).
                #
                # MODE NOTE (vec/scalar): this is the CACHED all-hands oracle —
                # each leaf's value is filled ONCE and reused, so it is safe in
                # BOTH scalar and vec traverser modes and we do NOT force scalar
                # here (vec_mode flows through unchanged from LiveSolveConfig).
                # The FRESH-BIAS-PER-VISIT variant (the later fidelity pass —
                # re-sampling the villain bias on every leaf visit) is NOT cache-
                # coherent across the vec traverser's simultaneous hands and
                # WILL require forcing scalar; that is out of scope for this
                # routing pass.
                _street_code = 1 if street == "flop" else 2
                (_ro_n_leaves, _ro_pot, _ro_bets, _ro_active,
                 _ro_hist, _ro_hist_len, _ro_seats) = _oracle_arrays
                # NEXUS_LEAF_CACHE_K: measurement knob (fidelity-rider Arm 1)
                # — raises THIS LAZY ORACLE's per-leaf bias-sample count only.
                # Deliberately scoped here, NOT at solve() entry: the eager
                # construct's num_samples_k must stay untouched so an env-set
                # measurement box never silently multiplies the eager anchor
                # (merge review 2026-07-13). Absent env = byte-identical.
                # Bounded [1, 64] with a fail-loud parse (a userdata typo must
                # reject once, clearly — not traceback on every spot).
                _env_k = os.environ.get("NEXUS_LEAF_CACHE_K")
                if _env_k:
                    try:
                        _k = int(_env_k)
                    except ValueError:
                        raise ValueError(
                            f"NEXUS_LEAF_CACHE_K={_env_k!r} is not an integer"
                        ) from None
                    if not (1 <= _k <= 64):
                        raise ValueError(
                            f"NEXUS_LEAF_CACHE_K={_k} outside [1, 64] "
                            "(refusing a silent wall-clock blowup)"
                        )
                    if _k != cache_k:
                        logger.info(
                            "lazy leaf-rollout cache_k %d -> %d "
                            "(NEXUS_LEAF_CACHE_K)", cache_k, _k,
                        )
                    cache_k = _k
                cs = self._CSolver.from_real_oracle(
                    gpu._tree,
                    self._bp_handle.value,  # BPSReader* address (int)
                    street=_street_code,
                    stack_chips=stack_chips,
                    n_leaves=_ro_n_leaves,
                    max_h=max_h,
                    pot=_ro_pot, bets=_ro_bets, active=_ro_active,
                    hist_flat=_ro_hist, hist_len=_ro_hist_len,
                    player_seats=_ro_seats,
                    num_rollouts=num_rollouts,
                    cache_k=cache_k,
                    master_seed=master_seed,
                    seed=1,
                )
            elif _leaf_bias_mode != "off":
                _bias_oracle_mode = (
                    0 if _leaf_bias_mode == "marginal" else 1
                )
                _bias_n_leaves = (
                    int(table.shape[0])
                    if _bias_oracle_mode == 0
                    else int(table.shape[0]) // 16
                )
                cs = self._CSolver.from_bias_table(
                    gpu._tree, table, mode=_bias_oracle_mode,
                    np_=n_players, max_h=max_h, n_leaves=_bias_n_leaves,
                    seed=1,
                )
            else:
                cs = self._CSolver.from_table(
                    gpu._tree, table, n_players, max_h, seed=1)
            if _actual_player_idx is not None and _actual_hand_idx is not None:
                cs.set_traverser_actual_hand(
                    _actual_player_idx,
                    _actual_hand_idx,
                    _actual_fraction,
                )
            cs.set_cfr_mode(1)  # Linear CFR (matches the GPU, ~3x more iter-efficient)
            if _sampled_requested:
                # The C default inherits ES_OPP_ENUMERATE when opp_mode is -1.
                # Pin sampling explicitly so a box-level research env cannot
                # defeat sampled mode's no-opponent-enumeration precedence.
                cs.set_opp_mode(0)

            # In-tree CHANCE handling. A 2p flop/turn tree extends through a chance
            # node (OP3 turn-extension). Single-card-per-iter sampling is high-
            # variance + cross-line-incoherent -> degenerate/oscillating solve
            # (EC2 2026-06-14). For a CORRECT chance solve: ENUMERATE the card
            # outcomes (zero chance-variance, the GPU's full-reach equivalent),
            # ENUMERATE opponent actions (full CFR — removes opp-action sampling
            # variance; only tractable at small player counts), and use the VEC
            # traverser (updates all traverser hands per walk -> no traverser-hand
            # sampling variance). Residual variance = opp-hand sampling only.
            # 3p+ has no chance node so leave its (validated) sampling untouched —
            # opponent enumeration would blow up at high N anyway.
            tree_has_chance = int(getattr(gpu._tree, "num_chance_nodes", 0)) > 0
            # REACH MODE (2026-06-14): for a 2p in-tree-chance tree, prefer the
            # full-per-hand-REACH vector CFR (the GPU's own math, ported to CPU) —
            # it converges to the CORRECT mixed strategy deterministically and is
            # NOT degenerate, unlike the sampling+enum path (which still floors at
            # a pure over-bet on the chance-tree fixed point: turn toy spot 36% of
            # pot exploitable vs reach's 0.84% — validate_reach_cfr.py). Gated to
            # 2p (np==2): full reach is O(nodes*nh) per iter with an O(nh*nh_opp)
            # showdown term, tractable only at small player counts; 5-6p MUST keep
            # sampling and 3p+ trees have no chance node anyway. Off by default
            # (NEXUS_ES_REACH unset/0 falls back to the prior sampling+enum path so
            # this is a no-op until explicitly enabled for the validated 2p case).
            _reach = os.environ.get("NEXUS_ES_REACH", "0")[:1] in (
                "1", "t", "T", "y", "Y")
            use_reach = (
                not _sampled_requested
                and tree_has_chance and n_players == 2 and _reach
                and hasattr(cs, "set_reach_mode")
            )
            if use_reach:
                # NOTE (2026-06-17): reach mode only RUNS for the 2p turn when the
                # operator opted in (NEXUS_ALLOW_CHANCE_SOLVE -> POKER_SEARCH_DEPTH
                # =full -> chance node built). It is correct + converging but
                # iter-hungry; the default 2p turn never reaches here (gated to
                # lookup above). Research / future higher-iter path.
                cs.set_reach_mode(1)    # full reach; ignores cfr/chance/opp/vec
                if readout != "both":
                    # Preserve the established reach-mode σ̄ contract for the
                    # historical single-readout callers.  The new scalar
                    # measurement readout explicitly requests both.
                    readout = "avg"
                vec_mode = 0
            elif tree_has_chance and hasattr(cs, "set_chance_mode"):
                # NEXUS_CHANCE_SAMPLE=1: deal ONE next-street card per
                # iteration (keyed by iteration and chance node, so every
                # visit inside one iteration sees the same runout) instead of
                # enumerating all outcomes. Measured 2026-09-09 on a 2p turn
                # with the river in-tree: enumerate 75 ms/iteration, sample
                # 0.16 ms/iteration at 1128 rows; sampling is the only way
                # the river-extended turn fits a live budget. Unbiased in
                # expectation; convergence comes from iterations.
                _chance_sampled = os.environ.get(
                    "NEXUS_CHANCE_SAMPLE", "1")[:1] in ("1", "t", "T", "y", "Y")
                cs.set_chance_mode(0 if _chance_sampled else 1)
                if (not _sampled_requested and n_players <= 3
                        and hasattr(cs, "set_opp_mode")):
                    if readout == "both":
                        raise ValueError(
                            "readout='both' is unsupported on chance-node trees "
                            "(internal vec forcing would make the avg readout "
                            "vec-Linear-AVERAGE)"
                        )
                    cs.set_opp_mode(1)  # full CFR over opponent actions (2-3p)
                    cs.set_vec_mode(1)  # all traverser hands per walk
                    vec_mode = 1        # force vec for the readout below
            if vec_mode and not _sampled_requested:
                cs.set_vec_mode(1)
            # ── EXACT ALL-IN LEAVES (2026-09-09) ──────────────────────────
            # The eager leaf table indexes by the traverser's hand only, so a
            # betting-closed (all-in) leaf priced hero's hand against the
            # villain's whole leaf range and ignored the hand in the walk;
            # measured: a bare 6-2 flush and ace-high called a 190 jam at
            # 1.00 at pot odds against a random range. With the resolver's
            # exact per-pair oracle enabled, such leaves are the exact
            # expected showdown of the traverser's hand against the sampled
            # opponent hands over every remaining runout. Depth-limit trees
            # only (river trees have no leaves); NEXUS_EXACT_ALLIN_LEAVES=0
            # disables it for measurement. Older resolver builds without the
            # entry point simply keep the table path.
            _exact_allin = (
                not _leaf_sampled and n_leaves > 0
                and os.environ.get("NEXUS_EXACT_ALLIN_LEAVES", "1")[:1]
                in ("1", "t", "T", "y", "Y")
                and hasattr(cs, "set_exact_allin_leaves")
            )
            if _exact_allin:
                cs.set_exact_allin_leaves(True)
            # SMALL-TREE -> SINGLE-THREAD (2026-06-16). Multi-thread Hogwild on a
            # tiny tree is dominated by race noise: the Pluribus-faithful 2p flop
            # is only ~21 nodes / 8 decision nodes, and at num_threads=0 (all
            # cores) racing on those few shared regret arrays produced a DEGENERATE
            # uniform 0.500 mix, while single-thread gave a sensible check ~0.92
            # (EC2 2026-06-16). Hogwild buys nothing on a tree this small (the
            # per-iter work is trivial), so force single-thread below a node
            # threshold — deterministic AND correct. Large trees (873k flop ext,
            # 5-6p) keep multi-thread. Reach mode already runs single-thread.
            _nn = int(getattr(gpu._tree, "num_nodes", 0))
            # Sampled leaves are exempt from the small-tree single-thread
            # force: their per-iteration work is rollout-dominated (~ms per
            # leaf visit vs ~ns table reads), so Hogwild race windows are
            # negligible relative to work and wall-clock scales with threads
            # (measured 2026-07-17: 3p sampled 50k iters = 836s single-thread).
            _eff_threads = (1 if (0 < _nn <= 2000 and not use_reach
                                  and not _leaf_sampled)
                            else num_threads)
            effective_deadline_ms = (
                int(_sampled_deadline) if _leaf_sampled
                else (int(deadline_ms) if deadline_ms is not None else None)
            )
            if effective_deadline_ms is not None:
                cs.set_deadline_ms(effective_deadline_ms)
            if progress_callback is not None:
                progress_callback(
                    "pre_es_run",
                    {"requested_iterations": int(iterations)},
                )
            t1 = time.time()
            cs.run(iterations, num_threads=_eff_threads)
            solve_s = time.time() - t1
            # Preserve the default production path: query the native counter
            # only for sampled mode (historical behavior) or when a measurement
            # caller explicitly supplies deadline_ms (including zero).
            iterations_run_observed = (
                (_leaf_sampled or deadline_ms is not None)
                and hasattr(cs, "iterations_run")
            )
            if iterations_run_observed:
                _sampled_iterations_run = cs.iterations_run()
            else:
                _sampled_iterations_run = int(iterations)
            if progress_callback is not None:
                progress_callback(
                    "post_es_run",
                    {
                        "solve_s": float(solve_s),
                        "iterations_run": int(_sampled_iterations_run),
                    },
                )
            readout_t0 = time.time()
            if readout == "both":
                read_strategies = {
                    "avg": cs.avg_strategy,
                    "final": cs.final_strategy,
                }
                # Legacy optional diagnostics and the legacy BR-gap ``sigma``
                # field still need one reader.  Keep σ̄ as their reference;
                # the BR-gap stash adds both readers separately below.
                read_strategy = read_strategies["avg"]
            else:
                read_strategy = (
                    cs.final_strategy
                    if readout == "final"
                    else cs.avg_strategy
                )
                read_strategies = {readout: read_strategy}

            # BELIEF CARRY STASH (NEXUS_BELIEF_CARRY, default OFF). The output
            # dict is the entire gate: with the default None there is no import,
            # tree copy, sigma-bar read, or metadata access. Extraction must
            # happen here while `cs` (and its strategy sums) is alive. It always
            # reads avg_strategy, independent of the strategy served above.
            if belief_stash_out is not None and use_root_geometry:
                _belief_meta = dict(belief_stash_out)
                belief_stash_out.clear()
                belief_stash_out.update(
                    status="empty",
                    reason=(
                        "belief-stash extraction is unsupported for "
                        "unequal-stack root geometry"
                    ),
                    entry_slots=_belief_meta.get("entry_slots"),
                    hist_start=_belief_meta.get("hist_start"),
                )
            elif belief_stash_out is not None:
                _belief_meta = dict(belief_stash_out)
                try:
                    from app.engine.agent.belief_stash import pack_belief_stash
                    pack_belief_stash(
                        belief_stash_out,
                        cs=cs,
                        tree=tree,
                        player_seats=player_seats,
                        board=board_ints[:len(board_strs)],
                        bet_sizes=bet_sizes,
                        subsequent_bet_sizes=subsequent_bet_sizes,
                        add_explicit_allin=add_explicit_allin,
                    )
                except Exception as _e:
                    # Belief carry is an optional refinement. Never let a
                    # packing failure corrupt or abort the solved strategy.
                    belief_stash_out.clear()
                    belief_stash_out.update(
                        status="empty",
                        reason=("belief-stash extraction failed: "
                                f"{type(_e).__name__}: {_e}"),
                        entry_slots=_belief_meta.get("entry_slots"),
                        hist_start=_belief_meta.get("hist_start"),
                        street=len(board_strs) - 2,
                        board=tuple(board_ints[:len(board_strs)]),
                    )
                    logger.warning(
                        "CpuEsSolver belief-stash extraction failed",
                        exc_info=True,
                    )

            # LAZY rollout oracle: snapshot the fill/cache stats AFTER the run
            # (the leaves were filled on-demand during the walk). None on the
            # eager path — _last_solve_stats.leaf_mode disambiguates. Guarded so
            # a stats read can never break a real solve.
            if _leaf_rollout:
                try:
                    _fills, _hits, _sims, _errs = cs.real_oracle_stats()
                    self._last_leaf_rollout_stats = {
                        "leaf_fills": int(_fills),
                        "cache_hits": int(_hits),
                        "rollout_sims": int(_sims),
                        "oracle_errors": int(_errs),
                    }
                    logger.info(
                        "CpuEsSolver leaf-rollout: %dp %s fills=%d cache_hits=%d "
                        "sims=%d errors=%d", n_players, street, _fills, _hits,
                        _sims, _errs)
                    if _errs:
                        # A nonzero oracle error count means some leaf rollout
                        # failed silently in C (fell back to a default value) —
                        # surface it so a degraded solve is visible to ops.
                        logger.warning(
                            "CpuEsSolver leaf-rollout: %d oracle errors — some "
                            "leaf values fell back to a default in C", _errs)
                except Exception as _e:  # never break a real solve for stats
                    logger.warning("leaf-rollout stats read failed: %s", _e)
                    self._last_leaf_rollout_stats = None

            if _leaf_sampled:
                try:
                    (_visits, _rollouts, _errs, _uniform_visits,
                     _uniform_rollouts, _regret_visits,
                     _regret_rollouts) = cs.sampled_oracle_stats()
                    self._last_sampled_stats = {
                        "leaf_visits": int(_visits),
                        "rollouts": int(_rollouts),
                        "oracle_errors": int(_errs),
                        "uniform_visits": int(_uniform_visits),
                        "uniform_rollouts": int(_uniform_rollouts),
                        "regret_visits": int(_regret_visits),
                        "regret_rollouts": int(_regret_rollouts),
                    }
                    logger.info(
                        "CpuEsSolver sampled-leaf: %dp %s policy=%d "
                        "visits=%d rollouts=%d errors=%d uniform=%d/%d "
                        "regret=%d/%d iterations=%d/%d deadline_ms=%d",
                        n_players, street, _sampled_policy, _visits,
                        _rollouts, _errs, _uniform_visits, _uniform_rollouts,
                        _regret_visits, _regret_rollouts,
                        _sampled_iterations_run, iterations,
                        _sampled_deadline,
                    )
                    if _errs:
                        logger.warning(
                            "CpuEsSolver sampled-leaf: %d oracle errors", _errs)
                except Exception as _e:  # observability must not break a solve
                    logger.warning("sampled-leaf stats read failed: %s", _e)
                    self._last_sampled_stats = None

            if _leaf_bias_mode != "off":
                try:
                    _visits, _errs = cs.bias_oracle_stats()
                    self._last_bias_oracle_stats = {
                        "visits": int(_visits),
                        "errors": int(_errs),
                    }
                    logger.info(
                        "CpuEsSolver bias-table: %dp %s mode=%s visits=%d "
                        "errors=%d", n_players, street, _leaf_bias_mode,
                        _visits, _errs)
                    if _errs:
                        logger.warning(
                            "CpuEsSolver bias-table: %d oracle errors", _errs)
                except Exception as _e:  # observability must not break a solve
                    logger.warning("bias-table stats read failed: %s", _e)
                    self._last_bias_oracle_stats = None

            # BR-GAP EXPOSE (NEXUS_BR_GAP_EXPOSE=1, default OFF). Stash the GENUINE
            # production solve objects (tree snapshot + leaf table + served sigma)
            # so an offline per-spot correctness harness can compute the in-tree
            # best-response gap on EXACTLY what production served — conditioning,
            # small-tree single-thread, the lot — instead of replicating (and
            # risking diverging from) the solve. Pure read-only stash, no behavior
            # change; only allocates when the flag is set. See
            # scripts/measure_per_spot_correctness.py.
            _br_gap_expose = os.environ.get(
                "NEXUS_BR_GAP_EXPOSE", ""
            )[:1] in ("1", "t", "T", "y", "Y")
            if _br_gap_expose and use_root_geometry:
                # Root-pot reach/exploitability CFVs are explicitly outside
                # the additive ABI. Never expose a scalar PlainTree snapshot
                # that an offline harness could score as if it were faithful.
                self._last_br_gap_objs = None
                logger.info(
                    "BR-gap expose unavailable for unequal-stack root geometry"
                )
            elif _br_gap_expose:
                try:
                    from es_resolver.es_solver import snapshot_tree as _snap
                    _ranges = [[(c0, c1, w) for (c0, c1, w) in rng]
                               for rng in player_ranges]
                    _tree_snap = _snap(gpu._tree, _ranges)
                    _sigmas = {}
                    for _readout_name, _strategy_reader in (
                            read_strategies.items()):
                        _readout_sigma = {}
                        for _ni in _tree_snap.decision_node_indices:
                            _a = _strategy_reader(_ni)
                            if _a is not None:
                                _readout_sigma[_ni] = _a
                        _sigmas[_readout_name] = _readout_sigma
                    _sigma = (
                        _sigmas["avg"]
                        if readout == "both"
                        else _sigmas[readout]
                    )
                    # LEAF TABLE for the BR-gap best-responder. The EAGER path has
                    # a materialized `table`; the ROLLOUT path (NEXUS_LEAF_ROLLOUT)
                    # sets table=None (no eager fill — leaf values live in the C
                    # oracle's lazy cache). exploitability() needs a dense
                    # [n_leaves, np, max_h] table either way, so on the rollout
                    # path read the oracle's cache back (fills-all then copies;
                    # see CSolver.get_oracle_leaf_table / es_real_oracle.c). A
                    # None here (older DLL / not a real-oracle solve) is stashed
                    # explicitly with a marker so the harness reports "rollout
                    # table unavailable" rather than crashing on table=None.
                    _br_table = table
                    _br_table_source = "eager"
                    if _leaf_rollout:
                        _br_table = None
                        _br_table_source = "rollout_oracle"
                        try:
                            _br_table = cs.get_oracle_leaf_table()
                        except Exception as _te:  # never break a real solve
                            logger.warning(
                                "BR-gap rollout leaf-table readout failed: %s",
                                _te)
                            _br_table = None
                        if _br_table is None:
                            _br_table_source = "rollout_oracle_unavailable"
                    elif _leaf_bias_mode != "off":
                        _br_table = None
                        _br_table_source = "bias_oracle_effective"
                        try:
                            _br_table = cs.get_bias_effective_table()
                        except Exception as _te:  # never break a real solve
                            logger.warning(
                                "BR-gap bias leaf-table readout failed: %s",
                                _te)
                        if _br_table is None:
                            _br_table_source = (
                                "bias_oracle_effective_unavailable")
                    elif _leaf_sampled:
                        # A fresh per-visit oracle has no coherent dense table
                        # to expose for the optional offline BR-gap harness.
                        _br_table = None
                        _br_table_source = "sampled_oracle_unavailable"
                    _br_gap_objs = {
                        "tree": _tree_snap, "table": _br_table, "sigma": _sigma,
                        "table_source": _br_table_source,
                        "n_players": n_players, "street": street,
                        "pot_chips": int(round(pot_bb * scale)),
                    }
                    if readout == "both":
                        # Additive research-only surface.  Keep the legacy
                        # ``sigma`` key as σ̄ so every existing scorer retains
                        # its exact input shape.
                        _br_gap_objs["sigmas"] = _sigmas
                    self._last_br_gap_objs = _br_gap_objs
                except Exception as _e:  # never break a real solve for the stash
                    logger.warning("BR-gap expose stash failed: %s", _e)
                    self._last_br_gap_objs = None

            # DIAG (NEXUS_ES_DIAG=1): dump the RAW first-decision-node strategy for
            # each player (unfiltered by the freq>0.001 display) + per-hand spread,
            # to see whether the solver produced action-discriminating mixes or a
            # uniform 0.5 (the 2p-flop mystery). Off by default.
            if os.environ.get("NEXUS_ES_DIAG", "")[:1] in ("1", "t", "T", "y", "Y"):
                try:
                    import numpy as _np
                    for _p in range(n_players):
                        _ni = None
                        for _cand in gpu._tree.decision_node_indices:
                            _nd = gpu._tree.nodes[_cand]
                            if _nd.type == 0 and _nd.player == _p and _nd.num_children > 0:
                                _ni = _cand
                                break
                        if _ni is None:
                            continue
                        _sig = read_strategy(_ni)
                        if _sig is None:
                            logger.info("CpuEsSolver DIAG: p%d node %d sig=None", _p, _ni)
                            continue
                        _sig = _np.asarray(_sig)
                        print(
                            f"CpuEsSolver DIAG: p{_p} node={_ni} na={_sig.shape[1]} "
                            f"nh={_sig.shape[0]} colmean={_np.round(_sig.mean(axis=0), 4).tolist()} "
                            f"action0_spread={float(_sig[:, 0].min()):.3f}..{float(_sig[:, 0].max()):.3f}",
                            file=sys.stderr, flush=True)
                except Exception as _e:
                    print(f"CpuEsSolver DIAG: strategy dump failed: {_e}", file=sys.stderr, flush=True)

            # The hero's replayed node (mid-street entry), if requested.
            hero_idx: Optional[int] = None
            hero_node: Optional[int] = None
            hero_projection: Optional[NodeProjection] = None
            _offtree_replayed_through = False
            _offtree_added_child_chips: Dict[int, int] = {}
            if hero_replay is not None and hero_replay.get("street_actions"):
                if _offtree_replay is not None:
                    hero_idx = int(
                        hero_replay["player_idx"]  # type: ignore[arg-type]
                    )
                    _walk = replay_offtree_street_node(
                        tree,
                        list(hero_replay["street_actions"]),  # type: ignore[arg-type]
                        n_players=n_players,
                        base_bet_sizes=_base_bet_sizes,
                        base_subsequent_bet_sizes=_base_subsequent_sizes,
                        bet_sizes=bet_sizes,
                        subsequent_bet_sizes=subsequent_bet_sizes,
                        add_explicit_allin=add_explicit_allin,
                        event_fractions=_event_fractions,
                        expected_event_amounts=_expected_event_amounts,
                    )
                    hero_node = int(_walk.node_index)
                    _offtree_replayed_through = bool(
                        _walk.replayed_through_added
                    )
                    _offtree_added_child_chips = dict(
                        _walk.added_child_chips
                    )
                    reached = self._require_replayed_hero_node(
                        tree, hero_node, hero_idx,
                    )
                    hero_projection = build_node_projection(
                        tree,
                        hero_node,
                        n_players=n_players,
                        base_bet_sizes=_base_bet_sizes,
                        base_subsequent_bet_sizes=_base_subsequent_sizes,
                        bet_sizes=bet_sizes,
                        subsequent_bet_sizes=subsequent_bet_sizes,
                        add_explicit_allin=add_explicit_allin,
                    )
                    expected = hero_replay.get("expected_actions")
                    if (expected is not None
                            and len(hero_projection.base_child_indices)
                            != int(expected)):  # type: ignore[arg-type]
                        raise ReplayUnsupported(
                            "hero node has "
                            f"{len(hero_projection.base_child_indices)} base-"
                            "vocabulary tree actions but the agent's legal "
                            f"set has {expected} — the solved tree cannot "
                            "represent this decision"
                        )
                else:
                    # The replay path is validated for the single-size,
                    # no-explicit-allin vocab only (the 8.05B shape). With
                    # multi-size / explicit-allin vocabs the agent's surviving-
                    # raise count can silently diverge from the tree's child
                    # count (different all-in baselines) and the structural
                    # check cannot distinguish raise children — route to the
                    # lookup fallback rather than risk a wrong-node read
                    # (review 2026-06-10).
                    if (add_explicit_allin or len(bet_sizes) > 1
                            or len(subsequent_bet_sizes) > 1):
                        raise ReplayUnsupported(
                            "street replay is validated only for single-size, "
                            "no-explicit-allin vocabs (got bet_sizes="
                            f"{list(bet_sizes)}, subsequent="
                            f"{list(subsequent_bet_sizes)}, allin="
                            f"{add_explicit_allin})"
                        )
                    hero_idx = int(hero_replay["player_idx"])  # type: ignore[arg-type]
                    hero_node = self._replay_street_node(
                        tree,
                        list(hero_replay["street_actions"]),  # type: ignore[arg-type]
                        n_players=n_players,
                    )
                    reached = self._require_replayed_hero_node(
                        tree, hero_node, hero_idx,
                    )
                    expected = hero_replay.get("expected_actions")
                    if expected is not None and reached.num_children != int(expected):  # type: ignore[arg-type]
                        # The tree's action set at the hero's node differs from
                        # the agent's legal set: either the multiway raise cap
                        # (tree capped at 2 raises, vocab allows a 3rd — the mix
                        # would silently lack it) or a real-chips commitment
                        # prune the canonical tree chips don't have. Both are
                        # KNOWN abstraction limits -> lookup fallback.
                        raise ReplayUnsupported(
                            f"hero node has {reached.num_children} tree actions "
                            f"but the agent's legal set has {expected} — the "
                            "solved tree cannot represent this decision"
                        )

            # ── Extract strategies → all_by_player, with the weight-0 /
            #    never-visited filter the GPU's get_all_avg_strategies applies
            #    (freq>0.001; skip uniform-default rows so out-of-range hands
            #    stay fold-100% placeholders in _build_response_strategies).
            #    The historical paths iterate exactly once; ``both`` reuses the
            #    solved object and runs the same read-only extraction twice. ──
            _hero_row_strategy_sum_mass_by_row: Dict[int, float] = {}
            if _actual_player_idx is not None and _actual_hand_idx is not None:
                coverage_node = (
                    hero_node
                    if (
                        hero_node is not None
                        and hero_idx == _actual_player_idx
                    )
                    else self._first_decision_node_for(
                        tree, _actual_player_idx
                    )
                )
                if coverage_node is None:
                    raise RuntimeError(
                        "actual hero player has no decision node for coverage"
                    )
                _hero_row_strategy_sum_mass_by_row[_actual_hand_idx] = float(
                    cs.strategy_sum_mass(
                        coverage_node, _actual_hand_idx,
                    )
                )

            all_by_readout: Dict[
                str, Dict[int, Dict[int, Dict[str, float]]]
            ] = {}
            for _readout_name, read_strategy in read_strategies.items():
                all_by_player: Dict[int, Dict[int, Dict[str, float]]] = {}
                _hero_merged_mass_by_row: Dict[int, float] = {}
                _hero_dropped_mass_by_row: Dict[int, float] = {}
                _hero_drop_reason_by_row: Dict[int, str] = {}
                _projection_skipped_players: List[Dict[str, object]] = []
                for p_idx in range(n_players):
                    if hero_idx is not None and p_idx == hero_idx:
                        node_idx = hero_node
                    else:
                        node_idx = self._first_decision_node_for(tree, p_idx)
                    raw: Dict[int, Dict[str, float]] = {}
                    if node_idx is not None:
                        if _offtree_replay is not None:
                            # Extension labels come from projected base
                            # children, never the richer child universe.
                            labels: List[str] = []
                        elif hero_idx is not None and p_idx == hero_idx:
                            # Replayed (possibly faced-bet) node: synthesize
                            # labels from the tree's own chip deltas —
                            # _build_labels is wrong off the street root.
                            if use_root_geometry:
                                # The additive binding's helper recognizes
                                # structural short-call slots and uses the
                                # acting player's own root stack.
                                labels = self._build_labels(
                                    gpu._tree,
                                    node_idx,
                                    root_stacks_chips=gpu.root_stacks_chips,
                                )
                            else:
                                labels = self._node_action_labels(
                                    tree, node_idx, n_players
                                )
                        else:
                            if use_root_geometry:
                                labels = self._build_labels(
                                    gpu._tree,
                                    node_idx,
                                    root_stacks_chips=gpu.root_stacks_chips,
                                )
                            else:
                                labels = self._build_labels(
                                    gpu._tree, node_idx
                                )
                        sigma = read_strategy(node_idx)  # [nh, na] or None
                        if _offtree_replay is not None and sigma is not None:
                            try:
                                projection = (
                                    hero_projection
                                    if (hero_idx is not None
                                        and p_idx == hero_idx)
                                    else build_node_projection(
                                        tree,
                                        node_idx,
                                        n_players=n_players,
                                        base_bet_sizes=_base_bet_sizes,
                                        base_subsequent_bet_sizes=(
                                            _base_subsequent_sizes
                                        ),
                                        bet_sizes=bet_sizes,
                                        subsequent_bet_sizes=(
                                            subsequent_bet_sizes
                                        ),
                                        add_explicit_allin=(
                                            add_explicit_allin
                                        ),
                                    )
                                )
                                if projection is None:
                                    raise ReplayUnsupported(
                                        "missing base projection for player "
                                        f"{p_idx} node {node_idx}"
                                    )
                                if (len(projection.base_action_labels)
                                        != len(
                                            projection.base_child_indices
                                        )):
                                    raise ReplayUnsupported(
                                        f"node {node_idx} has "
                                        f"{len(projection.base_child_indices)} "
                                        "base children but "
                                        f"{len(projection.base_action_labels)} "
                                        "base labels"
                                    )
                                labels = list(
                                    projection.base_action_labels
                                )
                                (sigma, _merged_by_row, _dropped_by_row,
                                 _drop_reason_by_row) = (
                                    self._project_offtree_strategy(
                                        sigma, projection
                                    )
                                )
                            except ReplayUnsupported as projection_error:
                                # Only the replayed hero node is load-bearing
                                # for the served mix.  A projection miss at
                                # another player's display node must not
                                # discard an otherwise faithful hero solve.
                                if (hero_idx is not None
                                        and p_idx == hero_idx):
                                    raise
                                _projection_skipped_players.append(
                                    {
                                        "player_idx": int(p_idx),
                                        "node_index": int(node_idx),
                                        "reason": str(projection_error),
                                    }
                                )
                                all_by_player[p_idx] = raw
                                continue
                            if (hero_idx is not None
                                    and p_idx == hero_idx):
                                _hero_merged_mass_by_row = _merged_by_row
                                _hero_dropped_mass_by_row = _dropped_by_row
                                _hero_drop_reason_by_row = _drop_reason_by_row
                        na = sigma.shape[1] if sigma is not None else 0
                        nh = len(player_ranges[p_idx])
                        n_sigma_rows = (
                            sigma.shape[0] if sigma is not None else 0
                        )
                        for h in range(min(nh, n_sigma_rows)):
                            # weight-0 hand → never visited → skip
                            # (fold-100% placeholder)
                            if (
                                player_ranges[p_idx][h][2] <= 0.0
                                and not (
                                    p_idx == _actual_player_idx
                                    and h == _actual_hand_idx
                                )
                            ):
                                continue
                            strat = {}
                            for a in range(min(na, len(labels))):
                                freq = float(sigma[h, a])
                                # Match the shipped readout convention after
                                # projection too: discard numerical dust and
                                # omit all-zero rows.
                                if freq > 0.001:
                                    strat[labels[a]] = freq
                            if strat:
                                raw[h] = strat
                    all_by_player[p_idx] = raw
                all_by_readout[_readout_name] = all_by_player

            solve_result = (
                all_by_readout
                if readout == "both"
                else all_by_readout[readout]
            )

            readout_s = time.time() - readout_t0
            if progress_callback is not None:
                progress_callback(
                    "post_readout", {"readout_s": float(readout_s)}
                )

            # OBSERVABILITY STASH (2026-07-05, latency ladder): construct/solve
            # walls + tree shape for the LAST solve, as instance attributes —
            # the same read-only convention as _last_cont_ev_stats. Offline
            # measurement harnesses read this instead of parsing the log line
            # below (full float precision; num_chance_nodes is the per-arm
            # proof of which POKER_SEARCH_DEPTH mode actually built the tree).
            # Pure stash, no behavior change.
            self._last_solve_stats = {
                "n_players": n_players,
                "street": street,
                # leaf-value strategy for THIS solve: "table_eager" (shipped
                # default: exhaustive fill + from_table), "rollout_lazy"
                # (cached all-hands oracle), "sampled_per_visit", or one of
                # the table-backed bias-CFR modes.
                "leaf_mode": leaf_mode,
                "leaf_rollout_stats": self._last_leaf_rollout_stats,
                "bias_oracle_stats": self._last_bias_oracle_stats,
                "sampled_stats": self._last_sampled_stats,
                "construct_s": float(construct_s),
                "solve_s": float(solve_s),
                "readout_s": float(readout_s),
                "projected_fill_bytes": int(projected_fill_bytes),
                "iterations": int(iterations),
                "iterations_run": int(_sampled_iterations_run),
                "iterations_run_observed": bool(iterations_run_observed),
                "deadline_ms": effective_deadline_ms,
                "num_rollouts": int(num_rollouts),
                "master_seed": int(master_seed),
                "sampled_policy": (
                    int(_sampled_policy) if _leaf_sampled else None
                ),
                "sampled_deadline_ms": (
                    int(_sampled_deadline) if _leaf_sampled else None
                ),
                "vec_mode": int(vec_mode),
                "readout": readout,
                "num_threads": int(num_threads),
                # reach mode ignores num_threads in C (single-threaded pass)
                "eff_threads": 1 if use_reach else int(_eff_threads),
                "use_reach": bool(use_reach),
                "hero_frozen_steps": int(len(_frozen_steps)),
                "chance_sampled": bool(_chance_sampled),
                "exact_allin_leaves": bool(_exact_allin),
                "num_nodes": int(_nn),
                "num_leaf_nodes": int(n_leaf_nodes),
                "num_leaves": int(n_leaves),
                "num_chance_nodes": int(getattr(gpu._tree, "num_chance_nodes", 0)),
                "num_decision_nodes": int(
                    getattr(gpu._tree, "num_decision_nodes", 0)),
                "max_depth": int(getattr(gpu._tree, "max_depth", 0)),
                "hero_traverser_actual_fraction": (
                    float(_actual_fraction)
                    if _actual_player_idx is not None else None
                ),
                "hero_row_strategy_sum_mass": (
                    next(iter(_hero_row_strategy_sum_mass_by_row.values()))
                    if _hero_row_strategy_sum_mass_by_row else None
                ),
                "hero_row_strategy_sum_mass_by_row": dict(
                    _hero_row_strategy_sum_mass_by_row
                ),
                "offtree_added": {
                    "first": [
                        float(value) for value in bet_sizes
                        if float(value) not in _base_bet_sizes
                    ] if _offtree_replay is not None else [],
                    "subsequent": [
                        float(value) for value in subsequent_bet_sizes
                        if float(value) not in _base_subsequent_sizes
                    ] if _offtree_replay is not None else [],
                },
                "offtree_replayed_through": bool(
                    _offtree_replayed_through
                ),
                # CpuEsSolver returns every grid row. LiveResolveSource picks
                # the served hero row and exposes its scalar mass.
                "offtree_hero_merged_mass": 0.0,
                "offtree_hero_merged_mass_by_row": dict(
                    _hero_merged_mass_by_row
                ),
                "offtree_hero_dropped_mass_by_row": dict(
                    _hero_dropped_mass_by_row
                ),
                "offtree_hero_drop_reason_by_row": dict(
                    _hero_drop_reason_by_row
                ),
                "offtree_projection_skipped_players": list(
                    _projection_skipped_players
                ),
                "offtree_added_child_chips": dict(
                    _offtree_added_child_chips
                ),
            }

            logger.info(
                "CpuEsSolver solve: %dp %s construct=%.1fs solve=%.1fs "
                "(%d iters, %s threads, off-lock)",
                n_players, street, construct_s, solve_s,
                (_sampled_iterations_run
                 if _sampled_iterations_run is not None else iterations),
                num_threads or "all",
            )
            return solve_result
        finally:
            gpu._freed = True  # native ss_free_tree hangs on this box; skip it

    def _build_real_oracle_leaf_state(
        self, ss_tree, leaf_infos, n_players: int, player_seats,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """Per-leaf flat int32 state arrays for ``from_real_oracle`` or
        ``from_sampled_oracle``. Returns
        (n_leaves, pot, bets, active, hist_flat, hist_len, player_seats).

        Mirrors poker-solver python/es_resolver/run_real_spot_c.py
        ._extract_leaf_state — the VALIDATED caller of this oracle — but reuses
        the `leaf_infos` the solve already extracted (via the shared
        extract_leaf_info_from_tree) instead of re-extracting. pot / bets /
        flop_history come from those LeafInfo records (the load-bearing
        flop_history_train, encoded under the trained manifest vocab); ACTIVE is
        read PER-LEAF from the LEAF SSNodes (nd.type==4 -> active_players).
        player_seats is the SAME actual-seat vector the eager fill passes (NOT a
        synthetic arange like the run_real_spot_c smoke test), so the rollout's
        seat-relative blueprint lookup matches the served spot.

        The C ctx COPIES all arrays, so the numpy buffers here need only outlive
        the factory call. Both c_solver factories coerce them to C-contiguous
        int32.

        FIDELITY NOTE: the per-leaf active read here is the oracle's contract
        and is the more correct quantity than the eager fill's global
        active=[True]*n_players (which deals folded players into the runout).
        See the construction-site comment; validate the resulting leaf-value
        difference on EC2."""
        n_leaves = int(ss_tree.num_leaves)
        pot = np.zeros(n_leaves, dtype=np.int32)
        bets = np.zeros(n_leaves * n_players, dtype=np.int32)
        active = np.zeros(n_leaves * n_players, dtype=np.int32)
        hist = np.zeros(n_leaves * _ESRO_MAX_FLOP_HIST, dtype=np.int32)
        hist_len = np.zeros(n_leaves, dtype=np.int32)

        # Per-leaf active: first LEAF node for each leaf_idx carries the
        # active_players vector (identical across nodes sharing a leaf_idx).
        leaf_active: Dict[int, List[int]] = {}
        for i in range(int(ss_tree.num_nodes)):
            nd = ss_tree.nodes[i]
            if nd.type == 4 and nd.leaf_idx >= 0 and nd.leaf_idx not in leaf_active:
                leaf_active[nd.leaf_idx] = [
                    int(nd.active_players[p]) for p in range(n_players)]

        for li in leaf_infos:
            idx = int(li.leaf_idx)
            if idx < 0 or idx >= n_leaves:
                continue
            pot[idx] = int(li.pot)
            lb = list(li.bets)
            for p in range(n_players):
                bets[idx * n_players + p] = int(lb[p]) if p < len(lb) else 0
            av = leaf_active.get(idx, [1] * n_players)
            for p in range(n_players):
                active[idx * n_players + p] = int(av[p])
            h = list(li.flop_history_train)
            hl = min(len(h), _ESRO_MAX_FLOP_HIST)
            for k in range(hl):
                hist[idx * _ESRO_MAX_FLOP_HIST + k] = int(h[k])
            hist_len[idx] = hl

        seats = np.ascontiguousarray(list(player_seats), dtype=np.int32)
        if seats.shape[0] != n_players:
            raise RuntimeError(
                f"player_seats has {seats.shape[0]} entries for an "
                f"{n_players}-player spot (real-oracle seat map mismatch)")
        return (n_leaves, pot, bets, active, hist, hist_len, seats)

    def _first_decision_node_for(self, tree, player_idx: int) -> Optional[int]:
        """The first decision node owned by `player_idx` (BFS order), matching
        the GPU's get_all_avg_strategies which returns the first such node."""
        for ni in tree.decision_node_indices:
            nd = tree.nodes[ni]
            if nd.type == 0 and nd.player == player_idx and nd.num_children > 0:
                return ni
        return None

    def _project_offtree_strategy(
        self,
        sigma: np.ndarray,
        projection: NodeProjection,
    ) -> Tuple[np.ndarray, Dict[int, float], Dict[int, float], Dict[int, str]]:
        """Project every strategy row from added children to base columns."""

        from app.engine.agent.offtree_add import merge_added_action_mass

        if sigma.ndim != 2:
            raise ReplayUnsupported(
                f"extended strategy readout has shape {sigma.shape}, expected "
                "a [rows, actions] matrix"
            )
        referenced = list(projection.base_child_indices) + [
            child.child_index for child in projection.added_children
        ]
        if referenced and max(referenced) >= sigma.shape[1]:
            raise ReplayUnsupported(
                f"extended strategy has {sigma.shape[1]} columns but node "
                f"projection references column {max(referenced)}"
            )

        base_positions = [
            int(position) for position, _fraction
            in projection.base_raise_targets
        ]
        base_fractions = [
            float(fraction) for _position, fraction
            in projection.base_raise_targets
        ]
        rows: List[Tuple[float, ...]] = []
        merged_by_row: Dict[int, float] = {}
        dropped_by_row: Dict[int, float] = {}
        drop_reason_by_row: Dict[int, str] = {}
        for row_index in range(int(sigma.shape[0])):
            base_masses = [
                float(sigma[row_index, child_index])
                for child_index in projection.base_child_indices
            ]
            added_actions = [
                (
                    float(child.fraction),
                    float(sigma[row_index, child.child_index]),
                )
                for child in projection.added_children
            ]
            merged = merge_added_action_mass(
                base_masses,
                base_fractions,
                base_positions,
                added_actions,
            )
            rows.append(tuple(float(value) for value in merged.masses))
            merged_by_row[int(row_index)] = float(merged.merged_mass)
            if merged.dropped_mass > 0.0:
                dropped_by_row[int(row_index)] = float(merged.dropped_mass)
            if merged.drop_reason:
                drop_reason_by_row[int(row_index)] = str(
                    merged.drop_reason
                )

        projected = (
            np.asarray(rows, dtype=sigma.dtype)
            if rows
            else np.empty(
                (0, len(projection.base_child_indices)), dtype=sigma.dtype
            )
        )
        return (
            projected,
            merged_by_row,
            dropped_by_row,
            drop_reason_by_row,
        )

    @staticmethod
    def _require_replayed_hero_node(tree, node_idx: int, hero_idx: int):
        """Return the replay target or surface a known tree-shape limit."""
        reached = tree.nodes[node_idx]
        if reached.type != 0 or reached.player != hero_idx:
            raise ReplayUnsupported(
                f"replayed street actions lead to node {node_idx} "
                f"(type={reached.type}, player={reached.player}) — not a "
                f"decision node owned by hero player {hero_idx}. The "
                "recorded street history does not match the solved tree."
            )
        return reached

    def _replay_street_node(
        self,
        tree,
        street_actions: List[int],
        *,
        n_players: int,
    ) -> int:
        """Walk the solved tree from the street root through this street's
        recorded agent action indices; return the reached node index.

        The tree builder (poker-solver src/ss_tree_builder.c:272-370) emits
        children in the TRAINER's action order — [fold iff facing a bet]
        [check/call][raises in size order][explicit all-in last] — which is
        the SAME convention the agent's action indices use
        (legal_categories_chips / _solver_label_to_action_index). So the
        recorded agent index IS the child index. Each step is STRUCTURALLY
        verified against the child's chips/active state (fold → actor
        inactive; check/call → actor matched the max bet; raise → actor
        exceeds it) so a convention drift fails loud instead of silently
        walking the wrong line. (Do NOT map via _build_labels_at_node here:
        its labels are chip-amount-only and wrong at faced-bet nodes —
        found empirically 2026-06-10.)

        Fail-loud (ValueError) on: a non-decision node mid-replay, an
        out-of-range action index (off-tree action — bet-size translation
        is out of scope), or a structural-verification mismatch.
        """
        ni = 0  # street root
        for step, want_idx in enumerate(street_actions):
            nd = tree.nodes[ni]
            if nd.type != 0 or nd.num_children <= 0:
                raise ValueError(
                    f"street replay step {step}: node {ni} is not a decision "
                    f"node (type={nd.type}) — recorded history does not match "
                    "the solved tree"
                )
            want = int(want_idx)
            actor = nd.player
            mx = max(nd.bets[q] for q in range(n_players))
            to_call = mx - nd.bets[actor]
            n_nonraise = 2 if to_call > 0 else 1
            if not (0 <= want < nd.num_children):
                if nd.num_children == n_nonraise and want >= n_nonraise:
                    # The recorded action is a RAISE but this tree node has
                    # no raise children: the tree's raise cap
                    # (SS_MULTIWAY_MAX_RAISES=2 multiway vs the agent/trainer
                    # cap of 3) — a known abstraction limit, NOT drift.
                    # Surface as ReplayUnsupported so the hybrid falls back
                    # to LOOKUP, whose vocab covers the 3rd raise.
                    raise ReplayUnsupported(
                        f"street replay step {step}: recorded raise (index "
                        f"{want}) at a raise-capped tree node {ni} "
                        f"({nd.num_children} children) — tree raise cap is "
                        "below the agent vocab's"
                    )
                raise ValueError(
                    f"street replay step {step}: action index {want} out of "
                    f"range [0,{nd.num_children}) at node {ni} — vocab drift "
                    "or off-tree action"
                )
            child = tree.nodes[nd.children[want]]
            if to_call > 0 and want == 0:
                # fold: actor leaves the hand, chips unchanged.
                if child.active[actor] or child.bets[actor] != nd.bets[actor]:
                    raise ValueError(
                        f"street replay step {step}: child {want} of node {ni} "
                        "does not look like a FOLD — tree child-order "
                        "convention drift (fail loud)"
                    )
            elif want == n_nonraise - 1:
                # check/call: scalar trees match the outstanding max bet;
                # an additive tree may instead emit a short all-in call.
                root_stacks = getattr(tree, "root_stacks_chips", None)
                expected_call_bet = mx
                if root_stacks is not None:
                    remaining = max(
                        int(root_stacks[actor]) - int(nd.bets[actor]),
                        0,
                    )
                    expected_call_bet = int(nd.bets[actor]) + min(
                        int(to_call), remaining
                    )
                if (
                    not child.active[actor]
                    or child.bets[actor] != expected_call_bet
                ):
                    raise ValueError(
                        f"street replay step {step}: child {want} of node {ni} "
                        "does not look like a CHECK/CALL — tree child-order "
                        "convention drift (fail loud)"
                    )
            else:
                # bet/raise: actor exceeds the prior max bet.
                if not child.active[actor] or child.bets[actor] <= mx:
                    raise ValueError(
                        f"street replay step {step}: child {want} of node {ni} "
                        "does not look like a BET/RAISE — tree child-order "
                        "convention drift (fail loud)"
                    )
            ni = nd.children[want]
        return ni

    def _node_action_labels(self, tree, node_idx: int, n_players: int) -> List[str]:
        """Synthesize correct action labels for a decision node from the
        tree's own chip/active deltas — the exact inverse of the tree
        BUILDER's STORAGE convention.

        The production _build_labels_at_node is chip-amount-only and labels
        faced-bet nodes wrongly (a fold reads "Check", a call reads as a
        pot-fraction "Raise") — fine for the display path, which only reads
        street roots, but wrong for a replayed mid-street node.

        Builder convention (review 2026-06-10 — NOT the trainer's): at a
        faced-bet node ss_tree_builder.c:326 computes bet_amount = to_call +
        size*(pot+to_call) and :340 STORES the child level as
        mx + bet_amount (on top of the max bet, not the actor's own bet), so
        the child's delta vs the actor's bet = 2*to_call + size*(pot+to_call)
        and size = (delta - 2*to_call)/(pot + to_call). The all-in clamp is
        applied to bet_amount (= delta - to_call) against the CONSTANT
        street-entry effective stack (:328).
        """
        nd = tree.nodes[node_idx]
        actor = nd.player
        mx = max(nd.bets[q] for q in range(n_players))
        to_call = mx - nd.bets[actor]
        pot = float(nd.pot)
        stack = int(tree.effective_stack)
        labels: List[str] = []
        for c in range(nd.num_children):
            ch = tree.nodes[nd.children[c]]
            if to_call > 0 and c == 0:
                labels.append("Fold")
                continue
            if c == (1 if to_call > 0 else 0):
                labels.append("Call" if to_call > 0 else "Check")
                continue
            delta = int(ch.bets[actor]) - int(nd.bets[actor])
            bet_amount = delta - to_call  # the builder's clamped quantity
            if bet_amount >= stack:
                labels.append("All-in")
                continue
            if to_call > 0:
                frac = (delta - 2 * to_call) / max(pot + to_call, 1.0)
                labels.append(f"Raise {int(round(frac * 100))}%")
            else:
                frac = delta / max(pot, 1.0)
                labels.append(f"Bet {int(round(frac * 100))}%")
        return labels
