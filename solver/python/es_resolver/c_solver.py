"""CSolver — ctypes wrapper around build/es_resolver.dll (the production C ES
re-solver), with the SAME surface as es_solver.ESSolver so the existing Python
validation harnesses (validate_tiny, validate_fixed_profile, validate_vs_gpu)
run against the C solver UNCHANGED by swapping ESSolver(...) -> CSolver(...).

Design (advisor 2026-06-01):
  - The C solver consumes the IDENTICAL SSTreeData the GPU builds — we pass
    ctypes.byref(gpu_solver._tree) straight in. No PlainTree snapshot (that was
    only the Python reference's ctypes-lifetime workaround).
  - The leaf oracle is a C function pointer + ctx (es_oracles.c). NEVER a Python
    callback in the hot path. Deterministic validation factories include:
      from_table(table, np, max_h)  — deterministic table lookup
      joint_zero_sum(strength0, strength1) — 2p zero-sum strength showdown
    For the real-blueprint rollout oracles use from_real_oracle (cached all-
    hands) or from_sampled_oracle (fresh per-visit sampled hands/biases).
  - avg_strategy(node_idx) returns an [nh, na] float64 numpy array (matches
    ESSolver.avg_strategy). The C side fills a flat [nh*na] buffer.

Numerics: the C solver uses float64 regrets/strat-sum (matches the reference).
σ̄ is read back as float64. RNG/float-order differences mean the C and Python
solvers are NOT bit-identical on a stochastic solve — the gate is metric
convergence (BR-gap→0, fixed-profile within 3·SE), not bit equality. The C
joint/table oracles ARE bit-exact, so the deterministic primitives match.
"""
from __future__ import annotations

import ctypes
import operator
import os
import sys
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
# ES_DLL selects which build to load:
#   "es_resolver.dll"      — lightweight, table/joint oracles only (validation)
#   "es_resolver_real.dll" — links rollout + bps_reader; adds cached + sampled
#                            real oracles and BPS loader symbols
# Platform-aware shared-library extension. Windows builds .dll; Linux (the prod
# EC2 target) builds .so. ES_DLL may be given with or without an extension; if
# bare ("es_resolver_real"), the platform extension is appended. NOTE: the Linux
# .so build is NOT yet exercised — the Windows build flags (-Dfsync=_commit
# -include io.h) are Windows-only; Linux has fsync natively, so the Makefile/
# build recipe needs a Linux variant before the prod (Linux) deploy. See
# docs/CPU_ES_RESOLVER_BENCHMARK.md and the deploy note.
_SHLIB_EXT = ".dll" if sys.platform == "win32" else ".so"
_ES_DLL_ENV = os.environ.get("ES_DLL", "es_resolver")
if _ES_DLL_ENV.endswith((".dll", ".so")):
    _DLL_NAME = _ES_DLL_ENV
else:
    _DLL_NAME = _ES_DLL_ENV + _SHLIB_EXT
_DLL_PATH = os.path.join(_REPO, "build", _DLL_NAME)


def _load_dll() -> ctypes.CDLL:
    if not os.path.exists(_DLL_PATH):
        raise FileNotFoundError(
            f"{_DLL_NAME} not found at {_DLL_PATH}. Build it:\n"
            "  gcc -O2 -Wall -Wextra -fopenmp -shared -static -static-libgcc -Isrc "
            "src/es_resolver.c src/es_oracles.c -o build/es_resolver.dll -lm\n"
            "  (real: add es_real_oracle.c rollout.c bps_reader.c card_abstraction.c "
            "mccfr_*.c, -w -Dfsync=_commit -include io.h -o build/es_resolver_real.dll)"
        )
    return ctypes.CDLL(_DLL_PATH)


_lib = _load_dll()
_HAS_REAL = hasattr(_lib, "es_make_real_oracle")
_HAS_SAMPLED = hasattr(_lib, "es_make_sampled_oracle")
_HAS_ROOT_POTS = hasattr(_lib, "es_create_with_root_pots")
_HAS_EXACT_ALLIN_LEAVES = hasattr(_lib, "es_set_exact_allin_leaves")
_BIAS_SYMBOLS = (
    "es_make_bias_oracle", "es_bias_oracle_fn", "es_free_bias_oracle",
    "es_bias_oracle_stats", "es_bias_oracle_get_effective_table",
)
_HAS_BIAS = all(hasattr(_lib, name) for name in _BIAS_SYMBOLS)
# void* es_create(const SSTreeData*, ESLeafOracle, void* ctx, uint64_t seed)
_lib.es_create.restype = ctypes.c_void_p
_lib.es_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
if _HAS_ROOT_POTS:
    _lib.es_create_with_root_pots.restype = ctypes.c_void_p
    _lib.es_create_with_root_pots.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]
_lib.es_free.argtypes = [ctypes.c_void_p]
if hasattr(_lib, "es_set_cfr_mode"):
    _lib.es_set_cfr_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
if hasattr(_lib, "es_set_vec_mode"):
    _lib.es_set_vec_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
if _HAS_EXACT_ALLIN_LEAVES:
    _lib.es_set_exact_allin_leaves.restype = None
    _lib.es_set_exact_allin_leaves.argtypes = [ctypes.c_void_p, ctypes.c_int]
if hasattr(_lib, "es_set_traverser_actual_hand"):
    _lib.es_set_traverser_actual_hand.restype = ctypes.c_int
    _lib.es_set_traverser_actual_hand.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double,
    ]
if hasattr(_lib, "es_set_chance_mode"):
    _lib.es_set_chance_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
if hasattr(_lib, "es_set_opp_mode"):
    _lib.es_set_opp_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
if hasattr(_lib, "es_set_reach_mode"):
    _lib.es_set_reach_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
if hasattr(_lib, "es_set_deadline_ms"):
    _lib.es_set_deadline_ms.restype = None
    _lib.es_set_deadline_ms.argtypes = [ctypes.c_void_p, ctypes.c_int64]
if hasattr(_lib, "es_iterations_run"):
    _lib.es_iterations_run.restype = ctypes.c_int64
    _lib.es_iterations_run.argtypes = [ctypes.c_void_p]
if hasattr(_lib, "es_exploitability_2p"):
    _lib.es_exploitability_2p.restype = ctypes.c_double
    _lib.es_exploitability_2p.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)]
if hasattr(_lib, "es_reach_profile_read"):
    _lib.es_reach_profile_read.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
if hasattr(_lib, "es_enable_visit_counter"):
    _lib.es_enable_visit_counter.argtypes = [ctypes.c_void_p]
    _lib.es_get_visit_count.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.es_get_visit_count.restype = ctypes.c_int64
# int es_run(void*, int64 iters, int num_threads)
_lib.es_run.restype = ctypes.c_int
_lib.es_run.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
# int es_forward_ev_uniform(void*, int player, int64 iters, double*, double*)
_lib.es_forward_ev_uniform.restype = ctypes.c_int
_lib.es_forward_ev_uniform.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int64,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]
# int es_avg_strategy(void*, int node, double* out, int cap)
_lib.es_avg_strategy.restype = ctypes.c_int
_lib.es_avg_strategy.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
_lib.es_final_strategy.restype = ctypes.c_int
_lib.es_final_strategy.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
if hasattr(_lib, "es_strategy_sum_mass"):
    _lib.es_strategy_sum_mass.restype = ctypes.c_double
    _lib.es_strategy_sum_mass.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.es_get_last_error.restype = ctypes.c_char_p

# oracle factories
_lib.es_make_table_oracle.restype = ctypes.c_void_p
_lib.es_make_table_oracle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.es_table_oracle_fn.restype = ctypes.c_void_p
_lib.es_make_joint_oracle.restype = ctypes.c_void_p
_lib.es_make_joint_oracle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.es_joint_oracle_fn.restype = ctypes.c_void_p
_lib.es_free_oracle_ctx.argtypes = [ctypes.c_void_p]

# Table-backed continuation-bias oracle (lightweight and real DLLs).
if _HAS_BIAS:
    _lib.es_make_bias_oracle.restype = ctypes.c_void_p
    _lib.es_make_bias_oracle.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.es_bias_oracle_fn.restype = ctypes.c_void_p
    _lib.es_free_bias_oracle.restype = None
    _lib.es_free_bias_oracle.argtypes = [ctypes.c_void_p]
    _lib.es_bias_oracle_stats.restype = None
    _lib.es_bias_oracle_stats.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    _lib.es_bias_oracle_get_effective_table.restype = ctypes.c_int32
    _lib.es_bias_oracle_get_effective_table.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
    ]

# real-blueprint rollout oracle (only in es_resolver_real.dll)
ESRO_MAX_FLOP_HIST = 256  # = ROLLOUT_MAX_ACTION_HISTORY
if _HAS_REAL:
    _lib.es_make_real_oracle.restype = ctypes.c_void_p
    _lib.es_make_real_oracle.argtypes = [
        ctypes.c_void_p,   # bp (BPSReader*)
        ctypes.c_void_p,   # tree (SSTreeData*)
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # street,stack,nroll,cache_k
        ctypes.c_uint64,   # master_seed
        ctypes.c_int,      # n_leaves
        ctypes.c_int,      # max_h
        ctypes.POINTER(ctypes.c_int),  # pot
        ctypes.POINTER(ctypes.c_int),  # bets
        ctypes.POINTER(ctypes.c_int),  # active
        ctypes.POINTER(ctypes.c_int),  # hist_flat
        ctypes.POINTER(ctypes.c_int),  # hist_len
        ctypes.POINTER(ctypes.c_int),  # player_seats
    ]
    _lib.es_real_oracle_fn.restype = ctypes.c_void_p
    _lib.es_real_oracle_stats.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    _lib.es_real_oracle_stats2.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64)]
    # Leaf-table readout (BR-gap scoring of a rollout-mode solve). dims() sizes
    # the buffer; get_leaf_table() fills-all then copies cache_val out.
    if hasattr(_lib, "es_real_oracle_dims"):
        _lib.es_real_oracle_dims.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        _lib.es_real_oracle_dims.restype = None
    if hasattr(_lib, "es_real_oracle_get_leaf_table"):
        _lib.es_real_oracle_get_leaf_table.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        _lib.es_real_oracle_get_leaf_table.restype = ctypes.c_int
    _lib.es_free_real_oracle.argtypes = [ctypes.c_void_p]

# Per-visit sampled blueprint-rollout oracle (only in es_resolver_real.dll).
# Unlike the cached real oracle, this consumes only the resolver's passed RNG;
# there is deliberately no separate oracle master-seed parameter.
if _HAS_SAMPLED:
    _lib.es_make_sampled_oracle.restype = ctypes.c_void_p
    _lib.es_make_sampled_oracle.argtypes = [
        ctypes.c_void_p,   # bp (BPSReader*)
        ctypes.c_void_p,   # tree (SSTreeData*)
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        # street, stack_chips, num_rollouts, bias_policy
        ctypes.c_int,      # n_leaves
        ctypes.c_int,      # max_h
        ctypes.POINTER(ctypes.c_int),  # pot
        ctypes.POINTER(ctypes.c_int),  # bets
        ctypes.POINTER(ctypes.c_int),  # active
        ctypes.POINTER(ctypes.c_int),  # hist_flat
        ctypes.POINTER(ctypes.c_int),  # hist_len
        ctypes.POINTER(ctypes.c_int),  # player_seats
    ]
    _lib.es_sampled_oracle_fn.restype = ctypes.c_void_p
    _lib.es_sampled_oracle_stats.restype = None
    _lib.es_sampled_oracle_stats.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    _lib.es_sampled_oracle_stats2.restype = None
    _lib.es_sampled_oracle_stats2.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64)]
    _lib.es_free_sampled_oracle.restype = None
    _lib.es_free_sampled_oracle.argtypes = [ctypes.c_void_p]


class ESStats(ctypes.Structure):
    _fields_ = [
        ("traversals", ctypes.c_int64),
        ("leaf_rollouts", ctypes.c_int64),
        ("fold_terminals", ctypes.c_int64),
        ("showdown_terminals", ctypes.c_int64),
        ("decision_visits", ctypes.c_int64),
    ]


_lib.es_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ESStats)]
_lib.es_reset_stats.argtypes = [ctypes.c_void_p]


class _StatsView:
    """Mirror of es_solver.Stats so harness code reading es.stats.* works."""
    def __init__(self, s: ESStats):
        self.traversals = s.traversals
        self.leaf_rollouts = s.leaf_rollouts
        self.fold_terminals = s.fold_terminals
        self.showdown_terminals = s.showdown_terminals
        self.decision_visits = s.decision_visits


class _InfoView:
    """Minimal shim so harness code that iterates `es.info.strat_sum` to find
    visited decision nodes still works. Backed by the C tree's decision-node
    indices; avg_strategy(node) for any of them returns the C σ̄."""
    def __init__(self, solver: "CSolver"):
        self._solver = solver

    @property
    def strat_sum(self):
        # Harnesses use `for ni in es.info.strat_sum` then es.avg_strategy(ni).
        # Return the set of decision-node indices (keys-only usage).
        return self._solver._decision_node_indices


class CSolver:
    """External-sampling MCCFR (C). Drop-in for ESSolver in the harnesses.

    Construct with the GPU solver's tree and an oracle spec:
        CSolver.from_table(gpu._tree, table, np, max_h, seed=1)
        CSolver.joint_zero_sum(gpu._tree, strength0, strength1, seed=1)

    The legacy ESSolver(tree, fn, seed=, linear_cfr=) signature is ALSO accepted
    for the deterministic-table harnesses: if `leaf_rollout_fn` is a numpy table
    (or a closure created from one) we cannot introspect it, so the harnesses
    are updated to call the explicit factory. See validate_*_c.py.
    """

    def __init__(self, tree, oracle_fn_ptr, oracle_ctx, seed: int = 0,
                 _keepalive=None, _sampled_oracle: bool = False,
                 _root_pot_amounts=None, _root_pot_eligible_masks=None):
        self._tree = tree
        self._oracle_ctx = oracle_ctx
        self._real_oracle_ctx = None  # set by from_real_oracle (different free)
        self._bias_oracle_ctx = None
        self._bias_values_keepalive = None
        self._bias_dims = None
        # Tag sampled ownership before es_create so a constructor failure is
        # still cleaned up by the type-specific destructor (the generic oracle
        # free only releases the outer allocation).
        self._sampled_oracle_ctx = oracle_ctx if _sampled_oracle else None
        self._keepalive = _keepalive  # numpy buffers the oracle borrows
        if _root_pot_amounts is None:
            self._handle = _lib.es_create(
                ctypes.byref(tree), oracle_fn_ptr, oracle_ctx,
                ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF))
            create_name = "es_create"
        else:
            if not _HAS_ROOT_POTS:
                raise RuntimeError(
                    "es_create_with_root_pots is missing; rebuild es_resolver"
                )
            self._handle = _lib.es_create_with_root_pots(
                ctypes.byref(tree), oracle_fn_ptr, oracle_ctx,
                ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
                _root_pot_amounts.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_int)),
                _root_pot_eligible_masks.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_uint32)),
                int(_root_pot_amounts.size))
            create_name = "es_create_with_root_pots"
        if not self._handle:
            err = _lib.es_get_last_error()
            if _root_pot_amounts is not None and self._oracle_ctx:
                _lib.es_free_oracle_ctx(self._oracle_ctx)
                self._oracle_ctx = None
            raise RuntimeError(
                f"{create_name} failed: {err.decode() if err else '?'}")
        self.num_players = tree.num_players_field
        self._np = self.num_players
        # decision-node indices from the tree (for the info-view shim)
        self._decision_node_indices = [
            tree.decision_node_indices[i] for i in range(tree.num_decision_nodes)
        ]
        self.info = _InfoView(self)

    # ── factories ────────────────────────────────────────────────────────
    @classmethod
    def from_table(cls, tree, table: np.ndarray, np_players: int, max_h: int,
                   seed: int = 0) -> "CSolver":
        """Deterministic table oracle. table shape [num_leaves, np, max_h]
        (float32), C-contiguous; out = table[leaf, traverser, hand]."""
        t = np.ascontiguousarray(table, dtype=np.float32)
        ctx = _lib.es_make_table_oracle(
            t.ctypes.data_as(ctypes.c_void_p), np_players, max_h)
        fn = _lib.es_table_oracle_fn()
        return cls(tree, fn, ctx, seed=seed, _keepalive=t)

    @classmethod
    def from_table_with_root_pots(
        cls, tree, table: np.ndarray, np_players: int, max_h: int,
        root_pot_amounts, root_pot_eligible_masks, seed: int = 0,
    ) -> "CSolver":
        """Native-river table solver with explicit root-pot eligibility.

        Root masks use compact-player bits, not blueprint seat ids.  The C
        constructor copies both arrays and validates that amounts sum exactly
        to ``tree.starting_pot``.
        """
        if not _HAS_ROOT_POTS:
            raise RuntimeError(
                "es_create_with_root_pots is missing; rebuild es_resolver"
            )
        if np_players < 2 or np_players > 3:
            raise ValueError("root-pot geometry requires 2-3 players")
        if np_players != int(tree.num_players_field):
            raise ValueError(
                f"np_players={np_players} does not match tree players="
                f"{tree.num_players_field}"
            )
        if not bool(tree.is_river) or int(tree.num_leaf_nodes) != 0 \
                or int(tree.num_chance_nodes) != 0:
            raise ValueError("root-pot geometry currently requires a native river tree")

        try:
            amounts_py = tuple(operator.index(x) for x in root_pot_amounts)
            masks_py = tuple(
                operator.index(x) for x in root_pot_eligible_masks)
        except TypeError as exc:
            raise ValueError("root-pot amounts and masks must be integers") from exc
        if not amounts_py or len(amounts_py) != len(masks_py) \
                or len(amounts_py) > 3:
            raise ValueError("root-pot arrays must have the same length in [1, 3]")
        if any(x < 0 or x > np.iinfo(np.int32).max for x in amounts_py):
            raise ValueError("root-pot amounts must fit non-negative int32")
        if any(x <= 0 or x > np.iinfo(np.uint32).max for x in masks_py):
            raise ValueError("root-pot masks must fit nonzero uint32")

        amounts = np.ascontiguousarray(amounts_py, dtype=np.int32)
        masks = np.ascontiguousarray(masks_py, dtype=np.uint32)
        t = np.ascontiguousarray(table, dtype=np.float32)
        ctx = _lib.es_make_table_oracle(
            t.ctypes.data_as(ctypes.c_void_p), np_players, max_h)
        if not ctx:
            raise RuntimeError("es_make_table_oracle returned NULL")
        fn = _lib.es_table_oracle_fn()
        return cls(
            tree, fn, ctx, seed=seed, _keepalive=t,
            _root_pot_amounts=amounts,
            _root_pot_eligible_masks=masks,
        )

    @classmethod
    def from_bias_table(cls, tree, values: np.ndarray, mode: int, np_: int,
                        max_h: int, n_leaves: int,
                        seed: int = 1) -> "CSolver":
        """Table-backed continuation-bias oracle. mode=0 consumes
        [n_leaves,4,np,max_h] marginal values. mode=1 is
        JOINT2P and consumes [n_leaves*16,np,max_h]. The C context borrows the
        float32 values buffer, so this instance retains it for the full solver
        lifetime.
        """
        missing = [name for name in _BIAS_SYMBOLS if not hasattr(_lib, name)]
        if missing:
            raise RuntimeError(
                "bias oracle symbols are missing from this DLL "
                f"({', '.join(missing)}); rebuild es_resolver / "
                "es_resolver_real with es_bias_oracle.c"
            )
        if mode not in (0, 1):
            raise ValueError("mode must be 0 (MARGINAL) or 1 (JOINT2P)")
        if np_ != int(tree.num_players_field):
            raise ValueError(
                f"np_={np_} does not match tree players="
                f"{tree.num_players_field}"
            )
        if mode == 1 and np_ != 2:
            raise ValueError("JOINT2P mode requires np_=2")
        if n_leaves <= 0 or max_h <= 0:
            raise ValueError("n_leaves and max_h must be positive")
        if n_leaves < int(tree.num_leaves):
            raise ValueError(
                f"n_leaves={n_leaves} is smaller than tree.num_leaves="
                f"{tree.num_leaves}"
            )
        for player in range(np_):
            if max_h < int(tree.num_hands[player]):
                raise ValueError(
                    f"max_h={max_h} does not cover player {player}'s "
                    f"{tree.num_hands[player]} hands"
                )

        t = np.ascontiguousarray(values, dtype=np.float32)
        bias_slots = 4 if mode == 0 else 16
        expected = n_leaves * bias_slots * np_ * max_h
        if t.size != expected:
            layout = (
                f"({n_leaves}, 4, {np_}, {max_h})"
                if mode == 0
                else f"({n_leaves * 16}, {np_}, {max_h})"
            )
            raise ValueError(
                f"bias values contain {t.size} floats, expected {expected} "
                f"for layout {layout}"
            )

        ctx = _lib.es_make_bias_oracle(
            t.ctypes.data_as(ctypes.c_void_p),
            int(mode), int(np_), int(max_h), int(n_leaves),
        )
        if not ctx:
            raise RuntimeError(
                "es_make_bias_oracle returned NULL "
                "(invalid arguments, size overflow, or OOM)"
            )
        fn = _lib.es_bias_oracle_fn()
        solver = cls(tree, fn, ctx, seed=seed, _keepalive=t)
        solver._bias_oracle_ctx = ctx
        solver._bias_values_keepalive = t
        solver._bias_dims = (int(n_leaves), int(np_), int(max_h))
        return solver

    @classmethod
    def joint_zero_sum(cls, tree, strength0: np.ndarray, strength1: np.ndarray,
                       seed: int = 0) -> "CSolver":
        """2p zero-sum strength oracle: payoff to p0 = sign(s0-s1)*pot."""
        s0 = np.ascontiguousarray(strength0, dtype=np.float64)
        s1 = np.ascontiguousarray(strength1, dtype=np.float64)
        ctx = _lib.es_make_joint_oracle(
            s0.ctypes.data_as(ctypes.c_void_p), s1.ctypes.data_as(ctypes.c_void_p))
        fn = _lib.es_joint_oracle_fn()
        return cls(tree, fn, ctx, seed=seed, _keepalive=(s0, s1))

    @classmethod
    def with_oracle(cls, tree, oracle_fn_ptr, oracle_ctx, seed: int = 0,
                    keepalive=None) -> "CSolver":
        """Generic: bring your own C oracle fn-ptr + ctx (e.g. the real
        blueprint rollout oracle)."""
        return cls(tree, oracle_fn_ptr, oracle_ctx, seed=seed, _keepalive=keepalive)

    @classmethod
    def from_real_oracle(cls, tree, bp_handle, street: int, stack_chips: int,
                         n_leaves: int, max_h: int, pot, bets, active, hist_flat,
                         hist_len, player_seats, num_rollouts: int = 3,
                         cache_k: int = 16, master_seed: int = 0xABCDEF,
                         seed: int = 0) -> "CSolver":
        """Real blueprint-rollout oracle (requires es_resolver_real.dll).
        Leaf values are CACHED on first hit (strategy-independent), filled all-
        hands averaged over cache_k bias samples × num_rollouts. Per-leaf state
        arrays are int32 numpy, C-contiguous:
          pot [n_leaves], bets/active [n_leaves*np], hist_flat [n_leaves*256],
          hist_len [n_leaves], player_seats [np]. bp_handle = a BPSReader* as a
          ctypes pointer/int. The C ctx owns COPIES of the arrays."""
        if not _HAS_REAL:
            raise RuntimeError("es_make_real_oracle not in this DLL; set ES_DLL=es_resolver_real.dll")
        # keep arrays alive through the make call (C copies them)
        ka = [np.ascontiguousarray(a, dtype=np.int32)
              for a in (pot, bets, active, hist_flat, hist_len, player_seats)]
        ctx = _lib.es_make_real_oracle(
            ctypes.c_void_p(bp_handle), ctypes.byref(tree),
            street, stack_chips, num_rollouts, cache_k,
            ctypes.c_uint64(master_seed & 0xFFFFFFFFFFFFFFFF),
            n_leaves, max_h,
            ka[0].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ka[1].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ka[2].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ka[3].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ka[4].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ka[5].ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        if not ctx:
            raise RuntimeError("es_make_real_oracle returned NULL (OOM?)")
        fn = _lib.es_real_oracle_fn()
        s = cls(tree, fn, ctx, seed=seed, _keepalive=None)
        s._real_oracle_ctx = ctx  # for stats + free
        return s

    @classmethod
    def from_sampled_oracle(cls, tree, bp_handle, street: int, stack_chips: int,
                            n_leaves: int, max_h: int, pot, bets, active,
                            hist_flat, hist_len, player_seats,
                            num_rollouts: int = 3, bias_policy: int = 1,
                            seed: int = 0) -> "CSolver":
        """Per-visit sampled blueprint-rollout oracle.

        Policy 0 draws every active player's continuation bias uniformly and
        freshly on each leaf visit. Policy 1 regret-matches opponent biases and
        enumerates all four traverser biases before updating the traverser's
        dense bias-regret row. Opponent hands are the traversal's given sampled
        hands; there is no cache, all-hands fill, or leaf-value table.

        Requires es_resolver_real. Per-leaf arrays have the same int32,
        C-contiguous layouts as from_real_oracle; the C context copies them.
        ``seed`` is the resolver's sole master seed and therefore controls all
        hand, action, continuation-bias, and rollout randomness. Sampled leaf
        mode is scalar-only; set_vec_mode(1) and set_reach_mode(1) are rejected.
        """
        if not _HAS_SAMPLED:
            raise RuntimeError(
                "es_make_sampled_oracle not in this DLL; "
                "set ES_DLL=es_resolver_real.dll")
        if bias_policy not in (0, 1):
            raise ValueError("bias_policy must be 0 (uniform-fresh) or 1 (regret-matched)")

        # Keep arrays alive through the constructor call; C takes owned copies.
        ka = [np.ascontiguousarray(a, dtype=np.int32)
              for a in (pot, bets, active, hist_flat, hist_len, player_seats)]
        ctx = _lib.es_make_sampled_oracle(
            ctypes.c_void_p(bp_handle), ctypes.byref(tree),
            street, stack_chips, num_rollouts, bias_policy,
            n_leaves, max_h,
            ka[0].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ka[1].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ka[2].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ka[3].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ka[4].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ka[5].ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        if not ctx:
            raise RuntimeError("es_make_sampled_oracle returned NULL (invalid arguments or OOM)")
        fn = _lib.es_sampled_oracle_fn()
        return cls(tree, fn, ctx, seed=seed, _keepalive=None,
                   _sampled_oracle=True)

    def real_oracle_stats(self):
        """(leaf_fills, cache_hits, rollout_sims, oracle_errors)."""
        ctx = getattr(self, "_real_oracle_ctx", None)
        if ctx is None:
            return (0, 0, 0, 0)
        fills = ctypes.c_uint64(0); hits = ctypes.c_uint64(0)
        sims = ctypes.c_uint64(0); errs = ctypes.c_uint64(0)
        _lib.es_real_oracle_stats2(ctx, ctypes.byref(fills), ctypes.byref(hits),
                                   ctypes.byref(sims), ctypes.byref(errs))
        return (fills.value, hits.value, sims.value, errs.value)

    def sampled_oracle_stats(self):
        """Sampled-oracle counters.

        Returns ``(leaf_visits, rollouts, oracle_errors, uniform_visits,
        uniform_rollouts, regret_visits, regret_rollouts)``. The policy split is
        redundant for one immutable-policy context, but makes mixed-run
        instrumentation and accidental policy drift visible.
        """
        ctx = getattr(self, "_sampled_oracle_ctx", None)
        if ctx is None:
            return (0, 0, 0, 0, 0, 0, 0)
        visits = ctypes.c_uint64(0); rollouts = ctypes.c_uint64(0)
        errs = ctypes.c_uint64(0)
        uniform_visits = ctypes.c_uint64(0); uniform_rollouts = ctypes.c_uint64(0)
        regret_visits = ctypes.c_uint64(0); regret_rollouts = ctypes.c_uint64(0)
        _lib.es_sampled_oracle_stats(
            ctx, ctypes.byref(visits), ctypes.byref(rollouts), ctypes.byref(errs))
        _lib.es_sampled_oracle_stats2(
            ctx, ctypes.byref(uniform_visits), ctypes.byref(uniform_rollouts),
            ctypes.byref(regret_visits), ctypes.byref(regret_rollouts))
        return (visits.value, rollouts.value, errs.value,
                uniform_visits.value, uniform_rollouts.value,
                regret_visits.value, regret_rollouts.value)

    def bias_oracle_stats(self):
        """Return ``(leaf_visits, oracle_errors)`` for a bias-table solve."""
        ctx = getattr(self, "_bias_oracle_ctx", None)
        if ctx is None:
            return (0, 0)
        if not hasattr(_lib, "es_bias_oracle_stats"):
            raise RuntimeError(
                "es_bias_oracle_stats not in this DLL; rebuild "
                "es_resolver / es_resolver_real"
            )
        visits = ctypes.c_uint64(0)
        errors = ctypes.c_uint64(0)
        _lib.es_bias_oracle_stats(
            ctx, ctypes.byref(visits), ctypes.byref(errors))
        return (visits.value, errors.value)

    def get_bias_effective_table(self):
        """Post-solve [n_leaves,np,max_h] float32 reporting table. JOINT2P
        folds opponent biases uniformly for this reporting-only
        projection; live solve calls use the sampled opponent hand's
        regret-matched bias row.
        """
        ctx = getattr(self, "_bias_oracle_ctx", None)
        if ctx is None:
            return None
        if not hasattr(_lib, "es_bias_oracle_get_effective_table"):
            raise RuntimeError(
                "es_bias_oracle_get_effective_table not in this DLL; "
                "rebuild es_resolver / es_resolver_real"
            )
        dims = getattr(self, "_bias_dims", None)
        values = getattr(self, "_bias_values_keepalive", None)
        if dims is None or values is None:
            raise RuntimeError(
                "bias oracle dimensions/values keepalive are missing"
            )
        n_leaves, np_players, max_h = dims
        need = n_leaves * np_players * max_h
        if need <= 0:
            raise RuntimeError(
                "bias oracle reported a non-positive effective-table size"
            )
        buf = np.empty(need, dtype=np.float32)
        got = _lib.es_bias_oracle_get_effective_table(
            ctx, values.ctypes.data_as(ctypes.c_void_p),
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), need,
        )
        if got != need:
            raise RuntimeError(
                "es_bias_oracle_get_effective_table returned "
                f"{got}, expected {need} "
                f"(dims {n_leaves}x{np_players}x{max_h})"
            )
        return buf.reshape((n_leaves, np_players, max_h))

    def get_oracle_leaf_table(self):
        """Dense leaf-value table [n_leaves, np, max_h] read from the lazy
        rollout oracle's cache, for BR-gap scoring of a rollout-mode solve.

        Returns a C-contiguous float32 ndarray of shape (n_leaves, np, max_h) —
        the EXACT layout exploitability_c.py indexes as table[leaf, player, hand]
        (leaf in [0,n_leaves), player in [0,np), hand in [0, num_hands[player])
        with num_hands[player] <= max_h, so table[leaf, player, :nh_p] is valid).
        Returns None if this solver was NOT built via from_real_oracle (no real-
        oracle ctx) or if the DLL predates the accessor.

        The C side FILLS ALL leaves before copying (the solve's cache is lazy —
        only visited leaves are populated — but the BR-gap walk reads every
        leaf). Post-solve, single-threaded; never touches the hot path.
        """
        ctx = getattr(self, "_real_oracle_ctx", None)
        if ctx is None:
            return None
        if not (hasattr(_lib, "es_real_oracle_dims")
                and hasattr(_lib, "es_real_oracle_get_leaf_table")):
            return None
        n_leaves = ctypes.c_int(0); npl = ctypes.c_int(0); max_h = ctypes.c_int(0)
        _lib.es_real_oracle_dims(ctx, ctypes.byref(n_leaves),
                                 ctypes.byref(npl), ctypes.byref(max_h))
        need = int(n_leaves.value) * int(npl.value) * int(max_h.value)
        if need <= 0:
            return None
        buf = np.empty(need, dtype=np.float32)
        got = _lib.es_real_oracle_get_leaf_table(
            ctx, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), need)
        if got == -2:
            # A leaf failed to fill during the readout (OOM / all-rollouts-
            # errored). The C side refused to return a partially-zeroed table
            # (which would silently bias the BR-gap toward 0). Fail loud.
            raise RuntimeError(
                "es_real_oracle_get_leaf_table: a leaf failed to fill (rc=-2) — "
                "the leaf table would be partially zeroed; refusing to score a "
                "corrupted BR-gap")
        if got != need:
            raise RuntimeError(
                f"es_real_oracle_get_leaf_table returned {got}, expected {need} "
                f"(dims {n_leaves.value}x{npl.value}x{max_h.value})")
        return buf.reshape(
            (int(n_leaves.value), int(npl.value), int(max_h.value)))

    def set_cfr_mode(self, mode: int) -> None:
        """0 = vanilla (validated reference), 1 = Linear CFR (matches the GPU).
        Set before run()."""
        if hasattr(_lib, "es_set_cfr_mode"):
            _lib.es_set_cfr_mode(self._handle, int(mode))
        elif mode == 1:
            raise RuntimeError("es_set_cfr_mode not in this DLL (rebuild)")

    def set_vec_mode(self, mode: int) -> None:
        """0 = scalar (validated reference), 1 = vectorized traverser (all
        traverser hands per walk; same fixed point, lower variance). Set before
        run(). Raises if the DLL predates the feature and mode==1."""
        if mode == 1 and getattr(self, "_sampled_oracle_ctx", None) is not None:
            raise ValueError(
                "sampled leaf mode is scalar-only: fresh per-visit biases are "
                "incoherent across the vectorized per-hand leaf loop")
        if hasattr(_lib, "es_set_vec_mode"):
            _lib.es_set_vec_mode(self._handle, int(mode))
        elif mode == 1:
            raise RuntimeError("es_set_vec_mode not in this DLL (rebuild)")

    def set_exact_allin_leaves(self, enabled: bool) -> None:
        """Use exact sampled-pair runout values at betting-closed leaves.

        This is opt-in and should be configured before run(). Disabling is a
        no-op with an older shared library; enabling fails clearly so callers
        cannot silently fall back to eager hand-versus-range leaf values.
        """
        enabled_int = 1 if enabled else 0
        if not _HAS_EXACT_ALLIN_LEAVES:
            if enabled_int:
                raise RuntimeError(
                    f"{_DLL_NAME} is missing es_set_exact_allin_leaves; "
                    "rebuild es_resolver or es_resolver_real"
                )
            return
        _lib.es_set_exact_allin_leaves(self._handle, enabled_int)

    def set_traverser_actual_hand(
        self, player_idx: int, hand_idx: int, fraction: float,
    ) -> None:
        """Configure the scalar traverser-hand mixture for one actual row."""
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be finite and within [0, 1]")
        if not hasattr(_lib, "es_set_traverser_actual_hand"):
            raise RuntimeError(
                "es_set_traverser_actual_hand not in this DLL (rebuild)"
            )
        rc = _lib.es_set_traverser_actual_hand(
            self._handle, int(player_idx), int(hand_idx), float(fraction),
        )
        if rc != 0:
            err = _lib.es_get_last_error()
            raise RuntimeError(
                "es_set_traverser_actual_hand failed: "
                f"{err.decode() if err else '?'}"
            )

    def set_chance_mode(self, mode: int) -> None:
        """In-tree chance-node handling: 1 = ENUMERATE (correct, zero chance-
        variance — average over all dealt cards), 0 = SAMPLE (one card/iter,
        high variance), -1 = inherit env default. REQUIRED=1 for a correct 2p
        HU flop/turn (in-tree-chance) solve. Set before run()."""
        if hasattr(_lib, "es_set_chance_mode"):
            _lib.es_set_chance_mode(self._handle, int(mode))
        elif mode == 0:
            raise RuntimeError("es_set_chance_mode not in this DLL (rebuild)")

    def set_opp_mode(self, mode: int) -> None:
        """Opponent-action handling: 1 = ENUMERATE (full CFR — recurse all opp
        actions; SMALL player counts only), 0 = SAMPLE (external sampling —
        mandatory at 5-6p), -1 = inherit env default. For the 2p chance solve set
        1 (+ vec mode) to kill the single-sample oscillation. Set before run()."""
        if hasattr(_lib, "es_set_opp_mode"):
            _lib.es_set_opp_mode(self._handle, int(mode))
        elif mode == 1:
            raise RuntimeError("es_set_opp_mode not in this DLL (rebuild)")

    def set_reach_mode(self, mode: int) -> None:
        """Full-per-hand-reach vector CFR: 1 = ON (the interactive 2p chance
        solve — carry per-hand opponent reach, integrate ALL opponent hands each
        iter with NO sampling; Linear CFR + chance-enumerate by construction;
        deterministic; np==2 only), 0 = OFF (default; sampling traversers). When
        ON it supersedes cfr/vec/chance/opp modes. Set before run(). Raises if the
        DLL predates the feature and mode==1."""
        if mode == 1 and getattr(self, "_sampled_oracle_ctx", None) is not None:
            raise ValueError(
                "sampled leaf mode is scalar-only and cannot use full-hand reach mode")
        if hasattr(_lib, "es_set_reach_mode"):
            _lib.es_set_reach_mode(self._handle, int(mode))
        elif mode == 1:
            raise RuntimeError("es_set_reach_mode not in this DLL (rebuild)")

    def exploitability_2p(self):
        """2p exploitability (best-response gap) of the current σ̄, in pot-chip
        units per hand. Returns (total, gap0, gap1); each gap >= 0; total DROPS
        toward 0 as the solve converges. Reuses the reach machinery's showdown/
        fold evaluators (grades chance trees). Raises if the DLL predates it."""
        if not hasattr(_lib, "es_exploitability_2p"):
            raise RuntimeError("es_exploitability_2p not in this DLL (rebuild)")
        g0 = ctypes.c_double(0.0)
        g1 = ctypes.c_double(0.0)
        tot = _lib.es_exploitability_2p(self._handle, ctypes.byref(g0), ctypes.byref(g1))
        return float(tot), float(g0.value), float(g1.value)

    def enable_visit_counter(self) -> None:
        """DEBUG: count per-node traverser visits. Call before run()."""
        if hasattr(_lib, "es_enable_visit_counter"):
            _lib.es_enable_visit_counter(self._handle)
        else:
            raise RuntimeError("es_enable_visit_counter not in this DLL (rebuild)")

    def visit_count(self, node_idx: int) -> int:
        return int(_lib.es_get_visit_count(self._handle, int(node_idx)))

    def set_deadline_ms(self, ms: int) -> None:
        """Set a relative solve deadline in milliseconds; 0 disables it.

        The deadline is applied by the next run() and checked periodically by
        the C iteration loop. Use iterations_run() to read how many complete
        iterations the most recent run actually finished.
        """
        if ms < 0 or ms > 0x7FFFFFFFFFFFFFFF:
            raise ValueError("deadline must fit a nonnegative int64 millisecond value")
        if not hasattr(_lib, "es_set_deadline_ms"):
            if ms != 0:
                raise RuntimeError("es_set_deadline_ms not in this DLL (rebuild)")
            return
        _lib.es_set_deadline_ms(self._handle, ctypes.c_int64(int(ms)))

    def iterations_run(self) -> int:
        """Complete iterations finished by the most recent run()."""
        if not hasattr(_lib, "es_iterations_run"):
            raise RuntimeError("es_iterations_run not in this DLL (rebuild)")
        return int(_lib.es_iterations_run(self._handle))

    # ── run ───────────────────────────────────────────────────────────────
    def run(self, iterations: int, num_threads: int = 1) -> None:
        rc = _lib.es_run(self._handle, ctypes.c_int64(int(iterations)),
                         ctypes.c_int(num_threads))
        if rc != 0:
            err = _lib.es_get_last_error()
            raise RuntimeError(f"es_run failed: {err.decode() if err else '?'}")

    def forward_ev_uniform(self, player: int, iterations: int):
        mean = ctypes.c_double(0.0)
        se = ctypes.c_double(0.0)
        rc = _lib.es_forward_ev_uniform(
            self._handle, player, ctypes.c_int64(int(iterations)),
            ctypes.byref(mean), ctypes.byref(se))
        if rc != 0:
            err = _lib.es_get_last_error()
            raise RuntimeError(f"es_forward_ev_uniform failed: {err.decode() if err else '?'}")
        return mean.value, se.value

    # ── readout ───────────────────────────────────────────────────────────
    def avg_strategy(self, node_idx: int) -> Optional[np.ndarray]:
        """σ̄ at a decision node: [nh, na] float64. None if not a decision node."""
        node = self._tree.nodes[node_idx]
        if node.type != 0 or node.player < 0 or node.num_children <= 0:
            return None
        acting = node.player
        nh = self._tree.num_hands[acting]
        na = node.num_children
        buf = (ctypes.c_double * (nh * na))()
        n = _lib.es_avg_strategy(self._handle, node_idx, buf, nh * na)
        if n < 0:
            return None
        return np.ctypeslib.as_array(buf).reshape(nh, na).copy()

    def strategy_sum_mass(self, node_idx: int, hand_idx: int) -> float:
        """Pre-normalization strategy-sum mass for one decision-node row."""
        if not hasattr(_lib, "es_strategy_sum_mass"):
            raise RuntimeError("es_strategy_sum_mass not in this DLL (rebuild)")
        mass = float(_lib.es_strategy_sum_mass(
            self._handle, int(node_idx), int(hand_idx),
        ))
        if not np.isfinite(mass) or mass < 0.0:
            err = _lib.es_get_last_error()
            raise RuntimeError(
                "es_strategy_sum_mass failed: "
                f"{err.decode() if err else '?'}"
            )
        return mass

    def final_strategy(self, node_idx: int) -> Optional[np.ndarray]:
        node = self._tree.nodes[node_idx]
        if node.type != 0 or node.player < 0 or node.num_children <= 0:
            return None
        acting = node.player
        nh = self._tree.num_hands[acting]
        na = node.num_children
        buf = (ctypes.c_double * (nh * na))()
        n = _lib.es_final_strategy(self._handle, node_idx, buf, nh * na)
        if n < 0:
            return None
        return np.ctypeslib.as_array(buf).reshape(nh, na).copy()

    @property
    def stats(self) -> _StatsView:
        s = ESStats()
        _lib.es_get_stats(self._handle, ctypes.byref(s))
        return _StatsView(s)

    def reset_stats(self) -> None:
        _lib.es_reset_stats(self._handle)

    def __del__(self):
        try:
            if getattr(self, "_handle", None):
                _lib.es_free(self._handle)
                self._handle = None
            if getattr(self, "_real_oracle_ctx", None):
                _lib.es_free_real_oracle(self._real_oracle_ctx)
                self._real_oracle_ctx = None
                self._oracle_ctx = None  # same ptr; don't double-free
            elif getattr(self, "_bias_oracle_ctx", None):
                _lib.es_free_bias_oracle(self._bias_oracle_ctx)
                self._bias_oracle_ctx = None
                self._bias_values_keepalive = None
                self._oracle_ctx = None  # same ptr; don't double-free
            elif getattr(self, "_sampled_oracle_ctx", None):
                _lib.es_free_sampled_oracle(self._sampled_oracle_ctx)
                self._sampled_oracle_ctx = None
                self._oracle_ctx = None  # same ptr; don't double-free
            elif getattr(self, "_oracle_ctx", None):
                _lib.es_free_oracle_ctx(self._oracle_ctx)
                self._oracle_ctx = None
        except Exception:
            pass
