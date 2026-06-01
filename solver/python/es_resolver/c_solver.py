"""CSolver — ctypes wrapper around build/es_resolver.dll (the production C ES
re-solver), with the SAME surface as es_solver.ESSolver so the existing Python
validation harnesses (validate_tiny, validate_fixed_profile, validate_vs_gpu)
run against the C solver UNCHANGED by swapping ESSolver(...) -> CSolver(...).

Design (advisor 2026-06-01):
  - The C solver consumes the IDENTICAL SSTreeData the GPU builds — we pass
    ctypes.byref(gpu_solver._tree) straight in. No PlainTree snapshot (that was
    only the Python reference's ctypes-lifetime workaround).
  - The leaf oracle is a C function pointer + ctx (es_oracles.c). NEVER a Python
    callback in the hot path. We expose two oracle factories:
      from_table(table, np, max_h)  — deterministic table lookup
      joint_zero_sum(strength0, strength1) — 2p zero-sum strength showdown
    For the real-blueprint rollout oracle see real_oracle_c (separate module).
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
import os
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
# ES_DLL selects which build to load:
#   "es_resolver.dll"      — lightweight, table/joint oracles only (validation)
#   "es_resolver_real.dll" — links rollout + bps_reader; adds the real oracle +
#                            the BPS loader symbols (real-blueprint path / bench)
_DLL_NAME = os.environ.get("ES_DLL", "es_resolver.dll")
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

# void* es_create(const SSTreeData*, ESLeafOracle, void* ctx, uint64_t seed)
_lib.es_create.restype = ctypes.c_void_p
_lib.es_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
_lib.es_free.argtypes = [ctypes.c_void_p]
if hasattr(_lib, "es_set_cfr_mode"):
    _lib.es_set_cfr_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
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
_lib.es_get_last_error.restype = ctypes.c_char_p

# oracle factories
_lib.es_make_table_oracle.restype = ctypes.c_void_p
_lib.es_make_table_oracle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.es_table_oracle_fn.restype = ctypes.c_void_p
_lib.es_make_joint_oracle.restype = ctypes.c_void_p
_lib.es_make_joint_oracle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.es_joint_oracle_fn.restype = ctypes.c_void_p
_lib.es_free_oracle_ctx.argtypes = [ctypes.c_void_p]

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
    _lib.es_free_real_oracle.argtypes = [ctypes.c_void_p]


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
                 _keepalive=None):
        self._tree = tree
        self._oracle_ctx = oracle_ctx
        self._real_oracle_ctx = None  # set by from_real_oracle (different free)
        self._keepalive = _keepalive  # numpy buffers the oracle borrows
        self._handle = _lib.es_create(
            ctypes.byref(tree), oracle_fn_ptr, oracle_ctx, ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF))
        if not self._handle:
            err = _lib.es_get_last_error()
            raise RuntimeError(f"es_create failed: {err.decode() if err else '?'}")
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

    def set_cfr_mode(self, mode: int) -> None:
        """0 = vanilla (validated reference), 1 = Linear CFR (matches the GPU).
        Set before run()."""
        if hasattr(_lib, "es_set_cfr_mode"):
            _lib.es_set_cfr_mode(self._handle, int(mode))
        elif mode == 1:
            raise RuntimeError("es_set_cfr_mode not in this DLL (rebuild)")

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
            elif getattr(self, "_oracle_ctx", None):
                _lib.es_free_oracle_ctx(self._oracle_ctx)
                self._oracle_ctx = None
        except Exception:
            pass
