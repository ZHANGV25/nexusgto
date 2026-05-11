"""ctypes binding for src/rollout.c — the C port of python/rollout.py.

Produces the same output array shape as compute_subgame_leaf_values,
but orders of magnitude faster. The Python module remains the
correctness reference; this module is the shipping path for
nexusgto-api's live flop solves.

See docs/C2_CPU_ROLLOUT_PLAN.md for the port's design, known parity-
with-Python limitations, and test plan.

Usage:

    from bps_reader_py import BPSReaderC
    from rollout_c import compute_subgame_leaf_values_c

    bp = BPSReaderC(".../unified_blueprint_v3_1.5B.bps")
    leaf_values = compute_subgame_leaf_values_c(
        leaf_infos=..., board=[...], street="flop",
        player_hands=..., active=..., acting_order=...,
        player_seats=..., game_action_history=...,
        blueprint=bp, starting_pot_chips=..., stack_chips=...,
        max_hands=1326, num_rollouts=3,
    )
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from leaf_values import LeafValueUnavailable  # type: ignore[import-untyped]
except ImportError:
    from python.leaf_values import LeafValueUnavailable  # type: ignore[no-redef]


# Must match ROLLOUT_MAX_PLAYERS in src/rollout.h
_ROLLOUT_MAX_PLAYERS = 6
# Must match ROLLOUT_MAX_ACTION_HISTORY in src/rollout.h
_ROLLOUT_MAX_ACTION_HISTORY = 256


# ─── ctypes structs mirroring src/rollout.h ──────────────────────────────


class _RolloutHand(ctypes.Structure):
    _fields_ = [
        ("c0", ctypes.c_int32),
        ("c1", ctypes.c_int32),
        ("weight", ctypes.c_float),
    ]


class _RolloutHandList(ctypes.Structure):
    _fields_ = [
        ("hands", ctypes.POINTER(_RolloutHand)),
        ("n_hands", ctypes.c_int32),
    ]


class _RolloutLeafInfo(ctypes.Structure):
    # Field order MUST match RolloutLeafInfo in src/rollout.h exactly.
    # A sizeof parity check is asserted at module load (see _lib() below)
    # — if the C struct layout drifts, the rollout misbehaves silently
    # because per-leaf flop_history would land at the wrong offset.
    _fields_ = [
        ("leaf_idx", ctypes.c_int32),
        ("pot_chips", ctypes.c_int32),
        ("bets", ctypes.c_int32 * _ROLLOUT_MAX_PLAYERS),
        ("flop_history", ctypes.c_int32 * _ROLLOUT_MAX_ACTION_HISTORY),
        ("flop_history_len", ctypes.c_int32),
    ]


# ─── Library loader ──────────────────────────────────────────────────────


def _find_lib() -> ctypes.CDLL:
    base = Path(__file__).resolve().parent.parent / "build"
    ext = ".dll" if sys.platform == "win32" else ".so"
    candidates = [base / f"rollout{ext}"]
    env_path = os.environ.get("POKER_SOLVER_PATH")
    if env_path:
        candidates.append(Path(env_path) / "build" / f"rollout{ext}")
    for p in candidates:
        if p.exists():
            return ctypes.CDLL(str(p))
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"rollout shared library not found. Searched: {searched}. "
        f"Run 'make rollout' (or 'mingw32-make rollout' on Windows) "
        f"in the poker-solver directory."
    )


def _setup_ctypes(lib: ctypes.CDLL) -> None:
    # Pure-math exports (unit-testable bit-exact)
    lib.rollout_apply_bias_c.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int32,
        ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rollout_apply_bias_c.restype = None

    lib.rollout_classify_postflop_actions_c.argtypes = [
        ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
    ]
    lib.rollout_classify_postflop_actions_c.restype = ctypes.c_int32

    lib.rollout_action_amount_chips_c.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
    ]
    lib.rollout_action_amount_chips_c.restype = None

    lib.rollout_compute_showdown_payoff_c.argtypes = [
        ctypes.POINTER(ctypes.c_int),       # full_board
        ctypes.POINTER(ctypes.c_int),       # hands_c0
        ctypes.POINTER(ctypes.c_int),       # hands_c1
        ctypes.c_int32,                     # num_players
        ctypes.POINTER(ctypes.c_int32),     # active
        ctypes.POINTER(ctypes.c_int32),     # bets
        ctypes.c_int32,                     # pot_chips
        ctypes.c_int32,                     # hero_player
    ]
    lib.rollout_compute_showdown_payoff_c.restype = ctypes.c_float

    # Full subgame entry (API-facing)
    lib.rollout_compute_subgame_leaf_values_c.argtypes = [
        ctypes.c_void_p,                    # BPSReader *
        ctypes.POINTER(ctypes.c_int),       # board
        ctypes.c_int32,                     # n_board_cards
        ctypes.c_int32,                     # street
        ctypes.POINTER(_RolloutLeafInfo),   # leaf_infos
        ctypes.c_int32,                     # n_leaves
        ctypes.POINTER(ctypes.c_int32),     # active
        ctypes.c_int32,                     # num_players
        ctypes.POINTER(_RolloutHandList),   # player_hands
        ctypes.POINTER(ctypes.c_int32),     # acting_order
        ctypes.c_int32,                     # acting_order_len
        ctypes.POINTER(ctypes.c_int32),     # player_seats
        ctypes.POINTER(ctypes.c_int32),     # game_action_history
        ctypes.c_int32,                     # gah_len
        ctypes.c_int32,                     # stack_chips
        ctypes.c_int32,                     # max_hands
        ctypes.c_int32,                     # num_rollouts
        ctypes.c_uint64,                    # master_seed
        ctypes.POINTER(ctypes.c_float),     # out
        ctypes.c_int32,                     # out_size
    ]
    lib.rollout_compute_subgame_leaf_values_c.restype = ctypes.c_int32

    # Sampled-cont-strat variant (OP4 / Option c)
    lib.rollout_compute_subgame_leaf_values_sampled_c.argtypes = [
        ctypes.c_void_p,                    # BPSReader *
        ctypes.POINTER(ctypes.c_int),       # board
        ctypes.c_int32,                     # n_board_cards
        ctypes.c_int32,                     # street
        ctypes.POINTER(_RolloutLeafInfo),   # leaf_infos
        ctypes.c_int32,                     # n_leaves
        ctypes.POINTER(ctypes.c_int32),     # active
        ctypes.c_int32,                     # num_players
        ctypes.POINTER(_RolloutHandList),   # player_hands
        ctypes.POINTER(ctypes.c_int32),     # acting_order
        ctypes.c_int32,                     # acting_order_len
        ctypes.POINTER(ctypes.c_int32),     # player_seats
        ctypes.POINTER(ctypes.c_int32),     # game_action_history
        ctypes.c_int32,                     # gah_len
        ctypes.c_int32,                     # stack_chips
        ctypes.c_int32,                     # max_hands
        ctypes.c_int32,                     # num_rollouts
        ctypes.c_int32,                     # num_samples_k
        ctypes.c_uint64,                    # master_seed
        ctypes.POINTER(ctypes.c_float),     # out
        ctypes.c_int32,                     # out_size
    ]
    lib.rollout_compute_subgame_leaf_values_sampled_c.restype = ctypes.c_int32

    lib.rollout_compute_leaf_value_c.argtypes = [
        ctypes.c_void_p,                    # BPSReader *
        ctypes.POINTER(ctypes.c_int),       # board
        ctypes.c_int32,                     # n_board_cards
        ctypes.c_int32,                     # street
        ctypes.c_int32,                     # pot_chips
        ctypes.POINTER(ctypes.c_int32),     # active
        ctypes.c_int32,                     # num_players
        ctypes.POINTER(ctypes.c_int32),     # stacks
        ctypes.POINTER(ctypes.c_int32),     # bets
        ctypes.POINTER(_RolloutHandList),   # player_hands
        ctypes.POINTER(ctypes.c_int32),     # acting_order
        ctypes.c_int32,                     # acting_order_len
        ctypes.POINTER(ctypes.c_int32),     # player_seats
        ctypes.POINTER(ctypes.c_int32),     # game_action_history
        ctypes.c_int32,                     # gah_len
        ctypes.POINTER(ctypes.c_int32),     # bias_profile
        ctypes.c_int32,                     # hero_player
        ctypes.c_int32,                     # hero_hand_idx
        ctypes.c_int32,                     # num_rollouts
        ctypes.c_uint64,                    # master_seed
        ctypes.c_int32,                     # leaf_idx_for_seed
        ctypes.c_int32,                     # combo_for_seed
        ctypes.POINTER(ctypes.c_float),     # out_value
    ]
    lib.rollout_compute_leaf_value_c.restype = ctypes.c_int32

    lib.rollout_get_last_error.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.rollout_get_last_error.restype = None

    # Observability counters (uniform-on-zero substitution).
    lib.rollout_reset_observability.argtypes = []
    lib.rollout_reset_observability.restype = None

    lib.rollout_get_observability.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.rollout_get_observability.restype = None

    lib.rollout_get_bucket_miss_count.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.rollout_get_bucket_miss_count.restype = None

    lib.rollout_set_tolerate_bucket_miss.argtypes = [ctypes.c_int]
    lib.rollout_set_tolerate_bucket_miss.restype = None

    lib.rollout_get_tolerate_bucket_miss.argtypes = []
    lib.rollout_get_tolerate_bucket_miss.restype = ctypes.c_int

    lib.rollout_add_observability.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
    ]
    lib.rollout_add_observability.restype = None

    if hasattr(lib, "rollout_get_per_street_miss_counts"):
        lib.rollout_get_per_street_miss_counts.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        lib.rollout_get_per_street_miss_counts.restype = None
    if hasattr(lib, "rollout_add_per_street_miss_counts"):
        lib.rollout_add_per_street_miss_counts.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        lib.rollout_add_per_street_miss_counts.restype = None

    # Phase F Step 14: cont_ev_table consumer at end-of-flop multiway
    # boundary. See src/rollout.h "Phase F" block.
    if hasattr(lib, "rollout_set_cont_ev_table"):
        lib.rollout_set_cont_ev_table.argtypes = [ctypes.c_void_p]
        lib.rollout_set_cont_ev_table.restype = None
    if hasattr(lib, "rollout_get_cont_ev_table"):
        lib.rollout_get_cont_ev_table.argtypes = []
        lib.rollout_get_cont_ev_table.restype = ctypes.c_void_p
    if hasattr(lib, "rollout_set_use_cont_ev_at_flop_boundary"):
        lib.rollout_set_use_cont_ev_at_flop_boundary.argtypes = [ctypes.c_int]
        lib.rollout_set_use_cont_ev_at_flop_boundary.restype = None
    if hasattr(lib, "rollout_get_use_cont_ev_at_flop_boundary"):
        lib.rollout_get_use_cont_ev_at_flop_boundary.argtypes = []
        lib.rollout_get_use_cont_ev_at_flop_boundary.restype = ctypes.c_int
    if hasattr(lib, "rollout_get_cont_ev_boundary_hit_count"):
        lib.rollout_get_cont_ev_boundary_hit_count.argtypes = [
            ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.rollout_get_cont_ev_boundary_hit_count.restype = None
    if hasattr(lib, "rollout_reset_cont_ev_boundary_hit_count"):
        lib.rollout_reset_cont_ev_boundary_hit_count.argtypes = []
        lib.rollout_reset_cont_ev_boundary_hit_count.restype = None

    lib.rollout_leaf_info_sizeof.argtypes = []
    lib.rollout_leaf_info_sizeof.restype = ctypes.c_int32


_LIB: Optional[ctypes.CDLL] = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _find_lib()
        _setup_ctypes(_LIB)
        _assert_struct_parity(_LIB)
    return _LIB


def _assert_struct_parity(lib: ctypes.CDLL) -> None:
    """Hard-fail on RolloutLeafInfo layout drift between C and Python.

    The per-leaf flop_history field's offset MUST match between
    src/rollout.h's RolloutLeafInfo and the _RolloutLeafInfo ctypes
    mirror. A silent mismatch sends junk bytes to the rollout's blueprint
    queries — exactly the failure the sentinel battery struggled to
    diagnose. Verify at module load instead of at first use.
    """
    c_size = int(lib.rollout_leaf_info_sizeof())
    py_size = ctypes.sizeof(_RolloutLeafInfo)
    if c_size != py_size:
        raise RuntimeError(
            f"RolloutLeafInfo size mismatch: C reports {c_size} bytes, "
            f"Python ctypes mirror is {py_size} bytes. Field order or "
            f"padding drifted — update _RolloutLeafInfo in rollout_c.py "
            f"to match src/rollout.h."
        )


def _last_error() -> Tuple[str, int, int]:
    lib = _lib()
    reason = ctypes.c_char_p()
    street = ctypes.c_int32(0)
    num_players = ctypes.c_int32(0)
    lib.rollout_get_last_error(
        ctypes.byref(reason), ctypes.byref(street), ctypes.byref(num_players)
    )
    msg = reason.value.decode("utf-8", "replace") if reason.value else ""
    return msg, int(street.value), int(num_players.value)


def reset_rollout_observability() -> None:
    """Zero the per-thread σ̄-substitution counters before a solve."""
    _lib().rollout_reset_observability()


def get_rollout_observability() -> Tuple[int, int]:
    """Read the per-thread σ̄-substitution counters.

    Returns (blueprint_miss_count, zerosum_count) where:
      - blueprint_miss_count: σ̄ entry not found (coverage gap; expected
        outside the trained tree). Substituted with uniform-on-legal-
        actions per Pluribus Algorithm 1 supplementary.
      - zerosum_count: σ̄ entry present but truncated strategy summed
        to zero (data integrity issue; should be rare).

    The bucket-miss counter (used when the tolerate-miss gate is
    enabled) is exposed separately via get_rollout_bucket_miss_count.

    Callers that care about coverage should reset before each solve and
    read after to scope counts to that solve.
    """
    lib = _lib()
    miss = ctypes.c_uint64(0)
    zerosum = ctypes.c_uint64(0)
    lib.rollout_get_observability(ctypes.byref(miss), ctypes.byref(zerosum))
    return (int(miss.value), int(zerosum.value))


def get_rollout_bucket_miss_count() -> int:
    """Read the per-thread bucket-lookup miss counter.

    Bumped each time the rollout would have aborted on a hand_to_bucket
    failure but the tolerate-miss gate let it fall through to uniform-
    on-legal-actions instead. Distinct from blueprint miss (bucket
    existed but no σ̄ stored).
    """
    lib = _lib()
    bucket_miss = ctypes.c_uint64(0)
    lib.rollout_get_bucket_miss_count(ctypes.byref(bucket_miss))
    return int(bucket_miss.value)


def get_rollout_per_street_miss_counts() -> Tuple[int, int, int, int]:

    Returns (preflop, flop, turn, river) miss counts. Distinguishes
    action_history hash mismatch (high miss everywhere) from blueprint
    coverage sparseness (miss skewed to deeper streets).
    Returns (0,0,0,0) on builds without the symbol.
    """
    lib = _lib()
    if not hasattr(lib, "rollout_get_per_street_miss_counts"):
        return (0, 0, 0, 0)
    arr = (ctypes.c_uint64 * 4)()
    lib.rollout_get_per_street_miss_counts(arr)
    return (int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]))


def add_rollout_per_street_miss_counts(counts: Tuple[int, int, int, int]) -> None:
    """Fold per-worker per-street counts into the parent's view (fork-pool aggregation)."""
    lib = _lib()
    if not hasattr(lib, "rollout_add_per_street_miss_counts"):
        return
    arr = (ctypes.c_uint64 * 4)(*[int(c) for c in counts])
    lib.rollout_add_per_street_miss_counts(arr)


def set_rollout_tolerate_bucket_miss(enabled: bool) -> None:
    """Enable/disable the rollout's bucket-miss tolerance gate.

    When enabled, a hand_to_bucket failure no longer aborts the rollout;
    instead the call falls through to uniform-on-legal-actions (Pluribus
    Algorithm 1 supplementary). Each substitution bumps the bucket-miss
    counter (see get_rollout_bucket_miss_count).

    Off by default. Also configurable via the env
    ROLLOUT_TOLERATE_BUCKET_MISS=1 (read lazily on first hot-path call).
    Calling this overrides the env value.
    """
    _lib().rollout_set_tolerate_bucket_miss(1 if enabled else 0)


def get_rollout_tolerate_bucket_miss() -> bool:
    return bool(_lib().rollout_get_tolerate_bucket_miss())


def add_rollout_observability(
    add_miss: int, add_zerosum: int, add_bucket_miss: int
) -> None:
    """Add to the per-thread observability counters.

    Used by orchestrations that fan out via multiprocessing.Pool — each
    worker process has its own thread-local counter copy, so the
    parent's view stays at zero despite real misses. The parent reads
    each worker's counters at end-of-task and folds the totals back in
    via this call.
    """
    _lib().rollout_add_observability(
        ctypes.c_uint64(int(add_miss)),
        ctypes.c_uint64(int(add_zerosum)),
        ctypes.c_uint64(int(add_bucket_miss)),
    )


# ─── Pure-math thin wrappers (used by parity tests) ──────────────────────


def apply_bias_c(
    strategy: np.ndarray, bias_profile: int, categories: Sequence[int]
) -> np.ndarray:
    lib = _lib()
    s = np.ascontiguousarray(strategy, dtype=np.float64)
    n = s.size
    cats = np.ascontiguousarray(np.asarray(categories, dtype=np.int32))
    out = np.zeros(n, dtype=np.float64)
    lib.rollout_apply_bias_c(
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(n),
        ctypes.c_int32(int(bias_profile)),
        cats.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return out


def classify_postflop_actions_c(
    facing_bet: bool, raises_so_far: int
) -> List[int]:
    lib = _lib()
    buf = (ctypes.c_int32 * 16)()
    n = lib.rollout_classify_postflop_actions_c(
        ctypes.c_int32(1 if facing_bet else 0),
        ctypes.c_int32(int(raises_so_far)),
        buf, ctypes.c_int32(16),
    )
    return [int(buf[i]) for i in range(int(n))]


def action_amount_chips_c(
    action_idx: int, facing_bet: bool, raises_so_far: int,
    pot: int, to_call: int, stack: int,
) -> Tuple[str, int]:
    lib = _lib()
    kind = ctypes.c_int32(0)
    amt = ctypes.c_int32(0)
    lib.rollout_action_amount_chips_c(
        ctypes.c_int32(action_idx),
        ctypes.c_int32(1 if facing_bet else 0),
        ctypes.c_int32(int(raises_so_far)),
        ctypes.c_int32(int(pot)),
        ctypes.c_int32(int(to_call)),
        ctypes.c_int32(int(stack)),
        ctypes.byref(kind), ctypes.byref(amt),
    )
    kind_str = {0: "fold", 1: "call", 2: "check",
                3: "bet", 4: "raise", 5: "allin"}.get(int(kind.value), "?")
    return (kind_str, int(amt.value))


def compute_showdown_payoff_c(
    full_board: List[int],
    hands: List[Tuple[int, int]],
    active: List[bool],
    bets: List[int],
    pot_chips: int,
    hero_player: int,
) -> float:
    lib = _lib()
    n = len(hands)
    board_arr = (ctypes.c_int * 5)(*full_board[:5])
    c0_arr = (ctypes.c_int * n)(*[h[0] for h in hands])
    c1_arr = (ctypes.c_int * n)(*[h[1] for h in hands])
    active_arr = (ctypes.c_int32 * n)(*[1 if a else 0 for a in active])
    bets_arr = (ctypes.c_int32 * n)(*bets)
    return float(lib.rollout_compute_showdown_payoff_c(
        board_arr, c0_arr, c1_arr,
        ctypes.c_int32(n), active_arr, bets_arr,
        ctypes.c_int32(pot_chips),
        ctypes.c_int32(hero_player),
    ))


# ─── Full subgame entry (API-facing) ─────────────────────────────────────


_STREET_NUM = {"flop": 1, "turn": 2, "river": 3}


def _hand_list_struct(
    hands: Sequence[Tuple[int, int, float]],
    _stash: list,
) -> _RolloutHandList:
    """Pack one player's hand list into a RolloutHandList. The raw
    ctypes array is kept alive by appending it to _stash."""
    n = len(hands)
    buf = (_RolloutHand * n)()
    for i, (c0, c1, w) in enumerate(hands):
        buf[i].c0 = int(c0)
        buf[i].c1 = int(c1)
        buf[i].weight = float(w)
    _stash.append(buf)
    hl = _RolloutHandList()
    if n > 0:
        hl.hands = ctypes.cast(buf, ctypes.POINTER(_RolloutHand))
    hl.n_hands = n
    return hl


def compute_subgame_leaf_values_c(
    leaf_infos: List,
    board: List[int],
    street: str,
    player_hands: List[List[Tuple[int, int, float]]],
    active: List[bool],
    acting_order: List[int],
    player_seats: List[int],
    game_action_history: List[int],
    blueprint,
    starting_pot_chips: int,   # unused here; kept for signature parity with Python
    stack_chips: int,
    max_hands: int,
    num_rollouts: int = 3,
    rng=None,                  # accepted for drop-in compat with python/rollout.py
    master_seed: int = 0x5a17_e8c0_feed_beef,
) -> np.ndarray:
    """Drop-in faster replacement for python/rollout.py::compute_subgame_leaf_values.

    Returns float32 numpy array of shape [total, num_players, max_hands]
    where total = (max leaf_idx + 1) * (4 ** num_players).

    Blueprint coverage misses are absorbed inline via uniform-on-legal-
    actions (Pluribus Algorithm 1 supplementary); call
    get_rollout_observability() to count substitutions. Raises
    LeafValueUnavailable on genuine errors (deck exhausted, board
    overflow, bucket lookup failure).
    """
    if street not in _STREET_NUM:
        raise ValueError(f"bad street {street!r}")
    # `rng` kwarg is accepted for drop-in-compat with python/rollout.py
    # but unused — the C path uses per-tuple splitmix64 from master_seed.
    # If caller passes a numpy Generator, optionally derive a 64-bit
    # seed from it so two calls with the same generator produce the
    # same C-side stream.
    if rng is not None:
        try:
            master_seed = int(rng.integers(0, 2**64 - 1, dtype=np.uint64))
        except Exception:
            pass
    lib = _lib()

    # BPSReader * — the existing BPSReaderC exposes this as self._reader.
    if not hasattr(blueprint, "_reader"):
        raise TypeError(
            "C rollout requires a BPSReaderC blueprint (has ._reader handle); "
            f"got {type(blueprint).__name__}. Fall back to python/rollout.py "
            f"for other blueprint types."
        )
    bp_ptr = blueprint._reader

    num_players = len(player_hands)
    if num_players <= 0 or num_players > _ROLLOUT_MAX_PLAYERS:
        raise ValueError(f"num_players {num_players} out of range [1, 6]")

    cont_per_leaf = 4 ** num_players
    max_leaf_idx = max((li.leaf_idx for li in leaf_infos), default=-1)
    total = (max_leaf_idx + 1) * cont_per_leaf
    floats_needed = total * num_players * max_hands

    out = np.zeros((total, num_players, max_hands), dtype=np.float32)

    # Pack inputs
    board_arr = (ctypes.c_int * len(board))(*board) if board else None
    board_ptr = board_arr if board_arr is not None else ctypes.POINTER(ctypes.c_int)()

    # leaf_infos → RolloutLeafInfo array. flop_history_train is a tuple
    # of training-encoded action indices produced by
    # python/leaf_values.py::extract_leaf_info_from_tree; missing on
    # legacy LeafInfo objects (treated as empty → falls back to the old
    # dropped-history behavior).
    n_leaves = len(leaf_infos)
    li_arr = (_RolloutLeafInfo * n_leaves)()
    for i, li in enumerate(leaf_infos):
        li_arr[i].leaf_idx = int(li.leaf_idx)
        li_arr[i].pot_chips = int(li.pot)
        bets_src = list(li.bets) if hasattr(li, "bets") else [0] * num_players
        for p in range(_ROLLOUT_MAX_PLAYERS):
            li_arr[i].bets[p] = int(bets_src[p]) if p < num_players else 0
        fh = getattr(li, "flop_history_train", ()) or ()
        if len(fh) > _ROLLOUT_MAX_ACTION_HISTORY:
            raise ValueError(
                f"leaf {i}: flop_history_train length {len(fh)} exceeds "
                f"ROLLOUT_MAX_ACTION_HISTORY={_ROLLOUT_MAX_ACTION_HISTORY}"
            )
        for j, idx in enumerate(fh):
            li_arr[i].flop_history[j] = int(idx)
        li_arr[i].flop_history_len = len(fh)

    active_arr = (ctypes.c_int32 * num_players)(
        *[1 if a else 0 for a in active]
    )

    # player_hands: array of RolloutHandList + stash for the per-player RolloutHand buffers
    stash: list = []
    hl_arr = (_RolloutHandList * num_players)()
    for p in range(num_players):
        hl = _hand_list_struct(player_hands[p], stash)
        hl_arr[p] = hl

    acting_order_arr = (ctypes.c_int32 * len(acting_order))(*acting_order)
    player_seats_arr = (ctypes.c_int32 * len(player_seats))(*player_seats)
    gah_arr = (ctypes.c_int32 * len(game_action_history))(*game_action_history) \
        if game_action_history else (ctypes.c_int32 * 0)()

    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    rc = lib.rollout_compute_subgame_leaf_values_c(
        bp_ptr,
        board_ptr, ctypes.c_int32(len(board)),
        ctypes.c_int32(_STREET_NUM[street]),
        li_arr, ctypes.c_int32(n_leaves),
        active_arr, ctypes.c_int32(num_players),
        hl_arr,
        acting_order_arr, ctypes.c_int32(len(acting_order)),
        player_seats_arr,
        gah_arr, ctypes.c_int32(len(game_action_history)),
        ctypes.c_int32(int(stack_chips)),
        ctypes.c_int32(int(max_hands)),
        ctypes.c_int32(int(num_rollouts)),
        ctypes.c_uint64(int(master_seed) & 0xFFFFFFFFFFFFFFFF),
        out_ptr, ctypes.c_int32(int(floats_needed)),
    )
    if rc != 0:
        msg, street_n, np_n = _last_error()
        raise LeafValueUnavailable(
            street=street, num_players=num_players,
            reason=f"[rollout.c] {msg}",
        )
    return out


def compute_subgame_leaf_values_sampled_c(
    leaf_infos: List,
    board: List[int],
    street: str,
    player_hands: List[List[Tuple[int, int, float]]],
    active: List[bool],
    acting_order: List[int],
    player_seats: List[int],
    game_action_history: List[int],
    blueprint,
    starting_pot_chips: int,
    stack_chips: int,
    max_hands: int,
    num_rollouts: int = 3,
    num_samples_k: int = 1,
    rng=None,
    master_seed: int = 0x5a17_e8c0_feed_beef,
) -> np.ndarray:
    """Drop-in replacement for python/rollout.py::compute_subgame_leaf_values_sampled.

    Returns float32 numpy array of shape [total, num_players, max_hands]
    where total = max leaf_idx + 1 (no cont-strat axis — SAMPLED mode).

    Pairs with the SAMPLED tree builder (SS_CONT_STRAT_MODE==SAMPLED in
    street_solve.cuh). See docs/OP4_CONT_STRAT_MARGINAL_PLAN.md.

    Blueprint coverage misses are absorbed inline via uniform-on-legal-
    actions (Pluribus Algorithm 1 supplementary); call
    get_rollout_observability() to count substitutions. Raises
    LeafValueUnavailable on genuine errors only.
    """
    if street not in _STREET_NUM:
        raise ValueError(f"bad street {street!r}")
    if num_samples_k < 1:
        raise ValueError(f"num_samples_k must be >= 1, got {num_samples_k}")

    if rng is not None:
        try:
            master_seed = int(rng.integers(0, 2**64 - 1, dtype=np.uint64))
        except Exception:
            pass
    lib = _lib()

    if not hasattr(blueprint, "_reader"):
        raise TypeError(
            "C rollout requires a BPSReaderC blueprint (has ._reader handle); "
            f"got {type(blueprint).__name__}. Fall back to python/rollout.py "
            f"for other blueprint types."
        )
    bp_ptr = blueprint._reader

    num_players = len(player_hands)
    if num_players <= 0 or num_players > _ROLLOUT_MAX_PLAYERS:
        raise ValueError(f"num_players {num_players} out of range [1, 6]")

    max_leaf_idx = max((li.leaf_idx for li in leaf_infos), default=-1)
    total = max_leaf_idx + 1
    floats_needed = total * num_players * max_hands

    out = np.zeros((total, num_players, max_hands), dtype=np.float32)

    board_arr = (ctypes.c_int * len(board))(*board) if board else None
    board_ptr = board_arr if board_arr is not None else ctypes.POINTER(ctypes.c_int)()

    n_leaves = len(leaf_infos)
    li_arr = (_RolloutLeafInfo * n_leaves)()
    for i, li in enumerate(leaf_infos):
        li_arr[i].leaf_idx = int(li.leaf_idx)
        li_arr[i].pot_chips = int(li.pot)
        bets_src = list(li.bets) if hasattr(li, "bets") else [0] * num_players
        for p in range(_ROLLOUT_MAX_PLAYERS):
            li_arr[i].bets[p] = int(bets_src[p]) if p < num_players else 0

    active_arr = (ctypes.c_int32 * num_players)(
        *[1 if a else 0 for a in active]
    )

    stash: list = []
    hl_arr = (_RolloutHandList * num_players)()
    for p in range(num_players):
        hl = _hand_list_struct(player_hands[p], stash)
        hl_arr[p] = hl

    acting_order_arr = (ctypes.c_int32 * len(acting_order))(*acting_order)
    player_seats_arr = (ctypes.c_int32 * len(player_seats))(*player_seats)
    gah_arr = (ctypes.c_int32 * len(game_action_history))(*game_action_history) \
        if game_action_history else (ctypes.c_int32 * 0)()

    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    rc = lib.rollout_compute_subgame_leaf_values_sampled_c(
        bp_ptr,
        board_ptr, ctypes.c_int32(len(board)),
        ctypes.c_int32(_STREET_NUM[street]),
        li_arr, ctypes.c_int32(n_leaves),
        active_arr, ctypes.c_int32(num_players),
        hl_arr,
        acting_order_arr, ctypes.c_int32(len(acting_order)),
        player_seats_arr,
        gah_arr, ctypes.c_int32(len(game_action_history)),
        ctypes.c_int32(int(stack_chips)),
        ctypes.c_int32(int(max_hands)),
        ctypes.c_int32(int(num_rollouts)),
        ctypes.c_int32(int(num_samples_k)),
        ctypes.c_uint64(int(master_seed) & 0xFFFFFFFFFFFFFFFF),
        out_ptr, ctypes.c_int32(int(floats_needed)),
    )
    if rc != 0:
        msg, street_n, np_n = _last_error()
        raise LeafValueUnavailable(
            street=street, num_players=num_players,
            reason=f"[rollout.c sampled] {msg}",
        )
    return out


# ─── Phase F Step 14: cont_ev_table consumer ──────────────────────────


def set_rollout_cont_ev_table(table_ptr: Optional[int]) -> None:
    """Hand the rollout a cont_ev_table pointer (or None to clear).

    `table_ptr` is the integer address of a populated BPContEVTable —
    typically obtained via `load_cont_ev_table_from_bpr5` below. The
    rollout stores the pointer in process-global state; the table must
    outlive any rollout call that may consult it.

    Process-global. In multiprocessing fork pools, set BEFORE forking
    so workers inherit the same pointer via COW (the table itself is
    read-only at consume time).
    """
    lib = _lib()
    if not hasattr(lib, "rollout_set_cont_ev_table"):
        raise RuntimeError(
            "rollout.so missing rollout_set_cont_ev_table — rebuild with "
            "Phase F changes (src/rollout.c Step 14)."
        )
    lib.rollout_set_cont_ev_table(
        ctypes.c_void_p(table_ptr if table_ptr else 0)
    )


def get_rollout_cont_ev_table() -> Optional[int]:
    """Return the currently-set cont_ev_table pointer, or None."""
    lib = _lib()
    if not hasattr(lib, "rollout_get_cont_ev_table"):
        return None
    p = lib.rollout_get_cont_ev_table()
    return int(p) if p else None


def set_use_cont_ev_at_flop_boundary(enabled: bool) -> None:
    """Enable / disable the end-of-flop multiway boundary lookup.

    When enabled (and a table pointer is set), the rollout short-
    circuits at the boundary: queries cont_ev_table for hero's slot
    at all 4 traverser-bias indices, returns the uniform mean as the
    leaf payoff. MISS / EMPTY_BIAS / INVALID is fail-loud (raises
    LeafValueUnavailable downstream).

    When disabled (default), the rollout runs legacy turn + river
    simulation regardless of table state. Process-global flag.
    """
    lib = _lib()
    if not hasattr(lib, "rollout_set_use_cont_ev_at_flop_boundary"):
        raise RuntimeError(
            "rollout.so missing rollout_set_use_cont_ev_at_flop_boundary — "
            "rebuild with Phase F changes (src/rollout.c Step 14)."
        )
    lib.rollout_set_use_cont_ev_at_flop_boundary(1 if enabled else 0)


def get_use_cont_ev_at_flop_boundary() -> bool:
    lib = _lib()
    if not hasattr(lib, "rollout_get_use_cont_ev_at_flop_boundary"):
        return False
    return bool(lib.rollout_get_use_cont_ev_at_flop_boundary())


def get_cont_ev_boundary_hit_count() -> int:
    """Number of times the boundary lookup successfully short-circuited
    in the current thread. MISS / EMPTY_BIAS / INVALID outcomes do NOT
    increment (they raise via rollout_set_error). Reset between solves
    for scoped reporting via reset_cont_ev_boundary_hit_count.
    """
    lib = _lib()
    if not hasattr(lib, "rollout_get_cont_ev_boundary_hit_count"):
        return 0
    out = ctypes.c_uint64(0)
    lib.rollout_get_cont_ev_boundary_hit_count(ctypes.byref(out))
    return int(out.value)


def reset_cont_ev_boundary_hit_count() -> None:
    lib = _lib()
    if not hasattr(lib, "rollout_reset_cont_ev_boundary_hit_count"):
        return
    lib.rollout_reset_cont_ev_boundary_hit_count()


# ── BPR5 trailer loader (mccfr_blueprint.so) ────────────────────────────
#
# To get a populated cont_ev_table pointer, we construct a BPSolver with
# enough configuration to mirror the trainer's slot key space (preflop
# tiers + add_explicit_allin), then call bp_load_regrets — which parses
# the BPR4 entries (most dropped if hash_table_size is small) and
# loads the BPR5 trailer at the end into s->cont_ev_table.
#
# Use a smaller hash_table_size than training (e.g., 128M instead of
# 4B) — this lets the BPInfoTable fit in ~10 GB instead of 256 GB.
# The vast majority of BPR4 entries fail to insert (counted via
# insertion_failures); they're irrelevant for our consumer (we only
# want the cont_ev_table). The trailer loads independently at the end.


_PREFLOP_TIERS_PHASE3 = {
    0: [0.5, 0.7, 1.0],   # open raise
    1: [0.7, 1.0],        # 3-bet
    2: [1.0],             # 4-bet
}
_PREFLOP_MAX_RAISES_PHASE3 = 3
_POSTFLOP_BET_SIZES_PHASE3 = [0.5, 1.0]


class _BPConfig(ctypes.Structure):
    """Mirror of BPConfig in src/mccfr_blueprint.h. Match field order
    EXACTLY — see precompute/export_v2.py for the canonical list (the
    six int64_t iter/interval fields were silently broken for ~8 months
    when the Python mirror used `int`; we match the corrected layout)."""
    _fields_ = [
        ("discount_stop_iter",  ctypes.c_int64),
        ("discount_interval",   ctypes.c_int64),
        ("prune_start_iter",    ctypes.c_int64),
        ("snapshot_start_iter", ctypes.c_int64),
        ("snapshot_interval",   ctypes.c_int64),
        ("strategy_interval",   ctypes.c_int64),
        ("num_threads",         ctypes.c_int),
        ("hash_table_size",     ctypes.c_int64),
        ("snapshot_dir",        ctypes.c_char_p),
        ("include_preflop",     ctypes.c_int),
        ("postflop_num_buckets", ctypes.c_int),
    ]


class ContEVTableHolder:
    """Owns a BPSolver allocated for a cont_ev_table load. The table
    pointer is valid as long as this holder is alive. `__del__` calls
    bp_free to release the BPInfoTable + cont_ev_table memory.

    Get the table pointer via `holder.cont_ev_table_ptr` and pass it to
    `set_rollout_cont_ev_table`. Lifetime contract: keep the holder
    object alive for the duration of any rollout calls that may consult
    the table.
    """

    def __init__(self, bp_lib: ctypes.CDLL, solver_buf, solver_ptr: int,
                 table_ptr: int, n_slots: int, bin_path: str):
        self._bp = bp_lib
        self._solver_buf = solver_buf
        self._solver_ptr = solver_ptr
        self.cont_ev_table_ptr: int = table_ptr
        self.n_slots: int = n_slots
        self.bin_path: str = bin_path
        self._freed = False

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def free(self) -> None:
        if self._freed:
            return
        self._freed = True
        try:
            self._bp.bp_free(ctypes.c_void_p(self._solver_ptr))
        except Exception:
            pass


def _find_blueprint_lib() -> ctypes.CDLL:
    base = Path(__file__).resolve().parent.parent / "build"
    ext = ".dll" if sys.platform == "win32" else ".so"
    candidates = [base / f"mccfr_blueprint{ext}"]
    env_path = os.environ.get("POKER_SOLVER_PATH")
    if env_path:
        candidates.append(Path(env_path) / "build" / f"mccfr_blueprint{ext}")
    for p in candidates:
        if p.exists():
            return ctypes.CDLL(str(p))
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"mccfr_blueprint shared library not found. Searched: {searched}. "
        f"Run 'make blueprint' in the poker-solver directory."
    )


def load_cont_ev_table_from_bpr5(
    bin_path: str,
    *,
    hash_table_size: int = 128_000_000,
    num_threads: int = 1,
    num_players: int = 6,
    sb: int = 50,
    bb: int = 100,
    initial_stack: int = 10000,
    add_explicit_allin: bool = False,
    verbose: bool = True,
) -> ContEVTableHolder:
    """Load a Step 10–style BPR4+BPR5 .bin file and return a holder
    with the cont_ev_table pointer ready for `set_rollout_cont_ev_table`.

    Mirrors the loader pattern in
    tools/phase_r_probe_analysis/verify_bpr5_roundtrip.c — uses the
    same v3/phase3 preflop tier setup so action_hash key space matches
    the trained file.

    `hash_table_size` defaults to 128M (≈10 GB BPInfoTable) — much
    smaller than training's 4B (256 GB). Most BPR4 regret entries will
    fail to insert (counted as insertion_failures); they're irrelevant
    for our consumer. The BPR5 trailer at the end of the file loads
    INDEPENDENTLY into s->cont_ev_table.

    `add_explicit_allin=False` matches v3/phase3 (deep-stack nodes
    have no allin in the trained action set; trainer spec §4 +
    bp_set_add_explicit_allin(0) at verify_bpr5_roundtrip.c:73).

    Raises FileNotFoundError if the .bin doesn't exist; RuntimeError
    on any C-side init/load failure.
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"BPR5 .bin not found: {bin_path}")

    # Step 15 instrumentation: every milestone gets a BEFORE/AFTER
    # print with wall timing and flush=True so the live S3 uploader
    # can show progress mid-run. Step 14 attempt 6 ran 2 hours silent
    # because we had no visibility into which step was the long pole.
    _t_load_start = time.monotonic()

    def _log(msg: str) -> None:
        if verbose:
            print(f"[cont_ev_loader t+{time.monotonic() - _t_load_start:6.1f}s] {msg}",
                  flush=True)

    _log(f"BEGIN load_cont_ev_table_from_bpr5(bin_path={bin_path!r}, "
         f"hash_table_size={hash_table_size:,}, num_threads={num_threads})")

    _t = time.monotonic()
    _log("BEFORE _find_blueprint_lib (CDLL load)")
    bp = _find_blueprint_lib()
    _log(f"AFTER  _find_blueprint_lib ({time.monotonic() - _t:.2f}s)")

    # Wire up the bp_* APIs we need.
    bp.bp_default_config.argtypes = [ctypes.c_void_p]
    bp.bp_default_config.restype = None
    bp.bp_init_unified.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_void_p,
    ]
    bp.bp_init_unified.restype = ctypes.c_int
    bp.bp_set_preflop_tier.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int,
    ]
    bp.bp_set_preflop_tier.restype = ctypes.c_int
    bp.bp_set_add_explicit_allin.argtypes = [ctypes.c_void_p, ctypes.c_int]
    bp.bp_set_add_explicit_allin.restype = ctypes.c_int
    bp.bp_load_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    bp.bp_load_regrets.restype = ctypes.c_int64
    bp.bp_get_cont_ev_table_ptr.argtypes = [ctypes.c_void_p]
    bp.bp_get_cont_ev_table_ptr.restype = ctypes.c_void_p
    bp.bp_free.argtypes = [ctypes.c_void_p]
    bp.bp_free.restype = None

    # Slot count getter (defined in mccfr_cont_ev_table.c, exported).
    if not hasattr(bp, "bp_cont_ev_count"):
        raise RuntimeError(
            "mccfr_blueprint.so missing bp_cont_ev_count — "
            "the linked mccfr_cont_ev_table.c is too old."
        )
    bp.bp_cont_ev_count.argtypes = [ctypes.c_void_p]
    bp.bp_cont_ev_count.restype = ctypes.c_int64

    # Allocate the BPSolver struct (2 MB ought to be plenty; verify_bpr5
    # uses the same).
    solver_buf = (ctypes.c_char * (2 * 1024 * 1024))()
    solver_ptr_int = ctypes.cast(solver_buf, ctypes.c_void_p).value or 0

    # Configure.
    config = _BPConfig()
    bp.bp_default_config(ctypes.byref(config))
    config.num_threads = int(num_threads)
    config.include_preflop = 1
    config.hash_table_size = int(hash_table_size)
    config.postflop_num_buckets = 200

    c_post = (ctypes.c_float * len(_POSTFLOP_BET_SIZES_PHASE3))(
        *_POSTFLOP_BET_SIZES_PHASE3)
    # bp_init_unified takes a "preflop bet sizes" template too; the
    # actual preflop tree comes from bp_set_preflop_tier calls below.
    pf_template = _PREFLOP_TIERS_PHASE3[0]
    c_pre = (ctypes.c_float * len(pf_template))(*pf_template)

    _t = time.monotonic()
    _log(f"BEFORE bp_init_unified (hash_size={hash_table_size:,}, "
         f"threads={num_threads}, players={num_players}, "
         f"sb/bb={sb}/{bb}, stack={initial_stack})")
    rc = bp.bp_init_unified(
        ctypes.c_void_p(solver_ptr_int), num_players,
        sb, bb, initial_stack,
        c_post, len(_POSTFLOP_BET_SIZES_PHASE3),
        c_pre, len(pf_template),
        ctypes.byref(config),
    )
    _log(f"AFTER  bp_init_unified rc={rc} ({time.monotonic() - _t:.2f}s)")
    if rc != 0:
        raise RuntimeError(f"bp_init_unified rc={rc}")

    # Apply tiered preflop sizing (mirrors verify_bpr5_roundtrip.c:65-72).
    for level, sizes in sorted(_PREFLOP_TIERS_PHASE3.items()):
        _t = time.monotonic()
        _log(f"BEFORE bp_set_preflop_tier(level={level}, "
             f"sizes={list(sizes)}, max_raises={_PREFLOP_MAX_RAISES_PHASE3})")
        c_sizes = (ctypes.c_float * len(sizes))(*sizes)
        rc = bp.bp_set_preflop_tier(
            ctypes.c_void_p(solver_ptr_int), level,
            c_sizes, len(sizes), _PREFLOP_MAX_RAISES_PHASE3,
        )
        _log(f"AFTER  bp_set_preflop_tier(level={level}) "
             f"rc={rc} ({time.monotonic() - _t:.2f}s)")
        if rc != 0:
            bp.bp_free(ctypes.c_void_p(solver_ptr_int))
            raise RuntimeError(f"bp_set_preflop_tier level {level} rc={rc}")

    _t = time.monotonic()
    _log(f"BEFORE bp_set_add_explicit_allin({1 if add_explicit_allin else 0})")
    rc = bp.bp_set_add_explicit_allin(
        ctypes.c_void_p(solver_ptr_int), 1 if add_explicit_allin else 0)
    _log(f"AFTER  bp_set_add_explicit_allin rc={rc} "
         f"({time.monotonic() - _t:.2f}s)")
    if rc != 0:
        bp.bp_free(ctypes.c_void_p(solver_ptr_int))
        raise RuntimeError(f"bp_set_add_explicit_allin rc={rc}")

    # Load regrets — populates BPInfoTable (most entries dropped if hash
    # too small) AND, at end-of-file, the BPR5 trailer into
    # s->cont_ev_table.
    file_gb = os.path.getsize(bin_path) / (1024 ** 3)
    _t = time.monotonic()
    _log(f"BEFORE bp_load_regrets({bin_path}, {file_gb:.2f} GB) "
         f"— this is the long pole (~minutes); if stuck here the BPR4 "
         f"insertion-failures hash thrash is the suspect")
    n_loaded = bp.bp_load_regrets(
        ctypes.c_void_p(solver_ptr_int), bin_path.encode("utf-8"))
    _log(f"AFTER  bp_load_regrets n_loaded={n_loaded} "
         f"({time.monotonic() - _t:.2f}s, "
         f"{file_gb / max(time.monotonic() - _t, 0.001):.2f} GB/s)")
    if n_loaded < 0:
        bp.bp_free(ctypes.c_void_p(solver_ptr_int))
        raise RuntimeError(f"bp_load_regrets rc={n_loaded}")

    _t = time.monotonic()
    _log("BEFORE bp_get_cont_ev_table_ptr")
    table_ptr = bp.bp_get_cont_ev_table_ptr(ctypes.c_void_p(solver_ptr_int))
    table_ptr_int = int(table_ptr) if table_ptr else 0
    _log(f"AFTER  bp_get_cont_ev_table_ptr ptr=0x{table_ptr_int:x} "
         f"({time.monotonic() - _t:.2f}s)")
    if not table_ptr_int:
        bp.bp_free(ctypes.c_void_p(solver_ptr_int))
        raise RuntimeError("bp_get_cont_ev_table_ptr returned NULL")

    _t = time.monotonic()
    _log("BEFORE bp_cont_ev_count")
    n_slots = int(bp.bp_cont_ev_count(ctypes.c_void_p(table_ptr_int)))
    _log(f"AFTER  bp_cont_ev_count n_slots={n_slots:,} "
         f"({time.monotonic() - _t:.2f}s)")
    if n_slots == 0:
        bp.bp_free(ctypes.c_void_p(solver_ptr_int))
        raise RuntimeError(
            f"BPR5 trailer present but 0 slots loaded from {bin_path} — "
            f"either the file has no trailer (legacy BPR4) or load was "
            f"truncated."
        )

    _log(f"END   load_cont_ev_table_from_bpr5 — {n_slots:,} slots, "
         f"{n_loaded:,} BPR4 entries (most dropped — hash too small), "
         f"total {time.monotonic() - _t_load_start:.2f}s")

    return ContEVTableHolder(
        bp_lib=bp, solver_buf=solver_buf, solver_ptr=solver_ptr_int,
        table_ptr=table_ptr_int, n_slots=n_slots, bin_path=bin_path,
    )

