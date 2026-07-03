"""ctypes binding for src/rollout.c — the C port of python/rollout.py.

Produces the same output array shape as compute_subgame_leaf_values,
but orders of magnitude faster. The Python module remains the
correctness reference; this module is the shipping path for
nexusgto-api's live flop solves.

See docs/C2_CPU_ROLLOUT_PLAN.md for the port's design, known parity-
with-Python limitations, and test plan.

Usage:

    from engine.bps_reader_py import BPSReaderC
    from engine.rollout_c import compute_subgame_leaf_values_c

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
    from engine.leaf_values import LeafValueUnavailable  # type: ignore[import-untyped]
except ImportError:
    from engine.leaf_values import LeafValueUnavailable  # type: ignore[no-redef]


# Must match ROLLOUT_MAX_PLAYERS in src/rollout.h
_ROLLOUT_MAX_PLAYERS = 6
# Must match ROLLOUT_MAX_ACTION_HISTORY in src/rollout.h
_ROLLOUT_MAX_ACTION_HISTORY = 256
# Must match ROLLOUT_MAX_RAISE_SIZES in src/rollout.h
_ROLLOUT_MAX_RAISE_SIZES = 8


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


def _pack_leaf_flop_history(dst, li, leaf_idx_for_error: int) -> None:
    """Copy a leaf's training-encoded flop action history into the C struct.

    flop_history_train is the this-street action-index path the trainer used to
    KEY the blueprint's postflop nodes; the rollout concatenates it after the
    preflop game_action_history to reconstruct the blueprint key for its
    continuation queries. BOTH the sampled and non-sampled wrappers must pack it
    — omitting it (the sampled-path bug, 2026-06-16) drops the postflop key and
    every continuation query misses -> uniform fallback -> over-bet. Missing on
    legacy LeafInfo objects is tolerated (empty -> old dropped-history behavior).
    """
    fh = getattr(li, "flop_history_train", ()) or ()
    if len(fh) > _ROLLOUT_MAX_ACTION_HISTORY:
        raise ValueError(
            f"leaf {leaf_idx_for_error}: flop_history_train length {len(fh)} "
            f"exceeds ROLLOUT_MAX_ACTION_HISTORY={_ROLLOUT_MAX_ACTION_HISTORY}"
        )
    for j, idx in enumerate(fh):
        dst.flop_history[j] = int(idx)
    dst.flop_history_len = len(fh)


# ─── Library loader ──────────────────────────────────────────────────────


def _find_lib() -> ctypes.CDLL:
    base = Path(__file__).resolve().parent.parent.parent / "build"  # engine/ -> python/ -> repo root (2026-06 relocation)
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

    # Per-leaf conditioned variant (2026-06-17): same as sampled_c + an extra
    # per_leaf_player_hands array (n_leaves*num_players RolloutHandList). Optional
    # symbol — only the 2p-flop range-conditioning path binds it.
    if hasattr(lib, "rollout_compute_subgame_leaf_values_sampled_perleaf_c"):
        lib.rollout_compute_subgame_leaf_values_sampled_perleaf_c.argtypes = [
            ctypes.c_void_p,                    # BPSReader *
            ctypes.POINTER(ctypes.c_int),       # board
            ctypes.c_int32,                     # n_board_cards
            ctypes.c_int32,                     # street
            ctypes.POINTER(_RolloutLeafInfo),   # leaf_infos
            ctypes.c_int32,                     # n_leaves
            ctypes.POINTER(ctypes.c_int32),     # active
            ctypes.c_int32,                     # num_players
            ctypes.POINTER(_RolloutHandList),   # player_hands (shared fallback)
            ctypes.POINTER(_RolloutHandList),   # per_leaf_player_hands
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
        lib.rollout_compute_subgame_leaf_values_sampled_perleaf_c.restype = \
            ctypes.c_int32

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
    if hasattr(lib, "rollout_get_cont_ev_boundary_attempt_count"):
        lib.rollout_get_cont_ev_boundary_attempt_count.argtypes = [
            ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.rollout_get_cont_ev_boundary_attempt_count.restype = None

    # SYNTHETIC mode (DEBUG/MEASUREMENT — armed-residual probe). Applies to
    # both boundaries; default OFF.
    if hasattr(lib, "rollout_set_cont_ev_synthetic"):
        lib.rollout_set_cont_ev_synthetic.argtypes = [ctypes.c_int]
        lib.rollout_set_cont_ev_synthetic.restype = None
    if hasattr(lib, "rollout_get_cont_ev_synthetic"):
        lib.rollout_get_cont_ev_synthetic.argtypes = []
        lib.rollout_get_cont_ev_synthetic.restype = ctypes.c_int

    # Dual-boundary build (2026-06-25): TURN-boundary consumer. Mirrors the
    # flop bindings above; the SAME table (rollout_set_cont_ev_table) backs
    # both boundaries. See src/search/rollout.h "Dual-boundary" block.
    if hasattr(lib, "rollout_set_use_cont_ev_at_turn_boundary"):
        lib.rollout_set_use_cont_ev_at_turn_boundary.argtypes = [ctypes.c_int]
        lib.rollout_set_use_cont_ev_at_turn_boundary.restype = None
    if hasattr(lib, "rollout_get_use_cont_ev_at_turn_boundary"):
        lib.rollout_get_use_cont_ev_at_turn_boundary.argtypes = []
        lib.rollout_get_use_cont_ev_at_turn_boundary.restype = ctypes.c_int
    if hasattr(lib, "rollout_get_cont_ev_turn_boundary_hit_count"):
        lib.rollout_get_cont_ev_turn_boundary_hit_count.argtypes = [
            ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.rollout_get_cont_ev_turn_boundary_hit_count.restype = None
    if hasattr(lib, "rollout_get_cont_ev_turn_boundary_attempt_count"):
        lib.rollout_get_cont_ev_turn_boundary_attempt_count.argtypes = [
            ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.rollout_get_cont_ev_turn_boundary_attempt_count.restype = None
    if hasattr(lib, "rollout_reset_cont_ev_turn_boundary_hit_count"):
        lib.rollout_reset_cont_ev_turn_boundary_hit_count.argtypes = []
        lib.rollout_reset_cont_ev_turn_boundary_hit_count.restype = None

    # Unit 3 (Stage-1 mean-field, 2026-07-01): bias-as-hero-action consumer
    # mode. Optional symbols — hasattr-guarded like the turn-boundary bindings
    # so a pre-Unit-3 .so still loads (the wrapper raises if absent). See
    # src/search/rollout.h "Unit 3" block.
    if hasattr(lib, "rollout_set_cont_ev_bias_action"):
        lib.rollout_set_cont_ev_bias_action.argtypes = [ctypes.c_int]
        lib.rollout_set_cont_ev_bias_action.restype = None
    if hasattr(lib, "rollout_get_cont_ev_bias_action"):
        lib.rollout_get_cont_ev_bias_action.argtypes = []
        lib.rollout_get_cont_ev_bias_action.restype = ctypes.c_int
    if hasattr(lib, "rollout_get_cont_ev_bias_action_stats"):
        lib.rollout_get_cont_ev_bias_action_stats.argtypes = [
            ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.rollout_get_cont_ev_bias_action_stats.restype = None
    if hasattr(lib, "rollout_reset_cont_ev_bias_regrets"):
        lib.rollout_reset_cont_ev_bias_regrets.argtypes = []
        lib.rollout_reset_cont_ev_bias_regrets.restype = None

    # Stage-2' pinned-tail MISS fallback (2026-07-03, MEASUREMENT ONLY,
    # default OFF): on MISS/EMPTY_BIAS the querying rollout falls through to
    # the legacy live playout instead of failing loud. Optional symbols —
    # hasattr-guarded like the turn/bias-action bindings above. See
    # src/search/rollout.h "Stage-2' pinned-tail" block.
    if hasattr(lib, "rollout_set_cont_ev_miss_fallback"):
        lib.rollout_set_cont_ev_miss_fallback.argtypes = [ctypes.c_int]
        lib.rollout_set_cont_ev_miss_fallback.restype = None
    if hasattr(lib, "rollout_get_cont_ev_miss_fallback"):
        lib.rollout_get_cont_ev_miss_fallback.argtypes = []
        lib.rollout_get_cont_ev_miss_fallback.restype = ctypes.c_int
    if hasattr(lib, "rollout_get_cont_ev_fallback_count"):
        lib.rollout_get_cont_ev_fallback_count.argtypes = [
            ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.rollout_get_cont_ev_fallback_count.restype = None
    if hasattr(lib, "rollout_get_cont_ev_turn_fallback_count"):
        lib.rollout_get_cont_ev_turn_fallback_count.argtypes = [
            ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.rollout_get_cont_ev_turn_fallback_count.restype = None
    if hasattr(lib, "rollout_reset_cont_ev_fallback_counts"):
        lib.rollout_reset_cont_ev_fallback_counts.argtypes = []
        lib.rollout_reset_cont_ev_fallback_counts.restype = None

    # Training bet-vocab setter (bet-vocab desync fix, 2026-07-01). Optional
    # symbol — hasattr-guarded like the turn-boundary bindings above, so a
    # pre-fix .so still loads (the caller's set_rollout_train_vocab raises if
    # the symbol is absent). See src/search/rollout.h.
    if hasattr(lib, "rollout_set_train_vocab"):
        lib.rollout_set_train_vocab.argtypes = [
            ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
        ]
        lib.rollout_set_train_vocab.restype = ctypes.c_int32
    if hasattr(lib, "rollout_get_train_vocab"):
        lib.rollout_get_train_vocab.argtypes = [
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
        ]
        lib.rollout_get_train_vocab.restype = None

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


# ─── Training bet-vocab setter (bet-vocab desync fix, 2026-07-01) ─────────


def set_rollout_train_vocab(
    first_fracs: Sequence[float],
    subseq_fracs: Sequence[float],
) -> None:
    """Set the postflop bet vocabulary the C rollout enumerates.

    The rollout must enumerate the SAME first/subsequent postflop raise sizes
    the served blueprint was TRAINED with, or its action-index sequence hashes
    to a key training never stored (all-in mis-key — see
    docs/BET_VOCAB_DESYNC_FINDING_2026-06-26.md). The caller reads
    ``first_fracs`` / ``subseq_fracs`` from the loaded .bps manifest
    (``postflop_bet_sizes`` / ``subsequent_bet_sizes``) and calls this ONCE at
    startup, before any rollout.

    Fail-loud: raises RuntimeError if the .so predates the fix (symbol absent),
    ValueError on an empty/oversized list, and RuntimeError if the C setter
    rejects the args. Never silently no-ops.
    """
    ff = [float(x) for x in first_fracs]
    sf = [float(x) for x in subseq_fracs]
    if not ff or len(ff) > _ROLLOUT_MAX_RAISE_SIZES:
        raise ValueError(
            f"first_fracs must be 1..{_ROLLOUT_MAX_RAISE_SIZES} sizes, "
            f"got {ff}"
        )
    if not sf or len(sf) > _ROLLOUT_MAX_RAISE_SIZES:
        raise ValueError(
            f"subseq_fracs must be 1..{_ROLLOUT_MAX_RAISE_SIZES} sizes, "
            f"got {sf}"
        )
    lib = _lib()
    if not hasattr(lib, "rollout_set_train_vocab"):
        raise RuntimeError(
            "rollout.so missing rollout_set_train_vocab — rebuild with the "
            "bet-vocab desync fix (src/search/rollout.c, 2026-07-01)."
        )
    ff_arr = (ctypes.c_float * len(ff))(*ff)
    sf_arr = (ctypes.c_float * len(sf))(*sf)
    rc = lib.rollout_set_train_vocab(
        ctypes.c_int32(len(ff)),
        ctypes.cast(ff_arr, ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int32(len(sf)),
        ctypes.cast(sf_arr, ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(
            f"rollout_set_train_vocab rejected vocab first={ff} subseq={sf} "
            f"(rc={rc})"
        )


def get_rollout_train_vocab() -> Tuple[List[float], List[float]]:
    """Read back the C rollout's currently-configured training vocab.

    Returns (first_fracs, subseq_fracs). Returns the compiled defaults
    ([0.5, 1.0], [1.0]) if the setter was never called; raises RuntimeError
    if the .so predates the fix.
    """
    lib = _lib()
    if not hasattr(lib, "rollout_get_train_vocab"):
        raise RuntimeError(
            "rollout.so missing rollout_get_train_vocab — rebuild with the "
            "bet-vocab desync fix (src/search/rollout.c, 2026-07-01)."
        )
    n_first = ctypes.c_int32(0)
    n_subseq = ctypes.c_int32(0)
    ff_arr = (ctypes.c_float * _ROLLOUT_MAX_RAISE_SIZES)()
    sf_arr = (ctypes.c_float * _ROLLOUT_MAX_RAISE_SIZES)()
    lib.rollout_get_train_vocab(
        ctypes.byref(n_first),
        ctypes.cast(ff_arr, ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(n_subseq),
        ctypes.cast(sf_arr, ctypes.POINTER(ctypes.c_float)),
    )
    return (
        [float(ff_arr[i]) for i in range(int(n_first.value))],
        [float(sf_arr[i]) for i in range(int(n_subseq.value))],
    )


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
        _pack_leaf_flop_history(li_arr[i], li, i)

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
        # SAMPLED path must copy flop_history_train too (2026-06-16) — it was
        # OMITTED here (the non-sampled path above packs it), so the rollout's
        # blueprint CONTINUATION queries keyed on the preflop prefix ONLY and
        # MISSED on every postflop street (EC2: per_street flop=1235,
        # bucket_miss=0), falling back to UNIFORM random continuation -> the
        # cont-strat leaf over-rewarded BETTING (2p flop QJs bet 0.99 vs the
        _pack_leaf_flop_history(li_arr[i], li, i)

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


def compute_subgame_leaf_values_sampled_perleaf_c(
    leaf_infos: List,
    board: List[int],
    street: str,
    player_hands: List[List[Tuple[int, int, float]]],
    per_leaf_player_hands: List[List[List[Tuple[int, int, float]]]],
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
    """Per-leaf-conditioned variant of compute_subgame_leaf_values_sampled_c.

    Identical contract/return shape, except each leaf samples opponents from its
    OWN conditioned ranges: per_leaf_player_hands[leaf_local_idx][player] is that
    leaf's per-player [(c0,c1,weight), ...]. Used ONLY by the 2p-flop
    opponent-range conditioning (leaf_villain_conditioning.py); the C side keys the
    per-leaf array by the LOOP index (1:1 with leaf_infos), so the ordering of
    per_leaf_player_hands MUST match leaf_infos. player_hands is still passed (the
    shared-fallback contract) but is ignored by the C core when per-leaf is set.

    Fail-loud on a shape mismatch (wrong leaf count / wrong player count) — a
    silent mis-shape would scramble the conditioning.
    """
    if street not in _STREET_NUM:
        raise ValueError(f"bad street {street!r}")
    if num_samples_k < 1:
        raise ValueError(f"num_samples_k must be >= 1, got {num_samples_k}")
    lib = _lib()
    if not hasattr(lib, "rollout_compute_subgame_leaf_values_sampled_perleaf_c"):
        raise RuntimeError(
            "rollout lib lacks rollout_compute_subgame_leaf_values_sampled_perleaf_c "
            "— rebuild es_resolver_real / rollout with the 2026-06-17 conditioning"
        )

    if rng is not None:
        try:
            master_seed = int(rng.integers(0, 2**64 - 1, dtype=np.uint64))
        except Exception:
            pass

    if not hasattr(blueprint, "_reader"):
        raise TypeError(
            "C rollout requires a BPSReaderC blueprint (has ._reader handle); "
            f"got {type(blueprint).__name__}."
        )
    bp_ptr = blueprint._reader

    num_players = len(player_hands)
    if num_players <= 0 or num_players > _ROLLOUT_MAX_PLAYERS:
        raise ValueError(f"num_players {num_players} out of range [1, 6]")

    n_leaves = len(leaf_infos)
    if len(per_leaf_player_hands) != n_leaves:
        raise ValueError(
            f"per_leaf_player_hands has {len(per_leaf_player_hands)} entries, "
            f"expected one per leaf ({n_leaves})"
        )
    for lf, plr in enumerate(per_leaf_player_hands):
        if len(plr) != num_players:
            raise ValueError(
                f"per_leaf_player_hands[{lf}] has {len(plr)} player ranges, "
                f"expected {num_players}"
            )

    max_leaf_idx = max((li.leaf_idx for li in leaf_infos), default=-1)
    total = max_leaf_idx + 1
    floats_needed = total * num_players * max_hands
    out = np.zeros((total, num_players, max_hands), dtype=np.float32)

    board_arr = (ctypes.c_int * len(board))(*board) if board else None
    board_ptr = board_arr if board_arr is not None else ctypes.POINTER(ctypes.c_int)()

    li_arr = (_RolloutLeafInfo * n_leaves)()
    for i, li in enumerate(leaf_infos):
        li_arr[i].leaf_idx = int(li.leaf_idx)
        li_arr[i].pot_chips = int(li.pot)
        bets_src = list(li.bets) if hasattr(li, "bets") else [0] * num_players
        for p in range(_ROLLOUT_MAX_PLAYERS):
            li_arr[i].bets[p] = int(bets_src[p]) if p < num_players else 0
        _pack_leaf_flop_history(li_arr[i], li, i)

    active_arr = (ctypes.c_int32 * num_players)(
        *[1 if a else 0 for a in active]
    )

    stash: list = []
    # Shared fallback range (unused by the core when per-leaf is set, but the
    # signature requires a valid pointer).
    hl_arr = (_RolloutHandList * num_players)()
    for p in range(num_players):
        hl_arr[p] = _hand_list_struct(player_hands[p], stash)

    # Per-leaf array: n_leaves*num_players RolloutHandList, row-major
    # [leaf*num_players + player] — exactly how the C core indexes it.
    pl_arr = (_RolloutHandList * (n_leaves * num_players))()
    for lf in range(n_leaves):
        for p in range(num_players):
            pl_arr[lf * num_players + p] = _hand_list_struct(
                per_leaf_player_hands[lf][p], stash)

    acting_order_arr = (ctypes.c_int32 * len(acting_order))(*acting_order)
    player_seats_arr = (ctypes.c_int32 * len(player_seats))(*player_seats)
    gah_arr = (ctypes.c_int32 * len(game_action_history))(*game_action_history) \
        if game_action_history else (ctypes.c_int32 * 0)()

    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    rc = lib.rollout_compute_subgame_leaf_values_sampled_perleaf_c(
        bp_ptr,
        board_ptr, ctypes.c_int32(len(board)),
        ctypes.c_int32(_STREET_NUM[street]),
        li_arr, ctypes.c_int32(n_leaves),
        active_arr, ctypes.c_int32(num_players),
        hl_arr,
        pl_arr,
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
            reason=f"[rollout.c sampled perleaf] {msg}",
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
    """Reset BOTH the hit and attempt counters for the current thread."""
    lib = _lib()
    if not hasattr(lib, "rollout_reset_cont_ev_boundary_hit_count"):
        return
    lib.rollout_reset_cont_ev_boundary_hit_count()


def get_cont_ev_boundary_attempt_count() -> int:
    """Number of flop-boundary query ATTEMPTS in the current thread (the
    hit-rate denominator). Counts every time the predicate fired, including
    attempts that then MISS/fail loud — so hit_rate = hits / attempts.
    Reset alongside the hit count via reset_cont_ev_boundary_hit_count.
    """
    lib = _lib()
    if not hasattr(lib, "rollout_get_cont_ev_boundary_attempt_count"):
        return 0
    out = ctypes.c_uint64(0)
    lib.rollout_get_cont_ev_boundary_attempt_count(ctypes.byref(out))
    return int(out.value)


# ── SYNTHETIC mode wrappers (DEBUG/MEASUREMENT — armed-residual probe) ───


def set_cont_ev_synthetic(enabled: bool) -> None:
    """Toggle synthetic always-hit mode (default OFF). On a boundary MISS the
    consumer fills the slot with a dummy finite EV instead of failing loud, so
    the armed-residual probe can measure latency without a populated table.
    Applies to BOTH boundaries. NEVER enable in production (EVs are dummies)."""
    lib = _lib()
    if not hasattr(lib, "rollout_set_cont_ev_synthetic"):
        raise RuntimeError(
            "rollout.so missing rollout_set_cont_ev_synthetic — rebuild with "
            "the dual-boundary (2026-06-25) changes.")
    lib.rollout_set_cont_ev_synthetic(1 if enabled else 0)


def get_cont_ev_synthetic() -> bool:
    lib = _lib()
    if not hasattr(lib, "rollout_get_cont_ev_synthetic"):
        return False
    return bool(lib.rollout_get_cont_ev_synthetic())


# ── Dual-boundary build (2026-06-25): TURN-boundary consumer wrappers ────
#
# Mirror the flop wrappers above. The SAME cont_ev_table (set via
# set_rollout_cont_ev_table) backs both boundaries — street disambiguation
# is in the key-hash on the C side. INDEPENDENT enable flag and SEPARATE
# turn hit/attempt counters; the flop default is unchanged.


def set_use_cont_ev_at_turn_boundary(enabled: bool) -> None:
    """Enable / disable the end-of-TURN multiway boundary lookup.

    When enabled (and a table pointer is set), the rollout short-circuits
    at the turn boundary: queries cont_ev_table for hero's slot (TURN
    street salt + 4-card turn hand index) at all 4 biases, returns the
    uniform mean as the leaf payoff, short-circuiting the legacy RIVER
    simulation. MISS / EMPTY_BIAS / INVALID is fail-loud.

    INDEPENDENT of the flop flag; default OFF. Process-global.
    """
    lib = _lib()
    if not hasattr(lib, "rollout_set_use_cont_ev_at_turn_boundary"):
        raise RuntimeError(
            "rollout.so missing rollout_set_use_cont_ev_at_turn_boundary — "
            "rebuild with the dual-boundary (2026-06-25) changes."
        )
    lib.rollout_set_use_cont_ev_at_turn_boundary(1 if enabled else 0)


def get_use_cont_ev_at_turn_boundary() -> bool:
    lib = _lib()
    if not hasattr(lib, "rollout_get_use_cont_ev_at_turn_boundary"):
        return False
    return bool(lib.rollout_get_use_cont_ev_at_turn_boundary())


def get_cont_ev_turn_boundary_hit_count() -> int:
    """Number of TURN-boundary lookups that successfully short-circuited
    (separate from the flop hit counter)."""
    lib = _lib()
    if not hasattr(lib, "rollout_get_cont_ev_turn_boundary_hit_count"):
        return 0
    out = ctypes.c_uint64(0)
    lib.rollout_get_cont_ev_turn_boundary_hit_count(ctypes.byref(out))
    return int(out.value)


def get_cont_ev_turn_boundary_attempt_count() -> int:
    """Number of TURN-boundary query ATTEMPTS (the turn hit-rate
    denominator; separate from the flop attempt counter)."""
    lib = _lib()
    if not hasattr(lib, "rollout_get_cont_ev_turn_boundary_attempt_count"):
        return 0
    out = ctypes.c_uint64(0)
    lib.rollout_get_cont_ev_turn_boundary_attempt_count(ctypes.byref(out))
    return int(out.value)


def reset_cont_ev_turn_boundary_hit_count() -> None:
    """Reset BOTH the turn hit and attempt counters (separate from flop)."""
    lib = _lib()
    if not hasattr(lib, "rollout_reset_cont_ev_turn_boundary_hit_count"):
        return
    lib.rollout_reset_cont_ev_turn_boundary_hit_count()


# ── Unit 3 (Stage-1 mean-field): bias-as-hero-action consumer wrappers ───
#
# Toggle the leaf bias-combination semantics for an ARMED cont_ev boundary
# (flop or turn): default OFF = uniform mean of the 4 stored per-bias EVs
# (legacy, byte-identical); ON = regret-matched bias-as-hero-action per leaf
# infoset (Gap 2). Independent of the flop/turn enable flags. See
# src/search/rollout.h "Unit 3" block.


def set_cont_ev_bias_action(enabled: bool) -> None:
    """Enable / disable the bias-as-hero-action leaf semantics (default OFF).

    When ON (and a cont_ev boundary is armed + the table is set), the consumer
    regret-matches over the querying player's 4 stored per-bias EVs per leaf
    infoset and returns the regret-matched mixture value instead of the uniform
    mean; the per-infoset regret store persists within a solve and must be
    reset between solves via ``reset_cont_ev_bias_regrets``.

    Fail-loud: raises RuntimeError if the .so predates Unit 3 (symbol absent) —
    the caller must not silently run legacy uniform-mean when it asked for
    bias-action. Process-global; set once at startup.
    """
    lib = _lib()
    if not hasattr(lib, "rollout_set_cont_ev_bias_action"):
        raise RuntimeError(
            "rollout.so missing rollout_set_cont_ev_bias_action — rebuild with "
            "the Unit 3 mean-field bias-action changes (src/search/rollout.c, "
            "2026-07-01)."
        )
    lib.rollout_set_cont_ev_bias_action(1 if enabled else 0)


def get_cont_ev_bias_action() -> bool:
    """Whether the bias-as-hero-action leaf mode is enabled (False on a .so
    that predates Unit 3)."""
    lib = _lib()
    if not hasattr(lib, "rollout_get_cont_ev_bias_action"):
        return False
    return bool(lib.rollout_get_cont_ev_bias_action())


def get_cont_ev_bias_action_stats() -> int:
    """Number of bias-action mixtures returned since the last reset (per
    solve). Zeroed by ``reset_cont_ev_bias_regrets``. Returns 0 on a .so that
    predates Unit 3."""
    lib = _lib()
    if not hasattr(lib, "rollout_get_cont_ev_bias_action_stats"):
        return 0
    out = ctypes.c_uint64(0)
    lib.rollout_get_cont_ev_bias_action_stats(ctypes.byref(out))
    return int(out.value)


def reset_cont_ev_bias_regrets() -> None:
    """Free the persistent per-solve bias-regret store and zero the select
    counter. Call between solves (never mid-rollout) so regret state does not
    leak across solves — the same lifecycle as the hit-count reset. No-op on a
    .so that predates Unit 3."""
    lib = _lib()
    if not hasattr(lib, "rollout_reset_cont_ev_bias_regrets"):
        return
    lib.rollout_reset_cont_ev_bias_regrets()


# ── Stage-2' pinned-tail MISS fallback wrappers (MEASUREMENT ONLY) ────────
#
# Default OFF (fail-loud serving contract unchanged). When ON, a cont_ev
# boundary query whose slot is MISS/EMPTY_BIAS sends that ONE rollout down
# the legacy live playout instead of raising; INVALID still fails loud.
# Per-(leaf, combo) RNG derivation makes the fallback deterministic and
# arm-identical when both A/B tables miss the same key (the pinned tail).
# See src/search/rollout.h "Stage-2' pinned-tail" block.


def set_cont_ev_miss_fallback(enabled: bool) -> None:
    """Enable / disable the pinned-tail MISS fallback (default OFF).

    Fail-loud: raises RuntimeError if the .so predates the Stage-2' change —
    a measurement that asked for fallback must not silently run fail-loud
    (the gate would crash mid-solve an hour in, not judge).
    """
    lib = _lib()
    if not hasattr(lib, "rollout_set_cont_ev_miss_fallback"):
        raise RuntimeError(
            "rollout.so missing rollout_set_cont_ev_miss_fallback — rebuild "
            "with the Stage-2' pinned-tail changes (src/search/rollout.c, "
            "2026-07-03)."
        )
    lib.rollout_set_cont_ev_miss_fallback(1 if enabled else 0)


def get_cont_ev_miss_fallback() -> bool:
    """Whether the pinned-tail MISS fallback is enabled (False on a .so that
    predates Stage-2')."""
    lib = _lib()
    if not hasattr(lib, "rollout_get_cont_ev_miss_fallback"):
        return False
    return bool(lib.rollout_get_cont_ev_miss_fallback())


def get_cont_ev_fallback_count() -> int:
    """FLOP-boundary rollouts that fell back to the live playout since the
    last reset. 0 on a pre-Stage-2' .so."""
    lib = _lib()
    if not hasattr(lib, "rollout_get_cont_ev_fallback_count"):
        return 0
    out = ctypes.c_uint64(0)
    lib.rollout_get_cont_ev_fallback_count(ctypes.byref(out))
    return int(out.value)


def get_cont_ev_turn_fallback_count() -> int:
    """TURN-boundary rollouts that fell back to the live playout since the
    last reset. 0 on a pre-Stage-2' .so."""
    lib = _lib()
    if not hasattr(lib, "rollout_get_cont_ev_turn_fallback_count"):
        return 0
    out = ctypes.c_uint64(0)
    lib.rollout_get_cont_ev_turn_fallback_count(ctypes.byref(out))
    return int(out.value)


def reset_cont_ev_fallback_counts() -> None:
    """Zero both per-boundary fallback counters (call between solves, same
    lifecycle as the hit-count resets). No-op on a pre-Stage-2' .so."""
    lib = _lib()
    if not hasattr(lib, "rollout_reset_cont_ev_fallback_counts"):
        return
    lib.rollout_reset_cont_ev_fallback_counts()


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
    EXACTLY — see precompute/blueprint_worker_unified.py for the canonical
    layout (the six int64_t iter/interval fields were silently broken for
    ~8 months when the Python mirror used `int`).

    Step 15 incident: This struct was MISSING the trailing
    river_ochs_{enabled,num_clusters} fields. When bp_default_config does
    memset(config, 0, sizeof(BPConfig)) on the FULL C BPConfig size, it
    walks past our ctypes buffer's end and the river_ochs_enabled byte
    is read from adjacent heap garbage. Attempt 1 lucked into 0 (skip
    OCHS init); attempt 2 lucked into nonzero, triggering a 30+ min
    single-threaded k-means k=200 / dim=32608 / n=60k inside
    bp_init_unified. Fix: include the trailing two fields explicitly.
    """
    _fields_ = [
        ("discount_stop_iter",     ctypes.c_int64),
        ("discount_interval",      ctypes.c_int64),
        ("prune_start_iter",       ctypes.c_int64),
        ("snapshot_start_iter",    ctypes.c_int64),
        ("snapshot_interval",      ctypes.c_int64),
        ("strategy_interval",      ctypes.c_int64),
        ("num_threads",            ctypes.c_int),
        ("hash_table_size",        ctypes.c_int64),
        ("snapshot_dir",           ctypes.c_char_p),
        ("include_preflop",        ctypes.c_int),
        ("postflop_num_buckets",   ctypes.c_int),
        ("river_ochs_enabled",     ctypes.c_int),
        ("river_ochs_num_clusters", ctypes.c_int),
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


# ── Standalone CEVS cache holder (Step 15) ───────────────────────────


class _BPContEVTableStandalone(ctypes.Structure):
    """Mirror of BPContEVTable in src/mccfr_cont_ev_table.h. ONLY used
    by the standalone loader (which owns the C-side struct directly).
    The inline-in-BPSolver path uses bp_get_cont_ev_table_ptr instead.
    Match field order EXACTLY — drift here scrambles every cont_ev_table
    query."""
    _fields_ = [
        ("entries",            ctypes.c_void_p),   # BPContEVEntry *
        ("table_size",         ctypes.c_int64),
        ("num_entries",        ctypes.c_int64),
        ("insertion_failures", ctypes.c_int64),
        ("max_probe_observed", ctypes.c_int64),
    ]


class ContEVTableStandaloneHolder:
    """Owns a heap-allocated BPContEVTable populated from a standalone
    CEVS cache file. Faster than the BPR5 path (~10s vs ~20-30 min on
    a 124 GB checkpoint) and doesn't need a BPSolver at all.

    Lifetime contract: keep this object alive for the duration of any
    rollout calls that consult the table. Free deterministically via
    `.free()` or rely on `__del__`.
    """

    def __init__(self, bp_lib: ctypes.CDLL,
                 table_struct: _BPContEVTableStandalone, path: str):
        self._bp = bp_lib
        self._table_struct = table_struct
        self.cont_ev_table_ptr: int = ctypes.addressof(table_struct)
        self.n_slots: int = int(table_struct.num_entries)
        self.path: str = path
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
            self._bp.bp_cont_ev_free(
                ctypes.byref(self._table_struct))
        except Exception:
            pass


def load_cont_ev_table_standalone(path: str, *,
                                   verbose: bool = True
                                   ) -> ContEVTableStandaloneHolder:
    """Load a standalone CEVS cache file written by either:
      - `bp_cont_ev_save_standalone` from C, or
      - `dump_cont_ev_table_to_file` Python wrapper below.

    Returns a holder whose `.cont_ev_table_ptr` matches the layout the
    rollout's `rollout_set_cont_ev_table` expects (a `BPContEVTable*`
    cast to void*). Lifetime contract: keep the holder alive for the
    duration of rollout calls.

    Raises FileNotFoundError / RuntimeError on missing path / load
    failure. The load itself is fast (~10s for 2 GB CEVS file).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CEVS cache not found: {path}")

    bp = _find_blueprint_lib()
    bp.bp_cont_ev_load_standalone.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    bp.bp_cont_ev_load_standalone.restype = ctypes.c_int
    bp.bp_cont_ev_free.argtypes = [ctypes.c_void_p]
    bp.bp_cont_ev_free.restype = None

    table = _BPContEVTableStandalone()
    # Zero-init (the C side does memset internally but we want a known
    # state in case the load fails partway).
    ctypes.memset(ctypes.byref(table), 0, ctypes.sizeof(table))

    if verbose:
        size_gb = os.path.getsize(path) / (1024 ** 3)
        print(f"[cev_standalone] BEFORE bp_cont_ev_load_standalone "
              f"({path}, {size_gb:.2f} GB)",
              flush=True)
    t0 = time.monotonic()
    rc = bp.bp_cont_ev_load_standalone(
        ctypes.byref(table),
        path.encode("utf-8"),
    )
    wall = time.monotonic() - t0
    if verbose:
        print(f"[cev_standalone] AFTER  bp_cont_ev_load_standalone "
              f"rc={rc} ({wall:.2f}s, {table.num_entries:,} slots)",
              flush=True)
    if rc != 0:
        raise RuntimeError(
            f"bp_cont_ev_load_standalone({path}) failed rc={rc} — check "
            f"stderr for the specific reason (bad magic/version/checksum/"
            f"truncation)."
        )
    if table.num_entries == 0:
        # Empty table is a legal save-state but useless for the bridge.
        bp.bp_cont_ev_free(ctypes.byref(table))
        raise RuntimeError(
            f"CEVS file {path} loaded with 0 slots — likely an "
            f"empty-state save. Re-dump from a populated source."
        )
    return ContEVTableStandaloneHolder(bp_lib=bp, table_struct=table,
                                        path=path)


def dump_cont_ev_table_to_file(holder, output_path: str,
                                *, verbose: bool = True) -> None:
    """Write the cont_ev_table owned by `holder` to a standalone CEVS
    cache file. `holder` can be either a `ContEVTableHolder` (BPR5 load
    path) or a `ContEVTableStandaloneHolder` (CEVS reload path) — the
    function uses `cont_ev_table_ptr` from either.

    Atomic write: data goes to <path>.tmp first, then rename(2). If the
    process dies mid-write, no readable cache exists.

    Raises RuntimeError on write failure.
    """
    bp = _find_blueprint_lib()
    bp.bp_cont_ev_save_standalone.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    bp.bp_cont_ev_save_standalone.restype = ctypes.c_int

    if verbose:
        print(f"[cev_standalone] BEFORE bp_cont_ev_save_standalone "
              f"({output_path}, n_slots={holder.n_slots:,})",
              flush=True)
    t0 = time.monotonic()
    rc = bp.bp_cont_ev_save_standalone(
        ctypes.c_void_p(holder.cont_ev_table_ptr),
        output_path.encode("utf-8"),
    )
    wall = time.monotonic() - t0
    if verbose:
        print(f"[cev_standalone] AFTER  bp_cont_ev_save_standalone "
              f"rc={rc} ({wall:.2f}s)",
              flush=True)
    if rc != 0:
        raise RuntimeError(
            f"bp_cont_ev_save_standalone({output_path}) failed rc={rc}"
        )


def _find_blueprint_lib() -> ctypes.CDLL:
    base = Path(__file__).resolve().parent.parent.parent / "build"  # engine/ -> python/ -> repo root (2026-06 relocation)
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

