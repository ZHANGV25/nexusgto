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
    _fields_ = [
        ("leaf_idx", ctypes.c_int32),
        ("pot_chips", ctypes.c_int32),
        ("bets", ctypes.c_int32 * _ROLLOUT_MAX_PLAYERS),
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


_LIB: Optional[ctypes.CDLL] = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _find_lib()
        _setup_ctypes(_LIB)
    return _LIB


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

    Callers that care about coverage should reset before each solve and
    read after to scope counts to that solve.
    """
    lib = _lib()
    miss = ctypes.c_uint64(0)
    zerosum = ctypes.c_uint64(0)
    lib.rollout_get_observability(ctypes.byref(miss), ctypes.byref(zerosum))
    return (int(miss.value), int(zerosum.value))


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

    # leaf_infos → RolloutLeafInfo array
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
