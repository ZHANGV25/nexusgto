"""Unit tests for the 2p-flop opponent-range conditioning (pure, no blueprint).

Uses a FAKE blueprint that returns a controllable per-bucket flop strategy, so we
can assert the conditioning formula end-to-end without the 12GB artifact:
  - no-op (x1.0) on bucket / strategy miss,
  - the BET-call leaf down-weights hands the blueprint folds,
  - the CHECK-check root leaf is unchanged,
  - immutability (inputs untouched, new lists returned),
  - fail-loud on n_players != 2,
  - underflow guard (never an all-zero villain range).

See python/leaf_villain_conditioning.py and
docs/CPU_ES_CHANCE_NODE_FINDINGS_2026_06_14.md.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from leaf_values import LeafInfo  # type: ignore[import-untyped]
from leaf_villain_conditioning import (  # type: ignore[import-untyped]
    build_conditioned_per_leaf_ranges,
)


# ── A fake BPSReaderC-shaped blueprint ───────────────────────────────────


class _FakeBlueprint:
    """Returns a fixed flop strategy per (player seat, bucket). hand_to_bucket
    maps hole-card c0 -> a small bucket id so tests can steer hands into 'folds'
    or 'continues' buckets deterministically."""

    def __init__(self, *, strat_by_bucket=None, bucket_of_c0=None,
                 miss_bucket=False, miss_strat=False):
        # strat_by_bucket: {bucket: [p_action0, p_action1, ...]} on the FLOP node
        # (action indices match _train_action_idx: facing -> [fold, call, ...];
        # !facing -> [check, bet, ...]).
        self._strat_by_bucket = strat_by_bucket or {}
        self._bucket_of_c0 = bucket_of_c0 or {}
        self._miss_bucket = miss_bucket
        self._miss_strat = miss_strat

    def hand_to_bucket(self, board, c0, c1):
        if self._miss_bucket:
            return -1
        return int(self._bucket_of_c0.get(int(c0), 0))

    def get_strategy(self, *, board, action_history, player, bucket, street):
        if self._miss_strat:
            return None
        s = self._strat_by_bucket.get(int(bucket))
        return list(s) if s is not None else None


def _leaf(history, actors, leaf_idx=0):
    return LeafInfo(leaf_idx=leaf_idx, pot=200, bets=(0, 0),
                    flop_history_train=tuple(history),
                    flop_actor_train=tuple(actors))


# Two-hand ranges per player; villain (player 1) is the seat we condition.
def _ranges():
    # (c0, c1, weight). Hand A (c0=0) is a "continue" hand, hand B (c0=4) a "fold".
    return [
        [(8, 9, 1.0), (12, 13, 1.0)],     # hero (player 0)
        [(0, 1, 1.0), (4, 5, 1.0)],       # villain (player 1)
    ]


# ── Tests ────────────────────────────────────────────────────────────────


def test_raises_on_non_2p():
    with pytest.raises(ValueError):
        build_conditioned_per_leaf_ranges(
            leaf_infos=[_leaf([], [])], player_ranges=[[], [], []],
            flop_board=[0, 1, 2], player_seats=[0, 1, 2],
            game_action_history=[], blueprint=_FakeBlueprint(), n_players=3,
        )


def test_no_op_on_bucket_miss():
    bp = _FakeBlueprint(miss_bucket=True)
    ranges = _ranges()
    # BET-call leaf: P0 bet (idx 1, !facing), P1 call (idx 1, facing).
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 1], [0, 1])], player_ranges=ranges,
        flop_board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[2, 0, 0, 0, 0, 1], blueprint=bp, n_players=2,
    )
    # Every multiplier 1.0 -> villain weights unchanged from static.
    assert per_leaf[0][1] == [(0, 1, 1.0), (4, 5, 1.0)]
    assert stats["bucket_miss"] > 0


def test_no_op_on_strat_miss():
    bp = _FakeBlueprint(miss_strat=True, bucket_of_c0={0: 0, 4: 1})
    ranges = _ranges()
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 1], [0, 1])], player_ranges=ranges,
        flop_board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    assert per_leaf[0][1] == [(0, 1, 1.0), (4, 5, 1.0)]
    assert stats["strat_miss"] > 0


def test_bet_call_leaf_filters_folders():
    # facing node action set [fold(0), call(1), ...]. Villain hand A (bucket 0)
    # calls 0.9; hand B (bucket 1) calls 0.05 (mostly folds).
    bp = _FakeBlueprint(
        bucket_of_c0={0: 0, 4: 1},
        strat_by_bucket={0: [0.1, 0.9], 1: [0.95, 0.05]},
    )
    ranges = _ranges()
    # P0 bet (idx1), P1 call (idx1) -> on the line, P1's action index is 1 (call).
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 1], [0, 1])], player_ranges=ranges,
        flop_board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    villain = per_leaf[0][1]
    # Hand A keeps ~0.9, hand B drops to ~0.05 (strongly down-weighted).
    assert villain[0] == (0, 1, pytest.approx(0.9))
    assert villain[1] == (4, 5, pytest.approx(0.05))
    # Hero (player 0) row unchanged — only the villain (acting seat) is conditioned
    # on its own action; on this line P0's only action is the bet (idx1) so its
    # weight is multiplied by the FLOP no-facing bet prob — but P0 has no bucket
    # mapping here so it stays a no-op... assert hero weights are still positive.
    assert all(w > 0 for (_, _, w) in per_leaf[0][0])
    assert stats["nodes_conditioned"] >= 1


def test_check_check_root_leaf_unchanged():
    # Both check: the !facing node action set is [check(0), bet(1), ...]; each
    # player's taken action is check (idx 0). If the blueprint checks ~1.0 the
    # weights are ~unchanged; we give a non-degenerate check prob and assert the
    # RELATIVE ordering is preserved (both hands scaled by the same check prob,
    # since bucket maps both to the same node strat here).
    bp = _FakeBlueprint(
        bucket_of_c0={0: 0, 4: 0},
        strat_by_bucket={0: [1.0, 0.0]},  # always check
    )
    ranges = _ranges()
    per_leaf, _ = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([0, 0], [0, 1])], player_ranges=ranges,
        flop_board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    villain = per_leaf[0][1]
    # check prob 1.0 -> weights unchanged.
    assert villain[0] == (0, 1, pytest.approx(1.0))
    assert villain[1] == (4, 5, pytest.approx(1.0))


def test_immutability_inputs_untouched():
    bp = _FakeBlueprint(
        bucket_of_c0={0: 0, 4: 1},
        strat_by_bucket={0: [0.1, 0.9], 1: [0.95, 0.05]},
    )
    ranges = _ranges()
    snapshot = [list(r) for r in ranges]
    build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 1], [0, 1])], player_ranges=ranges,
        flop_board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    assert ranges == snapshot  # the input ranges object was not mutated


def test_underflow_guard_falls_back_to_static():
    # A villain that essentially never continues (call prob ~0) would drive the
    # whole range to ~0; the guard must fall back to static weights for that seat.
    bp = _FakeBlueprint(
        bucket_of_c0={0: 0, 4: 0},
        strat_by_bucket={0: [1.0, 1e-12]},  # call ~0 for everyone
    )
    ranges = _ranges()
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 1], [0, 1])], player_ranges=ranges,
        flop_board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    # Fell back to static (non-zero) weights rather than an all-zero range.
    assert per_leaf[0][1] == [(0, 1, 1.0), (4, 5, 1.0)]
    assert stats["underflow_fallback"] >= 1


def test_legacy_leafinfo_without_actor_is_no_op():
    # A LeafInfo missing flop_actor_train (legacy) -> cannot attribute -> the
    # range is left unconditioned (strict no-op), never guessed.
    bp = _FakeBlueprint(
        bucket_of_c0={0: 0, 4: 1},
        strat_by_bucket={0: [0.1, 0.9], 1: [0.95, 0.05]},
    )
    ranges = _ranges()
    legacy = LeafInfo(leaf_idx=0, pot=200, bets=(0, 0),
                      flop_history_train=(1, 1))  # no flop_actor_train
    per_leaf, _ = build_conditioned_per_leaf_ranges(
        leaf_infos=[legacy], player_ranges=ranges,
        flop_board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    assert per_leaf[0][1] == [(0, 1, 1.0), (4, 5, 1.0)]
