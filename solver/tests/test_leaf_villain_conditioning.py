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

from engine.leaf_values import LeafInfo  # type: ignore[import-untyped]
from engine.leaf_villain_conditioning import (  # type: ignore[import-untyped]
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

    def hand_to_bucket_turn(self, board, c0, c1):
        # Same mapping as flop for the fake (the test steers by c0 regardless of
        # street); the real blueprint uses k-means turn centroids.
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
            board=[0, 1, 2], player_seats=[0, 1, 2],
            game_action_history=[], blueprint=_FakeBlueprint(), n_players=3,
        )


def test_no_op_on_bucket_miss():
    bp = _FakeBlueprint(miss_bucket=True)
    ranges = _ranges()
    # BET-call leaf: P0 bet (idx 1, !facing), P1 call (idx 1, facing).
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 1], [0, 1])], player_ranges=ranges,
        board=[0, 1, 2], player_seats=[1, 2],
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
        board=[0, 1, 2], player_seats=[1, 2],
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
        board=[0, 1, 2], player_seats=[1, 2],
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
        board=[0, 1, 2], player_seats=[1, 2],
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
        board=[0, 1, 2], player_seats=[1, 2],
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
        board=[0, 1, 2], player_seats=[1, 2],
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
        board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    assert per_leaf[0][1] == [(0, 1, 1.0), (4, 5, 1.0)]


def test_turn_street_filters_via_turn_bucketizer():
    """street=2 (turn): same filtering mechanism, dispatched on the 4-card board
    via hand_to_bucket_turn + get_strategy(street=2). The BET-call leaf
    down-weights the villain's folding hands exactly as on the flop."""
    bp = _FakeBlueprint(
        bucket_of_c0={0: 0, 4: 1},
        strat_by_bucket={0: [0.1, 0.9], 1: [0.95, 0.05]},
    )
    ranges = _ranges()
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 1], [0, 1])], player_ranges=ranges,
        board=[0, 1, 2, 6], player_seats=[1, 2],  # 4-card turn board
        game_action_history=[], blueprint=bp, n_players=2, street=2,
    )
    villain = per_leaf[0][1]
    assert villain[0] == (0, 1, pytest.approx(0.9))   # continues
    assert villain[1] == (4, 5, pytest.approx(0.05))  # mostly folds
    assert stats["nodes_conditioned"] >= 1


def test_hero_pure_zero_action_is_floored_not_zeroed():
    """REGRESSION (review 2026-06-18, MEDIUM): a HERO hand the blueprint plays
    with pure-0 probability on the leaf's line must be FLOORED to a tiny epsilon,
    NOT driven to exactly 0.0. The C all-hands construct reuses this list to
    enumerate each seat as hero for its own table row and zero-gates weight<=0 to
    a SPURIOUS 0.0 leaf EV (rollout.c:1250), which would bias hero's solved
    strategy. Here HERO (player 0) bet a hand (c0=8, bucket 5) the blueprint bets
    with probability 0.0 -> multiplier 0.0 -> must be floored, not zeroed.
    """
    from engine.leaf_villain_conditioning import _HERO_ENUM_FLOOR  # type: ignore
    bp = _FakeBlueprint(
        # hero hand A c0=8 -> bucket 5 (pure-0 bet); hand B c0=12 -> bucket 6 (bets 0.7)
        # villain hands unchanged (c0=0 -> bucket 0, c0=4 -> bucket 1)
        bucket_of_c0={8: 5, 12: 6, 0: 0, 4: 1},
        strat_by_bucket={
            5: [1.0, 0.0],    # hero bucket 5: NEVER bets (pure check) -> bet mult 0.0
            6: [0.3, 0.7],    # hero bucket 6: bets 0.7
            0: [0.1, 0.9], 1: [0.95, 0.05],
        },
    )
    ranges = _ranges()
    # BET-call leaf: hero (P0) BET (idx 1, !facing), villain (P1) CALL (idx 1).
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 1], [0, 1])], player_ranges=ranges,
        board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    hero = per_leaf[0][0]
    # Hand A (pure-0 bet) is FLOORED, not zeroed — still enumerable as hero.
    assert hero[0][2] == pytest.approx(_HERO_ENUM_FLOOR)
    assert hero[0][2] > 0.0
    # Hand B (bets 0.7) keeps its conditioned weight, well above the floor.
    assert hero[1][2] == pytest.approx(0.7)
    # The floor fired at least once and is reported in stats.
    assert stats["hero_floor"] >= 1


def test_villain_pure_zero_also_floored_but_sampling_negligible():
    """A villain hand driven to a pure-0 multiplier is floored too (same list,
    same C zero-gate), but the floor (1e-12) is so far below any real σ̄ mass that
    it is sampling-negligible — the conditioning's relative down-weighting of the
    folding range is preserved."""
    from engine.leaf_villain_conditioning import _HERO_ENUM_FLOOR  # type: ignore
    bp = _FakeBlueprint(
        bucket_of_c0={0: 0, 4: 1},
        strat_by_bucket={0: [0.1, 0.9], 1: [1.0, 0.0]},  # bucket 1: pure fold -> call 0.0
    )
    ranges = _ranges()
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 1], [0, 1])], player_ranges=ranges,
        board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    villain = per_leaf[0][1]
    assert villain[0][2] == pytest.approx(0.9)              # continues
    assert villain[1][2] == pytest.approx(_HERO_ENUM_FLOOR)  # pure-fold floored, not 0
    # Floored weight is >12 orders of magnitude below the continuing hand -> the
    # opponent sampler (weight-proportional) draws it with ~0 probability.
    assert villain[1][2] < villain[0][2] / 1e10
    assert stats["hero_floor"] >= 1


def test_turn_street_needs_4card_board():
    """street=2 fail-loud on a 3-card board (would mis-key the turn node)."""
    with pytest.raises(ValueError):
        build_conditioned_per_leaf_ranges(
            leaf_infos=[_leaf([1, 1], [0, 1])], player_ranges=_ranges(),
            board=[0, 1, 2], player_seats=[1, 2],  # only 3 cards
            game_action_history=[], blueprint=_FakeBlueprint(), n_players=2,
            street=2,
        )


# ── Prefix-sensitive fake: proves the MULTI-STEP product math ─────────────
#
# The single-factor cases above lock P_continue at ONE decision node. The fix's
# actual formula is a PRODUCT over every node on the line where the villain acts,
# each keyed by the INCREMENTAL action-history prefix (preflop ++ this-street
# actions BEFORE that step). A fake that returns the same strategy regardless of
# prefix cannot distinguish "product of two nodes" from "one node, applied twice".
# This fake keys on (bucket, prefix length) so a regression that drops a step,
# reuses the wrong prefix, or stops being a true product fails loud.


class _PrefixBlueprint:
    """Returns a strategy that depends on (bucket, len(action_history)) so each
    decision node on a line can carry a DISTINCT frequency — the only way to
    verify the conditioning multiplier is the PRODUCT across the villain's steps,
    keyed by the correct incremental prefix."""

    def __init__(self, *, bucket_of_c0, strat_by_bucket_and_prefixlen):
        self._bucket_of_c0 = bucket_of_c0
        # {(bucket, prefix_len): [a0, a1, ...]}
        self._table = strat_by_bucket_and_prefixlen

    def hand_to_bucket(self, board, c0, c1):
        return int(self._bucket_of_c0.get(int(c0), 0))

    def hand_to_bucket_turn(self, board, c0, c1):
        return int(self._bucket_of_c0.get(int(c0), 0))

    def get_strategy(self, *, board, action_history, player, bucket, street):
        s = self._table.get((int(bucket), len(action_history)))
        return list(s) if s is not None else None


def test_multistep_product_locks_incremental_prefix_math():
    """Villain (P1) acts TWICE on the line — bet then call — so its conditioning
    multiplier must be the PRODUCT of the two node frequencies, each keyed by the
    incremental prefix. Line: P0 check, P1 bet, P0 raise, P1 call.
        history = [0, 1, 2, 1]   actors = [0, 1, 0, 1]
    P1's steps (where actor==1):
        step i=1: action idx 1 (bet, !facing), prefix len = len(preflop)+1 = 1
        step i=3: action idx 1 (call, facing),  prefix len = len(preflop)+3 = 3
    Continue hand (bucket 0): P(bet|len1)=0.6, P(call|len3)=0.5 -> 0.30.
    Fold hand   (bucket 1): P(bet|len1)=0.4, P(call|len3)=0.10 -> 0.04.
    A regression that keys the wrong prefix, drops a step, or applies a single
    factor twice cannot reproduce BOTH products from these distinct per-node
    frequencies.
    """
    preflop = [2, 0, 0, 0, 0, 1]  # non-empty: indices 1 and 3 land at len 7 and 9
    bp = _PrefixBlueprint(
        bucket_of_c0={0: 0, 4: 1},
        strat_by_bucket_and_prefixlen={
            # bucket 0 (continue hand): bet@len7 -> 0.6 ; call@len9 -> 0.5
            (0, len(preflop) + 1): [0.4, 0.6],          # !facing [check, bet]
            (0, len(preflop) + 3): [0.5, 0.5],          # facing  [fold, call]
            # bucket 1 (fold-y hand): bet@len7 -> 0.4 ; call@len9 -> 0.10
            (1, len(preflop) + 1): [0.6, 0.4],
            (1, len(preflop) + 3): [0.9, 0.1],
        },
    )
    ranges = _ranges()
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([0, 1, 2, 1], [0, 1, 0, 1])], player_ranges=ranges,
        board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=preflop, blueprint=bp, n_players=2,
    )
    villain = per_leaf[0][1]
    # PRODUCT of the two distinct node frequencies — NOT a single factor.
    assert villain[0] == (0, 1, pytest.approx(0.6 * 0.5))   # continue -> 0.30
    assert villain[1] == (4, 5, pytest.approx(0.4 * 0.10))  # fold-y  -> 0.04
    # Two villain nodes conditioned PER hand -> >= 2 across the two-hand range.
    assert stats["nodes_conditioned"] >= 2
    assert stats["underflow_fallback"] == 0


def test_folding_villain_drives_weight_toward_zero():
    """The product can collapse a folding villain's weight toward ~0 (the QJs fix
    in miniature: the BET-call leaf should be valued vs the CALLING subset, so a
    hand the blueprint almost never continues with carries almost no villain
    weight). Hand A continues (call 0.92); hand B is a near-pure fold (call 0.01).
    Assert the RATIO collapses by ~the call-frequency ratio, not merely 'not
    uniform' — locks the magnitude, the thing the EC2 QJs run validated."""
    bp = _FakeBlueprint(
        bucket_of_c0={0: 0, 4: 1},
        strat_by_bucket={0: [0.08, 0.92], 1: [0.99, 0.01]},
    )
    ranges = _ranges()
    per_leaf, _ = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 1], [0, 1])], player_ranges=ranges,
        board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    cont_w = per_leaf[0][1][0][2]   # continuing hand weight
    fold_w = per_leaf[0][1][1][2]   # folding hand weight
    assert cont_w == pytest.approx(0.92)
    assert fold_w == pytest.approx(0.01)
    # Folding hand is ~92x lighter than the continuing hand — the leaf is now
    # valued overwhelmingly against the calling range (the over-bet cure).
    assert fold_w < cont_w / 50.0


def test_off_vocabulary_allin_conditions_on_most_aggressive_column():
    """An explicit all-in (or an added off-tree size) has no column in the
    blueprint node: the trainer vocab is [0.75] first / [1.0] subsequent. Until
    2026-09-09 such a step was a x1.0 no-op, so a villain who JAMMED was
    conditioned only on his earlier BET and the all-in leaf faced his betting
    range (measured: ace-high called a 190 jam at 1.00 on Kh Qh 5h 9s). The
    step now conditions on the node's most aggressive column (its last entry)
    and is counted separately as idx_oor_proxied."""
    # Facing-a-bet node with [fold, call, raise]; action index 3 = all-in.
    bp = _FakeBlueprint(
        bucket_of_c0={0: 0, 4: 1},
        strat_by_bucket={0: [0.10, 0.30, 0.60], 1: [0.50, 0.49, 0.01]},
    )
    ranges = _ranges()
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 3, 1], [0, 1, 0])], player_ranges=ranges,
        board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    jam_w = per_leaf[0][1][0][2]     # hand A: raise column 0.60
    nojam_w = per_leaf[0][1][1][2]   # hand B: raise column 0.01
    assert jam_w == pytest.approx(0.60)
    assert nojam_w == pytest.approx(0.01)
    # villain: 2 hands x 1 proxied step; hero: 2 hands x 2 in-vocab steps.
    assert stats["idx_oor_proxied"] == 2
    assert stats["idx_oor"] == 0
    assert stats["nodes_conditioned"] == 6


def test_off_vocabulary_index_on_one_column_node_stays_no_op():
    """A one-column node cannot represent aggression at all: keep the historical
    x1.0 no-op and the idx_oor count so nothing is silently invented."""
    bp = _FakeBlueprint(
        bucket_of_c0={0: 0, 4: 1},
        strat_by_bucket={0: [1.0], 1: [1.0]},
    )
    ranges = _ranges()
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, 3, 1], [0, 1, 0])], player_ranges=ranges,
        board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    assert per_leaf[0][1][0][2] == pytest.approx(1.0)
    assert per_leaf[0][1][1][2] == pytest.approx(1.0)
    # every step (villain's jam and hero's two in-vocab steps) is out of range
    # on a one-column node: 2 villain + 4 hero no-ops, nothing proxied.
    assert stats["idx_oor"] == 6
    assert stats["idx_oor_proxied"] == 0


def test_negative_index_still_no_op():
    bp = _FakeBlueprint(
        bucket_of_c0={0: 0, 4: 1},
        strat_by_bucket={0: [0.1, 0.9], 1: [0.9, 0.1]},
    )
    per_leaf, stats = build_conditioned_per_leaf_ranges(
        leaf_infos=[_leaf([1, -1], [0, 1])], player_ranges=_ranges(),
        board=[0, 1, 2], player_seats=[1, 2],
        game_action_history=[], blueprint=bp, n_players=2,
    )
    assert per_leaf[0][1][0][2] == pytest.approx(1.0)
    assert stats["idx_oor"] == 2
