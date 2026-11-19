"""action_translation tests.

Map an opponent's (action, amount) to the on-tree action INDEX that
GameState.advance / key_builder consume. On-tree actions translate exactly;
off-tree bet sizes snap via pseudoharmonic to the nearest tree size. Reuses
the card_codec label helpers and the pseudoharmonic primitive (no parallel
reimplementation).

100% coverage target (action_translation is key-construction-adjacent core).
"""

from __future__ import annotations

import pytest

from app.engine.agent.action_translation import (
    OFF_TREE_PSEUDOHARMONIC,
    ON_TREE_EXACT,
    translate_action,
)

# 8.05B vocab: single 0.75-pot postflop bet, no explicit all-in.
SIZES = (0.75,)
ADD_ALLIN = False


# ── No-bet context: [check0, bet1] ───────────────────────────────────────


def test_check_when_no_bet():
    res = translate_action("check", None, pot=100, bet_sizes=SIZES,
                           add_explicit_allin=ADD_ALLIN, facing_bet=False,
                           raises_so_far=0)
    assert res.action_idx == 0
    assert res.kind == ON_TREE_EXACT


def test_on_tree_bet_maps_to_index_1():
    # Bet exactly 0.75 pot -> on-tree, index 1.
    res = translate_action("raise", 75.0, pot=100, bet_sizes=SIZES,
                           add_explicit_allin=ADD_ALLIN, facing_bet=False,
                           raises_so_far=0)
    assert res.action_idx == 1
    assert res.kind == ON_TREE_EXACT


def test_off_tree_bet_snaps_to_nearest_size():
    # Bet 0.5 pot with only a 0.75 tree size -> below all -> snaps to 0.75
    # (index 1). Single-size vocab => pseudoharmonic degenerates to nearest.
    res = translate_action("raise", 50.0, pot=100, bet_sizes=SIZES,
                           add_explicit_allin=ADD_ALLIN, facing_bet=False,
                           raises_so_far=0)
    assert res.action_idx == 1
    assert res.kind == OFF_TREE_PSEUDOHARMONIC


# ── Facing a bet: [fold0, call1, raise2] ─────────────────────────────────


def test_fold_when_facing_bet():
    res = translate_action("fold", None, pot=100, bet_sizes=SIZES,
                           add_explicit_allin=ADD_ALLIN, facing_bet=True,
                           raises_so_far=1)
    assert res.action_idx == 0
    assert res.kind == ON_TREE_EXACT


def test_call_when_facing_bet():
    res = translate_action("call", 75.0, pot=175, bet_sizes=SIZES,
                           add_explicit_allin=ADD_ALLIN, facing_bet=True,
                           raises_so_far=1)
    assert res.action_idx == 1
    assert res.kind == ON_TREE_EXACT


def test_raise_when_facing_bet_maps_to_index_2():
    # Re-raise vocab is subsequent (1.0). A pot-sized raise -> on-tree raise
    # at index 2.
    res = translate_action("raise", 100.0, pot=100, bet_sizes=SIZES,
                           add_explicit_allin=ADD_ALLIN, facing_bet=True,
                           raises_so_far=1)
    assert res.action_idx == 2


# ── All-in handling ──────────────────────────────────────────────────────


def test_allin_no_explicit_slot_snaps_to_top_of_vocab():
    # add_explicit_allin=False: an all-in has no dedicated slot, snaps to the
    # top bet/raise index (on-tree).
    res = translate_action("allin", 9999.0, pot=100, bet_sizes=SIZES,
                           add_explicit_allin=False, facing_bet=False,
                           raises_so_far=0)
    # no-bet: [check0, bet1] -> top index = 1 (len(sizes))
    assert res.action_idx == 1


def test_allin_facing_bet_no_explicit_slot():
    res = translate_action("allin", 9999.0, pot=100, bet_sizes=SIZES,
                           add_explicit_allin=False, facing_bet=True,
                           raises_so_far=1)
    # facing-bet: [fold0, call1, raise2] -> top = 1 + len(sizes) = 2
    assert res.action_idx == 2


def test_allin_explicit_slot_when_enabled():
    res = translate_action("allin", 9999.0, pot=100, bet_sizes=SIZES,
                           add_explicit_allin=True, facing_bet=False,
                           raises_so_far=0)
    # no-bet with explicit allin: [check0, bet1, allin2] -> 1 + len(sizes) = 2
    assert res.action_idx == 2


# ── Index stays within the legal action set (consistency with key_builder) ─


def test_index_within_legal_set_no_bet():
    from engine.rollout import classify_postflop_actions  # type: ignore[import-untyped]
    res = translate_action("raise", 75.0, pot=100, bet_sizes=SIZES,
                           add_explicit_allin=ADD_ALLIN, facing_bet=False,
                           raises_so_far=0)
    n = len(classify_postflop_actions(facing_bet=False, raises_so_far=0))
    assert 0 <= res.action_idx < n


def test_index_within_legal_set_facing_bet():
    from engine.rollout import classify_postflop_actions  # type: ignore[import-untyped]
    res = translate_action("raise", 100.0, pot=100, bet_sizes=SIZES,
                           add_explicit_allin=ADD_ALLIN, facing_bet=True,
                           raises_so_far=1)
    n = len(classify_postflop_actions(facing_bet=True, raises_so_far=1))
    assert 0 <= res.action_idx < n


# ── Validation / errors ──────────────────────────────────────────────────


def test_empty_bet_sizes_raises():
    with pytest.raises(ValueError):
        translate_action("raise", 75.0, pot=100, bet_sizes=(),
                         add_explicit_allin=False, facing_bet=False,
                         raises_so_far=0)


def test_unknown_action_raises():
    with pytest.raises(ValueError):
        translate_action("teleport", None, pot=100, bet_sizes=SIZES,
                         add_explicit_allin=False, facing_bet=False,
                         raises_so_far=0)


def test_raise_without_amount_defaults_to_smallest_size():
    # No amount given for a raise -> default to smallest tree size (on-tree).
    res = translate_action("raise", None, pot=100, bet_sizes=(0.5, 1.0),
                           add_explicit_allin=False, facing_bet=False,
                           raises_so_far=0)
    # no-bet [check0, bet0.5=1, bet1.0=2] -> smallest = index 1
    assert res.action_idx == 1


def test_off_tree_between_two_sizes_picks_a_valid_neighbor():
    # 0.7 pot bet with tree [0.5, 1.0] -> pseudoharmonic between them; the
    # resulting on-tree index must be one of the two neighbors (1 or 2).
    res = translate_action("raise", 70.0, pot=100, bet_sizes=(0.5, 1.0),
                           add_explicit_allin=False, facing_bet=False,
                           raises_so_far=0)
    assert res.action_idx in (1, 2)
    assert res.kind == OFF_TREE_PSEUDOHARMONIC
