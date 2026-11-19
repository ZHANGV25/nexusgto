"""run_match start_group shards reproduce the unsharded run's groups."""
from __future__ import annotations

import pytest

from scripts import run_match


def _zero_hand(deal, policies):
    return run_match.PlayedHand(
        deltas=(0, 0, 0, 0, 0, 0), action_count=0, terminal="preflop_fold",
        went_to_showdown=False,
    )


def _run(deals, start_group=0):
    return run_match.run_match(
        deals=deals, mode="aa", master_seed=31, a_factory=lambda seed: object(),
        hand_player=_zero_hand, start_group=start_group,
    )


def test_shard_matches_tail_of_full_run():
    full = _run(5)
    shard = _run(2, start_group=3)
    assert [g.group_index for g in shard.groups] == [3, 4]
    assert [g.deal for g in shard.groups] == [g.deal for g in full.groups[3:]]


def test_negative_start_group_is_rejected():
    with pytest.raises(ValueError, match="start_group"):
        _run(1, start_group=-1)
