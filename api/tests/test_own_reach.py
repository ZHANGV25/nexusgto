"""Own-reach hero wrapper (scripts/own_reach.py) on stub states: no blueprint, no solver."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pytest

from scripts.lbr_core import COMBOS
from scripts.own_reach import OwnReachPolicy, OwnReachStats


@dataclass
class _Pre:
    action_history: Tuple[int, ...] = ()
    street: int = 0
    board: Tuple[int, ...] = ()

    def num_legal_actions(self) -> int:
        return 2


@dataclass
class _Post:
    street: int
    board: Tuple[int, ...] = (40, 41, 42, 43, 44)
    street_history: Tuple[int, ...] = ()

    def num_legal_actions(self) -> int:
        return 2


class _Mix:
    def __init__(self, freqs):
        self.freqs = freqs


class _Lookup:
    """Action 1 with probability 0.75 for combos whose low card is even, else 0.25."""

    def get_mix(self, _state, cards):
        p = 0.75 if int(cards[0]) % 2 == 0 else 0.25
        return _Mix({0: 1.0 - p, 1: p})


class _Live:
    hero_range_override = None


@dataclass
class _Source:
    _live: _Live = field(default_factory=_Live)


class _Engine:
    _last_br_gap_objs = None


class _AlwaysOne:
    def act(self, _state, _cards):
        return 1


def _p1(index: int) -> float:
    return 0.75 if COMBOS[index][0] % 2 == 0 else 0.25


def _policy():
    source = _Source()
    stats = OwnReachStats()
    return OwnReachPolicy(_AlwaysOne(), _Lookup(), _Engine(), [source], stats), source, stats


def test_override_is_the_reach_at_the_street_root():
    policy, source, stats = _policy()
    cards = (0, 1)
    policy.act(_Pre(), cards)  # preflop: reach *= p(action 1)
    policy.act(_Post(street=1), cards)  # flop root: override = preflop reach
    expected_flop = np.array([_p1(i) for i in range(len(COMBOS))])
    assert source._live.hero_range_override == pytest.approx(expected_flop)
    policy.act(_Post(street=1, street_history=(0, 1)), cards)  # second flop decision: still the root reach
    assert source._live.hero_range_override == pytest.approx(expected_flop)
    policy.act(_Post(street=2), cards)  # turn root: three action-1 factors so far
    # the flop decisions' tables are zero for combos holding a flop card (40, 41, 42)
    on_flop = np.array([not ({40, 41, 42} & set(COMBOS[i])) for i in range(len(COMBOS))])
    assert source._live.hero_range_override == pytest.approx(np.where(on_flop, expected_flop ** 3, 0.0))
    assert stats.as_dict()["decisions"] == 4
    assert stats.as_dict()["lookup_tables"] == 4 and stats.as_dict()["stash_tables"] == 0


def test_new_hand_resets_the_reach():
    policy, source, _stats = _policy()
    policy.act(_Pre(), (0, 1))
    policy.act(_Post(street=1), (0, 1))
    policy.act(_Pre(), (2, 3))  # new hole cards: a new hand
    policy.act(_Post(street=1), (2, 3))
    expected = np.array([_p1(i) for i in range(len(COMBOS))])
    assert source._live.hero_range_override == pytest.approx(expected)


def test_same_cards_back_at_preflop_is_a_new_hand():
    policy, source, _stats = _policy()
    policy.act(_Pre(action_history=(1, 1)), (0, 1))
    policy.act(_Post(street=1), (0, 1))
    policy.act(_Pre(action_history=(1, 1)), (0, 1))  # duplicate deal, same seat
    policy.act(_Post(street=1), (0, 1))
    expected = np.array([_p1(i) for i in range(len(COMBOS))])
    assert source._live.hero_range_override == pytest.approx(expected)


def test_wrapper_returns_the_inner_result_and_forwards_attributes():
    policy, _source, _stats = _policy()
    assert policy.act(_Pre(), (0, 1)) == 1
    policy._inner.fallback_count = 7
    assert policy.fallback_count == 7
