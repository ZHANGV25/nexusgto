"""TDD: action_vocab — metadata-driven legal-action counts (plan §5).

The 8.05B trained with postflop_bet_sizes=[0.75], so a no-bet node has 2
actions (check, bet), not the 3 rollout's (0.5,1.0) constant implies. These
pin the count for the 8.05B vocab and a 2-size vocab, mirroring training.
"""

from __future__ import annotations

from app.agent.action_vocab import (
    DEFAULT_ADD_EXPLICIT_ALLIN,
    DEFAULT_POSTFLOP_BET_SIZES,
    DEFAULT_SUBSEQUENT_BET_SIZES,
    legal_categories,
    num_legal,
)

# 8.05B vocab.
P = DEFAULT_POSTFLOP_BET_SIZES      # (0.75,)
S = DEFAULT_SUBSEQUENT_BET_SIZES    # (1.0,)
A = DEFAULT_ADD_EXPLICIT_ALLIN      # False


def test_8_05b_no_bet_has_two_actions():
    # [check, bet0.75]
    assert num_legal(False, 0, P, S, A) == 2


def test_8_05b_facing_first_bet_has_three_actions():
    # [fold, call, raise(0.75 first-raise vocab)]
    assert num_legal(True, 0, P, S, A) == 3


def test_8_05b_facing_reraise_uses_subsequent_vocab():
    # raises_so_far>0 -> subsequent sizes (1.0,) -> [fold, call, raise] = 3
    assert num_legal(True, 1, P, S, A) == 3


def test_two_size_vocab_no_bet_has_three_actions():
    # v3-style [0.5, 1.0]: [check, bet0.5, bet1.0]
    assert num_legal(False, 0, (0.5, 1.0), (1.0,), False) == 3


def test_two_size_vocab_facing_bet_has_four_actions():
    # [fold, call, raise0.5, raise1.0]
    assert num_legal(True, 0, (0.5, 1.0), (1.0,), False) == 4


def test_explicit_allin_adds_a_slot():
    assert num_legal(False, 0, P, S, True) == 3  # check, bet, allin
    assert num_legal(True, 0, P, S, True) == 4   # fold, call, raise, allin


def test_tree_closed_at_max_raises():
    from app.agent.action_vocab import POSTFLOP_MAX_RAISES
    # facing a bet at the raise cap -> [fold, call] only
    assert num_legal(True, POSTFLOP_MAX_RAISES, P, S, A) == 2
    # no bet at the cap -> [check] only
    assert num_legal(False, POSTFLOP_MAX_RAISES, P, S, A) == 1


def test_categories_have_correct_semantics():
    from rollout import ActionCategory  # type: ignore[import-untyped]
    cats = legal_categories(True, 0, P, S, A)
    assert cats[0] == ActionCategory.FOLD
    assert cats[1] == ActionCategory.CALL
    assert cats[2] == ActionCategory.RAISE
    cats2 = legal_categories(False, 0, P, S, A)
    assert cats2[0] == ActionCategory.CALL  # check
    assert cats2[1] == ActionCategory.RAISE  # bet
