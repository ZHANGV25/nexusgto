"""Tests for scripts/match_aivat.py (no blueprint; fake evaluator unless noted)."""
from __future__ import annotations

import numpy as np
import pytest

from app.engine.agent.strategy_source import ActionMix
from scripts import match_aivat, run_match

FOLDED_EXCEPT = lambda *live: tuple(seat not in live for seat in range(6))  # noqa: E731


def _fake_eval(cards):
    ranks = np.sort(np.asarray(cards) // 4, axis=1)[:, ::-1]
    return (ranks * 13 ** np.arange(6, -1, -1)).sum(axis=1).astype(np.int64)


@pytest.fixture
def fake_evaluator(monkeypatch):
    monkeypatch.setattr(match_aivat, "evaluate_7card_batch", _fake_eval)
    match_aivat.reset_cache()
    yield
    match_aivat.reset_cache()


class UniformStub:
    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)

    def act(self, state, hero_cards):
        mix = ActionMix.uniform(int(state.num_legal_actions()))
        return run_match.ActionChoice(mix.sample(self._rng), mix=mix)


class FirstAction:
    def act(self, state, hero_cards):
        return run_match.ActionChoice(0, mix=ActionMix({0: 1.0}))


def _ranks(*rows):
    return np.asarray(rows, dtype=np.int64)


def test_frozen_pot_winner_collects_live_and_folded_chips():
    values = match_aivat.frozen_pot_values(
        _ranks([9, 3, 0, 0, 0, 0]), [100, 100, 50, 0, 0, 0], FOLDED_EXCEPT(0, 1)
    )
    assert values.tolist() == [150.0, -100.0, -50.0, 0.0, 0.0, 0.0]


def test_frozen_pot_layers_side_pots():
    values = match_aivat.frozen_pot_values(
        _ranks([9, 5, 1, 0, 0, 0]), [50, 150, 150, 0, 0, 0], FOLDED_EXCEPT(0, 1, 2)
    )
    assert values.tolist() == [100.0, 50.0, -150.0, 0.0, 0.0, 0.0]


def test_frozen_pot_returns_uncalled_excess():
    values = match_aivat.frozen_pot_values(
        _ranks([9, 5, 0, 0, 0, 0]), [100, 300, 0, 0, 0, 0], FOLDED_EXCEPT(0, 1)
    )
    assert values.tolist() == [100.0, -100.0, 0.0, 0.0, 0.0, 0.0]


def test_frozen_pot_splits_ties_and_averages_runouts():
    tie = match_aivat.frozen_pot_values(
        _ranks([7, 7, 0, 0, 0, 0]), [100, 100, 0, 0, 0, 0], FOLDED_EXCEPT(0, 1)
    )
    coin = match_aivat.frozen_pot_values(
        _ranks([9, 5, 0, 0, 0, 0], [5, 9, 0, 0, 0, 0]),
        [100, 100, 0, 0, 0, 0],
        FOLDED_EXCEPT(0, 1),
    )
    assert tie.tolist() == [0.0] * 6
    assert coin.tolist() == [0.0] * 6


def test_frozen_pot_single_live_seat_takes_everything():
    values = match_aivat.frozen_pot_values(
        _ranks([0] * 6), [1, 2, 0, 10, 0, 0], FOLDED_EXCEPT(3)
    )
    assert values.tolist() == [-1.0, -2.0, 0.0, 3.0, 0.0, 0.0]


@pytest.mark.parametrize("board_count", [3, 4])
def test_value_equals_mean_over_next_card(fake_evaluator, board_count):
    """The exactness the turn/river chance terms rely on."""

    deal = run_match.deal_from_rng(np.random.default_rng(5))
    chips = ((300, 300, 120, 0, 0, 0), FOLDED_EXCEPT(0, 1, 2))
    current = match_aivat.DealValues(deal).value(*chips, board_count)
    known = tuple(deal.board[:board_count])
    next_cards = [c for c in match_aivat._unseen_cards(deal) if c not in known]
    later = [
        match_aivat.DealValues(
            run_match.Deal(
                board=(known + (card,) * 5)[:5], hole_cards=deal.hole_cards
            )
        ).value(*chips, board_count + 1)
        for card in next_cards
    ]
    np.testing.assert_allclose(np.mean(later, axis=0), current, atol=1e-7)


def test_sampling_probs_for_mix_explicit_and_stub():
    mix = ActionMix({0: 0.25, 2: 0.75})
    assert run_match.sampling_probs_for(
        run_match.ActionChoice(2, mix=mix), 3
    ) == (0.25, 0.0, 0.75)
    explicit = run_match.ActionChoice(0, mix=mix, sampling_probs=(0.5, 0.5, 0.0))
    assert run_match.sampling_probs_for(explicit, 3) == (0.5, 0.5, 0.0)
    assert run_match.sampling_probs_for(1, 3) is None


def test_epsilon_sampling_probs_matches_policy_law():
    probs = run_match.epsilon_sampling_probs((0.25, 0.0, 0.75), 0.2)
    assert probs == pytest.approx((0.275, 0.1, 0.625))
    assert sum(probs) == pytest.approx(1.0)


def test_epsilon_policy_reports_true_distribution():
    state = type("S", (), {"num_legal_actions": lambda self: 2})()
    result = run_match.EpsilonPolicy(FirstAction(), epsilon=0.1, seed=3).act(
        state, (0, 1)
    )
    assert result.sampling_probs == pytest.approx((0.9, 0.1))


def test_deterministic_fold_around_has_zero_correction(fake_evaluator):
    deal = run_match.deal_from_rng(np.random.default_rng(1))
    hand = match_aivat.play_full_hand_aivat(deal, [FirstAction() for _ in range(6)])
    assert hand.terminal == "preflop_fold"
    assert hand.adjusted_deltas == tuple(float(delta) for delta in hand.deltas)


def test_unpriced_policy_is_refused(fake_evaluator):
    class MixlessStub:
        def act(self, state, hero_cards):
            return run_match.ActionChoice(0)

    deal = run_match.deal_from_rng(np.random.default_rng(2))
    with pytest.raises(match_aivat.AivatUnpricedDecision):
        match_aivat.play_full_hand_aivat(deal, [MixlessStub() for _ in range(6)])


def test_corrections_have_zero_mean_under_random_play(fake_evaluator):
    rng = np.random.default_rng(20260912)
    seat0 = []
    for hand_index in range(200):
        deal = run_match.deal_from_rng(rng)
        policies = [UniformStub(hand_index * 10 + seat) for seat in range(6)]
        hand = match_aivat.play_full_hand_aivat(deal, policies)
        assert abs(sum(hand.adjusted_deltas)) < 1e-9
        seat0.append(hand.adjusted_deltas[0] - hand.deltas[0])
    values = np.asarray(seat0)
    assert np.count_nonzero(values) > 0
    assert abs(values.mean()) <= 4.0 * values.std(ddof=1) / np.sqrt(len(values))


def test_aa_match_with_aivat_keeps_exact_null(fake_evaluator):
    run = run_match.run_match(
        deals=3,
        mode="aa",
        master_seed=4,
        a_factory=UniformStub,
        hand_player=match_aivat.play_full_hand_aivat,
        allin_equity_adjust=True,
    )
    assert run.summary["mean_bb_per_100_adjusted"] == 0.0


def test_c_and_python_evaluators_order_hands_identically():
    c_eval = match_aivat._load_c_eval7()
    if c_eval is None:
        pytest.skip("hand_eval_batch library not built under POKER_SOLVER_PATH")
    rng = np.random.default_rng(9)
    cards = np.stack([rng.choice(52, 7, replace=False) for _ in range(300)])
    c_ranks = c_eval(cards)
    py_ranks = match_aivat._python_eval7_batch(cards)
    assert np.array_equal(
        np.sign(c_ranks[:, None] - c_ranks[None, :]),
        np.sign(py_ranks[:, None] - py_ranks[None, :]),
    )
