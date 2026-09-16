"""Local best response: pure math and the state adapter (no blueprint needed)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import lbr_core as core
from scripts import lbr_game as game
from scripts.match_aivat import evaluate_7card_batch


def card(rank: str, suit: str) -> int:
    return "23456789TJQKA".index(rank) * 4 + "cdhs".index(suit)


NUT_BOARD = [card("2", "s"), card("3", "s"), card("4", "s"), card("K", "d"), card("9", "h")]
NUT_HAND = (card("5", "s"), card("6", "s"))
AIR_HAND = (card("7", "c"), card("2", "d"))


class StationPolicy:
    """Checks or calls with every hand."""

    def table(self, state, cards=None):
        table = np.zeros((core.N_COMBOS, state.num_legal_actions()))
        table[:, game.call_index(state)] = 1.0
        return table

    def decide(self, state, cards):
        return game.call_index(state), self.table(state)


class FolderPolicy:
    """Folds to any bet, checks otherwise."""

    def table(self, state, cards=None):
        kinds = game.action_kinds(state)
        table = np.zeros((core.N_COMBOS, len(kinds)))
        table[:, kinds.index("fold") if "fold" in kinds else kinds.index("call")] = 1.0
        return table

    def decide(self, state, cards):
        table = self.table(state)
        return int(np.argmax(table[0])), table


def river_state(pot: int = 1000, stack: int = 9500):
    from app.engine.agent.game_state import new_postflop_state

    return new_postflop_state(
        seats=["SB", "BB"], preflop_prefix=[0, 0, 0, 0, 1, 0], board=NUT_BOARD,
        pot=pot, stacks=[stack, stack], street=3,
    )


def test_live_mask_and_range_conditioning():
    mask = core.live_mask([0, 1])
    assert mask.sum() == core.N_COMBOS - (51 + 50)
    rng = core.ComboRange.uniform([0, 1])
    probs = np.zeros(core.N_COMBOS)
    probs[core.combo_index(2, 3)] = 0.5
    conditioned = rng.conditioned(probs)
    assert conditioned.mass() == pytest.approx(0.5)
    assert conditioned.normalized()[core.combo_index(2, 3)] == pytest.approx(1.0)
    assert conditioned.effective_rows() == pytest.approx(1.0)


def test_sample_runouts_shapes():
    rng = np.random.default_rng(0)
    assert core.sample_runouts(rng, [0, 1], 0, 10).shape == (1, 0)
    assert core.sample_runouts(rng, [0, 1, 2, 3, 4, 5], 1, 10).shape == (46, 1)
    sampled = core.sample_runouts(rng, [0, 1, 2, 3, 4], 2, 50)
    assert sampled.shape == (50, 2) and all(len(set(row)) == 2 for row in sampled.tolist())
    exhaustive = core.sample_runouts(rng, list(range(40)), 2, 1000)
    assert exhaustive.shape == (66, 2)


def test_equity_is_one_with_the_nuts_on_the_river():
    runouts = core.sample_runouts(np.random.default_rng(0), list(NUT_HAND) + NUT_BOARD, 0, 1)
    equity = core.equity_vs_combos(NUT_HAND, NUT_BOARD, runouts, evaluate_7card_batch)
    live = core.live_mask(list(NUT_HAND) + NUT_BOARD)
    assert np.all(equity[live] == 1.0) and np.all(equity[~live] == 0.0)


def test_outcome_and_candidate_values():
    equity = np.full(core.N_COMBOS, 0.5)
    assert core.outcome_values(core.Outcome("showdown", 30, 100), equity)[0] == pytest.approx(20.0)
    assert core.outcome_values(core.Outcome("p_fold", 30, 130), equity)[0] == pytest.approx(100.0)
    assert core.outcome_values(core.Outcome("fold", 0, 100), equity)[0] == 0.0
    responses = np.zeros((core.N_COMBOS, 2))
    responses[:, 0], responses[:, 1] = 0.25, 0.75
    candidate = core.Candidate(
        1, "raise", responses=responses,
        outcomes=(core.Outcome("p_fold", 30, 130), core.Outcome("showdown", 30, 160)),
    )
    weights = np.full(core.N_COMBOS, 1.0 / core.N_COMBOS)
    assert core.candidate_ev(candidate, weights, equity) == pytest.approx(0.25 * 100 + 0.75 * (80 - 30))


def test_choose_action_prefers_check_call_on_ties():
    call = core.Candidate(0, "call", direct=core.Outcome("showdown", 0, 100))
    raise_ = core.Candidate(1, "raise", direct=core.Outcome("showdown", 50, 200))
    assert core.choose_action([raise_, call], [10.0, 10.0]) is call
    assert core.choose_action([raise_, call], [11.0, 10.0]) is raise_


def _decide(policy, hand):
    state = river_state()
    folders = frozenset(range(6)) - {0, 1}
    policy_range = core.ComboRange.uniform(hand)
    action, info = game.exploiter_decide(
        state, hand, 0, 1, folders, policy_range, NUT_BOARD, lambda s: policy.table(s), evaluate_7card_batch,
        np.random.default_rng(0), game.ExploiterConfig(),
    )
    return action, info


def test_exploiter_value_bets_the_nuts_into_a_station_and_checks_air():
    _action, nuts = _decide(StationPolicy(), NUT_HAND)
    assert nuts["chosen"] == "raise" and nuts["kinds"] == ["call", "raise"]
    assert nuts["evs"][0] == pytest.approx(1000.0)  # check: wins the pot at showdown
    assert nuts["evs"][1] == pytest.approx(1000.0 + 750.0)  # 0.75-pot bet, always called and won
    _action, air = _decide(StationPolicy(), AIR_HAND)
    assert air["chosen"] == "call"


def test_exploiter_bluffs_a_folder_with_air():
    _action, air = _decide(FolderPolicy(), AIR_HAND)
    assert air["chosen"] == "raise" and air["evs"][1] == pytest.approx(1000.0)


def test_forced_folds_and_allin_advances():
    from app.engine.agent.preflop_play import new_preflop_state

    state = game.advance_forced(new_preflop_state(), frozenset({2, 3, 4, 5}))
    assert state.to_act_seat() == 0 and state.n_live == 2
    assert game.action_kinds(state)[:2] == ["fold", "call"]


def test_play_hand_against_a_station_is_consistent():
    deal = game.Deal(hole_cards=((8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19)), board=(20, 21, 22, 23, 24))
    record = game.play_hand(
        deal, 1, 0, StationPolicy(), evaluate_7card_batch, np.random.default_rng(3), game.ExploiterConfig(runouts_preflop=20, runouts_flop=20, terminal_samples=20),
    )
    assert record["terminal"] in ("showdown", "p_fold", "fold")
    assert record["pot"] >= 150 and record["l_contrib"] >= 50
    assert np.isfinite(record["l_net_expected"]) and np.isfinite(record["l_net_realized"])
    assert any(d["actor"] == "exploiter" for d in record["decisions"])


def test_expected_equals_realized_when_the_range_is_the_actual_hand():
    deal = game.Deal(hole_cards=(AIR_HAND, NUT_HAND, (30, 31), (32, 33), (34, 35), (36, 37)), board=tuple(NUT_BOARD))
    state = river_state()
    one_hot = np.zeros(core.N_COMBOS)
    one_hot[core.combo_index(*NUT_HAND)] = 1.0
    trace = game.HandTrace(policy_range=core.ComboRange(one_hot))
    realized, expected = game._showdown_values(
        state, deal, 1, 0, trace, evaluate_7card_batch, np.random.default_rng(0), game.ExploiterConfig(), 500,
    )
    assert realized == pytest.approx(-500.0) and expected == pytest.approx(realized)
