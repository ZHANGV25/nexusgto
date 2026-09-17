"""Root value audit: believed-value walk, blueprint rollouts, and range mapping (no blueprint needed)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import lbr_core as core
from scripts import root_value_audit as audit
from scripts.match_aivat import evaluate_7card_batch


def card(rank: str, suit: str) -> int:
    return "23456789TJQKA".index(rank) * 4 + "cdhs".index(suit)


BOARD = [card("2", "s"), card("3", "s"), card("4", "s"), card("K", "d"), card("9", "h")]


class _Mix:
    def __init__(self, freqs):
        self.freqs = freqs

    def sample(self, rng):
        return int(rng.choice(list(self.freqs), p=list(self.freqs.values())))


class _CheckCallLookup:
    def get_mix(self, state, cards):
        from scripts.lbr_game import call_index

        return _Mix({call_index(state): 1.0})


class _AlwaysBetLookup:
    def get_mix(self, state, cards):
        from scripts.lbr_game import action_kinds

        kinds = action_kinds(state)
        return _Mix({kinds.index("raise") if "raise" in kinds else kinds.index("call"): 1.0})


def flop_root():
    from app.engine.agent.game_state import new_postflop_state

    return new_postflop_state(seats=["SB", "BB"], preflop_prefix=[0, 0, 0, 0, 1, 0], board=BOARD, pot=1000, stacks=[9500, 9500], street=1)


def test_root_action_values_reads_leaves_and_folds():
    from es_resolver.es_solver import T_DECISION, T_FOLD, T_LEAF

    hero = [(card("A", "c"), card("A", "d"), 1.0), (card("7", "c"), card("2", "d"), 1.0)]
    opp = [(card("Q", "c"), card("Q", "d"), 1.0), (card("8", "c"), card("6", "d"), 3.0)]
    nodes = [
        SimpleNamespace(type=T_DECISION, player=0, children=(1, 2), pot=1000, bets=(0, 0), leaf_idx=-1),
        SimpleNamespace(type=T_LEAF, player=-1, children=(), pot=1000, bets=(0, 0), leaf_idx=0),
        SimpleNamespace(type=T_DECISION, player=1, children=(3, 4), pot=1750, bets=(750, 0), leaf_idx=-1),
        SimpleNamespace(type=T_FOLD, player=0, children=(), pot=1750, bets=(750, 0), leaf_idx=-1),
        SimpleNamespace(type=T_LEAF, player=-1, children=(), pot=2500, bets=(750, 750), leaf_idx=1),
    ]
    tree = SimpleNamespace(num_players=2, num_hands=[2, 2], player_hands=[hero, opp], nodes=nodes)
    table = np.zeros((2, 2, 2))
    table[0, 0] = [600.0, 200.0]  # check leaf: hero values per hero hand
    table[1, 0] = [1500.0, -400.0]  # bet-call leaf
    sigma = {2: np.array([[0.0, 1.0], [1.0, 0.0]])}  # QQ calls, 86 folds
    values = audit.root_action_values(tree, table, sigma, 0)
    # opponent prior mass: QQ 1, 86 3 -> fold share 0.75, call share 0.25
    assert values[:, 0] == pytest.approx([600.0, 200.0])
    assert values[:, 1] == pytest.approx([0.75 * 1000.0 + 0.25 * 1500.0, 0.75 * 1000.0 + 0.25 * -400.0])


def test_mixed_profile_values_the_root_strategy_under_each_opponent_model():
    from es_resolver.es_solver import T_DECISION, T_FOLD, T_LEAF
    from scripts import opponent_model_check as opp

    hero = [(card("A", "c"), card("A", "d"), 1.0), (card("7", "c"), card("2", "d"), 1.0)]
    villain = [(card("Q", "c"), card("Q", "d"), 1.0), (card("8", "c"), card("6", "d"), 1.0)]
    nodes = [
        SimpleNamespace(type=T_DECISION, player=0, children=(1, 2), pot=1000, bets=(0, 0), leaf_idx=-1),
        SimpleNamespace(type=T_LEAF, player=-1, children=(), pot=1000, bets=(0, 0), leaf_idx=0),
        SimpleNamespace(type=T_DECISION, player=1, children=(3, 4), pot=1750, bets=(750, 0), leaf_idx=-1),
        SimpleNamespace(type=T_FOLD, player=0, children=(), pot=1750, bets=(750, 0), leaf_idx=-1),
        SimpleNamespace(type=T_LEAF, player=-1, children=(), pot=2500, bets=(750, 750), leaf_idx=1),
    ]
    tree = SimpleNamespace(num_players=2, num_hands=[2, 2], player_hands=[hero, villain], nodes=nodes)
    table = np.zeros((2, 2, 2))
    table[0, 0], table[1, 0] = [500.0, 500.0], [1000.0, -750.0]
    search = {0: np.array([[0.0, 1.0], [0.0, 1.0]]), 2: np.array([[0.0, 1.0], [0.0, 1.0]])}   # hero always bets, villain always calls
    blueprint = {0: np.array([[1.0, 0.0], [1.0, 0.0]]), 2: np.array([[1.0, 0.0], [1.0, 0.0]])}  # hero checks, villain folds
    profile = opp.mixed_profile(tree, 0, search, blueprint)
    assert set(profile) == {0, 2} and np.all(profile[0] == search[0]) and np.all(profile[2] == blueprint[2])
    assert opp.strategy_value(tree, table, opp.mixed_profile(tree, 0, search, search), 0) == pytest.approx([1000.0, -750.0])
    assert opp.strategy_value(tree, table, profile, 0) == pytest.approx([1000.0, 1000.0])  # bets into a villain who folds
    assert opp.strategy_value(tree, table, opp.mixed_profile(tree, 0, blueprint, blueprint), 0) == pytest.approx([500.0, 500.0])


def test_project_blueprint_covers_flop_decision_nodes_and_skips_leaves():
    from es_resolver.es_solver import T_DECISION, T_FOLD, T_LEAF
    from scripts import opponent_model_check as opp

    hands = [(card("A", "c"), card("A", "d"), 1.0), (card("7", "c"), card("2", "d"), 0.0)]
    nodes = [
        SimpleNamespace(type=T_DECISION, player=0, children=(1, 2)),
        SimpleNamespace(type=T_LEAF, player=-1, children=()),
        SimpleNamespace(type=T_DECISION, player=1, children=(3, 4, 5)),
        SimpleNamespace(type=T_FOLD, player=0, children=()),
        SimpleNamespace(type=T_LEAF, player=-1, children=()),
        SimpleNamespace(type=T_LEAF, player=-1, children=()),  # the raise line is cut at a leaf for this test
    ]
    tree = SimpleNamespace(nodes=nodes, player_hands=[hands, hands])

    class _Lookup:
        def get_mix(self, state, cards):
            from scripts.lbr_game import action_kinds

            kinds = action_kinds(state)
            return _Mix({kinds.index("call"): 0.7, len(kinds) - 1: 0.3})

    sigma = opp.project_blueprint(tree, flop_root(), _Lookup())
    assert set(sigma) == {0, 2}
    assert sigma[0][0] == pytest.approx([0.7, 0.3]) and sigma[0][1] == pytest.approx([0.5, 0.5])  # zero-weight row stays uniform
    assert sigma[2][0] == pytest.approx([0.0, 0.7, 0.3])  # facing a bet: fold, call, raise


def test_rollout_accounting_check_through_and_bet_lines():
    root = flop_root()
    cards = [(card("5", "s"), card("6", "s")), (card("A", "c"), card("A", "d"))]  # SB straight flush beats BB aces
    rng = np.random.default_rng(0)
    assert audit.rollout(root, 0, 0, cards, _CheckCallLookup(), rng, evaluate_7card_batch) == pytest.approx(1000.0)
    assert audit.rollout(root, 0, 1, cards, _CheckCallLookup(), rng, evaluate_7card_batch) == pytest.approx(0.0)
    bet_line = audit.rollout(root, 1, 0, cards, _AlwaysBetLookup(), rng, evaluate_7card_batch)
    assert bet_line > 1000.0  # the hero wins a pot grown by the bets it made


def test_assumed_reach_spreads_rows_over_isomorphic_members():
    from app.services.suit_isomorphism import group_by_canonical

    live = [core.COMBOS[int(i)] for i in np.nonzero(core.live_mask(BOARD[:3]))[0]]
    hands = [(canon[0], canon[1], float(len(members))) for canon, members in group_by_canonical(live, BOARD[:3])]
    tree = SimpleNamespace(num_players=2, num_hands=[len(hands), len(hands)], player_hands=[hands, hands], nodes=[])
    weights = audit.assumed_reach(tree, 0, BOARD[:3])
    assert weights.sum() == pytest.approx(sum(w for _c0, _c1, w in hands))
    assert np.all(weights[core.live_mask(BOARD[:3])] == pytest.approx(1.0))
    mapping = audit.row_of_combo(tree, 0, BOARD[:3])
    assert len(mapping) == int(core.live_mask(BOARD[:3]).sum())


def test_leaf_states_follow_every_line_to_the_street_end():
    from es_resolver.es_solver import T_DECISION, T_FOLD, T_LEAF
    from scripts import leaf_value_check as leaf

    def node(kind, player, children, leaf_idx=-1):
        return SimpleNamespace(type=kind, player=player, children=children, leaf_idx=leaf_idx)

    nodes = [
        node(T_DECISION, 0, (1, 2)),          # hero: check -> 1, bet -> 2
        node(T_DECISION, 1, (3, 4)),          # opp after check: check -> leaf, bet -> hero
        node(T_DECISION, 1, (7, 8)),          # opp after bet: fold, call -> leaf
        node(T_LEAF, -1, (), 0),
        node(T_DECISION, 0, (5, 6)),          # hero facing bet: fold, call -> leaf
        node(T_FOLD, 1, ()),
        node(T_LEAF, -1, (), 1),
        node(T_FOLD, 0, ()),
        node(T_LEAF, -1, (), 2),
    ]
    tree = SimpleNamespace(nodes=nodes)
    root = flop_root()
    leaves = leaf.leaf_states(tree, root)
    by_line = {entry["line"]: entry for entry in leaves}
    assert set(by_line) == {"cc", "cbc", "bc"}
    assert int(by_line["cc"]["state"].street) == 2 and by_line["cc"]["state"].pot == root.pot
    assert by_line["bc"]["state"].pot == root.pot + 2 * 750 and by_line["bc"]["leaf_idx"] == 2
    assert [player for _s, player, _a in by_line["cbc"]["decisions"]] == [0, 1, 0]

    class _Lookup:
        def get_mix(self, state, cards):
            return _Mix({0: 0.5, 1: 0.5})

    weights = leaf.conditioned_opponent(np.ones(core.N_COMBOS), by_line["cbc"]["decisions"], 1, _Lookup())
    live = core.live_mask(BOARD[:3])
    assert np.allclose(weights[live], 0.5) and np.all(weights[~live] == 0.0)  # one opponent decision, board-blocked combos dropped


def test_true_reach_multiplies_action_probabilities():
    class _Lookup:
        def get_mix(self, state, cards):
            return _Mix({0: 0.25, 1: 0.75}) if cards == core.COMBOS[0] else _Mix({0: 1.0})

    state = SimpleNamespace(street=0, board=(), num_legal_actions=lambda: 2)
    weights = audit.true_reach([(state, 3, 1), (state, 4, 0)], 3, _Lookup())
    assert weights[0] == pytest.approx(0.75) and weights[1] == pytest.approx(0.0)
