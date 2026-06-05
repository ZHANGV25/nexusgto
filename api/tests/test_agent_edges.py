"""Edge / validation coverage for the agent package.

Targets the error branches and the OFFLINE_LIFTED populated-cache path so the
critical core stays at 100% and the package overall stays well above 80%.
No blueprint needed.
"""

from __future__ import annotations

import pytest

from app.agent.game_state import new_postflop_state
from app.agent.modes import ModeConfig
from app.agent.preflop_constructor import (
    PreflopAction,
    PreflopConstructionUnsupported,
    build_preflop_prefix,
)
from app.agent.strategy_source import (
    ActionMix,
    LiftedCacheEntry,
    OfflineLiftedSource,
)

# ── modes validation ──────────────────────────────────────────────────────


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        ModeConfig(mode="TELEPORT")


# ── game_state validation ─────────────────────────────────────────────────


def test_factory_rejects_mismatched_stacks():
    with pytest.raises(ValueError):
        new_postflop_state(["BB", "UTG"], [2, 0, 0, 0, 0, 1], [], 440, [9780])


def test_factory_rejects_single_player():
    with pytest.raises(ValueError):
        new_postflop_state(["BB"], [2], [], 100, [9780])


def test_factory_rejects_unknown_position():
    with pytest.raises(ValueError):
        new_postflop_state(["BB", "LJ"], [2, 1], [], 440, [9780, 9780])


def test_advance_on_terminal_raises():
    st = new_postflop_state(["BB", "UTG"], [2, 0, 0, 0, 0, 1], [], 440,
                            [9780, 9780])
    st = st.advance(1)  # BB bet
    st = st.advance(0)  # UTG fold -> terminal
    assert st.terminal
    with pytest.raises(ValueError):
        st.advance(0)


def test_advance_rejects_out_of_range_index():
    st = new_postflop_state(["BB", "UTG"], [2, 0, 0, 0, 0, 1], [], 440,
                            [9780, 9780])
    with pytest.raises(ValueError):
        st.advance(99)


def test_n_live_and_facing_bet_helpers():
    st = new_postflop_state(["BB", "UTG", "BTN"], [2, 0, 0, 1, 0, 1], [], 720,
                            [9660, 9780, 9780])
    assert st.n_live == 3
    assert st.facing_bet is False
    assert st.to_act_seat() == 1
    st = st.advance(1)  # BB bet
    assert st.facing_bet is True


def test_terminal_state_to_act_seat_is_none():
    st = new_postflop_state(["BB", "UTG"], [2, 0, 0, 0, 0, 1], [], 440,
                            [9780, 9780])
    st = st.advance(1).advance(0)  # BB bet, UTG fold
    assert st.to_act_seat() is None
    assert st.facing_bet is False


# ── preflop_constructor edge branches ─────────────────────────────────────


def test_call_before_opener_is_limp_unsupported():
    # SB calls before any raise (a limp from SB) -> unsupported.
    actions = [
        PreflopAction("SB", "call", 50),
        PreflopAction("BB", "raise", 300),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_explicit_fold_action_emits_zero():
    # An explicit fold in the list (UTG folds, MP opens, BB calls).
    actions = [
        PreflopAction("UTG", "fold"),
        PreflopAction("MP", "raise", 250),
        PreflopAction("BB", "call", 250),
    ]
    prefix = build_preflop_prefix(actions)
    # UTG fold=0, MP raise=2, CO fold(inferred)=0, BTN fold=0, SB fold=0, BB call=1
    assert prefix == [0, 2, 0, 0, 0, 1]


# ── OFFLINE_LIFTED populated cache path ───────────────────────────────────


def test_offline_lifted_serves_a_populated_cache_entry():
    st = new_postflop_state(["BB", "UTG"], [2, 0, 0, 0, 0, 1],
                            [10, 20, 30], 440, [9780, 9780])
    # Build a cache entry keyed on the flop-root node the agent will query.
    entry = LiftedCacheEntry(
        action_history=(2, 0, 0, 0, 0, 1), player=1, street=1, bucket=0,
        freqs={0: 0.3, 1: 0.7},
    )
    cache = {(entry.action_history, entry.player, entry.street, entry.bucket): entry}
    src = OfflineLiftedSource(ModeConfig(mode="OFFLINE_LIFTED"), cache=cache)
    mix = src.get_mix(st, hero_cards=(0, 1))
    assert isinstance(mix, ActionMix)
    assert abs(mix.freqs[1] - 0.7) < 1e-9


def test_offline_lifted_sparse_cache_miss_raises():
    st = new_postflop_state(["BB", "UTG"], [2, 0, 0, 0, 0, 1],
                            [10, 20, 30], 440, [9780, 9780])
    other = LiftedCacheEntry(
        action_history=(9, 9, 9), player=0, street=1, bucket=0,
        freqs={0: 1.0},
    )
    cache = {(other.action_history, other.player, other.street, other.bucket): other}
    src = OfflineLiftedSource(ModeConfig(mode="OFFLINE_LIFTED"), cache=cache)
    with pytest.raises(NotImplementedError):
        src.get_mix(st, hero_cards=(0, 1))


def test_two_different_raisers_unsupported():
    # UTG opens, BTN 3-bets (two different raisers) -> unsupported.
    actions = [
        PreflopAction("UTG", "raise", 220),
        PreflopAction("BTN", "raise", 700),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_unsupported_action_string_raises():
    actions = [
        PreflopAction("UTG", "raise", 220),
        PreflopAction("BB", "straddle", 200),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


# ── agent guards (no blueprint) ───────────────────────────────────────────


def test_agent_act_on_terminal_raises():
    from app.agent.agent import PokerAgent
    from app.agent.strategy_source import StrategySource

    class _Src(StrategySource):
        def get_mix(self, state, hero_cards):
            return ActionMix({0: 1.0})

    st = new_postflop_state(["BB", "UTG"], [2, 0, 0, 0, 0, 1], [], 440,
                            [9780, 9780])
    st = st.advance(1).advance(0)  # terminal
    agent = PokerAgent(_Src(), ModeConfig(mode="LOOKUP", seed=0))
    with pytest.raises(ValueError):
        agent.act(st, (0, 1))


def test_agent_mode_property():
    from app.agent.agent import PokerAgent
    from app.agent.strategy_source import StrategySource

    class _Src(StrategySource):
        def get_mix(self, state, hero_cards):
            return ActionMix({0: 1.0})

    agent = PokerAgent(_Src(), ModeConfig(mode="LIVE_RESOLVE", seed=0))
    assert agent.mode == "LIVE_RESOLVE"
