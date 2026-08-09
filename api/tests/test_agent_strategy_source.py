"""TDD: strategy_source — LookupSource (plan §3.4, §4c).

LookupSource reads blueprint σ̄ for the node key_builder constructs and
returns a {action_idx: freq} mix. Behavioral tests walk realistic 2p + 3p
lines and assert HIT (the key is right) + decisive-node poker-sanity (the
strategy is right) — the HIT/sanity split convention. LOOKUP MUST NOT call
the GPU (no StreetSolverGPU import / construction).

Pure-unit tests (mix shape, miss-handling) use a fake source; the blueprint
behavioral tests SKIP if the 8.05B isn't present.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.engine.agent.game_state import new_postflop_state
from app.engine.agent.modes import ModeConfig
from app.engine.agent.strategy_source import (
    ActionMix,
    LookupSource,
    OfflineLiftedSource,
    StrategySource,
)


def _ci(card: str) -> int:
    ranks = "AKQJT98765432"
    so = ("c", "d", "h", "s")
    return (12 - ranks.index(card[0].upper())) * 4 + so.index(card[1].lower())


class _StubStrategyReader:
    def __init__(self, strategy):
        self.strategy = strategy

    def get_metadata(self):
        return {}

    def hand_to_bucket(self, *args):
        return 0

    def get_strategy(self, *args, **kwargs):
        return self.strategy


def _stub_lookup_state(*, facing_bet: bool):
    state = new_postflop_state(
        seats=["BB", "UTG"],
        preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td")],
        pot=440,
        stacks=[9780, 9780],
    )
    return state.advance(1) if facing_bet else state


def _stub_lookup(strategy):
    return LookupSource(
        ModeConfig(mode="LOOKUP"), reader=_StubStrategyReader(strategy)
    )


# ── ActionMix invariants ──────────────────────────────────────────────────


def test_action_mix_normalizes_and_samples():
    mix = ActionMix({0: 0.25, 1: 0.75})
    assert abs(sum(mix.freqs.values()) - 1.0) < 1e-9
    # deterministic sample with a seeded rng
    rng = np.random.default_rng(0)
    samples = [mix.sample(rng) for _ in range(2000)]
    frac1 = sum(1 for s in samples if s == 1) / len(samples)
    assert 0.70 < frac1 < 0.80  # ~0.75


def test_action_mix_rejects_empty():
    with pytest.raises(ValueError):
        ActionMix({})


def test_action_mix_uniform_factory():
    mix = ActionMix.uniform(3)
    assert set(mix.freqs.keys()) == {0, 1, 2}
    assert all(abs(f - 1 / 3) < 1e-9 for f in mix.freqs.values())




def test_lookup_shorter_facing_bet_zero_fills_pruned_raise_tail():
    state = _stub_lookup_state(facing_bet=True)
    assert state.num_legal_actions() == 3

    mix = _stub_lookup((2.0, 6.0)).get_mix(
        state, hero_cards=(_ci("Qh"), _ci("Jh"))
    )

    assert mix.freqs == pytest.approx({0: 0.25, 1: 0.75, 2: 0.0})
    assert sum(mix.freqs.values()) == pytest.approx(1.0)
    assert not mix.is_uniform_fallback


def test_lookup_shorter_not_facing_bet_zero_fills_pruned_bet_tail():
    state = _stub_lookup_state(facing_bet=False)
    assert state.num_legal_actions() == 2

    mix = _stub_lookup((4.0,)).get_mix(
        state, hero_cards=(_ci("Qh"), _ci("Jh"))
    )

    assert mix.freqs == pytest.approx({0: 1.0, 1: 0.0})
    assert sum(mix.freqs.values()) == pytest.approx(1.0)
    assert not mix.is_uniform_fallback


def test_lookup_shorter_than_nonraise_slots_raises_value_error():
    state = _stub_lookup_state(facing_bet=True)

    with pytest.raises(ValueError, match="blueprint stores 1 actions"):
        _stub_lookup((1.0,)).get_mix(
            state, hero_cards=(_ci("Qh"), _ci("Jh"))
        )


def test_lookup_exact_action_count_path_remains_unchanged():
    state = _stub_lookup_state(facing_bet=False)

    mix = _stub_lookup((2.0, 6.0)).get_mix(
        state, hero_cards=(_ci("Qh"), _ci("Jh"))
    )

    assert mix.freqs == pytest.approx({0: 0.25, 1: 0.75})


def test_lookup_longer_action_count_still_truncates_raise_tail():
    state = _stub_lookup_state(facing_bet=False)

    mix = _stub_lookup((2.0, 6.0, 9.0)).get_mix(
        state, hero_cards=(_ci("Qh"), _ci("Jh"))
    )

    assert mix.freqs == pytest.approx({0: 0.25, 1: 0.75})


# ── ABC contract ──────────────────────────────────────────────────────────


def test_lookup_source_is_a_strategy_source(blueprint_path):
    src = LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path))
    assert isinstance(src, StrategySource)


def test_offline_lifted_source_is_a_strategy_source():
    src = OfflineLiftedSource(ModeConfig(mode="OFFLINE_LIFTED", cache_path=None))
    assert isinstance(src, StrategySource)


# ── LookupSource behavioral (blueprint oracle) ───────────────────────────


def test_lookup_3p_flop_root_hits_and_is_legal(blueprint_path):
    src = LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path))
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td")], pot=720,
        stacks=[9660, 9780, 9780],
    )
    mix = src.get_mix(st, hero_cards=(_ci("Qh"), _ci("Jh")))
    # no-bet vocab [check0, bet1] -> 2 legal actions
    assert set(mix.freqs.keys()) <= {0, 1}
    assert abs(sum(mix.freqs.values()) - 1.0) < 1e-6
    assert not mix.is_uniform_fallback  # genuine HIT


def test_lookup_faced_bet_3way_is_call_heavy(blueprint_path):
    src = LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path))
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td")], pot=720,
        stacks=[9660, 9780, 9780],
    )
    st = st.advance(1)  # BB bet -> UTG (QhJh) faces it
    mix = src.get_mix(st, hero_cards=(_ci("Qh"), _ci("Jh")))
    # facing-bet [fold0, call1, raise2]
    assert set(mix.freqs.keys()) <= {0, 1, 2}
    assert mix.freqs.get(1, 0.0) > 0.25  # meaningful call mass
    assert not mix.is_uniform_fallback


def test_lookup_miss_falls_back_to_uniform_when_configured(blueprint_path):
    # An impossible/never-trained node (deep bogus history) -> MISS -> uniform.
    src = LookupSource(ModeConfig(
        mode="LOOKUP", blueprint_path=blueprint_path, uniform_on_miss=True))
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"],
        preflop_prefix=[2, 0, 0, 1, 0, 1, 1, 1, 1, 1],  # garbage tail
        board=[_ci("Ah"), _ci("Ks"), _ci("Td")], pot=720,
        stacks=[9660, 9780, 9780],
    )
    mix = src.get_mix(st, hero_cards=(_ci("Qh"), _ci("Jh")))
    assert mix.is_uniform_fallback
    assert abs(sum(mix.freqs.values()) - 1.0) < 1e-6


def test_lookup_miss_raises_when_not_configured(blueprint_path):
    src = LookupSource(ModeConfig(
        mode="LOOKUP", blueprint_path=blueprint_path, uniform_on_miss=False))
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"],
        preflop_prefix=[2, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td")], pot=720,
        stacks=[9660, 9780, 9780],
    )
    with pytest.raises(LookupError):
        src.get_mix(st, hero_cards=(_ci("Qh"), _ci("Jh")))


def test_lookup_does_not_import_gpu_solver(blueprint_path):
    # LOOKUP is sub-second and MUST NOT touch the GPU. Assert no StreetSolverGPU
    # was imported as a side effect of a LookupSource get_mix.
    import sys
    sys.modules.pop("street_solver_gpu", None)
    src = LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint_path))
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td")], pot=440,
        stacks=[9780, 9780],
    )
    src.get_mix(st, hero_cards=(_ci("Qh"), _ci("Jh")))
    assert "street_solver_gpu" not in sys.modules


# ── OfflineLiftedSource stub ──────────────────────────────────────────────


def test_offline_lifted_cache_miss_raises_without_cache():
    src = OfflineLiftedSource(ModeConfig(mode="OFFLINE_LIFTED", cache_path=None))
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[_ci("Ah"), _ci("Ks"), _ci("Td")], pot=440,
        stacks=[9780, 9780],
    )
    with pytest.raises(NotImplementedError):
        src.get_mix(st, hero_cards=(_ci("Qh"), _ci("Jh")))
