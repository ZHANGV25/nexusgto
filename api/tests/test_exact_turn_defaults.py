"""The production defaults of the exact-turn configuration (2026-09-09).

Source-level pins: these knobs are read from the environment inside
CpuEsSolver.solve, so the defaults are asserted where they are read.
"""
from __future__ import annotations

import inspect

from app.engine.cpu_es_solver import CpuEsSolver
from app.engine.engine_constants import (
    DEFAULT_LIVE_CONFIG_BY_N,
    LIVE_2P3P_ITERATIONS,
    PLAY_LIVE_CONFIG_BY_N,
    live_config_for,
)


def test_two_and_three_player_defaults_are_vec_final_20k():
    assert LIVE_2P3P_ITERATIONS == 20_000
    for n in (2, 3):
        cfg = DEFAULT_LIVE_CONFIG_BY_N[n]
        assert cfg.vec_mode is True
        assert cfg.readout == "final"
        assert cfg.iterations == LIVE_2P3P_ITERATIONS
        assert PLAY_LIVE_CONFIG_BY_N[n] is cfg
        assert live_config_for(n).readout == "final"
        assert live_config_for(n).vec_mode is True


def test_solver_defaults_extend_the_turn_sample_the_river_and_skip_reach():
    source = inspect.getsource(CpuEsSolver.solve)
    assert source.count('"NEXUS_ALLOW_CHANCE_SOLVE", "1")') == 2
    assert source.count('"NEXUS_ES_REACH", "0")') == 2
    assert '"NEXUS_CHANCE_SAMPLE", "1")' in source
    assert '"NEXUS_FREEZE_HERO_LINE", "1")' in source
    assert '"NEXUS_EXACT_ALLIN_LEAVES", "1")' in source
