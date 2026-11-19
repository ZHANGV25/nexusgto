"""Env parsing and gating for the river average-readout research A/B."""
from __future__ import annotations

import pytest

from app.engine import cpu_es_solver as solver
from scripts import run_match_cli as cli


@pytest.mark.parametrize("raw,expected", [
    (None, False), ("", False), ("0", False), ("1", True), ("true", True), ("Yes", True),
])
def test_river_avg_flag_parses(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("NEXUS_RIVER_AVG_READOUT", raising=False)
    else:
        monkeypatch.setenv("NEXUS_RIVER_AVG_READOUT", raw)
    assert solver._river_avg_readout_enabled() is expected


def test_river_avg_iterations_default_and_override(monkeypatch):
    monkeypatch.delenv("NEXUS_RIVER_AVG_ITERATIONS", raising=False)
    assert solver._river_avg_iterations() == 5_000_000
    monkeypatch.setenv("NEXUS_RIVER_AVG_ITERATIONS", "1500000")
    assert solver._river_avg_iterations() == 1_500_000
    for bad in ("x", "0", "-5"):
        monkeypatch.setenv("NEXUS_RIVER_AVG_ITERATIONS", bad)
        with pytest.raises(ValueError):
            solver._river_avg_iterations()


def test_river_avg_applies_only_to_heads_up_rivers(monkeypatch):
    monkeypatch.setenv("NEXUS_RIVER_AVG_READOUT", "1")
    assert solver.river_avg_applies("river", 2)
    assert not solver.river_avg_applies("river", 3)
    assert not solver.river_avg_applies("turn", 2)
    monkeypatch.delenv("NEXUS_RIVER_AVG_READOUT", raising=False)
    assert not solver.river_avg_applies("river", 2)


def test_run_match_accepts_the_river_avg_env_overrides():
    assert cli._env_override("NEXUS_RIVER_AVG_READOUT=1") == ("NEXUS_RIVER_AVG_READOUT", "1")
    assert cli._env_override("NEXUS_RIVER_AVG_ITERATIONS=2000000") == (
        "NEXUS_RIVER_AVG_ITERATIONS", "2000000"
    )
