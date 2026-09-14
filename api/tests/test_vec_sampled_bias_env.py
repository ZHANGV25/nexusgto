"""Env parsing for the vec-sampled leaf oracle's research bias modes."""
from __future__ import annotations

import pytest

from app.engine import cpu_es_solver as solver
from scripts import run_match
from scripts import run_match_cli as cli


@pytest.mark.parametrize("raw,expected", [
    (None, 0), ("", 0), ("regret", 0), ("uniform", 1), ("UNBIASED", 2),
])
def test_opp_bias_mode_parses_names(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("NEXUS_VEC_SAMPLED_OPP_BIAS", raising=False)
    else:
        monkeypatch.setenv("NEXUS_VEC_SAMPLED_OPP_BIAS", raw)
    assert solver._vec_sampled_opp_bias_mode() == expected


def test_opp_bias_mode_rejects_unknown(monkeypatch):
    monkeypatch.setenv("NEXUS_VEC_SAMPLED_OPP_BIAS", "fold")
    with pytest.raises(ValueError, match="regret, uniform or unbiased"):
        solver._vec_sampled_opp_bias_mode()


@pytest.mark.parametrize("raw,expected", [
    (None, False), ("0", False), ("1", True), ("true", True), ("no", False),
])
def test_hero_bias_fixed_parses_flag(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("NEXUS_VEC_SAMPLED_HERO_FIXED", raising=False)
    else:
        monkeypatch.setenv("NEXUS_VEC_SAMPLED_HERO_FIXED", raw)
    assert solver._vec_sampled_hero_bias_fixed() is expected


@pytest.mark.parametrize("raw,expected", [
    (None, 0), ("", 0), ("traversal", 0), ("ROOT", 1),
])
def test_opp_hand_mode_parses_names(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("NEXUS_VEC_SAMPLED_OPP_HAND", raising=False)
    else:
        monkeypatch.setenv("NEXUS_VEC_SAMPLED_OPP_HAND", raw)
    assert solver._vec_sampled_opp_hand_mode() == expected


def test_opp_hand_mode_rejects_unknown(monkeypatch):
    monkeypatch.setenv("NEXUS_VEC_SAMPLED_OPP_HAND", "uniform")
    with pytest.raises(ValueError, match="traversal or root"):
        solver._vec_sampled_opp_hand_mode()


def test_opp_hand_flag_is_production_overridable(tmp_path):
    args = cli.parse_args(run_match, [
        "--mode", "ab", "--out", str(tmp_path), "--hero-source", "production",
        "--field-source", "lookup",
        "--production-env", "NEXUS_VEC_SAMPLED_OPP_HAND=root",
    ])
    assert args.production_env == [("NEXUS_VEC_SAMPLED_OPP_HAND", "root")]


def test_bias_mode_flags_are_production_overridable(tmp_path):
    args = cli.parse_args(run_match, [
        "--mode", "ab", "--out", str(tmp_path), "--hero-source", "production",
        "--field-source", "lookup",
        "--production-env", "NEXUS_VEC_SAMPLED_OPP_BIAS=unbiased",
        "--production-env", "NEXUS_VEC_SAMPLED_HERO_FIXED=1",
    ])
    assert args.production_env == [
        ("NEXUS_VEC_SAMPLED_OPP_BIAS", "unbiased"),
        ("NEXUS_VEC_SAMPLED_HERO_FIXED", "1"),
    ]
