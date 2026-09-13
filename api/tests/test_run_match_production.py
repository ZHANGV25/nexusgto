"""Pure tests for the run_match 'production' source (no blueprint needed)."""
from __future__ import annotations

import argparse
import os

import pytest

from scripts import run_match
from scripts import run_match_cli as cli

SERVE_ENV = dict(cli.PRODUCTION_SERVE_ENV)


def test_production_env_problems_accepts_exact_serve_env():
    assert cli.production_env_problems(SERVE_ENV) == []


def test_production_env_problems_reports_missing_wrong_and_foreign_flags():
    environ = {**SERVE_ENV, "NEXUS_OFFTREE_ADD": "0", "NEXUS_LEAF_SAMPLED": "1"}
    del environ["NEXUS_LEAF_VEC_SAMPLED"]
    problems = cli.production_env_problems(environ)
    assert any(p.startswith("NEXUS_LEAF_VEC_SAMPLED=None") for p in problems)
    assert any(p.startswith("NEXUS_OFFTREE_ADD='0'") for p in problems)
    assert any(p.startswith("NEXUS_LEAF_SAMPLED='1'") for p in problems)


class _Refused(RuntimeError):
    pass


class _Live:
    def __init__(self, error=None):
        self.error = error

    def get_mix(self, state, hero_cards):
        if self.error is not None:
            raise self.error
        return "live"


class _Lookup:
    def get_mix(self, state, hero_cards):
        return "lookup"


def test_fallback_source_serves_live_mix():
    source = cli.ProductionFallbackSource(_Live(), _Lookup(), (_Refused,))
    assert source.get_mix(None, (0, 1)) == "live"
    assert source.fallbacks == {}


def test_fallback_source_acts_on_lookup_after_refusal():
    source = cli.ProductionFallbackSource(_Live(_Refused()), _Lookup(), (_Refused,))
    assert source.get_mix(None, (0, 1)) == "lookup"
    assert source.get_mix(None, (0, 1)) == "lookup"
    assert source.fallbacks == {"_Refused": 2}


def test_fallback_source_propagates_unlisted_errors():
    source = cli.ProductionFallbackSource(
        _Live(KeyError("bug")), _Lookup(), (_Refused,)
    )
    with pytest.raises(KeyError):
        source.get_mix(None, (0, 1))


def test_production_scope_leaves_environment_untouched(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_SAMPLED", "sentinel")
    assert cli.resolved_leaf_mode_env("production") == {}
    with cli.scoped_leaf_mode("production"):
        assert os.environ["NEXUS_LEAF_SAMPLED"] == "sentinel"


def test_parse_args_accepts_production_ab_with_threads(tmp_path):
    args = cli.parse_args(
        run_match,
        [
            "--mode", "ab", "--out", str(tmp_path), "--hero-source", "production",
            "--field-source", "lookup", "--live-threads", "8", "--aivat",
        ],
    )
    assert args.hero_source == "production"
    assert args.live_threads == 8
    assert args.aivat is True


def _namespace(**overrides):
    base = dict(
        seats=6, mode="ab", hero_source="production", field_source="lookup",
        blueprint_a=None, blueprint_b=None, live_iterations=250_000,
        live_num_rollouts=None, max_wall_s=None, live_threads=0,
    )
    return argparse.Namespace(**{**base, **overrides})


def test_build_factories_refuses_production_without_serve_env(monkeypatch):
    for name in SERVE_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="server environment"):
        cli.build_factories(run_match, _namespace())


def test_build_factories_refuses_mixed_live_environments(monkeypatch):
    for name, value in SERVE_ENV.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="cannot share a process"):
        cli.build_factories(run_match, _namespace(field_source="live_sampled"))
