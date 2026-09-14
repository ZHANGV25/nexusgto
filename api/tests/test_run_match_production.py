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


def test_fallback_source_records_replayable_spot():
    from types import SimpleNamespace

    state = SimpleNamespace(
        street=2, active=(True, False, True), board=(1, 2, 3, 4, 5),
        preflop_prefix=(3, 1), postflop_history=(1, 2), seat_of_slot=(0, 3, 5),
        stacks=(900, 1000, 800), pot=450, to_act_slot=2,
    )
    source = cli.ProductionFallbackSource(
        _Live(_Refused("hero row mass 0.5 < 10")), _Lookup(), (_Refused,)
    )
    source.get_mix(state, (40, 41))
    assert source.spots == ({
        "error": "_Refused", "message": "hero row mass 0.5 < 10", "street": 2,
        "n_live": 2, "board": [1, 2, 3, 4, 5], "preflop_prefix": [3, 1],
        "postflop_history": [1, 2], "seat_of_slot": [0, 3, 5],
        "stacks": [900, 1000, 800], "pot": 450, "to_act_slot": 2,
        "hero_cards": [40, 41],
    },)


def test_fallback_source_gates_streets_to_lookup_without_solving():
    from types import SimpleNamespace

    live = _Live(_Refused("must not solve"))
    source = cli.ProductionFallbackSource(
        live, _Lookup(), (_Refused,), live_streets=frozenset({3})
    )
    assert source.get_mix(SimpleNamespace(street=1), (0, 1)) == "lookup"
    assert source.gated == 1 and source.fallbacks == {}
    live.error = None
    assert source.get_mix(SimpleNamespace(street=3), (0, 1)) == "live"
    assert source.gated == 1


def test_fallback_source_counts_belief_carry_engagement():
    live = _Live()
    source = cli.ProductionFallbackSource(live, _Lookup(), (_Refused,))
    source.get_mix(None, (0, 1))
    assert source.belief_carry == {}  # flag-off contract: no summary
    live.last_belief_carry = {"applied": True, "dropped": False, "folded_streets": [{}]}
    source.get_mix(None, (0, 1))
    live.last_belief_carry = {"applied": False, "dropped": True, "folded_streets": [],
                              "drop_reason": "belief_carry: dropped (x)"}
    live.error = _Refused("no")
    source.get_mix(None, (0, 1))
    assert source.belief_carry == {"solves": 2, "applied": 1, "folded": 1, "dropped": 1}
    assert source.belief_drop_reasons == {"belief_carry: dropped (x)": 1}


def test_parse_args_river_capture_requires_production(tmp_path):
    base = ["--mode", "ab", "--out", str(tmp_path), "--field-source", "lookup"]
    args = cli.parse_args(
        run_match,
        base + ["--hero-source", "production", "--capture-river-brgap", str(tmp_path / "rv"),
                "--capture-limit", "40"],
    )
    assert args.capture_river_brgap == str(tmp_path / "rv") and args.capture_limit == 40
    with pytest.raises(SystemExit):
        cli.parse_args(run_match, base + ["--capture-river-brgap", str(tmp_path / "rv")])


def test_parse_args_production_live_streets(tmp_path):
    base = ["--mode", "ab", "--out", str(tmp_path), "--field-source", "lookup"]
    args = cli.parse_args(
        run_match,
        base + ["--hero-source", "production", "--production-live-streets", "turn, river"],
    )
    assert args.production_live_streets == frozenset({2, 3})
    assert cli.street_names(args.production_live_streets) == ["turn", "river"]
    for bad in (["--hero-source", "production", "--production-live-streets", "preflop"],
                ["--production-live-streets", "flop"]):
        with pytest.raises(SystemExit):
            cli.parse_args(run_match, base + bad)


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


def test_parse_args_rejects_ab_overrides_without_production(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args(
            run_match,
            ["--mode", "aa", "--out", str(tmp_path), "--belief-carry"],
        )


class _Cfg:
    def __init__(self, n_players, iterations, deadline_ms):
        self.n_players = n_players
        self.iterations = iterations
        self.vec_mode = True
        self.readout = "final"
        self.num_rollouts = 3
        self.deadline_ms = deadline_ms
        self.hero_traverser_actual_fraction = 0.5
        self.hero_row_min_strategy_sum_mass = 10.0


class _FakeLive:
    created = []

    def __init__(self, config, **kwargs):
        self.kwargs = kwargs
        _FakeLive.created.append(self)


def _production(monkeypatch, **kwargs):
    import app.engine.agent.strategy_source as strategy_source

    _FakeLive.created = []
    monkeypatch.setattr(strategy_source, "LiveResolveSource", _FakeLive)
    calls = []

    def live_config_for(n, iterations_override=None, deadline_ms_override=None, purpose="display"):
        calls.append((n, iterations_override, deadline_ms_override, purpose))
        return _Cfg(n, iterations_override or 100_000, deadline_ms_override or 3_000)

    strategy, sources = cli._production_strategy(
        path="bp", lookup=_Lookup(), engine=object(),
        live_config_for_fn=live_config_for, num_threads=8, **kwargs,
    )
    return strategy, sources, calls


def test_production_strategy_shares_identical_configs_and_caps_players(monkeypatch):
    strategy, sources, calls = _production(monkeypatch)
    assert [c[0] for c in calls] == [2, 3]
    assert all(c[3] == "play" for c in calls)
    assert len(sources) == 1 and len(_FakeLive.created) == 1
    live_by_count = strategy._live_by_count
    assert live_by_count[2] is live_by_count[3] is sources[0]
    assert all(isinstance(live_by_count[n], _Lookup) for n in (4, 5, 6))
    assert _FakeLive.created[0].kwargs["belief_carry"] is None
    assert _FakeLive.created[0].kwargs["deadline_ms"] == 3_000


def test_production_strategy_applies_ab_overrides(monkeypatch):
    _, _, calls = _production(
        monkeypatch, iterations_override=1_000_000, deadline_ms_override=10_000,
        belief_carry=True,
    )
    assert calls[0][1:3] == (1_000_000, 10_000)
    kwargs = _FakeLive.created[0].kwargs
    assert kwargs["iterations"] == 1_000_000
    assert kwargs["deadline_ms"] == 10_000
    assert kwargs["belief_carry"] is True


def test_production_strategy_passes_live_streets(monkeypatch):
    _, sources, _ = _production(monkeypatch, live_streets=frozenset({1}))
    assert sources[0]._live_streets == frozenset({1})


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
