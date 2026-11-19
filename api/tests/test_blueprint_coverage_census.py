"""Pure tests for scripts/blueprint_coverage_census.py (no blueprint needed)."""
from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace

import pytest

from scripts import blueprint_coverage_census as census_mod


def test_classify_row_absent_for_none_and_empty():
    assert census_mod.classify_row(None) == "absent"
    assert census_mod.classify_row([]) == "absent"


def test_classify_row_uniform_matches_reader_padding():
    # bps_reader.c pads with 255/na bytes, which normalize to exactly 1/na.
    assert census_mod.classify_row([1 / 3, 1 / 3, 1 / 3]) == "uniform"
    assert census_mod.classify_row([0.5, 0.5]) == "uniform"


def test_classify_row_trained_for_any_non_uniform_row():
    assert census_mod.classify_row([0.2, 0.8]) == "trained"
    assert census_mod.classify_row([0.0, 0.0, 1.0]) == "trained"


def test_summarize_rates_by_street_and_player_count():
    counts = {
        (1, 3, "trained"): 6,
        (1, 3, "uniform"): 2,
        (2, 2, "absent"): 2,
    }
    summary = census_mod.summarize(counts)
    assert summary["overall"]["lookups"] == 10
    assert summary["overall"]["trained_rate"] == pytest.approx(0.6)
    assert summary["by_street"]["flop"]["uniform_rate"] == pytest.approx(0.25)
    assert summary["by_street_players"]["turn/2p"]["absent_rate"] == 1.0


def test_summarize_empty_counts_has_null_rates():
    summary = census_mod.summarize({})
    assert summary["overall"]["lookups"] == 0
    assert summary["overall"]["trained_rate"] is None


Key = namedtuple("Key", "action_history player street")


class _FakeReader:
    def __init__(self, strategy):
        self._strategy = strategy
        self.queries = []

    def get_strategy(self, board, history, player, bucket, street=1):
        self.queries.append((tuple(board), tuple(history), player, bucket, street))
        return self._strategy


class _FakeLookup:
    def __init__(self, strategy, bucket=7):
        self._reader = _FakeReader(strategy)
        self._bucket_value = bucket
        self.delegated = 0
        self.extra = "passthrough"

    def _bucket(self, state, hero_cards):
        return self._bucket_value

    def get_mix(self, state, hero_cards):
        self.delegated += 1
        return "mix"


@pytest.fixture
def fake_key(monkeypatch):
    import app.engine.agent.strategy_source as strategy_source

    monkeypatch.setattr(
        strategy_source, "build_key", lambda state: Key((1, 2), 0, 1)
    )


def test_census_source_records_and_delegates(fake_key):
    inner = _FakeLookup([0.1, 0.9])
    source = census_mod.CensusSource(inner)
    state = SimpleNamespace(board=[1, 2, 3], active=[True, True, False])
    assert source.get_mix(state, (4, 5)) == "mix"
    assert inner.delegated == 1
    assert inner._reader.queries == [((1, 2, 3), (1, 2), 0, 7, 1)]
    assert source.counts == {(1, 2, "trained"): 1}
    assert source.extra == "passthrough"


def test_census_source_negative_bucket_counts_absent(fake_key):
    inner = _FakeLookup([0.1, 0.9], bucket=-1)
    source = census_mod.CensusSource(inner)
    state = SimpleNamespace(board=[1, 2, 3], active=[True, True])
    source.get_mix(state, (4, 5))
    assert inner._reader.queries == []
    assert source.counts == {(1, 2, "absent"): 1}
