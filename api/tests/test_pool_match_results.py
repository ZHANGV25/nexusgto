"""Tests for scripts/pool_match_results.py."""
from __future__ import annotations

import json

import pytest

from scripts import pool_match_results as pool_mod


def _write_run(path, seed, values, adjusted=None, deal_tag=0):
    path.mkdir()
    rows = [
        {
            "master_seed": seed,
            "group_index": i,
            "deal": {"board": [seed, i, deal_tag]},
            "hero_net_chips_per_hand": value,
            **(
                {"hero_net_chips_per_hand_adjusted": adjusted[i]}
                if adjusted is not None else {}
            ),
        }
        for i, value in enumerate(values)
    ]
    (path / "deal_groups.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_pool_concatenates_groups_across_runs(tmp_path):
    _write_run(tmp_path / "a", 1, [100.0, -100.0], adjusted=[50.0, -50.0])
    _write_run(tmp_path / "b", 2, [300.0, -100.0], adjusted=[150.0, 50.0])
    result = pool_mod.pool([tmp_path / "a", tmp_path / "b"])
    assert result["raw"]["n_groups"] == 4
    assert result["raw"]["mean_bb_per_100"] == pytest.approx(50.0)
    assert result["adjusted"]["mean_bb_per_100"] == pytest.approx(50.0)
    assert result["corrections"]["mean_bb_per_100"] == pytest.approx(0.0)


def test_pool_skips_adjusted_when_any_run_lacks_it(tmp_path):
    _write_run(tmp_path / "a", 1, [100.0, -100.0])
    result = pool_mod.pool([tmp_path / "a"])
    assert "adjusted" not in result


def test_paired_difference_uses_same_seed_groups(tmp_path):
    _write_run(tmp_path / "base", 5, [100.0, -100.0, 0.0], adjusted=[10.0, 20.0, 30.0])
    _write_run(tmp_path / "cand", 5, [200.0, 0.0], adjusted=[20.0, 40.0])
    result = pool_mod.paired([tmp_path / "cand"], [tmp_path / "base"])
    assert result["raw_difference"]["n_groups"] == 2
    assert result["raw_difference"]["mean_bb_per_100"] == pytest.approx(100.0)
    assert result["adjusted_difference"]["mean_bb_per_100"] == pytest.approx(15.0)


def test_paired_rejects_missing_pair_and_different_deals(tmp_path):
    _write_run(tmp_path / "base", 5, [1.0])
    _write_run(tmp_path / "cand", 5, [1.0, 2.0])
    with pytest.raises(ValueError, match="no baseline pair"):
        pool_mod.paired([tmp_path / "cand"], [tmp_path / "base"])
    _write_run(tmp_path / "other", 5, [1.0], deal_tag=9)
    with pytest.raises(ValueError, match="dealt differently"):
        pool_mod.paired([tmp_path / "other"], [tmp_path / "base"])


def test_pool_rejects_repeated_seeds(tmp_path):
    _write_run(tmp_path / "a", 7, [1.0, 2.0])
    _write_run(tmp_path / "b", 7, [3.0, 4.0])
    with pytest.raises(ValueError, match="repeats master seeds"):
        pool_mod.pool([tmp_path / "a", tmp_path / "b"])
