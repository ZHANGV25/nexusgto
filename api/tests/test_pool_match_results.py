"""Tests for scripts/pool_match_results.py."""
from __future__ import annotations

import json

import pytest

from scripts import pool_match_results as pool_mod


def _write_run(path, seed, values, adjusted=None):
    path.mkdir()
    rows = [
        {
            "master_seed": seed,
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


def test_pool_rejects_repeated_seeds(tmp_path):
    _write_run(tmp_path / "a", 7, [1.0, 2.0])
    _write_run(tmp_path / "b", 7, [3.0, 4.0])
    with pytest.raises(ValueError, match="repeats master seeds"):
        pool_mod.pool([tmp_path / "a", tmp_path / "b"])
