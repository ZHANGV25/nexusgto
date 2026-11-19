"""Convergence ladder report: strategy distance, run and pair summaries, audit solve stats."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import ladder_agreement as ladder
from scripts import root_value_audit as audit


def test_strategy_distance_is_reach_weighted_total_variation():
    a = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 0.0]])
    b = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    # Hand 0 differs fully (TV 1), hand 1 agrees, hand 2 is empty in table a and skipped.
    assert ladder.strategy_distance(a, b, np.array([1.0, 3.0, 5.0])) == pytest.approx(0.25)
    assert ladder.strategy_distance(a, a, np.array([1.0, 1.0, 1.0])) == 0.0


def test_strategy_distance_handles_no_live_hands_and_shape_errors():
    a = np.array([[1.0, 0.0]])
    assert ladder.strategy_distance(a, a, np.array([0.0])) is None
    with pytest.raises(ValueError):
        ladder.strategy_distance(a, np.array([[1.0, 0.0, 0.0]]), np.array([1.0]))


def _write_run(path, values, tables, iterations):
    path.mkdir()
    rows = [
        {"deal": i + 1, "board": [0, 1, 2], "pot": 300.0, "hero_seat": 1,
         "solve_stats": {"iterations_run": iterations, "solve_s": 3.0, "num_nodes": 100},
         "true": {"search_minus_blueprint_pct": v, "bet_freq_search": 0.3, "bet_freq_blueprint": 0.25}}
        for i, v in enumerate(values)
    ]
    (path / "spots.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    np.savez_compressed(path / "tables.npz", **tables)


def test_run_and_pair_reports(tmp_path):
    bet, check = np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]])
    reach = np.array([1.0])
    common = {"blueprint_0": check, "true_hero_0": reach, "blueprint_1": check, "true_hero_1": reach}
    _write_run(tmp_path / "a", [1.0, 3.0], {**common, "search_0": bet, "search_1": check}, 1000)
    _write_run(tmp_path / "b", [0.0, 1.0], {**common, "search_0": check, "search_1": check}, 4000)
    run_a, run_b = ladder.load_run(tmp_path / "a"), ladder.load_run(tmp_path / "b")
    report = ladder.run_report(run_a)
    assert report["iterations_run"]["mean"] == 1000
    assert report["search_minus_blueprint_pct"]["mean"] == pytest.approx(2.0)
    assert report["distance_to_blueprint"]["mean"] == pytest.approx(0.5)
    pair = ladder.pair_report(run_a, run_b)
    assert pair["matched_spots"] == 2
    assert pair["strategy_distance"]["mean"] == pytest.approx(0.5)
    assert pair["value_a_minus_b_pct"]["mean"] == pytest.approx(1.5)


def test_main_writes_report(tmp_path, capsys):
    tables = {"search_0": np.array([[1.0, 0.0]]), "blueprint_0": np.array([[1.0, 0.0]]), "true_hero_0": np.array([1.0])}
    _write_run(tmp_path / "a", [1.0], tables, 10)
    _write_run(tmp_path / "b", [2.0], tables, 20)
    out = tmp_path / "report.json"
    assert ladder.main([str(tmp_path / "a"), str(tmp_path / "b"), "--pairs", "a:b", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["pairs"]["a:b"]["value_a_minus_b_pct"]["mean"] == pytest.approx(-1.0)
    assert "runs" in json.loads(capsys.readouterr().out)


def test_last_solve_stats_reads_the_engine_stash():
    engine = SimpleNamespace(_last_solve_stats={"iterations_run": 12000, "solve_s": 3.0, "num_nodes": 20000, "other": 1})
    assert audit.last_solve_stats(SimpleNamespace(_engine=engine)) == {"iterations_run": 12000, "solve_s": 3.0, "num_nodes": 20000}
    assert audit.last_solve_stats(SimpleNamespace()) == {}
