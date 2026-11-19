"""results/: numbers.yaml, all_arms.csv, MANIFEST.sha256 and reproduce.py agree.

The data files are a release asset, not in git. Without them these tests check
the tables against each other and the script's parser and statistics on small
inputs; with results/data/ present they also run the headline check.
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DATA = RESULTS / "data"


def _load_reproduce():
    spec = importlib.util.spec_from_file_location("reproduce", RESULTS / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reproduce"] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load_reproduce()
ROWS = R.load_numbers(RESULTS / "numbers.yaml")
MANIFEST = dict(reversed(line.split()) for line in
                (RESULTS / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines())


def test_scalar_keeps_published_precision():
    assert R._scalar("5.73").decimals == 2
    assert R._scalar("-2.30").decimals == 2
    assert R._scalar("1.3").decimals == 1
    assert R._scalar("14250") == 14250
    assert R._scalar("[2.74, 8.72]") == [2.74, 8.72]
    assert R._scalar('"a: b"') == "a: b"
    assert R._close(5.7308, R._scalar("5.73"))
    assert not R._close(5.7360, R._scalar("5.73"))


def test_headline_row_is_the_published_one():
    head = next(r for r in ROWS if r["id"] == R.HEADLINE)
    assert head["reproducible"] is True
    assert (head["value"], head["se"], head["deals"]) == (5.73, 1.53, 14250)
    assert head["ci95"] == [2.74, 8.72]
    assert head["metric"] == "adjusted"


def test_every_row_is_complete():
    ids = [r["id"] for r in ROWS]
    assert len(ids) == len(set(ids))
    for row in ROWS:
        assert row.get("fact") and row.get("value") is not None, row["id"]
        if row["reproducible"]:
            assert row["compute"] in R.COMPUTE, row["id"]
            assert len(row["data"]) == len(row["sha256"]) >= 1, row["id"]
            for name, digest in zip(row["data"], row["sha256"]):
                assert MANIFEST[name] == digest, (row["id"], name)
        else:
            assert row.get("reason"), row["id"]
            assert "data" not in row, row["id"]


def test_corrections_carry_a_note():
    for row in ROWS:
        if any(f"corrected_{k}" in row for k in R.CORRECTABLE):
            assert row.get("note"), row["id"]


def test_all_arms_table_matches_the_manifest():
    with (RESULTS / "all_arms.csv").open(newline="", encoding="utf-8") as f:
        arms = list(csv.DictReader(f))
    assert {a["data_available"] for a in arms} <= {"yes", "partial", "no"}
    for arm in arms:
        if arm["data_available"] == "no":
            assert not arm["data_file"], arm["arm_id"]
        else:
            assert arm["data_file"] in MANIFEST, arm["arm_id"]
            assert int(arm["data_deals"]) <= int(arm["deals"]), arm["arm_id"]


def _write_groups(path: Path, values: dict) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for (seed, group), (adjusted, deal) in values.items():
            f.write(json.dumps({"master_seed": seed, "group_index": group, "deal": deal,
                                "hero_net_chips_per_hand": adjusted,
                                "hero_net_chips_per_hand_adjusted": adjusted}) + "\n")


def test_pooled_and_paired_statistics(tmp_path):
    _write_groups(tmp_path / "match.a.blueprint.deal_groups.jsonl.gz",
                  {(1, 0): (1.0, {"d": 0}), (1, 1): (3.0, {"d": 1}), (2, 0): (5.0, {"d": 2})})
    _write_groups(tmp_path / "match.b.blueprint.deal_groups.jsonl.gz",
                  {(1, 0): (0.0, {"d": 0}), (1, 1): (1.0, {"d": 1}), (2, 0): (2.0, {"d": 2})})
    data = R.Data(tmp_path, tmp_path / "none.sha256")
    pooled = R.c_pooled(data, {"arm": "a", "field": "blueprint", "metric": "adjusted"})
    assert pooled["value"] == pytest.approx(3.0)
    assert pooled["se"] == pytest.approx(2.0 / math.sqrt(3))
    assert pooled["n"] == 3
    one_seed = R.c_pooled(data, {"arm": "a", "field": "blueprint", "metric": "raw", "seeds": "1"})
    assert one_seed["n"] == 2
    paired = R.c_paired(data, {"arm": "a", "baseline": "b", "field": "blueprint", "metric": "adjusted"})
    assert paired["value"] == pytest.approx(2.0)


def test_paired_refuses_different_deals(tmp_path):
    _write_groups(tmp_path / "match.a.blueprint.deal_groups.jsonl.gz", {(1, 0): (1.0, {"d": 0}),
                                                                        (1, 1): (2.0, {"d": 1})})
    _write_groups(tmp_path / "match.b.blueprint.deal_groups.jsonl.gz", {(1, 0): (1.0, {"d": 9}),
                                                                        (1, 1): (2.0, {"d": 1})})
    data = R.Data(tmp_path, tmp_path / "none.sha256")
    with pytest.raises(ValueError, match="dealt differently"):
        R.c_paired(data, {"arm": "a", "baseline": "b", "field": "blueprint", "metric": "raw"})


def test_paired_baseline_split_over_two_arms(tmp_path):
    _write_groups(tmp_path / "match.a.nit.deal_groups.jsonl.gz",
                  {(1, 0): (4.0, {"d": 0}), (2, 0): (6.0, {"d": 1})})
    _write_groups(tmp_path / "match.b.nit.deal_groups.jsonl.gz", {(2, 0): (1.0, {"d": 1})})
    _write_groups(tmp_path / "match.b_rerun.nit.deal_groups.jsonl.gz", {(1, 0): (1.0, {"d": 0})})
    data = R.Data(tmp_path, tmp_path / "none.sha256")
    row = {"arm": "a", "baseline": ["b", "b_rerun"], "field": "nit", "metric": "adjusted"}
    assert R.c_paired(data, row)["value"] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="no baseline deal"):
        R.c_paired(data, {**row, "baseline": ["b"]})
    _write_groups(tmp_path / "match.b_dup.nit.deal_groups.jsonl.gz", {(2, 0): (1.0, {"d": 1})})
    with pytest.raises(ValueError, match="repeat group"):
        R.c_paired(data, {**row, "baseline": ["b", "b_rerun", "b_dup"]})


def test_list_baseline_parses():
    row = next(r for r in ROWS if r["id"] == "own_range_vs_blueprint_nit")
    assert row["baseline"] == ["blueprint", "blueprint_rerun"]


@pytest.mark.skipif(not (DATA / "match.all_streets_own_range.blueprint.deal_groups.jsonl.gz").is_file(),
                    reason="results/data/ (the release asset) is not unpacked")
def test_headline_reproduces_from_the_data(capsys):
    assert R.main(["--only", R.HEADLINE]) == 0
    out = capsys.readouterr().out
    assert "14,250 deals, +5.73 ± 1.53 bb/100" in out
