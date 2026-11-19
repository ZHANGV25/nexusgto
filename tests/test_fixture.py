"""The tiny test blueprint and the shipped abstraction files.

tests/fixtures/fixture.bps is made by tools/make_fixture.py from
configs/fixture.json. These tests check that the committed bytes are the
recorded ones, that the file is a well-formed BPS3 blueprint of the published
game, and (on Linux with the trainer built) that the script makes the same
bytes again.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CONFIG = json.loads((REPO / "configs" / "fixture.json").read_text())
FIXTURE = REPO / CONFIG["output"]
ABSTRACTION = REPO / CONFIG["abstraction_dir"]
TRAINER_LIB = REPO / CONFIG["build_dir"] / "mccfr_blueprint.so"
PUBLISHED_TIERS = {"0": [0.7, 1.0], "1": [0.7, 1.0], "2": [1.0]}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _manifest() -> dict:
    data = FIXTURE.read_bytes()
    clen, mlen = struct.unpack_from("<QI", data, 4)
    return json.loads(data[16 + clen:16 + clen + mlen])


def test_abstraction_files_match_their_sums():
    for line in (ABSTRACTION / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((ABSTRACTION / name).read_bytes()).hexdigest() == digest, name


def test_fixture_is_the_recorded_bytes():
    assert FIXTURE.stat().st_size < 1_000_000
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == CONFIG["expected_sha256"]


def test_fixture_manifest_is_the_published_game():
    meta = _manifest()
    assert meta["config_hash"] == CONFIG["expected_config_hash"]
    assert meta["preflop_tiers"] == PUBLISHED_TIERS
    assert meta["preflop_max_raises"] == 3
    assert meta["postflop_bet_sizes"] == [0.75]
    assert meta["add_explicit_allin"] == 0
    assert meta["hash_mixer"] == "splitmix64"
    assert meta["iterations"] == CONFIG["iterations"]
    assert meta["hash_insertion_failures"] == 0
    assert (meta["time_seconds"], meta["time_hours"]) == (0, 0.0)
    assert meta["subsequent_bet_sizes"] == [1.0]


def test_fixture_payload_parses_and_has_the_preflop_root():
    ex = _load(REPO / "solver" / "precompute" / "extract_preflop_json.py", "extract_preflop_json")
    by_node, _evs, meta = ex.load_bps3_preflop_direct(str(FIXTURE))
    ex._HASH_MIXER = meta["hash_mixer"]
    rules = ex.preflop_rules_from_manifest(meta)
    walk = ex.compute_nodes_by_tree_walk(rules, max_depth=2)
    root = walk[ex.HASH_SEED]
    rows = by_node[(ex.HASH_SEED, root["player"])]
    assert len(rows) > 100  # most of the 169 hand classes were dealt to UTG
    assert all(len(r) == len(root["labels"]) for r in rows.values())
    missing = [h for h, node in walk.items() if (h, node["player"]) not in by_node]
    assert not missing  # every node up to two actions deep was trained


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the trainer runs on Linux")
@pytest.mark.skipif(not TRAINER_LIB.is_file(), reason="trainer not built (make -C solver mccfr_blueprint)")
def test_make_fixture_reproduces_the_committed_bytes():
    mk = _load(REPO / "tools" / "make_fixture.py", "make_fixture")
    assert mk.main(["--check"]) == 0
