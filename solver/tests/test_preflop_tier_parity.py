"""The preflop tree the extractor walks must be the tree that trained.

The trainer (blueprint_worker_unified.py) sets tiered preflop sizes and stamps
them into every manifest it writes. The JSON extractor reads them back from the
manifest instead of keeping its own copy: a hard-coded copy can drift (for
example to an extra 0.5-pot open), which would build a different tree and
scramble the extracted grid.

The expected values are the published blueprint's manifest: tiers
{0: [0.7, 1.0], 1: [0.7, 1.0], 2: [1.0]}, at most 3 raises, no explicit all-in.
Its preflop table has 16,306 nodes, and the walk below must enumerate exactly
that many with those settings.
"""

import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_PY = os.path.join(REPO, "precompute", "blueprint_worker_unified.py")
EXTRACT_PY = os.path.join(REPO, "precompute", "extract_preflop_json.py")

TRAINED_TIERS = {0: [0.7, 1.0], 1: [0.7, 1.0], 2: [1.0]}
TRAINED_MAX_RAISES = 3
PUBLISHED_PREFLOP_NODES = 16_306


def _load(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _worker_manifest(worker, add_explicit_allin=0):
    """The preflop fields exactly as the worker's exporters stamp them."""
    return {
        "preflop_tiers": {str(k): v for k, v in sorted(worker.PREFLOP_TIERS.items())},
        "preflop_max_raises": worker.PREFLOP_MAX_RAISES,
        "add_explicit_allin": add_explicit_allin,
    }


def test_worker_tiers_are_the_trained_config():
    worker = _load(WORKER_PY, "blueprint_worker_unified")
    assert worker.PREFLOP_TIERS == TRAINED_TIERS
    assert worker.PREFLOP_MAX_RAISES == TRAINED_MAX_RAISES


def test_extractor_has_no_hardcoded_tiers():
    extract = _load(EXTRACT_PY, "extract_preflop_json")
    assert not hasattr(extract, "PREFLOP_TIERS")
    assert not hasattr(extract, "PREFLOP_MAX_RAISES")


def test_extractor_reads_the_worker_manifest():
    worker = _load(WORKER_PY, "blueprint_worker_unified")
    extract = _load(EXTRACT_PY, "extract_preflop_json")
    rules = extract.preflop_rules_from_manifest(_worker_manifest(worker))
    assert rules.tiers == worker.PREFLOP_TIERS
    assert rules.max_raises == worker.PREFLOP_MAX_RAISES
    assert rules.add_explicit_allin is False
    assert rules.tiers_json() == _worker_manifest(worker)["preflop_tiers"]


def test_manifest_without_tiers_is_rejected():
    extract = _load(EXTRACT_PY, "extract_preflop_json")
    with pytest.raises(ValueError):
        extract.preflop_rules_from_manifest({"iterations": 1})
    with pytest.raises(ValueError):
        extract.preflop_rules_from_manifest({"preflop_tiers": {"0": [1.0]}})
    with pytest.raises(ValueError):
        extract.preflop_rules_from_manifest(
            {"preflop_tiers": {"1": [1.0]}, "preflop_max_raises": 3})


def test_missing_allin_flag_means_explicit_allin():
    extract = _load(EXTRACT_PY, "extract_preflop_json")
    rules = extract.preflop_rules_from_manifest(
        {"preflop_tiers": {"0": [1.0]}, "preflop_max_raises": 2})
    assert rules.add_explicit_allin is True


def test_trained_tree_matches_the_published_table():
    worker = _load(WORKER_PY, "blueprint_worker_unified")
    extract = _load(EXTRACT_PY, "extract_preflop_json")
    extract._HASH_MIXER = "splitmix64"
    rules = extract.preflop_rules_from_manifest(_worker_manifest(worker))
    nodes = extract.compute_nodes_by_tree_walk(rules, max_depth=8)
    assert len(nodes) == PUBLISHED_PREFLOP_NODES
    root = nodes[extract.HASH_SEED]
    assert root["player"] == extract.POS_TO_PLAYER["UTG"]
    assert root["labels"] == ["fold", "call", "raise_2.8", "raise_3.5"]


def test_explicit_allin_adds_a_root_action():
    worker = _load(WORKER_PY, "blueprint_worker_unified")
    extract = _load(EXTRACT_PY, "extract_preflop_json")
    extract._HASH_MIXER = "splitmix64"
    rules = extract.preflop_rules_from_manifest(
        _worker_manifest(worker, add_explicit_allin=1))
    nodes = extract.compute_nodes_by_tree_walk(rules, max_depth=2)
    assert nodes[extract.HASH_SEED]["labels"] == [
        "fold", "call", "raise_2.8", "raise_3.5", "allin"]
