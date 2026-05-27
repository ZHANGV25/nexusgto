"""Regression test: the preflop tier config in blueprint_worker_unified.py
and extract_preflop_json.py must stay in lockstep. A mismatch causes
35k+ info-set divergence between training and JSON extraction, silently
scrambling the entire frontend grid.

This test fails loud on drift.
"""

import os
import sys
import importlib.util

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_PY = os.path.join(REPO, "precompute", "blueprint_worker_unified.py")
EXTRACT_PY = os.path.join(REPO, "precompute", "extract_preflop_json.py")


def _load(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_preflop_tiers_match():
    """Training worker and JSON extractor must share PREFLOP_TIERS / MAX_RAISES."""
    worker = _load(WORKER_PY, "blueprint_worker_unified")
    extract = _load(EXTRACT_PY, "extract_preflop_json")

    w_tiers = worker.PREFLOP_TIERS
    e_tiers = extract.PREFLOP_TIERS

    assert set(w_tiers.keys()) == set(e_tiers.keys()), (
        f"Tier-level keys differ: worker={sorted(w_tiers.keys())} "
        f"extract={sorted(e_tiers.keys())}"
    )

    for level in sorted(w_tiers.keys()):
        w_sizes = list(w_tiers[level])
        e_sizes = list(e_tiers[level])
        assert w_sizes == e_sizes, (
            f"Tier {level} sizes differ: worker={w_sizes} extract={e_sizes}. "
            "A mismatch here silently scrambles the frontend strategy grid."
        )

    assert worker.PREFLOP_MAX_RAISES == extract.PREFLOP_MAX_RAISES, (
        f"PREFLOP_MAX_RAISES differs: worker={worker.PREFLOP_MAX_RAISES} "
        f"extract={extract.PREFLOP_MAX_RAISES}"
    )


def test_preflop_max_raises_is_three():
    """Phase 2c pruning (Step 18, 2026-05-12): MAX_RAISES=3 — open / 3-bet /
    4-bet, with all-in auto-added at the 4-bet level. The earlier 4-tier setup
    with an explicit 5-bet shove (worker.PREFLOP_TIERS[3] = [8.0]) was dropped
    when Step 17 evidence showed only 0.0003% of action_hashes had all-6
    player coverage and the action set needed pruning."""
    worker = _load(WORKER_PY, "blueprint_worker_unified")
    assert worker.PREFLOP_MAX_RAISES == 3


def test_tiers_cover_open_3bet_4bet():
    """Three tiers must exist (open=0, 3bet=1, 4bet=2). The 5-bet level
    is reachable but collapses to all-in via the auto-added all-in action."""
    worker = _load(WORKER_PY, "blueprint_worker_unified")
    assert sorted(worker.PREFLOP_TIERS.keys()) == [0, 1, 2]


def test_sizes_are_monotonically_nonincreasing_in_count():
    """Fewer sizes at deeper raise levels — matches Pluribus-style tiering."""
    worker = _load(WORKER_PY, "blueprint_worker_unified")
    counts = [len(worker.PREFLOP_TIERS[l]) for l in [0, 1, 2]]
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1], (
            f"Tier sizes must be monotonically non-increasing in count: "
            f"got {counts}"
        )


if __name__ == "__main__":
    test_preflop_tiers_match()
    test_preflop_max_raises_is_three()
    test_tiers_cover_open_3bet_4bet()
    test_sizes_are_monotonically_nonincreasing_in_count()
    print("OK: preflop tier parity tests passed")
