"""Shared test fixtures.

Puts the solver's Python packages on sys.path and exposes the blueprint
fixtures. Blueprint-oracle tests SKIP unless AGENT_BLUEPRINT_PATH points at a
blueprint file, so the pure-unit tests run on a fresh checkout.

`fixture_blueprint_path` is the tiny trained blueprint in tests/fixtures/
(see tools/make_fixture.py). Tests that only need a real, loadable blueprint
of the published game use it; tests that check the 8.05B's own strategy
values use `blueprint_path`.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_BPS = _REPO_ROOT / "tests" / "fixtures" / "fixture.bps"
_ABSTRACTION_DIR = _REPO_ROOT / "data" / "abstraction"
_ABSTRACTION_FILES = ("texture_cache.bin", "turn_centroids.bin")
_POKER_SOLVER_PATH = os.environ.get("POKER_SOLVER_PATH", str(_REPO_ROOT / "solver"))
os.environ.setdefault("POKER_SOLVER_PATH", _POKER_SOLVER_PATH)
_PS_PYTHON = Path(_POKER_SOLVER_PATH) / "python"
if _PS_PYTHON.is_dir():
    for p in (str(_POKER_SOLVER_PATH), str(_PS_PYTHON)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve_blueprint_path() -> Optional[str]:
    p = os.environ.get("AGENT_BLUEPRINT_PATH")
    if p and Path(p).exists():
        return p
    return None


@pytest.fixture(scope="session")
def blueprint_path() -> str:
    path = _resolve_blueprint_path()
    if path is None:
        pytest.skip("blueprint not found (set AGENT_BLUEPRINT_PATH); skipping blueprint-oracle tests.")
    return path


def _reader_library_built() -> bool:
    ext = ".dll" if sys.platform == "win32" else ".so"
    build = Path(_POKER_SOLVER_PATH) / "build"
    return any((build / f"{name}{ext}").is_file() for name in ("bps_ev_walk", "bps_reader"))


@pytest.fixture(scope="session")
def fixture_blueprint_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The tiny fixture blueprint, next to copies of its abstraction files.

    The reader looks for texture_cache.bin and turn_centroids.bin beside the
    blueprint, and would otherwise rebuild the turn centroids (minutes) and
    write them into tests/fixtures/. A temp dir keeps the checkout clean.
    """
    if not _reader_library_built():
        pytest.skip("bps_reader library not built (make -C solver bps_reader)")
    if not _FIXTURE_BPS.is_file():
        pytest.skip(f"fixture blueprint missing: {_FIXTURE_BPS}")
    work = tmp_path_factory.mktemp("fixture_blueprint")
    for name in _ABSTRACTION_FILES:
        shutil.copyfile(_ABSTRACTION_DIR / name, work / name)
    target = work / _FIXTURE_BPS.name
    shutil.copyfile(_FIXTURE_BPS, target)
    return str(target)


@pytest.fixture(scope="session")
def blueprint(blueprint_path: str):
    """Session-scoped BPSReaderC on the blueprint."""
    from engine.bps_reader_py import BPSReaderC  # type: ignore[import-untyped]

    return BPSReaderC(blueprint_path)
