"""Fixed-seed goldens: replay recorded matches and compare every decision.

Each tests/goldens/<name>.json holds a run_match command and the digest that
tools/golden_match.py recorded for it: per kind (blueprint lookup, search
decision, solve) the number of calls and the SHA-256 of all their rows. The
goldens were recorded on the original code this release was cut from and on
this release, and the two were identical (see each file's `recorded`).

The fixture_* goldens run on tests/fixtures/fixture.bps and need only files
in git and the built libraries. The published_* goldens run on the 8.05B
blueprint, and only when NEXUSGTO_GOLDEN_8B points at the published
unified_blueprint.bps (about 12 GB; they need about 23 GB of memory, and the
reader writes an 11 GB .cache beside the file on first use).

Like the fixture blueprint's hash, the digests hold for gcc 13.3 on x86-64
Linux; another compiler or libm can change float rounding in the solver.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GOLDENS = REPO / "tests" / "goldens"
FIXTURE = REPO / "tests" / "fixtures" / "fixture.bps"
ABSTRACTION = REPO / "data" / "abstraction"
ABSTRACTION_FILES = ("texture_cache.bin", "turn_centroids.bin")
LIBS = ("bps_reader", "es_resolver_real", "rollout", "hand_eval_batch", "street_solve_cpu")
KINDS = ("lookup", "search", "solve")


def _golden(name: str) -> dict:
    return json.loads((GOLDENS / f"{name}.json").read_text(encoding="utf-8"))


def _libs_built() -> bool:
    build = REPO / "solver" / "build"
    return sys.platform.startswith("linux") and all((build / f"{n}.so").is_file() for n in LIBS)


def _run(name: str, blueprint: Path, work: Path) -> dict:
    spec = _golden(name)
    out = work / name
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("NEXUS_", "BPS_", "ENGINE_", "BP_"))}
    env.pop("PYTHONPATH", None)
    env.update(spec["env"])
    env.update({"AGENT_BLUEPRINT_PATH": str(blueprint), "BPS_TMPDIR": str(work / "bpstmp"),
                "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"})
    (work / "bpstmp").mkdir(exist_ok=True)
    cmd = [sys.executable, str(REPO / "tools" / "golden_match.py"),
           str(REPO / "api"), str(REPO / "solver"), str(out), "--",
           "--blueprint-a", str(blueprint), "--blueprint-b", str(blueprint),
           "--out", str(out / "match"), *spec["args"]]
    proc = subprocess.run(cmd, cwd=REPO / "api", env=env, capture_output=True, text=True,
                          timeout=spec.get("timeout_s", 1800))
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads((out / "digest.json").read_text())


def _check(name: str, got: dict) -> None:
    want = _golden(name)["digest"]
    for kind in KINDS:
        assert got[kind]["count"] == want[kind]["count"], (name, kind, got[kind], want[kind])
        assert got[kind]["sha256"] == want[kind]["sha256"], (name, kind)


@pytest.fixture(scope="module")
def fixture_copy(tmp_path_factory) -> Path:
    """The fixture blueprint beside its abstraction files, in a temp dir."""
    if not _libs_built():
        pytest.skip("native libraries not built (make all), or not Linux")
    work = tmp_path_factory.mktemp("golden_fixture")
    for name in ABSTRACTION_FILES:
        shutil.copyfile(ABSTRACTION / name, work / name)
    shutil.copyfile(FIXTURE, work / FIXTURE.name)
    return work / FIXTURE.name


FIXTURE_GOLDENS = sorted(p.stem for p in GOLDENS.glob("fixture_*.json"))
PUBLISHED_GOLDENS = sorted(p.stem for p in GOLDENS.glob("published_*.json"))


def test_goldens_are_well_formed():
    assert FIXTURE_GOLDENS and PUBLISHED_GOLDENS
    for name in FIXTURE_GOLDENS + PUBLISHED_GOLDENS:
        spec = _golden(name)
        assert spec["name"] == name
        assert set(spec["digest"]) >= set(KINDS)
        assert "--live-threads" not in spec["args"] or \
            spec["args"][spec["args"].index("--live-threads") + 1] == "1"
        for path_like in spec["args"] + list(spec["env"].values()):
            assert not str(path_like).startswith("/"), (name, path_like)


@pytest.mark.parametrize("name", FIXTURE_GOLDENS)
def test_fixture_golden(name, fixture_copy, tmp_path):
    _check(name, _run(name, fixture_copy, tmp_path))


@pytest.mark.parametrize("name", PUBLISHED_GOLDENS)
def test_published_golden(name, tmp_path):
    path = os.environ.get("NEXUSGTO_GOLDEN_8B")
    if not path or not Path(path).is_file():
        pytest.skip("set NEXUSGTO_GOLDEN_8B to the 8.05B unified_blueprint.bps")
    if not _libs_built():
        pytest.skip("native libraries not built (make all), or not Linux")
    _check(name, _run(name, Path(path), tmp_path))
