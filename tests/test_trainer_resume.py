"""Trainer resume: a checkpoint is kept unless it loaded, and a bad manifest stops the run.

Runs the trainer (solver/precompute/blueprint_worker_unified.py) for a few
hundred iterations on one thread with the fixture's settings
(configs/fixture.json), then resumes from what it wrote. A resume needs the
regret checkpoint to load and checkpoint_meta.json to be readable: the
manifest's iteration count keeps the discount and pruning schedule in step,
and its config_hash is checked against the run's settings.

Linux only; needs the trainer library (make all). Each run takes a few seconds.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CONFIG = json.loads((REPO / "configs" / "fixture.json").read_text())
BUILD = REPO / CONFIG["build_dir"]
TRAINER = REPO / CONFIG["trainer"]
FIRST_ITERATIONS = 200
CHECKPOINT = "regrets_latest.bin"
MANIFEST = "checkpoint_meta.json"

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the trainer runs on Linux"),
    pytest.mark.skipif(not (BUILD / "mccfr_blueprint.so").is_file(),
                       reason="trainer not built (make all)"),
]


def _train(workdir: Path, iterations: int, *extra: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BPS_", "BP_", "NEXUS_"))}
    env.update(CONFIG["env"])
    env["BP_CACHE_DIR"] = str(workdir)
    cmd = [sys.executable, str(TRAINER),
           "--iterations", str(iterations),
           "--num-threads", str(CONFIG["num_threads"]),
           "--hash-size", str(CONFIG["hash_size"]),
           "--build-dir", str(BUILD),
           "--output-dir", str(workdir), *CONFIG["args"], *extra]
    return subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True,
                          timeout=600)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Path:
    """A directory holding a 200-iteration checkpoint and its manifest."""
    work = tmp_path_factory.mktemp("trained")
    for name in CONFIG["abstraction_files"]:
        shutil.copyfile(REPO / CONFIG["abstraction_dir"] / name, work / name)
    proc = _train(work, FIRST_ITERATIONS)
    assert proc.returncode == 0, _output(proc)[-3000:]
    assert (work / CHECKPOINT).is_file()
    assert json.loads((work / MANIFEST).read_text())["iterations"] == FIRST_ITERATIONS
    return work


@pytest.fixture
def run(trained, tmp_path) -> Path:
    """A private copy of the trained directory."""
    work = tmp_path / "run"
    shutil.copytree(trained, work)
    return work


def test_resume_continues_the_iteration_count(run):
    proc = _train(run, 2 * FIRST_ITERATIONS, "--resume")
    assert proc.returncode == 0, _output(proc)[-3000:]
    assert f"{FIRST_ITERATIONS:,} iterations already done" in proc.stdout
    assert json.loads((run / MANIFEST).read_text())["iterations"] == 2 * FIRST_ITERATIONS


def test_resume_of_a_finished_run_keeps_its_checkpoint(run):
    before = _digest(run / CHECKPOINT)
    proc = _train(run, FIRST_ITERATIONS, "--resume")
    assert proc.returncode == 0, _output(proc)[-3000:]
    assert "Already completed" in proc.stdout
    assert _digest(run / CHECKPOINT) == before


def test_unreadable_manifest_stops_and_keeps_the_checkpoint(run):
    before = _digest(run / CHECKPOINT)
    (run / MANIFEST).write_text('{"iterations": ')
    proc = _train(run, 2 * FIRST_ITERATIONS, "--resume")
    assert proc.returncode == 2, _output(proc)[-3000:]
    assert f"cannot read {run / MANIFEST}" in proc.stdout
    assert _digest(run / CHECKPOINT) == before


def test_missing_manifest_stops_and_keeps_the_checkpoint(run):
    before = _digest(run / CHECKPOINT)
    (run / MANIFEST).unlink()
    proc = _train(run, 2 * FIRST_ITERATIONS, "--resume")
    assert proc.returncode == 2, _output(proc)[-3000:]
    assert "Refusing to resume" in proc.stdout
    assert _digest(run / CHECKPOINT) == before


def test_manifest_without_an_iteration_count_stops(run):
    meta = json.loads((run / MANIFEST).read_text())
    (run / MANIFEST).write_text(json.dumps({k: v for k, v in meta.items() if k != "iterations"}))
    proc = _train(run, 2 * FIRST_ITERATIONS, "--resume")
    assert proc.returncode == 2, _output(proc)[-3000:]
    assert "no valid 'iterations' count" in proc.stdout
    assert (run / CHECKPOINT).is_file()


def test_config_hash_mismatch_stops_and_keeps_the_checkpoint(run):
    before = _digest(run / CHECKPOINT)
    meta = json.loads((run / MANIFEST).read_text())
    (run / MANIFEST).write_text(json.dumps({**meta, "config_hash": "0" * 64}))
    proc = _train(run, 2 * FIRST_ITERATIONS, "--resume")
    assert proc.returncode == 2, _output(proc)[-3000:]
    assert "Config hash mismatch" in proc.stdout
    assert _digest(run / CHECKPOINT) == before


def test_resume_from_a_file_that_does_not_load_keeps_it(run):
    bad = run / "not_a_checkpoint.bin"
    bad.write_bytes(b"this is not a regret checkpoint")
    proc = _train(run, 2 * FIRST_ITERATIONS, "--resume", "--resume-from", bad.name)
    assert proc.returncode == 2, _output(proc)[-3000:]
    assert "did not load" in proc.stdout
    assert bad.read_bytes() == b"this is not a regret checkpoint"
    assert (run / CHECKPOINT).is_file()
