#!/usr/bin/env python3
"""Train the tiny test blueprint (tests/fixtures/fixture.bps).

Runs the public trainer for a few seconds on one thread with the settings in
configs/fixture.json: the published blueprint's game and abstraction, a small
hash table and 1,000 iterations. The run is deterministic, so the same build
gives the same bytes. The manifest gets the overrides listed in the config:
the wall-clock fields (time_seconds, time_hours) become 0 so the file can be
hash-checked, and subsequent_bet_sizes is added because the trainer's final
export, unlike its per-chunk snapshot export, does not stamp it.

Usage (Linux, after `make -C solver mccfr_blueprint`):
    python tools/make_fixture.py            # write tests/fixtures/fixture.bps
    python tools/make_fixture.py --check    # regenerate and compare, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "fixture.json"
BPS_MAGIC = b"BPS3"
HEADER = struct.Struct("<QI")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[name.strip().lstrip("*")] = digest
    return sums


def check_abstraction(cfg: dict) -> dict[str, str]:
    """Verify the shipped abstraction files; return name -> sha256."""
    adir = REPO / cfg["abstraction_dir"]
    sums = read_sums(adir / "SHA256SUMS")
    expected = {}
    for name in cfg["abstraction_files"]:
        got = sha256_file(adir / name)
        if sums.get(name) != got:
            raise SystemExit(f"{adir / name}: sha256 {got} does not match SHA256SUMS")
        expected[name] = got
    return expected


def check_trainer_caches(workdir: Path, expected: dict[str, str]) -> list[str]:
    """Compare the caches the trainer computed in workdir with the recorded hashes."""
    notes = []
    for name, digest in expected.items():
        path = workdir / name
        if not path.is_file():
            notes.append(f"the trainer did not write {name}")
        elif sha256_file(path) != digest:
            notes.append(f"{name} differs from configs/fixture.json (toolchain?)")
    return notes


def run_trainer(cfg: dict, workdir: Path, build_dir: Path, python: str) -> Path:
    for name in cfg["abstraction_files"]:
        shutil.copyfile(REPO / cfg["abstraction_dir"] / name, workdir / name)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BPS_", "NEXUS_"))}
    env.update(cfg["env"])
    # The trainer reads and writes its abstraction caches here, and only here.
    env["BP_CACHE_DIR"] = str(workdir)
    cmd = [python, str(REPO / cfg["trainer"]),
           "--iterations", str(cfg["iterations"]),
           "--num-threads", str(cfg["num_threads"]),
           "--hash-size", str(cfg["hash_size"]),
           "--build-dir", str(build_dir),
           "--output-dir", str(workdir), *cfg["args"]]
    log = workdir / "train.log"
    with open(log, "w") as out:
        rc = subprocess.run(cmd, cwd=workdir, env=env, stdout=out,
                            stderr=subprocess.STDOUT).returncode
    bps = workdir / "unified_blueprint.bps"
    if rc != 0 or not bps.is_file():
        tail = log.read_text(errors="replace").splitlines()[-20:]
        raise SystemExit(f"trainer failed (rc={rc}); last lines of {log}:\n" + "\n".join(tail))
    return bps


def split_bps(data: bytes) -> tuple[bytes, dict, bytes]:
    if data[:4] != BPS_MAGIC:
        raise ValueError("not a BPS3 file")
    clen, mlen = HEADER.unpack_from(data, 4)
    start = 4 + HEADER.size
    payload = data[start:start + clen]
    meta = json.loads(data[start + clen:start + clen + mlen])
    return payload, meta, data[start + clen + mlen:]


def normalize(data: bytes, cfg: dict) -> bytes:
    """Apply the config's manifest overrides and check the config hash."""
    payload, meta, tail = split_bps(data)
    if meta.get("config_hash") != cfg["expected_config_hash"]:
        raise SystemExit(f"config_hash {meta.get('config_hash')} != "
                         f"{cfg['expected_config_hash']}; the settings drifted")
    overrides = cfg["manifest_overrides"]
    fixed = {**meta, **overrides}
    meta_bytes = json.dumps(fixed, separators=(",", ":")).encode("utf-8")
    return BPS_MAGIC + HEADER.pack(len(payload), len(meta_bytes)) + payload + meta_bytes + tail


def build(cfg: dict, build_dir: Path, python: str, keep: bool) -> bytes:
    check_abstraction(cfg)
    workdir = Path(tempfile.mkdtemp(prefix="fixture-"))
    try:
        bps = run_trainer(cfg, workdir, build_dir, python)
        for note in check_trainer_caches(workdir, cfg["trainer_cache_files"]):
            print(f"note: {note}")
        return normalize(bps.read_bytes(), cfg)
    finally:
        if keep:
            print(f"kept {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare with the committed fixture; write nothing")
    ap.add_argument("--build-dir", default=None, help="trainer library dir (default solver/build)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=None, help="output path (default from the config)")
    ap.add_argument("--keep-workdir", action="store_true")
    args = ap.parse_args(argv)

    cfg = json.loads(CONFIG.read_text())
    build_dir = Path(args.build_dir or REPO / cfg["build_dir"]).resolve()
    out = Path(args.out or REPO / cfg["output"])
    data = build(cfg, build_dir, args.python, args.keep_workdir)
    digest = hashlib.sha256(data).hexdigest()
    print(f"fixture: {len(data):,} bytes, sha256 {digest}")
    if args.check:
        committed = sha256_file(out) if out.is_file() else None
        ok = digest == committed == cfg["expected_sha256"]
        print("matches the committed fixture" if ok else
              f"MISMATCH: committed {committed}, config {cfg['expected_sha256']}")
        return 0 if ok else 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    if digest != cfg["expected_sha256"]:
        print(f"note: configs/fixture.json expected_sha256 is {cfg['expected_sha256'] or 'empty'}; "
              "update it if this change is intended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
