"""data/training/: the 8.05B run's checkpoint manifests are intact and consistent."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAINING = REPO / "data" / "training"
CONFIG_HASH = json.loads((REPO / "configs" / "fixture.json").read_text())["expected_config_hash"]


def _sums() -> dict[str, str]:
    sums = {}
    for line in (TRAINING / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        sums[name] = digest
    return sums


def _iterations(name: str) -> int:
    m = re.fullmatch(r"meta_(\d+)([MB])\.json", name)
    return int(m.group(1)) * (10**6 if m.group(2) == "M" else 10**9)


def test_files_match_sha256sums():
    sums = _sums()
    on_disk = {p.name for p in TRAINING.glob("*.json")}
    assert on_disk == set(sums)
    assert len(sums) == 81
    for name, digest in sums.items():
        assert hashlib.sha256((TRAINING / name).read_bytes()).hexdigest() == digest, name


def test_manifests_are_one_run():
    metas = sorted((_iterations(p.name), json.loads(p.read_text()))
                   for p in TRAINING.glob("meta_*.json"))
    assert [it for it, _ in metas] == [100_000_000 * k for k in range(1, 81)]
    previous = 0
    for it, meta in metas:
        assert meta["iterations"] == it
        assert meta["config_hash"] == CONFIG_HASH
        assert meta["hash_insertion_failures"] == 0
        assert meta["num_info_sets"] > previous
        previous = meta["num_info_sets"]
    final = json.loads((TRAINING / "checkpoint_meta.json").read_text())
    assert final == metas[-1][1]
