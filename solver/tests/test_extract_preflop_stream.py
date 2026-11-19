"""The preflop extractor's streaming BPS3 parser.

load_bps3_preflop_direct decompresses and parses the strategy payload in
bounded chunks so a multi-GB blueprint fits in memory. These tests build a
small BPS3 file by hand and check the streamed parse against a direct parse,
with chunk sizes small enough that records straddle chunk boundaries.
"""

import importlib.util
import json
import lzma
import os
import random
import struct
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_PY = os.path.join(REPO, "precompute", "extract_preflop_json.py")


def _load_extractor():
    spec = importlib.util.spec_from_file_location("extract_preflop_json", EXTRACT_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extract_preflop_json"] = mod
    spec.loader.exec_module(mod)
    return mod


def _records(seed=7, n=400):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        street = rng.choice([0, 0, 1, 2, 3])
        na = rng.randint(1, 5)
        out.append((rng.randrange(6), street, rng.randrange(200),
                    rng.getrandbits(64), rng.choice([11, 22, 33, rng.getrandbits(64)]),
                    bytes(rng.randrange(256) for _ in range(na))))
    return out


def _payload(records):
    body = [b"BPS3", struct.pack("<II", len(records), 6)]
    for player, street, bucket, board, action, strat in records:
        body.append(struct.pack("<BBHQQB", player, street, bucket, board, action, len(strat)))
        body.append(strat)
    return b"".join(body)


def _expected(records):
    by_node = {}
    for player, street, bucket, _board, action, strat in records:
        if street == 0:
            by_node.setdefault((action, player), {})[bucket] = strat
    return by_node


def _as_bytes(by_node):
    return {k: {b: bytes(v) for b, v in d.items()} for k, d in by_node.items()}


@pytest.mark.parametrize("chunk", [1, 7, 64, 1 << 20])
def test_stream_parse_matches_direct_parse(tmp_path, chunk):
    ex = _load_extractor()
    records = _records()
    compressed = lzma.compress(_payload(records), preset=1)
    path = tmp_path / "x.bin"
    path.write_bytes(compressed)
    with open(path, "rb") as f:
        by_node, n_entries, kept, skipped, raw = ex._scan_preflop_stream(
            ex._iter_lzma_chunks(f, len(compressed), chunk=chunk))
    assert n_entries == len(records)
    assert kept == sum(1 for r in records if r[1] == 0)
    assert kept + skipped == len(records)
    assert raw == len(_payload(records))
    assert _as_bytes(by_node) == _expected(records)


def test_load_bps3_reads_manifest_and_preflop_rows(tmp_path):
    ex = _load_extractor()
    records = _records(seed=11)
    compressed = lzma.compress(_payload(records), preset=1)
    meta = {"iterations": 5, "num_info_sets": len(records), "hash_mixer": "splitmix64",
            "preflop_tiers": {"0": [0.7, 1.0], "1": [0.7, 1.0], "2": [1.0]},
            "preflop_max_raises": 3, "add_explicit_allin": 0}
    meta_bytes = json.dumps(meta).encode()
    path = tmp_path / "tiny.bps"
    path.write_bytes(b"BPS3" + struct.pack("<QI", len(compressed), len(meta_bytes))
                     + compressed + meta_bytes)
    by_node, by_node_evs, got_meta = ex.load_bps3_preflop_direct(str(path))
    assert got_meta == meta
    assert by_node_evs == {}
    assert _as_bytes(by_node) == _expected(records)
    for rows in by_node.values():
        for row in rows.values():
            assert row.dtype == np.uint8


def test_truncated_payload_is_an_error(tmp_path):
    ex = _load_extractor()
    records = _records(seed=3)
    payload = _payload(records)
    compressed = lzma.compress(payload[:-5], preset=1)
    path = tmp_path / "short.bin"
    path.write_bytes(compressed)
    with open(path, "rb") as f, pytest.raises(ValueError):
        ex._scan_preflop_stream(ex._iter_lzma_chunks(f, len(compressed), chunk=64))
