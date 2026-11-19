#!/usr/bin/env python3
"""Run one match and record every blueprint lookup, search decision and solve.

The records are what the fixed-seed goldens in tests/goldens/ are made of. The
script takes the api/ and solver/ directories to run, so the same match can be
recorded on two checkouts and the records compared byte for byte.

    python tools/golden_match.py <api_dir> <solver_dir> <out_dir> -- <run_match args...>

Writes <out_dir>/records.jsonl.gz (one JSON row per event, floats exact) and
<out_dir>/digest.json (per kind: the count and the SHA-256 of the rows).

Kinds:
  lookup  every LookupSource.get_mix call: info-set key, board, hole cards, the
          returned action frequencies and whether it was the uniform fallback
  search  every LiveResolveSource.get_mix call (a search decision): the same
          fields, with the frequencies the search played
  solve   every CpuEsSolver.solve call: SHA-256 of its inputs and of its output,
          the number of players, the board, the iteration ceiling and the
          iterations run

The digests are exact only for a deterministic run: one solve thread
(OMP_NUM_THREADS=1, --live-threads 1), a fixed iteration ceiling that the solves
reach before the deadline, and PYTHONHASHSEED fixed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys

KINDS = ("lookup", "search", "solve")
# Solve arguments that are callbacks or output buffers, not inputs.
SOLVE_SKIP = {"progress_callback", "belief_stash_out"}


def canon(obj):
    """JSON-ready copy with sorted keys and numpy values as Python numbers."""
    if isinstance(obj, dict):
        return {str(k): canon(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [canon(v) for v in obj]
    if hasattr(obj, "tolist"):
        return canon(obj.tolist())
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    return repr(obj)


def dumps(obj) -> str:
    return json.dumps(canon(obj), sort_keys=True, separators=(",", ":"))


def digest(rows: list) -> dict:
    out = {}
    for kind in KINDS:
        sel = [dumps(r) for r in rows if r["kind"] == kind]
        out[kind] = {"count": len(sel),
                     "sha256": hashlib.sha256("\n".join(sel).encode()).hexdigest()}
    return out


def install_recorders(rows: list) -> None:
    """Wrap the three entry points so each call appends a row."""
    from app.engine.agent import strategy_source as ss
    from app.engine.agent.key_builder import build_key
    from app.engine import cpu_es_solver as ces

    def state_id(state, hero_cards):
        return {"key": list(build_key(state).as_tuple()), "board": list(state.board),
                "hero": list(hero_cards)}

    orig_lookup = ss.LookupSource.get_mix
    orig_live = ss.LiveResolveSource.get_mix
    orig_solve = ces.CpuEsSolver.solve

    def lookup(self, state, hero_cards):
        mix = orig_lookup(self, state, hero_cards)
        rows.append({"kind": "lookup", **state_id(state, hero_cards),
                     "freqs": sorted(mix.freqs.items()),
                     "fallback": bool(mix.is_uniform_fallback)})
        return mix

    def live(self, state, hero_cards):
        mix = orig_live(self, state, hero_cards)
        rows.append({"kind": "search", **state_id(state, hero_cards),
                     "freqs": sorted(mix.freqs.items())})
        return mix

    def solve(self, **kwargs):
        out = orig_solve(self, **kwargs)
        stats = getattr(self, "_last_solve_stats", None) or {}
        inputs = {k: v for k, v in kwargs.items() if k not in SOLVE_SKIP}
        rows.append({
            "kind": "solve",
            "inputs_sha256": hashlib.sha256(dumps(inputs).encode()).hexdigest(),
            "output_sha256": hashlib.sha256(dumps(out).encode()).hexdigest(),
            "players": len(kwargs.get("player_ranges") or ()),
            "board": list(kwargs.get("board_ints") or ()),
            "iterations": kwargs.get("iterations"),
            "iterations_run": stats.get("iterations_run"),
        })
        return out

    ss.LookupSource.get_mix = lookup
    ss.LiveResolveSource.get_mix = live
    ces.CpuEsSolver.solve = solve


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[3] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    api_dir, solver_dir, out_dir = (os.path.abspath(a) for a in argv[:3])
    for p in (solver_dir, os.path.join(solver_dir, "python"), api_dir):
        sys.path.insert(0, p)
    os.environ["POKER_SOLVER_PATH"] = solver_dir
    os.makedirs(out_dir, exist_ok=True)

    rows: list = []
    install_recorders(rows)
    from scripts import run_match

    rc = run_match.main(argv[4:])

    with gzip.open(os.path.join(out_dir, "records.jsonl.gz"), "wt") as f:
        for r in rows:
            f.write(dumps(r) + "\n")
    result = {"rc": rc, **digest(rows)}
    with open(os.path.join(out_dir, "digest.json"), "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print("GOLDEN_DIGEST", json.dumps(result, sort_keys=True))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
