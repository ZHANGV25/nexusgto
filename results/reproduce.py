#!/usr/bin/env python3
"""Recompute the published numbers from the per-deal result files.

    python results/reproduce.py [--data DIR] [--only ID ...]

Reads numbers.yaml (next to this script) and, for every row marked
`reproducible: true`, recomputes the number from the release-asset files in
DIR (default: results/data/), checks each file against MANIFEST.sha256, and
compares the result with the published value at the precision it was
published. Prints the headline first. Then recomputes every all_arms.csv row
whose data survives. Exit code 1 if anything differs.

Standard library only. Statistics follow api/scripts/pool_match_results.py:
the deal group (one deal played in all six seat rotations) is the
independent unit; the standard error uses the n-1 sample variance; the 95%
interval is mean +/- 1.96 SE. Chips per hand equal bb/100 at a 100-chip
big blind.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADLINE = "all_streets_own_range_vs_blueprint"


# ── numbers.yaml (a small, fixed subset of YAML) ─────────────────────────────

class Num(float):
    """A float that remembers how many decimals it was written with."""

    decimals = 0


def _scalar(token: str):
    token = token.strip()
    if token.startswith('"'):
        return json.loads(token)
    if token in ("true", "false"):
        return token == "true"
    if token in ("null", "~", ""):
        return None
    if token.startswith("["):
        inner = token[1:-1].strip()
        return [_scalar(t) for t in inner.split(",")] if inner else []
    try:
        return int(token.replace("_", ""))
    except ValueError:
        pass
    try:
        num = Num(float(token))
    except ValueError:
        return token
    num.decimals = len(token.split(".")[1]) if "." in token else 0
    return num


def load_numbers(path: Path) -> list:
    """Parse numbers.yaml: a `rows:` list of flat mappings, comments on own lines."""
    rows, current = [], None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#") or line.strip() == "rows:":
            continue
        if line.startswith("  - "):
            current = {}
            rows.append(current)
            line = "    " + line[4:]
        if current is None or not line.startswith("    "):
            continue
        key, _, value = line.strip().partition(":")
        current[key] = _scalar(value)
    return rows


# ── data files ───────────────────────────────────────────────────────────────

class Data:
    def __init__(self, root: Path, manifest: Path):
        self.root = root
        self.manifest = {}
        if manifest.is_file():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                digest, name = line.split()
                self.manifest[name] = digest
        self._cache = {}

    def sha256(self, name: str) -> str:
        return hashlib.sha256((self.root / name).read_bytes()).hexdigest()

    def rows(self, name: str) -> list:
        if name not in self._cache:
            path = self.root / name
            if not path.is_file():
                raise FileNotFoundError(f"missing data file {path}")
            if self.manifest and self.manifest.get(name) != self.sha256(name):
                raise ValueError(f"{name} does not match MANIFEST.sha256")
            with gzip.open(path, "rt", encoding="utf-8") as f:
                self._cache[name] = [json.loads(line) for line in f if line.strip()]
        return self._cache[name]


def _seeds(spec) -> set | None:
    if spec in (None, "all"):
        return None
    lo, _, hi = str(spec).partition("-")
    return set(range(int(lo), int(hi or lo) + 1))


def _estimate(values) -> dict:
    n = len(values)
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return {"value": mean, "se": se, "ci95": [mean - 1.96 * se, mean + 1.96 * se], "n": n}


def _groups(data: Data, arm: str, field: str, seeds) -> dict:
    keep = _seeds(seeds)
    rows = data.rows(f"match.{arm}.{field}.deal_groups.jsonl.gz")
    return {(r["master_seed"], r["group_index"]): r for r in rows
            if keep is None or r["master_seed"] in keep}


def _metric(row: dict, metric: str) -> float:
    key = "hero_net_chips_per_hand" + ("_adjusted" if metric == "adjusted" else "")
    return float(row[key])


# ── computations, by `compute:` in numbers.yaml ──────────────────────────────

def c_pooled(data: Data, row: dict) -> dict:
    groups = _groups(data, row["arm"], row["field"], row.get("seeds"))
    return _estimate([_metric(g, row["metric"]) for g in groups.values()])


def _baseline_groups(data: Data, row: dict) -> dict:
    """The baseline's deals; `baseline` may list several arms that together cover the seeds."""
    arms = row["baseline"] if isinstance(row["baseline"], list) else [row["baseline"]]
    merged = {}
    for arm in arms:
        groups = _groups(data, arm, row.get("baseline_field", row["field"]), row.get("seeds"))
        repeated = merged.keys() & groups.keys()
        if repeated:
            raise ValueError(f"baseline arms repeat group {min(repeated)}")
        merged.update(groups)
    return merged


def c_paired(data: Data, row: dict) -> dict:
    cand = _groups(data, row["arm"], row["field"], row.get("seeds"))
    base = _baseline_groups(data, row)
    diffs = []
    for key, c in sorted(cand.items()):
        if key not in base:
            raise ValueError(f"group {key} has no baseline deal")
        b = base[key]
        if c["deal"] != b["deal"]:
            raise ValueError(f"group {key} was dealt differently in the two runs")
        diffs.append(_metric(c, row["metric"]) - _metric(b, row["metric"]))
    return _estimate(diffs)


def _lbr_units(data: Data, arm: str, metric: str, pairing: str = "deal") -> dict:
    """Exploiter result per deal (both seats averaged), or per seat arm if pairing == "seat"."""
    by_unit = defaultdict(list)
    for hand in data.rows(f"lbr.{arm}.hands.jsonl.gz"):
        key = (hand["seed"], hand["deal"])
        if pairing == "seat":
            key += (hand["p_seat"],)
        by_unit[key].append(float(hand[f"l_net_{metric}"]))
    return {k: math.fsum(v) / len(v) for k, v in by_unit.items()}


def c_lbr(data: Data, row: dict) -> dict:
    per_deal = _lbr_units(data, row["arm"], row["metric"])
    return _estimate(list(per_deal.values()))


def _lbr_cards(data: Data, arm: str) -> dict:
    return {(h["seed"], h["deal"], h["p_seat"]): (h["p_cards"], h["l_cards"], h["board"])
            for h in data.rows(f"lbr.{arm}.hands.jsonl.gz")}


def c_lbr_paired(data: Data, row: dict) -> dict:
    if _lbr_cards(data, row["arm"]) != _lbr_cards(data, row["baseline"]):
        raise ValueError(f"{row['arm']} and {row['baseline']} were not dealt the same cards")
    pairing = row.get("pairing", "deal")
    cand = _lbr_units(data, row["arm"], row["metric"], pairing)
    base = _lbr_units(data, row["baseline"], row["metric"], pairing)
    return _estimate([cand[k] - base[k] for k in sorted(cand)])


def _audit_bb100(scores: list, values_of, deals: int) -> dict:
    """Cell-weighted bb/100, as api/scripts/decision_audit_core.summarize_report."""
    total, var, n = 0.0, 0.0, 0
    for cell in sorted({s["cell"]["key"] for s in scores}):
        rows = [s for s in scores if s["cell"]["key"] == cell]
        vals = [values_of(s) for s in rows if values_of(s) is not None]
        if not vals:
            continue
        n_c = statistics.fmean(1.0 / float(s["inclusion_probability"]) for s in rows) * len(rows)
        scale = n_c / (6.0 * deals)
        mean = statistics.fmean(vals)
        se = statistics.stdev(vals) / math.sqrt(len(vals))
        total += mean * scale
        var += (se * scale) ** 2
        n += len(vals)
    se = math.sqrt(var)
    return {"value": total, "se": se, "ci95": [total - 1.96 * se, total + 1.96 * se], "n": n}


def _adv(score: dict, arm: str):
    entry = score["arms"].get(arm)
    return None if entry is None else float(entry["advantage_chips"]["mean"])


def c_audit(data: Data, row: dict) -> dict:
    deals = int(data.rows("decision_audit.meta.json.gz")[0]["deals"])
    scores = data.rows("decision_audit.scores.jsonl.gz")
    arm, base = row["arm"], row.get("baseline")
    if base is None:
        return _audit_bb100(scores, lambda s: _adv(s, arm), deals)
    both = [s for s in scores if arm in s["arms"] and base in s["arms"]]
    return _audit_bb100(both, lambda s: _adv(s, arm) - _adv(s, base), deals)


def _summaries(data: Data, row: dict) -> list:
    keep = _seeds(row.get("seeds"))
    rows = data.rows(f"match.{row['arm']}.{row['field']}.summaries.jsonl.gz")
    return [s for s in rows if keep is None or s["master_seed"] in keep]


def c_solves(data: Data, row: dict) -> dict:
    sums = _summaries(data, row)
    hero = [s["solve_wall_s_by_source"]["hero"] for s in sums]
    solves = sum(h["count"] for h in hero)
    fallbacks = sum(sum((s.get("production_live_fallbacks") or {}).values()) for s in sums)
    own = [s.get("hero_own_reach_stats") or {} for s in sums]
    stat = row["stat"]
    values = {
        "solves": solves,
        "mean_s": math.fsum(h["mean"] * h["count"] for h in hero) / solves,
        "p50_s_median_over_jobs": statistics.median(h["p50"] for h in hero),
        "p95_s_median_over_jobs": statistics.median(h["p95"] for h in hero),
        "max_s": max(h["max"] for h in hero),
        "fallbacks": fallbacks,
        "fallback_pct": 100.0 * fallbacks / solves,
        "own_range_overrides": sum(o.get("overrides_set", 0) for o in own),
        "own_range_unapplied": sum(o.get("overrides_unapplied", 0) for o in own),
    }
    return {"value": values[stat], "n": len(sums)}


def c_audit_meta(data: Data, row: dict) -> dict:
    meta = data.rows("decision_audit.meta.json.gz")[0]
    return {"value": meta[row["stat"]]}


def c_audit_iterations(data: Data, row: dict) -> dict:
    """Median CFR iterations per solve for one arm in one street / player-count cell."""
    street, players = row["street"], int(row["players"])
    iters = [log["solve_stats"]["iterations_run"] for log in data.rows("decision_audit.solve_logs.jsonl.gz")
             if log["arm"] == row["arm"] and log["cell"]["street"] == street
             and log["cell"]["entry_players"] == players
             and not log.get("fallback") and (log.get("solve_stats") or {}).get("iterations_run")]
    return {"value": statistics.median(iters), "n": len(iters)}


def _fallback_cause(spot: dict) -> str:
    """Classify a recorded search fallback by its error message."""
    message = spot["message"]
    if "exactly uniform" in message:
        all_in = all(stack == 0 for stack in spot["stacks"])
        return "all_in_forced_check" if all_in else "uniform_strategy_other"
    if "tree actions but the agent's legal set has" in message:
        return "tree_lacks_legal_raise"
    if "raise-capped tree" in message:
        return "past_raise_cap"
    if "strategy-sum mass" in message:
        return "low_mass"
    return "other"


def c_fallback_causes(data: Data, row: dict) -> dict:
    spots = data.rows(f"match.{row['arm']}.{row['field']}.fallback_spots.jsonl.gz")
    return {"value": sum(1 for s in spots if _fallback_cause(s) == row["cause"])}


COMPUTE = {
    "pooled": c_pooled, "paired": c_paired, "lbr": c_lbr, "lbr_paired": c_lbr_paired,
    "audit": c_audit, "solves": c_solves, "audit_meta": c_audit_meta,
    "audit_iterations": c_audit_iterations, "fallback_causes": c_fallback_causes,
}


# ── comparison and output ────────────────────────────────────────────────────

def _close(got: float, want) -> bool:
    """Equal at the precision the published value was written with."""
    if want is None:
        return True
    tolerance = 0.5 * 10 ** (-getattr(want, "decimals", 0)) + 1e-9
    return abs(got - float(want)) <= tolerance


CORRECTABLE = ("value", "se", "ci95")


def expected(row: dict, key: str):
    """The figure the data must match: `corrected_<key>` when present, else `<key>`."""
    return row.get(f"corrected_{key}", row.get(key))


def corrections(row: dict) -> list:
    return [f"published {key} {row.get(key)}; the data gives {row[f'corrected_{key}']}"
            for key in CORRECTABLE if f"corrected_{key}" in row]


def check_row(row: dict, result: dict) -> list:
    """Compare a result with its row, at the precision each figure is written with."""
    problems = []
    want = expected(row, "value")
    if not _close(result["value"], want):
        problems.append(f"value {result['value']:.4f} != {want}")
    se = expected(row, "se")
    if "se" in result and not _close(result["se"], se):
        problems.append(f"se {result['se']:.4f} != {se}")
    ci = expected(row, "ci95")
    if ci and "ci95" in result:
        if not (_close(result["ci95"][0], ci[0]) and _close(result["ci95"][1], ci[1])):
            problems.append(f"ci95 [{result['ci95'][0]:.2f}, {result['ci95'][1]:.2f}] != {ci}")
    if row.get("deals") is not None and result.get("n") not in (None, row["deals"]):
        problems.append(f"deals {result['n']} != {row['deals']}")
    return problems


def check_files(data: Data, row: dict) -> list:
    problems = []
    for name, digest in zip(row.get("data") or [], row.get("sha256") or []):
        if not (data.root / name).is_file():
            problems.append(f"missing {name}")
        elif data.sha256(name) != digest:
            problems.append(f"{name} sha256 differs from numbers.yaml")
    return problems


def fmt(row: dict, result: dict) -> str:
    want = row.get("value")
    places = want.decimals if isinstance(want, Num) else 0
    value = result["value"]
    if isinstance(value, float) and not places and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        text = f"{value:,}"
    else:
        text = f"{value:{'+' if 'se' in result else ''}.{places}f}"
    if "se" in result:
        text += f" ± {result['se']:.{places}f}"
    if "ci95" in result and row.get("ci95"):
        ci_places = max(getattr(bound, "decimals", 2) for bound in row["ci95"])
        low, high = result["ci95"]
        text += f" (95% CI {low:.{ci_places}f} to {high:.{ci_places}f})"
    if result.get("n") is not None and row.get("deals") is not None:
        text += f", {result['n']:,} {row.get('unit_n', 'deals')}"
    return text


def check_arms(data: Data, path: Path) -> int:
    """Recompute each all_arms.csv row with surviving data; return the number that differ."""
    failures = 0
    with path.open(newline="", encoding="utf-8") as f:
        arms = [r for r in csv.DictReader(f) if r["data_available"] in ("yes", "partial")]
    print(f"\nall_arms.csv: {len(arms)} rows with data")
    for arm in arms:
        spec = {"arm": arm["data_file"].split(".")[1], "field": "blueprint",
                "metric": "adjusted", "seeds": arm["seeds"]}
        result = _estimate([_metric(g, "adjusted") for g in _groups(data, **{
            k: spec[k] for k in ("arm", "field", "seeds")}).values()])
        want = {"value": _scalar(arm["data_value"]), "se": _scalar(arm["data_se"]),
                "deals": int(arm["data_deals"])}
        problems = check_row(want, result)
        failures += bool(problems)
        print(f"{'OK ' if not problems else 'DIFF'} {arm['arm_id']} ({arm['seeds']}, {arm['data_available']}): "
              f"{result['value']:+.2f} ± {result['se']:.2f}, {result['n']:,} deals")
        for problem in problems:
            print(f"     {problem}")
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--numbers", type=Path, default=HERE / "numbers.yaml")
    ap.add_argument("--manifest", type=Path, default=HERE / "MANIFEST.sha256")
    ap.add_argument("--arms", type=Path, default=HERE / "all_arms.csv")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
        sys.stdout.reconfigure(encoding="utf-8")  # for the ± sign when output is piped

    data = Data(args.data, args.manifest)
    rows = [r for r in load_numbers(args.numbers) if args.only is None or r["id"] in args.only]
    rows.sort(key=lambda r: r["id"] != HEADLINE)
    failures = 0
    for row in rows:
        if not row.get("reproducible"):
            continue
        result = COMPUTE[row["compute"]](data, row)
        problems = check_files(data, row) + check_row(row, result)
        status = "OK " if not problems else "DIFF"
        if row["id"] == HEADLINE:
            print(f"Headline: {row['fact']}")
            print(f"  {result['n']:,} deals, {result['value']:+.2f} ± {result['se']:.2f} {row['unit']} "
                  f"(95% CI {result['ci95'][0]:.2f} to {result['ci95'][1]:.2f})"
                  f"  [{'matches' if not problems else 'DIFFERS FROM'} numbers.yaml]")
            print()
        print(f"{status} {row['id']}: {fmt(row, result)} {row.get('unit', '')}".rstrip())
        for note in corrections(row):
            print(f"     {note}")
        if row.get("published"):
            print(f'     published as "{row["published"]}"')
        for problem in problems:
            print(f"     {problem}")
        failures += bool(problems)
    skipped = sum(1 for r in rows if not r.get("reproducible"))
    print(f"\n{len(rows) - skipped - failures} reproduced, {failures} differ, "
          f"{skipped} not reproducible (data lost; see numbers.yaml)")
    if args.only is None and args.arms.is_file():
        arm_failures = check_arms(data, args.arms)
        print(f"all_arms.csv: {arm_failures} differ")
        failures += arm_failures
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
