"""Opponent model check: value the flop subgame under mixed strategy profiles.

usage (published solve environment exported):
  python scripts/opponent_model_check.py --spots 60 --out DIR --production-env NEXUS_LEAF_VEC_SAMPLED=

For each heads-up flop root the hero is re-solved once with the eager leaf
table, and the blueprint's flop strategy is projected onto the same tree.
The hero's range-weighted root value is then computed on the same leaf table
under four profiles: search vs search (what the solver believes), search vs
blueprint (what the search's flop play is worth against the field's actual
flop play), blueprint vs blueprint, and blueprint vs search.  The gap between
the first two is the solver's opponent-model error; the gap between the
second and third is the search's in-subgame edge over the blueprint against
the blueprint field, leaf values held fixed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lbr_core import mean_se  # noqa: E402
from scripts.lbr_game import deal_from_rng  # noqa: E402
from scripts.root_value_audit import BUCKETS, blueprint_play_to_root, bucket_of, root_action_values, row_equities  # noqa: E402

PROFILES = ("search_vs_search", "search_vs_blueprint", "blueprint_vs_blueprint", "blueprint_vs_search")


def project_blueprint(tree: Any, root: Any, lookup: Any) -> Dict[int, np.ndarray]:
    """Blueprint strategy per decision node and hand row of a flop or turn tree.

    Unlike river_brgap_capture.project_blueprint_sigma, street-end leaves map
    to non-terminal states (the next street) and are simply not expanded.
    """
    from scripts.river_brgap_capture import _row_mix

    sigma: Dict[int, np.ndarray] = {}
    pending = [(0, root)]
    while pending:
        node_idx, state = pending.pop()
        node = tree.nodes[node_idx]
        if node.type != 0:
            continue
        if state.terminal or state.to_act_slot is None:
            raise ValueError(f"decision node {node_idx} reached a terminal state")
        n_legal = int(state.num_legal_actions())
        if n_legal != len(node.children):
            raise ValueError(f"node {node_idx} has {len(node.children)} children, state has {n_legal}")
        hands = tree.player_hands[node.player]
        rows = np.full((len(hands), n_legal), 1.0 / n_legal)
        for row, (c0, c1, weight) in enumerate(hands):
            if weight <= 0.0:
                continue
            rows[row], _missed = _row_mix(lookup, state, (int(c0), int(c1)), n_legal)
        sigma[node_idx] = rows
        pending.extend((child, state.advance(action)) for action, child in enumerate(node.children))
    return sigma


def mixed_profile(tree: Any, hero: int, hero_sigma: Dict[int, np.ndarray], opp_sigma: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Decision nodes owned by the hero from ``hero_sigma``, the rest from ``opp_sigma``."""
    profile: Dict[int, np.ndarray] = {}
    for node_idx, node in enumerate(tree.nodes):
        if node.type != 0:
            continue
        source = hero_sigma if node.player == hero else opp_sigma
        sigma = source.get(node_idx)
        if sigma is not None:
            profile[node_idx] = np.asarray(sigma, dtype=np.float64)
    return profile


def strategy_value(tree: Any, table: np.ndarray, profile: Dict[int, np.ndarray], hero: int) -> np.ndarray:
    """Per hero row: the root strategy's value under ``profile`` (achieved, not best response)."""
    values = root_action_values(tree, table, profile, hero)
    root_sigma = profile.get(0)
    if root_sigma is None:
        root_sigma = np.full(values.shape, 1.0 / values.shape[1])
    return (root_sigma * values).sum(axis=1)


def audit_spot(spot: Dict[str, Any], hero_cards: Any, policy: Any, lookup: Any, evaluate: Any,
               rng: np.random.Generator, equity_runouts: int) -> Dict[str, Any]:
    from scripts.root_value_audit import assumed_reach

    root = spot["state"]
    hero = int(root.to_act_slot)
    board = list(root.board)[:3]
    policy.table(root, hero_cards)
    objs = getattr(policy, "last_objs", None)
    if objs is None or objs.get("table") is None:
        raise RuntimeError("no fresh eager leaf table for the root")
    tree, table = objs["tree"], objs["table"]
    search_sigma = {int(k): np.asarray(v, dtype=np.float64) for k, v in objs["sigma"].items()}
    blueprint_sigma = project_blueprint(tree, root, lookup)
    projection = {"nodes": len(blueprint_sigma)}
    profiles = {
        "search_vs_search": mixed_profile(tree, hero, search_sigma, search_sigma),
        "search_vs_blueprint": mixed_profile(tree, hero, search_sigma, blueprint_sigma),
        "blueprint_vs_blueprint": mixed_profile(tree, hero, blueprint_sigma, blueprint_sigma),
        "blueprint_vs_search": mixed_profile(tree, hero, blueprint_sigma, search_sigma),
    }
    weights = np.array([float(w) for _c0, _c1, w in tree.player_hands[hero]])
    weights = weights / weights.sum()
    pot = float(root.pot)
    equities = row_equities(tree, hero, assumed_reach(tree, 1 - hero, board), board, rng, evaluate, equity_runouts)
    buckets = np.array([bucket_of(e) for e in equities])
    result: Dict[str, Any] = {
        "pot": pot, "hero_seat": int(root.seat_of_slot[hero]), "board": board, "rows": len(weights),
        "projection": projection, "values_pct": {}, "by_bucket": {name: {} for name in BUCKETS},
        "root_bet_freq": {
            "search": float((weights * (1.0 - search_sigma[0][:, 0])).sum()) if 0 in search_sigma else None,
            "blueprint": float((weights * (1.0 - blueprint_sigma[0][:, 0])).sum()) if 0 in blueprint_sigma else None,
        },
    }
    for name, profile in profiles.items():
        per_row = strategy_value(tree, table, profile, hero)
        result["values_pct"][name] = 100.0 * float((weights * per_row).sum()) / pot
        for bucket in BUCKETS:
            mask = buckets == bucket
            if weights[mask].sum() > 0:
                result["by_bucket"][bucket][name] = 100.0 * float((weights[mask] * per_row[mask]).sum() / weights[mask].sum()) / pot
    return result


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"spots": len(rows), "values_pct": {}, "differences_pct": {}, "by_bucket": {}}
    for name in PROFILES:
        out["values_pct"][name] = mean_se([r["values_pct"][name] for r in rows])
    pairs = {
        "opponent_model_gap (search_vs_search - search_vs_blueprint)": ("search_vs_search", "search_vs_blueprint"),
        "search_edge_vs_field (search_vs_blueprint - blueprint_vs_blueprint)": ("search_vs_blueprint", "blueprint_vs_blueprint"),
        "field_edge_vs_search_opponent (blueprint_vs_search - blueprint_vs_blueprint)": ("blueprint_vs_search", "blueprint_vs_blueprint"),
    }
    for label, (a, b) in pairs.items():
        out["differences_pct"][label] = mean_se([r["values_pct"][a] - r["values_pct"][b] for r in rows])
    for bucket in BUCKETS:
        out["by_bucket"][bucket] = {
            name: mean_se([r["by_bucket"][bucket][name] for r in rows if name in r["by_bucket"][bucket]]) for name in PROFILES
        }
    out["root_bet_freq"] = {
        key: mean_se([r["root_bet_freq"][key] for r in rows if r["root_bet_freq"][key] is not None]) for key in ("search", "blueprint")
    }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", type=int, default=60)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--equity-runouts", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blueprint", default=None)
    ap.add_argument("--live-threads", type=int, default=8)
    ap.add_argument("--production-env", action="append", default=[])
    args = ap.parse_args(argv)
    blueprint = args.blueprint or os.environ.get("AGENT_BLUEPRINT_PATH")
    if not blueprint:
        raise SystemExit("set --blueprint or AGENT_BLUEPRINT_PATH")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from scripts.lbr_policies import build_policy
    from scripts.lbr_run import _env_pairs
    from scripts.match_aivat import evaluate_7card_batch

    policy = build_policy("production", blueprint, np.random.default_rng(args.seed + 1), live_threads=args.live_threads,
                          env_overrides=_env_pairs(args.production_env))
    lookup = policy._lookup
    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    with (out_dir / "spots.jsonl").open("w", encoding="utf-8") as fh:
        while len(rows) < int(args.spots):
            deal = deal_from_rng(rng)
            spot = blueprint_play_to_root(deal, lookup, rng, 1)
            if spot is None:
                continue
            try:
                row = audit_spot(spot, deal.hole_cards[spot["state"].to_act_seat()], policy, lookup, evaluate_7card_batch, rng, int(args.equity_runouts))
            except Exception as exc:
                print(f"[opp] spot failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            rows.append(row)
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            v = row["values_pct"]
            print(
                f"[opp] {len(rows)}/{args.spots} pot={row['pot']:.0f} SS={v['search_vs_search']:+.1f} SB={v['search_vs_blueprint']:+.1f} "
                f"BB={v['blueprint_vs_blueprint']:+.1f} BS={v['blueprint_vs_search']:+.1f} bet s/bp {row['root_bet_freq']['search']:.2f}/{row['root_bet_freq']['blueprint']:.2f}",
                flush=True,
            )
    summary = summarize(rows)
    summary["wall_s"] = time.perf_counter() - started
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
