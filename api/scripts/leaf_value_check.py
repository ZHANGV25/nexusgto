"""Leaf value check: the oracle's value at every flop-tree leaf versus blueprint
rollouts from that leaf's game state.

usage (published solve environment exported):
  python scripts/leaf_value_check.py --spots 60 --out DIR --production-env NEXUS_LEAF_VEC_SAMPLED=

For each heads-up flop root the hero is re-solved once with the eager leaf
table.  Every street-end leaf of the solved tree is mapped to the game state
after that betting line, and its table value per hero row (net chips from the
flop root, the oracle's convention) is compared with rollouts from that state
in which both players play the blueprint.  Opponent hands are drawn (a) from
the root range, as the oracle does, and (b) from the root range conditioned
on the blueprint's probability of the opponent's actions along the line.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lbr_core import COMBOS, N_COMBOS, live_mask, mean_se, sample_runouts  # noqa: E402
from scripts.lbr_game import action_kinds, deal_from_rng  # noqa: E402
from scripts.lbr_policies import lookup_table  # noqa: E402
from scripts.root_value_audit import (  # noqa: E402
    BUCKETS, assumed_reach, blueprint_play_to_root, bucket_of, play_out, row_equities, row_of_combo, sample_pair,
)

KIND_LETTER = {"fold": "f", "call": "c", "raise": "b"}


def leaf_states(tree: Any, root: Any) -> List[Dict[str, Any]]:
    """Every street-end leaf with the game state after its line and the opponent's decisions on it."""
    from es_resolver.es_solver import T_DECISION, T_LEAF

    out = []
    pending = [(0, root, "", [])]
    while pending:
        node_idx, state, line, decisions = pending.pop()
        node = tree.nodes[node_idx]
        if node.type == T_LEAF:
            out.append({"node": node_idx, "leaf_idx": int(node.leaf_idx), "state": state, "line": line, "decisions": decisions})
            continue
        if node.type != T_DECISION:
            continue
        kinds = action_kinds(state)
        for action, child in enumerate(node.children):
            step = KIND_LETTER[kinds[action]] if action < len(kinds) else "?"
            pending.append((child, state.advance(action), line + step, decisions + [(state, int(node.player), action)]))
    return out


def conditioned_opponent(root_weights: np.ndarray, decisions: Sequence[Tuple[Any, int, int]], opp_slot: int, lookup: Any) -> np.ndarray:
    weights = root_weights.copy()
    for state, player, action in decisions:
        if player == opp_slot:
            weights = weights * lookup_table(lookup, state)[:, action]
    return weights


def check_leaf(
    leaf: Dict[str, Any], root: Any, hero_slot: int, hero_w: np.ndarray, opp_root: np.ndarray, opp_cond: np.ndarray,
    table: np.ndarray, rows_of: Dict[int, int], equities: np.ndarray, samples: int, lookup: Any,
    rng: np.random.Generator, evaluate: Any,
) -> Dict[str, Any]:
    state = leaf["state"]
    visible = list(root.board)[:3]
    root_stack = int(root.stacks[hero_slot])
    pot = float(root.pot)
    records: Dict[str, List[Tuple[int, float, float, float]]] = {name: [] for name in BUCKETS}
    for _ in range(samples):
        pair = sample_pair(rng, hero_w, opp_root, visible)
        if pair is None:
            break
        hero, _opp_root = pair
        row = rows_of.get(hero)
        if row is None:
            continue
        opp_p = np.where(live_mask(visible + list(COMBOS[hero])), opp_cond, 0.0)
        opp_cond_hand = int(rng.choice(N_COMBOS, p=opp_p / opp_p.sum())) if opp_p.sum() > 0 else _opp_root
        values = []
        for opp in (_opp_root, opp_cond_hand):
            dead = visible + list(COMBOS[hero]) + list(COMBOS[opp])
            runout = sample_runouts(rng, dead, 2, 1)[0].tolist()
            rolled = replace(state, board=tuple(visible + runout))
            cards = [COMBOS[hero] if slot == hero_slot else COMBOS[opp] for slot in range(2)]
            award, final_stack = play_out(rolled, hero_slot, cards, lookup, np.random.default_rng(int(rng.integers(0, 2 ** 31 - 1))), evaluate)
            values.append(award - (root_stack - final_stack))
        oracle = float(table[leaf["leaf_idx"], hero_slot, row])
        records[bucket_of(equities[row])].append((row, oracle, values[0], values[1]))
    out: Dict[str, Any] = {"line": leaf["line"], "leaf_idx": leaf["leaf_idx"], "pot_at_leaf": int(state.pot), "by_bucket": {}}
    everything = [r for name in BUCKETS for r in records[name]]
    for name, group in list(records.items()) + [("all", everything)]:
        if not group:
            continue
        arr = np.array([(o, a, b) for _row, o, a, b in group])
        out["by_bucket"][name] = {
            "n": len(group), "oracle_pct": 100.0 * arr[:, 0].mean() / pot,
            "rollout_root_pct": 100.0 * arr[:, 1].mean() / pot, "rollout_cond_pct": 100.0 * arr[:, 2].mean() / pot,
        }
    return out


def audit_spot(spot: Dict[str, Any], hero_cards: Tuple[int, int], policy: Any, lookup: Any, evaluate: Any,
               rng: np.random.Generator, samples: int, equity_runouts: int) -> Dict[str, Any]:
    root = spot["state"]
    hero_slot = int(root.to_act_slot)
    opp_slot = 1 - hero_slot
    board = list(root.board)[:3]
    policy.table(root, hero_cards)
    objs = getattr(policy, "last_objs", None)
    if objs is None or objs.get("table") is None:
        raise RuntimeError("no fresh eager leaf table for the root")
    tree, table = objs["tree"], objs["table"]
    hero_w, opp_root = assumed_reach(tree, hero_slot, board), assumed_reach(tree, opp_slot, board)
    rows_of = row_of_combo(tree, hero_slot, board)
    equities = row_equities(tree, hero_slot, opp_root, board, rng, evaluate, equity_runouts)
    leaves = []
    for leaf in leaf_states(tree, root):
        opp_cond = conditioned_opponent(opp_root, leaf["decisions"], opp_slot, lookup)
        leaves.append(check_leaf(leaf, root, hero_slot, hero_w, opp_root, opp_cond, table, rows_of, equities, samples, lookup, rng, evaluate))
    return {
        "pot": float(root.pot), "hero_seat": int(root.seat_of_slot[hero_slot]), "board": board,
        "preflop_prefix": list(root.preflop_prefix), "leaves": leaves, "table_source": objs.get("table_source"),
    }


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lines: Dict[str, Dict[str, List[Tuple[float, float, float]]]] = {}
    for row in rows:
        for leaf in row["leaves"]:
            for name, entry in leaf["by_bucket"].items():
                lines.setdefault(leaf["line"], {}).setdefault(name, []).append(
                    (entry["oracle_pct"], entry["rollout_root_pct"], entry["rollout_cond_pct"])
                )
    out: Dict[str, Any] = {"spots": len(rows), "lines": {}}
    for line, buckets in sorted(lines.items(), key=lambda kv: -len(kv[1].get("all", []))):
        out["lines"][line] = {}
        for name, values in buckets.items():
            arr = np.array(values)
            out["lines"][line][name] = {
                "leaves": len(values), "oracle_pct": mean_se(arr[:, 0]), "rollout_root_pct": mean_se(arr[:, 1]),
                "rollout_cond_pct": mean_se(arr[:, 2]), "oracle_minus_rollout_root_pct": mean_se(arr[:, 0] - arr[:, 1]),
                "oracle_minus_rollout_cond_pct": mean_se(arr[:, 0] - arr[:, 2]), "root_minus_cond_pct": mean_se(arr[:, 1] - arr[:, 2]),
            }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", type=int, default=60)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--samples", type=int, default=400, help="rollout pairs per leaf")
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
            t0 = time.perf_counter()
            try:
                row = audit_spot(spot, deal.hole_cards[spot["state"].to_act_seat()], policy, lookup, evaluate_7card_batch, rng, int(args.samples), int(args.equity_runouts))
            except Exception as exc:
                print(f"[leaf] spot failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            rows.append(row)
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            first = row["leaves"][0]["by_bucket"].get("all", {})
            print(
                f"[leaf] {len(rows)}/{args.spots} {time.perf_counter() - t0:.0f}s pot={row['pot']:.0f} leaves={len(row['leaves'])} "
                f"{row['leaves'][0]['line']}: oracle={first.get('oracle_pct', float('nan')):.1f} root={first.get('rollout_root_pct', float('nan')):.1f} cond={first.get('rollout_cond_pct', float('nan')):.1f}",
                flush=True,
            )
            if len(rows) % 10 == 0:
                (out_dir / "summary.json").write_text(json.dumps(summarize(rows), indent=2, default=float), encoding="utf-8")
    summary = summarize(rows)
    summary["wall_s"] = time.perf_counter() - started
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "lines"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
