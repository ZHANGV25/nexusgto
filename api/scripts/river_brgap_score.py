"""Grade captured heads-up river spots: search vs blueprint best-response gap.

usage: python scripts/river_brgap_score.py CAPTURE_DIR [CAPTURE_DIR ...] [--json OUT]

Each ``river_brgap_v1`` record (scripts/river_brgap_capture.py) holds one served
river tree with the search strategy and the blueprint strategy projected onto the
same tree and ranges. The gap is poker-solver ``exploitability_c``'s range-prior
BR-gap extended to an exact heads-up river showdown: for each player, the
range-weighted mean over their hands of best-response value minus value under
the profile, both against the opponent's seeded range, summed over the two
players. Reported as percent of the river starting pot.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

T_DECISION = 0
T_FOLD = 1
T_SHOWDOWN = 2
Evaluator = Callable[[np.ndarray], np.ndarray]


def _rows(tree: Mapping[str, Any], player: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hands = tree["player_hands"][player]
    cards = np.array([(h[0], h[1]) for h in hands], dtype=np.int64).reshape(-1, 2)
    weights = np.array([h[2] for h in hands], dtype=np.float64)
    bits = np.array([(1 << int(a)) | (1 << int(b)) for a, b in cards], dtype=np.uint64)
    return cards, weights, bits


def _board_bits(tree: Mapping[str, Any]) -> np.uint64:
    return np.uint64(sum(1 << int(card) for card in tree["board"]))


def showdown_strengths(tree: Mapping[str, Any], evaluate: Evaluator) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row hand strength for both players from one evaluator call (higher wins)."""
    board = [int(card) for card in tree["board"]]
    if len(board) != 5:
        raise ValueError(f"river tree board has {len(board)} cards")
    blocked = _board_bits(tree)
    parts = [_rows(tree, player) for player in (0, 1)]
    cards = np.vstack([part[0] for part in parts])
    valid = np.concatenate([(part[2] & blocked) == 0 for part in parts])
    strengths = np.zeros(len(cards), dtype=np.int64)
    if valid.any():
        seven = np.hstack([np.tile(board, (int(valid.sum()), 1)), cards[valid]])
        strengths[valid] = evaluate(seven.astype(np.int32))
    split = len(parts[0][0])
    return strengths[:split], strengths[split:]


def player_gap(
    tree: Mapping[str, Any],
    sigma: Mapping[int, np.ndarray],
    player: int,
    strengths: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(best response, achieved, hand weights) per row of ``player``."""
    if int(tree["num_players"]) != 2:
        raise ValueError("river BR-gap supports heads-up trees only")
    opp = 1 - player
    _, w_p, bits_p = _rows(tree, player)
    _, w_q, bits_q = _rows(tree, opp)
    blocked = _board_bits(tree)
    valid_p = (bits_p & blocked) == 0
    valid_q = (bits_q & blocked) == 0
    nc = (
        ((bits_p[:, None] & bits_q[None, :]) == 0)
        & valid_p[:, None]
        & valid_q[None, :]
    ).astype(np.float64)
    diff = strengths[player][:, None].astype(np.float64) - strengths[opp][None, :].astype(np.float64)
    cmp = np.sign(diff) * nc
    nodes = tree["nodes"]
    nh_p, nh_q = len(w_p), len(w_q)

    def terminal(node: Mapping[str, Any], reach_q: np.ndarray) -> np.ndarray:
        my_bet = float(node["bets"][player])
        pot = float(node["pot"])
        mass = nc @ reach_q
        if node["type"] == T_FOLD:
            return (pot - my_bet if node["player"] == player else -my_bet) * mass
        if not node["active"][opp]:
            return (pot - my_bet) * mass
        if not node["active"][player]:
            return -my_bet * mass
        return 0.5 * pot * (cmp @ reach_q) + (0.5 * pot - my_bet) * mass

    def walk(node_idx: int, reach_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        node = nodes[node_idx]
        if node["type"] in (T_FOLD, T_SHOWDOWN):
            values = terminal(node, reach_q)
            return values, values
        if node["type"] != T_DECISION:
            raise ValueError(f"node {node_idx} type {node['type']} is not river fold/showdown/decision")
        children = node["children"]
        n_actions = len(children)
        acting_rows = nh_p if node["player"] == player else nh_q
        sig = sigma.get(node_idx)
        if sig is None:
            sig = np.full((acting_rows, n_actions), 1.0 / n_actions)
        if node["player"] == player:
            pairs = [walk(child, reach_q) for child in children]
            br_children = np.stack([pair[0] for pair in pairs])
            ach_children = np.stack([pair[1] for pair in pairs])
            return br_children.max(axis=0), (sig * ach_children.T).sum(axis=1)
        br = np.zeros(nh_p)
        ach = np.zeros(nh_p)
        for action, child in enumerate(children):
            child_br, child_ach = walk(child, reach_q * sig[:, action])
            br = br + child_br
            ach = ach + child_ach
        return br, ach

    br, ach = walk(0, w_q.copy())
    norm = nc @ w_q
    norm = np.where(norm > 0.0, norm, 1.0)
    return br / norm, ach / norm, np.where(valid_p, w_p, 0.0)


def exploitability(
    tree: Mapping[str, Any], sigma: Mapping[int, np.ndarray], strengths: Sequence[np.ndarray]
) -> Tuple[float, List[float]]:
    """Sum over both players of the range-weighted mean BR gain, in tree chips."""
    gaps = []
    for player in (0, 1):
        br, ach, weights = player_gap(tree, sigma, player, strengths)
        total = float(weights.sum())
        gaps.append(float(((br - ach) * weights).sum() / total) if total > 0.0 else 0.0)
    return float(sum(gaps)), gaps


def score_record(record: Mapping[str, Any], evaluate: Evaluator) -> Dict[str, Any]:
    tree = record["tree"]
    strengths = showdown_strengths(tree, evaluate)
    pot = float(tree["starting_pot"])
    result: Dict[str, Any] = {"starting_pot": pot, "projection": record.get("projection_stats")}
    for name in ("search", "blueprint"):
        total, per_player = exploitability(tree, record[f"{name}_sigma"], strengths)
        result[f"{name}_pct_pot"] = 100.0 * total / pot
        result[f"{name}_per_player_pct_pot"] = [100.0 * gap / pot for gap in per_player]
    return result


def _mean_se(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else float("nan")
    return {"mean": float(array.mean()), "se": se, "median": float(np.median(array))}


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    search = [row["search_pct_pot"] for row in rows]
    blueprint = [row["blueprint_pct_pot"] for row in rows]
    diff = [s - b for s, b in zip(search, blueprint)]
    rows_total = sum((row["projection"] or {}).get("rows", 0) for row in rows)
    misses = sum((row["projection"] or {}).get("misses", 0) for row in rows)
    return {
        "spots": len(rows),
        "search_pct_pot": _mean_se(search),
        "blueprint_pct_pot": _mean_se(blueprint),
        "search_minus_blueprint_pct_pot": _mean_se(diff),
        "search_less_exploitable_share": float(np.mean([d < 0.0 for d in diff])),
        "blueprint_lookup_miss_rate": (misses / rows_total) if rows_total else None,
    }


def load_records(dirs: Sequence[str]) -> List[Dict[str, Any]]:
    records = []
    for directory in dirs:
        for path in sorted(Path(directory).rglob("spot_*.pkl")):
            with path.open("rb") as fh:
                record = pickle.load(fh)
            if record.get("schema") != "river_brgap_v1":
                raise ValueError(f"{path}: not a river_brgap_v1 record")
            records.append(record)
    return records


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", default=None, help="write per-spot rows and the summary here")
    args = ap.parse_args(argv)
    from scripts.match_aivat import evaluate_7card_batch

    rows = [score_record(record, evaluate_7card_batch) for record in load_records(args.dirs)]
    if not rows:
        print("no river_brgap_v1 records found")
        return 1
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "spots": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
