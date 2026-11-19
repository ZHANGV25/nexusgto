"""Audit the ranges a heads-up river re-solve is posed against.

usage: python scripts/river_range_audit.py CAPTURE_DIR [CAPTURE_DIR ...] --out DIR
         [--limit N] [--blueprint PATH] [--no-resolve]

The live re-solve seeds every seat's range from the PREFLOP strategy only
(app.engine.agent.live_resolve_ranges): a river solve is posed against each
player's whole preflop range, as if no flop or turn action had happened. For
each captured heads-up river spot (scripts/river_brgap_capture.py records:
the assumed ranges are the tree's hand rows) this script replays the hand's
line with the blueprint lookup, multiplies each row's weight by the
blueprint's probability of the actions that seat actually took on the flop
and turn (the field plays the blueprint, so this is its true river range),
and reports how far the assumed range is from the true one.

With re-solving on, the same tree is solved twice under the published solve
configuration and the average readout (NEXUS_RIVER_AVG_READOUT=1): once with
the assumed ranges, once with the true ones. Every hero strategy (the captured
search strategy, labeled ``served``; both re-solves; the blueprint) is then
scored against the blueprint's own river strategy with the true ranges:
expected value in % of pot and the strong-hand bet frequency at the root.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.river_brgap_score import (  # noqa: E402
    T_DECISION, _board_bits, _rows, player_gap, showdown_strengths,
)

STRONG_EQUITY = 0.65
TEMPLATE_KWARGS = {
    "bet_sizes": [0.75], "subsequent_bet_sizes": [1.0], "add_explicit_allin": False,
    "iterations": 100_000, "deadline_ms": 3_000, "num_threads": 8, "num_rollouts": 3,
    "cache_k": 1, "vec_mode": 1, "readout": "final",
}


def load_records(dirs: Sequence[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for directory in dirs:
        for path in sorted(Path(directory).rglob("spot_*.pkl")):
            with path.open("rb") as fh:
                record = pickle.load(fh)
            if record.get("schema") != "river_brgap_v1":
                continue
            out.append({**record, "path": str(path)})
            if limit is not None and len(out) >= limit:
                return out
    return out


def rebuild_root(record: Mapping[str, Any]) -> Tuple[Any, List[Tuple[int, Any, int]], List[int]]:
    """Replay the captured line to the river root.

    Returns the root state, the postflop decisions before it as
    (slot, state_before, action), and the solver-player -> slot map.
    """
    from app.engine.agent.game_state import new_postflop_state
    from app.engine.agent.preflop_play import replay_preflop

    tree = record["tree"]
    # Flop-entry chips from the shared preflop transition (the canonical
    # derivation refuses limp-raise and other non-canonical lines).
    pre = replay_preflop(list(record["preflop_prefix"]), require_terminal=True)
    state = new_postflop_state(
        seats=list(pre.live_seat_names),
        preflop_prefix=list(record["preflop_prefix"]),
        board=list(tree["board"]),
        pot=int(pre.pot),
        stacks=list(pre.live_stacks),
        true_hu=bool(record.get("true_hu", False)),
    )
    if tuple(int(s) for s in state.seat_of_slot) != tuple(int(s) for s in record["seat_of_slot"]):
        raise ValueError("replayed seats differ from the captured slots")
    decisions: List[Tuple[int, Any, int]] = []
    for action in record["postflop_history"]:
        slot = state.to_act_slot
        if slot is None:
            raise ValueError("line ends before the river root")
        decisions.append((int(slot), state, int(action)))
        state = state.advance(int(action))
    if int(state.street) != 3 or tuple(state.street_history) != ():
        raise ValueError(f"replay did not land on a river root (street {state.street})")
    entry = [s for s in range(state.n_slots) if state.active[s] or state.acted[s]]
    if len(entry) != int(tree["num_players"]):
        raise ValueError(f"{len(entry)} entry slots for {tree['num_players']} players")
    if state.to_act_slot != entry[int(record["hero_player"])]:
        raise ValueError("hero slot disagrees with the replayed to-act slot")
    if tuple(int(c) for c in state.board[:5]) != tuple(tree["board"]):
        raise ValueError("replayed board differs from the tree board")
    return state, decisions, entry


def postflop_factors(
    tree: Mapping[str, Any], decisions: Sequence[Tuple[int, Any, int]],
    entry: Sequence[int], lookup: Any,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """Per player and row: product of the blueprint's probabilities of the
    postflop actions that player's seat took before the river root."""
    stats = {"decisions": 0, "lookups": 0, "uniform_rows": 0}
    factors: List[np.ndarray] = []
    for player, slot in enumerate(entry):
        rows = tree["player_hands"][player]
        factor = np.ones(len(rows), dtype=np.float64)
        for dec_slot, state_before, action in decisions:
            if dec_slot != slot or int(state_before.street) >= 3:
                continue
            stats["decisions"] += 1
            for row, (c0, c1, weight) in enumerate(rows):
                if weight <= 0.0:
                    continue
                mix = lookup.get_mix(state_before, (int(c0), int(c1)))
                stats["lookups"] += 1
                stats["uniform_rows"] += int(bool(getattr(mix, "is_uniform_fallback", False)))
                factor[row] *= float(mix.freqs.get(int(action), 0.0))
        factors.append(factor)
    return factors, stats


def _normalize(weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    return weights / total if total > 0.0 else weights


def range_stats(assumed: np.ndarray, true: np.ndarray, strengths: np.ndarray) -> Dict[str, float]:
    a, t = _normalize(assumed), _normalize(true)
    order = np.argsort(strengths)
    percentile = np.empty(len(strengths))
    percentile[order] = np.linspace(0.0, 1.0, len(strengths))

    def effective(p: np.ndarray) -> float:
        nz = p[p > 0]
        return float(np.exp(-(nz * np.log(nz)).sum()))

    return {
        "tv": float(0.5 * np.abs(a - t).sum()),
        "mass_kept": float(true.sum() / assumed.sum()) if assumed.sum() > 0 else float("nan"),
        "effective_rows_assumed": effective(a),
        "effective_rows_true": effective(t),
        "mean_percentile_assumed": float((a * percentile).sum()),
        "mean_percentile_true": float((t * percentile).sum()),
        "top_quartile_assumed": float(a[percentile >= 0.75].sum()),
        "top_quartile_true": float(t[percentile >= 0.75].sum()),
    }


def hero_equity(tree: Mapping[str, Any], hero: int, opp_weights: np.ndarray,
                strengths: Sequence[np.ndarray]) -> np.ndarray:
    opp = 1 - hero
    _, _, bits_h = _rows(tree, hero)
    _, _, bits_o = _rows(tree, opp)
    blocked = _board_bits(tree)
    nc = (((bits_h[:, None] & bits_o[None, :]) == 0)
          & ((bits_h & blocked) == 0)[:, None] & ((bits_o & blocked) == 0)[None, :]).astype(np.float64)
    win = (strengths[hero][:, None] > strengths[opp][None, :]).astype(np.float64)
    tie = (strengths[hero][:, None] == strengths[opp][None, :]).astype(np.float64)
    mass = nc @ opp_weights
    score = ((win + 0.5 * tie) * nc) @ opp_weights
    return np.where(mass > 0.0, score / np.where(mass > 0.0, mass, 1.0), 0.0)


def with_weights(tree: Mapping[str, Any], weights: Sequence[np.ndarray]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(tree))
    out["player_hands"] = [
        [(c0, c1, float(w[i])) for i, (c0, c1, _old) in enumerate(hands)]
        for hands, w in zip(tree["player_hands"], weights)
    ]
    return out


def joint_profile(tree: Mapping[str, Any], hero: int, hero_sigma: Mapping[int, np.ndarray],
                  opp_sigma: Mapping[int, np.ndarray]) -> Dict[int, np.ndarray]:
    joint: Dict[int, np.ndarray] = {}
    for ni, node in enumerate(tree["nodes"]):
        if node["type"] != T_DECISION:
            continue
        source = hero_sigma if node["player"] == hero else opp_sigma
        if ni in source:
            joint[ni] = np.asarray(source[ni], dtype=np.float64)
    return joint


def score_strategy(
    tree_true: Mapping[str, Any], hero: int, hero_sigma: Mapping[int, np.ndarray],
    blueprint_sigma: Mapping[int, np.ndarray], strengths: Sequence[np.ndarray],
    strong_mask: np.ndarray, hero_weights: np.ndarray,
) -> Dict[str, Optional[float]]:
    joint = joint_profile(tree_true, hero, hero_sigma, blueprint_sigma)
    br, ach, weights = player_gap(tree_true, joint, hero, strengths)
    total = float(weights.sum())
    pot = float(tree_true["starting_pot"])
    root = np.asarray(hero_sigma[0], dtype=np.float64) if 0 in hero_sigma else None
    strong_bet = None
    if root is not None and tree_true["nodes"][0]["player"] == hero:
        bet = 1.0 - root[:, 0]
        mass = float((hero_weights * strong_mask).sum())
        strong_bet = float((bet * hero_weights * strong_mask).sum() / mass) if mass > 0 else None
    return {
        "ev_pct_pot": 100.0 * float((ach * weights).sum() / total) / pot if total > 0 else None,
        "br_pct_pot": 100.0 * float((br * weights).sum() / total) / pot if total > 0 else None,
        "strong_bet": strong_bet,
    }


class Resolver:
    """Re-solve one captured river tree under given ranges (average readout)."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def solve(self, root_state: Any, tree: Mapping[str, Any], hero: int,
              weights: Sequence[np.ndarray], hero_cards: Tuple[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        from app.engine.agent.live_resolve_adapter import build_solve_inputs
        from app.services.suit_isomorphism import canonical_combo

        inputs = build_solve_inputs(root_state, range_weight_dicts=None)
        ranges = []
        for player, rows in enumerate(inputs["player_ranges"]):
            tree_rows = tree["player_hands"][player]
            if len(rows) != len(tree_rows) or any(
                (int(r[0]), int(r[1])) != (int(t[0]), int(t[1])) for r, t in zip(rows, tree_rows)
            ):
                raise ValueError(f"player {player}: rebuilt grid rows differ from the captured tree")
            ranges.append([(int(r[0]), int(r[1]), float(weights[player][i])) for i, r in enumerate(rows)])
        canon = canonical_combo(int(hero_cards[0]), int(hero_cards[1]), list(tree["board"]))
        hero_rows = [i for i, r in enumerate(ranges[hero]) if (r[0], r[1]) == (int(canon[0]), int(canon[1]))]
        if not hero_rows:
            raise ValueError("hero combo has no row in the rebuilt grid")
        kwargs = {
            **TEMPLATE_KWARGS,
            "board_strs": inputs["board_strs"], "board_ints": inputs["board_ints"],
            "player_ranges": ranges, "positions_sorted": inputs["positions_sorted"],
            "player_seats": inputs["player_seats"], "pot_bb": inputs["pot_bb"],
            "stack_bb": inputs["stack_bb"], "game_action_indices": inputs["game_action_indices"],
            "hero_replay": {"player_idx": hero, "street_actions": [], "expected_actions": inputs["n_legal"]},
            "hero_actual_hand": {"player_idx": hero, "hand_idx": hero_rows[0], "traverser_fraction": 0.5},
        }
        before = getattr(self._engine, "_last_br_gap_objs", None)
        self._engine.solve(**kwargs)
        objs = getattr(self._engine, "_last_br_gap_objs", None)
        if objs is None or objs is before:
            raise RuntimeError("no fresh BR-gap stash after the re-solve")
        stats = getattr(self._engine, "_last_solve_stats", None) or {}
        sigma = {int(k): np.asarray(v, dtype=np.float64) for k, v in objs["sigma"].items()}
        return sigma, {"iterations_run": stats.get("iterations_run"), "solve_s": stats.get("solve_s"),
                       "readout": stats.get("readout")}


def audit_record(record: Mapping[str, Any], lookup: Any, evaluate: Any,
                 resolver: Optional[Resolver]) -> Dict[str, Any]:
    tree = record["tree"]
    hero = int(record["hero_player"])
    opp = 1 - hero
    root_state, decisions, entry = rebuild_root(record)
    factors, lookup_stats = postflop_factors(tree, decisions, entry, lookup)
    assumed = [np.array([h[2] for h in hands], dtype=np.float64) for hands in tree["player_hands"]]
    true = [assumed[p] * factors[p] for p in range(len(assumed))]
    strengths = showdown_strengths(tree, evaluate)
    row: Dict[str, Any] = {
        "spot": record["path"], "hero": hero, "postflop_history": list(record["postflop_history"]),
        "preflop_prefix": list(record["preflop_prefix"]), "rows": len(assumed[0]),
        "lookup": lookup_stats, "pot": tree["starting_pot"],
        "opp_range": range_stats(assumed[opp], true[opp], strengths[opp]),
        "hero_range": range_stats(assumed[hero], true[hero], strengths[hero]),
    }
    # The hero played search on earlier streets, so its blueprint-reach range
    # is only a stand-in; when the hero's own line has zero blueprint mass
    # (an action the blueprint never takes there) keep its assumed range.
    hero_range_source = "blueprint_reach"
    if float(true[hero].sum()) <= 0.0:
        true = [assumed[hero] if p == hero else true[p] for p in range(len(true))]
        hero_range_source = "assumed"
    row["hero_range_source"] = hero_range_source
    equity_true = hero_equity(tree, hero, true[opp], strengths)
    strong = (equity_true > STRONG_EQUITY).astype(np.float64)
    tree_true = with_weights(tree, true)
    bits_hero = np.asarray([(1 << int(h[0])) | (1 << int(h[1])) for h in tree["player_hands"][hero]], dtype=np.uint64)
    hero_w = np.where((bits_hero & _board_bits(tree)) == 0, true[hero], 0.0)
    bp = {int(k): np.asarray(v, dtype=np.float64) for k, v in record["blueprint_sigma"].items()}
    served = {int(k): np.asarray(v, dtype=np.float64) for k, v in record["search_sigma"].items()}
    strategies: Dict[str, Mapping[int, np.ndarray]] = {"served": served, "blueprint": bp}
    if resolver is not None:
        for label, weights in (("resolve_assumed", assumed), ("resolve_true", true)):
            sigma, stats = resolver.solve(root_state, tree, hero, weights, tuple(record["hero_cards"]))
            strategies[label] = sigma
            row[f"{label}_solve"] = stats
    row["strategies"] = {
        label: score_strategy(tree_true, hero, sigma, bp, strengths, strong, hero_w)
        for label, sigma in strategies.items()
    }
    row["strong_mass"] = float((hero_w * strong).sum() / hero_w.sum()) if hero_w.sum() > 0 else None
    return row


def _mean_se(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    clean = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not clean:
        return {"mean": None, "se": None, "n": 0}
    arr = np.asarray(clean)
    return {"mean": float(arr.mean()), "se": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else None,
            "n": len(arr)}


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"spots": len(rows)}
    for side in ("opp_range", "hero_range"):
        out[side] = {key: _mean_se([r[side][key] for r in rows]) for key in rows[0][side]} if rows else {}
    labels = sorted({label for r in rows for label in r["strategies"]})
    out["strategies"] = {
        label: {metric: _mean_se([r["strategies"].get(label, {}).get(metric) for r in rows])
                for metric in ("ev_pct_pot", "br_pct_pot", "strong_bet")}
        for label in labels
    }
    if "resolve_true" in labels and "resolve_assumed" in labels:
        paired = [r for r in rows if "resolve_true" in r["strategies"] and "resolve_assumed" in r["strategies"]]
        out["resolve_true_minus_assumed_ev"] = _mean_se([
            r["strategies"]["resolve_true"]["ev_pct_pot"] - r["strategies"]["resolve_assumed"]["ev_pct_pot"]
            for r in paired
        ])
        out["blueprint_minus_resolve_assumed_ev"] = _mean_se([
            r["strategies"]["blueprint"]["ev_pct_pot"] - r["strategies"]["resolve_assumed"]["ev_pct_pot"]
            for r in paired
        ])
        out["blueprint_minus_resolve_true_ev"] = _mean_se([
            r["strategies"]["blueprint"]["ev_pct_pot"] - r["strategies"]["resolve_true"]["ev_pct_pot"]
            for r in paired
        ])
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--blueprint", default=None)
    ap.add_argument("--no-resolve", action="store_true", default=False)
    args = ap.parse_args(argv)
    blueprint = args.blueprint or os.environ.get("AGENT_BLUEPRINT_PATH")
    if not blueprint:
        raise SystemExit("set --blueprint or AGENT_BLUEPRINT_PATH")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from app.engine.agent.modes import ModeConfig
    from app.engine.agent.strategy_source import LookupSource
    from scripts.match_aivat import evaluate_7card_batch

    lookup = LookupSource(ModeConfig(mode="LOOKUP", blueprint_path=blueprint, uniform_on_miss=True))
    resolver = None
    if not args.no_resolve:
        from scripts.run_match_cli import production_env_problems

        problems = production_env_problems(os.environ)
        if problems:
            raise SystemExit("published solve environment needed for re-solving: " + "; ".join(problems))
        os.environ["NEXUS_BR_GAP_EXPOSE"] = "1"
        os.environ["NEXUS_RIVER_AVG_READOUT"] = "1"
        from app.engine.cpu_es_solver import CpuEsSolver

        resolver = Resolver(CpuEsSolver(blueprint))
    records = load_records(args.dirs, args.limit)
    print(f"[audit] {len(records)} records", flush=True)
    rows: List[Dict[str, Any]] = []
    failures: Dict[str, int] = {}
    with (out_dir / "audit.jsonl").open("w", encoding="utf-8") as fh:
        for index, record in enumerate(records, 1):
            t0 = time.perf_counter()
            try:
                row = audit_record(record, lookup, evaluate_7card_batch, resolver)
            except Exception as exc:  # keep going; report the failure classes
                key = f"{type(exc).__name__}: {str(exc)[:80]}"
                failures[key] = failures.get(key, 0) + 1
                print(f"[audit] {index}/{len(records)} failed: {key}", flush=True)
                continue
            rows.append(row)
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            s = row["strategies"]

            def ev(label: str) -> str:
                value = s.get(label, {}).get("ev_pct_pot")
                return "-" if value is None else f"{value:.1f}"

            print(
                f"[audit] {index}/{len(records)} {time.perf_counter() - t0:.1f}s rows={row['rows']} "
                f"opp tv={row['opp_range']['tv']:.3f} kept={row['opp_range']['mass_kept']:.2f} "
                f"hero={row['hero_range_source']} ev served={ev('served')} bp={ev('blueprint')} "
                f"assumed={ev('resolve_assumed')} true={ev('resolve_true')}",
                flush=True,
            )
    summary = summarize(rows)
    summary["failures"] = failures
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
