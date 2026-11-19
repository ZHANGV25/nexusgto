"""Root value audit: does the re-solve's valuation of its root actions agree
with the blueprint's actual continuation?

usage (published solve environment exported):
  python scripts/root_value_audit.py --street flop --spots 150 --out DIR \
      --production-env NEXUS_LEAF_VEC_SAMPLED= [--production-env NAME=VALUE ...]

The blueprint plays all six seats until a street root with exactly two live
players.  The hero (first to act) is re-solved once; the solver's stash gives
its root strategy for every hand, its tree, and (with the eager leaf table)
the values it BELIEVES each root action has.  The TRUE value of each root
action is estimated by rolling the hand out with the blueprint for both
players after that action, sampling hands from the solver's own ranges
("assumed") and from each seat's exact blueprint reach with the deal prior
("true").  Values are net chips from the root, reported in % of the pot.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lbr_core import COMBOS, N_COMBOS, combo_index, equity_vs_combos, live_mask, mean_se, sample_runouts  # noqa: E402
from scripts.lbr_game import advance_forced, betting_closed_by_allin, deal_from_rng  # noqa: E402
from scripts.lbr_policies import lookup_table  # noqa: E402

STREET_INDEX = {"flop": 1, "turn": 2}
STRONG, WEAK = 0.65, 0.35
BUCKETS = ("weak", "mid", "strong")


# -- spot generation ----------------------------------------------------------


def blueprint_play_to_root(deal: Any, lookup: Any, rng: np.random.Generator, street: int) -> Optional[Dict[str, Any]]:
    """All six seats play the blueprint until ``street``'s root with two live players."""
    from app.engine.agent.game_state import new_postflop_state
    from app.engine.agent.preflop_play import new_preflop_state

    decisions: List[Tuple[Any, int, int]] = []
    state = new_preflop_state()
    while not state.terminal:
        seat = state.to_act_seat()
        action = int(lookup.get_mix(state, deal.hole_cards[seat]).sample(rng))
        decisions.append((state, seat, action))
        state = state.advance(action)
    if state.uncontested or state.n_live != 2:
        return None
    post = new_postflop_state(
        seats=list(state.live_seat_names), preflop_prefix=list(state.preflop_prefix),
        board=list(deal.board), pot=int(state.pot), stacks=list(state.live_stacks),
    )
    while True:
        post = advance_forced(post, frozenset())
        if post.terminal or post.n_live != 2 or betting_closed_by_allin(post):
            return None
        if int(post.street) == street:
            break
        seat = post.to_act_seat()
        action = int(lookup.get_mix(post, deal.hole_cards[seat]).sample(rng))
        decisions.append((post, seat, action))
        post = post.advance(action)
    if post.street_history:
        return None
    return {"state": post, "decisions": decisions}


def true_reach(decisions: Sequence[Tuple[Any, int, int]], seat: int, lookup: Any) -> np.ndarray:
    """Per-combo blueprint reach of ``seat``: deal prior times its action probabilities."""
    weights = np.ones(N_COMBOS, dtype=np.float64)
    for state, acting, action in decisions:
        if acting == seat:
            weights = weights * lookup_table(lookup, state)[:, int(action)]
    return weights


def assumed_reach(tree: Any, player: int, board: Sequence[int]) -> np.ndarray:
    """The solver's row weights spread evenly over each row's isomorphic members."""
    from app.services.suit_isomorphism import group_by_canonical

    live = [COMBOS[int(i)] for i in np.nonzero(live_mask(board))[0]]
    members = {canon: group for canon, group in group_by_canonical(live, list(board))}
    weights = np.zeros(N_COMBOS, dtype=np.float64)
    for c0, c1, weight in tree.player_hands[player]:
        group = members.get((int(c0), int(c1)))
        if not group or weight <= 0.0:
            continue
        for a, b in group:
            weights[combo_index(a, b)] = float(weight) / len(group)
    return weights


def row_of_combo(tree: Any, player: int, board: Sequence[int]) -> Dict[int, int]:
    from app.services.suit_isomorphism import canonical_combo

    index = {(int(c0), int(c1)): i for i, (c0, c1, _w) in enumerate(tree.player_hands[player])}
    mapping = {}
    for j in np.nonzero(live_mask(board))[0]:
        c0, c1 = COMBOS[int(j)]
        row = index.get(canonical_combo(c0, c1, list(board)))
        if row is not None:
            mapping[int(j)] = row
    return mapping


# -- believed values (solver tree + eager leaf table) --------------------------


def root_action_values(tree: Any, table: np.ndarray, sigma: Dict[int, np.ndarray], player: int) -> np.ndarray:
    """``[n_hero_rows, n_root_actions]`` values the solver assigns each root action.

    The walk mirrors es_resolver.exploitability_c.player_gap: opponent reach is
    split by its sigma, leaves read the table, folds pay pot minus this street's
    bet, and every hero hand is normalised by its non-conflicting opponent mass.
    """
    from es_resolver.es_solver import T_DECISION, T_FOLD, T_LEAF, T_SHOWDOWN

    if int(tree.num_players) != 2:
        raise ValueError("root value audit supports heads-up trees only")
    opp = 1 - player
    nh_p, nh_q = int(tree.num_hands[player]), int(tree.num_hands[opp])
    bits_p = np.array([(1 << int(c0)) | (1 << int(c1)) for c0, c1, _w in tree.player_hands[player]], dtype=np.uint64)
    bits_q = np.array([(1 << int(c0)) | (1 << int(c1)) for c0, c1, _w in tree.player_hands[opp]], dtype=np.uint64)
    conflict_free = ((bits_p[:, None] & bits_q[None, :]) == 0).astype(np.float64)
    prior_q = np.array([float(w) for _c0, _c1, w in tree.player_hands[opp]])

    def walk(node_idx: int, reach_q: np.ndarray) -> np.ndarray:
        node = tree.nodes[node_idx]
        if node.type in (T_LEAF, T_FOLD, T_SHOWDOWN):
            mass = conflict_free @ reach_q
            if node.type == T_LEAF:
                return np.asarray(table[node.leaf_idx, player, :nh_p], dtype=np.float64) * mass
            if node.type == T_FOLD:
                my_bet = float(node.bets[player])
                value = float(node.pot) - my_bet if node.player == player else -my_bet
                return value * mass
            return np.zeros(nh_p)
        if node.type != T_DECISION:
            raise ValueError(f"unexpected node type {node.type}")
        n_actions = len(node.children)
        acting_rows = nh_p if node.player == player else nh_q
        sig = sigma.get(node_idx)
        sig = np.full((acting_rows, n_actions), 1.0 / n_actions) if sig is None else np.asarray(sig, dtype=np.float64)
        if node.player == player:
            children = [walk(child, reach_q) for child in node.children]
            return sum(sig[:, a] * children[a] for a in range(n_actions))
        return sum(walk(child, reach_q * sig[:, a]) for a, child in enumerate(node.children))

    root = tree.nodes[0]
    if root.player != player:
        raise ValueError("hero is not the root actor")
    norm = conflict_free @ prior_q
    norm = np.where(norm > 0.0, norm, 1.0)
    return np.stack([walk(child, prior_q.copy()) for child in root.children], axis=1) / norm[:, None]


# -- true values (blueprint rollouts) ------------------------------------------


def play_out(state: Any, hero_slot: int, cards_by_slot: Sequence[Tuple[int, int]], lookup: Any,
             rng: np.random.Generator, evaluate: Any) -> Tuple[float, int]:
    """Both players play the blueprint from ``state``; returns (hero award, hero final stack)."""
    while True:
        state = advance_forced(state, frozenset())
        if state.terminal:
            break
        slot = state.to_act_slot
        state = state.advance(int(lookup.get_mix(state, cards_by_slot[slot]).sample(rng)))
    opp_slot = 1 - hero_slot
    if not state.active[hero_slot]:
        award = 0.0
    elif not state.active[opp_slot]:
        award = float(state.pot)
    else:
        rows = np.array([list(state.board) + list(cards_by_slot[hero_slot]), list(state.board) + list(cards_by_slot[opp_slot])], dtype=np.int32)
        strength = evaluate(rows)
        award = float(state.pot) if strength[0] > strength[1] else 0.5 * float(state.pot) if strength[0] == strength[1] else 0.0
    return award, int(state.stacks[hero_slot])


def rollout(root: Any, action: int, hero_slot: int, cards_by_slot: Sequence[Tuple[int, int]], lookup: Any,
            rng: np.random.Generator, evaluate: Any) -> float:
    """Hero's net chips from the root when it takes ``action`` and both then play the blueprint."""
    award, final_stack = play_out(root.advance(int(action)), hero_slot, cards_by_slot, lookup, rng, evaluate)
    return award - (int(root.stacks[hero_slot]) - final_stack)


def sample_pair(rng: np.random.Generator, hero_w: np.ndarray, opp_w: np.ndarray, board: Sequence[int]) -> Optional[Tuple[int, int]]:
    hero_p = np.where(live_mask(board), hero_w, 0.0)
    if hero_p.sum() <= 0.0:
        return None
    hero = int(rng.choice(N_COMBOS, p=hero_p / hero_p.sum()))
    opp_p = np.where(live_mask(list(board) + list(COMBOS[hero])), opp_w, 0.0)
    if opp_p.sum() <= 0.0:
        return None
    return hero, int(rng.choice(N_COMBOS, p=opp_p / opp_p.sum()))


def rollout_values(
    root: Any, hero_slot: int, hero_w: np.ndarray, opp_w: np.ndarray, n_actions: int, samples: int,
    lookup: Any, rng: np.random.Generator, evaluate: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(hero combo index per sample, [samples, n_actions] net chips)`` with common random runouts."""
    board = list(root.board)
    visible = board[: {1: 3, 2: 4, 3: 5}[int(root.street)]]
    hands = np.zeros(samples, dtype=np.int64)
    values = np.zeros((samples, n_actions), dtype=np.float64)
    from dataclasses import replace

    for k in range(samples):
        pair = sample_pair(rng, hero_w, opp_w, visible)
        if pair is None:
            raise ValueError("empty range for rollouts")
        hero, opp = pair
        dead = visible + list(COMBOS[hero]) + list(COMBOS[opp])
        runout = sample_runouts(rng, dead, 5 - len(visible), 1)[0].tolist() if len(visible) < 5 else []
        rolled = replace(root, board=tuple(visible + runout))
        cards = [COMBOS[hero] if slot == hero_slot else COMBOS[opp] for slot in range(2)]
        seed = int(rng.integers(0, 2 ** 31 - 1))
        hands[k] = hero
        for a in range(n_actions):
            values[k, a] = rollout(rolled, a, hero_slot, cards, lookup, np.random.default_rng(seed + a), evaluate)
    return hands, values


# -- per-spot audit -------------------------------------------------------------


def row_equities(tree: Any, hero: int, opp_w: np.ndarray, board: Sequence[int], rng: np.random.Generator,
                 evaluate: Any, runouts: int) -> np.ndarray:
    """Equity of each hero row's representative combo against the opponent's combo weights."""
    out = np.zeros(int(tree.num_hands[hero]))
    for i, (c0, c1, weight) in enumerate(tree.player_hands[hero]):
        if weight <= 0.0 or c0 in board or c1 in board:
            continue
        runs = sample_runouts(rng, list(board) + [int(c0), int(c1)], 5 - len(board), runouts)
        equity = equity_vs_combos((int(c0), int(c1)), board, runs, evaluate)
        weights = np.where(live_mask(list(board) + [int(c0), int(c1)]), opp_w, 0.0)
        out[i] = float(np.dot(weights, equity) / weights.sum()) if weights.sum() > 0 else 0.0
    return out


def bucket_of(equity: float) -> str:
    return "strong" if equity > STRONG else "weak" if equity < WEAK else "mid"


def _pct(values: np.ndarray, pot: float) -> float:
    return 100.0 * float(values.mean()) / pot if values.size else float("nan")


SOLVE_STAT_KEYS = ("iterations_run", "solve_s", "num_nodes", "flop_chance", "master_seed")


def last_solve_stats(policy: Any) -> Dict[str, Any]:
    """The engine's stats for the solve that produced the policy's last table."""
    stats = getattr(getattr(policy, "_engine", None), "_last_solve_stats", None) or {}
    return {key: stats.get(key) for key in SOLVE_STAT_KEYS if key in stats}


def audit_spot(spot: Dict[str, Any], hero_cards: Tuple[int, int], policy: Any, lookup: Any, evaluate: Any,
               rng: np.random.Generator, samples: int, equity_runouts: int) -> Dict[str, Any]:
    root = spot["state"]
    hero_slot = int(root.to_act_slot)
    hero_seat, opp_seat = root.seat_of_slot[hero_slot], root.seat_of_slot[1 - hero_slot]
    board = list(root.board)[: {1: 3, 2: 4}[int(root.street)]]
    n_actions = int(root.num_legal_actions())
    search_table = policy.table(root, hero_cards)
    objs = getattr(policy, "last_objs", None)
    if objs is None:
        raise RuntimeError("no fresh solver stash for the root")
    tree, sigma, table = objs["tree"], objs["sigma"], objs.get("table")
    pot = float(root.pot)
    blueprint_table = lookup_table(lookup, root)
    hero_rows = row_of_combo(tree, hero_slot, board)
    assumed_hero, assumed_opp = assumed_reach(tree, hero_slot, board), assumed_reach(tree, 1 - hero_slot, board)
    true_hero = true_reach(spot["decisions"], hero_seat, lookup)
    true_opp = true_reach(spot["decisions"], opp_seat, lookup)
    equities = row_equities(tree, hero_slot, assumed_opp, board, rng, evaluate, equity_runouts)
    believed = None
    believed_error = None
    if table is not None:
        try:
            believed = root_action_values(tree, table, sigma, hero_slot)
        except ValueError as exc:
            # Turn trees carry chance and showdown nodes the single-street walk
            # does not value; the strategy comparison below still applies.
            believed_error = str(exc)
    if any(np.asarray(sig).shape[0] != len(tree.player_hands[tree.nodes[ni].player]) for ni, sig in sigma.items()):
        raise ValueError("solver readout has fewer rows than the hand list")
    result: Dict[str, Any] = {
        "street": int(root.street), "hero_seat": int(hero_seat), "pot": pot, "n_actions": n_actions,
        "preflop_prefix": list(root.preflop_prefix), "postflop_history": list(root.postflop_history),
        "board": board, "hero_rows": int(tree.num_hands[hero_slot]), "table_source": objs.get("table_source"),
        "believed_error": believed_error,
        "solve": {"iterations": policy.stats.get("solves"), "wall_s": policy.stats.get("solve_wall_s")},
        "solve_stats": last_solve_stats(policy),
        "_tables": {
            "search": np.asarray(search_table, dtype=np.float32), "blueprint": np.asarray(blueprint_table, dtype=np.float32),
            "true_hero": np.asarray(true_hero, dtype=np.float32),
        },
    }
    for label, hero_w, opp_w in (("assumed", assumed_hero, assumed_opp), ("true", true_hero, true_opp)):
        hands, values = rollout_values(root, hero_slot, hero_w, opp_w, n_actions, samples, lookup, rng, evaluate)
        rows = np.array([hero_rows.get(int(h), -1) for h in hands])
        keep = rows >= 0
        sig_search = np.array([search_table[h] for h in hands])
        sig_bp = np.array([blueprint_table[h] for h in hands])
        ev_search = (sig_search * values).sum(axis=1)
        ev_bp = (sig_bp * values).sum(axis=1)
        ev_br = values.max(axis=1)
        block = {
            "ev_search_pct": _pct(ev_search, pot), "ev_blueprint_pct": _pct(ev_bp, pot), "ev_best_root_pct": _pct(ev_br, pot),
            "search_minus_blueprint_pct": _pct(ev_search - ev_bp, pot),
            "true_action_pct": [_pct(values[:, a], pot) for a in range(n_actions)],
            "bet_freq_search": float(sig_search[:, 1:].sum(axis=1).mean()), "bet_freq_blueprint": float(sig_bp[:, 1:].sum(axis=1).mean()),
            "by_bucket": {},
        }
        buckets = np.array([bucket_of(equities[r]) if r >= 0 else "mid" for r in rows])
        for name in BUCKETS:
            mask = keep & (buckets == name)
            if not mask.any():
                continue
            entry = {
                "share": float(mask.mean()), "true_action_pct": [_pct(values[mask, a], pot) for a in range(n_actions)],
                "bet_freq_search": float(sig_search[mask, 1:].sum(axis=1).mean()), "bet_freq_blueprint": float(sig_bp[mask, 1:].sum(axis=1).mean()),
                "search_minus_blueprint_pct": _pct(ev_search[mask] - ev_bp[mask], pot),
            }
            if believed is not None:
                entry["believed_action_pct"] = [_pct(believed[rows[mask], a], pot) for a in range(n_actions)]
            block["by_bucket"][name] = entry
        if believed is not None and label == "assumed":
            block["believed_action_pct"] = [_pct(believed[rows[keep], a], pot) for a in range(n_actions)]
            block["believed_minus_true_pct"] = [
                _pct(believed[rows[keep], a] - values[keep, a], pot) for a in range(n_actions)
            ]
        result[label] = block
    return result


# -- driver ---------------------------------------------------------------------


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"spots": len(rows)}
    for label in ("assumed", "true"):
        block: Dict[str, Any] = {}
        for key in ("ev_search_pct", "ev_blueprint_pct", "ev_best_root_pct", "search_minus_blueprint_pct", "bet_freq_search", "bet_freq_blueprint"):
            block[key] = mean_se([r[label][key] for r in rows])
        for a in range(2):
            block[f"true_action{a}_pct"] = mean_se([r[label]["true_action_pct"][a] for r in rows if len(r[label]["true_action_pct"]) > a])
            block[f"believed_action{a}_pct"] = mean_se([r[label]["believed_action_pct"][a] for r in rows if "believed_action_pct" in r[label]])
            block[f"believed_minus_true{a}_pct"] = mean_se([r[label]["believed_minus_true_pct"][a] for r in rows if "believed_minus_true_pct" in r[label]])
        block["by_bucket"] = {}
        for name in BUCKETS:
            entries = [r[label]["by_bucket"][name] for r in rows if name in r[label]["by_bucket"]]
            block["by_bucket"][name] = {
                "spots": len(entries),
                "bet_freq_search": mean_se([e["bet_freq_search"] for e in entries]),
                "bet_freq_blueprint": mean_se([e["bet_freq_blueprint"] for e in entries]),
                "search_minus_blueprint_pct": mean_se([e["search_minus_blueprint_pct"] for e in entries]),
                "true_check_pct": mean_se([e["true_action_pct"][0] for e in entries]),
                "true_bet_pct": mean_se([e["true_action_pct"][1] for e in entries if len(e["true_action_pct"]) > 1]),
                "believed_check_pct": mean_se([e["believed_action_pct"][0] for e in entries if "believed_action_pct" in e]),
                "believed_bet_pct": mean_se([e["believed_action_pct"][1] for e in entries if "believed_action_pct" in e and len(e["believed_action_pct"]) > 1]),
            }
        out[label] = block
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--street", choices=("flop", "turn"), default="flop")
    ap.add_argument("--spots", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=3000, help="rollouts per range variant per spot")
    ap.add_argument("--equity-runouts", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blueprint", default=None)
    ap.add_argument("--live-threads", type=int, default=8)
    ap.add_argument("--production-env", action="append", default=[])
    ap.add_argument("--deadline-ms", type=int, default=None, help="solve deadline override (research ladder)")
    ap.add_argument("--iterations", type=int, default=None, help="iteration ceiling override (research ladder)")
    ap.add_argument("--save-tables", action="store_true", help="write each spot's root tables to tables.npz")
    ap.add_argument("--max-consecutive-failures", type=int, default=25, help="abort instead of looping on a broken configuration")
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
                          env_overrides=_env_pairs(args.production_env),
                          iterations_override=args.iterations, deadline_ms_override=args.deadline_ms)
    lookup = policy._lookup
    rng = np.random.default_rng(args.seed)
    street = STREET_INDEX[args.street]
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "blueprint": blueprint}, indent=2), encoding="utf-8")
    rows: List[Dict[str, Any]] = []
    tables: Dict[str, np.ndarray] = {}
    dealt = 0
    failures_in_a_row = 0
    started = time.perf_counter()
    with (out_dir / "spots.jsonl").open("w", encoding="utf-8") as fh:
        while len(rows) < int(args.spots):
            deal = deal_from_rng(rng)
            dealt += 1
            spot = blueprint_play_to_root(deal, lookup, rng, street)
            if spot is None:
                continue
            hero_seat = spot["state"].to_act_seat()
            t0 = time.perf_counter()
            try:
                row = audit_spot(spot, deal.hole_cards[hero_seat], policy, lookup, evaluate_7card_batch, rng, int(args.samples), int(args.equity_runouts))
            except Exception as exc:
                print(f"[audit] spot failed: {type(exc).__name__}: {exc}", flush=True)
                failures_in_a_row += 1
                if failures_in_a_row >= int(args.max_consecutive_failures):
                    raise SystemExit(f"[audit] {failures_in_a_row} consecutive spot failures; aborting") from exc
                continue
            failures_in_a_row = 0
            spot_tables = row.get("_tables", {})
            row = {**{k: v for k, v in row.items() if k != "_tables"}, "deal": dealt}
            if args.save_tables:
                tables = {**tables, **{f"{name}_{len(rows)}": arr for name, arr in spot_tables.items()}}
            rows.append(row)
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            a, t = row["assumed"], row["true"]
            print(
                f"[audit] {len(rows)}/{args.spots} {time.perf_counter() - t0:.0f}s pot={row['pot']:.0f} "
                f"assumed: believed={a.get('believed_action_pct')} true={[round(v, 1) for v in a['true_action_pct']]} "
                f"| true ranges: search-bp={t['search_minus_blueprint_pct']:+.1f} bet {t['bet_freq_search']:.2f}/{t['bet_freq_blueprint']:.2f}",
                flush=True,
            )
            if len(rows) % 10 == 0:
                (out_dir / "summary.json").write_text(json.dumps(summarize(rows), indent=2, default=float), encoding="utf-8")
                if tables:
                    np.savez_compressed(out_dir / "tables.npz", **tables)
    summary = summarize(rows)
    summary["dealt"] = dealt
    summary["wall_s"] = time.perf_counter() - started
    summary["policy_stats"] = dict(policy.stats)
    if tables:
        np.savez_compressed(out_dir / "tables.npz", **tables)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
