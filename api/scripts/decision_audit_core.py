"""Core helpers for scripts/decision_audit.py."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.engine.agent.preflop_play import DEFAULT_INITIAL_STACK, SEAT_NAME
from scripts.lbr_core import (
    COMBOS,
    N_COMBOS,
    equity_vs_combos,
    live_mask,
    mean_se,
    sample_runouts,
)
from scripts.lbr_game import (
    STREET_BOARD_LEN,
    action_kinds,
    betting_closed_by_allin,
    deal_from_rng,
    seat_stack,
)
from scripts.lbr_policies import lookup_table

STREET_NAME = {1: "flop", 2: "turn", 3: "river"}
AUDIT_STREETS = (2, 3)
STRONG_EQUITY = 0.65
WEAK_EQUITY = 0.35


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return float(value) if isinstance(value, float) else str(value)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shard_selected(index: int, shard: Optional[Tuple[int, int]]) -> bool:
    if shard is None:
        return True
    part, total = shard
    if not 0 <= part < total:
        raise ValueError(f"invalid shard {part}/{total}")
    return index % total == part


def parse_shard(raw: Optional[str]) -> Optional[Tuple[int, int]]:
    if raw is None:
        return None
    if "/" not in raw:
        raise ValueError(f"--shard expects i/n, got {raw!r}")
    left, right = raw.split("/", 1)
    part, total = int(left), int(right)
    if total <= 0 or not 0 <= part < total:
        raise ValueError(f"invalid shard {raw!r}")
    return part, total


def card_pair_json(pair: Sequence[int]) -> List[int]:
    return [int(pair[0]), int(pair[1])]


def full_deal_json(deal: Any) -> Dict[str, Any]:
    return {
        "hole_cards": [card_pair_json(pair) for pair in deal.hole_cards],
        "board": [int(card) for card in deal.board],
    }


def _entry_slots(state: Any) -> List[int]:
    return [s for s in range(state.n_slots) if state.active[s] or state.acted[s]]


def _entry_seats(state: Any) -> List[str]:
    return [SEAT_NAME[int(state.seat_of_slot[s])] for s in _entry_slots(state)]


def _visible_board(state: Any) -> List[int]:
    return [int(c) for c in list(state.board)[: STREET_BOARD_LEN[int(state.street)]]]


def _post_action_record(state: Any, action: int) -> Dict[str, int]:
    slot = state.to_act_slot
    if slot is None:
        raise ValueError("terminal state has no postflop action")
    return {
        "street": int(state.street),
        "slot": int(slot),
        "seat": int(state.seat_of_slot[slot]),
        "action": int(action),
    }


def advance_forced_recorded(
    state: Any,
    records: List[Dict[str, int]],
) -> Any:
    """Apply forced all-in calls/checks and record their action indices."""
    while not state.terminal:
        seat = state.to_act_seat()
        if seat is None:
            break
        kinds = action_kinds(state)
        if betting_closed_by_allin(state) or seat_stack(state, int(seat)) <= 0:
            action = kinds.index("call")
            records.append({**_post_action_record(state, action), "forced": 1})
            state = state.advance(action)
            continue
        break
    return state


def _hero_acted_on_street(
    post_records: Sequence[Mapping[str, int]],
    street: int,
    slot: int,
) -> bool:
    return any(
        int(record["street"]) == int(street) and int(record["slot"]) == int(slot)
        and not record.get("forced")
        for record in post_records
    )


def spot_cell(state: Any, post_records: Sequence[Mapping[str, int]]) -> Dict[str, Any]:
    slot = state.to_act_slot
    if slot is None:
        raise ValueError("terminal state has no cell")
    entry_slots = _entry_slots(state)
    return {
        "street": STREET_NAME[int(state.street)],
        "street_index": int(state.street),
        "entry_players": int(len(entry_slots)),
        "facing_bet": bool(state.facing_bet),
        "hero_acted": _hero_acted_on_street(post_records, int(state.street), int(slot)),
    }


def cell_key(cell: Mapping[str, Any]) -> str:
    facing = "facing" if cell["facing_bet"] else "open"
    acted = "acted" if cell["hero_acted"] else "fresh"
    return f"{cell['street']}_{cell['entry_players']}p_{facing}_{acted}"


def blueprint_spots_for_deal(
    deal: Any,
    deal_index: int,
    lookup: Any,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Return every eligible turn/river decision reached by blueprint self-play."""
    from app.engine.agent.game_state import new_postflop_state
    from app.engine.agent.preflop_play import new_preflop_state

    pre = new_preflop_state()
    preflop_actions: List[int] = []
    sequence: List[Dict[str, int | str]] = []
    while not pre.terminal:
        seat = pre.to_act_seat()
        if seat is None:
            raise ValueError("nonterminal preflop state without actor")
        action = int(lookup.get_mix(pre, deal.hole_cards[seat]).sample(rng))
        preflop_actions.append(action)
        sequence.append({"phase": "preflop", "seat": int(seat), "action": action})
        pre = pre.advance(action)

    if pre.uncontested:
        return []
    post = new_postflop_state(
        seats=list(pre.live_seat_names),
        preflop_prefix=list(pre.preflop_prefix),
        board=list(deal.board),
        pot=int(pre.pot),
        stacks=list(pre.live_stacks),
    )
    post_records: List[Dict[str, int]] = []
    spots: List[Dict[str, Any]] = []
    while True:
        post = advance_forced_recorded(post, post_records)
        if post.terminal:
            break
        entry_count = len(_entry_slots(post))
        if int(post.street) in AUDIT_STREETS and entry_count in (2, 3):
            slot = post.to_act_slot
            if slot is None:
                raise ValueError("eligible nonterminal spot has no actor")
            cell = spot_cell(post, post_records)
            spots.append({
                "schema": "decision_audit_spot_v1",
                "deal_index": int(deal_index),
                "deal": full_deal_json(deal),
                "preflop_actions": list(preflop_actions),
                "postflop_actions": [int(r["action"]) for r in post_records],
                "postflop_forced": [int(r.get("forced", 0)) for r in post_records],
                "action_sequence": [
                    dict(record) for record in sequence
                    if record.get("phase") == "preflop"
                ] + [
                    {"phase": "postflop", **dict(r)} for r in post_records
                ],
                "cell": {**cell, "key": cell_key(cell)},
                "pot": int(post.pot),
                "hero_seat": int(post.seat_of_slot[slot]),
                "hero_slot": int(slot),
                "entry_seats": _entry_seats(post),
            })
        slot = post.to_act_slot
        if slot is None:
            raise ValueError("nonterminal postflop state without actor")
        seat = int(post.seat_of_slot[slot])
        action = int(lookup.get_mix(post, deal.hole_cards[seat]).sample(rng))
        record = _post_action_record(post, action)
        post_records.append(record)
        post = post.advance(action)
    return spots


def collect_sampled_spots(
    deals: int,
    per_cell: int,
    seed: int,
    lookup: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    all_spots: List[Dict[str, Any]] = []
    for deal_index in range(int(deals)):
        all_spots.extend(
            blueprint_spots_for_deal(deal_from_rng(rng), deal_index, lookup, rng)
        )
    by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for spot in all_spots:
        by_cell.setdefault(str(spot["cell"]["key"]), []).append(spot)
    sampler = np.random.default_rng(seed + 10_003)
    sampled: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    selected: Dict[str, int] = {}
    for key in sorted(by_cell):
        rows = by_cell[key]
        counts[key] = len(rows)
        take = min(int(per_cell), len(rows))
        selected[key] = take
        if take <= 0:
            continue
        picks = sampler.choice(len(rows), size=take, replace=False)
        inclusion = float(take) / float(len(rows))
        for pick in sorted(int(i) for i in picks):
            spot = dict(rows[pick])
            spot["spot_id"] = f"{key}_d{spot['deal_index']}_{pick}"
            spot["inclusion_probability"] = inclusion
            sampled.append(spot)
    meta = {
        "schema": "decision_audit_meta_v1",
        "deals": int(deals),
        "per_cell": int(per_cell),
        "seed": int(seed),
        "n_c": counts,
        "sampled_c": selected,
        "eligible": len(all_spots),
        "sampled": len(sampled),
    }
    return sampled, meta


def replay_spot(
    spot: Mapping[str, Any],
) -> Tuple[Any, List[Tuple[Any, int, int]], List[Tuple[Any, int, int]]]:
    """Replay a spot to its node; return state, all decisions, street-start decisions."""
    from app.engine.agent.game_state import new_postflop_state
    from app.engine.agent.preflop_play import new_preflop_state

    deal = spot["deal"]
    pre = new_preflop_state()
    all_decisions: List[Tuple[Any, int, int]] = []
    for action in spot["preflop_actions"]:
        seat = pre.to_act_seat()
        if seat is None:
            raise ValueError("preflop actions continue after terminal")
        all_decisions.append((pre, int(seat), int(action)))
        pre = pre.advance(int(action))
    if pre.uncontested:
        raise ValueError("spot preflop replay ended uncontested")
    post = new_postflop_state(
        seats=list(pre.live_seat_names),
        preflop_prefix=list(pre.preflop_prefix),
        board=list(deal["board"]),
        pot=int(pre.pot),
        stacks=list(pre.live_stacks),
    )
    target_street = int(spot["cell"]["street_index"])
    start_decisions = list(all_decisions)
    actions = [int(a) for a in spot["postflop_actions"]]
    forced = [int(f) for f in spot.get("postflop_forced", [0] * len(actions))]
    if len(forced) != len(actions):
        raise ValueError("postflop_forced length differs from postflop_actions")
    for action, is_forced in zip(actions, forced):
        if post.to_act_slot is None:
            raise ValueError("postflop actions continue after terminal")
        # Forced all-in checks/calls are not choices: they advance the state
        # but carry no range information.
        if not is_forced:
            decision = (post, int(post.to_act_seat()), action)
            if int(post.street) < target_street:
                start_decisions.append(decision)
            all_decisions.append(decision)
        post = post.advance(action)
    if int(post.street) != target_street:
        raise ValueError(
            f"replay landed on street {post.street}, expected {target_street}"
        )
    return post, all_decisions, start_decisions


def true_reaches(
    decisions: Sequence[Tuple[Any, int, int]],
    seats: Sequence[int],
    lookup: Any,
) -> Dict[int, np.ndarray]:
    weights = {int(seat): np.ones(N_COMBOS, dtype=np.float64) for seat in seats}
    for state, acting, action in decisions:
        seat = int(acting)
        if seat not in weights:
            continue
        table = lookup_table(lookup, state)
        weights[seat] = weights[seat] * table[:, int(action)]
    return weights


def oracle_range_overrides(
    state: Any,
    start_decisions: Sequence[Tuple[Any, int, int]],
    lookup: Any,
) -> Dict[str, np.ndarray]:
    seats = [int(state.seat_of_slot[s]) for s in _entry_slots(state)]
    reaches = true_reaches(start_decisions, seats, lookup)
    return {SEAT_NAME[seat]: reaches[seat] for seat in seats}


def node_reaches(
    state: Any,
    node_decisions: Sequence[Tuple[Any, int, int]],
    lookup: Any,
) -> Dict[int, np.ndarray]:
    seats = [int(state.seat_of_slot[s]) for s in _entry_slots(state)]
    return true_reaches(node_decisions, seats, lookup)


def sample_joint_combos(
    rng: np.random.Generator,
    weights_by_slot: Mapping[int, np.ndarray],
    slots: Sequence[int],
    board: Sequence[int],
    max_tries: int = 10_000,
) -> Dict[int, int]:
    live = live_mask(board)
    probs_by_slot: Dict[int, np.ndarray] = {}
    for slot in slots:
        weights = np.where(live, np.asarray(weights_by_slot[int(slot)], dtype=np.float64), 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError(f"empty reach for slot {slot}")
        probs_by_slot[int(slot)] = weights / total
    for _ in range(max_tries):
        draw = {
            int(slot): int(rng.choice(N_COMBOS, p=probs_by_slot[int(slot)]))
            for slot in slots
        }
        seen = set(int(c) for c in board)
        ok = True
        for combo in draw.values():
            c0, c1 = COMBOS[int(combo)]
            if c0 in seen or c1 in seen:
                ok = False
                break
            seen.add(c0)
            seen.add(c1)
        if ok:
            return draw
    raise ValueError("joint combo sampler exceeded rejection limit")


def _check_no_unequal_allin_showdown(state: Any) -> None:
    active = [slot for slot in range(state.n_slots) if state.active[slot]]
    if len(active) < 2:
        return
    if not any(int(state.stacks[slot]) == 0 for slot in active):
        return
    contrib = [DEFAULT_INITIAL_STACK - int(state.stacks[slot]) for slot in active]
    if len(set(contrib)) != 1:
        raise ValueError(
            "terminal state implies unequal all-in contributions; side pots "
            "are outside decision_audit"
        )


def play_out_blueprint(
    state: Any,
    hero_slot: int,
    cards_by_slot: Sequence[Tuple[int, int]],
    lookup: Any,
    rng: np.random.Generator,
    evaluate: Any,
) -> Tuple[float, int]:
    while True:
        state = advance_forced_recorded(state, [])
        if state.terminal:
            break
        slot = state.to_act_slot
        if slot is None:
            raise ValueError("nonterminal rollout state without actor")
        cards = cards_by_slot[int(slot)]
        action = int(lookup.get_mix(state, cards).sample(rng))
        state = state.advance(action)
    live = [slot for slot in range(state.n_slots) if state.active[slot]]
    if not state.active[hero_slot]:
        return 0.0, int(state.stacks[hero_slot])
    if len(live) == 1:
        return float(state.pot), int(state.stacks[hero_slot])
    _check_no_unequal_allin_showdown(state)
    rows = np.array(
        [list(state.board) + list(cards_by_slot[slot]) for slot in live],
        dtype=np.int32,
    )
    strengths = evaluate(rows)
    best = max(int(v) for v in strengths)
    winners = [slot for slot, strength in zip(live, strengths) if int(strength) == best]
    award = float(state.pot) / float(len(winners)) if hero_slot in winners else 0.0
    return award, int(state.stacks[hero_slot])


def rollout_action_value(
    root: Any,
    action: int,
    hero_slot: int,
    cards_by_slot: Sequence[Tuple[int, int]],
    lookup: Any,
    rng: np.random.Generator,
    evaluate: Any,
) -> float:
    award, final_stack = play_out_blueprint(
        root.advance(int(action)), hero_slot, cards_by_slot, lookup, rng, evaluate
    )
    return award - (int(root.stacks[hero_slot]) - final_stack)


def rollout_values_multiway(
    root: Any,
    reaches_by_slot: Mapping[int, np.ndarray],
    n_actions: int,
    samples: int,
    lookup: Any,
    rng: np.random.Generator,
    evaluate: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    hero_slot = int(root.to_act_slot)
    slots = _entry_slots(root)
    visible = _visible_board(root)
    hands = np.zeros(int(samples), dtype=np.int64)
    values = np.zeros((int(samples), int(n_actions)), dtype=np.float64)
    for row in range(int(samples)):
        draw = sample_joint_combos(rng, reaches_by_slot, slots, visible)
        dead = list(visible)
        for combo in draw.values():
            dead.extend(COMBOS[int(combo)])
        runout = (
            sample_runouts(rng, dead, 5 - len(visible), 1)[0].tolist()
            if len(visible) < 5 else []
        )
        rolled = replace(root, board=tuple(visible + runout))
        cards_by_slot = [(0, 1)] * root.n_slots
        for slot, combo in draw.items():
            cards_by_slot[int(slot)] = COMBOS[int(combo)]
        seed = int(rng.integers(0, 2 ** 31 - 1))
        hands[row] = int(draw[hero_slot])
        for action in range(int(n_actions)):
            values[row, action] = rollout_action_value(
                rolled, action, hero_slot, cards_by_slot, lookup,
                np.random.default_rng(seed + action), evaluate,
            )
    return hands, values


def bucket_of_equity(equity: float) -> str:
    if equity < WEAK_EQUITY:
        return "weak"
    if equity > STRONG_EQUITY:
        return "strong"
    return "mid"


def equity_bucket_for_combo(
    combo: int,
    board: Sequence[int],
    opp_weights: np.ndarray,
    rng: np.random.Generator,
    evaluate: Any,
    runouts: int = 100,
) -> str:
    c0, c1 = COMBOS[int(combo)]
    runs = sample_runouts(rng, list(board) + [c0, c1], 5 - len(board), runouts)
    equities = equity_vs_combos((c0, c1), board, runs, evaluate)
    weights = np.where(live_mask(list(board) + [c0, c1]), opp_weights, 0.0)
    total = float(weights.sum())
    equity = float(np.dot(weights, equities) / total) if total > 0.0 else 0.5
    return bucket_of_equity(equity)


def action_frequency(table: np.ndarray, hands: np.ndarray, kinds: Sequence[str]) -> Dict[str, float]:
    rows = np.asarray(table, dtype=np.float64)[hands]
    fold_cols = [i for i, kind in enumerate(kinds) if kind == "fold"]
    aggressive_cols = [i for i, kind in enumerate(kinds) if kind == "raise"]
    return {
        "fold": float(rows[:, fold_cols].sum(axis=1).mean()) if fold_cols else 0.0,
        "bet_raise": (
            float(rows[:, aggressive_cols].sum(axis=1).mean())
            if aggressive_cols else 0.0
        ),
    }


def score_tables_for_spot(
    spot: Mapping[str, Any],
    arm_tables: Mapping[str, np.ndarray],
    lookup: Any,
    evaluate: Any,
    rng: np.random.Generator,
    samples: int,
) -> Dict[str, Any]:
    state, node_decisions, _start_decisions = replay_spot(spot)
    if state.to_act_slot is None:
        raise ValueError("spot replay is terminal")
    hero_slot = int(state.to_act_slot)
    seats_by_slot = {slot: int(state.seat_of_slot[slot]) for slot in _entry_slots(state)}
    reach_by_seat = node_reaches(state, node_decisions, lookup)
    reaches_by_slot = {
        slot: reach_by_seat[seat] for slot, seat in seats_by_slot.items()
    }
    n_actions = int(state.num_legal_actions())
    hands, values = rollout_values_multiway(
        state, reaches_by_slot, n_actions, int(samples), lookup, rng, evaluate
    )
    blueprint = lookup_table(lookup, state)
    kinds = action_kinds(state)
    pot = float(state.pot)
    opp_slots = [slot for slot in _entry_slots(state) if slot != hero_slot]
    first_opp_weights = reaches_by_slot[opp_slots[0]]
    visible = _visible_board(state)
    runouts = 1 if len(visible) == 5 else 40
    bucket_of_combo = {
        int(combo): equity_bucket_for_combo(
            int(combo), visible, first_opp_weights, rng, evaluate, runouts=runouts,
        )
        for combo in np.unique(hands)
    }
    buckets = np.array([bucket_of_combo[int(combo)] for combo in hands])
    arms: Dict[str, Any] = {}
    for arm, table in arm_tables.items():
        table = np.asarray(table, dtype=np.float64)
        if table.shape != blueprint.shape:
            raise ValueError(
                f"{arm} table shape {table.shape} != blueprint {blueprint.shape}"
            )
        diff = table[hands] - blueprint[hands]
        adv = (diff * values).sum(axis=1)
        pct = 100.0 * adv / pot if pot > 0.0 else adv * float("nan")
        arm_freq = action_frequency(table, hands, kinds)
        bp_freq = action_frequency(blueprint, hands, kinds)
        by_bucket: Dict[str, Any] = {}
        for bucket in ("weak", "mid", "strong"):
            mask = buckets == bucket
            if not bool(mask.any()):
                continue
            by_bucket[bucket] = {
                "share": float(mask.mean()),
                "advantage_chips": mean_se(adv[mask].tolist()),
                "advantage_pct_pot": mean_se(pct[mask].tolist()),
                "arm_freq": action_frequency(table, hands[mask], kinds),
                "blueprint_freq": action_frequency(blueprint, hands[mask], kinds),
            }
        arms[arm] = {
            "advantage_chips": mean_se(adv.tolist()),
            "advantage_pct_pot": mean_se(pct.tolist()),
            "arm_freq": arm_freq,
            "blueprint_freq": bp_freq,
            "by_bucket": by_bucket,
        }
    return {
        "schema": "decision_audit_score_v1",
        "spot_id": spot["spot_id"],
        "cell": spot["cell"],
        "pot": pot,
        "hero_seat": int(spot["hero_seat"]),
        "inclusion_probability": float(spot["inclusion_probability"]),
        "samples": int(samples),
        "arms": arms,
    }


def _cell_rows(scores: Sequence[Mapping[str, Any]], cell: str, arms: Sequence[str]) -> List[Mapping[str, Any]]:
    return [
        row for row in scores
        if str(row["cell"]["key"]) == cell and all(arm in row["arms"] for arm in arms)
    ]


def _bb100(values: Sequence[float], rows: Sequence[Mapping[str, Any]], deals: int) -> Dict[str, Optional[float]]:
    """Per-hand chip contribution of a cell (bb/100 at 100 chips per bb), with its SE.

    Inside a cell every spot has the same inclusion probability p = K/n_c, so
    sum(adv / p) / (6 deals) = mean(adv) * n_c / (6 deals); the SE scales the same way.
    """
    if not rows:
        return {"mean": 0.0, "se": 0.0, "n": 0}
    n_c = float(np.mean([1.0 / float(row["inclusion_probability"]) for row in rows])) * len(rows)
    scale = n_c / (6.0 * float(deals))
    stats = mean_se(values)
    se = None if stats["se"] is None else float(stats["se"]) * scale
    return {"mean": float(stats["mean"]) * scale, "se": se, "n": len(rows)}


def _combine(parts: Sequence[Mapping[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    mean = float(sum(float(p["mean"] or 0.0) for p in parts))
    ses = [p["se"] for p in parts if p["n"]]
    se = None if any(v is None for v in ses) else float(np.sqrt(sum(float(v) ** 2 for v in ses)))
    return {"mean": mean, "se": se, "n": int(sum(int(p["n"]) for p in parts))}


def _adv(row: Mapping[str, Any], arm: str) -> float:
    return float(row["arms"][arm]["advantage_chips"]["mean"])


def summarize_report(
    scores: Sequence[Mapping[str, Any]],
    deals: int,
) -> Dict[str, Any]:
    arms = sorted({arm for row in scores for arm in row["arms"]})
    cells = sorted({str(row["cell"]["key"]) for row in scores})
    out: Dict[str, Any] = {"deals": int(deals), "spots": len(scores), "arms": {}}
    for arm in arms:
        cell_out: Dict[str, Any] = {}
        for cell in cells:
            rows = _cell_rows(scores, cell, [arm])
            fallbacks = sum(
                1 for row in rows
                if (row.get("table_sources") or {}).get(arm, {}).get("fallback")
            )
            cell_out[cell] = {
                "spots": len(rows),
                "fallbacks": fallbacks,
                "advantage_chips": mean_se([_adv(row, arm) for row in rows]),
                "advantage_pct_pot": mean_se([
                    float(row["arms"][arm]["advantage_pct_pot"]["mean"]) for row in rows
                ]),
                "bb100": _bb100([_adv(row, arm) for row in rows], rows, deals),
                "behaviour": {
                    key: mean_se([row["arms"][arm][src][freq] for row in rows])
                    for key, src, freq in (
                        ("arm_bet_raise", "arm_freq", "bet_raise"),
                        ("arm_fold", "arm_freq", "fold"),
                        ("blueprint_bet_raise", "blueprint_freq", "bet_raise"),
                        ("blueprint_fold", "blueprint_freq", "fold"),
                    )
                },
            }
        out["arms"][arm] = {
            "cells": cell_out,
            "total_bb100": _combine([cell_out[c]["bb100"] for c in cells]),
        }
    out["paired"] = paired_arm_deltas(scores, arms, cells, deals)
    return out


def paired_arm_deltas(
    scores: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    cells: Sequence[str],
    deals: int,
) -> Dict[str, Any]:
    paired: Dict[str, Any] = {}
    for i, left in enumerate(arms):
        for right in arms[i + 1:]:
            by_cell: Dict[str, Any] = {}
            for cell in cells:
                rows = _cell_rows(scores, cell, [left, right])
                by_cell[cell] = _bb100([_adv(row, left) - _adv(row, right) for row in rows], rows, deals)
            paired[f"{left}_minus_{right}"] = {
                "cells_bb100": by_cell,
                "total_bb100": _combine(list(by_cell.values())),
            }
    return paired


def _fmt(stat: Mapping[str, Optional[float]], digits: int = 2) -> str:
    if stat.get("mean") is None:
        return "n/a"
    se = stat.get("se")
    return f"{float(stat['mean']):+.{digits}f}" + ("" if se is None else f" +/- {float(se):.{digits}f}")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Decision audit report",
        "",
        f"Deals: {report['deals']}; scored spots: {report['spots']}.",
        "Advantage = arm mix minus blueprint mix at the node, blueprint continuation for everyone,",
        "true (blueprint-reach) ranges. bb/100 = per-hand contribution weighted by cell frequency.",
        "",
    ]
    for arm, arm_report in report["arms"].items():
        lines.extend([
            f"## {arm}  (total {_fmt(arm_report['total_bb100'])} bb/100)", "",
            "| Cell | Spots | Fallbacks | % pot | bb/100 | bet/raise arm vs bp | fold arm vs bp |",
            "|---|---:|---:|---:|---:|---|---|",
        ])
        for cell, row in arm_report["cells"].items():
            beh = row["behaviour"]
            lines.append(
                f"| {cell} | {row['spots']} | {row['fallbacks']} | {_fmt(row['advantage_pct_pot'])} | "
                f"{_fmt(row['bb100'])} | {_fmt(beh['arm_bet_raise'])} vs {_fmt(beh['blueprint_bet_raise'])} | "
                f"{_fmt(beh['arm_fold'])} vs {_fmt(beh['blueprint_fold'])} |"
            )
        lines.append("")
    lines.extend(["## Paired arm differences (bb/100)", "", "| Pair | Total | " + " | ".join(
        sorted({c for v in report["paired"].values() for c in v["cells_bb100"]})) + " |"])
    cells = sorted({c for v in report["paired"].values() for c in v["cells_bb100"]})
    lines.append("|---|---:|" + "---:|" * len(cells))
    for label, pair in report["paired"].items():
        lines.append(f"| {label} | {_fmt(pair['total_bb100'])} | " + " | ".join(
            _fmt(pair["cells_bb100"][c]) for c in cells) + " |")
    return "\n".join(lines) + "\n"
