"""Local best response over the engine's preflop and postflop states.

Two seats take part; every other seat folds when asked.  The policy under
test acts through ``decide(state, cards) -> (action, table)`` and answers
hypothetical nodes through ``table(state, cards)``, where a table is the
policy's action distribution for every one of the 1326 combos.  The exploiter
tracks the policy's range from those tables and scores its own actions with
``scripts.lbr_core``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from scripts.lbr_core import (
    Candidate,
    ComboRange,
    Evaluator,
    Outcome,
    candidate_ev,
    choose_action,
    equity_vs_combos,
    sample_runouts,
)

STREET_BOARD_LEN = {0: 0, 1: 3, 2: 4, 3: 5}
TableFn = Callable[[Any], np.ndarray]


@dataclass(frozen=True)
class ExploiterConfig:
    """Runout budgets for the exploiter's equity estimates."""

    runouts_preflop: int = 200
    runouts_flop: int = 100
    terminal_samples: int = 400

    def samples_for(self, visible: int) -> int:
        if visible == 0:
            return self.runouts_preflop
        if visible == 3:
            return self.runouts_flop
        return 10 ** 9  # turn and river enumerate exactly


# -- state adapters ---------------------------------------------------------


def is_preflop(state: Any) -> bool:
    return int(state.street) == 0


def to_act_seat(state: Any) -> Optional[int]:
    return state.to_act_seat()


def _category_kind(category: Any) -> str:
    name = str(getattr(category, "name", category)).upper()
    if "FOLD" in name:
        return "fold"
    if "CALL" in name or "CHECK" in name:
        return "call"
    return "raise"


def action_kinds(state: Any) -> List[str]:
    """Node-local kinds in action-index order: "fold" | "call" | "raise"."""
    if is_preflop(state):
        return ["call" if a.kind in ("check", "call") else a.kind for a in state.legal_actions()]
    return [_category_kind(c) for c in state.legal_action_categories()]


def call_index(state: Any) -> int:
    return action_kinds(state).index("call")


def seat_stack(state: Any, seat: int) -> int:
    if is_preflop(state):
        return int(state.stacks[seat])
    return int(state.stacks[list(state.seat_of_slot).index(seat)])


def seat_active(state: Any, seat: int) -> bool:
    if is_preflop(state):
        return bool(state.active[seat])
    slots = list(state.seat_of_slot)
    return seat in slots and bool(state.active[slots.index(seat)])


def visible_board(state: Any, board: Sequence[int]) -> List[int]:
    return [int(card) for card in board[: STREET_BOARD_LEN[int(state.street)]]]


def street_closed(before: Any, after: Any) -> bool:
    return bool(after.terminal) or int(after.street) != int(before.street)


def betting_closed_by_allin(state: Any) -> bool:
    """At most one live player still has chips, so no further decision exists."""
    if state.terminal:
        return False
    if is_preflop(state):
        live = [seat for seat in range(6) if state.active[seat]]
        with_chips = [seat for seat in live if state.stacks[seat] > 0]
    else:
        live = [slot for slot in range(state.n_slots) if state.active[slot]]
        with_chips = [slot for slot in live if state.stacks[slot] > 0]
    return len(live) >= 2 and len(with_chips) <= 1


def advance_forced(state: Any, folders: FrozenSet[int]) -> Any:
    """Apply every action that is not a decision: bystander folds, all-in checks."""
    while not state.terminal:
        seat = to_act_seat(state)
        kinds = action_kinds(state)
        if seat in folders:
            if "fold" not in kinds:
                raise ValueError(f"bystander seat {seat} cannot fold at {kinds}")
            state = state.advance(kinds.index("fold"))
        elif betting_closed_by_allin(state) or seat_stack(state, seat) <= 0:
            state = state.advance(kinds.index("call"))
        else:
            break
    return state


# -- exploiter --------------------------------------------------------------


def _after_exploiter_action(
    state: Any, action: int, kind: str, l_seat: int, p_seat: int,
    folders: FrozenSet[int], table_fn: TableFn,
) -> Candidate:
    l_stack = seat_stack(state, l_seat)
    after = advance_forced(state.advance(action), folders)
    if street_closed(state, after):
        return Candidate(action, kind, direct=Outcome("showdown", l_stack - seat_stack(after, l_seat), int(after.pot)))
    if to_act_seat(after) != p_seat:
        raise ValueError(f"expected the policy to act after the exploiter's {kind}")
    table = table_fn(after)
    outcomes: List[Outcome] = []
    for k, response in enumerate(action_kinds(after)):
        if response == "fold":
            outcomes.append(Outcome("p_fold", l_stack - seat_stack(after, l_seat), int(after.pot)))
            continue
        settled = advance_forced(after.advance(k), folders)
        if not street_closed(after, settled) and to_act_seat(settled) == l_seat:
            settled = advance_forced(settled.advance(call_index(settled)), folders)
        outcomes.append(Outcome("showdown", l_stack - seat_stack(settled, l_seat), int(settled.pot)))
    if table.shape[1] != len(outcomes):
        raise ValueError(f"policy table has {table.shape[1]} actions, node has {len(outcomes)}")
    return Candidate(action, kind, responses=table, outcomes=tuple(outcomes))


def build_candidates(
    state: Any, l_seat: int, p_seat: int, folders: FrozenSet[int], table_fn: TableFn,
) -> List[Candidate]:
    candidates = []
    for action, kind in enumerate(action_kinds(state)):
        if kind == "fold":
            candidates.append(Candidate(action, kind, direct=Outcome("fold", 0, int(state.pot))))
        else:
            candidates.append(_after_exploiter_action(state, action, kind, l_seat, p_seat, folders, table_fn))
    return candidates


def exploiter_decide(
    state: Any, l_cards: Tuple[int, int], l_seat: int, p_seat: int, folders: FrozenSet[int],
    policy_range: ComboRange, board: Sequence[int], table_fn: TableFn,
    evaluate: Evaluator, rng: np.random.Generator, config: ExploiterConfig,
) -> Tuple[int, Dict[str, Any]]:
    board_now = visible_board(state, board)
    dead = list(l_cards) + board_now
    runouts = sample_runouts(rng, dead, 5 - len(board_now), config.samples_for(len(board_now)))
    equity = equity_vs_combos(l_cards, board_now, runouts, evaluate)
    weights = policy_range.normalized(dead)
    candidates = build_candidates(state, l_seat, p_seat, folders, table_fn)
    evs = [candidate_ev(c, weights, equity) for c in candidates]
    chosen = choose_action(candidates, evs)
    info = {
        "street": int(state.street), "kinds": [c.kind for c in candidates], "evs": [float(ev) for ev in evs],
        "chosen": chosen.kind, "action": int(chosen.action),
        "policy_effective_rows": policy_range.effective_rows(dead),
    }
    return int(chosen.action), info


# -- hand driver ------------------------------------------------------------


@dataclass(frozen=True)
class Deal:
    hole_cards: Tuple[Tuple[int, int], ...]
    board: Tuple[int, ...]


def deal_from_rng(rng: np.random.Generator) -> Deal:
    """Board first, then one two-card packet per seat (run_match's convention)."""
    cards = [int(card) for card in rng.permutation(52)]
    board = tuple(cards[:5])
    holes = tuple((cards[5 + 2 * seat], cards[6 + 2 * seat]) for seat in range(6))
    return Deal(hole_cards=holes, board=board)


@dataclass
class HandTrace:
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    closure_board_len: Optional[int] = None
    policy_range: ComboRange = field(default_factory=lambda: ComboRange.uniform(()))


def _drive(
    state: Any, deal: Deal, p_seat: int, l_seat: int, folders: FrozenSet[int], policy: Any,
    trace: HandTrace, evaluate: Evaluator, rng: np.random.Generator, config: ExploiterConfig,
) -> Any:
    p_cards, l_cards = deal.hole_cards[p_seat], deal.hole_cards[l_seat]
    set_own_range = getattr(policy, "set_own_range", None)

    def table_fn(node_state: Any) -> np.ndarray:
        if set_own_range is not None:
            set_own_range(trace.policy_range.weights)
        return policy.table(node_state, p_cards)

    while True:
        state = advance_forced(state, folders)
        if state.terminal:
            return state
        seat = to_act_seat(state)
        acting_street = int(state.street)
        if seat == p_seat:
            if set_own_range is not None:
                set_own_range(trace.policy_range.weights)
            action, table = policy.decide(state, p_cards)
            trace.policy_range = trace.policy_range.conditioned(table[:, int(action)])
            trace.decisions.append({"street": acting_street, "actor": "policy", "action": int(action)})
        elif seat == l_seat:
            action, info = exploiter_decide(
                state, l_cards, l_seat, p_seat, folders, trace.policy_range, deal.board, table_fn, evaluate, rng, config,
            )
            trace.decisions.append({"actor": "exploiter", **info})
        else:
            raise ValueError(f"unexpected seat {seat} to act")
        after = state.advance(action)
        if trace.closure_board_len is None and betting_closed_by_allin(after):
            trace.closure_board_len = STREET_BOARD_LEN[acting_street]
        state = after


def _showdown_values(
    state: Any, deal: Deal, p_seat: int, l_seat: int, trace: HandTrace,
    evaluate: Evaluator, rng: np.random.Generator, config: ExploiterConfig, l_contrib: int,
) -> Tuple[float, float]:
    pot = int(state.pot)
    l_cards, p_cards = deal.hole_cards[l_seat], deal.hole_cards[p_seat]
    strengths = evaluate(np.array([list(deal.board) + list(l_cards), list(deal.board) + list(p_cards)], dtype=np.int32))
    share = 1.0 if strengths[0] > strengths[1] else 0.5 if strengths[0] == strengths[1] else 0.0
    realized = share * pot - l_contrib
    board_len = 5 if trace.closure_board_len is None else int(trace.closure_board_len)
    board_now = list(deal.board[:board_len])
    runouts = sample_runouts(rng, list(l_cards) + board_now, 5 - board_len, config.terminal_samples)
    equity = equity_vs_combos(l_cards, board_now, runouts, evaluate)
    weights = trace.policy_range.normalized(list(l_cards) + board_now)
    expected = float(np.dot(weights, equity)) * pot - l_contrib
    return realized, expected


def play_hand(
    deal: Deal, p_seat: int, l_seat: int, policy: Any, evaluate: Evaluator,
    rng: np.random.Generator, config: ExploiterConfig,
) -> Dict[str, Any]:
    """One hand: the policy in ``p_seat``, the exploiter in ``l_seat``, others fold."""
    from app.engine.agent.game_state import INITIAL_STACK, new_postflop_state
    from app.engine.agent.preflop_play import new_preflop_state

    started = time.perf_counter()
    folders = frozenset(range(6)) - {p_seat, l_seat}
    trace = HandTrace(policy_range=ComboRange.uniform(deal.hole_cards[l_seat]))
    pre = _drive(new_preflop_state(), deal, p_seat, l_seat, folders, policy, trace, evaluate, rng, config)
    state = pre
    if not pre.uncontested:
        post = new_postflop_state(
            seats=list(pre.live_seat_names), preflop_prefix=list(pre.preflop_prefix),
            board=list(deal.board), pot=int(pre.pot), stacks=list(pre.live_stacks),
        )
        state = _drive(post, deal, p_seat, l_seat, folders, policy, trace, evaluate, rng, config)
    l_contrib = INITIAL_STACK - seat_stack(state, l_seat)
    pot = int(state.pot)
    if not seat_active(state, l_seat):
        terminal, realized, expected = "fold", -float(l_contrib), -float(l_contrib)
    elif not seat_active(state, p_seat):
        terminal, realized, expected = "p_fold", float(pot - l_contrib), float(pot - l_contrib)
    else:
        terminal = "showdown"
        realized, expected = _showdown_values(state, deal, p_seat, l_seat, trace, evaluate, rng, config, l_contrib)
    return {
        "p_seat": p_seat, "l_seat": l_seat,
        "p_cards": list(deal.hole_cards[p_seat]), "l_cards": list(deal.hole_cards[l_seat]),
        "board": list(deal.board), "preflop_prefix": list(pre.preflop_prefix),
        "postflop_history": list(getattr(state, "postflop_history", ())),
        "terminal": terminal, "pot": pot, "l_contrib": int(l_contrib),
        "closure_board_len": trace.closure_board_len,
        "l_net_realized": float(realized), "l_net_expected": float(expected),
        "policy_range_mass": trace.policy_range.mass(),
        "decisions": trace.decisions, "wall_s": time.perf_counter() - started,
    }
