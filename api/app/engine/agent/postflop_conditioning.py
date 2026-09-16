"""Condition live re-solve ranges on the postflop actions already taken.

Research flag (2026-09-16, SCOREBOARD_RESULTS section 32). The re-solve seeds
every seat's range from the preflop strategy only (``live_resolve_ranges``), so
a turn or river root is solved against each player's whole preflop range. With
``NEXUS_RANGE_POSTFLOP_CONDITIONING=1`` each entry player's grid row weight is
multiplied by the blueprint's probability of the actions that player's seat
took on the earlier postflop streets: the hand's line is replayed from flop
entry through the shared transition and the blueprint lookup is read at each
of that seat's decisions for the row's combo. Exact for an opponent who plays
the blueprint; the hero is conditioned the same way, a stand-in for its own
search reach. The current street's actions are not conditioned on: the solved
tree replays them itself. DEFAULT OFF.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Sequence, Tuple

ENV = "NEXUS_RANGE_POSTFLOP_CONDITIONING"
Decision = Tuple[int, Any, int]  # (compact slot, state before the action, action index)
Row = Tuple[int, int, float]


def enabled() -> bool:
    return os.environ.get(ENV, "")[:1] in ("1", "t", "T", "y", "Y")


def earlier_street_decisions(state: Any) -> List[Decision]:
    """Every postflop decision taken on streets before ``state.street``.

    Replays the hand from flop entry (chips from the shared preflop
    transition) through the recorded postflop history, stopping before the
    current street's actions, and checks the replay lands on the same street
    root the state describes.
    """
    from app.engine.agent.game_state import new_postflop_state
    from app.engine.agent.preflop_play import replay_preflop

    n_current = len(state.street_history)
    history = list(state.postflop_history)
    earlier = history[: len(history) - n_current] if n_current else history
    if not earlier:
        return []
    pre = replay_preflop(list(state.preflop_prefix), require_terminal=True)
    replay = new_postflop_state(
        seats=list(pre.live_seat_names),
        preflop_prefix=list(state.preflop_prefix),
        board=list(state.board),
        pot=int(pre.pot),
        stacks=list(pre.live_stacks),
        true_hu=bool(state.true_hu),
        postflop_bet_sizes=tuple(state.postflop_bet_sizes),
        subsequent_bet_sizes=tuple(state.subsequent_bet_sizes),
        add_explicit_allin=bool(state.add_explicit_allin),
    )
    if tuple(replay.seat_of_slot) != tuple(state.seat_of_slot):
        raise ValueError("replayed seats differ from the live state's slots")
    decisions: List[Decision] = []
    for action in earlier:
        slot = replay.to_act_slot
        if slot is None:
            raise ValueError("postflop history runs past a terminal state")
        decisions.append((int(slot), replay, int(action)))
        replay = replay.advance(int(action))
    if int(replay.street) != int(state.street) or tuple(replay.street_history) != ():
        raise ValueError(
            f"replay reached street {replay.street} with {len(replay.street_history)} "
            f"actions; expected the root of street {state.street}"
        )
    return decisions


def condition_player_ranges(
    decisions: Sequence[Decision],
    player_ranges: Sequence[Sequence[Row]],
    entry_slots: Sequence[int],
    get_mix: Callable[[Any, Tuple[int, int]], Any],
) -> Tuple[List[List[Row]], Dict[str, Any]]:
    """Multiply each player's row weights by the blueprint's probability of the
    earlier-street actions its seat took. A player whose conditioned range has
    no mass keeps its unconditioned range (disclosed in the stats)."""
    stats: Dict[str, Any] = {
        "decisions": len(decisions), "lookups": 0, "uniform_rows": 0,
        "players_conditioned": 0, "players_zero_mass": 0, "mass_kept": [],
    }
    out: List[List[Row]] = []
    for player, slot in enumerate(entry_slots):
        rows = list(player_ranges[player])
        mine = [d for d in decisions if d[0] == int(slot)]
        if not mine:
            out.append([(int(c0), int(c1), float(w)) for c0, c1, w in rows])
            continue
        factors = [1.0] * len(rows)
        for _slot, before, action in mine:
            for index, (c0, c1, weight) in enumerate(rows):
                if float(weight) <= 0.0 or factors[index] == 0.0:
                    continue
                mix = get_mix(before, (int(c0), int(c1)))
                stats["lookups"] += 1
                stats["uniform_rows"] += int(bool(getattr(mix, "is_uniform_fallback", False)))
                factors[index] *= float(mix.freqs.get(int(action), 0.0))
        conditioned = [(int(c0), int(c1), float(w) * factors[i]) for i, (c0, c1, w) in enumerate(rows)]
        before_mass = sum(float(w) for _c0, _c1, w in rows)
        after_mass = sum(w for _c0, _c1, w in conditioned)
        if after_mass <= 0.0:
            stats["players_zero_mass"] += 1
            out.append([(int(c0), int(c1), float(w)) for c0, c1, w in rows])
            continue
        stats["players_conditioned"] += 1
        stats["mass_kept"].append(after_mass / before_mass if before_mass > 0.0 else 0.0)
        out.append(conditioned)
    return out, stats
