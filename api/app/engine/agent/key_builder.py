"""GameState -> blueprint node key (plan §2, §3.2).

The key is `(action_history: List[int], player: int, street: int)`:
  - action_history = preflop_prefix + postflop_history (one index per action).
  - player = the FULL SEAT (0-5) of the to-act slot. NOT the compact slot —
    that confusion was last session's wrong-seat bug (the canary).
  - street = 1/2/3 (flop/turn/river).

This module is a THIN assembler. It does NOT decide who-acts-next or when a
street advances — that lives in GameState (which mirrors the capture-verified
forward-from-bettor / round-closure semantics). key_builder only reads the
already-correct GameState. The legal action SET ordering (what each index
means) comes from GameState.legal_action_categories ->
rollout.classify_postflop_actions; we never hand-roll it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.engine.agent.game_state import GameState


@dataclass(frozen=True)
class NodeKey:
    """An immutable blueprint lookup key.

    action_history is a fresh list (not a shared reference) so callers can't
    mutate GameState internals through it.
    """

    action_history: List[int]
    player: int
    street: int

    def as_tuple(self) -> tuple:
        return (tuple(self.action_history), self.player, self.street)


def build_key(state: GameState) -> NodeKey:
    """Construct the blueprint node key for the player to act in `state`.

    Raises:
        ValueError: if `state` is terminal (no one to act) — there is no
            decision node to key.
    """
    seat = state.to_act_seat()
    if seat is None:
        raise ValueError(
            "build_key called on a terminal GameState (no player to act)"
        )
    action_history = list(state.preflop_prefix) + list(state.postflop_history)
    return NodeKey(
        action_history=action_history,
        player=seat,
        street=state.street,
    )
