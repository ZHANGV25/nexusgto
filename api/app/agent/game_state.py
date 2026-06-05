"""Immutable GameState + pure betting-round advance (plan §2, §3.1).

GameState models one hand from a postflop street onward. It is FROZEN; the
only way to progress is `advance(action_idx)`, which returns a NEW state. The
betting semantics MIRROR poker-solver/python/rollout.py::_simulate_one — the
same SB-first acting order, bet-reopens-action, round-closure-advances-street
rules that produced the capture-verified blueprint keys. key_builder reads
GameState to construct the node key; it does NOT re-derive who-acts-next.

The slot<->seat seam (the root cause of last session's wrong-seat bug):
  - COMPACT SLOT: 0..n_live-1, position in this hand's acting order.
  - FULL SEAT: 0..5 (SB0,BB1,UTG2,MP3,CO4,BTN5), the blueprint's player id.
  `seat_of_slot[slot]` maps one to the other. who-acts-next iterates SLOTS;
  the blueprint key's `player` is the SEAT. Confusing them queries the wrong
  node (the canary tests pin this).

Scope (v1): single-street-at-a-time betting with the 8.05B postflop vocab
(one 0.75-pot bet size; facing-bet = fold/call/raise). All-in / side-pot /
multi-size raise trees beyond POSTFLOP_MAX_RAISES are out of scope (plan §6).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

# Action-set COUNT comes from the blueprint's trained vocab (metadata-driven,
# see action_vocab); the category SEMANTICS (fold/call/raise) reuse the
# solver's enum. The seat / who-acts-next logic lives below, unchanged.
from app.agent.action_vocab import (
    DEFAULT_ADD_EXPLICIT_ALLIN,
    DEFAULT_POSTFLOP_BET_SIZES,
    DEFAULT_SUBSEQUENT_BET_SIZES,
    legal_categories,
)

try:
    from rollout import ActionCategory  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only on bare checkouts
    from python.rollout import ActionCategory  # type: ignore[no-redef]


# Blueprint seat encoding (capture-verified — see agent_node_key_encoding memo).
FULL_SEAT: Dict[str, int] = {
    "SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5,
}
SEAT_NAME: Dict[int, str] = {v: k for k, v in FULL_SEAT.items()}

# street ints: 0=pre,1=flop,2=turn,3=river.
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3
_MAX_STREET = STREET_RIVER


@dataclass(frozen=True)
class GameState:
    """Immutable state of one hand from a postflop street onward.

    Fields (all immutable — tuples/ints, no lists):
        seat_of_slot: compact-slot -> full-seat (0-5). Ordered SB-first.
        stacks: per-slot remaining chips (entering this hand's postflop play).
        active: per-slot bool; False = folded this hand.
        pot: chips in the pot.
        board: tuple of card ints for the current street (3/4/5).
        street: 1/2/3 (flop/turn/river).
        preflop_prefix: the preflop action-index sequence (given; see plan §6
            and the preflop_constructor for how it's built).
        postflop_history: action indices taken across ALL postflop streets so
            far (one per action), in the order they were taken. This is the
            postflop tail appended to preflop_prefix to form action_history.
        to_act_slot: compact slot whose turn it is, or None if the hand/street
            is terminal (one player left or river betting complete).
        current_bet: chips the current outstanding bet requires (street-local).
        street_bets: per-slot chips committed on the CURRENT street.
        acted: per-slot bool; whether the slot has acted since the last
            bet/raise reopened the round.
        raises_this_street: number of bets/raises so far this street.
        betting_round_closed_count: how many postflop betting rounds have
            fully closed (drives street advance; flop=0 closed at entry).
    """

    seat_of_slot: Tuple[int, ...]
    stacks: Tuple[int, ...]
    active: Tuple[bool, ...]
    pot: int
    board: Tuple[int, ...]
    street: int
    preflop_prefix: Tuple[int, ...]
    postflop_history: Tuple[int, ...] = ()
    to_act_slot: Optional[int] = 0
    current_bet: int = 0
    street_bets: Tuple[int, ...] = ()
    acted: Tuple[bool, ...] = ()
    raises_this_street: int = 0
    betting_round_closed_count: int = 0

    # Blueprint-trained postflop vocab (metadata-driven). Governs the legal
    # action COUNT only. Defaults target the 8.05B (v1's blueprint);
    # LookupSource asserts these match the loaded blueprint's metadata.
    postflop_bet_sizes: Tuple[float, ...] = DEFAULT_POSTFLOP_BET_SIZES
    subsequent_bet_sizes: Tuple[float, ...] = DEFAULT_SUBSEQUENT_BET_SIZES
    add_explicit_allin: bool = DEFAULT_ADD_EXPLICIT_ALLIN

    # ── Derived / read-only helpers ──────────────────────────────────────

    @property
    def n_slots(self) -> int:
        return len(self.seat_of_slot)

    @property
    def n_live(self) -> int:
        return sum(1 for a in self.active if a)

    @property
    def facing_bet(self) -> bool:
        """Whether the to-act player faces an outstanding bet to call."""
        if self.to_act_slot is None:
            return False
        return self.current_bet > self.street_bets[self.to_act_slot]

    @property
    def terminal(self) -> bool:
        """Hand can no longer be acted on at this street."""
        return self.to_act_slot is None

    def legal_action_categories(self) -> List[ActionCategory]:
        """The legal action set for the to-act player.

        The action COUNT is driven by the blueprint's trained vocab (this
        state's postflop_bet_sizes/subsequent_bet_sizes/add_explicit_allin),
        NOT rollout's hardcoded (0.5,1.0) — the 8.05B trained with [0.75] so
        its no-bet node has 2 actions, not 3. The index ORDER + category
        semantics still match training (action_vocab mirrors the structure).
        """
        return legal_categories(
            facing_bet=self.facing_bet,
            raises_so_far=self.raises_this_street,
            postflop_bet_sizes=self.postflop_bet_sizes,
            subsequent_bet_sizes=self.subsequent_bet_sizes,
            add_explicit_allin=self.add_explicit_allin,
        )

    def num_legal_actions(self) -> int:
        return len(self.legal_action_categories())

    def to_act_seat(self) -> Optional[int]:
        if self.to_act_slot is None:
            return None
        return self.seat_of_slot[self.to_act_slot]

    # ── The single mutator (returns a NEW state) ─────────────────────────

    def advance(self, action_idx: int) -> "GameState":
        """Apply the to-act player's action; return the next GameState.

        action_idx indexes into legal_action_categories() (the same order the
        blueprint key uses). Pure: builds and returns a new frozen GameState.
        """
        if self.to_act_slot is None:
            raise ValueError("Cannot advance a terminal state")
        cats = self.legal_action_categories()
        if action_idx < 0 or action_idx >= len(cats):
            raise ValueError(
                f"action_idx {action_idx} out of range [0,{len(cats)}) for "
                f"facing_bet={self.facing_bet} raises={self.raises_this_street}"
            )

        slot = self.to_act_slot
        cat = cats[action_idx]

        active = list(self.active)
        acted = list(self.acted)
        street_bets = list(self.street_bets)
        current_bet = self.current_bet
        raises = self.raises_this_street
        pot = self.pot

        acted[slot] = True

        if cat == ActionCategory.FOLD:
            active[slot] = False
        elif cat == ActionCategory.CALL:
            # CALL covers both "call a bet" and "check". For a real call,
            # move chips to match the outstanding bet (tracked at chip level
            # for pot accounting; exact sizing is the resolver's concern, not
            # the key's — the key only needs the action index).
            to_match = current_bet - street_bets[slot]
            if to_match > 0:
                street_bets[slot] += to_match
                pot += to_match
        else:  # RAISE (covers bet and raise)
            # A bet/raise opens a new round. We do NOT reset other players'
            # `acted` flags — the `street_bets[q] < current_bet` predicate in
            # _next_to_act makes everyone who hasn't matched owe action
            # automatically (capture-verified: forward-from-bettor order, the
            # blueprint MISSES rollout's scan-from-slot0 order).
            raises += 1
            # Size is not needed for the key; advance current_bet by a
            # nominal unit so facing_bet flips for the others. Pot/stack
            # accounting for play uses real sizes via action_translation;
            # GameState only needs the betting STRUCTURE correct.
            current_bet = current_bet + 1
            street_bets[slot] = current_bet

        new_history = self.postflop_history + (action_idx,)

        # Determine next to act / round closure. Scan FORWARD from the slot
        # after the one that just acted (real-poker position order), which is
        # what the blueprint encodes.
        next_slot = self._next_to_act(
            active, acted, street_bets, current_bet, start_after=slot,
        )

        if next_slot is not None or self._one_left(active):
            # Either there is a next actor, or the hand collapsed to one
            # player (terminal). No street advance yet.
            return replace(
                self,
                active=tuple(active),
                acted=tuple(acted),
                street_bets=tuple(street_bets),
                current_bet=current_bet,
                raises_this_street=raises,
                pot=pot,
                postflop_history=new_history,
                to_act_slot=(None if self._one_left(active) else next_slot),
            )

        # No next actor AND >1 player live => the betting round CLOSED.
        return self._close_round(
            active=active, pot=pot, history=new_history,
            closed_count=self.betting_round_closed_count + 1,
        )

    # ── Internal betting mechanics (mirror rollout._simulate_one) ────────

    def _one_left(self, active: List[bool]) -> bool:
        return sum(1 for a in active if a) <= 1

    def _next_to_act(
        self,
        active: List[bool],
        acted: List[bool],
        street_bets: List[int],
        current_bet: int,
        start_after: int,
    ) -> Optional[int]:
        """Next slot that still owes an action, scanning FORWARD from
        `start_after` (i.e. starting at start_after+1, wrapping around).

        A slot owes action if it is active AND (hasn't acted yet OR hasn't
        matched the current bet). Forward-from-last-actor is the order the
        blueprint encodes (capture-verified) — after a bet, action proceeds
        in position order from the aggressor and only then wraps back to
        already-checked players. Returns None when no one owes action (round
        complete).
        """
        n = self.n_slots
        for step in range(1, n + 1):
            s = (start_after + step) % n
            if not active[s]:
                continue
            if (not acted[s]) or (street_bets[s] < current_bet):
                return s
        return None

    def _close_round(
        self,
        active: List[bool],
        pot: int,
        history: Tuple[int, ...],
        closed_count: int,
    ) -> "GameState":
        """Advance to the next street (or terminal if river closed)."""
        next_street = self.street + 1
        if next_street > _MAX_STREET:
            # River betting closed -> terminal showdown.
            return replace(
                self,
                active=tuple(active),
                pot=pot,
                postflop_history=history,
                to_act_slot=None,
                betting_round_closed_count=closed_count,
            )

        n_slots = self.n_slots
        first = self._first_live_slot(active)
        return replace(
            self,
            active=tuple(active),
            pot=pot,
            street=next_street,
            postflop_history=history,
            to_act_slot=first,
            current_bet=0,
            street_bets=tuple([0] * n_slots),
            acted=tuple([False] * n_slots),
            raises_this_street=0,
            betting_round_closed_count=closed_count,
        )

    def _first_live_slot(self, active: List[bool]) -> Optional[int]:
        for s in range(self.n_slots):
            if active[s]:
                return s
        return None


# ── Factory ───────────────────────────────────────────────────────────────


def new_postflop_state(
    seats: List[str],
    preflop_prefix: List[int],
    board: List[int],
    pot: int,
    stacks: List[int],
    street: int = STREET_FLOP,
    postflop_bet_sizes: Tuple[float, ...] = DEFAULT_POSTFLOP_BET_SIZES,
    subsequent_bet_sizes: Tuple[float, ...] = DEFAULT_SUBSEQUENT_BET_SIZES,
    add_explicit_allin: bool = DEFAULT_ADD_EXPLICIT_ALLIN,
) -> GameState:
    """Build a fresh GameState entering a postflop street.

    `seats` are position names of the players still in the hand (any order);
    they're reordered SB-first into compact slots (the postflop acting order).
    `preflop_prefix` is the given preflop action-index sequence.

    Stacks are aligned to `seats` BEFORE reordering: stacks[i] belongs to
    seats[i]. They're permuted to slot order internally.
    """
    if len(seats) != len(stacks):
        raise ValueError(
            f"seats ({len(seats)}) and stacks ({len(stacks)}) length mismatch"
        )
    if len(seats) < 2:
        raise ValueError("need at least 2 players for a postflop state")
    for p in seats:
        if p not in FULL_SEAT:
            raise ValueError(f"unknown position {p!r}; expected one of {list(FULL_SEAT)}")

    # Order players SB-first by full seat => compact slot order.
    order = sorted(range(len(seats)), key=lambda i: FULL_SEAT[seats[i]])
    seat_of_slot = tuple(FULL_SEAT[seats[i]] for i in order)
    slot_stacks = tuple(int(stacks[i]) for i in order)
    n = len(order)

    return GameState(
        seat_of_slot=seat_of_slot,
        stacks=slot_stacks,
        active=tuple([True] * n),
        pot=int(pot),
        board=tuple(board),
        street=int(street),
        preflop_prefix=tuple(preflop_prefix),
        postflop_history=(),
        to_act_slot=0,
        current_bet=0,
        street_bets=tuple([0] * n),
        acted=tuple([False] * n),
        raises_this_street=0,
        betting_round_closed_count=0,
        postflop_bet_sizes=tuple(postflop_bet_sizes),
        subsequent_bet_sizes=tuple(subsequent_bet_sizes),
        add_explicit_allin=bool(add_explicit_allin),
    )
