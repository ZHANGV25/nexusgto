"""Trainer-exact 6-max preflop state, replay, and play driver.

This module stays deliberately independent of :mod:`game_state` so the same
transition code can drive live preflop play *and* validate a completed prefix
before it is handed to ``new_postflop_state``.

The read-only trainer oracle is
``poker-solver/src/blueprint/mccfr_blueprint.c`` (2026-07-11 checkout):

* lines 954-988: action order, float32 bet arithmetic, clamp/prune/dedup;
* lines 1335-1345: round closure (all live players acted and matched);
* lines 2168-2199: preflop tier selection and raise cap;
* lines 2297-2326: fold/call/raise state transitions and action reopening;
* lines 4362-4398: blinds and UTG,MP,CO,BTN,SB,BB acting order.

Action indices are node-local and exactly trainer ordered: facing a bet gives
``[fold, call, raises...]``; with nothing to call it is
``[check, raises...]``. A raise's ``amount`` is the chip increment paid by the
actor, not a raise-to total.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from numbers import Integral
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol, Sequence, Tuple

if TYPE_CHECKING:
    from app.engine.agent.settlement import SettlementResult


FULL_SEAT = {"SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5}
SEAT_NAME = {seat: name for name, seat in FULL_SEAT.items()}
PREFLOP_ORDER: Tuple[int, ...] = (2, 3, 4, 5, 0, 1)

DEFAULT_SMALL_BLIND = 50
DEFAULT_BIG_BLIND = 100
DEFAULT_INITIAL_STACK = 10_000
DEFAULT_PREFLOP_TIERS: Tuple[Tuple[float, ...], ...] = (
    (0.7, 1.0),
    (0.7, 1.0),
    (1.0,),
)
DEFAULT_PREFLOP_MAX_RAISES = 3
DEFAULT_PREFLOP_ADD_EXPLICIT_ALLIN = False


class PreflopError(ValueError):
    """Base class for invalid preflop rules, actions, or replays."""


class PreflopReplayError(PreflopError):
    """An action prefix cannot be traversed through the trainer tree."""


@dataclass(frozen=True)
class PreflopRules:
    """Immutable trainer vocabulary and chip constants for one preflop tree."""

    tiers: Tuple[Tuple[float, ...], ...] = DEFAULT_PREFLOP_TIERS
    max_raises: int = DEFAULT_PREFLOP_MAX_RAISES
    add_explicit_allin: bool = DEFAULT_PREFLOP_ADD_EXPLICIT_ALLIN
    small_blind: int = DEFAULT_SMALL_BLIND
    big_blind: int = DEFAULT_BIG_BLIND
    initial_stack: int = DEFAULT_INITIAL_STACK

    def __post_init__(self) -> None:
        tiers = tuple(tuple(float(frac) for frac in tier) for tier in self.tiers)
        object.__setattr__(self, "tiers", tiers)
        if not tiers or any(not tier for tier in tiers):
            raise PreflopError("preflop tiers must contain at least one size each")
        if any(frac <= 0.0 for tier in tiers for frac in tier):
            raise PreflopError("preflop raise fractions must be positive")
        if self.max_raises <= 0:
            raise PreflopError("preflop max_raises must be positive")
        if not 0 < self.small_blind < self.big_blind < self.initial_stack:
            raise PreflopError(
                "expected 0 < small_blind < big_blind < initial_stack"
            )


SHIPPED_PREFLOP_RULES = PreflopRules()


@dataclass(frozen=True)
class PreflopLegalAction:
    """One trainer-emitted action at the current node.

    ``amount`` is the additional chips paid by the actor. ``bet_fraction`` is
    populated for a sized raise and is ``None`` for fold/check/call or an
    optional explicit all-in action.
    """

    kind: str
    amount: int
    bet_fraction: Optional[float] = None
    is_all_in: bool = False

    @property
    def is_raise(self) -> bool:
        return self.kind == "raise"


def preflop_raise_amount_f32(
    frac: float, pot: int, to_call: int, stack: int,
) -> int:
    """Return the trainer's preflop raise increment using C float arithmetic.

    ``mccfr_blueprint.c:971-975`` multiplies a ``float`` bet fraction and then
    truncates to ``int``. Packing both the fraction and product through IEEE
    float32 reproduces that contract before the stack clamp.
    """

    f32 = struct.unpack("f", struct.pack("f", float(frac)))[0]
    base = pot if to_call == 0 else pot + to_call
    product = struct.unpack("f", struct.pack("f", f32 * base))[0]
    amount = int(product) if to_call == 0 else to_call + int(product)
    return min(amount, stack)


@dataclass(frozen=True)
class PreflopState:
    """Immutable six-seat preflop state, indexed by full seat ``0..5``.

    The compatibility properties intentionally duck-type the existing
    ``PokerAgent`` / ``StrategySource`` / ``key_builder`` decision seam.
    """

    rules: PreflopRules
    stacks: Tuple[int, ...]
    bets: Tuple[int, ...]
    active: Tuple[bool, ...]
    acted: Tuple[bool, ...]
    pot: int
    num_raises: int
    action_history: Tuple[int, ...]
    to_act_order_idx: Optional[int]

    # GameState-compatible decision fields. They are class constants rather
    # than stored fields because a PreflopState can only ever be street zero.
    street: ClassVar[int] = 0
    board: ClassVar[Tuple[int, ...]] = ()
    postflop_history: ClassVar[Tuple[int, ...]] = ()

    def __post_init__(self) -> None:
        for name, values in (
            ("stacks", self.stacks),
            ("bets", self.bets),
            ("active", self.active),
            ("acted", self.acted),
        ):
            if len(values) != 6:
                raise PreflopError(f"{name} must have six full-seat entries")
        if any(stack < 0 for stack in self.stacks):
            raise PreflopError("preflop stack cannot be negative")
        if any(bet < 0 for bet in self.bets):
            raise PreflopError("preflop contribution cannot be negative")
        if any(
            self.stacks[seat] + self.bets[seat] != self.rules.initial_stack
            for seat in range(6)
        ):
            raise PreflopError("stack + contribution must equal initial_stack")
        if self.pot != sum(self.bets):
            raise PreflopError("preflop pot must equal total contributions")
        if not 0 <= self.num_raises <= self.rules.max_raises:
            raise PreflopError("num_raises is outside the configured cap")
        if any(
            isinstance(action, bool) or not isinstance(action, Integral)
            for action in self.action_history
        ):
            raise PreflopError("preflop action history must contain integer indices")
        if self.to_act_order_idx is not None:
            if not 0 <= self.to_act_order_idx < len(PREFLOP_ORDER):
                raise PreflopError("to_act_order_idx is out of range")
            if not self.active[PREFLOP_ORDER[self.to_act_order_idx]]:
                raise PreflopError("to-act seat must be active")

    @property
    def preflop_prefix(self) -> Tuple[int, ...]:
        return self.action_history

    @property
    def contributions(self) -> Tuple[int, ...]:
        return self.bets

    @property
    def preflop_tiers(self) -> Tuple[Tuple[float, ...], ...]:
        return self.rules.tiers

    @property
    def preflop_max_raises(self) -> int:
        return self.rules.max_raises

    @property
    def add_explicit_allin(self) -> bool:
        return self.rules.add_explicit_allin

    @property
    def small_blind(self) -> int:
        return self.rules.small_blind

    @property
    def big_blind(self) -> int:
        return self.rules.big_blind

    @property
    def initial_stack(self) -> int:
        return self.rules.initial_stack

    @property
    def n_live(self) -> int:
        return sum(1 for is_active in self.active if is_active)

    @property
    def terminal(self) -> bool:
        return self.to_act_order_idx is None

    @property
    def uncontested(self) -> bool:
        return self.terminal and self.n_live == 1

    @property
    def flop_ready(self) -> bool:
        return self.terminal and self.n_live >= 2

    @property
    def live_seats(self) -> Tuple[int, ...]:
        return tuple(seat for seat in range(6) if self.active[seat])

    @property
    def live_seat_names(self) -> Tuple[str, ...]:
        return tuple(SEAT_NAME[seat] for seat in self.live_seats)

    @property
    def live_stacks(self) -> Tuple[int, ...]:
        return tuple(self.stacks[seat] for seat in self.live_seats)

    @property
    def folded(self) -> Tuple[bool, ...]:
        return tuple(not is_active for is_active in self.active)

    @property
    def facing_bet(self) -> bool:
        seat = self.to_act_seat()
        if seat is None:
            return False
        return self.bets[seat] < _max_live_bet(self)

    def to_act_seat(self) -> Optional[int]:
        if self.to_act_order_idx is None:
            return None
        return PREFLOP_ORDER[self.to_act_order_idx]

    def legal_actions(self) -> Tuple[PreflopLegalAction, ...]:
        """Return the current trainer action list in blueprint index order."""

        seat = self.to_act_seat()
        if seat is None:
            return ()
        to_call = max(0, _max_live_bet(self) - self.bets[seat])
        stack = self.stacks[seat]
        actions = []
        if to_call > 0:
            actions.append(PreflopLegalAction("fold", 0))
            actions.append(
                PreflopLegalAction("call", min(to_call, stack), is_all_in=stack <= to_call)
            )
        else:
            actions.append(PreflopLegalAction("check", 0))

        if self.num_raises >= self.rules.max_raises:
            return tuple(actions)

        tier_idx = min(self.num_raises, len(self.rules.tiers) - 1)
        added_all_in = False
        for fraction in self.rules.tiers[tier_idx]:
            amount = preflop_raise_amount_f32(
                fraction, self.pot, to_call, stack
            )
            if amount <= to_call:
                continue
            is_all_in = amount >= stack
            if is_all_in and added_all_in:
                continue
            added_all_in = added_all_in or is_all_in
            actions.append(
                PreflopLegalAction(
                    "raise", amount, bet_fraction=fraction, is_all_in=is_all_in
                )
            )

        if (
            self.rules.add_explicit_allin
            and not added_all_in
            and stack > to_call
        ):
            actions.append(
                PreflopLegalAction("raise", stack, is_all_in=True)
            )
        return tuple(actions)

    def num_legal_actions(self) -> int:
        return len(self.legal_actions())

    def advance(self, action_idx: int) -> "PreflopState":
        return advance_preflop(self, action_idx)


def new_preflop_state(
    rules: PreflopRules = SHIPPED_PREFLOP_RULES,
) -> PreflopState:
    """Post blinds and return the first UTG decision of a six-max hand."""

    stacks = [rules.initial_stack] * 6
    bets = [0] * 6
    bets[FULL_SEAT["SB"]] = rules.small_blind
    bets[FULL_SEAT["BB"]] = rules.big_blind
    stacks[FULL_SEAT["SB"]] -= rules.small_blind
    stacks[FULL_SEAT["BB"]] -= rules.big_blind
    return PreflopState(
        rules=rules,
        stacks=tuple(stacks),
        bets=tuple(bets),
        active=(True,) * 6,
        acted=(False,) * 6,
        pot=rules.small_blind + rules.big_blind,
        num_raises=0,
        action_history=(),
        to_act_order_idx=0,
    )


def advance_preflop(state: PreflopState, action_idx: int) -> PreflopState:
    """Apply one node-local action index using the trainer transition."""

    action_idx = _require_action_index(action_idx)
    seat = state.to_act_seat()
    if seat is None:
        raise PreflopReplayError("cannot advance a terminal preflop state")
    actions = state.legal_actions()
    if action_idx < 0 or action_idx >= len(actions):
        raise PreflopReplayError(
            f"preflop action_idx {action_idx} out of range [0,{len(actions)}) "
            f"at history {list(state.action_history)} for seat {seat}"
        )
    action = actions[action_idx]
    stacks = list(state.stacks)
    bets = list(state.bets)
    active = list(state.active)
    acted = list(state.acted)
    pot = state.pot
    num_raises = state.num_raises

    if action.kind == "fold":
        active[seat] = False
    elif action.kind in ("check", "call"):
        bets[seat] += action.amount
        stacks[seat] -= action.amount
        pot += action.amount
        acted[seat] = True
    elif action.kind == "raise":
        bets[seat] += action.amount
        stacks[seat] -= action.amount
        pot += action.amount
        acted[seat] = True
        for other in range(6):
            if other != seat and active[other]:
                acted[other] = False
        num_raises += 1
    else:  # pragma: no cover - actions are constructed above
        raise AssertionError(f"unknown preflop action kind {action.kind!r}")

    history = state.action_history + (action_idx,)
    active_tuple = tuple(active)
    acted_tuple = tuple(acted)
    if sum(active_tuple) <= 1 or _round_done(tuple(bets), active_tuple, acted_tuple):
        next_idx = None
    else:
        next_idx = _next_active_order_idx(active_tuple, state.to_act_order_idx)
        if next_idx is None:  # pragma: no cover - <=1 active handled above
            raise PreflopReplayError("no next active preflop player")

    return replace(
        state,
        stacks=tuple(stacks),
        bets=tuple(bets),
        active=active_tuple,
        acted=acted_tuple,
        pot=pot,
        num_raises=num_raises,
        action_history=history,
        to_act_order_idx=next_idx,
    )


def replay_preflop(
    action_history: Sequence[int],
    rules: PreflopRules = SHIPPED_PREFLOP_RULES,
    *,
    require_terminal: bool = False,
) -> PreflopState:

    state = new_preflop_state(rules)
    for offset, action_idx in enumerate(action_history):
        if state.terminal:
            raise PreflopReplayError(
                f"preflop prefix has action {offset} after the hand/round closed"
            )
        state = advance_preflop(state, _require_action_index(action_idx))
    if require_terminal and not state.terminal:
        raise PreflopReplayError("preflop prefix is not closed")
    return state


def validate_preflop_state(state: PreflopState) -> PreflopState:
    """Fail loudly unless ``state`` exactly equals a fresh trainer replay."""

    replayed = replay_preflop(
        state.action_history, state.rules, require_terminal=state.terminal
    )
    if replayed != state:
        raise PreflopReplayError("live preflop state diverged from canonical replay")
    return replayed


class SampledActionLike(Protocol):
    action_idx: int


class PreflopAgent(Protocol):
    def act(
        self, state: PreflopState, hero_cards: Tuple[int, int]
    ) -> SampledActionLike:
        ...


def play_preflop(
    agents: Sequence[PreflopAgent],
    hole_cards: Sequence[Tuple[int, int]],
    state: Optional[PreflopState] = None,
) -> PreflopState:
    """Let six existing ``PokerAgent``-compatible policies close preflop.

    ``agents`` and ``hole_cards`` are indexed by full seat (SB=0 through
    BTN=5). Each agent must expose ``act(state, hero_cards)`` returning an
    object with ``action_idx`` (the existing ``SampledAction`` contract).
    """

    if len(agents) != 6 or len(hole_cards) != 6:
        raise ValueError("play_preflop requires six agents and six hole-card pairs")
    current = state if state is not None else new_preflop_state()
    _require_shipped_rules(current.rules, "play_preflop")
    for _ in range(128):
        if current.terminal:
            return validate_preflop_state(current)
        seat = current.to_act_seat()
        assert seat is not None
        sampled = agents[seat].act(current, hole_cards[seat])
        current = advance_preflop(current, sampled.action_idx)
    raise PreflopReplayError("preflop did not terminate within 128 actions")


def settle_preflop_terminal(
    state: PreflopState,
    hole_cards: Sequence[Tuple[int, int]],
) -> "SettlementResult":
    """Settle a fold-around hand through the shared settlement contract.

    A state with two or more live players is ``flop_ready`` and must continue;
    only an uncontested terminal can be settled with an empty board here.
    """

    _require_shipped_rules(state.rules, "settle_preflop_terminal")
    if len(hole_cards) != 6:
        raise ValueError("preflop settlement requires six hole-card pairs")
    if not state.uncontested:
        raise ValueError("only an uncontested terminal can settle preflop")
    validate_preflop_state(state)
    from app.engine.agent.settlement import settle_hand

    return settle_hand(
        board=(),
        hole_cards=hole_cards,
        contributions=state.contributions,
        folded=state.folded,
        button=FULL_SEAT["BTN"],
    )


def _max_live_bet(state: PreflopState) -> int:
    return max(
        (state.bets[seat] for seat in range(6) if state.active[seat]),
        default=0,
    )


def _round_done(
    bets: Tuple[int, ...], active: Tuple[bool, ...], acted: Tuple[bool, ...]
) -> bool:
    maximum = max(
        (bets[seat] for seat in range(6) if active[seat]), default=0
    )
    return all(
        not active[seat] or (acted[seat] and bets[seat] == maximum)
        for seat in range(6)
    )


def _next_active_order_idx(
    active: Tuple[bool, ...], current_idx: Optional[int]
) -> Optional[int]:
    if current_idx is None:
        return None
    for offset in range(1, len(PREFLOP_ORDER) + 1):
        candidate = (current_idx + offset) % len(PREFLOP_ORDER)
        if active[PREFLOP_ORDER[candidate]]:
            return candidate
    return None


def _require_action_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PreflopReplayError(
            f"preflop action index must be an integer, got {value!r}"
        )
    return int(value)


def _require_shipped_rules(rules: PreflopRules, operation: str) -> None:
    if rules != SHIPPED_PREFLOP_RULES:
        raise PreflopError(
            f"{operation} supports only SHIPPED_PREFLOP_RULES; "
            "custom rules are low-level replay-only"
        )
