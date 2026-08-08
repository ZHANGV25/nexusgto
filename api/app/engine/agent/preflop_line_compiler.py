"""Trainer-exact compiler for closed six-max preflop lines.

The compiler never derives raise indices or chips in Python.  It replays the
line through ``bp_generate_actions``, the exported action generator used by
the blueprint trainer, and maps supplied semantics by scanning each emitted
node's action types.  Missing actors are inferred as folds when (and only
when) the generated node contains a fold action.

Amounts passed to :func:`compile_preflop_line` are already normalized to the
trainer's bb=100 chip scale.  The engine translation boundary owns the wire
scale conversion.
"""

from __future__ import annotations

import ctypes
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.engine.agent.preflop_constructor import PreflopAction


PREFLOP_ORDER: Tuple[str, ...] = ("UTG", "MP", "CO", "BTN", "SB", "BB")
SEAT: Dict[str, int] = {
    "SB": 0,
    "BB": 1,
    "UTG": 2,
    "MP": 3,
    "CO": 4,
    "BTN": 5,
}
SMALL_BLIND = 50
BIG_BLIND = 100
INITIAL_STACK = 10_000
PREFLOP_TIERS: Tuple[Tuple[float, ...], ...] = (
    (0.7, 1.0),
    (0.7, 1.0),
    (1.0,),
)
PREFLOP_MAX_RAISES = 3
ADD_EXPLICIT_ALLIN = 0

_ACTION_TYPE = {"fold": 0, "check": 1, "call": 2, "raise": 3}
_MAX_GENERATED_ACTIONS = 16
_MAX_WALK_NODES = 64


class PreflopLineIllegal(Exception):
    """The supplied semantic line cannot be replayed in the trainer tree."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TrainerActionGeneratorUnavailable(RuntimeError):
    """The trainer action-generator shared library is unavailable."""


@dataclass(frozen=True)
class GeneratedAction:
    """One action emitted by ``bp_generate_actions``."""

    type: int
    bet_idx: int
    amount: int


@dataclass(frozen=True)
class PreflopNodeLayout:
    """The exact trainer node consumed for one explicit or inferred action."""

    position: str
    prefix_before: Tuple[int, ...]
    pot: int
    stack: int
    to_call: int
    num_raises: int
    bet_sizes: Tuple[float, ...]
    actions: Tuple[GeneratedAction, ...]
    chosen_index: int
    supplied_action_ordinal: Optional[int]
    inferred_fold: bool


@dataclass(frozen=True)
class PreflopActionSnap:
    """Size-selection provenance for one supplied semantic action."""

    action_ordinal: int
    position: str
    action: str
    action_index: int
    supplied_amount: Optional[float]
    canonical_amount: int
    snapped: bool
    log_distance: Optional[float]
    threebet_compatibility_override: bool = False


@dataclass(frozen=True)
class CompiledPreflopLine:
    """A trainer-replayed terminal preflop line."""

    prefix: List[int]
    live_seats: List[str]
    flop_pot: int
    flop_stacks: List[int]
    ends_preflop: bool
    snap_provenance: List[PreflopActionSnap]
    node_layouts: List[PreflopNodeLayout]

    @property
    def action_snaps(self) -> List[PreflopActionSnap]:
        """Compatibility alias for callers that prefer the shorter name."""
        return self.snap_provenance

    @property
    def snaps(self) -> List[PreflopActionSnap]:
        """Short alias for per-action snap provenance."""
        return self.snap_provenance

    @property
    def layouts(self) -> List[PreflopNodeLayout]:
        """Compatibility alias for the consumed per-node layouts."""
        return self.node_layouts


class _BPAction(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("bet_idx", ctypes.c_int),
        ("amount", ctypes.c_int),
    ]


@dataclass
class _WalkState:
    street_bets: Dict[str, int]
    stacks: Dict[str, int]
    pot: int
    current_bet: int
    num_raises: int
    folded: Set[str]
    acted: Set[str]
    prefix: List[int]
    cursor: int


_trainer_lib: Optional[Any] = None
_trainer_lib_lock = threading.Lock()


def resolve_poker_solver_root() -> Path:
    """Resolve poker-solver with the same precedence as ``engine_paths``."""
    root = os.environ.get("POKER_SOLVER_PATH")
    if not root:
        try:
            from app.config import settings

            if settings.poker_solver_path is not None:
                root = str(settings.poker_solver_path)
        except Exception:  # pragma: no cover - app config is normally present
            root = None
    if not root:
        root = str(Path(__file__).resolve().parents[3].parent / "poker-solver")
    return Path(root)


def _trainer_library_candidates() -> Tuple[Path, ...]:
    build = resolve_poker_solver_root() / "build"
    return (
        build / "mccfr_blueprint.so",
        build / "mccfr_blueprint.dll",
    )


def _load_trainer_library() -> Any:
    """Load and bind ``bp_generate_actions`` once, on first compilation."""
    global _trainer_lib
    if _trainer_lib is not None:
        return _trainer_lib
    with _trainer_lib_lock:
        if _trainer_lib is not None:
            return _trainer_lib
        candidates = _trainer_library_candidates()
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            searched = ", ".join(str(candidate) for candidate in candidates)
            raise TrainerActionGeneratorUnavailable(
                "trainer action generator is unavailable: expected "
                "mccfr_blueprint.so or mccfr_blueprint.dll under the resolved "
                f"poker-solver build directory (searched: {searched}); set "
                "POKER_SOLVER_PATH to the staged poker-solver checkout"
            )
        try:
            lib = ctypes.CDLL(str(path))
            generate = lib.bp_generate_actions
        except (OSError, AttributeError) as exc:
            raise TrainerActionGeneratorUnavailable(
                f"failed to load bp_generate_actions from {path}: {exc}"
            ) from exc
        generate.argtypes = [
            ctypes.POINTER(_BPAction),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        generate.restype = ctypes.c_int
        _trainer_lib = lib
        return lib


def _generate_actions(
    pot: int,
    stack: int,
    to_call: int,
    num_raises: int,
    bet_sizes: Sequence[float],
) -> Tuple[GeneratedAction, ...]:
    lib = _load_trainer_library()
    out = (_BPAction * _MAX_GENERATED_ACTIONS)()
    c_sizes = (ctypes.c_float * len(bet_sizes))(*bet_sizes)
    count = lib.bp_generate_actions(
        out,
        _MAX_GENERATED_ACTIONS,
        int(pot),
        int(stack),
        int(to_call),
        int(num_raises),
        PREFLOP_MAX_RAISES,
        c_sizes,
        len(bet_sizes),
        ADD_EXPLICIT_ALLIN,
    )
    if count <= 0 or count > _MAX_GENERATED_ACTIONS:
        raise RuntimeError(
            "bp_generate_actions emitted an invalid action count "
            f"{count} at pot={pot}, stack={stack}, to_call={to_call}, "
            f"num_raises={num_raises}"
        )
    return tuple(
        GeneratedAction(
            type=int(out[index].type),
            bet_idx=int(out[index].bet_idx),
            amount=int(out[index].amount),
        )
        for index in range(count)
    )


def _initial_state() -> _WalkState:
    street_bets = {seat: 0 for seat in PREFLOP_ORDER}
    street_bets["SB"] = SMALL_BLIND
    street_bets["BB"] = BIG_BLIND
    return _WalkState(
        street_bets=street_bets,
        stacks={
            seat: INITIAL_STACK - street_bets[seat]
            for seat in PREFLOP_ORDER
        },
        pot=SMALL_BLIND + BIG_BLIND,
        current_bet=BIG_BLIND,
        num_raises=0,
        folded=set(),
        acted=set(),
        prefix=[],
        cursor=0,
    )


def _live_seats(state: _WalkState) -> List[str]:
    return sorted(
        (seat for seat in PREFLOP_ORDER if seat not in state.folded),
        key=lambda seat: SEAT[seat],
    )


def _round_closed(state: _WalkState, live: Sequence[str]) -> bool:
    return all(
        seat in state.acted and state.street_bets[seat] >= state.current_bet
        for seat in live
    )


def _next_actor(state: _WalkState) -> Optional[Tuple[int, str]]:
    for offset in range(len(PREFLOP_ORDER)):
        index = (state.cursor + offset) % len(PREFLOP_ORDER)
        seat = PREFLOP_ORDER[index]
        if seat in state.folded:
            continue
        if seat not in state.acted or state.street_bets[seat] < state.current_bet:
            return index, seat
    return None


def _v1_threebet_compatibility_ordinal(
    actions: Sequence[PreflopAction],
) -> Optional[int]:
    """Return the v1 HU 3-bet action ordinal, solely for its size override."""
    raise_ordinals = [
        index for index, action in enumerate(actions) if action.action == "raise"
    ]
    if len(raise_ordinals) != 2:
        return None
    first_ordinal, second_ordinal = raise_ordinals
    opener = actions[first_ordinal].position
    threebettor = actions[second_ordinal].position
    if opener == threebettor:
        return None
    if PREFLOP_ORDER.index(threebettor) <= PREFLOP_ORDER.index(opener):
        return None
    calls = [
        (index, action)
        for index, action in enumerate(actions)
        if action.action == "call"
    ]
    if len(calls) != 1 or calls[0][1].position != opener:
        return None
    if calls[0][0] <= second_ordinal:
        return None
    if any(action.action == "check" for action in actions):
        return None
    for index, action in enumerate(actions):
        if index in (first_ordinal, second_ordinal, calls[0][0]):
            continue
        if action.action != "fold" or action.position in (opener, threebettor):
            return None
    ranks: List[int] = []
    for index, action in enumerate(actions):
        if index == calls[0][0]:
            ranks.append(len(PREFLOP_ORDER))
        else:
            ranks.append(PREFLOP_ORDER.index(action.position))
    if ranks != sorted(ranks):
        return None
    return second_ordinal


def _choose_supplied_action(
    actions: Tuple[GeneratedAction, ...],
    supplied: PreflopAction,
    ordinal: int,
    compatibility_ordinal: Optional[int],
    threebet_size_idx: int,
) -> Tuple[int, PreflopActionSnap]:
    if supplied.action != "raise" and supplied.amount > 0:
        raise PreflopLineIllegal(
            f"action #{ordinal} for {supplied.position}: amount is valid only "
            "for a raise"
        )

    wanted_type = _ACTION_TYPE[supplied.action]
    candidates = [
        (index, action)
        for index, action in enumerate(actions)
        if action.type == wanted_type
    ]
    if not candidates:
        raise PreflopLineIllegal(
            f"action #{ordinal} for {supplied.position}: "
            f"{supplied.action!r} is not legal at this trainer node; "
            f"generated action types are {[action.type for action in actions]}"
        )

    used_compatibility = supplied.action == "raise" and ordinal == compatibility_ordinal
    if used_compatibility:
        if threebet_size_idx >= len(candidates):
            raise PreflopLineIllegal(
                f"action #{ordinal} for {supplied.position}: v1 3-bet size "
                f"index {threebet_size_idx} is absent from the generated "
                f"{len(candidates)}-size node"
            )
        chosen_index, chosen = candidates[threebet_size_idx]
    elif supplied.action == "raise" and supplied.amount > 0:
        chosen_index, chosen = min(
            candidates,
            key=lambda item: abs(math.log(supplied.amount / item[1].amount)),
        )
    else:
        chosen_index, chosen = candidates[0]

    supplied_amount: Optional[float] = None
    snapped = False
    log_distance: Optional[float] = None
    if supplied.action == "raise" and supplied.amount > 0:
        supplied_amount = float(supplied.amount)
        log_distance = abs(math.log(supplied_amount / chosen.amount))
        snapped = supplied_amount != chosen.amount

    return chosen_index, PreflopActionSnap(
        action_ordinal=ordinal,
        position=supplied.position,
        action=supplied.action,
        action_index=chosen_index,
        supplied_amount=supplied_amount,
        canonical_amount=chosen.amount,
        snapped=snapped,
        log_distance=log_distance,
        threebet_compatibility_override=used_compatibility,
    )


def _apply_action(
    state: _WalkState,
    position: str,
    action: GeneratedAction,
) -> None:
    state.acted.add(position)
    if action.type == 0:
        state.folded.add(position)
        return
    if action.type in (2, 3):
        state.street_bets[position] += action.amount
        state.stacks[position] -= action.amount
        state.pot += action.amount
        if action.type == 3:
            state.current_bet = state.street_bets[position]
            state.num_raises += 1
        return
    if action.type != 1:
        raise RuntimeError(f"bp_generate_actions emitted unknown type {action.type}")


def _terminal_result(
    state: _WalkState,
    ends_preflop: bool,
    snaps: List[PreflopActionSnap],
    layouts: List[PreflopNodeLayout],
) -> CompiledPreflopLine:
    live = _live_seats(state)
    return CompiledPreflopLine(
        prefix=list(state.prefix),
        live_seats=live,
        flop_pot=state.pot,
        flop_stacks=[state.stacks[seat] for seat in live],
        ends_preflop=ends_preflop,
        snap_provenance=list(snaps),
        node_layouts=list(layouts),
    )


def compile_preflop_line(
    actions: List[PreflopAction],
    *,
    threebet_size_idx: int = 0,
) -> CompiledPreflopLine:
    """Replay an ordered semantic preflop line through the trainer generator.

    Supplied actions are voluntary actions in chronological order.  When the
    current owing seat is not the next supplied actor, its fold is inferred.
    A replay that reaches a closed betting round returns the exact flop-entry
    chips; a replay reduced to one live seat returns ``ends_preflop=True``.
    """
    if not actions:
        raise PreflopLineIllegal("actions must be non-empty")
    if threebet_size_idx not in (0, 1):
        raise PreflopLineIllegal(
            f"threebet_size_idx {threebet_size_idx!r} is invalid; expected 0 or 1"
        )
    for ordinal, action in enumerate(actions):
        if action.position not in SEAT:
            raise PreflopLineIllegal(
                f"action #{ordinal}: unknown position {action.position!r}; "
                f"expected one of {list(PREFLOP_ORDER)}"
            )
        if action.action not in _ACTION_TYPE:
            raise PreflopLineIllegal(
                f"action #{ordinal} for {action.position}: unknown semantic "
                f"{action.action!r}"
            )
        if not math.isfinite(action.amount) or action.amount < 0:
            raise PreflopLineIllegal(
                f"action #{ordinal} for {action.position}: amount must be a "
                f"finite non-negative number, got {action.amount!r}"
            )

    compatibility_ordinal = _v1_threebet_compatibility_ordinal(actions)

    state = _initial_state()
    supplied_ordinal = 0
    snaps: List[PreflopActionSnap] = []
    layouts: List[PreflopNodeLayout] = []

    for _node_count in range(_MAX_WALK_NODES):
        live = _live_seats(state)
        if len(live) <= 1:
            if supplied_ordinal != len(actions):
                pending = actions[supplied_ordinal]
                if pending.position in state.folded:
                    raise PreflopLineIllegal(
                        f"action #{supplied_ordinal} by {pending.position} is "
                        "out of order: that seat was already folded while "
                        "inferring an earlier acting-order gap"
                    )
                raise PreflopLineIllegal(
                    f"action #{supplied_ordinal} by {pending.position} occurs "
                    "after the hand already ended preflop"
                )
            return _terminal_result(state, True, snaps, layouts)
        if _round_closed(state, live):
            if supplied_ordinal != len(actions):
                pending = actions[supplied_ordinal]
                raise PreflopLineIllegal(
                    f"action #{supplied_ordinal} by {pending.position} is not "
                    "owed: the preflop betting round is already closed"
                )
            return _terminal_result(state, False, snaps, layouts)

        actor_info = _next_actor(state)
        if actor_info is None:
            raise RuntimeError(
                f"trainer preflop round is open with no owing actor at {state.prefix}"
            )
        actor_index, position = actor_info
        state.cursor = (actor_index + 1) % len(PREFLOP_ORDER)
        to_call = state.current_bet - state.street_bets[position]
        tier = min(state.num_raises, len(PREFLOP_TIERS) - 1)
        bet_sizes = PREFLOP_TIERS[tier]
        generated = _generate_actions(
            state.pot,
            state.stacks[position],
            to_call,
            state.num_raises,
            bet_sizes,
        )

        supplied: Optional[PreflopAction] = None
        if supplied_ordinal < len(actions):
            pending = actions[supplied_ordinal]
            if pending.position in state.folded:
                raise PreflopLineIllegal(
                    f"action #{supplied_ordinal} by {pending.position} is out "
                    "of order: that seat was already folded while inferring "
                    "an earlier acting-order gap"
                )
            if pending.position == position:
                supplied = pending

        inferred = supplied is None
        if inferred:
            fold_candidates = [
                (index, action)
                for index, action in enumerate(generated)
                if action.type == _ACTION_TYPE["fold"]
            ]
            if not fold_candidates:
                pending_desc = (
                    f" before action #{supplied_ordinal} by "
                    f"{actions[supplied_ordinal].position}"
                    if supplied_ordinal < len(actions)
                    else " at the end of the supplied line"
                )
                raise PreflopLineIllegal(
                    f"{position} cannot be skipped{pending_desc}: its trainer "
                    "node has no fold action (an explicit legal action is required)"
                )
            chosen_index, chosen = fold_candidates[0]
            snap = None
        else:
            chosen_index, snap = _choose_supplied_action(
                generated,
                supplied,
                supplied_ordinal,
                compatibility_ordinal,
                threebet_size_idx,
            )
            chosen = generated[chosen_index]

        layouts.append(
            PreflopNodeLayout(
                position=position,
                prefix_before=tuple(state.prefix),
                pot=state.pot,
                stack=state.stacks[position],
                to_call=to_call,
                num_raises=state.num_raises,
                bet_sizes=tuple(bet_sizes),
                actions=generated,
                chosen_index=chosen_index,
                supplied_action_ordinal=None if inferred else supplied_ordinal,
                inferred_fold=inferred,
            )
        )
        state.prefix.append(chosen_index)
        _apply_action(state, position, chosen)
        if snap is not None:
            snaps.append(snap)
            supplied_ordinal += 1

    raise RuntimeError(
        f"trainer preflop replay exceeded {_MAX_WALK_NODES} action nodes; "
        f"prefix={state.prefix}"
    )


__all__ = [
    "CompiledPreflopLine",
    "GeneratedAction",
    "PreflopActionSnap",
    "PreflopLineIllegal",
    "PreflopNodeLayout",
    "TrainerActionGeneratorUnavailable",
    "compile_preflop_line",
    "resolve_poker_solver_root",
]
