"""Pure planning and label-projection helpers for off-tree add-action.

The live solver consumes street-global first-raise and subsequent-raise size
lists.  This module turns the immutable per-street observations carried by a
GameState-like object into a bounded extension of those lists.  It deliberately
does not import GameState or any solver binding: planning is deterministic and
can be tested without constructing an engine.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence, Tuple

from app.engine.agent.pseudoharmonic import pseudoharmonic_pick


FIRST_RAISE = "first"
SUBSEQUENT_RAISE = "subsequent"

SKIP_INVALID_STEP = "invalid_event_step"
SKIP_INVALID_DENOMINATOR = "invalid_observation_denominator"
SKIP_NON_POSITIVE = "non_positive_fraction"
SKIP_ALL_IN_FAMILY = "all_in_family"
SKIP_DUPLICATE_BASE = "duplicate_base_child"
SKIP_DUPLICATE_ADDED = "duplicate_added_child"
SKIP_DUPLICATE_STEP = "duplicate_event_step"
SKIP_ADDED_SIZE_CAP = "added_size_cap"
SKIP_UNSORTED_BASE_SIZES = "unsorted_base_sizes"

_MAX_ADDED_PER_LIST = 3
_ALL_IN_FAMILY_FRACTION = 5.0
_CHIP_DEDUP_TOLERANCE = 1


class _OffTreeState(Protocol):
    """The state surface needed by :func:`build_offtree_extension`."""

    street_history: Tuple[int, ...]
    street_offtree_raises: Tuple[Tuple[int, int, int, int, int], ...]


@dataclass(frozen=True)
class OffTreeEvent:
    """One analyzed off-tree raise observation."""

    step: int
    increment: int
    pot_before: int
    to_call_before: int
    stack_before: int
    fraction: float
    raise_level: str


@dataclass(frozen=True)
class OffTreeSkip:
    """A rejected event with a stable machine-readable reason."""

    step: int
    fraction: float
    raise_level: str
    reason: str


class OffTreeExactUnsupported(ValueError):
    """A strict real-geometry event cannot be represented exactly."""

    def __init__(self, skip: OffTreeSkip):
        self.skip = skip
        super().__init__(
            "strict off-tree extension cannot represent street step "
            f"{skip.step}: {skip.reason}"
        )


@dataclass(frozen=True)
class ExtensionPlan:
    """The request-scoped size lists and surviving event-to-size mapping."""

    bet_sizes: Tuple[float, ...]
    subsequent_bet_sizes: Tuple[float, ...]
    added_first: Tuple[float, ...]
    added_subsequent: Tuple[float, ...]
    event_fractions: Dict[int, float]
    event_amounts: Dict[int, int]


@dataclass(frozen=True)
class OffTreeExtensionAnalysis:
    """Analysis exists for every non-empty event channel, even if all skip."""

    events: Tuple[OffTreeEvent, ...]
    skipped: Tuple[OffTreeSkip, ...]
    plan: Optional[ExtensionPlan]


@dataclass(frozen=True)
class HeroMergeResult:
    """One strategy row projected from the extended to the base vocabulary."""

    masses: Tuple[float, ...]
    merged_mass: float
    dropped_mass: float
    dropped: bool
    drop_reason: Optional[str]


def offtree_add_enabled() -> bool:
    """Read the default-OFF ``NEXUS_OFFTREE_ADD`` gate on every call."""

    value = os.environ.get("NEXUS_OFFTREE_ADD", "")
    return value[:1] in ("1", "t", "T", "y", "Y")


def _raise_level(street_history: Sequence[int], step: int) -> str:
    """Classify an event from the base-vocabulary history before its step.

    Before the first raise, check is the only non-raise action and has index
    zero.  Consequently the first positive index on a still-open street is
    the first raise (including an explicit-all-in slot); every event after it
    belongs to the subsequent-raise list.
    """

    prior = street_history[: max(0, min(step, len(street_history)))]
    return SUBSEQUENT_RAISE if any(idx > 0 for idx in prior) else FIRST_RAISE


def _fraction(
    increment: int,
    pot_before: int,
    to_call_before: int,
) -> Tuple[float, bool]:
    if to_call_before == 0:
        numerator = increment + 0.5
        denominator = pot_before
    else:
        numerator = increment - to_call_before + 0.5
        denominator = pot_before + to_call_before
    if denominator <= 0:
        return 0.0, False
    return float(numerator) / float(denominator), True


def _builder_amount(
    fraction: float,
    pot_before: int,
    to_call_before: int,
) -> int:
    """Builder truncation mirror before the remaining-stack clamp.

    Event fractions carry their half-chip margin from :func:`_fraction`; this
    consumer deliberately does not emulate float32 or alter base sizes.
    """

    if to_call_before == 0:
        return int(fraction * pot_before)
    return to_call_before + int(fraction * (pot_before + to_call_before))


def _clamped_builder_amount(
    fraction: float,
    pot_before: int,
    to_call_before: int,
    stack_before: int,
) -> int:
    """Builder raise increment after its remaining-stack clamp."""

    return min(
        _builder_amount(fraction, pot_before, to_call_before),
        stack_before,
    )


def build_offtree_extension(
    state: _OffTreeState,
    bet_sizes: Sequence[float],
    subsequent_bet_sizes: Sequence[float],
    *,
    strict: bool = False,
) -> Optional[OffTreeExtensionAnalysis]:
    """Analyze this street's events and build a bounded size-list extension.

    ``None`` is reserved for the zero-event invariant.  A non-empty event
    channel always returns an analysis, with ``plan=None`` when every event is
    skipped.  De-duplication compares integer child amounts at the observed
    node and treats a difference of at most one chip as already represented.
    Unsorted base lists fail safe with a disclosed skip because recorded
    indices, replay enumeration, and builder list order must agree.

    ``strict=True`` is the request-local real-geometry contract.  It uses an
    exact (zero-chip) duplicate test, admits large non-all-in sizes and implicit
    all-ins, and raises :class:`OffTreeExactUnsupported` for any event that
    cannot be replayed at the observed chip amount.  The default path retains
    the production planner's skip-and-disclose behavior.
    """

    raw_events = tuple(state.street_offtree_raises)
    if not raw_events:
        return None

    base_first = tuple(float(size) for size in bet_sizes)
    base_subsequent = tuple(float(size) for size in subsequent_bet_sizes)
    # Recorded raise indices and the C builder both follow list order.  The v2
    # replay may sort sizes while matching, so an unsorted base list is unsafe
    # even when the extension itself would sort the merged result.
    base_lists_ascending = all(
        left <= right for left, right in zip(base_first, base_first[1:])
    ) and all(
        left <= right
        for left, right in zip(base_subsequent, base_subsequent[1:])
    )
    accepted_first: list[float] = []
    accepted_subsequent: list[float] = []
    event_fractions: Dict[int, float] = {}
    event_amounts: Dict[int, int] = {}
    seen_steps: set[int] = set()
    events: list[OffTreeEvent] = []
    skipped: list[OffTreeSkip] = []
    dedup_tolerance = 0 if strict else _CHIP_DEDUP_TOLERANCE

    for raw in raw_events:
        step, increment, pot_before, to_call_before, stack_before = (
            int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]), int(raw[4])
        )
        raise_level = _raise_level(state.street_history, step)
        fraction, valid_denominator = _fraction(
            increment, pot_before, to_call_before,
        )
        event = OffTreeEvent(
            step=step,
            increment=increment,
            pot_before=pot_before,
            to_call_before=to_call_before,
            stack_before=stack_before,
            fraction=fraction,
            raise_level=raise_level,
        )
        events.append(event)

        reason: Optional[str] = None
        if not base_lists_ascending:
            reason = SKIP_UNSORTED_BASE_SIZES
        elif step < 0 or step >= len(state.street_history):
            reason = SKIP_INVALID_STEP
        elif step in seen_steps:
            reason = SKIP_DUPLICATE_STEP
        elif not valid_denominator:
            reason = SKIP_INVALID_DENOMINATOR
        elif increment > stack_before:
            reason = SKIP_ALL_IN_FAMILY
        elif increment == stack_before and not strict:
            reason = SKIP_ALL_IN_FAMILY
        elif fraction <= 0.0:
            reason = SKIP_NON_POSITIVE
        elif (fraction >= _ALL_IN_FAMILY_FRACTION and not strict):
            reason = SKIP_ALL_IN_FAMILY
        seen_steps.add(step)

        base = base_first if raise_level == FIRST_RAISE else base_subsequent
        accepted = (
            accepted_first
            if raise_level == FIRST_RAISE
            else accepted_subsequent
        )
        duplicate_base_fraction: Optional[float] = None
        duplicate_added_fraction: Optional[float] = None
        if reason is None:
            candidate_amount = _clamped_builder_amount(
                fraction, pot_before, to_call_before, stack_before,
            )
            duplicate_base_fraction = next((
                size for size in base
                if abs(
                    candidate_amount
                    - _clamped_builder_amount(
                        size, pot_before, to_call_before, stack_before,
                    )
                ) <= dedup_tolerance
            ), None)
            if duplicate_base_fraction is not None:
                reason = SKIP_DUPLICATE_BASE
            else:
                duplicate_added_fraction = next((
                    size for size in accepted
                    if abs(
                        candidate_amount
                        - _clamped_builder_amount(
                            size, pot_before, to_call_before, stack_before,
                        )
                    ) <= dedup_tolerance
                ), None)
            if reason is None and duplicate_added_fraction is not None:
                reason = SKIP_DUPLICATE_ADDED
            elif reason is None and len(accepted) >= _MAX_ADDED_PER_LIST:
                reason = SKIP_ADDED_SIZE_CAP

        if reason is not None:
            skip = OffTreeSkip(
                step=step,
                fraction=fraction,
                raise_level=raise_level,
                reason=reason,
            )
            skipped.append(skip)
            if (strict and reason == SKIP_DUPLICATE_BASE
                    and duplicate_base_fraction is not None):
                # The exact child already exists in the base vocabulary.  Map
                # this event to that fraction so amount-based replay, rather
                # than the recorded nearest-index replay, owns the transition.
                event_fractions[step] = duplicate_base_fraction
                event_amounts[step] = increment
            elif (reason == SKIP_DUPLICATE_ADDED
                    and duplicate_added_fraction is not None):
                # No second global size is needed, but replay must still walk
                # the already-present added child at this observed step.
                event_fractions[step] = duplicate_added_fraction
                event_amounts[step] = increment
            elif strict:
                raise OffTreeExactUnsupported(skip)
            continue

        accepted.append(fraction)
        event_fractions[step] = fraction
        event_amounts[step] = increment

    if not event_fractions:
        return OffTreeExtensionAnalysis(
            events=tuple(events),
            skipped=tuple(skipped),
            plan=None,
        )

    added_first = tuple(sorted(accepted_first))
    added_subsequent = tuple(sorted(accepted_subsequent))
    plan = ExtensionPlan(
        bet_sizes=tuple(sorted(base_first + added_first)),
        subsequent_bet_sizes=tuple(sorted(
            base_subsequent + added_subsequent,
        )),
        added_first=added_first,
        added_subsequent=added_subsequent,
        event_fractions=event_fractions,
        event_amounts=event_amounts,
    )
    return OffTreeExtensionAnalysis(
        events=tuple(events),
        skipped=tuple(skipped),
        plan=plan,
    )


def _validate_masses(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{name} must contain finite non-negative masses")
    return result


def merge_added_action_mass(
    base_masses: Sequence[float],
    base_raise_fractions: Sequence[float],
    base_raise_positions: Sequence[int],
    added_actions: Sequence[Tuple[float, float]],
) -> HeroMergeResult:
    """Project one extended strategy row onto the base action vocabulary.

    Each added action's mass is assigned to the pseudo-harmonic-nearest base
    raise.  Positions point into ``base_masses``, allowing fold/check/call to
    remain in their original label slots.  If commitment pruning leaves no
    base raise child, added mass is dropped and the remaining base mix is
    renormalized.  The helper is row-local so callers can apply it to every
    hand row and every player readout without changing label extraction.
    """

    masses = _validate_masses(base_masses, "base_masses")
    fractions = [float(value) for value in base_raise_fractions]
    positions = [int(value) for value in base_raise_positions]
    if len(fractions) != len(positions):
        raise ValueError(
            "base_raise_fractions and base_raise_positions length mismatch"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in fractions):
        raise ValueError(
            "base_raise_fractions must contain finite positive values"
        )
    if len(set(positions)) != len(positions):
        raise ValueError("base_raise_positions must be unique")
    if any(position < 0 or position >= len(masses) for position in positions):
        raise ValueError("base_raise_positions contains an out-of-range index")

    normalized_added: list[Tuple[float, float]] = []
    for fraction, mass in added_actions:
        fraction_value = float(fraction)
        if not math.isfinite(fraction_value) or fraction_value <= 0.0:
            raise ValueError("added action fractions must be finite and positive")
        mass_value = _validate_masses((mass,), "added action masses")[0]
        normalized_added.append((fraction_value, mass_value))

    if not normalized_added:
        return HeroMergeResult(
            masses=tuple(masses),
            merged_mass=0.0,
            dropped_mass=0.0,
            dropped=False,
            drop_reason=None,
        )

    added_mass = sum(mass for _, mass in normalized_added)
    if added_mass == 0.0:
        return HeroMergeResult(
            masses=tuple(masses),
            merged_mass=0.0,
            dropped_mass=0.0,
            dropped=False,
            drop_reason=None,
        )
    if not positions:
        remaining_mass = sum(masses)
        if remaining_mass > 0.0:
            masses = [mass / remaining_mass for mass in masses]
        elif masses:
            uniform = 1.0 / len(masses)
            masses = [uniform] * len(masses)
        return HeroMergeResult(
            masses=tuple(masses),
            merged_mass=0.0,
            dropped_mass=float(added_mass),
            dropped=True,
            drop_reason="no_base_raise_child",
        )

    for added_fraction, mass in normalized_added:
        raise_index = pseudoharmonic_pick(added_fraction, fractions)
        masses[positions[raise_index]] += mass

    return HeroMergeResult(
        masses=tuple(masses),
        merged_mass=float(added_mass),
        dropped_mass=0.0,
        dropped=False,
        drop_reason=None,
    )
