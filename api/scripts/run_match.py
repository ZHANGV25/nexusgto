"""Run variance-reduced, full-hand 6-max matches.

WHAT THIS NUMBER MEASURES
-------------------------
For each independently dealt deck, the same six hole-card packets and board are
replayed in six counterfactual arms.  One designated "hero" policy is evaluated
in every fixed FULL_SEAT position (SB, BB, UTG, MP, CO, BTN); the other five
seats use the field policy.  The six arms are paired into ONE independent
deal-group observation.  The HEADLINE metric is hero's own NET chips per hand
(mean, standard error, normal-approximation 95% CI, reported in bb/100 with
BB=100).  The paired seat differential (hero minus mean field seat) is also
recorded; under the enforced zero-sum invariant it equals exactly
(N/(N-1)) = 1.2x hero net and must NOT be read as bb/100.

The three modes are: ``aa`` (A against identical A, an exact plumbing null),
``epsilon`` (an epsilon-perturbed A against five unperturbed A seats), and
``ab`` (A against five B seats).  Every arm starts again at 100bb.  Exact decks,
fixed card-to-position assignment, and seat-keyed common policy seeds reduce
card, position and action-sampling variance.

With ``--allin-equity-adjust``, all-in showdowns are scored by their conditional
mean over the true remaining-deck runout distribution.  This Rao-Blackwellized
score is unbiased for the same mean while removing realized runout variance.

FULL_SEAT names are button-relative positions, not permanent player chairs.
Each deal group is one nominal session hand: group ``g`` advances its nominal
hero/button-relative position to ``g mod 6``.  Its other five rotations are
counterfactual duplicate arms of that same hand, not later session hands.

WHAT THIS NUMBER DOES NOT MEASURE
---------------------------------
It is realized EV for one policy inserted into a fixed five-player field.  It
is not exploitability, GTO distance, a best response, or six independent
samples per deck.  The independent sample is the deal group;
treating its six correlated rotations as independent would understate error.

Run from api/ after `make all` and `pip install -e .` at the repository root::

    export AGENT_BLUEPRINT_PATH=/path/to/unified_blueprint.bps
    python scripts/run_match.py --mode aa --deals 1000 --master-seed 0 --out out/aa

Each file in tests/goldens/ holds a complete match command and its environment.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))
_PS_PATH = os.environ.get("POKER_SOLVER_PATH", str(_API_ROOT.parent / "solver"))
for _p in (_PS_PATH, str(Path(_PS_PATH) / "python")):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402


BIG_BLIND = 100  # must equal SHIPPED_PREFLOP_RULES.big_blind; checked per hand
SEAT_NAMES = ("SB", "BB", "UTG", "MP", "CO", "BTN")
N_SEATS = len(SEAT_NAMES)
BUTTON_SEAT = 5
VALID_MODES = ("aa", "epsilon", "ab")
SCHEMA = "nexusgto_match_v1"
_ADJUSTED_ZERO_SUM_EPSILON = 1e-9
_STREET_BOARD_CARD_COUNT = {1: 3, 2: 4, 3: 5}

HoleCards = Tuple[int, int]
AdjustedDeltas = Tuple[float, float, float, float, float, float]


class ActingPolicy(Protocol):
    """The small runtime surface shared by PokerAgent and pure test stubs."""

    def act(self, state: object, hero_cards: HoleCards) -> object:
        ...


PolicyFactory = Callable[[int], ActingPolicy]


class SolveWallTracker(Protocol):
    """Per-group timing seam supplied by the live-source CLI factory."""

    @property
    def sources(self) -> Mapping[str, str]:
        ...

    def begin_group(self) -> None:
        ...

    def finish_group(self) -> Dict[str, Tuple[float, ...]]:
        ...


@dataclass(frozen=True)
class Deal:
    """One duplicate-able six-seat deal, indexed by fixed FULL_SEAT."""

    board: Tuple[int, int, int, int, int]
    hole_cards: Tuple[HoleCards, HoleCards, HoleCards, HoleCards, HoleCards, HoleCards]

    def as_dict(self) -> Dict[str, object]:
        return {
            "board": list(self.board),
            "hole_cards": {
                SEAT_NAMES[seat]: list(self.hole_cards[seat]) for seat in range(N_SEATS)
            },
        }


@dataclass(frozen=True)
class PlayedHand:
    """Full-hand output consumed by the duplicate/rotation aggregator."""

    deltas: Tuple[int, int, int, int, int, int]
    action_count: int
    terminal: str
    went_to_showdown: bool
    preflop_prefix: Tuple[int, ...] = ()
    fallback_count: int = 0
    contributions: Tuple[int, ...] = ()
    folded: Tuple[bool, ...] = ()
    adjusted_deltas: Optional[AdjustedDeltas] = None

    def __post_init__(self) -> None:
        if len(self.deltas) != N_SEATS:
            raise ValueError(f"need {N_SEATS} deltas, got {len(self.deltas)}")
        if sum(self.deltas) != 0:
            raise AssertionError(f"hand does not conserve chips: deltas={self.deltas}")
        if self.adjusted_deltas is not None:
            if len(self.adjusted_deltas) != N_SEATS:
                raise ValueError(
                    f"need {N_SEATS} adjusted deltas, "
                    f"got {len(self.adjusted_deltas)}"
                )
            # Averaging exact integer settlements can leave only floating-point
            # summation residue; anything larger is a real conservation bug.
            adjusted_sum = sum(self.adjusted_deltas)
            if (
                not all(math.isfinite(float(delta)) for delta in self.adjusted_deltas)
                or abs(adjusted_sum) > _ADJUSTED_ZERO_SUM_EPSILON
            ):
                raise AssertionError(
                    "adjusted hand does not conserve chips: "
                    f"deltas={self.adjusted_deltas} sum={adjusted_sum}"
                )
        if not 0 <= self.fallback_count <= self.action_count:
            raise ValueError("fallback_count must be between zero and action_count")
        if self.contributions and len(self.contributions) != N_SEATS:
            raise ValueError("contributions must have six entries when recorded")
        if self.folded and len(self.folded) != N_SEATS:
            raise ValueError("folded must have six entries when recorded")


@dataclass(frozen=True)
class RotationResult:
    """One duplicate arm: hero in exactly one FULL_SEAT position."""

    rotation_index: int
    session_hand_index: int
    hero_seat: int
    hand: PlayedHand
    seat_differential_numerator: int

    @property
    def hero_delta(self) -> int:
        return self.hand.deltas[self.hero_seat]

    @property
    def field_delta_sum(self) -> int:
        return sum(
            delta for seat, delta in enumerate(self.hand.deltas) if seat != self.hero_seat
        )

    @property
    def field_delta_mean(self) -> float:
        return self.field_delta_sum / float(N_SEATS - 1)

    @property
    def seat_differential_chips(self) -> float:
        return self.seat_differential_numerator / float(N_SEATS - 1)

    @property
    def hero_delta_adjusted(self) -> Optional[float]:
        if self.hand.adjusted_deltas is None:
            return None
        return self.hand.adjusted_deltas[self.hero_seat]

    @property
    def field_delta_mean_adjusted(self) -> Optional[float]:
        if self.hand.adjusted_deltas is None:
            return None
        field_sum = sum(
            delta
            for seat, delta in enumerate(self.hand.adjusted_deltas)
            if seat != self.hero_seat
        )
        return field_sum / float(N_SEATS - 1)

    @property
    def seat_differential_chips_adjusted(self) -> Optional[float]:
        hero = self.hero_delta_adjusted
        field_mean = self.field_delta_mean_adjusted
        if hero is None or field_mean is None:
            return None
        return hero - field_mean

    def as_dict(self, *, allin_equity_adjust: bool = False) -> Dict[str, object]:
        record = {
            "rotation_index": self.rotation_index,
            "session_hand_index": self.session_hand_index,
            "hero_seat": self.hero_seat,
            "hero_position": SEAT_NAMES[self.hero_seat],
            "button_seat": BUTTON_SEAT,
            "button_position": SEAT_NAMES[BUTTON_SEAT],
            "deltas": list(self.hand.deltas),
            "hero_delta_chips": self.hero_delta,
            "field_mean_delta_chips": self.field_delta_mean,
            "seat_differential_chips": self.seat_differential_chips,
            "action_count": self.hand.action_count,
            "fallback_count": self.hand.fallback_count,
            "fallback_rate": (
                self.hand.fallback_count / self.hand.action_count
                if self.hand.action_count else 0.0
            ),
            "contributions": list(self.hand.contributions),
            "folded": list(self.hand.folded),
            "terminal": self.hand.terminal,
            "went_to_showdown": self.hand.went_to_showdown,
            "preflop_prefix": list(self.hand.preflop_prefix),
        }
        if allin_equity_adjust:
            if self.hand.adjusted_deltas is None:
                raise AssertionError(
                    "adjusted group contains a hand without adjusted deltas"
                )
            record.update({
                "adjusted_deltas": list(self.hand.adjusted_deltas),
                "hero_delta_chips_adjusted": self.hero_delta_adjusted,
                "field_mean_delta_chips_adjusted": (
                    self.field_delta_mean_adjusted
                ),
                "seat_differential_chips_adjusted": (
                    self.seat_differential_chips_adjusted
                ),
            })
        return record


@dataclass(frozen=True)
class DealGroupResult:
    """The independent statistical unit: all six hero-seat rotations."""

    group_index: int
    deal: Deal
    rotations: Tuple[
        RotationResult,
        RotationResult,
        RotationResult,
        RotationResult,
        RotationResult,
        RotationResult,
    ]
    paired_numerator: int
    allin_equity_adjust: bool = False
    adjusted_paired_numerator: Optional[float] = None
    solve_wall_samples: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    solve_wall_sources: Dict[str, str] = field(default_factory=dict)

    @property
    def paired_differential_chips(self) -> float:
        # Each arm divides by five field seats; the group then averages six arms.
        return self.paired_numerator / float((N_SEATS - 1) * N_SEATS)

    @property
    def hero_net_chips_per_hand(self) -> float:
        # Under the enforced zero-sum invariant field_sum == -hero_delta, so
        # each arm's numerator (N-1)*hero - field_sum equals N*hero.  Dividing
        # the group numerator by N*N therefore gives mean hero NET chips per
        # hand -- the conventional own-winnings quantity behind bb/100.  The
        # paired seat differential above is exactly (N/(N-1)) = 1.2x this
        # number; reporting the differential as bb/100 would overstate a true
        # edge by 20%.
        return self.paired_numerator / float(N_SEATS * N_SEATS)

    @property
    def paired_differential_chips_adjusted(self) -> Optional[float]:
        if self.adjusted_paired_numerator is None:
            return None
        return self.adjusted_paired_numerator / float(
            (N_SEATS - 1) * N_SEATS
        )

    @property
    def hero_net_chips_per_hand_adjusted(self) -> Optional[float]:
        if self.adjusted_paired_numerator is None:
            return None
        return self.adjusted_paired_numerator / float(N_SEATS * N_SEATS)

    @property
    def action_count(self) -> int:
        return sum(rotation.hand.action_count for rotation in self.rotations)

    @property
    def fallback_count(self) -> int:
        return sum(rotation.hand.fallback_count for rotation in self.rotations)

    def as_dict(
        self, *, mode: str, master_seed: int, epsilon: Optional[float] = None
    ) -> Dict[str, object]:
        record = {
            "schema": SCHEMA,
            "record_type": "deal_group",
            "mode": mode,
            "epsilon": epsilon if mode == "epsilon" else None,
            "master_seed": master_seed,
            "group_index": self.group_index,
            "session_hand_index": self.group_index,
            "nominal_hero_seat": self.group_index % N_SEATS,
            "nominal_hero_position": SEAT_NAMES[self.group_index % N_SEATS],
            "n_rotations": N_SEATS,
            "button_model": (
                "group index advances the nominal button-relative assignment; rotations "
                "are counterfactual arms of one session hand"
            ),
            "deal": self.deal.as_dict(),
            "rotations": [
                rotation.as_dict(
                    allin_equity_adjust=self.allin_equity_adjust
                )
                for rotation in self.rotations
            ],
            "paired_differential_chips": self.paired_differential_chips,
            "hero_net_chips_per_hand": self.hero_net_chips_per_hand,
            "decisions": self.action_count,
            "fallback_count": self.fallback_count,
            "fallback_rate": (
                self.fallback_count / self.action_count if self.action_count else 0.0
            ),
        }
        if self.allin_equity_adjust:
            if self.adjusted_paired_numerator is None:
                raise AssertionError(
                    "adjusted group has no adjusted paired numerator"
                )
            record.update({
                "paired_differential_chips_adjusted": (
                    self.paired_differential_chips_adjusted
                ),
                "hero_net_chips_per_hand_adjusted": (
                    self.hero_net_chips_per_hand_adjusted
                ),
            })
        if self.solve_wall_sources:
            record["solve_wall_s_by_source"] = _solve_wall_summaries(
                self.solve_wall_samples, self.solve_wall_sources
            )
        return record


@dataclass(frozen=True)
class MatchRun:
    groups: Tuple[DealGroupResult, ...]
    summary: Dict[str, object]


@dataclass(frozen=True)
class ActionChoice:
    """Minimal SampledAction-compatible result for wrappers and pure stubs."""

    action_idx: int
    mix: Optional[object] = None
    # The exact distribution action_idx was drawn from, when a wrapper changes
    # it away from ``mix`` (AIVAT scoring needs the true sampling law).
    sampling_probs: Optional[Tuple[float, ...]] = None


def sampling_probs_for(result: object, n_legal: int) -> Optional[Tuple[float, ...]]:
    """Return the normalized distribution a policy sampled ``result`` from.

    Prefers an explicit ``sampling_probs``; otherwise reads ``mix.freqs``.
    Returns None when the policy exposes neither (pure integer stubs).
    """

    explicit = getattr(result, "sampling_probs", None)
    if explicit is not None:
        probs = [float(p) for p in explicit]
    else:
        freqs = getattr(getattr(result, "mix", None), "freqs", None)
        if freqs is None:
            return None
        probs = [0.0] * n_legal
        for idx, p in freqs.items():
            if not 0 <= int(idx) < n_legal:
                raise ValueError(f"mix index {idx} outside n_legal={n_legal}")
            probs[int(idx)] += float(p)
    if len(probs) != n_legal:
        raise ValueError(f"{len(probs)} sampling probabilities for n_legal={n_legal}")
    total = math.fsum(probs)
    if any(p < 0.0 or not math.isfinite(p) for p in probs) or not math.isclose(
        total, 1.0, abs_tol=1e-6
    ):
        raise ValueError(f"invalid sampling distribution {probs}")
    return tuple(p / total for p in probs)


def epsilon_sampling_probs(
    base: Sequence[float], epsilon: float
) -> Tuple[float, ...]:
    """EpsilonPolicy's law: keep w.p. 1-eps, else uniform over the others."""

    n_legal = len(base)
    if n_legal <= 1:
        return tuple(float(p) for p in base)
    return tuple(
        (1.0 - epsilon) * p + epsilon * (1.0 - p) / (n_legal - 1) for p in base
    )


class ObservedPolicy:
    """Count decisions and uniform lookup fallbacks without changing play."""

    def __init__(self, inner: ActingPolicy) -> None:
        self.inner = inner
        self.decisions = 0
        self.fallback_count = 0

    def act(self, state: object, hero_cards: HoleCards) -> object:
        result = self.inner.act(state, hero_cards)
        self.decisions += 1
        mix = getattr(result, "mix", None)
        self.fallback_count += int(bool(getattr(mix, "is_uniform_fallback", False)))
        return result

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)


def _validate_epsilon(epsilon: float) -> float:
    value = float(epsilon)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("epsilon must be finite and in [0, 1]")
    return value


class EpsilonPolicy:
    """Perturb an underlying policy by choosing a different legal action.

    The underlying A action is sampled first.  With probability epsilon, this
    wrapper replaces it uniformly from the other legal action indices.  Nodes
    with only one legal action are necessarily unchanged.
    """

    def __init__(self, inner: ActingPolicy, epsilon: float, seed: int) -> None:
        self._inner = inner
        self._epsilon = _validate_epsilon(epsilon)
        self._rng = np.random.default_rng(seed)
        self.perturbations = 0

    def act(self, state: object, hero_cards: HoleCards) -> ActionChoice:
        underlying = self._inner.act(state, hero_cards)
        chosen = _action_index(underlying)
        mix = getattr(underlying, "mix", None)
        n_legal_fn = getattr(state, "num_legal_actions", None)
        if not callable(n_legal_fn):
            raise TypeError("epsilon policy state needs num_legal_actions()")
        n_legal = int(n_legal_fn())
        if chosen < 0 or chosen >= n_legal:
            raise ValueError(f"underlying policy returned {chosen}, n_legal={n_legal}")
        base = sampling_probs_for(underlying, n_legal)
        probs = (
            epsilon_sampling_probs(base, self._epsilon) if base is not None else None
        )
        if n_legal <= 1 or self._rng.random() >= self._epsilon:
            return ActionChoice(chosen, mix=mix, sampling_probs=probs)
        alternatives = [idx for idx in range(n_legal) if idx != chosen]
        self.perturbations += 1
        return ActionChoice(
            int(self._rng.choice(alternatives)), mix=mix, sampling_probs=probs
        )


def _action_index(result: object) -> int:
    """Accept PokerAgent's SampledAction or an integer-returning pure stub."""

    if isinstance(result, (int, np.integer)):
        return int(result)
    idx = getattr(result, "action_idx", None)
    if idx is None:
        raise TypeError("policy act() must return int or an object with action_idx")
    return int(idx)


def _derived_seed(master_seed: int, group_index: int, seat: int, stream: int) -> int:
    """Stable uint32 child seed; deliberately excludes the rotation index.

    Omitting the rotation is the common-random-number coupling: when A and B
    are identical, a given physical seat receives the same policy RNG in all
    six duplicate arms and therefore produces the same trajectory.
    """

    if master_seed < 0 or group_index < 0 or seat < 0 or stream < 0:
        raise ValueError("seed components must be non-negative")
    entropy = [
        master_seed & 0xFFFF_FFFF,
        (master_seed >> 32) & 0xFFFF_FFFF,
        group_index & 0xFFFF_FFFF,
        seat & 0xFFFF_FFFF,
        stream & 0xFFFF_FFFF,
    ]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def deal_from_rng(rng: np.random.Generator) -> Deal:
    """Deal board first, then one two-card packet per fixed FULL_SEAT."""

    cards = [int(card) for card in rng.permutation(52)]
    board = tuple(cards[:5])
    holes = tuple(tuple(cards[5 + 2 * seat: 7 + 2 * seat]) for seat in range(N_SEATS))
    return Deal(board=board, hole_cards=holes)  # type: ignore[arg-type]


HandPlayer = Callable[..., PlayedHand]


def _policies_for_rotation(
    *,
    mode: str,
    hero_seat: int,
    group_index: int,
    master_seed: int,
    epsilon: float,
    a_factory: PolicyFactory,
    b_factory: Optional[PolicyFactory],
) -> Tuple[ActingPolicy, ...]:
    policies: List[ActingPolicy] = []
    for seat in range(N_SEATS):
        policy_seed = _derived_seed(master_seed, group_index, seat, stream=1)
        if seat != hero_seat and b_factory is not None:
            policy = b_factory(policy_seed)
        else:
            if mode == "ab" and seat != hero_seat:
                raise ValueError("mode='ab' requires b_factory")
            policy = a_factory(policy_seed)
        if mode == "epsilon" and seat == hero_seat:
            perturb_seed = _derived_seed(master_seed, group_index, seat, stream=2)
            policy = EpsilonPolicy(policy, epsilon=epsilon, seed=perturb_seed)
        policies.append(ObservedPolicy(policy))
    return tuple(policies)


def play_deal_group(
    deal: Deal,
    *,
    group_index: int,
    mode: str,
    master_seed: int,
    a_factory: PolicyFactory,
    b_factory: Optional[PolicyFactory] = None,
    epsilon: float = 0.05,
    hand_player: HandPlayer,
    solve_wall_tracker: Optional[SolveWallTracker] = None,
    allin_equity_adjust: bool = False,
) -> DealGroupResult:
    """Replay one exact deal with the designated hero in every position."""

    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
    epsilon = _validate_epsilon(epsilon)
    if solve_wall_tracker is not None:
        solve_wall_tracker.begin_group()

    rotations: List[RotationResult] = []
    total_numerator = 0
    adjusted_total_numerator: Optional[float] = (
        0.0 if allin_equity_adjust else None
    )
    for rotation_index in range(N_SEATS):
        # group_index advances the nominal button-relative assignment.  These
        # six rotations are counterfactual duplicate arms of that one hand.
        hero_seat = (group_index + rotation_index) % N_SEATS
        policies = _policies_for_rotation(
            mode=mode,
            hero_seat=hero_seat,
            group_index=group_index,
            master_seed=master_seed,
            epsilon=epsilon,
            a_factory=a_factory,
            b_factory=b_factory,
        )
        if allin_equity_adjust:
            hand = hand_player(
                deal, policies, allin_equity_adjust=True
            )
        else:
            # Keep the default call byte-for-byte compatible with injected
            # two-argument hand players used by existing match tooling.
            hand = hand_player(deal, policies)
        field_sum = sum(
            delta for seat, delta in enumerate(hand.deltas) if seat != hero_seat
        )
        # Exact integer numerator for hero - mean(field).  Keeping the group
        # numerator integral makes the AA conservation identity exactly zero,
        # rather than a tiny floating cancellation residue.
        numerator = (N_SEATS - 1) * hand.deltas[hero_seat] - field_sum
        if allin_equity_adjust:
            if hand.adjusted_deltas is None:
                raise AssertionError(
                    "allin_equity_adjust hand player returned no adjusted deltas"
                )
            adjusted_field_sum = sum(
                delta
                for seat, delta in enumerate(hand.adjusted_deltas)
                if seat != hero_seat
            )
            adjusted_numerator = (
                (N_SEATS - 1) * hand.adjusted_deltas[hero_seat]
                - adjusted_field_sum
            )
            if adjusted_total_numerator is None:
                raise AssertionError("adjusted numerator accumulator is missing")
            adjusted_total_numerator += adjusted_numerator
        rotations.append(
            RotationResult(
                rotation_index=rotation_index,
                session_hand_index=group_index,
                hero_seat=hero_seat,
                hand=hand,
                seat_differential_numerator=numerator,
            )
        )
        total_numerator += numerator

    if mode == "aa":
        first = rotations[0].hand.deltas
        if any(rotation.hand.deltas != first for rotation in rotations[1:]):
            raise AssertionError(
                "AA duplicate arms diverged: policy factories must reset from the supplied "
                "seat-keyed seeds and must not leak mutable hand state"
            )
        if total_numerator != 0:
            raise AssertionError(f"AA paired differential is not exactly zero: {total_numerator}")
        if allin_equity_adjust:
            first_adjusted = rotations[0].hand.adjusted_deltas
            if any(
                rotation.hand.adjusted_deltas != first_adjusted
                for rotation in rotations[1:]
            ):
                raise AssertionError(
                    "AA adjusted duplicate arms diverged: policy factories "
                    "must reset from the supplied seat-keyed seeds and must "
                    "not leak mutable hand state"
                )
            if (
                adjusted_total_numerator is None
                or abs(adjusted_total_numerator)
                > _ADJUSTED_ZERO_SUM_EPSILON
            ):
                raise AssertionError(
                    "AA adjusted paired differential is not zero: "
                    f"{adjusted_total_numerator}"
                )
            # Preserve the exact-null reporting contract despite harmless
            # floating summation residue across the six rotated hero seats.
            adjusted_total_numerator = 0.0

    return DealGroupResult(
        group_index=group_index,
        deal=deal,
        rotations=tuple(rotations),  # type: ignore[arg-type]
        paired_numerator=total_numerator,
        allin_equity_adjust=allin_equity_adjust,
        adjusted_paired_numerator=adjusted_total_numerator,
        solve_wall_samples=(
            solve_wall_tracker.finish_group()
            if solve_wall_tracker is not None else {}
        ),
        solve_wall_sources=(
            dict(solve_wall_tracker.sources)
            if solve_wall_tracker is not None else {}
        ),
    )


def summarize_groups(
    groups: Sequence[DealGroupResult], *, mode: str, master_seed: int, epsilon: float,
    truncated_by_wall_budget: bool = False,
    wall_budget_active: bool = False,
) -> Dict[str, object]:
    """Summarize independent deal groups, mirroring measure_bot_strength."""

    epsilon = _validate_epsilon(epsilon)
    if not groups:
        raise ValueError("at least one deal group is required")
    values = np.asarray(
        [group.hero_net_chips_per_hand for group in groups], dtype=np.float64
    )
    differentials = np.asarray(
        [group.paired_differential_chips for group in groups], dtype=np.float64
    )
    n = len(values)
    mean = float(values.mean())
    to_bb100 = 100.0 / BIG_BLIND
    if n > 1:
        std = float(values.std(ddof=1))
        stderr: Optional[float] = std / math.sqrt(n)
        stderr_bb: Optional[float] = round(stderr * to_bb100, 4)
        ci_bb: Optional[List[float]] = [
            round((mean - 1.96 * stderr) * to_bb100, 4),
            round((mean + 1.96 * stderr) * to_bb100, 4),
        ]
        ci_note = None
    else:
        # A single deal group has no between-group variance to estimate; a
        # zero-width CI here would misread as certainty.
        stderr = None
        stderr_bb = None
        ci_bb = None
        ci_note = "n < 2 deal groups: no variance estimate; run more deals"
    decisions = sum(group.action_count for group in groups)
    fallbacks = sum(group.fallback_count for group in groups)
    source_names: Dict[str, str] = {}
    wall_samples: Dict[str, List[float]] = {}
    for group in groups:
        source_names.update(group.solve_wall_sources)
        for role, role_samples in group.solve_wall_samples.items():
            wall_samples.setdefault(role, []).extend(role_samples)
    summary = {
        "schema": SCHEMA,
        "record_type": "summary",
        "mode": mode,
        "epsilon": epsilon if mode == "epsilon" else None,
        "master_seed": master_seed,
        "n": n,
        "n_deal_groups": n,
        "n_full_hands": n * N_SEATS,
        "decisions": decisions,
        "fallback_count": fallbacks,
        "fallback_rate": fallbacks / decisions if decisions else 0.0,
        "mean_hero_net_chips_per_hand": round(mean, 6),
        "stderr_hero_net_chips_per_hand": (
            round(stderr, 6) if stderr is not None else None
        ),
        "mean_paired_differential_chips_per_hand": round(
            float(differentials.mean()), 6
        ),
        "differential_identity": (
            "paired seat differential = (N/(N-1)) x hero net under the enforced "
            "zero-sum invariant; 1.2x at N=6. Headline bb/100 uses HERO NET."
        ),
        "mean_bb_per_100": round(mean * to_bb100, 4),
        "stderr_bb_per_100": stderr_bb,
        "ci95_bb_per_100": ci_bb,
        "ci_note": ci_note,
        "independent_unit": "six-arm duplicate-deck deal group",
        "scope": "full-hand 6-max; six counterfactual seat arms per nominal session hand",
        "interpretation": (
            "Hero's realized net winnings per hand (bb/100) with duplicate cards, "
            "seat rotation and common policy RNGs. Not exploitability, GTO "
            "distance, or a best response."
        ),
        "power_check_expected_sign": (
            "negative: epsilon-perturbed A should underperform the unperturbed A field"
            if mode == "epsilon"
            else None
        ),
    }
    if source_names:
        summary["solve_wall_s_by_source"] = _solve_wall_summaries(
            wall_samples, source_names
        )
    if wall_budget_active or truncated_by_wall_budget:
        summary["truncated_by_wall_budget"] = bool(
            truncated_by_wall_budget
        )
    adjusted_values = [
        group.hero_net_chips_per_hand_adjusted for group in groups
    ]
    if any(value is not None for value in adjusted_values):
        if any(value is None for value in adjusted_values):
            raise ValueError(
                "cannot summarize a mixture of adjusted and raw-only groups"
            )
        adjusted_array = np.asarray(adjusted_values, dtype=np.float64)
        adjusted_mean = float(adjusted_array.mean())
        if n > 1:
            adjusted_std = float(adjusted_array.std(ddof=1))
            adjusted_stderr: Optional[float] = (
                adjusted_std / math.sqrt(n)
            )
            adjusted_stderr_bb: Optional[float] = round(
                adjusted_stderr * to_bb100, 4
            )
            adjusted_ci_bb: Optional[List[float]] = [
                round(
                    (adjusted_mean - 1.96 * adjusted_stderr) * to_bb100,
                    4,
                ),
                round(
                    (adjusted_mean + 1.96 * adjusted_stderr) * to_bb100,
                    4,
                ),
            ]
        else:
            adjusted_stderr_bb = None
            adjusted_ci_bb = None
        summary.update({
            "mean_bb_per_100_adjusted": round(
                adjusted_mean * to_bb100, 4
            ),
            "stderr_bb_per_100_adjusted": adjusted_stderr_bb,
            "ci95_bb_per_100_adjusted": adjusted_ci_bb,
        })
    return summary


def run_match(
    *,
    deals: int,
    mode: str,
    master_seed: int,
    a_factory: PolicyFactory,
    b_factory: Optional[PolicyFactory] = None,
    epsilon: float = 0.05,
    hand_player: Optional[HandPlayer] = None,
    on_group: Optional[Callable[[DealGroupResult], None]] = None,
    solve_wall_tracker: Optional[SolveWallTracker] = None,
    max_wall_s: Optional[float] = None,
    clock: Optional[Callable[[], float]] = None,
    allin_equity_adjust: bool = False,
    start_group: int = 0,
) -> MatchRun:
    """Run ``deals`` independent groups (six full duplicate arms per group).

    ``on_group`` fires after each completed group; the CLI uses it for
    per-group progress and crash-safe partial persistence. ``start_group``
    skips that many decks first, so a shard reproduces groups
    ``start_group..start_group+deals-1`` of the same seed exactly (decks and
    seat-keyed policy seeds both key on the absolute group index).
    """

    if deals <= 0:
        raise ValueError("deals must be positive")
    if start_group < 0:
        raise ValueError("start_group must be non-negative")
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
    if max_wall_s is not None and (
        not math.isfinite(float(max_wall_s)) or float(max_wall_s) <= 0.0
    ):
        raise ValueError("max_wall_s must be finite and positive when provided")
    epsilon = _validate_epsilon(epsilon)
    player = hand_player or play_full_hand
    deal_rng = np.random.default_rng(master_seed)
    monotonic = clock or time.monotonic
    started = monotonic() if max_wall_s is not None else None
    truncated_by_wall_budget = False
    collected: List[DealGroupResult] = []
    for _ in range(start_group):
        deal_from_rng(deal_rng)
    for group_index in range(start_group, start_group + deals):
        deal = deal_from_rng(deal_rng)
        try:
            group = play_deal_group(
                deal,
                group_index=group_index,
                mode=mode,
                master_seed=master_seed,
                a_factory=a_factory,
                b_factory=b_factory,
                epsilon=epsilon,
                hand_player=player,
                solve_wall_tracker=solve_wall_tracker,
                allin_equity_adjust=allin_equity_adjust,
            )
        except Exception as exc:
            raise RuntimeError(
                f"deal group {group_index}/{deals} failed "
                f"(mode={mode}, master_seed={master_seed}, "
                f"deal={deal.as_dict()}): {exc}"
            ) from exc
        if on_group is not None:
            on_group(group)
        collected.append(group)
        if (
            max_wall_s is not None
            and started is not None
            and monotonic() - started > float(max_wall_s)
        ):
            truncated_by_wall_budget = True
            break
    groups = tuple(collected)
    summary = summarize_groups(
        groups, mode=mode, master_seed=master_seed, epsilon=epsilon,
        truncated_by_wall_budget=truncated_by_wall_budget,
        wall_budget_active=max_wall_s is not None,
    )
    return MatchRun(groups=groups, summary=summary)


def _raw_adjusted_deltas(deltas: Sequence[int]) -> AdjustedDeltas:
    return tuple(float(delta) for delta in deltas)  # type: ignore[return-value]


def _allin_equity_adjusted_deltas(
    *,
    full_board: Sequence[int],
    hole_cards: Sequence[HoleCards],
    contributions: Sequence[int],
    folded: Sequence[bool],
    live_board_card_count: int,
) -> AdjustedDeltas:
    """Mean settlement deltas over every unseen board completion."""

    from app.engine.agent.settlement import settle_hand

    if live_board_card_count not in (3, 4, 5):
        raise ValueError(
            "postflop live board count must be 3, 4, or 5, got "
            f"{live_board_card_count}"
        )
    live_seats = sum(1 for did_fold in folded if not did_fold)
    realized = settle_hand(
        list(full_board), hole_cards, contributions, folded,
        button=BUTTON_SEAT,
    )
    if live_seats < 2 or live_board_card_count == 5:
        return _raw_adjusted_deltas(realized.deltas)

    live_board = tuple(int(card) for card in full_board[:live_board_card_count])
    excluded = set(live_board)
    for hand in hole_cards:
        excluded.update(int(card) for card in hand)
    remaining_deck = tuple(card for card in range(52) if card not in excluded)
    cards_to_come = 5 - live_board_card_count
    totals = [0] * N_SEATS
    completion_count = 0
    # For a flop all-in, each unordered turn/river pair represents two equally
    # likely orders and settle_hand is order-insensitive, so combinations give
    # the same exact mean without evaluating both permutations.
    for runout in combinations(remaining_deck, cards_to_come):
        settled = settle_hand(
            list(live_board + runout),
            hole_cards,
            contributions,
            folded,
            button=BUTTON_SEAT,
        )
        for seat, delta in enumerate(settled.deltas):
            totals[seat] += int(delta)
        completion_count += 1
    if completion_count == 0:
        raise AssertionError("all-in board completion enumeration was empty")
    adjusted = tuple(
        total / float(completion_count) for total in totals
    )
    # Every enumerated settle_hand result is exact zero-sum. Division can
    # introduce only a tiny floating summation residue.
    adjusted_sum = sum(adjusted)
    if abs(adjusted_sum) > _ADJUSTED_ZERO_SUM_EPSILON:
        raise AssertionError(
            "all-in equity-adjusted deltas do not conserve chips: "
            f"deltas={adjusted} sum={adjusted_sum}"
        )
    return adjusted  # type: ignore[return-value]


def _betting_closed_by_allin(state: object) -> bool:
    active = tuple(bool(value) for value in getattr(state, "active"))
    stacks = tuple(int(value) for value in getattr(state, "stacks"))
    if len(active) != len(stacks):
        raise AssertionError("GameState active/stacks length mismatch")
    live_slots = [slot for slot, is_active in enumerate(active) if is_active]
    return (
        len(live_slots) >= 2
        and sum(1 for slot in live_slots if stacks[slot] > 0) <= 1
    )


def play_full_hand(
    deal: Deal,
    policies: Sequence[ActingPolicy],
    *,
    allin_equity_adjust: bool = False,
) -> PlayedHand:
    """Play preflop -> postflop -> multiway settlement through real states.

    Imports are deliberately local: importing this script or running injected
    pure plumbing tests cannot construct a blueprint reader or native solver.
    """

    if len(policies) != N_SEATS:
        raise ValueError(f"need {N_SEATS} policies, got {len(policies)}")

    from app.engine.agent.game_state import canonical_flop_chips, new_postflop_state
    from app.engine.agent.preflop_play import play_preflop, settle_preflop_terminal
    from app.engine.agent.settlement import settle_hand

    pre = play_preflop(policies, deal.hole_cards)
    rules_big_blind = int(pre.rules.big_blind)
    if rules_big_blind != BIG_BLIND:
        raise AssertionError(
            f"run_match BIG_BLIND={BIG_BLIND} disagrees with shipped preflop "
            f"rules big_blind={rules_big_blind}; the bb/100 conversion would "
            "be mis-scaled"
        )
    prefix = tuple(int(action) for action in pre.preflop_prefix)
    contributions = [int(value) for value in pre.contributions]
    folded = [not bool(value) for value in pre.active]
    action_count = len(prefix)

    if pre.uncontested:
        # The preflop module owns its terminal replay validation and delegates
        # to the same shared settlement primitive internally.
        settled = settle_preflop_terminal(pre, deal.hole_cards)
        return PlayedHand(
            deltas=tuple(settled.deltas),  # type: ignore[arg-type]
            action_count=action_count,
            terminal="preflop_fold",
            went_to_showdown=bool(settled.went_to_showdown),
            preflop_prefix=prefix,
            fallback_count=sum(int(getattr(p, "fallback_count", 0)) for p in policies),
            contributions=tuple(contributions),
            folded=tuple(folded),
            adjusted_deltas=(
                _raw_adjusted_deltas(settled.deltas)
                if allin_equity_adjust else None
            ),
        )

    live_names = list(pre.live_seat_names)
    validated_pot, validated_stacks = canonical_flop_chips(
        list(prefix), live_names, rules=pre.rules
    )
    if int(validated_pot) != int(pre.pot) or tuple(validated_stacks) != tuple(pre.live_stacks):
        raise AssertionError(
            "preflop state disagrees with canonical replay: "
            f"state={(pre.pot, tuple(pre.live_stacks))} "
            f"replay={(validated_pot, tuple(validated_stacks))}"
        )

    state = new_postflop_state(
        seats=live_names,
        preflop_prefix=list(prefix),
        board=list(deal.board),
        pot=int(validated_pot),
        stacks=list(validated_stacks),
    )
    # GameState advances all-in players through forced zero-chip actions on
    # later streets. A terminal state's street therefore cannot distinguish a
    # true preflop all-in from a postflop all-in. Per the conservative contract,
    # preflop all-ins fall back to raw scoring instead of guessing a 0-card
    # live prefix.
    preflop_allin = (
        _betting_closed_by_allin(state) if allin_equity_adjust else False
    )
    allin_closed_street: Optional[int] = None
    postflop_actions = 0
    max_actions = 512
    while not state.terminal:
        if postflop_actions >= max_actions:
            raise RuntimeError(f"postflop action guard exceeded {max_actions}")
        seat = state.to_act_seat()
        if seat is None:
            raise AssertionError("non-terminal GameState has no acting seat")
        action_idx = _action_index(policies[seat].act(state, deal.hole_cards[seat]))
        acting_street = int(state.street)
        next_state = state.advance(action_idx)
        if (
            allin_equity_adjust
            and not preflop_allin
            and allin_closed_street is None
            and acting_street < 3
            and _betting_closed_by_allin(next_state)
        ):
            # advance() may already expose the next street after the closing
            # call, so the live prefix comes from GameState.street immediately
            # before that action, never from action-count reconstruction.
            allin_closed_street = acting_street
        state = next_state
        postflop_actions += 1

    # PreflopState retains all six seats.  GameState compacts only the live
    # preflop seats, so merge its postflop stack loss back by FULL_SEAT.
    for slot, seat in enumerate(state.seat_of_slot):
        postflop_committed = int(pre.stacks[seat]) - int(state.stacks[slot])
        if postflop_committed < 0:
            raise AssertionError("postflop stack exceeds its preflop-entry stack")
        contributions[seat] += postflop_committed
        folded[seat] = not bool(state.active[slot])

    reconstructed_pot = sum(contributions)
    if reconstructed_pot != int(state.pot):
        raise AssertionError(
            "full-seat contribution merge disagrees with GameState pot: "
            f"contributions={reconstructed_pot} state.pot={state.pot}"
        )

    settled = settle_hand(
        list(deal.board), deal.hole_cards, contributions, folded, button=BUTTON_SEAT
    )
    adjusted_deltas: Optional[AdjustedDeltas] = None
    if allin_equity_adjust:
        if allin_closed_street is None:
            adjusted_deltas = _raw_adjusted_deltas(settled.deltas)
        else:
            adjusted_deltas = _allin_equity_adjusted_deltas(
                full_board=deal.board,
                hole_cards=deal.hole_cards,
                contributions=contributions,
                folded=folded,
                live_board_card_count=(
                    _STREET_BOARD_CARD_COUNT[allin_closed_street]
                ),
            )
    terminal = "showdown" if settled.went_to_showdown else "postflop_fold"
    fallback_count = sum(int(getattr(policy, "fallback_count", 0)) for policy in policies)
    return PlayedHand(
        deltas=tuple(settled.deltas),  # type: ignore[arg-type]
        action_count=action_count + postflop_actions,
        terminal=terminal,
        went_to_showdown=bool(settled.went_to_showdown),
        preflop_prefix=prefix,
        fallback_count=fallback_count,
        contributions=tuple(contributions),
        folded=tuple(folded),
        adjusted_deltas=adjusted_deltas,
    )


from scripts import run_match_cli as _cli  # noqa: E402

write_match = _cli.write_match
_scoped_leaf_mode = _cli.scoped_leaf_mode
_resolved_leaf_mode_env = _cli.resolved_leaf_mode_env
_percentile = _cli.percentile
_solve_wall_summaries = _cli.solve_wall_summaries


def _lookup_factories(
    args: argparse.Namespace,
) -> Tuple[PolicyFactory, Optional[PolicyFactory], Dict[str, object]]:
    return _cli.build_factories(sys.modules[__name__], args)


def _parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    return _cli.parse_args(sys.modules[__name__], argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch the CLI while keeping pure match machinery importable."""

    return _cli.main(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
