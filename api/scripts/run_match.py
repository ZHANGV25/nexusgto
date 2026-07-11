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

FULL_SEAT names are button-relative positions, not permanent player chairs.
Each deal group is one nominal session hand: group ``g`` advances its nominal
hero/button-relative position to ``g mod 6``.  Its other five rotations are
counterfactual duplicate arms of that same hand, not later session hands.

WHAT THIS NUMBER DOES NOT MEASURE
---------------------------------
It is realized EV for one policy inserted into a fixed five-player field.  It
is not exploitability, GTO distance, a best response, or
six independent samples per deck.  The independent sample is the deal group;
treating its six correlated rotations as independent would understate error.

Run from nexusgto-api/::

    $env:PYTHONUTF8=1
    $env:AGENT_BLUEPRINT_PATH="C:/.../unified_blueprint_8.05B.bps"
    python scripts/run_match.py --mode aa --deals 1000 --master-seed 0 --out results/aa
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))
_PS_PATH = os.environ.get("POKER_SOLVER_PATH", str(_API_ROOT.parent / "poker-solver"))
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

HoleCards = Tuple[int, int]


class ActingPolicy(Protocol):
    """The small runtime surface shared by PokerAgent and pure test stubs."""

    def act(self, state: object, hero_cards: HoleCards) -> object:
        ...


PolicyFactory = Callable[[int], ActingPolicy]


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

    def __post_init__(self) -> None:
        if len(self.deltas) != N_SEATS:
            raise ValueError(f"need {N_SEATS} deltas, got {len(self.deltas)}")
        if sum(self.deltas) != 0:
            raise AssertionError(f"hand does not conserve chips: deltas={self.deltas}")
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

    def as_dict(self) -> Dict[str, object]:
        return {
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
    def action_count(self) -> int:
        return sum(rotation.hand.action_count for rotation in self.rotations)

    @property
    def fallback_count(self) -> int:
        return sum(rotation.hand.fallback_count for rotation in self.rotations)

    def as_dict(
        self, *, mode: str, master_seed: int, epsilon: Optional[float] = None
    ) -> Dict[str, object]:
        return {
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
            "rotations": [rotation.as_dict() for rotation in self.rotations],
            "paired_differential_chips": self.paired_differential_chips,
            "hero_net_chips_per_hand": self.hero_net_chips_per_hand,
            "decisions": self.action_count,
            "fallback_count": self.fallback_count,
            "fallback_rate": (
                self.fallback_count / self.action_count if self.action_count else 0.0
            ),
        }


@dataclass(frozen=True)
class MatchRun:
    groups: Tuple[DealGroupResult, ...]
    summary: Dict[str, object]


@dataclass(frozen=True)
class ActionChoice:
    """Minimal SampledAction-compatible result for wrappers and pure stubs."""

    action_idx: int
    mix: Optional[object] = None


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
        if n_legal <= 1 or self._rng.random() >= self._epsilon:
            return ActionChoice(chosen, mix=mix)
        alternatives = [idx for idx in range(n_legal) if idx != chosen]
        self.perturbations += 1
        return ActionChoice(int(self._rng.choice(alternatives)), mix=mix)


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


HandPlayer = Callable[[Deal, Sequence[ActingPolicy]], PlayedHand]


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
        if mode == "ab" and seat != hero_seat:
            if b_factory is None:
                raise ValueError("mode='ab' requires b_factory")
            policy = b_factory(policy_seed)
        else:
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
) -> DealGroupResult:
    """Replay one exact deal with the designated hero in every position."""

    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
    epsilon = _validate_epsilon(epsilon)

    rotations: List[RotationResult] = []
    total_numerator = 0
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
        hand = hand_player(deal, policies)
        field_sum = sum(
            delta for seat, delta in enumerate(hand.deltas) if seat != hero_seat
        )
        # Exact integer numerator for hero - mean(field).  Keeping the group
        # numerator integral makes the AA conservation identity exactly zero,
        # rather than a tiny floating cancellation residue.
        numerator = (N_SEATS - 1) * hand.deltas[hero_seat] - field_sum
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

    return DealGroupResult(
        group_index=group_index,
        deal=deal,
        rotations=tuple(rotations),  # type: ignore[arg-type]
        paired_numerator=total_numerator,
    )


def summarize_groups(
    groups: Sequence[DealGroupResult], *, mode: str, master_seed: int, epsilon: float
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
    return {
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
) -> MatchRun:
    """Run ``deals`` independent groups (six full duplicate arms per group).

    ``on_group`` fires after each completed group; the CLI uses it for
    per-group progress and crash-safe partial persistence.
    """

    if deals <= 0:
        raise ValueError("deals must be positive")
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
    epsilon = _validate_epsilon(epsilon)
    player = hand_player or play_full_hand
    deal_rng = np.random.default_rng(master_seed)
    collected: List[DealGroupResult] = []
    for group_index in range(deals):
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
    groups = tuple(collected)
    summary = summarize_groups(
        groups, mode=mode, master_seed=master_seed, epsilon=epsilon
    )
    return MatchRun(groups=groups, summary=summary)


def play_full_hand(deal: Deal, policies: Sequence[ActingPolicy]) -> PlayedHand:
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
    postflop_actions = 0
    max_actions = 512
    while not state.terminal:
        if postflop_actions >= max_actions:
            raise RuntimeError(f"postflop action guard exceeded {max_actions}")
        seat = state.to_act_seat()
        if seat is None:
            raise AssertionError("non-terminal GameState has no acting seat")
        action_idx = _action_index(policies[seat].act(state, deal.hole_cards[seat]))
        state = state.advance(action_idx)
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
    )


def write_match(run: MatchRun, out_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """Write deterministic JSONL group records plus one summary JSON."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups_path = out / "deal_groups.jsonl"
    groups_tmp = out / "deal_groups.jsonl.tmp"
    mode = str(run.summary["mode"])
    master_seed = int(run.summary["master_seed"])
    epsilon_value = run.summary.get("epsilon")
    record_epsilon = float(epsilon_value) if epsilon_value is not None else None
    with groups_tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for group in run.groups:
            json.dump(
                group.as_dict(
                    mode=mode, master_seed=master_seed, epsilon=record_epsilon
                ),
                fh,
                sort_keys=True,
                separators=(",", ":"),
            )
            fh.write("\n")
    os.replace(str(groups_tmp), str(groups_path))

    summary_path = out / "summary.json"
    summary_tmp = out / "summary.json.tmp"
    with summary_tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(run.summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(str(summary_tmp), str(summary_path))
    return groups_path, summary_path


def _lookup_factories(
    args: argparse.Namespace,
) -> Tuple[PolicyFactory, Optional[PolicyFactory], Dict[str, Optional[str]]]:
    """Construct shared LOOKUP sources and cheap per-hand seeded PokerAgents.

    Returns the two factories plus the resolved absolute blueprint paths so
    the run summary records exactly which artifacts played.

    Path validation runs BEFORE the app imports so misconfiguration fails
    fast without touching blueprint machinery.
    """

    a_path = args.blueprint_a or os.environ.get("AGENT_BLUEPRINT_PATH")
    if not a_path or not Path(a_path).is_file():
        raise ValueError("set --blueprint-a or AGENT_BLUEPRINT_PATH to the 8.05B .bps")
    if args.mode == "ab":
        b_path = args.blueprint_b or os.environ.get("AGENT_BLUEPRINT_B_PATH")
        if not b_path:
            raise ValueError(
                "mode 'ab' requires --blueprint-b or AGENT_BLUEPRINT_B_PATH; "
                "refusing to silently fall back to blueprint A (that would run "
                "A-vs-A labeled as A-vs-B)"
            )
        if not Path(b_path).is_file():
            raise ValueError("--blueprint-b / AGENT_BLUEPRINT_B_PATH is not a file")
    else:
        b_path = a_path

    from app.engine.agent.agent import PokerAgent
    from app.engine.agent.modes import ModeConfig
    from app.engine.agent.strategy_source import LookupSource

    a_source = LookupSource(
        ModeConfig(mode="LOOKUP", blueprint_path=str(a_path), uniform_on_miss=True)
    )
    b_source = a_source
    if args.mode == "ab" and os.path.abspath(str(b_path)) != os.path.abspath(str(a_path)):
        b_source = LookupSource(
            ModeConfig(mode="LOOKUP", blueprint_path=str(b_path), uniform_on_miss=True)
        )

    def a_factory(seed: int) -> ActingPolicy:
        cfg = ModeConfig(
            mode="LOOKUP", seed=seed, blueprint_path=str(a_path), uniform_on_miss=True
        )
        return PokerAgent(source=a_source, config=cfg)

    def b_factory(seed: int) -> ActingPolicy:
        cfg = ModeConfig(
            mode="LOOKUP", seed=seed, blueprint_path=str(b_path), uniform_on_miss=True
        )
        return PokerAgent(source=b_source, config=cfg)

    blueprint_paths: Dict[str, Optional[str]] = {
        "blueprint_a": os.path.abspath(str(a_path)),
        "blueprint_b": os.path.abspath(str(b_path)) if args.mode == "ab" else None,
    }
    return a_factory, b_factory if args.mode == "ab" else None, blueprint_paths


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=VALID_MODES, required=True)
    ap.add_argument(
        "--deals",
        type=int,
        default=1000,
        help="independent deal groups (six full duplicate hands per group)",
    )
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--blueprint-a", default=None)
    ap.add_argument("--blueprint-b", default=None)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        a_factory, b_factory, blueprint_paths = _lookup_factories(args)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        # Crash-safe partials: every completed group is appended and flushed
        # immediately, so a crash or spot-death at group N-1 of N keeps the
        # first N-1 groups on disk instead of discarding the whole run.
        partial_path = out / "deal_groups.jsonl.partial"
        record_epsilon = args.epsilon if args.mode == "epsilon" else None
        progress: Dict[str, object] = {"n": 0, "fh": None}

        def on_group(group: DealGroupResult) -> None:
            # Lazy open: a run that fails before its FIRST completed group
            # never touches the partial file, so a bad re-run into the same
            # --out cannot truncate a previous crash's salvageable groups.
            if progress["fh"] is None:
                progress["fh"] = partial_path.open(
                    "w", encoding="utf-8", newline="\n"
                )
            fh = progress["fh"]
            json.dump(
                group.as_dict(
                    mode=args.mode,
                    master_seed=args.master_seed,
                    epsilon=record_epsilon,
                ),
                fh,
                sort_keys=True,
                separators=(",", ":"),
            )
            fh.write("\n")
            fh.flush()
            progress["n"] = int(progress["n"]) + 1
            print(
                f"[match] group {progress['n']}/{args.deals} complete",
                file=sys.stderr,
                flush=True,
            )

        try:
            run = run_match(
                deals=args.deals,
                mode=args.mode,
                master_seed=args.master_seed,
                epsilon=args.epsilon,
                a_factory=a_factory,
                b_factory=b_factory,
                on_group=on_group,
            )
        finally:
            if progress["fh"] is not None:
                progress["fh"].close()
        run = MatchRun(groups=run.groups, summary={**run.summary, **blueprint_paths})
        groups_path, summary_path = write_match(run, args.out)
        partial_path.unlink(missing_ok=True)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"[match] wrote {groups_path} and {summary_path}")
    print("MATCH_SUMMARY " + json.dumps(run.summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
