"""Preflop action-index prefix constructor (plan §3, user-requested).

Builds the preflop action_history prefix from an EXPLICIT ordered list of
preflop actions, walking the canonical 6-max acting order and emitting
fold0 / call1 / raise-size indices exactly as the trainer
(mccfr_blueprint.c generate_actions) and the production preflop provider do.

This is the GameState.preflop_prefix seed for the agent. It covers TWO
capture-verified shapes:

  1. CANONICAL SINGLE-OPEN (open + flat callers) — validated code-vs-code
     against preflop_provider._scenario_to_action_path and bit-exact against
     the capture-verified prefixes (tests, 2026-06-04).

  2. 3-BET HEADS-UP (open → 3-bet → folds back → opener calls → flop HU) —
     capture-verified 2026-06-10 for ALL 15 (opener, 3-bettor) seat pairs:
     the literal trainer C function (bp_generate_actions) replayed every
     line's index sequence + chips, and the 8.05B blueprint HIT present +
     trained σ̄ at every flop-root key with wrong-seat / truncated /
     extended / out-of-vocab-index discriminators MISSING. Evidence:
     poker-solver scripts/preflop_shapes_captured.json (+ the EC2-gated
     oracle replay tests). v1 maps EVERY 3-bet to index 2 — the SMALLEST
     (0.7x) trainer size, tier 1 of PREFLOP_TIERS {0:[0.7,1.0], 1:[0.7,1.0],
     2:[1.0]} — the same smallest-size convention the open uses; the 1.0x
     3-bet (index 3) is trained in the blueprint but not wired here.

Anything beyond — limps, squeezes (callers + 3-bet), 4-bet+, multiway 3-bet
pots — is NOT capture-verified and raises PreflopConstructionUnsupported
rather than emit an unverifiable key (the exact silent-wrong-key failure
this session fought).

NOTE: the index sequences are constructed by REASONING about the canonical
tree, which is acceptable ONLY because every covered shape is bit-exact
validated against independent ground truth (trainer-C replay + blueprint
probes). Do not extend to unvalidated shapes without new ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

# Canonical 6-max preflop acting order (matches
# preflop_provider._PREFLOP_POSITIONS and the trainer's enumeration).
PREFLOP_ORDER = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
_ORDER_SET = set(PREFLOP_ORDER)

# Action indices when FACING a bet (preflop is always facing the blinds, so
# the first raiser already "faces" the BB): [fold0, call1, raise2(smallest)].
# The smallest-raise = index 2 invariant holds across all blueprint tier
# vocabs (preflop_provider documents this; capture-verified at the vs-open
# node too: [fold, call, 3bet-0.7x, 3bet-1.0x] with na=4, so the smallest
# 3-bet is also index 2).
_FOLD = 0
_CALL = 1
_RAISE = 2


class PreflopConstructionUnsupported(Exception):  # noqa: N818 - matches sibling PreflopNodeNotFound naming
    """The preflop line is outside the capture-verified shapes.

    Covered: single open + flat callers; 3-bet heads-up (smallest size).
    Limps, squeezes, 4-bet+, multiway 3-bet pots and other shapes are
    deferred (plan §6) — building their index sequence by reasoning would be
    unverifiable and risk a silent wrong key.
    """


@dataclass(frozen=True)
class PreflopAction:
    """One voluntary preflop action.

    position: seat name (UTG..BB).
    action: "fold" | "call" | "raise" (limp is rejected upstream).
    amount: chips committed (informational; the index doesn't encode size —
        both the open and the 3-bet map to the SMALLEST trainer size,
        index 2).
    """

    position: str
    action: str
    amount: float = 0.0


def build_preflop_prefix(actions: List[PreflopAction]) -> List[int]:
    """Construct the preflop action-index prefix for `actions`.

    `actions` is the ordered list of VOLUNTARY actions (raises/calls);
    intervening folds are inferred from acting-order gaps. The returned list
    is the preflop portion of action_history (one index per action taken,
    including inferred folds).

    Raises:
        ValueError: empty list, or an unknown position.
        PreflopConstructionUnsupported: a non-capture-verified shape (limp,
            squeeze, 4bet+, out-of-order action).
    """
    if not actions:
        raise ValueError("actions must be non-empty")
    for a in actions:
        if a.position not in _ORDER_SET:
            raise ValueError(
                f"unknown position {a.position!r}; expected one of {PREFLOP_ORDER}"
            )

    raisers = [a for a in actions if a.action == "raise"]
    if len(raisers) == 0:
        raise PreflopConstructionUnsupported(
            "no raise in the preflop line (open-limp pots are not "
            "capture-verified)"
        )
    if len(raisers) == 1:
        return _single_open_prefix(actions)
    if len(raisers) == 2:
        return _threebet_hu_prefix(actions, raisers)
    raise PreflopConstructionUnsupported(
        "more than two raises (4bet+) is not capture-verified"
    )


# ── Shape 1: canonical single open + flat callers (verified 2026-06-04) ───


def _single_open_prefix(actions: List[PreflopAction]) -> List[int]:
    # Map position -> its voluntary action. Reject duplicate actors (with a
    # single raise, nobody legally acts twice before the round closes).
    acted = {}
    for a in actions:
        if a.position in acted:
            raise PreflopConstructionUnsupported(
                f"{a.position} acts more than once without a re-raise — not "
                f"a canonical single-open line"
            )
        acted[a.position] = a

    # Actions must be in canonical acting order.
    order_idx = [PREFLOP_ORDER.index(a.position) for a in actions]
    if order_idx != sorted(order_idx):
        raise PreflopConstructionUnsupported(
            "preflop actions are not in canonical acting order "
            "(UTG,MP,CO,BTN,SB,BB) — non-canonical multiway line"
        )

    opener_idx = PREFLOP_ORDER.index(
        next(a.position for a in actions if a.action == "raise")
    )
    last_actor_idx = max(order_idx)

    prefix: List[int] = []
    for seat_idx in range(last_actor_idx + 1):
        pos = PREFLOP_ORDER[seat_idx]
        a = acted.get(pos)
        if a is None:
            # Seat between/ before the actors that didn't voluntarily act:
            # they folded (everyone folds around the single open).
            prefix.append(_FOLD)
            continue
        if a.action == "fold":
            prefix.append(_FOLD)
        elif a.action == "call":
            # A call BEFORE the opener acts = a limp (no bet to call yet) →
            # unsupported. After the opener it's a flat call of the open.
            if seat_idx < opener_idx:
                raise PreflopConstructionUnsupported(
                    f"{pos} calls before the opener (limp) — not "
                    f"capture-verified"
                )
            prefix.append(_CALL)
        elif a.action == "raise":
            prefix.append(_RAISE)
        else:
            raise PreflopConstructionUnsupported(
                f"unsupported preflop action {a.action!r} for {pos}"
            )
    return prefix


# ── Shape 2: 3-bet heads-up (capture-verified 2026-06-10) ─────────────────


def _threebet_hu_prefix(
    actions: List[PreflopAction], raisers: List[PreflopAction],
) -> List[int]:
    """open → 3-bet → folds back → opener calls, heads-up to the flop.

    The emitted prefix is always length 7: one index per seat in the first
    orbit (UTG..BB — folds for everyone but the opener and the 3-bettor),
    then the opener's closing call. Capture-verified for all 15
    (opener, 3-bettor) pairs — e.g. BTN-open/BB-3bet → [0,0,0,2,0,2,1],
    UTG-open/BTN-3bet → [2,0,0,2,0,0,1].
    """
    opener, threebettor = raisers[0].position, raisers[1].position
    if opener == threebettor:
        raise PreflopConstructionUnsupported(
            f"{opener} raises twice (limp-reraise or 4-bet shape) — not "
            f"capture-verified"
        )
    if PREFLOP_ORDER.index(threebettor) <= PREFLOP_ORDER.index(opener):
        raise PreflopConstructionUnsupported(
            f"{threebettor} re-raises from an earlier seat than the opener "
            f"{opener} (limp-reraise shape) — not capture-verified"
        )

    calls = [a for a in actions if a.action == "call"]
    if len(calls) != 1 or calls[0].position != opener:
        raise PreflopConstructionUnsupported(
            "3-bet line must close with EXACTLY the opener's call (squeeze/"
            "cold-call/multiway 3-bet pots are not capture-verified; an "
            "opener fold ends the hand preflop — no flop key exists)"
        )

    folds = [a for a in actions if a.action == "fold"]
    fold_seen: set = set()
    for f in folds:
        if f.position in (opener, threebettor):
            raise PreflopConstructionUnsupported(
                f"{f.position} both folds and raises/calls — inconsistent line"
            )
        if f.position in fold_seen:
            raise PreflopConstructionUnsupported(
                f"{f.position} folds twice — inconsistent line"
            )
        fold_seen.add(f.position)
    if any(a.action not in ("fold", "call", "raise") for a in actions):
        bad = next(a for a in actions if a.action not in ("fold", "call", "raise"))
        raise PreflopConstructionUnsupported(
            f"unsupported preflop action {bad.action!r} for {bad.position}"
        )

    # Chronological-order validation: orbit-1 rank = seat index; the opener's
    # closing call ranks AFTER the whole orbit.
    def _rank(a: PreflopAction) -> int:
        if a.action == "call" and a.position == opener:
            return len(PREFLOP_ORDER)
        return PREFLOP_ORDER.index(a.position)

    ranks = [_rank(a) for a in actions]
    if ranks != sorted(ranks):
        raise PreflopConstructionUnsupported(
            "preflop actions are not in chronological acting order for a "
            "3-bet heads-up line (UTG..BB orbit, then the opener's call)"
        )

    # Emission: full first orbit + the opener's closing call (length 7).
    prefix = []
    for pos in PREFLOP_ORDER:
        if pos == opener or pos == threebettor:
            prefix.append(_RAISE)
        else:
            prefix.append(_FOLD)
    prefix.append(_CALL)
    return prefix
