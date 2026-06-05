"""Preflop action-index prefix constructor (plan §3, user-requested).

Builds the preflop action_history prefix from an EXPLICIT ordered list of
preflop actions, walking the canonical 6-max acting order and emitting
fold0 / call1 / raise-size indices exactly as the trainer
(mccfr_blueprint.c generate_actions) and the production preflop provider do.

This is the GameState.preflop_prefix seed for the agent. It is validated
code-vs-code against preflop_provider._scenario_to_action_path and bit-exact
against the capture-verified prefixes (tests). It deliberately covers only
the CANONICAL single-open shape (open + flat callers), the boundary every
captured line and the canonical builder agree on. Anything beyond — limps,
3bet/4bet/squeeze multiway, >1 raise per actor — is NOT capture-verified and
raises PreflopConstructionUnsupported rather than emit an unverifiable key
(the exact silent-wrong-key failure this session fought).

NOTE: this constructs the index sequence by REASONING about the canonical
tree, which is acceptable ONLY because it is bit-exact validated against two
independent ground truths (capture + production builder). Do not extend it to
unvalidated shapes without new ground truth.
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
# vocabs (preflop_provider documents this; we serve a single-open line so the
# smallest raise index is the only one we emit).
_FOLD = 0
_CALL = 1
_RAISE = 2


class PreflopConstructionUnsupported(Exception):  # noqa: N818 - matches sibling PreflopNodeNotFound naming
    """The preflop line is outside the capture-verified canonical shape.

    v1 covers a single open + flat callers. Limps, re-raises (3bet+), squeezes
    and other multiway shapes are deferred (plan §6) — building their index
    sequence by reasoning would be unverifiable and risk a silent wrong key.
    """


@dataclass(frozen=True)
class PreflopAction:
    """One voluntary preflop action.

    position: seat name (UTG..BB).
    action: "fold" | "call" | "raise" (limp is rejected upstream).
    amount: chips committed (informational; the index doesn't encode size for
        the single-open canonical shape).
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
        PreflopConstructionUnsupported: a non-canonical shape (limp, >1 raise,
            out-of-order action, multiple raises).
    """
    if not actions:
        raise ValueError("actions must be non-empty")
    for a in actions:
        if a.position not in _ORDER_SET:
            raise ValueError(
                f"unknown position {a.position!r}; expected one of {PREFLOP_ORDER}"
            )

    # Map position -> its voluntary action. Reject duplicate actors (a 3bet/
    # 4bet line has the same actor acting twice — unsupported in v1).
    acted = {}
    for a in actions:
        if a.position in acted:
            raise PreflopConstructionUnsupported(
                f"{a.position} acts more than once (3bet/4bet/squeeze line "
                f"not capture-verified in v1)"
            )
        acted[a.position] = a

    # Actions must be in canonical acting order.
    order_idx = [PREFLOP_ORDER.index(a.position) for a in actions]
    if order_idx != sorted(order_idx):
        raise PreflopConstructionUnsupported(
            "preflop actions are not in canonical acting order "
            "(UTG,MP,CO,BTN,SB,BB) — non-canonical multiway line"
        )

    # Exactly one raiser (the opener). Zero raisers = open-limp pot
    # (unsupported); >1 = re-raise line (rejected by duplicate-actor check
    # only if same actor; different-actor 3bet caught here).
    raisers = [a for a in actions if a.action == "raise"]
    if len(raisers) == 0:
        raise PreflopConstructionUnsupported(
            "no raise in the preflop line (open-limp pots are not "
            "capture-verified in v1)"
        )
    if len(raisers) > 1:
        raise PreflopConstructionUnsupported(
            "more than one raise (3bet+/squeeze) is not capture-verified in v1"
        )

    opener_idx = PREFLOP_ORDER.index(raisers[0].position)
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
                    f"capture-verified in v1"
                )
            prefix.append(_CALL)
        elif a.action == "raise":
            prefix.append(_RAISE)
        else:
            raise PreflopConstructionUnsupported(
                f"unsupported preflop action {a.action!r} for {pos}"
            )
    return prefix
