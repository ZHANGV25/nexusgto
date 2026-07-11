"""Exact integer-chip settlement for complete multiway hold'em hands.

The public contract uses one entry per table seat.  In the canonical 6-max
layout those entries are in ``FULL_SEAT`` order::

    SB=0, BB=1, UTG=2, MP=3, CO=4, BTN=5

``button`` is an index in that same vector.  Odd chips are awarded to tied
winners by scanning clockwise from the seat immediately after the button.
Consequently the default ``button=None`` (equivalent to ``n_seats - 1``)
scans ``0, 1, ...`` and gives the first odd chip to the first winning seat
clockwise from BTN in the canonical full-seat layout.

Contributions are TOTAL chips committed across the hand, including blinds and
chips committed before a fold.  Folded seats fund every layer reached by their
contribution but can never win one.  Only positive LIVE contribution levels
define pot boundaries; a folded contribution is clipped into those intervals
and never creates an artificial side pot.  Awards are gross chips returned
from the pot; deltas are awards minus contributions and therefore sum to zero.

Only an uncontested pot may settle before the river.  A showdown with two or
more live seats requires a five-card board because ``_eval7_py`` evaluates
exactly seven cards (five board cards plus two hole cards).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import List, Optional, Sequence, Tuple

from engine.leaf_values import _eval7_py  # type: ignore[import-untyped]


Cards = Tuple[int, int]


@dataclass(frozen=True)
class SettlementResult:
    """Gross awards and net chip deltas for every input seat."""

    awards: Tuple[int, ...]
    deltas: Tuple[int, ...]
    went_to_showdown: bool


def settle_hand(
    board: Sequence[int],
    hole_cards: Sequence[Cards],
    contributions: Sequence[int],
    folded: Sequence[bool],
    *,
    button: Optional[int] = None,
) -> SettlementResult:
    """Settle an uncontested or showdown pot with exact side-pot layering.

    Args:
        board: Zero, three, four, or five encoded cards.  More than one live
            seat requires all five board cards.
        hole_cards: One two-card hand per seat.  Cards use ``rank * 4 + suit``
            with ranks 2=0 through A=12 and suits 0 through 3.
        contributions: Total non-negative integer chips committed by each
            seat across the whole hand.
        folded: Per-seat fold status; ``False`` means eligible to win.
        button: Button seat index in these vectors.  Odd chips scan clockwise
            from ``button + 1``.  Defaults to the final vector entry, which is
            BTN (seat 5) for the canonical six-entry ``FULL_SEAT`` order.

    Returns:
        A frozen :class:`SettlementResult`.  ``awards`` are gross pot shares;
        ``deltas[i] == awards[i] - contributions[i]``.

    Raises:
        ValueError: Invalid shapes/cards/chips/button, no live player, an
            incomplete showdown board, or an impossible contribution layer
            with no live eligible seat.  The latter represents unreturned,
            uncalled excess and must be corrected before settlement.
    """
    board_cards, hands, contrib, folded_flags, button_idx = _validate_inputs(
        board, hole_cards, contributions, folded, button
    )
    n_seats = len(contrib)
    live = [seat for seat, did_fold in enumerate(folded_flags) if not did_fold]
    pot = sum(contrib)
    awards = [0] * n_seats

    # A completed hand cannot leave a folded player with more committed chips
    # than every live player.  That excess was never called and must be returned
    # before contributions reach settlement.  Validate this even on the
    # uncontested path: blindly awarding uncalled excess would fabricate a loss.
    max_live_contribution = max(contrib[seat] for seat in live)
    for seat, did_fold in enumerate(folded_flags):
        if did_fold and contrib[seat] > max_live_contribution:
            raise ValueError(
                f"folded seat {seat} contribution {contrib[seat]} exceeds "
                f"maximum live contribution {max_live_contribution}; return "
                "uncalled excess before settlement"
            )

    if len(live) == 1:
        awards[live[0]] = pot
        return _result(awards, contrib, went_to_showdown=False)

    if len(board_cards) != 5:
        raise ValueError(
            "showdown with more than one live seat needs a 5-card board, "
            f"got {len(board_cards)}"
        )

    strengths = {
        seat: _eval7_py(list(board_cards) + list(hands[seat])) for seat in live
    }
    clockwise = _clockwise_order(n_seats, button_idx)

    previous = 0
    live_levels = sorted({contrib[seat] for seat in live if contrib[seat] > 0})
    for level in live_levels:
        # Live investments alone delimit main/side pots (the trainer's
        # active-invested levels).  Every seat funds this interval only up to
        # the chips it actually contributed, so folded amounts between live
        # levels join the surrounding live-defined layer instead of splitting
        # it and changing integer odd-chip allocation.
        layer_pot = sum(
            max(0, min(chips, level) - previous) for chips in contrib
        )
        eligible = [seat for seat in live if contrib[seat] >= level]
        if not eligible:
            raise ValueError(
                "contribution layer has no live eligible seat at level "
                f"{level}; return uncalled excess before settlement"
            )

        best = max(strengths[seat] for seat in eligible)
        winners = [seat for seat in eligible if strengths[seat] == best]
        share, odd = divmod(layer_pot, len(winners))
        for seat in winners:
            awards[seat] += share
        if odd:
            winner_set = set(winners)
            ordered_winners = [seat for seat in clockwise if seat in winner_set]
            for seat in ordered_winners[:odd]:
                awards[seat] += 1
        previous = level

    return _result(awards, contrib, went_to_showdown=True)


def _result(
    awards: Sequence[int],
    contributions: Sequence[int],
    *,
    went_to_showdown: bool,
) -> SettlementResult:
    gross = tuple(int(chips) for chips in awards)
    contrib = tuple(int(chips) for chips in contributions)
    pot = sum(contrib)
    assert sum(gross) == pot, (
        f"settlement failed to award the whole pot: awards={sum(gross)} pot={pot}"
    )
    deltas = tuple(gross[i] - contrib[i] for i in range(len(gross)))
    assert sum(deltas) == 0, f"settlement did not conserve chips: deltas={deltas}"
    return SettlementResult(
        awards=gross,
        deltas=deltas,
        went_to_showdown=went_to_showdown,
    )


def _clockwise_order(n_seats: int, button: int) -> Tuple[int, ...]:
    return tuple((button + offset) % n_seats for offset in range(1, n_seats + 1))


def _validate_inputs(
    board: Sequence[int],
    hole_cards: Sequence[Cards],
    contributions: Sequence[int],
    folded: Sequence[bool],
    button: Optional[int],
) -> Tuple[Tuple[int, ...], Tuple[Cards, ...], Tuple[int, ...], Tuple[bool, ...], int]:
    try:
        board_values = tuple(board)
        hand_values = tuple(hole_cards)
        contribution_values = tuple(contributions)
        folded_values = tuple(folded)
    except TypeError as exc:
        raise ValueError("settlement inputs must be finite sequences") from exc

    if len(board_values) not in (0, 3, 4, 5):
        raise ValueError(
            f"board must contain 0, 3, 4, or 5 cards, got {len(board_values)}"
        )

    n_seats = len(contribution_values)
    if len(hand_values) != n_seats or len(folded_values) != n_seats:
        raise ValueError(
            "hole_cards, contributions, and folded must have equal lengths: "
            f"{len(hand_values)}, {n_seats}, {len(folded_values)}"
        )
    if n_seats < 2 or n_seats > 6:
        raise ValueError(f"settlement needs 2 to 6 seats, got {n_seats}")

    normalized_contrib: List[int] = []
    for seat, chips in enumerate(contribution_values):
        if isinstance(chips, bool) or not isinstance(chips, Integral):
            raise ValueError(
                f"contribution for seat {seat} must be an integer, got {chips!r}"
            )
        value = int(chips)
        if value < 0:
            raise ValueError(
                f"contribution for seat {seat} must be non-negative, got {value}"
            )
        normalized_contrib.append(value)

    normalized_folded: List[bool] = []
    for seat, flag in enumerate(folded_values):
        if not isinstance(flag, bool):
            raise ValueError(
                f"folded flag for seat {seat} must be bool, got {flag!r}"
            )
        normalized_folded.append(flag)
    if all(normalized_folded):
        raise ValueError("at least one seat must remain live")

    if button is None:
        button_idx = n_seats - 1
    else:
        if isinstance(button, bool) or not isinstance(button, Integral):
            raise ValueError(f"button must be a seat index, got {button!r}")
        button_idx = int(button)
        if button_idx < 0 or button_idx >= n_seats:
            raise ValueError(
                f"button index {button_idx} out of range for {n_seats} seats"
            )

    normalized_board = tuple(
        _validate_card(card, f"board[{index}]")
        for index, card in enumerate(board_values)
    )
    normalized_hands: List[Cards] = []
    for seat, hand in enumerate(hand_values):
        try:
            pair = tuple(hand)
        except TypeError as exc:
            raise ValueError(f"hole_cards[{seat}] must contain two cards") from exc
        if len(pair) != 2:
            raise ValueError(
                f"hole_cards[{seat}] must contain two cards, got {len(pair)}"
            )
        normalized_hands.append(
            (
                _validate_card(pair[0], f"hole_cards[{seat}][0]"),
                _validate_card(pair[1], f"hole_cards[{seat}][1]"),
            )
        )

    all_cards = list(normalized_board)
    for hand in normalized_hands:
        all_cards.extend(hand)
    if len(set(all_cards)) != len(all_cards):
        raise ValueError("board and hole cards must not contain duplicates")

    return (
        normalized_board,
        tuple(normalized_hands),
        tuple(normalized_contrib),
        tuple(normalized_folded),
        button_idx,
    )


def _validate_card(card: int, label: str) -> int:
    if isinstance(card, bool) or not isinstance(card, Integral):
        raise ValueError(f"{label} must be an integer card, got {card!r}")
    value = int(card)
    if value < 0 or value >= 52:
        raise ValueError(f"{label} out of range [0, 52): {value}")
    return value


__all__ = ["Cards", "SettlementResult", "settle_hand"]
