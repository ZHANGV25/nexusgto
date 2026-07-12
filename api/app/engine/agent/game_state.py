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

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

# Action-set COUNT comes from the blueprint's trained vocab (metadata-driven,
# see action_vocab); the category SEMANTICS (fold/call/raise) reuse the
# solver's enum. The seat / who-acts-next logic lives below, unchanged.
from app.engine.agent.action_vocab import (
    DEFAULT_ADD_EXPLICIT_ALLIN,
    DEFAULT_POSTFLOP_BET_SIZES,
    DEFAULT_SUBSEQUENT_BET_SIZES,
    legal_categories_chips,
    raise_to_amount,
)
from app.engine.agent.preflop_play import (
    PreflopReplayError,
    PreflopRules,
    SHIPPED_PREFLOP_RULES,
    preflop_raise_amount_f32 as _trainer_preflop_raise_amount_f32,
    replay_preflop,
)

from engine.rollout import ActionCategory  # type: ignore[import-untyped]


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

# Trainer chip convention (8.05B metadata, capture-verified). Load-bearing for
# the stack-commitment vocab prune (3a): the flop-entry pot/stacks must equal
# the trainer's canonical postflop chips, or the prune fires at the wrong nodes.
SMALL_BLIND = 50
BIG_BLIND = 100
INITIAL_STACK = 10000
# Preflop OPEN size = the smallest preflop bet size (tier 0, index 2 action).
# Verified: with this convention the chip-aware vocab length matches the
# blueprint σ̄ length at EVERY node along the raise-war line, both prefixes.
PREFLOP_OPEN_FRACTION = 0.7
# The LARGE 3-bet size (tier 1, index 3 action = 1.0x). Chips for the
# index-3 3-bet are capture-verified for BTN-open/BB-3bet (2026-06-10,
# preflop_shapes_captured.json: pot 1800, stacks 9125/9125, σ̄-length walks
# clean); wider pair coverage runs in the Lane-C probe. SERVING the 1.0x
# path is gated in engine_translation until those probes return.
PREFLOP_THREEBET_LARGE_FRACTION = 1.0

# Preflop acting order (matches preflop_constructor.PREFLOP_ORDER / the trainer).
_PREFLOP_ORDER = ("UTG", "MP", "CO", "BTN", "SB", "BB")


class CanonicalChipsUnsupported(Exception):  # noqa: N818 - matches sibling naming
    """The preflop line is outside the canonical verified shapes, so its
    flop-entry pot/stacks cannot be derived from the verified convention.

    Chips derive only for the capture-verified shapes (see
    canonical_flop_chips: single open + flat callers; 3-bet HU; limp-check
    pots). Anything else must pass explicit pot/stacks that the caller has
    independently validated.
    """


def preflop_raise_amount_f32(
    frac: float, pot: int, to_call: int, stack: int,
) -> int:
    """Preflop raise INCREMENT, bit-exact to the trainer's FLOAT32 arithmetic.

    The trainer (mccfr_blueprint.c generate_actions) computes
    `ba = to_call + (int)(bet_sizes[i] * (pot + to_call))` with bet_sizes[i]
    a C float (float32). Python doubles DIVERGE on some preflop nodes:
    (int)(0.7f * 700) == 490 in C but int(0.7 * 700) == 489 in doubles —
    the cold-3-bet vs-UTG-open node (capture-verified against the literal
    bp_generate_actions C function, poker-solver
    scripts/probe_preflop_shapes.py, 2026-06-10). float32(0.7) has 24
    significant bits, so its product with a chip count (< 2^20) is exact in
    a double; rounding that product back to float32 reproduces the C single
    multiply (mulss) bit-exactly.

    Postflop sizes (0.75, 1.0) are exactly representable in binary, so the
    double-based raise_to_amount never diverges postflop and stays unchanged.
    """
    return _trainer_preflop_raise_amount_f32(frac, pot, to_call, stack)


def canonical_flop_chips(
    preflop_prefix: List[int],
    seats: List[str],
    *,
    rules: Optional[PreflopRules] = None,
) -> Tuple[int, Tuple[int, ...]]:
    """Derive the flop-entry (pot, per-slot stacks) for a canonical preflop line.

    Walks the preflop action sequence through the TRAINER's amount formula
    (preflop_raise_amount_f32 — float32, bit-exact to bp_generate_actions)
    with the [50,100]/10000 convention — the 0.7 (smallest) size for raise
    index 2, the 1.0x size for the index-3 3-bet — then returns the pot and
    the per-slot (SB-first over `seats`) remaining stacks.

    With ``rules=None`` (the backward-compatible default), supported
    capture-verified shapes are:
      * single open + flat callers — VERIFIED (600, (9725,9725)) 2p and
        (875, (9725,9725,9725)) 3p, matching the blueprint σ̄ length along
        the raise-war line (2026-06-04).
      * 3-bet HEADS-UP: open → 3-bet (index 2 = 0.7x OR index 3 = 1.0x) →
        folds back → opener calls. The 0.7x path VERIFIED 2026-06-10
        against the literal trainer C amounts (bp_generate_actions walk,
        all 15 opener/3-bettor pairs) + blueprint σ̄-length walks to the
        river all-in clamp: e.g. BTN-open/BB-3bet → (1440, (9305, 9305));
        UTG-open/BTN-3bet → (1680, (9235, 9235)). The 1.0x path verified
        for BTN-open/BB-3bet → (1800, (9125, 9125)); wider 1.0x pair
        coverage pending the Lane-C probe (serving is gated upstream).
        Evidence: poker-solver scripts/preflop_shapes_captured.json.
      * LIMP-CHECK pots (no raise; the BB's check — index 0 at the
        BB-option node — closes the round). Pure blind arithmetic, pinned
        to the 2026-06-11 trainer walks (limp_shapes_captured.json):
        UTG-limp/BTN-limp → (250, (9900, 9900)); SB-complete →
        (200, (9900, 9900)); the root-captured 3-way walks UTG+BTN →
        (350, (9900,)*3) and UTG+SB → (300, (9900,)*3). SERVING is
        restricted to the three verified HU lines upstream
        (preflop_constructor only emits those); this layer derives chips
        for any closed limp-check prefix.
      * Historical compatibility: the legacy walker also accepts a closed
        limp-then-single-raise/isolation line as its single-raised branch,
        despite older prose calling every limp-raise unsupported. That
        accepted behavior is intentionally preserved.

    With explicit ``rules``, the shared trainer-generic replay accepts every
    closed prefix in that tiered vocabulary (including 1.0x opens, isolation
    raises, squeezes, multiway 3-bets, and capped 4-bets). It still rejects an
    invalid action index, an unclosed prefix, a live-seat mismatch, or an
    uncontested hand (there is no flop entry).

    Raises:
        CanonicalChipsUnsupported: with ``rules=None``, a non-canonical
            preflop shape (4bet+, a 1.0x open, an out-of-vocab raise index,
            a 3-bet pot that isn't heads-up, or an unclosed limped pot); with
            explicit ``rules``, a prefix that cannot close
            through that trainer vocabulary or has no flop entry.
    """
    for s in seats:
        if s not in FULL_SEAT:
            raise ValueError(f"unknown position {s!r}")
    live_seats = set(seats)

    # Opt-in trainer-generic replay for the shipped game-loop vocabulary.
    # The no-rules path below intentionally retains its capture-limited
    # rejection contract (index-3 opens, 4-bets, squeezes/multiway 3-bets).
    # there is no second chip-arithmetic implementation to drift.
    if rules is not None:
        if rules != SHIPPED_PREFLOP_RULES:
            raise CanonicalChipsUnsupported(
                "generic flop-entry validation supports only "
                "SHIPPED_PREFLOP_RULES"
            )
        try:
            replayed = replay_preflop(
                preflop_prefix, rules, require_terminal=True
            )
        except PreflopReplayError as exc:
            raise CanonicalChipsUnsupported(str(exc)) from exc
        if replayed.uncontested:
            raise CanonicalChipsUnsupported(
                "preflop ended uncontested; no flop-entry state exists"
            )
        replay_live = set(replayed.live_seat_names)
        if replay_live != live_seats:
            raise ValueError(
                f"seats {sorted(live_seats)} do not match the live players "
                f"of the replayed preflop line {sorted(replay_live)}"
            )
        order = sorted(live_seats, key=lambda seat: FULL_SEAT[seat])
        return replayed.pot, tuple(
            replayed.stacks[FULL_SEAT[seat]] for seat in order
        )

    # Per-position street commitment (preflop). Blinds posted first.
    bets: Dict[str, int] = {pos: 0 for pos in _PREFLOP_ORDER}
    bets["SB"] = SMALL_BLIND
    bets["BB"] = BIG_BLIND
    pot = SMALL_BLIND + BIG_BLIND
    current_bet = BIG_BLIND
    raises_seen = 0
    folded: set = set()
    acted: set = set()
    bb_checked = False

    # The prefix is one index per preflop action taken, in the trainer's
    # owing-action order: a first orbit UTG..BB, then wrapping back to anyone
    # who hasn't matched the current bet (the opener facing a 3-bet). It may
    # be truncated after the last voluntary actor (single-open shape).
    idx = 0
    n = len(preflop_prefix)
    for _orbit in range(3):  # 2 raises close within two orbits; 3rd is a guard
        if idx >= n:
            break
        for pos in _PREFLOP_ORDER:
            if idx >= n:
                break
            if pos in folded:
                continue
            owes = (pos not in acted) or (bets[pos] < current_bet)
            if not owes:
                continue
            action = preflop_prefix[idx]
            idx += 1
            acted.add(pos)
            if bets[pos] == current_bet:
                # NOTHING TO CALL: the BB-option node of a limped pot (the
                # only preflop node with to_call == 0 — every other actor
                # faces at least the blind). Trainer layout (capture-verified
                # 2026-06-11): [check, 0.7x raise, 1.0x raise], check_idx=0.
                if action == 0:  # check — closes the limped pot
                    bb_checked = True
                    continue
                raise CanonicalChipsUnsupported(
                    f"raise index {action} at the BB-option node (a "
                    f"limp-RAISE pot) — chips not capture-verified"
                )
            if action == 0:  # fold
                folded.add(pos)
                continue
            if action == 1:  # call (limp / flat / close the 3-bet pot)
                to_call = current_bet - bets[pos]
                bets[pos] += to_call
                pot += to_call
            elif action in (2, 3):  # raise — 0.7x (idx 2) or 1.0x (idx 3)
                if raises_seen >= 2:
                    raise CanonicalChipsUnsupported(
                        "more than two preflop raises (4bet+) — chips not "
                        "capture-verified; pass explicit pot/stacks"
                    )
                if action == 3 and raises_seen == 0:
                    # Index 3 at the OPEN = a 1.0x open — NOT capture-verified
                    # (only the 0.7x open and the {0.7x, 1.0x} 3-bet are).
                    raise CanonicalChipsUnsupported(
                        "1.0x OPEN (index 3 at the first raise) — chips not "
                        "capture-verified; only the smallest (0.7x) open is"
                    )
                frac = (
                    PREFLOP_OPEN_FRACTION if action == 2
                    else PREFLOP_THREEBET_LARGE_FRACTION
                )
                to_call = current_bet - bets[pos]
                amt = preflop_raise_amount_f32(
                    frac, pot=pot, to_call=to_call,
                    stack=INITIAL_STACK - bets[pos],
                )
                bets[pos] += amt
                pot += amt
                current_bet = bets[pos]
                raises_seen += 1
            else:
                raise CanonicalChipsUnsupported(
                    f"unsupported preflop action index {action} (raise "
                    f"indices 2 = 0.7x and 3 = 1.0x-3-bet are the "
                    f"capture-verified vocabulary)"
                )

    if raises_seen == 0:
        # LIMP-CHECK pot: valid only when the BB's check CLOSED the round
        # with every action consumed and >= 2 live players all matched to
        # the blind (capture-pinned chips, limp_shapes_captured.json).
        if not bb_checked or idx != n:
            raise CanonicalChipsUnsupported(
                "limped pot without the BB's check closing the round "
                "(unclosed / overlong prefix) — chips not capture-verified"
            )
        replay_live = {
            pos for pos in _PREFLOP_ORDER
            if pos not in folded and pos in acted
        }
        if len(replay_live) < 2 or any(
            bets[pos] != current_bet for pos in replay_live
        ):
            raise CanonicalChipsUnsupported(
                "limped pot with fewer than 2 live players (or an unmatched "
                "blind) — no flop entry exists"
            )
        if replay_live != live_seats:
            raise ValueError(
                f"seats {sorted(live_seats)} do not match the live players "
                f"of the replayed limp-check line {sorted(replay_live)}"
            )
    if raises_seen == 2:
        # 3-bet pots are capture-verified HEADS-UP only (opener vs 3-bettor,
        # everyone else folded, opener's call closing the round).
        replay_live = {
            pos for pos in _PREFLOP_ORDER
            if pos not in folded and pos in acted
        }
        if len(replay_live) != 2 or any(
            bets[pos] != current_bet for pos in replay_live
        ):
            raise CanonicalChipsUnsupported(
                "3-bet pot that is not a closed heads-up line (squeeze/"
                "multiway/unclosed) — chips not capture-verified"
            )
        if replay_live != live_seats:
            raise ValueError(
                f"seats {sorted(live_seats)} do not match the live players "
                f"of the replayed 3-bet line {sorted(replay_live)}"
            )
    if raises_seen == 1:
        # Single-raised pot: the replayed live set must BE the seats the
        # caller claims reach the flop — the same invariant the limped
        # (raises_seen==0) and 3-bet (==2) branches above enforce. This
        # branch historically had NO check, which let every A3/Stage-2
        # harness spot since 2026-06-12 silently build a FICTIONAL state
        # (all six seats "live" over the heads-up prefix [2,0,0,0,0,1]):
        # its blueprint/boundary keys encode lines the trainer can never
        # traverse, so every cont_ev query 0-hit every walk table (the
        # Stage-2 hash-disjunction root cause, 2026-07-03).
        replay_live = {
            pos for pos in _PREFLOP_ORDER
            if pos not in folded and pos in acted
        }
        if replay_live != live_seats:
            raise ValueError(
                f"seats {sorted(live_seats)} do not match the live players "
                f"of the replayed single-raise line {sorted(replay_live)}"
            )

    # Per-slot stacks, SB-first over the LIVE seats (matches new_postflop_state
    # slot ordering). Each live seat invested bets[pos]; folded seats aren't at
    # the flop. The pot already includes folded blinds (SB's dead 50, etc.).
    order = sorted(live_seats, key=lambda s: FULL_SEAT[s])
    stacks = tuple(INITIAL_STACK - bets[pos] for pos in order)
    return pot, stacks


def canonical_threebet_increments(
    opener: str, threebettor: str,
) -> Tuple[int, int]:
    """The trainer's canonical 3-bet chip INCREMENTS (0.7x, 1.0x) at the
    3-bettor's vs-open node for this (opener, 3-bettor) pair.

    Replays the canonical heads-up 3-bet first orbit — blinds posted, folds
    to the opener, the 0.7x f32 open, folds to the 3-bettor — and evaluates
    both tier-1 sizes through preflop_raise_amount_f32 (bit-exact to
    bp_generate_actions). These are the candidates the engine-translation
    layer snaps a caller's 3-bet amount onto. E.g. BTN-open/BB-3bet →
    (595, 775); UTG-open/BTN-3bet (cold) → (765, 975).

    Raises:
        ValueError: unknown position, opener == threebettor, or a
            3-bettor seated BEFORE the opener (limp-reraise shape — not a
            vs-open node).
    """
    for s in (opener, threebettor):
        if s not in FULL_SEAT:
            raise ValueError(f"unknown position {s!r}")
    oi = _PREFLOP_ORDER.index(opener)
    ti = _PREFLOP_ORDER.index(threebettor)
    if ti <= oi:
        raise ValueError(
            f"3-bettor {threebettor} does not act after the opener {opener} "
            f"in the canonical orbit (limp-reraise shape)"
        )

    bets: Dict[str, int] = {pos: 0 for pos in _PREFLOP_ORDER}
    bets["SB"] = SMALL_BLIND
    bets["BB"] = BIG_BLIND
    pot = SMALL_BLIND + BIG_BLIND
    current_bet = BIG_BLIND

    open_amt = preflop_raise_amount_f32(
        PREFLOP_OPEN_FRACTION,
        pot=pot,
        to_call=current_bet - bets[opener],
        stack=INITIAL_STACK - bets[opener],
    )
    bets[opener] += open_amt
    pot += open_amt
    current_bet = bets[opener]

    to_call = current_bet - bets[threebettor]
    stack = INITIAL_STACK - bets[threebettor]
    return (
        preflop_raise_amount_f32(
            PREFLOP_OPEN_FRACTION, pot=pot, to_call=to_call, stack=stack,
        ),
        preflop_raise_amount_f32(
            PREFLOP_THREEBET_LARGE_FRACTION, pot=pot, to_call=to_call,
            stack=stack,
        ),
    )


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
        street_history: action indices taken on the CURRENT street only
            (a suffix of postflop_history; reset when a betting round
            closes). Live-resolve node replay reads this — it is tracked
            here so callers never re-derive street segmentation by hand.
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
    street_history: Tuple[int, ...] = ()
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

        Chip-aware (3a): the count is also pruned by STACK COMMITMENT exactly
        as the trainer's generate_actions does — a raise slot is dropped when
        the raise-to amount can't exceed the call (committed) and collapsed to
        one all-in when it reaches the stack. This is what makes num_legal
        match the blueprint σ̄ length on committed lines (no mid-hand crash).
        On deep stacks the prune never fires, so the set is identical to before.
        """
        slot = self.to_act_slot
        if slot is None:
            # Terminal — no actor. Return empty (callers guard on terminal).
            return []
        pot = int(self.pot)
        stack = int(self.stacks[slot]) if slot < len(self.stacks) else INITIAL_STACK
        to_call = max(0, self.current_bet - self.street_bets[slot])
        return legal_categories_chips(
            facing_bet=self.facing_bet,
            raises_so_far=self.raises_this_street,
            pot=pot,
            stack=stack,
            to_call=to_call,
            postflop_bet_sizes=self.postflop_bet_sizes,
            subsequent_bet_sizes=self.subsequent_bet_sizes,
            add_explicit_allin=self.add_explicit_allin,
        )

    def num_legal_actions(self) -> int:
        return len(self.legal_action_categories())

    def _surviving_raise_amounts(self, slot: int) -> List[int]:
        """The canonical raise INCREMENTS available to `slot` at this node,
        bit-exact to the trainer's generate_actions prune (drop raises that
        don't exceed the call; clamp+dedup at the stack; optional explicit
        all-in last). Position i here == raise slot i of
        legal_action_categories()."""
        pot = int(self.pot)
        stack = int(self.stacks[slot])
        to_call = max(0, self.current_bet - self.street_bets[slot])
        sizes = (
            self.subsequent_bet_sizes
            if self.raises_this_street > 0 else self.postflop_bet_sizes
        )
        survivors: List[int] = []
        added_allin = False
        for frac in sizes:
            ba = raise_to_amount(frac, pot=pot, to_call=to_call, stack=stack)
            if ba <= to_call:
                continue
            if ba >= stack:
                if added_allin:
                    continue
                added_allin = True
            survivors.append(ba)
        if self.add_explicit_allin and not added_allin and stack > to_call:
            survivors.append(stack)
        return survivors

    def _raise_amount_for(self, action_idx: int, slot: int) -> int:
        """Chip INCREMENT for the raise at `action_idx`, bit-exact to the
        trainer. action_idx indexes legal_action_categories(); the raise slots
        come AFTER the non-raise slots (fold?/call)."""
        n_nonraise = (1 if self.facing_bet else 0) + 1  # fold? + check/call
        raise_position = action_idx - n_nonraise
        if raise_position < 0:
            raise ValueError(
                f"action_idx {action_idx} is not a raise (n_nonraise="
                f"{n_nonraise})"
            )
        survivors = self._surviving_raise_amounts(slot)
        if raise_position >= len(survivors):
            # Should be unreachable — the vocab and this share the prune logic.
            raise ValueError(
                f"raise position {raise_position} exceeds {len(survivors)} "
                f"surviving raises (vocab/amount prune disagreement)"
            )
        return survivors[raise_position]

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
        stacks = list(self.stacks)
        current_bet = self.current_bet
        raises = self.raises_this_street
        pot = self.pot

        acted[slot] = True

        if cat == ActionCategory.FOLD:
            active[slot] = False
        elif cat == ActionCategory.CALL:
            # CALL covers both "call a bet" and "check". For a real call, move
            # REAL chips to match the outstanding bet (clamped to the stack) and
            # decrement the stack — chip accounting is now load-bearing because
            # the legal-action vocab prunes by commitment (3a).
            to_match = min(current_bet - street_bets[slot], stacks[slot])
            if to_match > 0:
                street_bets[slot] += to_match
                stacks[slot] -= to_match
                pot += to_match
        else:  # RAISE (covers bet and raise)
            # A bet/raise opens a new round. We do NOT reset other players'
            # `acted` flags — the `street_bets[q] < current_bet` predicate in
            # _next_to_act makes everyone who hasn't matched owe action
            # automatically (capture-verified: forward-from-bettor order, the
            # blueprint MISSES rollout's scan-from-slot0 order).
            #
            # Use the trainer's REAL raise sizing (bit-exact to generate_actions
            # via raise_to_amount), so pot/stack/to_call evolve correctly and
            # the next node's vocab prunes at the right depth. `amt` is the chip
            # INCREMENT added to this player's street bet.
            amt = self._raise_amount_for(action_idx, slot)
            street_bets[slot] += amt
            stacks[slot] -= amt
            pot += amt
            current_bet = street_bets[slot]
            raises += 1

        return self._after_action(
            slot=slot, action_idx=action_idx, active=active, acted=acted,
            street_bets=street_bets, stacks=stacks, current_bet=current_bet,
            raises=raises, pot=pot,
        )

    def _after_action(
        self,
        *,
        slot: int,
        action_idx: int,
        active: List[bool],
        acted: List[bool],
        street_bets: List[int],
        stacks: List[int],
        current_bet: int,
        raises: int,
        pot: int,
    ) -> "GameState":
        """Shared post-action tail for advance()/advance_observed():
        record the action index in both histories, find the next actor
        (forward-from-actor — the order the blueprint encodes), or close
        the round / terminate."""
        new_history = self.postflop_history + (action_idx,)
        new_street_history = self.street_history + (action_idx,)

        next_slot = self._next_to_act(
            active, acted, street_bets, current_bet, stacks, start_after=slot,
        )

        if next_slot is not None or self._one_left(active):
            # Either there is a next actor, or the hand collapsed to one
            # player (terminal). No street advance yet.
            return replace(
                self,
                active=tuple(active),
                acted=tuple(acted),
                street_bets=tuple(street_bets),
                stacks=tuple(stacks),
                current_bet=current_bet,
                raises_this_street=raises,
                pot=pot,
                postflop_history=new_history,
                street_history=new_street_history,
                to_act_slot=(None if self._one_left(active) else next_slot),
            )

        # No next actor AND >1 player live => the betting round CLOSED.
        return self._close_round(
            active=active, stacks=stacks, pot=pot, history=new_history,
            closed_count=self.betting_round_closed_count + 1,
        )

    def advance_observed(
        self, action: str, amount: Optional[int] = None,
    ) -> "ObservedAdvance":
        """Apply an OBSERVED (real-table, possibly off-tree) action.

        The split that makes a lookup/replay agent playable against humans:
        the HISTORIES record the SNAPPED on-tree action index (so blueprint
        keys and the live-resolve street replay stay on-tree), while the
        CHIP STATE applies the REAL amount (so pot odds, stack-commitment
        prunes and street-entry reconstruction stay faithful to the table).
        The residual divergence — the solved tree's canonical in-street
        amounts vs the real ones — is the standard action-abstraction gap
        (Pluribus closed it with live search; a lookup/replay agent accepts
        it and snaps).

        Args:
            action: "fold" | "check" | "call" | "raise" | "allin".
            amount: for "raise": the actor's chip INCREMENT this action
                (chips added on top of their current street bet — a bet of
                500 into a 600 pot, or a raise BY 1200 over the call).
                Ignored for the other actions ("allin" derives from stack).

        Snapping: the increment maps to the NEAREST surviving canonical
        raise (relative/log distance) from the same prune the vocab uses —
        position i == raise slot i of legal_action_categories(). An
        increment that doesn't exceed the call (under-raise / short
        all-in-call) is applied as a CALL.

        Raises:
            ValueError: terminal state; "fold"/"check" inconsistent with
                facing_bet; a raise with no amount; or a raise at a node
                whose vocab has no raise slot (commitment-pruned).
        """
        if self.to_act_slot is None:
            raise ValueError("Cannot advance a terminal state")
        if action not in ("fold", "check", "call", "raise", "allin"):
            raise ValueError(f"unknown observed action {action!r}")

        slot = self.to_act_slot
        facing = self.facing_bet
        to_call = max(0, self.current_bet - self.street_bets[slot])

        active = list(self.active)
        acted = list(self.acted)
        street_bets = list(self.street_bets)
        stacks = list(self.stacks)
        current_bet = self.current_bet
        raises = self.raises_this_street
        pot = self.pot
        acted[slot] = True

        n_nonraise = (1 if facing else 0) + 1

        def _apply_call() -> int:
            """Match the outstanding bet (clamped to stack). Returns chips."""
            chips = min(to_call, stacks[slot])
            if chips > 0:
                street_bets[slot] += chips
                stacks[slot] -= chips
                pot_add[0] = chips
            return chips

        pot_add = [0]

        if action == "fold":
            if not facing:
                raise ValueError(
                    "observed FOLD with nothing to call — not a legal node "
                    "transition (open-folds are not in the trainer vocab)"
                )
            active[slot] = False
            idx, off_tree, applied = 0, False, 0
        elif action == "check":
            if facing:
                raise ValueError(
                    "observed CHECK while facing a bet — inconsistent input"
                )
            idx, off_tree, applied = 0, False, 0
        elif action == "call":
            # "call" with no bet outstanding is a check (some feeds say it).
            applied = _apply_call() if facing else 0
            idx, off_tree = (1 if facing else 0), False
        else:
            # raise / allin — possibly off-tree.
            if action == "allin":
                increment = stacks[slot]
            else:
                if amount is None or int(amount) <= 0:
                    raise ValueError("observed RAISE requires a chip amount")
                increment = min(int(amount), stacks[slot])
            if increment <= 0:
                # The actor has NO chips left in the served canonical game
                # (e.g. a deeper-than-100bb table whose real player still has
                # chips after the canonical stack went all-in). Recording the
                # observed bet as a 0-chip CHECK with off_tree=False served a
                # silently wrong node (review 2026-06-10). Fail loud; the API
                # maps this ValueError to a 422.
                raise ValueError(
                    f"observed {action} by a player with no remaining chips "
                    "in the canonical 100bb game — the described action "
                    "cannot exist in the served blueprint game (off-depth "
                    "table state)"
                )
            if increment <= to_call:
                # Under-raise / short all-in-call: representable on-tree only
                # as a CALL. A legal under-raise is an all-in (stack ==
                # increment), so _apply_call's min(to_call, stack) puts in
                # exactly the real chips; a malformed feed normalizes to a
                # full call.
                _apply_call()
                applied = pot_add[0]
                idx, off_tree = (1 if facing else 0), increment != to_call
            elif len(self.legal_action_categories()) <= n_nonraise:
                # Raise-CAP / commitment-pruned node: the trainer vocab here
                # is [fold?, call] only (POSTFLOP_MAX_RAISES reached, or no
                # canonical raise survives), so a further real raise has NO
                # on-tree representation (review 2026-06-10: recording a
                # phantom raise index corrupts every later blueprint key and
                # crashes the street replay). Record the CALL — the only
                # on-tree continuation — while applying the REAL chips, and
                # keep raises_this_street at the cap so the remaining
                # players' [fold, call] vocab stays consistent; the unmatched
                # bet re-opens action for them via the street_bets predicate.
                street_bets[slot] += increment
                stacks[slot] -= increment
                pot_add[0] = increment
                current_bet = max(current_bet, street_bets[slot])
                applied = increment
                idx, off_tree = (1 if facing else 0), True
            else:
                survivors = self._surviving_raise_amounts(slot)
                if not survivors:
                    raise ValueError(
                        "observed raise at a node whose vocab has no raise "
                        "slot (stack-commitment prune) — cannot represent "
                        "on-tree"
                    )
                position = min(
                    range(len(survivors)),
                    key=lambda i: abs(math.log(increment / survivors[i])),
                )
                off_tree = survivors[position] != increment
                idx = n_nonraise + position
                street_bets[slot] += increment
                stacks[slot] -= increment
                pot_add[0] = increment
                current_bet = max(current_bet, street_bets[slot])
                raises += 1
                applied = increment

        pot += pot_add[0]

        next_state = self._after_action(
            slot=slot, action_idx=idx, active=active, acted=acted,
            street_bets=street_bets, stacks=stacks, current_bet=current_bet,
            raises=raises, pot=pot,
        )
        return ObservedAdvance(
            state=next_state, action_idx=idx, off_tree=off_tree,
            applied_amount=applied,
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
        stacks: List[int],
        start_after: int,
    ) -> Optional[int]:
        """Next slot that still owes an action, scanning FORWARD from
        `start_after` (i.e. starting at start_after+1, wrapping around).

        A slot owes action if it is active AND (hasn't acted yet OR hasn't
        matched the current bet AND can still put in chips). The stack
        condition is the rollout-reference all-in guard (rollout.py
        `if stacks[p] <= 0: continue`): a SHORT all-in caller cannot match
        the bet and owes nothing more — without this the round never closes
        and the state desyncs from the table (review 2026-06-10). NOTE the
        guard applies only to the unmatched-bet clause: an all-in player who
        has NOT acted this street still gets their (forced) turn — the
        trainer keeps all-in players in the rotation and those forced action
        indices are part of trained blueprint keys (pinned by
        test_deep_raise_war_node_now_serves_correct_length_mix).

        Forward-from-last-actor is the order the blueprint encodes
        (capture-verified) — after a bet, action proceeds in position order
        from the aggressor and only then wraps back to already-checked
        players. Returns None when no one owes action (round complete).
        """
        n = self.n_slots
        for step in range(1, n + 1):
            s = (start_after + step) % n
            if not active[s]:
                continue
            if not acted[s]:
                return s
            if street_bets[s] < current_bet and stacks[s] > 0:
                return s
        return None

    def _close_round(
        self,
        active: List[bool],
        stacks: List[int],
        pot: int,
        history: Tuple[int, ...],
        closed_count: int,
    ) -> "GameState":
        """Advance to the next street (or terminal if river closed).

        Stacks PERSIST across streets (chips removed this street stay removed);
        only the current-street bets/acted/raises reset — mirrors the trainer's
        street transition (pot carried, bets/has_acted/num_raises reset).
        """
        next_street = self.street + 1
        if next_street > _MAX_STREET:
            # River betting closed -> terminal showdown.
            return replace(
                self,
                active=tuple(active),
                stacks=tuple(stacks),
                pot=pot,
                postflop_history=history,
                street_history=(),
                to_act_slot=None,
                betting_round_closed_count=closed_count,
            )

        n_slots = self.n_slots
        first = self._first_live_slot(active)
        return replace(
            self,
            active=tuple(active),
            stacks=tuple(stacks),
            pot=pot,
            street=next_street,
            postflop_history=history,
            street_history=(),
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


@dataclass(frozen=True)
class ObservedAdvance:
    """Result of GameState.advance_observed (real-table action applied).

    state: the next GameState (REAL chips applied).
    action_idx: the SNAPPED on-tree action index recorded in the histories
        (what blueprint keys and the live-resolve street replay consume).
    off_tree: True when the observed chips didn't match the canonical
        on-tree amount (the abstraction-gap flag, for logging/analysis).
    applied_amount: real chips the actor added this action.
    """

    state: GameState
    action_idx: int
    off_tree: bool
    applied_amount: int


# ── Factory ───────────────────────────────────────────────────────────────


def new_postflop_state(
    seats: List[str],
    preflop_prefix: List[int],
    board: List[int],
    pot: Optional[int] = None,
    stacks: Optional[List[int]] = None,
    street: int = STREET_FLOP,
    postflop_bet_sizes: Tuple[float, ...] = DEFAULT_POSTFLOP_BET_SIZES,
    subsequent_bet_sizes: Tuple[float, ...] = DEFAULT_SUBSEQUENT_BET_SIZES,
    add_explicit_allin: bool = DEFAULT_ADD_EXPLICIT_ALLIN,
) -> GameState:
    """Build a fresh GameState entering a postflop street.

    `seats` are position names of the players still in the hand (any order);
    they're reordered SB-first into compact slots (the postflop acting order).
    `preflop_prefix` is the given preflop action-index sequence.

    Chips (3a): if `pot`/`stacks` are omitted they are DERIVED from the verified
    trainer convention via canonical_flop_chips (the correct flop-entry chips
    for the canonical single-open, 3-bet HU and limp-check lines). Passing
    them explicitly is supported
    for backward compatibility and non-canonical / non-flop entries — but note
    the flop-entry pot/stacks are load-bearing for the commitment prune, so
    prefer the derived path for canonical lines. `stacks` is aligned to `seats`
    BEFORE reordering: stacks[i] belongs to seats[i]; permuted to slot order.
    """
    if len(seats) < 2:
        raise ValueError("need at least 2 players for a postflop state")
    for p in seats:
        if p not in FULL_SEAT:
            raise ValueError(f"unknown position {p!r}; expected one of {list(FULL_SEAT)}")

    # Order players SB-first by full seat => compact slot order.
    order = sorted(range(len(seats)), key=lambda i: FULL_SEAT[seats[i]])
    seat_of_slot = tuple(FULL_SEAT[seats[i]] for i in order)
    n = len(order)

    if pot is None or stacks is None:
        # Derive the verified canonical flop-entry chips. Only valid for a
        # flop entry on a verified canonical preflop line.
        if street != STREET_FLOP:
            raise ValueError(
                "pot/stacks must be given for a non-flop entry "
                "(canonical chip derivation is flop-entry only)"
            )
        derived_pot, derived_slot_stacks = canonical_flop_chips(
            list(preflop_prefix), list(seats)
        )
        pot = derived_pot
        slot_stacks = derived_slot_stacks
    else:
        if len(seats) != len(stacks):
            raise ValueError(
                f"seats ({len(seats)}) and stacks ({len(stacks)}) length mismatch"
            )
        slot_stacks = tuple(int(stacks[i]) for i in order)

    return GameState(
        seat_of_slot=seat_of_slot,
        stacks=slot_stacks,
        active=tuple([True] * n),
        pot=int(pot),
        board=tuple(board),
        street=int(street),
        preflop_prefix=tuple(preflop_prefix),
        postflop_history=(),
        street_history=(),
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
