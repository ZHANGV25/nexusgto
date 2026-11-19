"""GameState -> CpuEsSolver translation for LIVE_RESOLVE (plan §3.6).

The agent's StrategySource.get_mix maps (GameState, hero_cards) -> ActionMix.
The live-solve engines are STATEFUL/imperative. This module is the thin adapter
that bridges the two for the LIVE_RESOLVE ceiling, GPU-free. Re-solves remain
per-call; a default-off BeliefCarrySession held by LiveResolveSource can carry
completed-street search beliefs onto later calls.

ENGINE CHOICE (empirically validated — docs/BENCHMARK_RESULTS.md §3c):
  - flop / turn  → CpuEsSolver (poker-solver CPU external-sampling ES). Builds
    real blueprint-grounded leaf values off the GPU lock; N-player generic, so
    it serves 2p AND 3p (the "5p only" in postflop_provider is a routing policy,
    not an engine limit). Latency depends on the configured iteration budget.
  - river        → CpuEsSolver too (2026-06-11). River trees terminate in
    exact showdown/fold terminals settled natively by the C ES
    (es_showdown_value — side-pot validated, poker-solver
    validate_sidepot.py: 100k-case fuzz < 1e-9 + the nh>1 vectorized gate);
    the leaf-table construct stage is bypassed fail-loud inside CpuEsSolver.
    The old §6 "river is the side-pot gap" rejection is GONE.

solve path (solver_v2 StreetSolver) crashes building multi-street trees for
flop/turn (infoset-cap NULL deref; verified exit 127). It works for river only.
So LIVE_RESOLVE wraps CpuEsSolver for the flop/turn surface this session targets.

SCOPE / SIMPLIFICATIONS (documented, honest):
  - PER-CALL re-solve: ranges are rebuilt fresh each call from the
    board (169-class grid, K=1 by default, blocked by the board). Ranges are
    SEEDED IN-DISTRIBUTION
    from the blueprint preflop σ̄ when range_weight_dicts is supplied (see
    LiveResolveSource + live_resolve_ranges) — uniform ranges make the re-solve
    over-bet (the 3b confound). With NEXUS_BELIEF_CARRY enabled, completed-
    street search-sigma likelihoods are composed onto those seeded weights;
    the default-off path remains the independent per-street behavior.
  - MID-STREET ENTRY (faced bet / post-raise) is served by solving from the
    STREET ROOT (chips/players reconstructed at street entry) and REPLAYING
    this street's recorded actions (GameState.street_history) inside the
    solved tree — CpuEsSolver's hero_replay reads the hero's strategy at the
    reached node, with its correct (fold/call/raise) action set. The old
    fail-loud guard (facing_bet or raises_this_street > 0) is GONE; what
    remains fail-loud is an OFF-TREE action (a recorded action that matches
    no tree child — bet-size translation is a later step) and a terminal /
    non-postflop state.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from app.engine.agent.game_state import BIG_BLIND, SEAT_NAME, GameState
from app.engine.agent.key_builder import build_key

# Reuse the production card encoder (rank<<2 | suit, A=12) — identical to the
# agent's card ints, so a round-trip int->str->int is the identity.
from app.services.postflop_provider import (
    GridK,
    HandGridMetadata,
    _build_hand_grid_range_with_metadata,
    _int_to_card,
)

# Reuse the production label↔index translation (metadata-driven, bet_sizes-aware).
from app.services.card_codec import _solver_label_to_action_index


class LiveResolveUnsupported(Exception):  # noqa: N818 - sibling-naming convention
    """The live re-solve cannot serve this node GPU-free in v1 scope.

    Raised for: a non-postflop street (the engine serves flop/turn/river), a
    terminal state, or a KNOWN abstraction-surface limit surfaced by the
    engine as ReplayUnsupported (tree raise cap, action-count divergence,
    unvalidated vocab shape) and converted by LiveResolveSource. Fail loud —
    never serve a wrong node.
    """


class LiveResolveConvergenceError(RuntimeError):
    """The live solve did not produce a usable strategy for the hero row.

    This is a server-side solve failure, not an unsupported abstraction node.
    Callers must surface it explicitly and must never route it to lookup.
    """


# Street ints (1=flop,2=turn,3=river) -> CpuEsSolver street name / board length.
_STREET_NAME = {1: "flop", 2: "turn", 3: "river"}
_STREET_BOARD_LEN = {1: 3, 2: 4, 3: 5}


def _has_to_call(state: GameState) -> bool:
    """Whether the to-act player faces an outstanding bet (drives fold/call vs
    check in _solver_label_to_action_index)."""
    return state.facing_bet


def _grid_k_from_env() -> GridK:
    """Read the hand-grid dose. Default ISO (2026-09-09): one row per
    board-relative suit-isomorphism class, exact for hand strength and about
    a third of the ALL rows on a turn (1128 -> 367), which is what puts a 2p
    turn re-solve at 0.7-2.5 s on r7a.2xlarge. ALL keeps every physical
    combo for measurement.

    The historical K=1 grid held ONE representative combo per 169 class,
    chosen clubs-first among unblocked suits, and served the hero's real
    cards from that class row. On any non-club flush board no hand in the
    model could make a flush or hold a flush draw; on a club board every
    suited hand was a flush (measured 2026-09-04: real flushes folded 100%
    to a river jam, non-club suited hands raised 100% on a club board).
    K=1 remains selectable for measurement only.
    """
    raw = os.environ.get("NEXUS_GRID_K", "ISO").strip()
    if raw.upper() in ("ALL", "ISO"):
        # ISO (2026-09-09): one row per board-relative suit-isomorphism
        # class (flush-irrelevant suits pooled); exact for hand strength,
        # roughly a third of the ALL rows on a turn. See suit_isomorphism.py.
        return raw.upper()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"NEXUS_GRID_K must be a positive integer, ALL or ISO, got {raw!r}"
        ) from None
    if value <= 0:
        raise ValueError(
            f"NEXUS_GRID_K must be a positive integer, ALL or ISO, got {raw!r}"
        )
    return value


def _hu_leaf_mirror_enabled() -> bool:
    """Read the strict per-call true-HU leaf-key experiment flag."""
    raw = os.environ.get("NEXUS_HU_LEAF_MIRROR", "0")
    if raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(
        f"NEXUS_HU_LEAF_MIRROR must be 0 or 1, got {raw!r}"
    )


COMBO_PRIOR_ENV = "NEXUS_RANGE_COMBO_PRIOR"


def _combo_prior_enabled() -> bool:
    return os.environ.get(COMBO_PRIOR_ENV, "")[:1] in ("1", "t", "T", "y", "Y")


def build_solve_inputs(
    state: GameState,
    range_weight_dicts: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, object]:
    """Translate a GameState into the keyword inputs for CpuEsSolver.solve().

    Returns a dict with: board_strs, board_ints, positions_sorted, player_seats,
    player_ranges, hero_player_idx, pot_bb, stack_bb, game_action_indices,
    street_actions (this street's recorded action indices, for CpuEsSolver's
    hero_replay; empty at a street root), plus n_legal/has_to_call for the
    response mapping.

    The solve is ALWAYS posed from the STREET ROOT: chips and the player set
    are reconstructed at street entry (street_bets rolled back; players who
    folded EARLIER streets excluded, players who folded THIS street included —
    their fold lines are part of the street tree). A mid-street entry is then
    a replay of street_actions inside the solved tree, not a different solve.

    range_weight_dicts: optional per-slot {hand_class_name: weight} (in
        compact-slot order, aligned to state.seat_of_slot). When given, each
        player's 169 grid is seeded IN-DISTRIBUTION (preflop σ̄) instead of
        uniform — removes the over-bet confound (see live_resolve_ranges). When
        None, the grid is uniform-on-legal. A wrong-length list is an internal
        contract violation and raises instead of silently discarding weights.

    Raises:
        LiveResolveUnsupported: terminal state or a non-postflop street.
    """
    if state.terminal:
        raise LiveResolveUnsupported("terminal GameState has no decision node")
    if state.street not in _STREET_NAME:
        raise LiveResolveUnsupported(
            f"live-resolve serves flop/turn/river only (street={state.street})"
        )

    hero_slot = state.to_act_slot
    assert hero_slot is not None  # guarded by `terminal` above

    # This street's recorded actions (empty at a street root / check-through
    # is NOT empty — checks are actions and are replayed like any other).
    street_actions = list(state.street_history)

    # Players in the hand at street ENTRY: still active, or folded THIS street
    # (active=False, acted=True — `acted` resets at street close, so a
    # prior-street folder shows acted=False). This-street folders belong in
    # the street tree; prior-street folders were never in it.
    entry_slots = [
        s for s in range(state.n_slots) if state.active[s] or state.acted[s]
    ]
    if hero_slot not in entry_slots:
        # A bug condition (the to-act player must be in the hand), NOT a
        # documented unsupported surface — fail loud, never lookup-fallback.
        raise ValueError(
            "to-act player is not in the street-entry player set — "
            "inconsistent GameState"
        )
    if len(state.board) < _STREET_BOARD_LEN[state.street]:
        raise ValueError(
            f"GameState street={state.street} needs >= "
            f"{_STREET_BOARD_LEN[state.street]} board cards but has "
            f"{len(state.board)} — the driver must supply the dealt card(s) "
            "(rebuild the state with the extended board) before a re-solve, "
            "or the solve silently runs the WRONG street."
        )

    # Chips at street ENTRY: roll the current street's bets back out.
    pot_entry = int(state.pot) - sum(int(b) for b in state.street_bets)
    # Effective stack for the SOLVE: min over players still CONTESTING
    # (active). A this-street folder stays IN the tree (their fold line is
    # replayed) but must not cap the live players' stack depth — a short
    # stack that folded would otherwise silently shrink the whole solved
    # game (review 2026-06-10).
    contesting_stacks = [
        int(state.stacks[s]) + int(state.street_bets[s])
        for s in entry_slots if state.active[s]
    ]

    board_ints = list(state.board)[: _STREET_BOARD_LEN[state.street]]
    board_strs = [_int_to_card(c) for c in board_ints]

    # positions_sorted = entry players' seat names in compact-slot order
    # (already SB-first); solver player index == position in entry_slots.
    positions_sorted = [SEAT_NAME[state.seat_of_slot[s]] for s in entry_slots]
    n = len(entry_slots)
    root_stack_chips = tuple(
        int(state.stacks[slot]) + int(state.street_bets[slot])
        for slot in entry_slots
    )
    unequal_stack_inputs: Dict[str, object] = {}
    requires_unequal_stack_abi = bool(
        getattr(state, "real_requires_unequal_stack_abi", False)
    )
    scalar_fallback_verified = bool(getattr(
        state, "real_scalar_stack_fallback_verified", False
    ))
    if (
        bool(getattr(state, "real_geometry", False))
        and not requires_unequal_stack_abi
        and (
            any(stack <= 0 for stack in root_stack_chips)
            or (
                len(set(root_stack_chips)) > 1
                and not scalar_fallback_verified
            )
        )
    ):
        raise LiveResolveUnsupported(
            "real unequal-stack geometry is missing its translation-owned "
            "root-pot ledger"
        )
    if requires_unequal_stack_abi:
        if not bool(getattr(state, "real_geometry", False)):
            raise ValueError(
                "unequal-stack solver sidecar exists outside real geometry"
            )
        if state.street != 3:
            raise LiveResolveUnsupported(
                "unequal-stack real geometry is supported only on the river"
            )
        if n not in (2, 3):
            raise LiveResolveUnsupported(
                "unequal-stack river geometry requires 2 or 3 players"
            )
        translated_stacks = tuple(
            int(value)
            for value in getattr(state, "real_root_stacks_chips", ())
        )
        root_pot_amounts = tuple(
            int(value)
            for value in getattr(state, "real_root_pot_amounts", ())
        )
        root_pot_masks = tuple(
            int(value)
            for value in getattr(
                state, "real_root_pot_eligible_masks", ()
            )
        )
        if translated_stacks != root_stack_chips:
            raise LiveResolveUnsupported(
                "real root-stack sidecar disagrees with the translated state"
            )
        if (
            not root_pot_amounts
            or len(root_pot_amounts) != len(root_pot_masks)
            or sum(root_pot_amounts) != pot_entry
        ):
            raise LiveResolveUnsupported(
                "real root-pot sidecar is incomplete or does not cover the pot"
            )
        unequal_stack_inputs = {
            "root_stacks_bb": [
                float(chips) / BIG_BLIND for chips in translated_stacks
            ],
            "root_stack_chips": list(translated_stacks),
            "root_pot_amounts": list(root_pot_amounts),
            "root_pot_eligible_masks": list(root_pot_masks),
        }
    # The C rollout keys the blueprint's postflop CONTINUATION strategies by FULL
    # seat id (mccfr_blueprint.c stores key.player = the seat; bps_query_strategy
    # looks up by that seat). player_seats passed compact indices range(n) here,
    # so for any spot whose compact ids differ from full seats (HU [BB,UTG] is
    # compact [0,1] but full [1,2]) the blueprint query used the WRONG seat and
    # task-mqgm2hpp). Pass the FULL seat per compact slot. (player_seats reaches
    # the engine ONLY as the rollout's blueprint seat key — cpu_es_solver passes
    # it straight to compute_subgame_leaf_values_sampled_c.)
    # NEXUS_HU_LEAF_MIRROR is a default-off true-HU experiment: swap only
    # the SB/BB leaf keys so BB-first sequence positions query the blueprint's
    # positionally corresponding folded-to-SB continuation (OOP->OOP, IP->IP).
    # Ranges and every game input stay put; 6-max states are never mirrored.
    player_seats = [state.seat_of_slot[s] for s in entry_slots]
    if _hu_leaf_mirror_enabled() and getattr(state, "true_hu", False):
        invalid_seats = [
            seat for seat in state.seat_of_slot if seat not in (0, 1)
        ]
        if invalid_seats:
            raise ValueError(
                "true_hu leaf mirror requires only full seats {0, 1}; "
                f"got seat_of_slot={list(state.seat_of_slot)}"
            )
        player_seats = [1 - seat for seat in player_seats]
    hero_player_idx = entry_slots.index(hero_slot)

    # K-combo grid per entry player, blocked by the board. NEXUS_GRID_K is a
    # measurement-only dose flag; its default 1 preserves the shipped grid.
    # Seeded in-distribution when range_weight_dicts is supplied (aligned to
    # COMPACT SLOT order), else uniform-on-legal. Flagged belief carry is
    # composed into the weight dicts by LiveResolveSource before this call.
    blocked = set(board_ints)
    if (
        range_weight_dicts is not None
        and len(range_weight_dicts) != state.n_slots
    ):
        raise ValueError(
            "range_weight_dicts length must match GameState slots: "
            f"got {len(range_weight_dicts)} for {state.n_slots} slots"
        )
    use_weights = range_weight_dicts is not None
    grid_k = _grid_k_from_env()
    player_ranges: List[List[Tuple[int, int, float]]] = []
    player_range_metadata: List[List[HandGridMetadata]] = []
    for slot in entry_slots:
        weights = range_weight_dicts[slot] if use_weights else None
        grid, metadata = _build_hand_grid_range_with_metadata(
            blocked, weights=weights, k=grid_k, board=board_ints,
        )
        player_ranges.append([(r[0], r[1], r[2]) for r in grid])
        player_range_metadata.append(metadata)
    # NEXUS_RANGE_COMBO_PRIOR (research, SCOREBOARD_RESULTS section 34): the
    # grid conserves each hand CLASS's weight, so per physical combo a pair
    # (6 combos) carries twice and a suited hand (4) three times the mass of
    # an offsuit hand (12). Scaling every row by its class's live combo count
    # restores the deal prior: each combo carries its class's seeded weight.
    combo_prior = _combo_prior_enabled()
    if combo_prior:
        player_ranges = [
            [(c0, c1, w * float(meta[2])) for (c0, c1, w), meta in zip(rows, row_metadata)]
            for rows, row_metadata in zip(player_ranges, player_range_metadata)
        ]

    # Chips -> big blinds (CpuEsSolver scales bb by 100 internally).
    pot_bb = float(pot_entry) / BIG_BLIND
    stack_bb = float(min(contesting_stacks)) / BIG_BLIND

    # Action history UP TO street entry — the solved tree itself adds this
    # street's actions, so they must NOT be in the leaf-rollout history too.
    key = build_key(state)
    full_history = list(key.action_history)
    n_street = len(street_actions)
    game_action_indices = (
        full_history[:-n_street] if n_street else full_history
    )

    result = {
        "board_strs": board_strs,
        "board_ints": board_ints,
        "positions_sorted": positions_sorted,
        "player_seats": player_seats,
        # Solver-player -> compact-slot map. This is inert translation metadata
        # unless the default-off belief stash path requests it.
        "entry_slots": entry_slots,
        "player_ranges": player_ranges,
        "player_range_metadata": player_range_metadata,
        "grid_k": grid_k,
        "hero_player_idx": hero_player_idx,
        "pot_bb": pot_bb,
        "stack_bb": stack_bb,
        "game_action_indices": game_action_indices,
        "street_actions": street_actions,
        "n_legal": state.num_legal_actions(),
        "has_to_call": _has_to_call(state),
        "combo_prior": combo_prior,
    }
    result.update(unequal_stack_inputs)
    return result


def hero_grid_row(player_ranges: Sequence, hero_player_idx: int,
                  hero_cards: Tuple[int, int]) -> int:
    """Index of the hero's hand-CLASS row in its legacy K=1 grid.

    CpuEsSolver returns σ̄ per grid row (one representative combo per 169 class);
    the hero's exact cards belong to one class, whose row carries that class's
    strategy. We match by class membership (same (rank-hi, rank-lo, suited)),
    NOT by exact combo, because the grid stores a single representative combo.
    """
    c0, c1 = hero_cards
    target = _class_key(c0, c1)
    grid = player_ranges[hero_player_idx]
    for i, (g0, g1, _w) in enumerate(grid):
        if _class_key(g0, g1) == target:
            return i
    # The hero's own combo can never be board-blocked, so a missing class
    # row is a range-construction BUG — fail loud, never lookup-fallback.
    raise ValueError(
        f"hero hand ({c0},{c1}) maps to no grid class — corrupted range grid"
    )


def hero_exact_row(
    player_ranges: Sequence,
    hero_player_idx: int,
    hero_rows: Sequence[Tuple[int, float]],
    hero_cards: Tuple[int, int],
    board: Optional[Sequence[int]] = None,
) -> Optional[int]:
    """Row holding the hero's PHYSICAL combo among its class rows, or None.

    K=ALL grids contain the combo; the K=1 grid holds one representative
    per class, so this returns None there and callers fall back to the
    class row. Serving the exact row is what makes suits real for hero.

    With ``board`` given (the ISO grid), the match is by board-relative
    canonical form: the row holding the canonical representative of the
    hero's combo is the hero's row (isomorphic under the pooled suits, so
    exact for strength).
    """
    grid = player_ranges[hero_player_idx]
    target = set(hero_cards)
    if board is not None:
        from app.services.suit_isomorphism import canonical_combo
        target = set(canonical_combo(hero_cards[0], hero_cards[1], board))
    for row_idx, _weight in hero_rows:
        if set(grid[row_idx][:2]) == target:
            return row_idx
    return None


# Weight given to the hero's own row when the seeded range for this line
# holds it at zero (the hero deviated from the blueprint's preflop play, e.g.
# flat-called KK from the BB). Without it the row is dropped at readout and
# the decision refuses (live_resolve_not_converged -> blueprint fallback).
# 1e-6 is far below any real per-row weight (class weight split over the
# class's live combos, ~0.01-0.3), so the villain's model of hero is
# unchanged in every practical sense while hero's row is trained and served.
HERO_ROW_WEIGHT_FLOOR = 1e-6


def floor_hero_row(
    player_ranges: Sequence,
    hero_player_idx: int,
    row: int,
    floor: float = HERO_ROW_WEIGHT_FLOOR,
) -> Tuple[List[List[Tuple[int, int, float]]], bool]:
    """Return ``(player_ranges_copy, floored)`` with the hero's row weight
    raised to ``floor`` when it is below it. Inputs are never mutated; when
    no floor is needed the copy holds the same row tuples."""
    new_ranges = [list(rows) for rows in player_ranges]
    c0, c1, weight = new_ranges[hero_player_idx][row][:3]
    if float(weight) >= floor:
        return new_ranges, False
    new_ranges[hero_player_idx][row] = (int(c0), int(c1), float(floor))
    return new_ranges, True


def hero_grid_rows(
    player_ranges: Sequence,
    hero_player_idx: int,
    hero_cards: Tuple[int, int],
    range_metadata: Optional[Sequence] = None,
) -> List[Tuple[int, float]]:
    """Return every ``(row_index, weight)`` for the hero's hand class.

    Metadata is optional for compatibility with hand-built grids. When given,
    fully blocked placeholders are skipped so their legacy ``(0, 1)`` cards
    cannot masquerade as pocket deuces during class matching.
    """
    target = _class_key(*hero_cards)
    grid = player_ranges[hero_player_idx]
    metadata = (
        range_metadata[hero_player_idx]
        if range_metadata is not None else None
    )
    rows: List[Tuple[int, float]] = []
    for i, (g0, g1, weight) in enumerate(grid):
        if metadata is not None and int(metadata[i][2]) == 0:
            continue
        if _class_key(g0, g1) == target:
            rows.append((i, float(weight)))
    if rows:
        return rows
    c0, c1 = hero_cards
    raise ValueError(
        f"hero hand ({c0},{c1}) maps to no grid class - corrupted range grid"
    )


def aggregate_grid_row_strategies(
    per_hand: Dict[int, Dict[str, float]],
    rows: Sequence[Tuple[int, float]],
) -> Optional[Dict[str, float]]:
    """Weight-average every positive-weight strategy row in one class.

    The one-row case returns the original mapping unchanged, preserving the
    shipped K=1 float path. A missing positive-weight row returns ``None``
    instead of silently renormalizing over an incomplete class sample.
    """
    if len(rows) == 1:
        row, weight = rows[0]
        return per_hand.get(row) if weight > 0.0 else None
    total_weight = sum(weight for _row, weight in rows if weight > 0.0)
    if total_weight <= 0.0:
        return None
    weighted: Dict[str, float] = {}
    for row, weight in rows:
        if weight <= 0.0:
            continue
        strategy = per_hand.get(row)
        if not strategy:
            return None
        for label, frequency in strategy.items():
            weighted[label] = (
                weighted.get(label, 0.0) + weight * float(frequency)
            )
    return {
        label: value / total_weight for label, value in weighted.items()
    }


def _class_key(c0: int, c1: int) -> Tuple[int, int, bool]:
    """(rank_hi, rank_lo, suited) — the 169-class identity of a 2-card hand."""
    r0, r1 = c0 >> 2, c1 >> 2
    suited = (c0 & 3) == (c1 & 3)
    hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
    return (hi, lo, suited)


def labels_to_index_freqs(
    strat: Dict[str, float], *, has_to_call: bool, n_legal: int,
    bet_sizes: Tuple[float, ...], add_explicit_allin: bool,
) -> Dict[int, float]:
    """Map {solver_label: freq} -> {action_index: freq} in the agent's vocab.

    Uses the production _solver_label_to_action_index (metadata-driven). With
    bet_sizes == the agent's vocab (e.g. (0.75,)), the solver's action count
    equals state.num_legal_actions(), so labels map 1:1 onto [0, n_legal).
    Frequencies for labels that map to the same index are summed (robust to
    label rounding, e.g. "Bet 74%"/"Bet 76%" both snapping to the 0.75 slot).

    Raises:
        ValueError: a label maps outside [0, n_legal) — a vocab BUG (wrong
            bet_sizes), which must fail loud, never lookup-fallback. (The
            legitimate tree-vs-vocab action-count divergences are routed to
            the lookup fallback EARLIER, via the solver's expected_actions
            check — by the time labels are mapped, counts must agree.)
    """
    out: Dict[int, float] = {}
    for label, freq in strat.items():
        f = float(freq)
        if f <= 0.0:
            continue
        idx = _solver_label_to_action_index(
            label, has_to_call, bet_sizes, add_explicit_allin
        )
        if not (0 <= idx < n_legal):
            raise ValueError(
                f"solver label {label!r} -> index {idx} outside legal "
                f"[0,{n_legal}); the solver bet_sizes must match the agent vocab."
            )
        out[idx] = out.get(idx, 0.0) + f
    return out
