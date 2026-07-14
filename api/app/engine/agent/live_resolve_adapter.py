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
    not an engine limit). 2p-flop ~3s, 3p-turn ~18s at a few hundred iters.
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
    board (169 grid, blocked by the board). Ranges are SEEDED IN-DISTRIBUTION
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

from typing import Dict, List, Optional, Sequence, Tuple

from app.engine.agent.game_state import BIG_BLIND, SEAT_NAME, GameState
from app.engine.agent.key_builder import build_key

# Reuse the production card encoder (rank<<2 | suit, A=12) — identical to the
# agent's card ints, so a round-trip int->str->int is the identity.
from app.services.postflop_provider import _build_hand_grid_range, _int_to_card

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


# Street ints (1=flop,2=turn,3=river) -> CpuEsSolver street name / board length.
_STREET_NAME = {1: "flop", 2: "turn", 3: "river"}
_STREET_BOARD_LEN = {1: 3, 2: 4, 3: 5}


def _has_to_call(state: GameState) -> bool:
    """Whether the to-act player faces an outstanding bet (drives fold/call vs
    check in _solver_label_to_action_index)."""
    return state.facing_bet


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
        None (or wrong length), the grid is uniform-on-legal.

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
    # The C rollout keys the blueprint's postflop CONTINUATION strategies by FULL
    # seat id (mccfr_blueprint.c stores key.player = the seat; bps_query_strategy
    # looks up by that seat). player_seats passed compact indices range(n) here,
    # so for any spot whose compact ids differ from full seats (HU [BB,UTG] is
    # compact [0,1] but full [1,2]) the blueprint query used the WRONG seat and
    # task-mqgm2hpp). Pass the FULL seat per compact slot. (player_seats reaches
    # the engine ONLY as the rollout's blueprint seat key — cpu_es_solver passes
    # it straight to compute_subgame_leaf_values_sampled_c.)
    player_seats = [state.seat_of_slot[s] for s in entry_slots]
    hero_player_idx = entry_slots.index(hero_slot)

    # 169 grid per entry player, blocked by the board (mirrors solve_cpu_es).
    # Seeded in-distribution when range_weight_dicts is supplied (aligned to
    # COMPACT SLOT order), else uniform-on-legal. Flagged belief carry is
    # composed into the weight dicts by LiveResolveSource before this call.
    blocked = set(board_ints)
    use_weights = (
        range_weight_dicts is not None
        and len(range_weight_dicts) == state.n_slots
    )
    player_ranges: List[List[Tuple[int, int, float]]] = []
    for slot in entry_slots:
        weights = range_weight_dicts[slot] if use_weights else None
        grid = _build_hand_grid_range(blocked, weights=weights)
        player_ranges.append([(r[0], r[1], r[2]) for r in grid])

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

    return {
        "board_strs": board_strs,
        "board_ints": board_ints,
        "positions_sorted": positions_sorted,
        "player_seats": player_seats,
        # Solver-player -> compact-slot map. This is inert translation metadata
        # unless the default-off belief stash path requests it.
        "entry_slots": entry_slots,
        "player_ranges": player_ranges,
        "hero_player_idx": hero_player_idx,
        "pot_bb": pot_bb,
        "stack_bb": stack_bb,
        "game_action_indices": game_action_indices,
        "street_actions": street_actions,
        "n_legal": state.num_legal_actions(),
        "has_to_call": _has_to_call(state),
    }


def hero_grid_row(player_ranges: Sequence, hero_player_idx: int,
                  hero_cards: Tuple[int, int]) -> int:
    """Index of the hero's hand-CLASS row in its 169 grid.

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
