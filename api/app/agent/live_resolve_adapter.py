"""GameState -> CpuEsSolver translation for LIVE_RESOLVE (plan §3.6).

The agent's StrategySource.get_mix is STATELESS: (GameState, hero_cards) ->
ActionMix. The live-solve engines are STATEFUL/imperative. This module is the
thin adapter that bridges the two for the LIVE_RESOLVE ceiling, GPU-free.

ENGINE CHOICE (empirically validated — docs/BENCHMARK_RESULTS.md §3c):
  - flop / turn  → CpuEsSolver (poker-solver CPU external-sampling ES). Builds
    real blueprint-grounded leaf values off the GPU lock; N-player generic, so
    it serves 2p AND 3p (the "5p only" in postflop_provider is a routing policy,
    not an engine limit). 2p-flop ~3s, 3p-turn ~18s at a few hundred iters.
  - river        → NOT served here. CpuEsSolver raises on river (showdowns,
    side-pot splitting that lives only in the C trainer). 2p river COULD use
    solver_v2's StreetSolver, but 3p river is the §6 side-pot gap. v1 surfaces
    river as LiveResolveUnsupported (fail loud), keeping 2p/3p flop+turn as the
    verified surface; river live-solve is deferred.

solve path (solver_v2 StreetSolver) crashes building multi-street trees for
flop/turn (infoset-cap NULL deref; verified exit 127). It works for river only.
So LIVE_RESOLVE wraps CpuEsSolver for the flop/turn surface this session targets.

SCOPE / SIMPLIFICATIONS (documented, honest):
  - PER-CALL STATELESS re-solve: ranges are rebuilt fresh each call from the
    board (169 grid, blocked by the board). Ranges are SEEDED IN-DISTRIBUTION
    from the blueprint preflop σ̄ when range_weight_dicts is supplied (see
    LiveResolveSource + live_resolve_ranges) — uniform ranges make the re-solve
    over-bet (the 3b confound). There is NO cross-street Bayesian narrowing —
    this is a per-street re-solve, not the full Pluribus narrowing ceiling.
  - The hero strategy is read at the to-act player's FIRST (BFS) decision node —
    the no-bet check-line node. That equals the CURRENT node only when no betting
    has happened yet this street (a street ROOT, or a check-through to the hero).
    A faced-bet first action or a post-raise re-entry is a DIFFERENT node, so we
    fail loud (LiveResolveUnsupported) rather than serve the check-line strategy
    remapped as a (wrong, no-fold) faced-bet mix. Guard: not facing_bet AND
    raises_this_street == 0.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from app.agent.game_state import BIG_BLIND, SEAT_NAME, GameState
from app.agent.key_builder import build_key

# Reuse the production card encoder (rank<<2 | suit, A=12) — identical to the
# agent's card ints, so a round-trip int->str->int is the identity.
from app.services.postflop_provider import _build_hand_grid_range, _int_to_card

# Reuse the production label↔index translation (metadata-driven, bet_sizes-aware).
from app.services.session_pipeline import _solver_label_to_action_index


class LiveResolveUnsupported(Exception):  # noqa: N818 - sibling-naming convention
    """The live re-solve cannot serve this node GPU-free in v1 scope.

    Raised for: river (showdown / side-pot splitting not available here), a
    terminal state, or a mid-street re-entry whose current node isn't the
    to-act player's first decision node. Fail loud — never serve a wrong node.
    """


# Street ints (1=flop,2=turn,3=river) -> CpuEsSolver street name / board length.
_STREET_NAME = {1: "flop", 2: "turn"}
_STREET_BOARD_LEN = {1: 3, 2: 4}


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
    player_ranges, hero_player_idx, pot_bb, stack_bb, game_action_indices, plus
    n_legal/has_to_call for the response mapping.

    range_weight_dicts: optional per-slot {hand_class_name: weight} (in
        compact-slot order, aligned to state.seat_of_slot). When given, each
        player's 169 grid is seeded IN-DISTRIBUTION (preflop σ̄) instead of
        uniform — removes the over-bet confound (see live_resolve_ranges). When
        None (or wrong length), the grid is uniform-on-legal.

    Raises:
        LiveResolveUnsupported: terminal state, river, or a non-root re-entry.
    """
    if state.terminal:
        raise LiveResolveUnsupported("terminal GameState has no decision node")
    if state.street not in _STREET_NAME:
        raise LiveResolveUnsupported(
            f"live-resolve serves flop/turn only (street={state.street}); river "
            "is the side-pot/showdown gap (§6) — deferred."
        )

    # The to-act player's slot == its solver player_idx (player_ranges are built
    # in slot order; player_seats = range(n)). The hero is whoever is to act.
    hero_slot = state.to_act_slot
    assert hero_slot is not None  # guarded by `terminal` above

    # Fail loud unless the current node IS the hero's first decision node on this
    # street. CpuEsSolver reads each player's FIRST (BFS) decision node — the
    # no-bet check-line node. That equals the current node ONLY when no betting
    # has happened yet this street (street root, or check-through to the hero).
    # The correct condition is "no betting this street": NOT facing a bet AND no
    # raises yet. A faced-bet first action (acted[hero]=False but facing_bet=True)
    # would otherwise read the check-line node and remap its {Check,Bet} labels as
    # {call,raise} — a silently wrong, no-fold mix. (acted[hero] alone misses it.)
    if state.facing_bet or state.raises_this_street > 0:
        raise LiveResolveUnsupported(
            "non-root node: live-resolve reads the street-root first-decision "
            "node; a faced-bet / post-raise / mid-street re-entry is not that "
            "node (would serve the wrong strategy). Out of v1 scope — fail loud."
        )

    board_ints = list(state.board)[: _STREET_BOARD_LEN[state.street]]
    board_strs = [_int_to_card(c) for c in board_ints]

    # positions_sorted = seat names in compact-slot order (already SB-first).
    positions_sorted = [SEAT_NAME[s] for s in state.seat_of_slot]
    n = len(positions_sorted)
    player_seats = list(range(n))

    # 169 grid per player, blocked by the board (mirrors solve_cpu_es). Seeded
    # in-distribution when range_weight_dicts is supplied (aligned to slot
    # order), else uniform-on-legal. No cross-street narrowing in v1.
    blocked = set(board_ints)
    use_weights = (
        range_weight_dicts is not None
        and len(range_weight_dicts) == n
    )
    player_ranges: List[List[Tuple[int, int, float]]] = []
    for slot in range(n):
        weights = range_weight_dicts[slot] if use_weights else None
        grid = _build_hand_grid_range(blocked, weights=weights)
        player_ranges.append([(r[0], r[1], r[2]) for r in grid])

    # Chips -> big blinds (CpuEsSolver scales bb by 100 internally).
    pot_bb = float(state.pot) / BIG_BLIND
    stack_bb = float(min(state.stacks)) / BIG_BLIND

    key = build_key(state)
    game_action_indices = list(key.action_history)

    return {
        "board_strs": board_strs,
        "board_ints": board_ints,
        "positions_sorted": positions_sorted,
        "player_seats": player_seats,
        "player_ranges": player_ranges,
        "hero_player_idx": hero_slot,
        "pot_bb": pot_bb,
        "stack_bb": stack_bb,
        "game_action_indices": game_action_indices,
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
    raise LiveResolveUnsupported(
        f"hero hand ({c0},{c1}) maps to no grid class (board-blocked?)"
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
        LiveResolveUnsupported: a label maps outside [0, n_legal) — a vocab
            mismatch (wrong bet_sizes), surfaced loud rather than dropped.
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
            raise LiveResolveUnsupported(
                f"solver label {label!r} -> index {idx} outside legal "
                f"[0,{n_legal}); the solver bet_sizes must match the agent vocab."
            )
        out[idx] = out.get(idx, 0.0) + f
    return out
