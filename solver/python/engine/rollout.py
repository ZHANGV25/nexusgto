"""Pluribus-faithful rollout leaf values.

See docs/C1_ROLLOUT_AUDIT.md for the four correctness deviations in the
pre-C1 implementation that motivated this ground-up rewrite.

Paper reference:
  Brown, Sandholm. "Superhuman AI for multiplayer poker."
  Science 2019, supplementary §4 "Real-Time Search" / Algorithm 2.

Algorithm (continuation-strategy depth-limited solving):

  1. At every subgame leaf, each player has already chosen one of 4
     continuation strategies:
       0 = blueprint σ̄ unmodified
       1 = blueprint, fold probability × 5, renormalized
       2 = blueprint, call probability × 5, renormalized
       3 = blueprint, every raise probability × 5, renormalized
     The choice is identical across leaves in the same info set and is
     decided by the subgame CFR (treated as another action).

  2. To estimate a leaf's value, Pluribus runs NUM_ROLLOUTS=3 rollouts
     of the remainder of the game:
       - Deal the remaining board cards one street at a time.
       - At each decision node, look up σ̄ for the acting player at
         their trained (bucket, full-game-action-history), apply their
         chosen bias transform, sample one action.
       - Continue until showdown or everyone-but-one folded; return
         the hero's net chip payoff.
     Average across the NUM_ROLLOUTS rollouts.

  3. The resulting per-leaf per-bias-combo value is the leaf value used
     by the subgame CFR on its next iteration.

This module is the canonical correctness reference. It is Python and
port are separate modules in a later stage. Both ports must produce
bit-identical values modulo RNG choice to this reference.
"""

from __future__ import annotations

import enum
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.leaf_values import (  # type: ignore[import-untyped]
    LeafValueUnavailable,
    _eval7_py,
)


# ─── Constants — matches Pluribus Science 2019 §4 exactly ────────────────

NUM_ROLLOUTS_DEFAULT: int = 3
"""Pluribus §4: 'We found the optimal number of rollouts to be three.'"""

BIAS_MULTIPLIER: float = 5.0
"""Pluribus §4: 'multiplied by 5, then renormalized.'"""

NUM_BIAS_PROFILES: int = 4

# Blueprint training postflop action tree — must match
# precompute/blueprint_worker_unified.py::POSTFLOP_BET_SIZES.
# First raise pot fractions: [0.5, 1.0]. Subsequent raise: [1.0].
POSTFLOP_BET_FRACTIONS: Tuple[float, ...] = (0.5, 1.0)
SUBSEQUENT_BET_FRACTIONS: Tuple[float, ...] = (1.0,)

# Maximum raises per street the training abstraction allows.
POSTFLOP_MAX_RAISES: int = 3


class ActionCategory(enum.IntEnum):
    """Categories used by the 4 bias profiles."""
    FOLD = 0
    CALL = 1   # also used for 'check' (semantically 'do not put money in')
    RAISE = 2  # any bet or raise size


# ─── Bias transform ──────────────────────────────────────────────────────


def apply_bias(
    strategy: np.ndarray,
    bias_profile: int,
    categories: Sequence[int],
) -> np.ndarray:
    """Apply one of the 4 Pluribus bias profiles to a strategy vector.

    Args:
        strategy: [num_actions] probability distribution (need not sum
            to exactly 1; we renormalize).
        bias_profile: 0, 1, 2, or 3.
            0 = unmodified blueprint
            1 = fold prob × 5, renormalize
            2 = call prob × 5, renormalize
            3 = every raise prob × 5, renormalize
        categories: same length as strategy; each is an ActionCategory
            int. CALL covers both 'call' and 'check' (both = "don't
            escalate money"). RAISE covers bet and raise of any size.

    Returns:
        [num_actions] biased + renormalized distribution (sums to 1).

    Behavior on degenerate bias (no mass on the bias target):
        Falls back to the *unbiased* strategy (renormalized), NOT to
        uniform. This corrects a defect in the pre-C1 rollout code where
        the fallback to uniform could silently replace a blueprint
        distribution that happened to have zero mass on the target
        (e.g., raise-biased at a check-only node).
    """
    s = np.asarray(strategy, dtype=np.float64)
    if s.sum() <= 0.0:
        # Totally empty strategy — can't even renormalize. Caller
        # shouldn't be feeding us this; treat as uniform over the
        # legal action set.
        return np.ones_like(s) / max(len(s), 1)

    # Unbiased: just renormalize the input.
    if bias_profile == 0:
        return s / s.sum()

    target = {1: ActionCategory.FOLD,
              2: ActionCategory.CALL,
              3: ActionCategory.RAISE}.get(bias_profile)
    if target is None:
        # Unknown profile — treat as unbiased (safer than uniform).
        return s / s.sum()

    biased = s.copy()
    for i, cat in enumerate(categories):
        if i < len(biased) and int(cat) == int(target):
            biased[i] *= BIAS_MULTIPLIER

    total = biased.sum()
    # Degenerate case: no mass on target to amplify (e.g., raise-biased
    # at check-only). Fall back to the unbiased input, NOT uniform.
    if total <= 0.0:
        return s / s.sum()

    return biased / total


# ─── Rollout leaf state ──────────────────────────────────────────────────


@dataclass
class RolloutLeaf:
    """Full state needed to simulate rollouts from a subgame leaf.

    Uses a COMPACT player layout: slots 0..num_players-1 represent the
    players who reached the subgame. `player_seats` maps compact slot
    to the BLUEPRINT's full seat (0=SB, 1=BB, 2=UTG, 3=MP, 4=CO, 5=BTN).
    The blueprint's `get_strategy`/`hand_to_bucket` are called with the
    full seat number; everything else (active, stacks, hands, acting
    order) uses compact slots.

    Fields:
        board: 3-5 card ints (flop/turn/river).
        pot_chips: chips in the pot entering this leaf's continuation.
        active: per-compact-slot bool. Folded players are False.
        stacks: per-compact-slot remaining chips.
        bets: per-compact-slot chips committed this hand total.
        street: "flop" | "turn" | "river".
        player_hands: per-compact-slot list of (c0, c1, reach_weight).
            Any hand with c0 or c1 in `board` is blocked.
        acting_order: compact slots in postflop order (first to act
            first). E.g., HU with UTG=slot 0, BB=slot 1: [1, 0].
        player_seats: compact slot → blueprint full seat (0-5).
            Required for blueprint lookups.
        game_action_history: full preflop + postflop-prior action
            indices leading INTO this leaf. Used as the prefix for
            blueprint σ̄ lookups — rollout actions are appended.
    """
    board: List[int]
    pot_chips: int
    active: List[bool]
    stacks: List[int]
    bets: List[int]
    street: str
    player_hands: List[List[Tuple[int, int, float]]]
    acting_order: List[int]
    game_action_history: List[int]
    # Compact slot → blueprint full seat. Default: identity (compat
    # with 6-player training-native callers).
    player_seats: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.player_seats is None:
            self.player_seats = list(range(len(self.active)))


# ─── Action tree classification ──────────────────────────────────────────


def _add_explicit_allin() -> bool:
    """Whether to append an explicit all-in slot to postflop action sets.

    MUST match the training-time generate_actions() add_explicit_allin gate.
    Phase3 trained with add_explicit_allin=0, so default OFF. Override via
    BPS_ROLLOUT_ADD_EXPLICIT_ALLIN=1 for blueprints trained with it on.
    """
    v = os.environ.get("BPS_ROLLOUT_ADD_EXPLICIT_ALLIN", "0")
    return v not in ("0", "", "false", "False", "no", "No", "off", "Off")


def classify_postflop_actions(
    facing_bet: bool, raises_so_far: int,
) -> List[ActionCategory]:
    """Build the category vector for the bias transform.

    This must match the action tree the solver exposes to the rollout
    AND the trainer's generate_actions() enumeration. The explicit all-in
    slot is gated by BPS_ROLLOUT_ADD_EXPLICIT_ALLIN (default OFF for phase3).
    """
    if raises_so_far >= POSTFLOP_MAX_RAISES:
        return [ActionCategory.FOLD, ActionCategory.CALL] if facing_bet \
            else [ActionCategory.CALL]  # check

    add_allin = _add_explicit_allin()

    if facing_bet:
        bet_fracs = SUBSEQUENT_BET_FRACTIONS if raises_so_far > 0 \
            else POSTFLOP_BET_FRACTIONS
        cats = [ActionCategory.FOLD, ActionCategory.CALL]
        cats.extend([ActionCategory.RAISE] * len(bet_fracs))
        if add_allin:
            cats.append(ActionCategory.RAISE)  # all-in
        return cats

    # No bet yet: [check, bet_0.5, bet_1.0, (all-in)?]
    cats = [ActionCategory.CALL]  # check
    cats.extend([ActionCategory.RAISE] * len(POSTFLOP_BET_FRACTIONS))
    if add_allin:
        cats.append(ActionCategory.RAISE)  # all-in
    return cats


def num_postflop_actions(facing_bet: bool, raises_so_far: int) -> int:
    return len(classify_postflop_actions(facing_bet, raises_so_far))


def action_amount_chips(
    action_idx: int, facing_bet: bool, raises_so_far: int,
    pot: int, to_call: int, stack: int,
) -> Tuple[str, int]:
    """Translate an action index into (kind, chip_amount).

    kind is one of 'fold', 'call', 'check', 'bet', 'raise', 'allin'.
    chip_amount is how many chips the player adds to the pot for this
    action (0 for fold/check).

    to_call is the chips the player needs to match to stay in
    (current_bet - their current street_bet).
    """
    cats = classify_postflop_actions(facing_bet, raises_so_far)
    if action_idx < 0 or action_idx >= len(cats):
        raise ValueError(f"action_idx {action_idx} out of range [0, {len(cats)})")

    tree_closed = raises_so_far >= POSTFLOP_MAX_RAISES

    add_allin = _add_explicit_allin()

    if facing_bet:
        if action_idx == 0:
            return ("fold", 0)
        if action_idx == 1:
            return ("call", min(to_call, stack))
        if tree_closed:
            # Should not reach — tree closed has only fold/call.
            return ("call", min(to_call, stack))
        # All-in is the last slot ONLY when the explicit gate is on.
        if add_allin and action_idx == len(cats) - 1:
            return ("allin", stack)
        # Sized raise
        frac_slot = action_idx - 2
        bet_fracs = SUBSEQUENT_BET_FRACTIONS if raises_so_far > 0 \
            else POSTFLOP_BET_FRACTIONS
        frac = bet_fracs[frac_slot]
        # Pluribus: raise_to = to_call + frac * (pot + to_call)
        raise_to = to_call + int(frac * (pot + to_call))
        amount = min(raise_to, stack)
        return ("raise", max(amount, to_call + 1) if stack > to_call + 1 else stack)

    # Not facing bet
    if action_idx == 0:
        return ("check", 0)
    if tree_closed:
        return ("check", 0)
    if add_allin and action_idx == len(cats) - 1:
        return ("allin", stack)
    frac_slot = action_idx - 1
    frac = POSTFLOP_BET_FRACTIONS[frac_slot]
    bet_amount = int(frac * pot)
    return ("bet", min(max(bet_amount, 1), stack))


# ─── Blueprint lookup ────────────────────────────────────────────────────


def _lookup_strategy(
    blueprint,
    board: List[int],
    action_history: List[int],
    player: int,
    bucket: int,
    street: int,
) -> Optional[np.ndarray]:
    """Look up σ̄(bucket) at a postflop decision node.

    Returns the normalized strategy on hit, or None on miss / zero-sum.
    Records coverage hit/miss via leaf_coverage.record() if a
    CoverageScope is active.

    Pluribus Algorithm 1 supplementary: when σ̄ is unavailable or sums
    to zero, the caller substitutes uniform-on-legal-actions. This
    function communicates the substitution-needed signal as None;
    callers MUST handle it.
    """
    from engine.leaf_coverage import record as _rec

    strat = blueprint.get_strategy(
        board, action_history, player, bucket, street=street,
    )
    if strat is None:
        _rec("rollout_sigma", hit=False,
             detail=f"miss_street={street}_p={player}_ah={len(action_history)}")
        return None
    arr = np.asarray(strat, dtype=np.float64)
    if arr.sum() <= 1e-10:
        _rec("rollout_sigma", hit=False,
             detail=f"zerosum_street={street}_p={player}")
        return None
    _rec("rollout_sigma", hit=True)
    return arr / arr.sum()


# ─── Chance / dealing ────────────────────────────────────────────────────


def _deal_one_card(
    used: set, rng: np.random.Generator,
) -> int:
    """Uniform random card from the 52 not in `used`. Raises if deck empty."""
    deck = [c for c in range(52) if c not in used]
    if not deck:
        raise RuntimeError("deck exhausted during rollout card deal")
    return int(rng.choice(deck))


def _sample_opp_hand(
    candidates: List[Tuple[int, int, float]],
    used: set, rng: np.random.Generator,
) -> Optional[Tuple[int, int]]:
    """Weighted sample of one opponent hand that doesn't conflict with `used`.

    Returns None if no valid hand remains (too many cards are used).
    """
    valid = [(c0, c1, w) for c0, c1, w in candidates
             if w > 0.0 and c0 not in used and c1 not in used]
    if not valid:
        return None
    weights = np.fromiter((w for _, _, w in valid), dtype=np.float64)
    total = weights.sum()
    if total <= 0.0:
        return None
    idx = int(rng.choice(len(valid), p=weights / total))
    return (valid[idx][0], valid[idx][1])


# ─── Simulation ──────────────────────────────────────────────────────────


def _street_num(name: str) -> int:
    return {"flop": 1, "turn": 2, "river": 3}[name]


def _next_street(name: str) -> Optional[str]:
    return {"flop": "turn", "turn": "river", "river": None}[name]


def _simulate_one(
    leaf: RolloutLeaf,
    bias_profile: Sequence[int],
    blueprint,
    hero_player: int,
    sampled_hands: List[Optional[Tuple[int, int]]],
    rng: np.random.Generator,
) -> float:
    """Run ONE rollout and return hero's net chip payoff.

    Mutates nothing caller-visible; works on local copies.
    """
    num_players = len(leaf.active)
    active = list(leaf.active)
    stacks = list(leaf.stacks)
    bets = list(leaf.bets)
    pot = leaf.pot_chips
    board = list(leaf.board)
    game_history = list(leaf.game_action_history)
    # Acting order for this street: already pre-ordered by caller for
    # the leaf's street, but subsequent streets need recomputation. We
    # keep the same order module inactive players (folded mid-rollout).
    acting_order = list(leaf.acting_order)

    street = leaf.street
    while True:
        # ── Deal chance cards for the NEXT street, if applicable ──
        # On entry the leaf's `street` is already dealt; we simulate
        # that street's betting, then deal next street's card(s), etc.

        # If every live player is all-in (stack 0), no more betting
        # action is possible. Skip directly to chance / showdown. Without
        # this check, a post-allin river would query σ̄ at a closed
        # betting line that doesn't exist in the trained tree.
        n_live_can_act = sum(
            1 for i in range(num_players) if active[i] and stacks[i] > 0
        )

        # ── Betting round for the current street ──
        street_bets = [0] * num_players
        current_bet = 0
        raises_this_street = 0
        acted = [False] * num_players
        safety = num_players * (POSTFLOP_MAX_RAISES + 2) + 4

        iter_idx = 0
        while iter_idx < safety and n_live_can_act >= 2:
            iter_idx += 1

            # Find next active player in acting_order who still owes action
            chosen = None
            for p in acting_order:
                if not active[p]:
                    continue
                if stacks[p] <= 0:
                    continue  # all-in player can't act
                if not acted[p] or street_bets[p] < current_bet:
                    chosen = p
                    break

            if chosen is None:
                break  # round closed

            # If only one active player remains, round ends.
            if sum(active) <= 1:
                break

            p = chosen
            pair = sampled_hands[p]
            if pair is None:
                # No hand sampled for this player (shouldn't happen for
                # active players); treat as check/fold conservatively.
                acted[p] = True
                continue

            c0, c1 = pair
            # Dispatch by street / board length so turn and river
            # decisions use their own bucketizers, not a silent flop
            # fallback. Fallback to plain hand_to_bucket for mocks that
            # don't define the new methods (e.g. tests/test_rollout_v2.py).
            if len(board) == 4 and hasattr(blueprint, "hand_to_bucket_turn"):
                bucket = blueprint.hand_to_bucket_turn(board, c0, c1)
            elif len(board) == 5 and hasattr(blueprint, "hand_to_bucket_river"):
                bucket = blueprint.hand_to_bucket_river(board, c0, c1)
            else:
                bucket = blueprint.hand_to_bucket(board, c0, c1)
            if bucket < 0:
                raise LeafValueUnavailable(
                    street=street,
                    reason=f"hand_to_bucket returned {bucket} for ({c0},{c1}) on {board}",
                    num_players=num_players,
                )

            facing_bet = current_bet > street_bets[p]
            to_call = current_bet - street_bets[p] if facing_bet else 0
            cats = classify_postflop_actions(facing_bet, raises_this_street)
            n_actions = len(cats)

            # Blueprint lookup: compact slot → full seat for the
            # blueprint's seat-indexed storage.
            assert leaf.player_seats is not None
            full_seat = leaf.player_seats[p]
            strat_full = _lookup_strategy(
                blueprint=blueprint,
                board=board,
                action_history=game_history,
                player=full_seat,
                bucket=bucket,
                street=_street_num(street),
            )
            if strat_full is None:
                # Pluribus Algorithm 1 supplementary: σ̄ unavailable →
                # uniform over the rollout tree's legal action set. The
                # leaf_coverage record from _lookup_strategy keeps the
                # substitution observable.
                strat = np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
            elif len(strat_full) >= n_actions:
                strat = strat_full[:n_actions]
            else:
                # Blueprint has fewer actions than our tree says; this
                # happens when the training tree at that spot is
                # narrower. Take what we have and rescale categories.
                strat = np.zeros(n_actions, dtype=np.float64)
                strat[:len(strat_full)] = strat_full
                cats = cats[:max(len(strat_full), 1)]

            # Post-truncate zero-sum → uniform. Mirrors src/rollout.c:
            # both branches above can produce zero strat (bp_na < n
            # padded with zeros, OR bp_na >= n with all mass in the
            # truncated tail). apply_bias's downstream defensive renorm
            # would silently absorb this; the explicit substitution
            # here keeps the leaf_coverage record visible.
            if strat_full is not None and strat.sum() <= 0.0:
                from engine.leaf_coverage import record as _rec
                _rec("rollout_sigma", hit=False,
                     detail=f"post_truncate_zerosum_p={p}_bucket={bucket}")
                strat = np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
                cats = classify_postflop_actions(facing_bet, raises_this_street)

            biased = apply_bias(strat, int(bias_profile[p]), cats)
            # Re-normalize defensively
            if biased.sum() <= 0.0:
                biased = np.ones_like(biased) / max(len(biased), 1)
            else:
                biased = biased / biased.sum()

            action_idx = int(rng.choice(len(biased), p=biased))

            # Execute action
            kind, amt = action_amount_chips(
                action_idx=action_idx,
                facing_bet=facing_bet,
                raises_so_far=raises_this_street,
                pot=pot,
                to_call=to_call,
                stack=stacks[p],
            )
            if kind == "fold":
                active[p] = False
                acted[p] = True
            elif kind in ("call", "check"):
                stacks[p] -= amt
                street_bets[p] += amt
                bets[p] += amt
                pot += amt
                acted[p] = True
            elif kind in ("bet", "raise", "allin"):
                stacks[p] -= amt
                street_bets[p] += amt
                bets[p] += amt
                pot += amt
                if street_bets[p] > current_bet:
                    current_bet = street_bets[p]
                raises_this_street += 1
                acted[p] = True
                # Opens action for others still live
                for q in range(num_players):
                    if q != p and active[q]:
                        acted[q] = False

            game_history.append(action_idx)

            if sum(active) <= 1:
                break

        # End of street. If only one player remains, terminal.
        if sum(active) <= 1:
            break

        # Deal next street if there is one
        nxt = _next_street(street)
        if nxt is None:
            break  # river betting done → showdown
        used = set(board)
        for pair in sampled_hands:
            if pair is not None:
                used.add(pair[0])
                used.add(pair[1])
        new_card = _deal_one_card(used, rng)
        board.append(new_card)
        street = nxt

    # Terminal — compute payoff
    if sum(active) <= 1:
        # Last player standing wins the pot
        if active[hero_player]:
            return float(pot) - float(bets[hero_player])
        return -float(bets[hero_player])

    # Showdown: deal any missing board cards (river leaves won't need this)
    while len(board) < 5:
        used = set(board)
        for pair in sampled_hands:
            if pair is not None:
                used.add(pair[0])
                used.add(pair[1])
        board.append(_deal_one_card(used, rng))

    return compute_showdown_payoff(
        full_board=board,
        hands=[sampled_hands[p] if sampled_hands[p] is not None else (0, 1)
               for p in range(num_players)],
        active=active,
        bets=bets,
        pot_chips=pot,
        hero_player=hero_player,
    )


# ─── Showdown payoff (winner-takes / chop; no side pots yet) ─────────────


def compute_showdown_payoff(
    full_board: List[int],
    hands: List[Tuple[int, int]],
    active: List[bool],
    bets: List[int],
    pot_chips: int,
    hero_player: int,
) -> float:
    """Compute hero's net chip payoff at showdown.

    Simplification: treats the pot as a single undivided pool; the
    winners split it evenly. Side pots (when players are all-in at
    different amounts) are NOT modeled — this is a known limitation
    for Pluribus faithfulness; Pluribus's rollouts also flatten side
    pots by this approximation per §4. Revisit if it proves material.

    Args:
        full_board: 5 card ints.
        hands: per-player (c0, c1). Inactive players' entries are
            ignored.
        active: per-player bool. False = folded earlier; excluded.
        bets: per-player total chips invested this hand (pre- +
            postflop). Used to compute hero's net.
        pot_chips: total pot at showdown.
        hero_player: whose net payoff we return.

    Returns:
        Hero's net chip payoff (pot share - hero's bets).
    """
    # Active players at showdown
    live = [i for i in range(len(hands)) if active[i]]
    if not live:
        return -float(bets[hero_player])

    # Compute strengths
    strengths = {}
    for p in live:
        c0, c1 = hands[p]
        strengths[p] = _eval7_py(list(full_board) + [c0, c1])

    best = max(strengths.values())
    winners = [p for p, s in strengths.items() if s == best]

    hero_share = 0.0
    if hero_player in winners:
        hero_share = float(pot_chips) / len(winners)

    return hero_share - float(bets[hero_player])


# ─── Public entry point (per-leaf-per-bias-per-hero-hand) ────────────────


def compute_leaf_value(
    leaf: RolloutLeaf,
    bias_profile: Sequence[int],
    blueprint,
    hero_player: int,
    hero_hand_idx: int,
    num_rollouts: int = NUM_ROLLOUTS_DEFAULT,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute the rollout-mean EV for a single (leaf, bias_profile,
    hero, hero_hand).

    For each of `num_rollouts` rollouts:
      1. Sample opponents' hands from their ranges (reach-weighted,
         conflict-avoiding).
      2. Run one rollout via _simulate_one.
      3. Accumulate hero's payoff.
    Return the mean.

    Blueprint coverage misses are absorbed inline via uniform-on-legal-
    actions (Pluribus Algorithm 1 supplementary); leaf_coverage records
    the substitutions. Raises LeafValueUnavailable on genuine errors
    (bucket lookup failure at site 479).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    num_players = len(leaf.active)
    if len(bias_profile) != num_players:
        raise ValueError(
            f"bias_profile length {len(bias_profile)} != num_players {num_players}"
        )
    if not leaf.active[hero_player]:
        # Hero is not in the hand at this leaf — they've folded.
        return -float(leaf.bets[hero_player])

    # Hero's hand
    if hero_hand_idx >= len(leaf.player_hands[hero_player]):
        raise ValueError(
            f"hero_hand_idx {hero_hand_idx} out of range "
            f"(player has {len(leaf.player_hands[hero_player])} hands)"
        )
    hc0, hc1, hweight = leaf.player_hands[hero_player][hero_hand_idx]
    if hweight <= 0.0:
        return 0.0  # out of range
    if hc0 in leaf.board or hc1 in leaf.board:
        return 0.0  # blocked

    payoff_sum = 0.0
    payoffs_recorded = 0
    for _ in range(num_rollouts):
        used = set(leaf.board) | {hc0, hc1}
        sampled: List[Optional[Tuple[int, int]]] = [None] * num_players
        sampled[hero_player] = (hc0, hc1)

        ok = True
        for p in range(num_players):
            if p == hero_player:
                continue
            if not leaf.active[p]:
                continue
            pair = _sample_opp_hand(leaf.player_hands[p], used, rng)
            if pair is None:
                ok = False
                break
            sampled[p] = pair
            used.add(pair[0]); used.add(pair[1])

        if not ok:
            continue

        payoff = _simulate_one(
            leaf=leaf,
            bias_profile=bias_profile,
            blueprint=blueprint,
            hero_player=hero_player,
            sampled_hands=sampled,
            rng=rng,
        )
        payoff_sum += payoff
        payoffs_recorded += 1

    if payoffs_recorded == 0:
        return 0.0
    return payoff_sum / payoffs_recorded


# ─── Multi-hand and multi-leaf wrappers (API-facing) ─────────────────────


def compute_leaf_values_all_hands(
    leaf: RolloutLeaf,
    bias_profile: Sequence[int],
    blueprint,
    max_hands: int,
    num_rollouts: int = NUM_ROLLOUTS_DEFAULT,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute per-player per-hand leaf value for ALL hands in the leaf.

    Returns [num_players, max_hands] float32. Inactive players get 0
    across all their hand slots.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    num_players = len(leaf.active)
    out = np.zeros((num_players, max_hands), dtype=np.float32)

    for p in range(num_players):
        if not leaf.active[p]:
            continue
        hands = leaf.player_hands[p]
        n = min(len(hands), max_hands)
        for h_idx in range(n):
            c0, c1, w = hands[h_idx]
            if w <= 0.0 or c0 in leaf.board or c1 in leaf.board:
                continue
            v = compute_leaf_value(
                leaf=leaf, bias_profile=bias_profile,
                blueprint=blueprint, hero_player=p, hero_hand_idx=h_idx,
                num_rollouts=num_rollouts, rng=rng,
            )
            out[p, h_idx] = np.float32(v)
    return out


def compute_subgame_leaf_values(
    leaf_infos: List,
    board: List[int],
    street: str,
    player_hands: List[List[Tuple[int, int, float]]],
    active: List[bool],
    acting_order: List[int],
    player_seats: List[int],
    game_action_history: List[int],
    blueprint,
    starting_pot_chips: int,
    stack_chips: int,
    max_hands: int,
    num_rollouts: int = NUM_ROLLOUTS_DEFAULT,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Build the full [total_leaves × num_players × max_hands] array.

    Pluribus-faithful implementation. The solver calls this once per CFR
    iteration (or once at setup, depending on how the cont-strat values
    are cached).

    Each leaf is enumerated against all 4^num_players bias combos;
    flat leaf index = leaf_infos[i].leaf_idx * cont_per_leaf + combo.

    Args:
        leaf_infos: from leaf_values.extract_leaf_info_from_tree.
            Each has .leaf_idx, .pot, and optionally .bets.
        board: current street's full board (3/4/5 card ints).
        street: "flop" | "turn" | "river".
        player_hands: per-compact-slot (c0, c1, reach_weight) lists.
        active: per-compact-slot bool. Folded players False.
        acting_order: compact slots, postflop order, filtered to
            active-at-leaf players.
        player_seats: compact slot -> blueprint full seat (0-5).
        game_action_history: full preflop+prior-postflop indices.
        blueprint: BPSReaderC or BlueprintV2.
        starting_pot_chips: pot entering this street.
        stack_chips: starting stack (same for every active player,
            pre-deduction of leaf bets).
        max_hands: max hands per player across the range.
        num_rollouts: Pluribus uses 3. More reduces variance.
        rng: numpy Generator; if None, seeds with 42 for determinism.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    num_players = len(player_hands)
    cont_per_leaf = NUM_BIAS_PROFILES ** num_players
    num_orig = len(leaf_infos)
    total = num_orig * cont_per_leaf

    out = np.zeros((total, num_players, max_hands), dtype=np.float32)

    for li in leaf_infos:
        leaf_bets = list(li.bets) if hasattr(li, "bets") else [0] * num_players
        # Stacks at this leaf: starting stack minus chips invested so
        # far (approximated by leaf_bets here). This is what the solver
        # provides per-leaf via its extract_leaf_info_from_tree.
        stacks = [max(0, stack_chips - b) for b in leaf_bets]

        leaf = RolloutLeaf(
            board=list(board),
            pot_chips=int(li.pot),
            active=list(active),
            stacks=stacks,
            bets=leaf_bets,
            street=street,
            player_hands=player_hands,
            acting_order=list(acting_order),
            game_action_history=list(game_action_history),
            player_seats=list(player_seats),
        )

        for combo in range(cont_per_leaf):
            bias = [(combo // (NUM_BIAS_PROFILES ** p)) % NUM_BIAS_PROFILES
                    for p in range(num_players)]
            flat = li.leaf_idx * cont_per_leaf + combo
            if flat >= total:
                continue
            arr = compute_leaf_values_all_hands(
                leaf=leaf, bias_profile=bias,
                blueprint=blueprint, max_hands=max_hands,
                num_rollouts=num_rollouts, rng=rng,
            )
            out[flat] = arr
    return out


def compute_subgame_leaf_values_sampled(
    leaf_infos: List,
    board: List[int],
    street: str,
    player_hands: List[List[Tuple[int, int, float]]],
    active: List[bool],
    acting_order: List[int],
    player_seats: List[int],
    game_action_history: List[int],
    blueprint,
    starting_pot_chips: int,
    stack_chips: int,
    max_hands: int,
    num_rollouts: int = NUM_ROLLOUTS_DEFAULT,
    num_samples_k: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sampled-cont-strat leaf-value integrator (OP4 / Option c).

    For each betting leaf, draws `num_samples_k` joint bias profiles
    uniformly from {0,1,2,3}^num_players and averages leaf values
    across samples. Each sample runs `num_rollouts` Monte Carlo
    rollouts internally.

    Produces `[num_betting_leaves × num_players × max_hands]` — one
    leaf_value slot per betting leaf, no cont-strat axis. The CFR
    kernel sees a tree where each betting leaf is a single terminal
    (SS_CONT_STRAT_MODE == SAMPLED in street_solve.cuh).

    Pluribus §4 cont-strat is "another action in the subgame,
    selected via MCCFR". MCCFR external-sampling draws one
    realization per iteration; this integrator realizes that draw at
    leaf-evaluation time rather than materializing 4^N terminals in
    the tree. Closes both the 4^N memory blowup AND the sequential-
    observation deviation in the old joint chain — the joint draw is
    atomic, no player observes any other player's choice.

    Variance accounting: at n_active players per betting leaf the
    joint assignment space has 4^n_active outcomes. num_samples_k=1
    is unbiased but noisy; at 6p (4^6 = 4096 outcomes) variance may
    require k=16 or higher for CFR convergence within the existing
    iter budget. The joint-equivalent variance at n_active=N is
    achieved at num_samples_k = 4^(N-n_player_target); e.g., k=16
    at 6p matches the 3p joint variance profile. Tune via
    tools/op4_bench.py. See docs/OP4_CONT_STRAT_MARGINAL_PLAN.md
    §"Variance accounting".

    Args (same as compute_subgame_leaf_values, plus):
        num_samples_k: number of joint cont-strat samples averaged
            per betting leaf. Default 1.
    """
    if num_samples_k < 1:
        raise ValueError(f"num_samples_k must be >= 1, got {num_samples_k}")

    if rng is None:
        rng = np.random.default_rng(42)

    num_players = len(player_hands)
    # num_leaves for the output array is (max leaf_idx + 1) — matches
    # the SAMPLED-mode tree where each betting leaf has its original
    # leaf_idx from the betting-tree pass.
    if not leaf_infos:
        return np.zeros((0, num_players, max_hands), dtype=np.float32)
    max_leaf_idx = max(li.leaf_idx for li in leaf_infos)
    total = max_leaf_idx + 1

    out = np.zeros((total, num_players, max_hands), dtype=np.float32)

    for li in leaf_infos:
        leaf_bets = list(li.bets) if hasattr(li, "bets") else [0] * num_players
        stacks = [max(0, stack_chips - b) for b in leaf_bets]

        leaf = RolloutLeaf(
            board=list(board),
            pot_chips=int(li.pot),
            active=list(active),
            stacks=stacks,
            bets=leaf_bets,
            street=street,
            player_hands=player_hands,
            acting_order=list(acting_order),
            game_action_history=list(game_action_history),
            player_seats=list(player_seats),
        )

        # Draw K independent joint bias profiles. Each sample's bias
        # for folded players is irrelevant (they don't act in the
        # rollout), but we draw for all num_players slots to keep the
        # RNG stream deterministic regardless of who folded.
        sample_sum = np.zeros((num_players, max_hands), dtype=np.float64)
        for sample_idx in range(num_samples_k):
            bias = rng.integers(0, NUM_BIAS_PROFILES, size=num_players).tolist()
            arr = compute_leaf_values_all_hands(
                leaf=leaf, bias_profile=bias,
                blueprint=blueprint, max_hands=max_hands,
                num_rollouts=num_rollouts, rng=rng,
            )
            sample_sum += arr.astype(np.float64)

        out[li.leaf_idx] = (sample_sum / float(num_samples_k)).astype(np.float32)

    return out
