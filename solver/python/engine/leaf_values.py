"""Leaf value computation for Pluribus-style depth-limited solving.

Implements the exact depth-limited approach from:
  Brown & Sandholm, "Depth-Limited Solving for Imperfect-Information Games",
  NeurIPS 2018, Section 4.

At each non-river depth-limit leaf, both players simultaneously choose among
4 continuation strategies for the remainder of the game:
    0. Unmodified blueprint strategy
    1. Fold-biased:  P(fold)  *= 5, renormalize
    2. Call-biased:  P(call)  *= 5, renormalize
    3. Raise-biased: P(raise) *= 5, renormalize

This is modeled as two sequential (but unobserved) decision nodes:
P0 picks among {0,1,2,3}, then P1 picks among {0,1,2,3}, giving 16 pairs.
At each terminal, the value is the EV of playing out the game under the
two chosen biased strategies.

For the flop solver:
  - Each leaf leads to the turn. The continuation EV is computed by
    averaging over all 49 possible turn cards, using precomputed
    per-action EVs from the blueprint.
  - EV under bias = sum_a biased_P(a|hand) * action_EV(a, hand)
  - This is the key insight: we DON'T re-solve — we combine the
    precomputed per-action EVs with the biased action probabilities.

For the turn solver:
  - Each leaf leads to the river. Since river is solved at runtime
    (no depth limit), we use river equity as the continuation value.
  - Both players' biased strategies at the turn leaf affect the
    expected pot share via the equity computation.

The leaf value the GPU kernel expects:
  leaf_values[leaf_idx * 2 * max_hands + player * max_hands + hand]
  = EV per unit of opponent reach for this hand at this leaf.

The kernel multiplies this by the actual opponent reach sum:
  cfv[hand] = leaf_value[hand] * sum(opponent_reach[non_conflicting])
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import namedtuple


class LeafValueUnavailable(RuntimeError):
    """Raised when no trustworthy leaf-value oracle is available for a spot.

    The equity-only N-player fallback produces degenerate strategies, so
    there is no silent fallback: the leaf-oracle selector raises this when
    neither the BPST per-scenario blueprint store nor the BPS3 per-action EV
    path can service a query. Callers should surface this as an explicit
    "data unavailable" error rather than fall through to a plausible-looking
    but wrong strategy.

    Attributes:
        street:   "flop" | "turn" | "river"
        reason:   human-readable explanation (why no oracle is available)
        num_players: player count in the spot
    """

    def __init__(self, street: str, reason: str, num_players: int = 0) -> None:
        self.street = street
        self.reason = reason
        self.num_players = num_players
        super().__init__(f"[{street} {num_players}p] leaf oracle unavailable: {reason}")

    # Ensure a pickle round-trip across multiprocessing pool
    # boundaries reconstructs the same exception. Default pickle uses
    # cls(*self.args) which only has the formatted message → __init__
    # signature mismatch → TypeError on result-handler thread, masking
    # the real error from the worker. Explicit __reduce__ keeps the
    # constructor args.
    def __reduce__(self):
        return (
            self.__class__,
            (self.street, self.reason, self.num_players),
        )


# ── First-order bias approximation (for .bps files without action EVs) ───
#
# Pluribus's depth-limited solving uses 4 continuation strategies (unmodified,
# fold-bias, call-bias, raise-bias) at every leaf and produces 4 different
# leaf values per (hand, leaf) pair via:
#
#     leaf_value(hand, bias_pair) = sum_a biased_P(a | hand, bias) * EV(a, hand)
#
# That requires per-action EVs `EV(a, hand)` from the blueprint, which the v2
# .bps export does not currently store (the binary BlueprintStore format does;
# v2 doesn't). Without per-action EVs, the obvious fallback is "use showdown
# equity for all 16 (s0, s1) pairs", which collapses Pluribus's variance
# reduction to nothing.
#
# This module's first-order bias approximation gives **16 different** leaf
# values per (hand, leaf) by adjusting the per-bias leaf computation along
# two axes:
#
# 1. **Fold equity from biased opponent.** A fold-biased opponent folds more
#    often on the next street; we capture that as a fractional "free win" of
#    the current pot, weighted by FOLD_PROB[bias].
# 2. **Pot-size adjustment from biased participation.** A raise-biased opponent
#    inflates the pot we end up in (when we both go to showdown); a fold-biased
#    opponent shrinks it. POT_FACTOR[bias] captures this multiplicatively.
#
# The result is a defensible first-order approximation: it correctly captures
# the SIGN and rough MAGNITUDE of how each bias profile changes the leaf value,
# even though the exact magnitudes differ from a true per-action EV computation.
# This restores meaningful variance reduction in the GPU CFR's leaf-value pass
# without requiring any blueprint export changes.
#
# Exact leaf values matching Pluribus need per-action EVs in the .bps, written
# by a post-hoc tree walk at export time (the trailing BPR3 section). This
# first-order bias approximation is the path for files without BPR3; the
# rollout-based leaf values (rollout.py, rollout_c.py) are the canonical
# flop/turn leaf oracle.

# Probability that a player folds on the next street, indexed by bias profile:
# 0 = unmodified, 1 = fold-bias, 2 = call-bias, 3 = raise-bias
BIAS_FOLD_PROB = [0.0, 0.35, 0.0, 0.0]

# Multiplicative factor on the pot we end up at if both players go to showdown,
# indexed by bias profile
BIAS_POT_FACTOR = [1.0, 0.7, 1.0, 1.4]


def biased_leaf_value(equity, leaf_pot, s_self, s_opp):
    """First-order bias-adjusted leaf value for a single (hand, leaf, bias-pair).

    Args:
        equity: showdown equity of self vs opp's range, in [0, 1]
        leaf_pot: chip count of the pot at this leaf (positive int)
        s_self: bias profile of self {0,1,2,3}
        s_opp: bias profile of opponent {0,1,2,3}

    Returns:
        EV of self at this leaf in chips, signed (positive = winning chips
        relative to leaf_pot/2 baseline)

    Math:
      - With probability fold_prob_opp = BIAS_FOLD_PROB[s_opp]:
            opponent folds → self wins the leaf_pot/2 they would have invested
                             (approximation: half the pot is "self contribution")
      - With probability 1 - fold_prob_opp:
            both go to showdown
            adjusted_pot = leaf_pot * BIAS_POT_FACTOR[s_self] * BIAS_POT_FACTOR[s_opp]
            self EV at showdown = (equity - 0.5) * adjusted_pot
    """
    fold_prob_opp = BIAS_FOLD_PROB[s_opp]
    pot_factor = BIAS_POT_FACTOR[s_self] * BIAS_POT_FACTOR[s_opp]
    showdown_pot = leaf_pot * pot_factor

    fold_payoff = 0.5 * leaf_pot     # we win the half opponent would have put in
    showdown_payoff = (equity - 0.5) * showdown_pot

    return fold_prob_opp * fold_payoff + (1.0 - fold_prob_opp) * showdown_payoff


# ── Continuation strategy biasing (Pluribus, 5x multiplier) ──────────────

def classify_actions(num_actions: int, first_action_is_fold: bool):
    """Classify action indices into fold/call/raise categories.

    In our tree, actions at the start of a new street (turn root) are:
      OOP first to act: [Check, Bet33%, Bet75%, All-in]
      IP facing a bet:  [Fold, Call, Raise33%, Raise75%, All-in]

    Since the precomputed blueprint stores strategies at the TURN ROOT
    (start of turn betting), there is NO fold — P0 acts first with
    check/bet options.

    Returns:
        dict mapping action index to category ('fold', 'call', 'raise')
    """
    categories = {}
    if first_action_is_fold:
        categories[0] = 'fold'
        categories[1] = 'call'
        for a in range(2, num_actions):
            categories[a] = 'raise'
    else:
        categories[0] = 'call'   # check
        for a in range(1, num_actions):
            categories[a] = 'raise'  # bets
    return categories


def bias_strategy(strategy: np.ndarray, bias_type: int,
                  action_categories: dict,
                  multiplier: float = 5.0) -> np.ndarray:
    """Apply Pluribus-style bias to a strategy array.

    Args:
        strategy: [num_hands, num_actions] — weighted avg P(action|hand)
        bias_type: 0=unmodified, 1=fold-biased, 2=call-biased, 3=raise-biased
        action_categories: dict {action_idx: 'fold'|'call'|'raise'}
        multiplier: Pluribus uses 5x (paper Section 4)

    Returns:
        [num_hands, num_actions] — biased strategy (renormalized)
    """
    if bias_type == 0:
        return strategy.copy()

    target = {1: 'fold', 2: 'call', 3: 'raise'}[bias_type]
    biased = strategy.copy()

    for a, cat in action_categories.items():
        if a < biased.shape[1] and cat == target:
            biased[:, a] *= multiplier

    # Renormalize rows
    row_sums = biased.sum(axis=1, keepdims=True)
    valid = row_sums > 1e-10
    biased = np.where(valid, biased / np.maximum(row_sums, 1e-10),
                      1.0 / biased.shape[1])
    return biased


# ── Leaf info extraction from tree ───────────────────────────────────────

# Postflop training action enumeration vocab.
#
# These module-level tuples are the compiled DEFAULTS ([0.5, 1.0]
# first / [1.0] subsequent), NOT the published training vocab. The published 8.05B
# blueprint was trained with postflop_bet_sizes=[0.75] / subsequent=[1.0]
# (manifest footer, config_hash bc8ac105). The leaf encoder MUST enumerate
# THAT vocab or it mis-keys every flop all-in line, because the all-in index
# moves with the vocab length. So the real vocab is THREADED
# from the loaded .bps manifest through `extract_leaf_info_from_tree` /
# `_train_action_idx`; these defaults exist ONLY so a caller that supplies no
# vocab keeps the original byte-for-byte indices (byte-identity). They
# are NOT load-bearing for the published configuration: its caller
# (api/app/engine/cpu_es_solver.py) always passes the manifest vocab.
#
# Subgame trees may use richer sizes (off-tree extras from Pluribus
# Algorithm 2 nested re-solve); _train_action_idx fuzzy-maps subgame
# sizes back to the training-time slots so the per-leaf flop_history
# fed into the rollout's blueprint queries hashes to the keys training
# stored.
_POSTFLOP_FIRST_RAISE_FRACS = (0.5, 1.0)
_POSTFLOP_SUBSEQUENT_RAISE_FRACS = (1.0,)
_POSTFLOP_MAX_RAISES = 3

# Default — most code paths supply a richer struct via the new positional
# field. The hasattr() guard in rollout_c.py handles the absent-field case
# for legacy callers (LeafInfo without flop_history_train works as before;
# the rollout falls back to the dropped-history behavior).
# flop_actor_train: the per-step ACTOR (node.player) parallel to
# flop_history_train — same length, same DFS order. Consumed by the 2p-flop
# opponent-range conditioning (leaf_villain_conditioning.py) to attribute each
# flop action on a leaf's line to the seat that took it. Appended LAST and
# defaulted so every existing positional/legacy caller is unaffected (the rollout
# never reads it; only the Python conditioning helper does).
LeafInfo = namedtuple(
    'LeafInfo',
    ['leaf_idx', 'pot', 'bets', 'flop_history_train', 'flop_actor_train'],
    defaults=((), ()),
)


def _train_action_idx(facing_bet, raises_so_far, kind, size_frac,
                      first_fracs=_POSTFLOP_FIRST_RAISE_FRACS,
                      subseq_fracs=_POSTFLOP_SUBSEQUENT_RAISE_FRACS):
    """Map a (kind, size_frac) action back to its training-time action index.

    Mirrors the trainer's generate_actions enumeration
    (solver/src/blueprint/mccfr_blueprint.c) and the C rollout's
    rollout_classify_postflop_actions_c so the per-leaf flop history hashes
    to the same action_hash training stored. The all-in index is
    ``num_bet_sizes`` PAST the last sized-raise slot, so it moves with the
    vocab length: a hard-coded vocab that differs from training mis-keys
    every all-in line.

    The vocab is a PARAMETER threaded from the loaded .bps manifest
    (postflop_bet_sizes -> first_fracs, subsequent_bet_sizes -> subseq_fracs);
    the module-level defaults are [0.5,1.0]/[1.0] and exist only
    so a caller that passes nothing keeps the original indices.

    Subgame trees may have richer raise sizes; closest-fraction match maps
    them to the training-time slots. This is lossy at subsequent-raise nodes
    when training stored fewer sizes there — the only correct option, the
    blueprint has nothing to return for sizes it never trained on.

    Enumerations (with the published [0.75] first / [1.0] subsequent vocab):
        tree_closed (raises_so_far >= 3) + facing: [fold(0), call(1)]
        tree_closed + !facing:                     [check(0)]
        facing + first raise:    [fold(0), call(1), 0.75(2), allin(3)]
        facing + subsequent:     [fold(0), call(1), 1.0(2), allin(3)]
        !facing:                 [check(0), 0.75(1), allin(2)]

    Args:
        facing_bet: True if to_call > 0 at the parent node.
        raises_so_far: number of raises already made this street.
        kind: 'fold' | 'check' | 'call' | 'bet' | 'raise' | 'allin'.
        size_frac: pot fraction for bet/raise; ignored otherwise.
        first_fracs: first-raise (opening bet) pot fractions from the manifest.
        subseq_fracs: subsequent-raise pot fractions from the manifest.
    """
    tree_closed = raises_so_far >= _POSTFLOP_MAX_RAISES

    if tree_closed:
        if facing_bet:
            return 0 if kind == 'fold' else 1
        return 0

    if facing_bet:
        if kind == 'fold':
            return 0
        if kind in ('call', 'check'):
            return 1
        if raises_so_far == 0:
            # [fold, call, first_fracs..., allin@2+len(first_fracs)]
            if kind == 'allin':
                return 2 + len(first_fracs)
            best_i = min(
                range(len(first_fracs)),
                key=lambda i: abs(size_frac - first_fracs[i]),
            )
            return 2 + best_i
        # subsequent raise: [fold, call, subseq_fracs..., allin@2+len(subseq)]
        if kind == 'allin':
            return 2 + len(subseq_fracs)
        best_i = min(
            range(len(subseq_fracs)),
            key=lambda i: abs(size_frac - subseq_fracs[i]),
        )
        return 2 + best_i

    # !facing: [check, first_fracs..., allin@1+len(first_fracs)]
    if kind == 'check':
        return 0
    if kind == 'allin':
        return 1 + len(first_fracs)
    best_i = min(
        range(len(first_fracs)),
        key=lambda i: abs(size_frac - first_fracs[i]),
    )
    return 1 + best_i


def extract_leaf_info_from_tree(
    tree_data, num_players=2,
    postflop_bet_sizes=None, subsequent_bet_sizes=None,
) -> List[LeafInfo]:
    """Extract pot/bets/flop_history for each subgame leaf.

    DFS from the SS tree root accumulates the action sequence taken to
    reach each leaf. Each action gets re-encoded to training's discrete
    index space via _train_action_idx; the resulting tuple is stored on
    LeafInfo.flop_history_train and threaded into the rollout so
    blueprint queries during turn/river include the real flop slice. With
    only the preflop prefix those queries miss and the solve collapses to
    uniform 1/n.

    postflop_bet_sizes / subsequent_bet_sizes: the TRAINING bet vocab from
    the loaded .bps manifest (postflop_bet_sizes / subsequent_bet_sizes).
    The encoded flop_history_train indices MUST match the vocab the
    blueprint was trained with, or every flop all-in line mis-keys. When
    None, falls back to the module defaults ([0.5,1.0]/[1.0]) so legacy
    callers are byte-identical; cpu_es_solver passes the manifest
    vocab explicitly.

    Handles both SS tree shapes (compile-time SS_CONT_STRAT_MODE):

      * JOINT (legacy): each betting leaf is wrapped in a 4^N cont-chain
        of decision nodes. Detected via `_is_cont_chain`; one LeafInfo
        emitted per cont-chain root, leaf_idx recovered from
        `base_leaf_idx // cont_per_leaf`.
      * SAMPLED (the default): no tree-level
        expansion. Each betting-tree terminal stays as SS_NODE_LEAF
        (type=4). One LeafInfo emitted per such terminal directly,
        leaf_idx taken from `node.leaf_idx`. A search for only the
        cont-chain pattern would return no leaves for SAMPLED trees, so
        CFR would run on zero leaf values and play uniform 1/n at every
        postflop spot.

    Args:
        tree_data: SSTreeData ctypes struct (after ss_build_tree or
            ss_build_tree2, the per-level bet_sizes variant; both
            produce the same SSTreeData layout)
        num_players: number of active players (2-6)

    Returns:
        list of LeafInfo, sorted by leaf_idx (matches original ordering).
    """
    cont_per_leaf = 4 ** num_players  # 16 for 2p, 64 for 3p, etc.
    effective_stack = int(tree_data.effective_stack)

    # Normalize the training vocab (None -> module defaults for legacy
    # callers; cpu_es_solver passes the manifest vocab). Tuples so the
    # closure below and _train_action_idx see stable, indexable sequences.
    first_fracs = (tuple(postflop_bet_sizes)
                   if postflop_bet_sizes is not None
                   else _POSTFLOP_FIRST_RAISE_FRACS)
    subseq_fracs = (tuple(subsequent_bet_sizes)
                    if subsequent_bet_sizes is not None
                    else _POSTFLOP_SUBSEQUENT_RAISE_FRACS)

    def _is_cont_chain(node_idx, depth):
        """Check if node_idx is the root of a N-deep continuation chain.
        Each level has num_children=4, depth levels total, terminals are
        SS_NODE_LEAF (type=4) at the bottom."""
        node = tree_data.nodes[node_idx]
        if node.type != 0 or node.num_children != 4:
            return False
        if depth == num_players - 1:
            for c in range(4):
                child_idx = tree_data.children[node.first_child + c]
                if tree_data.nodes[child_idx].type != 4:
                    return False
            return True
        for c in range(4):
            child_idx = tree_data.children[node.first_child + c]
            if not _is_cont_chain(child_idx, depth + 1):
                return False
        return True

    def _get_first_terminal(node_idx):
        """Walk down the first-child chain to find the first terminal leaf."""
        cur = node_idx
        for _ in range(num_players):
            cur = tree_data.children[tree_data.nodes[cur].first_child]
        return cur

    leaves = []
    # Iterative DFS — Python's default recursion limit is 1000 and SS
    # trees at 6p with off-tree sizes can exceed that. Stack carries
    # (node_idx, history_tuple, actors_tuple, raises_so_far). actors is the
    # parallel per-step acting seat (node.player at each decision on the line),
    # threaded for the 2p-flop range conditioning; same length as history.
    stack = [(0, (), (), 0)]
    while stack:
        node_idx, history, actors, raises_so_far = stack.pop()
        node = tree_data.nodes[node_idx]

        if node.type == 0 and node.num_children == 4 and \
                _is_cont_chain(node_idx, 0):
            first_term_idx = _get_first_terminal(node_idx)
            base_leaf_idx = tree_data.nodes[first_term_idx].leaf_idx
            orig_idx = base_leaf_idx // cont_per_leaf
            bets = tuple(node.bets[p] for p in range(num_players))
            leaves.append(LeafInfo(
                leaf_idx=orig_idx,
                pot=node.pot,
                bets=bets,
                flop_history_train=history,
                flop_actor_train=actors,
            ))
            continue

        if node.type == 4:  # SS_NODE_LEAF — SAMPLED-mode betting terminal
            bets = tuple(node.bets[p] for p in range(num_players))
            leaves.append(LeafInfo(
                leaf_idx=int(node.leaf_idx),
                pot=node.pot,
                bets=bets,
                flop_history_train=history,
                flop_actor_train=actors,
            ))
            continue

        if node.type != 0:  # FOLD / SHOWDOWN / CHANCE — no leaf_idx
            continue

        acting = node.player
        if acting < 0:
            continue

        max_parent_bet = max(node.bets[p] for p in range(num_players))
        parent_acting_bet = node.bets[acting]
        to_call = max_parent_bet - parent_acting_bet
        facing = to_call > 0

        # Iterate children in REVERSE so DFS pop order matches forward
        # order (mostly aesthetic — leaves get sorted by leaf_idx anyway).
        for c in range(node.num_children - 1, -1, -1):
            child_idx = tree_data.children[node.first_child + c]
            child = tree_data.nodes[child_idx]

            if child.type == 1:  # SS_NODE_FOLD
                kind, frac = 'fold', 0.0
                new_raises = raises_so_far
            else:
                bet_delta = child.bets[acting] - parent_acting_bet
                if bet_delta == to_call:
                    kind = 'check' if to_call == 0 else 'call'
                    frac = 0.0
                    new_raises = raises_so_far
                else:
                    # Detect all-in: acting player's CUMULATIVE this-street
                    # bet at the child equals their effective stack. Note
                    # bets[] is per-street in the SS tree, so this only
                    # fires when this single street's bet maxed the stack
                    # — which is the case the rollout's classify treats
                    # as the trailing 'allin' index. Sized raise that does
                    # NOT exhaust stack stays as 'bet'/'raise'.
                    is_allin = child.bets[acting] >= effective_stack
                    if to_call == 0:
                        kind = 'allin' if is_allin else 'bet'
                        frac = (bet_delta / node.pot) if node.pot > 0 else 0.0
                    else:
                        kind = 'allin' if is_allin else 'raise'
                        denom = node.pot + to_call
                        frac = ((bet_delta - to_call) / denom) if denom > 0 else 0.0
                    new_raises = raises_so_far + 1

            train_idx = _train_action_idx(
                facing, raises_so_far, kind, frac,
                first_fracs=first_fracs, subseq_fracs=subseq_fracs,
            )
            stack.append((child_idx, history + (train_idx,),
                          actors + (acting,), new_raises))

    leaves.sort(key=lambda x: x.leaf_idx)
    return leaves


# ── Flop leaf values via BPR3 per-action EVs (unified .bps) ──────────────


def compute_flop_leaf_values_bpr3(
    flop_board: List[int],
    player_hands: List[List[Tuple[int, int, float]]],
    blueprint_v2,
    game_action_indices: List[int],
    leaf_infos: List[LeafInfo],
    max_hands: int,
    starting_pot: int,
) -> Optional[np.ndarray]:
    """Compute flop leaf values using per-action EVs from the unified .bps.

    Uses BlueprintV2.get_all_bucket_action_evs() to look up the per-action
    expected values at the flop root for the given game_action_indices
    (preflop action sequence). Combines with 4 bias profiles per Pluribus §4
    to produce 4^N continuation values per leaf.

    This is faster than rollout (~instant lookup vs Monte Carlo simulation)
    and more accurate than the equity-only first-order approximation, because
    it uses the trained blueprint's actual strategic values.

    Raises RuntimeError if the blueprint lookup misses (info set not visited
    during the EV walk). Callers must ensure bps_compute_evs() has been run
    with sufficient iterations before using this function.

    Args:
        flop_board: [card0, card1, card2] ints
        player_hands: list of per-player hands [(c0, c1, weight), ...]
        blueprint_v2: BlueprintV2 instance with loaded action EVs
        game_action_indices: full game action indices from preflop
        leaf_infos: from extract_leaf_info_from_tree()
        max_hands: max hands across all players
        starting_pot: pot in chips at start of this street

    Returns:
        np.array[num_total_leaves, num_players, max_hands] float32
        or None if blueprint lookup fails.
    """
    num_players = len(player_hands)
    num_orig_leaves = len(leaf_infos)
    cont_per_leaf = 4 ** num_players
    total_leaves = num_orig_leaves * cont_per_leaf

    # Look up per-action EVs and strategies at the flop root for each player.
    # The action_hash is computed from game_action_indices (preflop sequence).
    # board=[] and street=1 (flop) with board_hash=0 per the unified format.
    try:
        from engine.leaf_coverage import record as _cov_record
    except ImportError:
        from engine.leaf_coverage import record as _cov_record  # type: ignore

    player_evs = []
    player_strats = []
    for p in range(num_players):
        evs = blueprint_v2.get_all_bucket_action_evs(
            board=[], action_history=game_action_indices,
            player=p, street=1,
        )
        strats = blueprint_v2.get_all_bucket_strategies(
            board=[], action_history=game_action_indices,
            player=p, street=1,
        )
        if evs is None or strats is None:
            _cov_record("bpr3_flop_root", hit=False,
                        detail=f"p{p}_ah{len(game_action_indices)}")
            raise RuntimeError(
                f"BPR3 lookup failed for player {p}, action_history="
                f"{game_action_indices}, street=1. "
                f"Run bps_compute_evs() to populate EV data."
            )
        _cov_record("bpr3_flop_root", hit=True)
        player_evs.append(evs)
        player_strats.append(strats)

    # Compute biased strategies (4 profiles per player)
    player_biased = []
    for p in range(num_players):
        strats = player_strats[p]  # [num_buckets, num_actions]
        na = strats.shape[1]
        # Flop root is first-to-act (OOP): check/bet, no fold
        cats = classify_actions(na, first_action_is_fold=False)
        biased = [bias_strategy(strats, bt, cats, 5.0) for bt in range(4)]
        player_biased.append(biased)

    # Build hand → bucket mapping for each player.
    # Uses the precomputed texture cache via blueprint_v2.hand_to_bucket()
    # when available (BPSReaderC with loaded texture_cache.bin). This gives
    # the exact k-means bucket assignments from training — Pluribus-faithful.
    # Falls back to EHS percentile approximation if no texture cache.
    num_buckets = player_evs[0].shape[0]  # typically 200

    has_texture_cache = hasattr(blueprint_v2, 'hand_to_bucket')
    board_set = set(flop_board)
    player_buckets = []
    for p in range(num_players):
        buckets = []
        for h in range(len(player_hands[p])):
            c0, c1 = player_hands[p][h][0], player_hands[p][h][1]
            if c0 in board_set or c1 in board_set:
                buckets.append(0)  # blocked hand — bucket 0 as placeholder
                continue

            if has_texture_cache:
                bucket = blueprint_v2.hand_to_bucket(flop_board, c0, c1)
                if bucket >= 0:
                    buckets.append(bucket)
                    continue

            # Fallback: EHS percentile approximation (inaccurate but functional)
            ehs_approx = h / max(len(player_hands[p]) - 1, 1)
            bucket = int(ehs_approx * (num_buckets - 1))
            bucket = max(0, min(bucket, num_buckets - 1))
            buckets.append(bucket)
        player_buckets.append(buckets)

    # Compute leaf values: for each leaf × bias combo × player × hand
    leaf_values = np.zeros((total_leaves, num_players, max_hands), dtype=np.float32)

    for li_idx, li in enumerate(leaf_infos):
        leaf_pot = li.pot

        for combo in range(cont_per_leaf):
            flat_idx = li.leaf_idx * cont_per_leaf + combo
            if flat_idx >= total_leaves:
                continue

            # Decode bias choices
            biases = []
            tmp = combo
            for _ in range(num_players):
                biases.append(tmp % 4)
                tmp //= 4

            for p in range(num_players):
                evs = player_evs[p]       # [num_buckets, num_actions]
                biased = player_biased[p]  # list of 4 [num_buckets, num_actions]
                na = evs.shape[1]
                s_p = biases[p]
                biased_strat = biased[s_p]  # [num_buckets, num_actions]

                for h in range(min(len(player_hands[p]), max_hands)):
                    bucket = player_buckets[p][h]
                    if bucket >= evs.shape[0]:
                        continue

                    # EV = sum_a(biased_strategy[bucket, a] * action_ev[bucket, a])
                    ev = 0.0
                    for a in range(na):
                        ev += float(biased_strat[bucket, a]) * float(evs[bucket, a])

                    # Scale by leaf pot relative to starting pot
                    if starting_pot > 0:
                        ev *= leaf_pot / starting_pot

                    leaf_values[flat_idx, p, h] = ev

    return leaf_values


# ── Python hand evaluator ────────────────────────────────────────────────

def _eval5_py(cards):
    """Evaluate 5-card hand. Returns comparable tuple."""
    ranks = sorted([c >> 2 for c in cards], reverse=True)
    suits = [c & 3 for c in cards]
    is_flush = len(set(suits)) == 1
    is_straight = False
    high = ranks[0]
    if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
        is_straight = True
    if ranks == [12, 3, 2, 1, 0]:
        is_straight = True
        high = 3

    from collections import Counter
    cnt = Counter(ranks)
    groups = sorted(cnt.items(), key=lambda x: (-x[1], -x[0]))

    if is_straight and is_flush:
        return (9, high)
    if groups[0][1] == 4:
        return (8, groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (7, groups[0][0], groups[1][0])
    if is_flush:
        return (6,) + tuple(ranks)
    if is_straight:
        return (5, high)
    if groups[0][1] == 3:
        kickers = sorted([r for r, c in groups if c == 1], reverse=True)
        return (4, groups[0][0]) + tuple(kickers)
    if groups[0][1] == 2 and groups[1][1] == 2:
        pairs = sorted([r for r, c in groups if c == 2], reverse=True)
        kicker = [r for r, c in groups if c == 1][0]
        return (3, pairs[0], pairs[1], kicker)
    if groups[0][1] == 2:
        kickers = sorted([r for r, c in groups if c == 1], reverse=True)
        return (2, groups[0][0]) + tuple(kickers)
    return (1,) + tuple(ranks)


def _eval7_py(cards):
    """Evaluate 7-card hand by trying all 21 5-card combinations."""
    from itertools import combinations
    best = None
    for combo in combinations(range(7), 5):
        hand = [cards[i] for i in combo]
        val = _eval5_py(hand)
        if best is None or val > best:
            best = val
    return best
