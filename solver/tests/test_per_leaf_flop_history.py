"""Validates the per-leaf flop_history extraction + threading.

Local part: synthetic 2p flop tree built directly via SSNode/SSTreeData
ctypes structs (no CUDA, no .so, no blueprint needed — the structs are
plain ctypes and extract_leaf_info_from_tree is pure python). Walk it,
verify the reconstructed flop_history matches what each leaf was reached
through. Verifies _train_action_idx + the DFS traversal path.

Bet vocab: the action-index re-encoder (_train_action_idx) and the DFS
extractor take the training bet vocab as a PARAMETER (threaded from the
loaded .bps manifest) instead of a hardcoded [0.5,1.0]/[1.0]. This file
therefore DERIVES the expected indices from the vocab under test and is
parametrized over BOTH:

  * the published 8.05B vocab [0.75] first / [1.0] subsequent
    (manifest footer, config_hash bc8ac105), and
  * the LEGACY default vocab [0.5,1.0] first / [1.0] subsequent
    (the older compiled default; passing None reproduces it).

Under the [0.75] vocab a hardcoded [0.5,1.0] encoder mis-keys every all-in
(trainer all-in idx 2 non-facing / 3 facing vs the encoder's 3 / 4). The
parametric expectations below encode the CORRECT (vocab-derived) indices
for each vocab, so the regression is pinned for [0.75] and the legacy
[0.5,1.0] behavior is preserved.

An end-to-end check (rollout miss rate against a full blueprint) is not
part of this file.
"""
from __future__ import annotations

import ctypes
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from engine.street_solver_gpu import (  # type: ignore[import-untyped]
    SSNode,
    SSTreeData,
    SS_MAX_PLAYERS,
)
from engine.leaf_values import (  # type: ignore[import-untyped]
    LeafInfo,
    _train_action_idx,
    extract_leaf_info_from_tree,
)


# ── Vocab parametrization ────────────────────────────────────────────────
#
# Each vocab is (first_fracs, subseq_fracs). SERVED is the 8.05B training
# vocab; LEGACY is the older compiled default. `None` for the extractor
# kwargs reproduces LEGACY exactly, so the LEGACY case with explicit fracs
# equal to the module defaults also covers the "vocab omitted"
# (byte-identity) path.
SERVED_FIRST = (0.75,)
SERVED_SUBSEQ = (1.0,)
LEGACY_FIRST = (0.5, 1.0)
LEGACY_SUBSEQ = (1.0,)

VOCABS = [
    pytest.param(SERVED_FIRST, SERVED_SUBSEQ, id="served_0.75"),
    pytest.param(LEGACY_FIRST, LEGACY_SUBSEQ, id="legacy_0.5_1.0"),
]


def _closest(fracs, size_frac):
    """Reference: index of the closest training fraction (fuzzy match)."""
    return min(range(len(fracs)), key=lambda i: abs(size_frac - fracs[i]))


# ── Action-index re-encoder direct tests (vocab-parametric) ──────────────
#
# _train_action_idx enumerates, per the trainer's generate_actions rule
# (src/blueprint/mccfr_blueprint.c:954-991):
#   !facing:        [check(0), first_fracs...(1..n), allin(1+n)]
#   facing, first:  [fold(0), call(1), first_fracs...(2..1+n), allin(2+n)]
#   facing, subseq: [fold(0), call(1), subseq_fracs...(2..1+m), allin(2+m)]
#   tree_closed + facing:  [fold(0), call(1)]
#   tree_closed + !facing: [check(0)]
# where n = len(first_fracs), m = len(subseq_fracs). The all-in index moves
# with the vocab length — the exact desync this parametrization guards.


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_train_action_idx_first_raise_facing(first_fracs, subseq_fracs):
    # facing + first raise: [fold(0), call(1), <first_fracs @2..>, allin@2+n]
    n = len(first_fracs)
    assert _train_action_idx(True, 0, "fold", 0.0,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == 0
    assert _train_action_idx(True, 0, "call", 0.0,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == 1
    # Each trained size lands at 2 + its slot.
    for i, frac in enumerate(first_fracs):
        assert _train_action_idx(True, 0, "raise", frac,
                                 first_fracs=first_fracs,
                                 subseq_fracs=subseq_fracs) == 2 + i
    # All-in sits ONE PAST the last sized-raise slot.
    assert _train_action_idx(True, 0, "allin", 1.0,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == 2 + n
    # Fuzzy: an off-tree size maps to the closest trained slot.
    for probe in (0.6, 0.8, 0.55, 0.95):
        assert _train_action_idx(True, 0, "raise", probe,
                                 first_fracs=first_fracs,
                                 subseq_fracs=subseq_fracs) == \
            2 + _closest(first_fracs, probe)


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_train_action_idx_subsequent_facing(first_fracs, subseq_fracs):
    # facing + subsequent (raises_so_far>=1):
    #   [fold(0), call(1), <subseq_fracs @2..>, allin@2+m]
    m = len(subseq_fracs)
    assert _train_action_idx(True, 1, "fold", 0.0,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == 0
    assert _train_action_idx(True, 1, "call", 0.0,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == 1
    for i, frac in enumerate(subseq_fracs):
        assert _train_action_idx(True, 1, "raise", frac,
                                 first_fracs=first_fracs,
                                 subseq_fracs=subseq_fracs) == 2 + i
    # Lossy: a subgame size with no subsequent counterpart maps to closest.
    assert _train_action_idx(True, 1, "raise", 0.5,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == \
        2 + _closest(subseq_fracs, 0.5)
    assert _train_action_idx(True, 1, "allin", 1.0,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == 2 + m


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_train_action_idx_no_facing(first_fracs, subseq_fracs):
    # !facing: [check(0), <first_fracs @1..>, allin@1+n]
    n = len(first_fracs)
    assert _train_action_idx(False, 0, "check", 0.0,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == 0
    for i, frac in enumerate(first_fracs):
        assert _train_action_idx(False, 0, "bet", frac,
                                 first_fracs=first_fracs,
                                 subseq_fracs=subseq_fracs) == 1 + i
    assert _train_action_idx(False, 0, "allin", 1.0,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == 1 + n


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_train_action_idx_tree_closed(first_fracs, subseq_fracs):
    # tree_closed (raises_so_far >= 3) + facing: [fold(0), call(1)]
    assert _train_action_idx(True, 3, "fold", 0.0,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == 0
    assert _train_action_idx(True, 3, "call", 0.0,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == 1
    # tree_closed + !facing: [check(0)]
    assert _train_action_idx(False, 3, "check", 0.0,
                             first_fracs=first_fracs,
                             subseq_fracs=subseq_fracs) == 0


# ── Regression: the exact all-in index divergence ───────────────────────
#
# Under the published [0.75] vocab an all-in must sit at idx 2
# (non-facing) / 3 (facing); a consumer hardcoded to [0.5,1.0] puts
# it at 3 / 4. Pin BOTH vocabs so a future edit that reintroduces the desync
# fails here.


def test_allin_index_divergence_served_vs_legacy():
    # SERVED [0.75]: non-facing all-in = 2, facing all-in = 3.
    assert _train_action_idx(False, 0, "allin", 1.0,
                             first_fracs=SERVED_FIRST,
                             subseq_fracs=SERVED_SUBSEQ) == 2
    assert _train_action_idx(True, 0, "allin", 1.0,
                             first_fracs=SERVED_FIRST,
                             subseq_fracs=SERVED_SUBSEQ) == 3
    # LEGACY [0.5,1.0]: non-facing all-in = 3, facing all-in = 4 (the
    # hardcoded values — correct FOR the [0.5,1.0] vocab, wrong for [0.75]).
    assert _train_action_idx(False, 0, "allin", 1.0,
                             first_fracs=LEGACY_FIRST,
                             subseq_fracs=LEGACY_SUBSEQ) == 3
    assert _train_action_idx(True, 0, "allin", 1.0,
                             first_fracs=LEGACY_FIRST,
                             subseq_fracs=LEGACY_SUBSEQ) == 4


def test_train_action_idx_default_vocab_is_legacy():
    # Omitting the vocab kwargs reproduces the legacy [0.5,1.0]/[1.0]
    # indices byte-for-byte (byte-identity invariant for legacy callers).
    assert _train_action_idx(False, 0, "allin", 1.0) == 3
    assert _train_action_idx(True, 0, "allin", 1.0) == 4
    assert _train_action_idx(False, 0, "bet", 0.5) == 1
    assert _train_action_idx(False, 0, "bet", 1.0) == 2
    assert _train_action_idx(True, 0, "raise", 0.5) == 2
    assert _train_action_idx(True, 0, "raise", 1.0) == 3


# ── LeafInfo backward compat ────────────────────────────────────────────


def test_leaf_info_backwards_compatible():
    # Existing 3-arg call still works (legacy callers).
    li = LeafInfo(leaf_idx=0, pot=100, bets=(50, 50))
    assert li.flop_history_train == ()
    # 4-arg with explicit history.
    li2 = LeafInfo(leaf_idx=1, pot=200, bets=(100, 100), flop_history_train=(1, 2))
    assert li2.flop_history_train == (1, 2)


# ── Synthetic SS tree fixtures ──────────────────────────────────────────
#
# Build a hand-rolled SSTreeData with raw nodes/children arrays. This avoids
# calling ss_build_tree; the fixtures set up only the small
# tree structure that extract_leaf_info_from_tree walks. The 2p cont-chain
# requires 2 levels of 4-children DECISION nodes terminating in 4 LEAF
# nodes per branch (16 LEAFs per cont-chain root).

_NODE_DECISION = 0
_NODE_FOLD = 1
_NODE_SHOWDOWN = 2
_NODE_LEAF = 4


def _make_node(
    type_, player, num_children, first_child, pot, bets,
    leaf_idx=-1,
):
    n = SSNode()
    n.type = type_
    n.player = player
    n.num_children = num_children
    n.first_child = first_child
    n.pot = pot
    n.num_board = 3
    n.leaf_idx = leaf_idx
    n.fold_player = -1
    n.num_players = 2
    for p in range(SS_MAX_PLAYERS):
        n.bets[p] = int(bets[p]) if p < len(bets) else 0
        n.active_players[p] = 1 if p < 2 else 0
        n.board_cards[p % 5] = 0
    n.chance_card_per_child_offset = 0
    n.chance_node_idx = -1
    return n


def _build_2p_synthetic_tree(
    actions_to_take, effective_stack=10000,
):
    """Build a minimal SSTreeData with one path of decision nodes leading
    to a single 2p cont-chain root (1 leaf).

    actions_to_take is a list of (acting_player, kind, frac, parent_pot,
    parent_bets) describing the chain from root to the cont-chain root.

    The cont-chain itself is 2 levels deep × 4 children = 16 LEAF terminals.
    """
    nodes = []
    children = []

    # Build the path of decision nodes. Each has num_children=1 (only the
    # action we took), pointing to the next node.
    pot = 200  # preflop residual pot
    bets = [0, 0]
    parent_idx = 0  # root will be created first

    # Track node indices as we go. Root is index 0 placeholder.
    # We append placeholders and patch first_child after.
    # For simplicity: linear chain (each step has 1 child).
    chain_node_indices = []
    state_pot = pot
    state_bets = list(bets)
    for step_i, (acting, kind, frac, _ppot, _pbets) in enumerate(actions_to_take):
        # Compute new pot/bets after this action
        max_other = max(b for i, b in enumerate(state_bets) if i != acting)
        to_call = max(0, max_other - state_bets[acting])
        if kind == "check":
            new_bet_acting = state_bets[acting]
            new_pot = state_pot
        elif kind == "call":
            new_bet_acting = state_bets[acting] + to_call
            new_pot = state_pot + to_call
        elif kind == "fold":
            new_bet_acting = state_bets[acting]
            new_pot = state_pot
        elif kind == "bet":
            amt = int(frac * state_pot)
            new_bet_acting = state_bets[acting] + amt
            new_pot = state_pot + amt
        elif kind == "raise":
            amt = to_call + int(frac * (state_pot + to_call))
            new_bet_acting = state_bets[acting] + amt
            new_pot = state_pot + amt
        elif kind == "allin":
            amt = effective_stack - state_bets[acting]
            new_bet_acting = effective_stack
            new_pot = state_pot + amt
        else:
            raise ValueError(kind)

        # Create the parent decision node
        parent_node = _make_node(
            _NODE_DECISION, acting, 1, -1,
            state_pot, list(state_bets),
        )
        chain_node_indices.append(len(nodes))
        nodes.append(parent_node)

        # Update state for child
        state_bets[acting] = new_bet_acting
        state_pot = new_pot

    # Now build the cont-chain root: a DECISION with num_children=4
    # (player=0). Each child is also a DECISION with num_children=4
    # (player=1). Each grandchild is a LEAF.
    cont_root_idx = len(nodes)
    cont_root_first_child = -1  # patch later
    cont_root = _make_node(
        _NODE_DECISION, 0, 4, -1, state_pot, list(state_bets),
    )
    nodes.append(cont_root)

    # Layer-1 nodes (player=1, 4 children of cont_root)
    layer1_indices = []
    for c0 in range(4):
        layer1 = _make_node(
            _NODE_DECISION, 1, 4, -1, state_pot, list(state_bets),
        )
        layer1_indices.append(len(nodes))
        nodes.append(layer1)

    # Layer-2 LEAF nodes (16 total, leaf_idx = 0..15)
    leaf_indices = []
    for c0 in range(4):
        for c1 in range(4):
            leaf_node = _make_node(
                _NODE_LEAF, -1, 0, 0, state_pot, list(state_bets),
                leaf_idx=c0 * 4 + c1,
            )
            leaf_indices.append(len(nodes))
            nodes.append(leaf_node)

    # Now build children[] array. Order: chain transitions, then
    # cont_root's 4 children, then each layer1's 4 children.
    #
    # We patch first_child on each node as we lay out the children array.

    # Chain transitions: each chain[i].first_child points to a children[]
    # entry that holds chain[i+1] index (or cont_root_idx for the last).
    children_arr = []
    for i, ci in enumerate(chain_node_indices):
        nodes[ci].first_child = len(children_arr)
        if i + 1 < len(chain_node_indices):
            children_arr.append(chain_node_indices[i + 1])
        else:
            children_arr.append(cont_root_idx)

    # Cont root's 4 children → layer1
    nodes[cont_root_idx].first_child = len(children_arr)
    for li in layer1_indices:
        children_arr.append(li)

    # Each layer1 node's 4 children → 4 leaves
    for c0, l1_idx in enumerate(layer1_indices):
        nodes[l1_idx].first_child = len(children_arr)
        for c1 in range(4):
            children_arr.append(leaf_indices[c0 * 4 + c1])

    # Build SSTreeData
    tree = SSTreeData()
    NodeArrayT = SSNode * len(nodes)
    nodes_arr = NodeArrayT(*nodes)
    tree.nodes = ctypes.cast(nodes_arr, ctypes.POINTER(SSNode))
    tree.num_nodes = len(nodes)

    ChildArrayT = ctypes.c_int * len(children_arr)
    cs = ChildArrayT(*children_arr)
    tree.children = ctypes.cast(cs, ctypes.POINTER(ctypes.c_int))
    tree.num_children_total = len(children_arr)

    tree.num_players_field = 2
    tree.starting_pot = pot
    tree.effective_stack = effective_stack
    tree.is_river = 0

    # Keep the underlying ctypes arrays alive for the test's lifetime.
    tree._keepalive = (nodes_arr, cs)
    return tree


# ── DFS extraction integration (vocab-parametric) ───────────────────────
#
# These build a synthetic SS tree whose bet sizes are drawn from the vocab
# under test, then assert extract_leaf_info_from_tree — threaded with THAT
# vocab — reconstructs the training-encoded flop_history_train.


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_extract_leaf_info_check_check(first_fracs, subseq_fracs):
    # 2p flop, both check around → cont-chain root reached via [check, check]
    tree = _build_2p_synthetic_tree([
        (0, "check", 0.0, 200, (0, 0)),
        (1, "check", 0.0, 200, (0, 0)),
    ])
    leaves = extract_leaf_info_from_tree(
        tree, num_players=2,
        postflop_bet_sizes=first_fracs, subsequent_bet_sizes=subseq_fracs,
    )
    assert len(leaves) == 1
    li = leaves[0]
    # !facing + check both → train_idx 0 each (vocab-independent).
    assert li.flop_history_train == (0, 0)
    assert li.bets == (0, 0)
    # flop_actor_train parallels flop_history_train: P0 acted, then P1.
    assert li.flop_actor_train == (0, 1)


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_extract_leaf_info_bet_call(first_fracs, subseq_fracs):
    # 2p flop: P0 bets the FIRST trained size (!facing → idx 1), P1 calls
    # (facing → idx 1). Use the vocab's own first size so the bet keys to
    # the trained slot rather than a fuzzy neighbor.
    bet_frac = first_fracs[0]
    tree = _build_2p_synthetic_tree([
        (0, "bet", bet_frac, 200, (0, 0)),
        (1, "call", 0.0, 200, (0, 0)),
    ])
    leaves = extract_leaf_info_from_tree(
        tree, num_players=2,
        postflop_bet_sizes=first_fracs, subsequent_bet_sizes=subseq_fracs,
    )
    assert len(leaves) == 1
    li = leaves[0]
    # bet first-size → !facing idx 1 (1 + slot 0); call → facing idx 1.
    assert li.flop_history_train == (1, 1)
    # The BET-call leaf line: P0 (the bettor), then P1 (the caller).
    assert li.flop_actor_train == (0, 1)


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_extract_leaf_info_bet_raise_call(first_fracs, subseq_fracs):
    # 2p flop: P0 bet first-size (raises_so_far=0, !facing → idx 1),
    # P1 raise subseq-size (raises_so_far=1, facing+SUBSEQUENT → idx 2,
    # since training increments raises_so_far on the opening bet too),
    # P0 call (facing subsequent → idx 1).
    bet_frac = first_fracs[0]
    raise_frac = subseq_fracs[0]
    tree = _build_2p_synthetic_tree([
        (0, "bet", bet_frac, 200, (0, 0)),
        (1, "raise", raise_frac, 200, (0, 0)),
        (0, "call", 0.0, 200, (0, 0)),
    ])
    leaves = extract_leaf_info_from_tree(
        tree, num_players=2,
        postflop_bet_sizes=first_fracs, subsequent_bet_sizes=subseq_fracs,
    )
    assert len(leaves) == 1
    li = leaves[0]
    # bet@idx1, subsequent-raise first-size@idx2, call@idx1.
    assert li.flop_history_train == (1, 2, 1)
    # P0 bet, P1 raise, P0 call.
    assert li.flop_actor_train == (0, 1, 0)


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_extract_leaf_info_check_bet_call(first_fracs, subseq_fracs):
    # 2p flop: P0 check (!facing → 0), P1 bet LAST trained first-size
    # (!facing → 1 + last slot), P0 call (facing → 1).
    bet_frac = first_fracs[-1]
    tree = _build_2p_synthetic_tree([
        (0, "check", 0.0, 200, (0, 0)),
        (1, "bet", bet_frac, 200, (0, 0)),
        (0, "call", 0.0, 200, (0, 0)),
    ])
    leaves = extract_leaf_info_from_tree(
        tree, num_players=2,
        postflop_bet_sizes=first_fracs, subsequent_bet_sizes=subseq_fracs,
    )
    assert len(leaves) == 1
    li = leaves[0]
    bet_idx = 1 + (len(first_fracs) - 1)  # last first-size slot
    assert li.flop_history_train == (0, bet_idx, 1)
    # P0 check, P1 bet, P0 call.
    assert li.flop_actor_train == (0, 1, 0)


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_extract_leaf_info_empty_root_is_cont_chain(first_fracs, subseq_fracs):
    # 2p flop: cont-chain root IS the SS tree root (no flop betting yet)
    tree = _build_2p_synthetic_tree([])
    leaves = extract_leaf_info_from_tree(
        tree, num_players=2,
        postflop_bet_sizes=first_fracs, subsequent_bet_sizes=subseq_fracs,
    )
    assert len(leaves) == 1
    li = leaves[0]
    assert li.flop_history_train == ()
    assert li.flop_actor_train == ()


# ── SAMPLED-mode tree shape ─────────────────────────────────────────────
#
# SS trees compile with SS_CONT_STRAT_MODE_SAMPLED by default. In that
# mode each betting-tree terminal stays as a single SS_NODE_LEAF — no
# cont-chain wrapping. extract_leaf_info_from_tree must emit one LeafInfo
# per such terminal (an extractor that only matches the JOINT cont-chain
# pattern returns [] for SAMPLED trees → rollout feeds CFR all-zero leaf
# values → uniform 1/n at every postflop spot).


def _build_2p_sampled_tree(bet_frac, effective_stack=10000):
    """3-leaf SAMPLED-mode tree, parametrized on the opening bet fraction:
        root P0 acts !facing → {check, bet bet_frac}
          check → P1 acts !facing → {check (LEAF 0), bet bet_frac}
                                         bet → P0 facing → {fold, call (LEAF 1)}
          bet → P1 facing → {fold, call (LEAF 2)}

    Expected histories (per _train_action_idx, bet_frac = first_fracs[0]):
        leaf 0: (0, 0)         check, check
        leaf 1: (0, 1, 1)      check, bet first-size, call
        leaf 2: (1, 1)         bet first-size, call
    """
    nodes = []
    children = []

    # Index plan (built bottom-up so first_child can be patched as we go).
    # 0: root DECISION p=0, 2 children {check_branch, bet_branch}
    # check_branch:
    #   1: DECISION p=1, 2 children {LEAF leaf_idx=0, bet_branch_2}
    #   2: LEAF leaf_idx=0 (P0 check, P1 check)
    #   3: DECISION p=0 facing (after P0 check, P1 bet), 2 children
    #   4: FOLD
    #   5: LEAF leaf_idx=1 (P0 check, P1 bet, P0 call)
    # bet_branch:
    #   6: DECISION p=1 facing (after P0 bet), 2 children
    #   7: FOLD
    #   8: LEAF leaf_idx=2 (P0 bet, P1 call)
    bet_amt = int(bet_frac * 200)

    # 0: root
    nodes.append(_make_node(_NODE_DECISION, 0, 2, -1, 200, (0, 0)))
    # 1: P1 acts after P0 check
    nodes.append(_make_node(_NODE_DECISION, 1, 2, -1, 200, (0, 0)))
    # 2: LEAF — both check
    nodes.append(_make_node(_NODE_LEAF, -1, 0, 0, 200, (0, 0), leaf_idx=0))
    # 3: P0 facing after [check, bet]
    nodes.append(_make_node(
        _NODE_DECISION, 0, 2, -1, 200 + bet_amt, (0, bet_amt),
    ))
    # 4: FOLD child of 3
    nodes.append(_make_node(
        _NODE_FOLD, -1, 0, 0, 200 + bet_amt, (0, bet_amt),
    ))
    # 5: LEAF after [check, bet, call] — P0 calls bet_amt
    nodes.append(_make_node(
        _NODE_LEAF, -1, 0, 0, 200 + 2 * bet_amt, (bet_amt, bet_amt),
        leaf_idx=1,
    ))
    # 6: P1 facing after [bet]
    nodes.append(_make_node(
        _NODE_DECISION, 1, 2, -1, 200 + bet_amt, (bet_amt, 0),
    ))
    # 7: FOLD child of 6
    nodes.append(_make_node(
        _NODE_FOLD, -1, 0, 0, 200 + bet_amt, (bet_amt, 0),
    ))
    # 8: LEAF after [bet, call]
    nodes.append(_make_node(
        _NODE_LEAF, -1, 0, 0, 200 + 2 * bet_amt, (bet_amt, bet_amt),
        leaf_idx=2,
    ))

    # children[] layout — patch first_child in order:
    # node 0 children: [1, 6]
    nodes[0].first_child = len(children); children.extend([1, 6])
    # node 1 children: [2, 3]
    nodes[1].first_child = len(children); children.extend([2, 3])
    # node 3 children: [4, 5]
    nodes[3].first_child = len(children); children.extend([4, 5])
    # node 6 children: [7, 8]
    nodes[6].first_child = len(children); children.extend([7, 8])

    tree = SSTreeData()
    NodeArrayT = SSNode * len(nodes)
    nodes_arr = NodeArrayT(*nodes)
    tree.nodes = ctypes.cast(nodes_arr, ctypes.POINTER(SSNode))
    tree.num_nodes = len(nodes)

    ChildArrayT = ctypes.c_int * len(children)
    cs = ChildArrayT(*children)
    tree.children = ctypes.cast(cs, ctypes.POINTER(ctypes.c_int))
    tree.num_children_total = len(children)

    tree.num_players_field = 2
    tree.starting_pot = 200
    tree.effective_stack = effective_stack
    tree.is_river = 0
    tree._keepalive = (nodes_arr, cs)
    return tree


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_extract_leaf_info_sampled_mode_emits_per_leaf(first_fracs, subseq_fracs):
    """Default-shape SAMPLED tree: 3 flat SS_NODE_LEAF terminals,
    no cont-chain wrapping. Each must yield one LeafInfo with
    flop_history_train captured from the path."""
    bet_frac = first_fracs[0]
    bet_amt = int(bet_frac * 200)
    tree = _build_2p_sampled_tree(bet_frac)
    leaves = extract_leaf_info_from_tree(
        tree, num_players=2,
        postflop_bet_sizes=first_fracs, subsequent_bet_sizes=subseq_fracs,
    )

    by_idx = {li.leaf_idx: li for li in leaves}
    assert set(by_idx) == {0, 1, 2}, \
        f"expected leaves 0,1,2 (one per SS_NODE_LEAF); got {sorted(by_idx)}"

    # check-check (vocab-independent)
    assert by_idx[0].flop_history_train == (0, 0)
    assert by_idx[0].bets == (0, 0)
    # check, bet first-size, call → (0, 1, 1)
    assert by_idx[1].flop_history_train == (0, 1, 1)
    assert by_idx[1].bets == (bet_amt, bet_amt)
    # bet first-size, call → (1, 1)
    assert by_idx[2].flop_history_train == (1, 1)
    assert by_idx[2].bets == (bet_amt, bet_amt)


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_extract_leaf_info_sampled_no_folds_no_phantom_leaves(
        first_fracs, subseq_fracs):
    """FOLD nodes (type=1) must not produce phantom LeafInfo entries —
    only SS_NODE_LEAF (type=4) terminals carry leaf_idx."""
    tree = _build_2p_sampled_tree(first_fracs[0])
    leaves = extract_leaf_info_from_tree(
        tree, num_players=2,
        postflop_bet_sizes=first_fracs, subsequent_bet_sizes=subseq_fracs,
    )
    # Tree has 3 SS_NODE_LEAF + 2 SS_NODE_FOLD; only the 3 LEAFs emit.
    assert len(leaves) == 3


@pytest.mark.parametrize("first_fracs,subseq_fracs", VOCABS)
def test_extract_leaf_info_allin(first_fracs, subseq_fracs):
    # 2p flop: P0 bets allin → child.bets[0] = effective_stack (10000).
    # The non-facing all-in index MOVES with the vocab length — this is
    # exactly the vocab desync this file guards: SERVED [0.75] → idx 2,
    # LEGACY [0.5,1.0] → idx 3.
    tree = _build_2p_synthetic_tree([
        (0, "allin", 0.0, 200, (0, 0)),
    ], effective_stack=10000)
    leaves = extract_leaf_info_from_tree(
        tree, num_players=2,
        postflop_bet_sizes=first_fracs, subsequent_bet_sizes=subseq_fracs,
    )
    assert len(leaves) == 1
    li = leaves[0]
    expected_allin = 1 + len(first_fracs)  # !facing all-in = 1 + n
    assert li.flop_history_train == (expected_allin,)


def test_extract_leaf_info_allin_regression_served_vs_legacy():
    """Explicit regression pin of the all-in index divergence at the
    extractor level: the SAME non-facing all-in tree yields idx 2 under the
    published [0.75] vocab and idx 3 under the legacy [0.5,1.0] vocab. An
    extractor that always emits 3 mis-keys every flop all-in line under the
    [0.75] vocab."""
    tree = _build_2p_synthetic_tree([
        (0, "allin", 0.0, 200, (0, 0)),
    ], effective_stack=10000)

    served = extract_leaf_info_from_tree(
        tree, num_players=2,
        postflop_bet_sizes=SERVED_FIRST, subsequent_bet_sizes=SERVED_SUBSEQ,
    )
    assert served[0].flop_history_train == (2,)

    legacy = extract_leaf_info_from_tree(
        tree, num_players=2,
        postflop_bet_sizes=LEGACY_FIRST, subsequent_bet_sizes=LEGACY_SUBSEQ,
    )
    assert legacy[0].flop_history_train == (3,)

    # Vocab omitted → legacy default (byte-identity).
    default = extract_leaf_info_from_tree(tree, num_players=2)
    assert default[0].flop_history_train == (3,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
