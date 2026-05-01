"""Validates the per-leaf flop_history extraction + threading.

Local part: synthetic 2p flop tree built directly via SSNode/SSTreeData
ctypes structs (no CUDA needed). Walk it, verify the reconstructed
flop_history matches what each leaf was reached through. Verifies
_train_action_idx + the DFS traversal path.

End-to-end (rollout miss rate against v3 blueprint) lives in
tools/diagnose_v3_rollout.py and runs only on EC2.
"""
from __future__ import annotations

import ctypes
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from street_solver_gpu import (  # type: ignore[import-untyped]
    SSNode,
    SSTreeData,
    SS_MAX_PLAYERS,
)
from leaf_values import (  # type: ignore[import-untyped]
    LeafInfo,
    _train_action_idx,
    extract_leaf_info_from_tree,
)


# ── Action-index re-encoder direct tests ────────────────────────────────


def test_train_action_idx_first_raise_facing():
    # facing + first raise: [fold(0), call(1), 0.5(2), 1.0(3), allin(4)]
    assert _train_action_idx(True, 0, "fold", 0.0) == 0
    assert _train_action_idx(True, 0, "call", 0.0) == 1
    assert _train_action_idx(True, 0, "raise", 0.5) == 2
    assert _train_action_idx(True, 0, "raise", 1.0) == 3
    assert _train_action_idx(True, 0, "allin", 1.0) == 4
    # Fuzzy: 0.6 closer to 0.5 than 1.0
    assert _train_action_idx(True, 0, "raise", 0.6) == 2
    assert _train_action_idx(True, 0, "raise", 0.8) == 3


def test_train_action_idx_subsequent_facing():
    # facing + subsequent: [fold(0), call(1), 1.0(2), allin(3)]
    assert _train_action_idx(True, 1, "fold", 0.0) == 0
    assert _train_action_idx(True, 1, "call", 0.0) == 1
    assert _train_action_idx(True, 1, "raise", 1.0) == 2
    # Lossy: subgame's 0.5 has no training counterpart at subsequent → maps to 1.0
    assert _train_action_idx(True, 1, "raise", 0.5) == 2
    assert _train_action_idx(True, 1, "allin", 1.0) == 3


def test_train_action_idx_no_facing():
    # !facing: [check(0), 0.5(1), 1.0(2), allin(3)]
    assert _train_action_idx(False, 0, "check", 0.0) == 0
    assert _train_action_idx(False, 0, "bet", 0.5) == 1
    assert _train_action_idx(False, 0, "bet", 1.0) == 2
    assert _train_action_idx(False, 0, "allin", 1.0) == 3


def test_train_action_idx_tree_closed():
    # tree_closed (raises_so_far >= 3) + facing: [fold(0), call(1)]
    assert _train_action_idx(True, 3, "fold", 0.0) == 0
    assert _train_action_idx(True, 3, "call", 0.0) == 1
    # tree_closed + !facing: [check(0)]
    assert _train_action_idx(False, 3, "check", 0.0) == 0


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
# Build a hand-rolled SSTreeData with raw nodes/children arrays. We can't
# call ss_build_tree without the CUDA path; instead, we set up the small
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


# ── DFS extraction integration ──────────────────────────────────────────


def test_extract_leaf_info_check_check():
    # 2p flop, both check around → cont-chain root reached via [check, check]
    tree = _build_2p_synthetic_tree([
        (0, "check", 0.0, 200, (0, 0)),
        (1, "check", 0.0, 200, (0, 0)),
    ])
    leaves = extract_leaf_info_from_tree(tree, num_players=2)
    assert len(leaves) == 1
    li = leaves[0]
    # !facing + check both → train_idx 0 each
    assert li.flop_history_train == (0, 0)
    assert li.bets == (0, 0)


def test_extract_leaf_info_bet_call():
    # 2p flop: P0 bet 0.5×pot (idx 1 in !facing), P1 calls (idx 1 in facing)
    tree = _build_2p_synthetic_tree([
        (0, "bet", 0.5, 200, (0, 0)),
        (1, "call", 0.0, 200, (0, 0)),
    ])
    leaves = extract_leaf_info_from_tree(tree, num_players=2)
    assert len(leaves) == 1
    li = leaves[0]
    # P0 bets 0.5 (no facing → idx 1), P1 calls (facing → idx 1)
    assert li.flop_history_train == (1, 1)


def test_extract_leaf_info_bet_raise_call():
    # 2p flop: P0 bet 0.5 (raises_so_far=0, !facing → idx 1),
    # P1 raise 1.0 (raises_so_far=1, facing+SUBSEQUENT → idx 2,
    # since training increments raises_so_far on the opening bet too),
    # P0 call (facing subsequent → idx 1).
    tree = _build_2p_synthetic_tree([
        (0, "bet", 0.5, 200, (0, 0)),
        (1, "raise", 1.0, 200, (0, 0)),
        (0, "call", 0.0, 200, (0, 0)),
    ])
    leaves = extract_leaf_info_from_tree(tree, num_players=2)
    assert len(leaves) == 1
    li = leaves[0]
    assert li.flop_history_train == (1, 2, 1)


def test_extract_leaf_info_check_bet_call():
    # 2p flop: P0 check (no-facing 0), P1 bet 1.0 (no-facing 2), P0 call (facing 1)
    tree = _build_2p_synthetic_tree([
        (0, "check", 0.0, 200, (0, 0)),
        (1, "bet", 1.0, 200, (0, 0)),
        (0, "call", 0.0, 200, (0, 0)),
    ])
    leaves = extract_leaf_info_from_tree(tree, num_players=2)
    assert len(leaves) == 1
    li = leaves[0]
    assert li.flop_history_train == (0, 2, 1)


def test_extract_leaf_info_empty_root_is_cont_chain():
    # 2p flop: cont-chain root IS the SS tree root (no flop betting yet)
    tree = _build_2p_synthetic_tree([])
    leaves = extract_leaf_info_from_tree(tree, num_players=2)
    assert len(leaves) == 1
    li = leaves[0]
    assert li.flop_history_train == ()


def test_extract_leaf_info_allin():
    # 2p flop: P0 bets allin → child.bets[0] = effective_stack (10000)
    tree = _build_2p_synthetic_tree([
        (0, "allin", 0.0, 200, (0, 0)),
    ], effective_stack=10000)
    # The cont-chain root is reached after P0's allin AND P1's call (which
    # we didn't model). We only modeled P0's action; the cont-chain
    # immediately follows in this test setup. Verify the allin idx.
    leaves = extract_leaf_info_from_tree(tree, num_players=2)
    assert len(leaves) == 1
    li = leaves[0]
    # !facing + allin → train_idx 3
    assert li.flop_history_train == (3,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
