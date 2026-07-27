"""Gap 2B Python binding, vec-coherence, and scalar-gate tests.

Run after building es_resolver_real. The tree is hand-rolled with ctypes, so
no CUDA device, blueprint artifact, or live rollout is required.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# The sampled-oracle regression needs the real target. A harness can override
# this test specifically with GAP2B_ES_DLL.
_PREVIOUS_ES_DLL = os.environ.get("ES_DLL")
os.environ["ES_DLL"] = os.environ.get("GAP2B_ES_DLL", "es_resolver_real")

try:
    try:
        from engine.street_solver_gpu import (  # type: ignore[import-untyped]
            SSNode,
            SSTreeData,
            SS_MAX_PLAYERS,
        )
        from es_resolver.c_solver import CSolver  # type: ignore[import-untyped]
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(
            f"Gap 2B resolver DLL unavailable: {exc}",
            allow_module_level=True,
        )
finally:
    if _PREVIOUS_ES_DLL is None:
        os.environ.pop("ES_DLL", None)
    else:
        os.environ["ES_DLL"] = _PREVIOUS_ES_DLL


def _node(type_: int, player: int, leaf_idx: int = -1) -> SSNode:
    node = SSNode()
    node.type = type_
    node.player = player
    node.leaf_idx = leaf_idx
    node.pot = 100
    node.num_players = 2
    node.fold_player = -1
    for p in range(SS_MAX_PLAYERS):
        node.active_players[p] = 1 if p < 2 else 0
    return node


def _tiny_bias_tree() -> SSTreeData:
    """Root player 0 chooses between two bias-valued leaves."""
    nodes = (SSNode * 3)()
    nodes[0] = _node(0, 0)
    nodes[0].num_children = 2
    nodes[0].first_child = 0
    nodes[1] = _node(4, -1, leaf_idx=0)
    nodes[2] = _node(4, -1, leaf_idx=1)
    children = (ctypes.c_int * 2)(1, 2)
    decisions = (ctypes.c_int * 1)(0)
    leaves = (ctypes.c_int * 2)(1, 2)

    tree = SSTreeData()
    tree.nodes = ctypes.cast(nodes, ctypes.POINTER(SSNode))
    tree.children = ctypes.cast(children, ctypes.POINTER(ctypes.c_int))
    tree.num_nodes = 3
    tree.num_children_total = 2
    tree.num_players_field = 2
    tree.num_leaves = 2
    tree.decision_node_indices = ctypes.cast(
        decisions, ctypes.POINTER(ctypes.c_int))
    tree.num_decision_nodes = 1
    tree.leaf_node_indices = ctypes.cast(
        leaves, ctypes.POINTER(ctypes.c_int))
    tree.num_leaf_nodes = 2

    hands = (
        ((0, 1), (2, 3)),
        ((4, 5), (6, 7)),
    )
    for player in range(2):
        tree.num_hands[player] = 2
        for hand, (c0, c1) in enumerate(hands[player]):
            tree.hands[player][hand][0] = c0
            tree.hands[player][hand][1] = c1
            tree.weights[player][hand] = 1.0

    tree._keepalive = (nodes, children, decisions, leaves)
    return tree


def _dominant_leaf_values() -> np.ndarray:
    values = np.zeros((2, 4, 2, 2), dtype=np.float32)
    for bias in range(4):
        # Player 0 learns dominant bias 3 at both leaves while leaf 1 remains
        # exactly five chips better than leaf 0 for the root action.
        values[0, bias, 0, :] = float(bias)
        values[1, bias, 0, :] = float(5 + bias)
        # Player 1 has the opposite zero-sum sign (and no tree decision).
        values[0, bias, 1, :] = float(-bias)
        values[1, bias, 1, :] = float(-(5 + bias))
    return values


def test_from_bias_table_keepalive_stats_and_effective_table() -> None:
    tree = _tiny_bias_tree()
    values = _dominant_leaf_values()
    solver = CSolver.from_bias_table(
        tree, values, mode=0, np_=2, max_h=2, n_leaves=2, seed=17)
    try:
        assert solver._bias_values_keepalive is not None
        assert solver._bias_values_keepalive.flags.c_contiguous
        solver.set_vec_mode(1)  # deterministic plain callback: accepted
        solver.run(256, num_threads=1)
        visits, errors = solver.bias_oracle_stats()
        assert visits > 0
        assert errors == 0
        effective = solver.get_bias_effective_table()
        assert effective is not None
        assert effective.shape == (2, 2, 2)
        assert effective.dtype == np.float32
        assert np.all(effective[1, 0, :] > 7.9)
    finally:
        solver.__del__()


def test_vec_and_scalar_reach_same_fixed_point_single_thread() -> None:
    tree_scalar = _tiny_bias_tree()
    tree_vec = _tiny_bias_tree()
    values = _dominant_leaf_values()
    scalar = CSolver.from_bias_table(
        tree_scalar, values, mode=0, np_=2, max_h=2,
        n_leaves=2, seed=123)
    vec = CSolver.from_bias_table(
        tree_vec, values, mode=0, np_=2, max_h=2,
        n_leaves=2, seed=123)
    try:
        scalar.set_vec_mode(0)
        vec.set_vec_mode(1)
        scalar.run(4000, num_threads=1)
        vec.run(4000, num_threads=1)
        scalar_sigma = scalar.final_strategy(0)
        vec_sigma = vec.final_strategy(0)
        assert scalar_sigma is not None and vec_sigma is not None
        assert np.allclose(scalar_sigma, vec_sigma, atol=0.02, rtol=0.0)
        assert np.all(scalar_sigma[:, 1] > 0.98)
        assert np.all(vec_sigma[:, 1] > 0.98)
    finally:
        scalar.__del__()
        vec.__del__()


def test_sampled_stays_scalar_only_while_bias_accepts_vec() -> None:
    tree_bias = _tiny_bias_tree()
    bias = CSolver.from_bias_table(
        tree_bias, _dominant_leaf_values(), mode=0, np_=2,
        max_h=2, n_leaves=2, seed=9)
    try:
        bias.set_vec_mode(1)
    finally:
        bias.__del__()

    tree_sampled = _tiny_bias_tree()
    n_leaves = 2
    sampled = CSolver.from_sampled_oracle(
        tree_sampled,
        bp_handle=1,  # Constructor borrows it; guard is tested before any run.
        street=1,
        stack_chips=100,
        n_leaves=n_leaves,
        max_h=2,
        pot=np.full(n_leaves, 100, dtype=np.int32),
        bets=np.zeros(n_leaves * 2, dtype=np.int32),
        active=np.ones(n_leaves * 2, dtype=np.int32),
        hist_flat=np.zeros(n_leaves * 256, dtype=np.int32),
        hist_len=np.zeros(n_leaves, dtype=np.int32),
        player_seats=np.array([0, 1], dtype=np.int32),
        num_rollouts=1,
        bias_policy=0,
        seed=9,
    )
    try:
        with pytest.raises(ValueError, match="scalar-only"):
            sampled.set_vec_mode(1)
    finally:
        sampled.__del__()
