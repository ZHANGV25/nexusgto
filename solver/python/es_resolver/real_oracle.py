"""Real blueprint-rollout leaf oracle for the ES solver. Wraps
rollout.compute_leaf_value as a LAZY single-(leaf, traverser-hand)
callback (the same leaf-value path the search uses), called only when an ES
traversal reaches a betting leaf.

This is the piece that makes the ES solver produce a REAL strategy for a REAL
spot against the loaded blueprint (vs the synthetic strength/random oracles used
for mechanics validation).

Per-leaf state (pot, bets, flop_history_train) comes from
leaf_values.extract_leaf_info_from_tree on the SAME SSTreeData the ES traverses.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.rollout import compute_leaf_value, RolloutLeaf, NUM_ROLLOUTS_DEFAULT, NUM_BIAS_PROFILES  # noqa: E402
from engine.leaf_values import extract_leaf_info_from_tree  # noqa: E402


def build_real_leaf_oracle(
    ss_tree,                       # the ctypes SSTreeData (live; needed for leaf extraction)
    plain_tree,                    # the PlainTree snapshot (for player_hands/board)
    blueprint,                     # BPSReaderC / BlueprintV2
    street: str,                   # "flop" | "turn"
    player_seats: List[int],       # compact slot -> blueprint full seat (0-5)
    stack_chips: int,
    num_rollouts: int = NUM_ROLLOUTS_DEFAULT,
    num_samples_k: int = 1,
    seed: int = 0,
) -> Tuple[Callable, Dict[str, float]]:
    """Return (leaf_rollout_fn, stats) where leaf_rollout_fn matches the ES
    solver's leaf_rollout_fn(leaf_idx, traverser, hand, sampled_hands, leaf_node)
    signature and computes a LAZY rollout value via compute_leaf_value.

    The cont-strat bias profile is sampled per (leaf, call) — Pluribus §4 treats
    the cont-strat choice as another action selected via MCCFR (external
    sampling draws one realization per visit). The oracle draws num_samples_k
    joint profiles and averages them (K=1 = plain ES).

    stats: collects timing — total rollout wall seconds + count, so the caller
    can report the REAL per-rollout wall cost (measured here, not assumed).
    """
    NP = plain_tree.num_players
    board = list(plain_tree.board)

    # Per-leaf bookkeeping from the search's leaf extractor: leaf_idx -> LeafInfo
    leaf_infos = extract_leaf_info_from_tree(ss_tree, num_players=NP)
    by_idx = {li.leaf_idx: li for li in leaf_infos}

    rng = np.random.default_rng(seed)
    stats = {"rollout_s": 0.0, "rollout_calls": 0}

    # acting order in compact slots (postflop): the solver tree uses 0..NP-1 in
    # acting order already (root player = slot 0). Identity is passed; the rollout
    # recomputes per-street acting order internally.
    acting_order = list(range(NP))

    def leaf_rollout_fn(leaf_idx, traverser, hand, sampled_hands, leaf_node):
        li = by_idx.get(leaf_idx)
        if li is None:
            return 0.0
        leaf_bets = list(li.bets) if hasattr(li, "bets") else [0] * NP
        stacks = [max(0, stack_chips - b) for b in leaf_bets]
        # active: a player is active at the leaf iff still in (the solver tree's
        # leaf nodes are reached only on no-fold betting lines; node.active marks
        # who is in). Use the snapshotted leaf_node.active.
        active = [bool(leaf_node.active[p]) for p in range(NP)]

        leaf = RolloutLeaf(
            board=board,
            pot_chips=int(li.pot),
            active=active,
            stacks=stacks,
            bets=leaf_bets,
            street=street,
            player_hands=plain_tree.player_hands,
            acting_order=acting_order,
            game_action_history=list(li.flop_history_train),
            player_seats=player_seats,
        )
        # draw K joint cont-strat bias profiles, average the traverser's value
        t0 = time.time()
        acc = 0.0
        for _ in range(num_samples_k):
            bias = rng.integers(0, NUM_BIAS_PROFILES, size=NP).tolist()
            acc += compute_leaf_value(
                leaf=leaf, bias_profile=bias, blueprint=blueprint,
                hero_player=traverser, hero_hand_idx=hand,
                num_rollouts=num_rollouts, rng=rng,
            )
        stats["rollout_s"] += time.time() - t0
        stats["rollout_calls"] += 1
        return acc / num_samples_k

    return leaf_rollout_fn, stats
