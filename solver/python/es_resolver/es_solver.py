"""External-sampling MCCFR re-solver over the identical SSTreeData subgame tree.

This is the correct-but-slow Python reference (guardrail #3: T is algorithmic /
language-independent — measure it here, scale per-traversal cost from the C
trainer's known throughput). It consumes the EXACT tree StreetSolverGPU builds
(ss_build_tree2), so there is no structural-mismatch risk vs the production
solve (advisor concern #3, resolved by construction).

DESIGN (locked with advisor 2026-06-01):
  - LAZY leaf evaluation (Option 2): no eager precompute floor. A leaf's value
    is computed only when a traversal reaches it, via a single sampled-opponent
    rollout — an unbiased estimate of the range-marginalized leaf EV the GPU
    precomputes. This is the entire reason ES avoids the 466s/6p construct wall.
  - SAMPLE the traverser's hand (one sampled hand per player per traversal),
    mirroring the C trainer's proven external-sampling structure. Keeps the
    crossover's "85 traversals/s/thread" anchor honest (that anchor measures a
    hand-sampling trainer). Vector-traverser-hands is a later quality
    optimization, not the first build.
  - External sampling: at the TRAVERSER's decision nodes, explore ALL actions
    (branch); at every OTHER player's decision node, sample ONE action from
    their current regret-matched strategy. Update regrets + strategy-sum only
    at traverser nodes. (Pluribus Algorithm 1, TRAVERSE-MCCFR.)
  - Linear CFR weighting on the strategy-sum (Pluribus: σ̄ weighted by iter t).

THE INVARIANT WE MEASURE (advisor): total LEAF-ROLLOUTS-to-converge, not
"traversals". rollouts_per_traversal is instrumented empirically (guardrail #2 —
never eyeball the nested term).

Node type codes (street_solve.cuh): 0=DECISION 1=FOLD 2=SHOWDOWN 4=LEAF.
Leaf value layout: leaf_values[leaf_idx*NP*max_h + player*max_h + hand].
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Node type codes
T_DECISION = 0
T_FOLD = 1
T_SHOWDOWN = 2
T_LEAF = 4

REGRET_FLOOR = -1e12  # generous; Pluribus uses -310M on int32. We use float.


@dataclass
class PlainNode:
    """A plain-Python copy of one SSNode (decouples from ctypes lifetime)."""
    idx: int
    type: int
    player: int
    num_children: int
    children: Tuple[int, ...]      # resolved child node indices
    pot: int
    bets: Tuple[int, ...]          # per-player chips in this round (chips)
    active: Tuple[bool, ...]       # active_players[]
    num_board: int
    leaf_idx: int
    fold_player: int
    num_players: int


@dataclass
class PlainTree:
    """Plain-Python snapshot of the SSTreeData structure we traverse."""
    nodes: List[PlainNode]
    num_players: int
    num_board: int
    board: Tuple[int, ...]
    starting_pot: int
    effective_stack: int
    # per-player hand list: (c0, c1, weight)
    player_hands: List[List[Tuple[int, int, float]]]
    num_hands: List[int]
    max_hands: int
    decision_node_indices: List[int]


def snapshot_tree(ss_tree, player_hands: List[List[Tuple[int, int, float]]]) -> PlainTree:
    """Copy an SSTreeData (ctypes) into a PlainTree (pure Python).

    player_hands: the SAME per-player (c0,c1,w) lists passed to StreetSolverGPU,
    in solver player order. We copy them rather than re-read from ss_tree.hands
    to keep weights exact.
    """
    NP = ss_tree.num_players_field
    nodes: List[PlainNode] = []
    for i in range(ss_tree.num_nodes):
        nd = ss_tree.nodes[i]
        kids = tuple(
            ss_tree.children[nd.first_child + c] for c in range(nd.num_children)
        )
        nodes.append(PlainNode(
            idx=i,
            type=nd.type,
            player=nd.player,
            num_children=nd.num_children,
            children=kids,
            pot=nd.pot,
            bets=tuple(nd.bets[p] for p in range(NP)),
            active=tuple(bool(nd.active_players[p]) for p in range(NP)),
            num_board=nd.num_board,
            leaf_idx=nd.leaf_idx,
            fold_player=nd.fold_player,
            num_players=NP,
        ))
    max_h = max((len(h) for h in player_hands), default=0)
    return PlainTree(
        nodes=nodes,
        num_players=NP,
        num_board=ss_tree.num_board,
        board=tuple(ss_tree.board[i] for i in range(ss_tree.num_board)),
        starting_pot=ss_tree.starting_pot,
        effective_stack=ss_tree.effective_stack,
        player_hands=player_hands,
        num_hands=[len(h) for h in player_hands],
        max_hands=max_h,
        decision_node_indices=[
            ss_tree.decision_node_indices[i]
            for i in range(ss_tree.num_decision_nodes)
        ],
    )


# ─── Regret / strategy storage, keyed by (node_idx, hand_idx) ────────────────
# An info set in this single-street subgame is (decision_node, traverser_hand).
# The decision node already encodes the board (abstracted) + action history; the
# traverser's hand distinguishes the private state. This mirrors the GPU's
# per-(node,hand) strategy table exactly.


@dataclass
class InfoSets:
    """Regret + strategy-sum tables, lazily allocated per (node_idx, hand)."""
    # regrets[node_idx] -> np.ndarray[num_hands_of_that_player, num_actions]
    regrets: Dict[int, np.ndarray] = field(default_factory=dict)
    strat_sum: Dict[int, np.ndarray] = field(default_factory=dict)
    # final-iteration strategy snapshot (Pluribus play-path) filled on demand
    last_strategy: Dict[int, np.ndarray] = field(default_factory=dict)

    def ensure(self, node_idx: int, num_hands: int, num_actions: int) -> None:
        if node_idx not in self.regrets:
            self.regrets[node_idx] = np.zeros((num_hands, num_actions), dtype=np.float64)
            self.strat_sum[node_idx] = np.zeros((num_hands, num_actions), dtype=np.float64)

    def strategy(self, node_idx: int, hand: int, num_actions: int) -> np.ndarray:
        """Regret-matching: σ(a) ∝ max(0, R(a)); uniform if all ≤ 0."""
        r = self.regrets[node_idx][hand]
        pos = np.maximum(r, 0.0)
        s = pos.sum()
        if s > 0:
            return pos / s
        return np.full(num_actions, 1.0 / num_actions, dtype=np.float64)


@dataclass
class Stats:
    traversals: int = 0
    leaf_rollouts: int = 0       # THE invariant: total leaf-rollouts (advisor)
    fold_terminals: int = 0
    showdown_terminals: int = 0
    decision_visits: int = 0


class ESSolver:
    """External-sampling MCCFR over a PlainTree subgame.

    leaf_rollout_fn(leaf_idx, traverser, hand, sampled_hands, leaf_node) -> float
        Returns the hero's net chip payoff at the betting leaf for the given
        traverser hand, with opponents holding `sampled_hands`. LAZY — called
        only when a traversal reaches the leaf. The caller supplies this (it
        wraps the existing rollout machinery). For the first validated build we
        also support a deterministic table oracle (see leaf_oracle_from_array).
    """

    def __init__(
        self,
        tree: PlainTree,
        leaf_rollout_fn: Callable[..., float],
        seed: int = 0,
        linear_cfr: bool = True,
    ) -> None:
        self.tree = tree
        self.leaf_rollout_fn = leaf_rollout_fn
        self.info = InfoSets()
        self.rng = np.random.default_rng(seed)
        self.linear_cfr = linear_cfr
        self.stats = Stats()
        self._iter = 0
        # Precompute, per player, the index of each hand for fast lookup.
        self._np = tree.num_players

    # ── hand sampling ───────────────────────────────────────────────────────
    def _sample_hands(self) -> List[int]:
        """Sample one hand index per player from their reach-weighted range,
        avoiding card conflicts (board + other players' sampled cards).

        Returns per-player hand index (into player_hands[p]); -1 if a player
        couldn't be assigned a conflict-free hand (rare; that traversal is
        skipped by the caller).
        """
        used = set(self.tree.board)
        chosen = [-1] * self._np
        order = list(range(self._np))
        # randomize assignment order so no player is systematically starved
        self.rng.shuffle(order)
        for p in order:
            hands = self.tree.player_hands[p]
            cand_idx = []
            cand_w = []
            for i, (c0, c1, w) in enumerate(hands):
                if w <= 0.0:
                    continue
                if c0 in used or c1 in used:
                    continue
                cand_idx.append(i)
                cand_w.append(w)
            if not cand_idx:
                chosen[p] = -1
                continue
            wv = np.asarray(cand_w, dtype=np.float64)
            wv /= wv.sum()
            pick = cand_idx[int(self.rng.choice(len(cand_idx), p=wv))]
            chosen[p] = pick
            c0, c1, _ = hands[pick]
            used.add(c0)
            used.add(c1)
        return chosen

    # ── terminal values ─────────────────────────────────────────────────────
    def _fold_value(self, node: PlainNode, traverser: int, sampled: List[int]) -> float:
        """Value at a FOLD terminal for the traverser's sampled hand.

        A fold node = exactly one player folded relative to its parent; the
        hand continues only if >1 active remain (the tree only emits a FOLD
        *terminal* when the fold leaves a single winner — otherwise play
        continues at a DECISION child). We mirror the GPU: at a fold terminal
        the last remaining active player wins the pot; everyone else loses
        their invested chips this round.

        net = (pot - my_bet) if I'm the winner else -my_bet.
        """
        # Winner = the sole remaining active player. node.active reflects who is
        # still in AFTER the fold. node.player is the "last remaining" (per
        # street_solve.cu:747 `winner = node->player`).
        winner = node.player
        my_bet = node.bets[traverser]
        if traverser == winner:
            return float(node.pot) - float(my_bet)
        return -float(my_bet)

    def _showdown_value(self, node: PlainNode, traverser: int, sampled: List[int]) -> float:
        """Value at a SHOWDOWN terminal (river only). Uses sampled hands +
        7-card eval. Single-pot (no side pots) to mirror the rollout
        convention (rollout.py compute_showdown_payoff)."""
        from leaf_values import _eval7_py  # local import; heavy module
        board = list(self.tree.board)
        live = [p for p in range(self._np) if node.active[p]]
        if traverser not in live:
            return -float(node.bets[traverser])
        # board must be 5 cards at a river showdown
        strengths = {}
        for p in live:
            hi = sampled[p]
            if hi < 0:
                return 0.0  # unassignable — skip (caller-handled upstream)
            c0, c1, _ = self.tree.player_hands[p][hi]
            strengths[p] = _eval7_py(board + [c0, c1])
        best = max(strengths.values())
        winners = [p for p in live if strengths[p] == best]
        my_bet = node.bets[traverser]
        if traverser in winners:
            return float(node.pot) / len(winners) - float(my_bet)
        return -float(my_bet)

    def _leaf_value(self, node: PlainNode, traverser: int, sampled: List[int]) -> float:
        """LAZY betting-leaf value via one sampled-opponent rollout."""
        self.stats.leaf_rollouts += 1
        return float(self.leaf_rollout_fn(
            leaf_idx=node.leaf_idx,
            traverser=traverser,
            hand=sampled[traverser],
            sampled_hands=sampled,
            leaf_node=node,
        ))

    # ── traversal ───────────────────────────────────────────────────────────
    def _traverse(self, node_idx: int, traverser: int, sampled: List[int]) -> float:
        node = self.tree.nodes[node_idx]
        t = node.type
        if t == T_FOLD:
            self.stats.fold_terminals += 1
            return self._fold_value(node, traverser, sampled)
        if t == T_SHOWDOWN:
            self.stats.showdown_terminals += 1
            return self._showdown_value(node, traverser, sampled)
        if t == T_LEAF:
            return self._leaf_value(node, traverser, sampled)

        # DECISION node
        self.stats.decision_visits += 1
        acting = node.player
        na = node.num_children
        hand = sampled[acting]
        if hand < 0:
            # acting player has no assignable hand — treat as uniform pass-through
            # (extremely rare; pick child 0 to keep the walk finite)
            return self._traverse(node.children[0], traverser, sampled)

        nh_acting = self.tree.num_hands[acting]
        self.info.ensure(node_idx, nh_acting, na)
        sigma = self.info.strategy(node_idx, hand, na)

        if acting == traverser:
            # Explore ALL actions (external sampling: traverser branches).
            child_vals = np.empty(na, dtype=np.float64)
            for a in range(na):
                child_vals[a] = self._traverse(node.children[a], traverser, sampled)
            node_val = float(np.dot(sigma, child_vals))
            # regret update (counterfactual: opponents already sampled, so the
            # reach weight of "everyone else" is folded into the sampled path —
            # external sampling makes the cf-reach 1 for the traverser).
            regrets = self.info.regrets[node_idx][hand]
            regrets += child_vals - node_val
            np.maximum(regrets, REGRET_FLOOR, out=regrets)
            # Strategy-sum update. Mirrors mccfr_blueprint.c:2036-2037 exactly:
            # plain per-visit accumulation of the CURRENT strategy at the
            # traverser's sampled hand (NO iteration weight, NO explicit reach
            # weight). Under external sampling the traverser's reach to this
            # node is already realized by the sampled path, so per-visit
            # accumulation is the correct average-strategy estimator (Pluribus
            # Algorithm 1 UPDATE-STRATEGY, applied to the subgame's single
            # betting round = the analogue of preflop street 0).
            self.info.strat_sum[node_idx][hand] += sigma
            return node_val
        else:
            # Opponent: sample ONE action from their current strategy.
            a = int(self.rng.choice(na, p=sigma))
            return self._traverse(node.children[a], traverser, sampled)

    def iterate(self) -> None:
        """One ES iteration: for each player as traverser, sample hands + walk."""
        self._iter += 1
        for traverser in range(self._np):
            sampled = self._sample_hands()
            if sampled[traverser] < 0:
                continue  # traverser unassignable this draw; skip
            self.stats.traversals += 1
            self._traverse(0, traverser, sampled)

    def run(self, iterations: int) -> None:
        for _ in range(iterations):
            self.iterate()

    # ── strategy readout ────────────────────────────────────────────────────
    def avg_strategy(self, node_idx: int) -> Optional[np.ndarray]:
        """σ̄ (sigma-bar) at a decision node: normalized strategy_sum per hand.
        Shape [num_hands, num_actions]. None if node never visited."""
        if node_idx not in self.info.strat_sum:
            return None
        ss = self.info.strat_sum[node_idx]
        out = np.zeros_like(ss)
        sums = ss.sum(axis=1, keepdims=True)
        nz = sums[:, 0] > 0
        out[nz] = ss[nz] / sums[nz]
        # hands never updated -> uniform
        na = ss.shape[1]
        out[~nz] = 1.0 / na
        return out

    def final_strategy(self, node_idx: int) -> Optional[np.ndarray]:
        """Final-iteration regret-matched strategy (Pluribus play-path)."""
        if node_idx not in self.info.regrets:
            return None
        r = self.info.regrets[node_idx]
        pos = np.maximum(r, 0.0)
        sums = pos.sum(axis=1, keepdims=True)
        out = np.where(sums > 0, pos / np.maximum(sums, 1e-12),
                       1.0 / r.shape[1])
        return out
