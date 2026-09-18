"""Sampled-card chance subsets for the in-tree flop solve (2026-09-17).

Gates (heads-up flop root, POKER_SEARCH_DEPTH=full, POKER_FLOP_DEPTH_LIMIT=0):
  1. POKER_CHANCE_TURN_CARDS=K: every turn chance node deals exactly K cards
     and every turn chance node deals the SAME K cards.
  2. POKER_CHANCE_RIVER_CARDS=M: every river chance node deals M cards, and
     river chance nodes under the same turn card deal the same M cards.
  3. The tree is small (far below the 873k enumerated shape) and still has
     no leaf nodes (terminals are showdown and fold).
  4. A different POKER_CHANCE_SUBSET_SEED draws a different turn subset.
  5. A turn root under the same env is unchanged (48 river children).
  6. Unset / 0 restores the full deal (49 turn children).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from engine.solver import card_to_int
from engine.street_solver_gpu import StreetSolverGPU

_PASS: List[str] = []
_FAIL: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        _PASS.append(name); print(f"  PASS  {name}")
    else:
        _FAIL.append(name); print(f"  FAIL  {name}")
        if detail: print(f"        {detail}")


def _build(board_strs: List[str], turn_cards: str, river_cards: str, seed: str) -> StreetSolverGPU:
    os.environ["POKER_SEARCH_DEPTH"] = "full"
    os.environ["POKER_FLOP_DEPTH_LIMIT"] = "0"
    for name, value in (("POKER_CHANCE_TURN_CARDS", turn_cards), ("POKER_CHANCE_RIVER_CARDS", river_cards),
                        ("POKER_CHANCE_SUBSET_SEED", seed)):
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)
    board_set = {card_to_int(c) for c in board_strs}
    hands: List[Tuple[int, int, float]] = []
    for c0 in range(52):
        if c0 in board_set:
            continue
        for c1 in range(c0 + 1, 52):
            if c1 not in board_set:
                hands.append((c0, c1, 1.0))
            if len(hands) >= 4:
                break
        if len(hands) >= 4:
            break
    return StreetSolverGPU(board=board_strs, player_ranges=[hands[:2], hands[2:4]], pot_bb=6.5, stack_bb=97.5,
                           bet_sizes=[0.75], use_cont_strats=False)


def _chance_deals(tree) -> Dict[int, Tuple[int, Tuple[int, ...], Tuple[int, ...]]]:
    """chance node index -> (num_board, board cards, dealt cards)."""
    out = {}
    for i in range(tree.num_chance_nodes):
        nidx = tree.chance_node_indices[i]
        node = tree.nodes[nidx]
        board = tuple(int(node.board_cards[b]) for b in range(node.num_board))
        cards = tuple(int(tree.chance_card_per_child[node.chance_card_per_child_offset + c]) for c in range(node.num_children))
        out[nidx] = (int(node.num_board), board, cards)
    return out


def test_subsets():
    s = _build(["Qs", "Kh", "2d"], "4", "3", "7")
    tree = s._tree
    deals = _chance_deals(tree)
    turn = [d for d in deals.values() if d[0] == 3]
    river = [d for d in deals.values() if d[0] == 4]
    check("subset: turn chance nodes deal 4 cards", turn and all(len(d[2]) == 4 for d in turn), str([len(d[2]) for d in turn]))
    check("subset: all turn chance nodes share the subset", len({d[2] for d in turn}) == 1, str({d[2] for d in turn}))
    check("subset: river chance nodes deal 3 cards", river and all(len(d[2]) == 3 for d in river), str(sorted({len(d[2]) for d in river})))
    by_turn_card: Dict[int, set] = {}
    for _nb, board, cards in river:
        by_turn_card.setdefault(board[3], set()).add(cards)
    check("subset: river subsets are shared per turn card", all(len(v) == 1 for v in by_turn_card.values()), str(by_turn_card))
    check("subset: river cards exclude the turn card", all(board[3] not in cards for _nb, board, cards in river))
    check("subset: tree is small", tree.num_nodes < 20000, f"nodes={tree.num_nodes}")
    check("subset: no leaf nodes", tree.num_leaf_nodes == 0, f"leaf={tree.num_leaf_nodes}")
    check("subset: showdown and fold terminals", tree.num_showdown_nodes > 0 and tree.num_fold_nodes > 0)
    first_turn_subset = turn[0][2]

    s2 = _build(["Qs", "Kh", "2d"], "4", "3", "8")
    turn2 = [d for d in _chance_deals(s2._tree).values() if d[0] == 3]
    check("subset: another seed draws another turn subset", turn2[0][2] != first_turn_subset, f"{turn2[0][2]} vs {first_turn_subset}")

    os.environ["POKER_TURN_DEPTH_LIMIT"] = "0"  # the turn root extends only under this flag
    s3 = _build(["Qs", "Kh", "2d", "7h"], "4", "3", "7")
    os.environ.pop("POKER_TURN_DEPTH_LIMIT", None)
    river3 = [d for d in _chance_deals(s3._tree).values() if d[0] == 4]
    check("subset: a turn root still deals every river card", river3 and all(len(d[2]) == 48 for d in river3), str(sorted({len(d[2]) for d in river3})))

    s4 = _build(["Qs", "Kh", "2d"], "", "", "")
    turn4 = [d for d in _chance_deals(s4._tree).values() if d[0] == 3]
    check("subset: unset restores the full 49-card turn deal", turn4 and all(len(d[2]) == 49 for d in turn4), str(sorted({len(d[2]) for d in turn4})))
    print(f"  nodes: subset K=4 M=3 -> {tree.num_nodes}; full -> {s4._tree.num_nodes}")


if __name__ == "__main__":
    test_subsets()
    print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
    sys.exit(1 if _FAIL else 0)
