"""OP3 single-street mode (POKER_SEARCH_DEPTH=single) regression —
verifies the legacy single-street tree builder path is untouched when
the env var is set to 'single'.

Must be run in a fresh Python process with POKER_SEARCH_DEPTH=single
set BEFORE import. There's a documented Windows-side issue where
sequential StreetSolverGPU constructions in the same process can
destabilize the shared CUDA runtime; we sidestep it by spawning one
subprocess per tree-build.

Gate: zero SS_NODE_CHANCE nodes for HU flop, 3p flop, HU turn, 3p turn
under POKER_SEARCH_DEPTH=single.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent.parent


_PASS: List[str] = []
_FAIL: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        _PASS.append(name); print(f"  PASS  {name}")
    else:
        _FAIL.append(name); print(f"  FAIL  {name}")
        if detail: print(f"        {detail}")


def _spawn_build(board: List[str], num_players: int) -> int:
    """Spawn a subprocess, build the tree once under POKER_SEARCH_DEPTH=single,
    print the chance-node count to stdout. Returns the count, or -1 on error.
    """
    board_repr = repr(board)
    prog = textwrap.dedent(f"""
        import os, sys
        os.environ['POKER_SEARCH_DEPTH'] = 'single'
        sys.path.insert(0, {str(ROOT / 'python')!r})
        from street_solver_gpu import StreetSolverGPU

        hands = []
        board_set = set([])  # cards parsed below
        from solver import card_to_int
        board_cards = [card_to_int(c) for c in {board_repr}]
        board_set = set(board_cards)
        for c0 in range(52):
            if c0 in board_set: continue
            for c1 in range(c0 + 1, 52):
                if c1 in board_set: continue
                hands.append((c0, c1, 1.0))
                if len(hands) >= {2 * num_players}: break
            if len(hands) >= {2 * num_players}: break
        ranges = [hands[2*i:2*i+2] for i in range({num_players})]

        s = StreetSolverGPU(
            board={board_repr},
            player_ranges=ranges,
            pot_bb=6.5, stack_bb=97.5,
            bet_sizes=[0.75], use_cont_strats=False,
        )
        sys.stdout.write(f'CHANCE={{s._tree.num_chance_nodes}}\\n')
        sys.stdout.flush()
        s._freed = True  # don't call ss_free_tree in the subprocess; skip cleanup
        os._exit(0)
    """)
    out = subprocess.run(
        [sys.executable, "-c", prog],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    for line in out.stdout.splitlines():
        if line.startswith("CHANCE="):
            return int(line.split("=", 1)[1])
    # Debug: dump captured output on failure.
    print(f"    [subprocess rc={out.returncode}]")
    print(f"    stdout: {out.stdout[:500]}")
    print(f"    stderr: {out.stderr[:500]}")
    return -1


def test_hu_flop():
    c = _spawn_build(["Qs", "Kh", "2d"], 2)
    check("single/hu_flop: zero chance", c == 0, f"chance={c}")


def test_hu_turn():
    c = _spawn_build(["Qs", "Kh", "2d", "7h"], 2)
    check("single/hu_turn: zero chance", c == 0, f"chance={c}")


def test_3p_flop():
    c = _spawn_build(["Qs", "Kh", "2d"], 3)
    check("single/3p_flop: zero chance", c == 0, f"chance={c}")


def test_3p_turn():
    c = _spawn_build(["Qs", "Kh", "2d", "7h"], 3)
    check("single/3p_turn: zero chance", c == 0, f"chance={c}")


if __name__ == "__main__":
    print("[test_op3_single_street_mode]")
    test_hu_flop()
    test_hu_turn()
    test_3p_flop()
    test_3p_turn()
    print(f"\n{'PASSED' if not _FAIL else 'FAILED'} {len(_PASS)}/{len(_PASS)+len(_FAIL)}")
    sys.exit(0 if not _FAIL else 1)
