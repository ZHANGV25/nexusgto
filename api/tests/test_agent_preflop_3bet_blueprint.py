"""Blueprint-as-oracle replay for the 3-bet HEADS-UP shape (Lane 2, EC2-gated).

The load-bearing anchor pattern (test_agent_key_builder): replay GameState
forward along 3-bet HU lines built END-TO-END by the agent's own constructors
(build_preflop_prefix + canonical_flop_chips via new_postflop_state) and
assert the 8.05B HITs at every emitted key with the EXACT stored sigma length.
A wrong seat / wrong index / wrong chips lands on a different (or absent)
node -> MISS or length mismatch -> fail.

Capture provenance: poker-solver scripts/preflop_shapes_captured.json
GOLDEN, discriminators MISS, trainer-C chips float32-exact, sigma-length
walks clean through the river all-in clamp.

SKIPs cleanly without the blueprint (pure tests live in
tests/test_agent_preflop_3bet.py).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from app.agent.game_state import new_postflop_state
from app.agent.key_builder import build_key
from app.agent.preflop_constructor import PreflopAction, build_preflop_prefix

from tests.test_agent_preflop_3bet import CAPTURED, _threebet_line


def _ci(card: str) -> int:
    ranks = "AKQJT98765432"
    so = ("c", "d", "h", "s")
    return (12 - ranks.index(card[0].upper())) * 4 + so.index(card[1].lower())


BOARD = [_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s")]
HERO = (_ci("Qh"), _ci("Jh"))


def _bucket(blueprint, board, street, hero):
    bd = list(board)
    if street == 1:
        return blueprint.hand_to_bucket(bd[:3], *hero)
    if street == 2:
        return blueprint.hand_to_bucket_turn(bd[:4], *hero)
    return blueprint.hand_to_bucket_river(bd[:5], *hero)


def _replay_assert(blueprint, seats, prefix, action_seq, hero=HERO):
    """Walk `action_seq` from the DERIVED-chips flop entry; assert the
    blueprint HITs with the exact sigma length at every decision node.
    Returns the per-node sigma lengths."""
    st = new_postflop_state(seats=seats, preflop_prefix=prefix, board=BOARD)
    lengths = []
    for act in action_seq:
        assert not st.terminal, "line ended before the scripted actions did"
        key = build_key(st)
        bucket = _bucket(blueprint, st.board, key.street, hero)
        strat = blueprint.get_strategy(
            list(st.board), key.action_history, key.player, bucket,
            street=key.street,
        )
        assert strat is not None, (
            f"MISS at {key.as_tuple()} — wrong seat/index/chips would land here"
        )
        arr = np.asarray(strat, dtype=float)
        assert arr.sum() > 0, f"zero-sum sigma at {key.as_tuple()}"
        assert st.num_legal_actions() == len(arr), (
            f"length mismatch at {key.as_tuple()}: state {st.num_legal_actions()}"
            f" vs stored {len(arr)} (chip/commitment-prune disagreement)"
        )
        lengths.append(len(arr))
        st = st.advance(act)
    return lengths, st


# ── Oracle replay: bet-call line to TERMINAL (river all-in clamp) ─────────


def test_oracle_3bet_hu_btn_bb_betline_to_terminal(blueprint):
    # Line A: BTN open / BB 3-bet / BTN call. bet-call x3 -> river all-in.
    prefix = build_preflop_prefix(_threebet_line("BTN", "BB"))
    assert prefix == [0, 0, 0, 2, 0, 2, 1]
    lengths, st = _replay_assert(
        blueprint, ["BB", "BTN"], prefix, [1, 1, 1, 1, 1, 1],
    )
    # Capture-pinned signature incl. the chips-discriminating river faced
    # node na=2 (single-open chips would store na=3 there).
    assert lengths == [2, 3, 2, 3, 2, 2]
    assert st.terminal


def test_oracle_3bet_hu_utg_btn_betline_to_terminal(blueprint):
    # Line B: UTG open / BTN 3-bet / UTG call (folds between AND after).
    prefix = build_preflop_prefix(_threebet_line("UTG", "BTN"))
    assert prefix == [2, 0, 0, 2, 0, 0, 1]
    lengths, st = _replay_assert(
        blueprint, ["UTG", "BTN"], prefix, [1, 1, 1, 1, 1, 1],
    )
    assert lengths == [2, 3, 2, 3, 2, 2]
    assert st.terminal


def test_oracle_3bet_hu_flop_raise_war(blueprint):
    # Line A flop war: bet -> raise -> all-in raise -> call (the commitment
    # prune sequence [2,3,3,2], capture-pinned).
    prefix = build_preflop_prefix(_threebet_line("BTN", "BB"))
    lengths, _ = _replay_assert(blueprint, ["BB", "BTN"], prefix, [1, 2, 2, 1])
    assert lengths == [2, 3, 3, 2]


# ── All 15 pairs: constructed root key HITs trained ───────────────────────


@pytest.mark.parametrize("pair", sorted(CAPTURED), ids=lambda p: f"{p[0]}-{p[1]}")
def test_all_15_pairs_flop_root_hits_trained(blueprint, pair):
    prefix_fixture, _, _, live = CAPTURED[pair]
    prefix = build_preflop_prefix(_threebet_line(*pair))
    assert prefix == prefix_fixture
    st = new_postflop_state(seats=live, preflop_prefix=prefix, board=BOARD[:3])
    key = build_key(st)
    hero = (_ci("As"), _ci("Ad"))  # AsAd was GOLDEN at all 15 roots
    bucket = _bucket(blueprint, st.board, key.street, hero)
    strat = blueprint.get_strategy(
        list(st.board), key.action_history, key.player, bucket, street=1,
    )
    assert strat is not None, f"MISS at {key.as_tuple()} for {pair}"
    arr = np.asarray(strat, dtype=float)
    assert arr.sum() > 0
    assert st.num_legal_actions() == len(arr)
    arr = arr / arr.sum()
    uniform = np.full(len(arr), 1.0 / len(arr))
    assert np.abs(arr - uniform).max() > 0.05, (
        f"untrained sigma at {key.as_tuple()} for {pair}: {arr}"
    )


# ── 1.0x 3-bet (index 3): artifact-gated oracle replay (Lane C) ───────────
#
# Ground truth comes from preflop_shapes_captured.json in the poker-solver
# checkout (the Lane-C probe APPENDS threebet_size_idx=1 walks for these
# pairs). Each test SKIPS until the artifact has the index-1 walk for its
# pair — they turn red/green honestly the moment the EC2 capture lands.
# BTN-BB was already captured clean 2026-06-10, so it runs today.

_SZ1_PAIRS = [("BTN", "BB"), ("UTG", "BB"), ("BTN", "SB"),
              ("SB", "BB"), ("UTG", "BTN")]


@lru_cache(maxsize=1)
def _sz1_walks():
    ps_path = Path(os.environ.get(
        "POKER_SOLVER_PATH",
        str(Path(__file__).resolve().parent.parent.parent / "poker-solver"),
    ))
    artifact = ps_path / "scripts" / "preflop_shapes_captured.json"
    if not artifact.exists():
        return {}
    data = json.loads(artifact.read_text())
    return {
        (w["opener"], w["threebettor"]): w
        for w in data.get("trainer_walks", [])
        if w.get("threebet_size_idx") == 1
    }


def _sz1_walk_or_skip(pair):
    w = _sz1_walks().get(pair)
    if w is None:
        pytest.skip(
            f"no threebet_size_idx=1 capture for {pair} in "
            f"preflop_shapes_captured.json yet (Lane-C probe pending)"
        )
    return w


@pytest.mark.parametrize("pair", _SZ1_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
def test_sz1_constructed_line_matches_capture_and_hits_trained(blueprint, pair):
    w = _sz1_walk_or_skip(pair)
    prefix = build_preflop_prefix(_threebet_line(*pair), threebet_size_idx=1)
    assert prefix == w["prefix"], "constructed prefix != trainer-C capture"
    live = list(w["live_seats"])
    st = new_postflop_state(seats=live, preflop_prefix=prefix, board=BOARD[:3])
    # Derived f32 chips must equal the literal trainer-C chips.
    assert (st.pot, list(st.stacks)) == (w["flop_pot"], w["flop_stacks"])
    key = build_key(st)
    hero = (_ci("As"), _ci("Ad"))
    bucket = _bucket(blueprint, st.board, key.street, hero)
    strat = blueprint.get_strategy(
        list(st.board), key.action_history, key.player, bucket, street=1,
    )
    assert strat is not None, f"MISS at {key.as_tuple()} for sz1 {pair}"
    arr = np.asarray(strat, dtype=float)
    assert arr.sum() > 0
    assert st.num_legal_actions() == len(arr)
    arr = arr / arr.sum()
    uniform = np.full(len(arr), 1.0 / len(arr))
    assert np.abs(arr - uniform).max() > 0.05, (
        f"untrained sigma at {key.as_tuple()} for sz1 {pair}: {arr}"
    )


@pytest.mark.parametrize("pair", _SZ1_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
def test_sz1_betline_to_terminal(blueprint, pair):
    # The 1.0x chip path's na signature: replay bet-call to the river all-in
    # clamp asserting state-vs-stored sigma length at EVERY node (the bigger
    # 3-bet shifts the commitment prune — exactly what sz0 chips would get
    # wrong).
    w = _sz1_walk_or_skip(pair)
    prefix = build_preflop_prefix(_threebet_line(*pair), threebet_size_idx=1)
    lengths, st = _replay_assert(
        blueprint, list(w["live_seats"]), prefix, [1, 1, 1, 1, 1, 1],
    )
    assert st.terminal
    assert len(lengths) == 6


# ── Canary: the wrong seat at the flop root must MISS ─────────────────────


def test_flop_root_wrong_seat_misses(blueprint):
    prefix = build_preflop_prefix(_threebet_line("BTN", "BB"))
    st = new_postflop_state(seats=["BB", "BTN"], preflop_prefix=prefix,
                            board=BOARD[:3])
    key = build_key(st)
    assert key.player == 1  # BB acts first at the 3-bet-pot flop root
    bucket = _bucket(blueprint, st.board, 1, HERO)
    # BTN (5) is NOT first to act: the same history at p=5 must MISS.
    assert blueprint.get_strategy(
        list(st.board), key.action_history, 5, bucket, street=1,
    ) is None
