"""Blueprint-as-oracle replay for the HU limp-check shapes (EC2-gated).

Same anchor pattern as tests/test_agent_preflop_3bet_blueprint.py: replay
GameState forward along the limp-check lines built END-TO-END by the agent's
own constructors (build_preflop_prefix + canonical_flop_chips via
new_postflop_state) and assert the 8.05B HITs at every emitted key with the
EXACT stored sigma length. Ground truth is READ FROM the captured artifact
(poker-solver scripts/limp_shapes_captured.json, 2026-06-11, ALL PROBES
CONSISTENT) — the walks' histories, pots and na values drive the asserts.

SKIPs cleanly without the blueprint (conftest fixture) or without the
artifact. Pure tests live in tests/test_agent_preflop_limp.py.
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


def _ci(card: str) -> int:
    ranks = "AKQJT98765432"
    so = ("c", "d", "h", "s")
    return (12 - ranks.index(card[0].upper())) * 4 + so.index(card[1].lower())


# The capture's board/heroes (meta.board = AhKsTdQc2s, heroes QhJh + AsAd).
BOARD = [_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s")]
HEROES = {"QhJh": (_ci("Qh"), _ci("Jh")), "AsAd": (_ci("As"), _ci("Ad"))}

# tag -> the limper seat whose line the agent constructs (HU lines only;
# the 3-way L4/L5 walks exist in the artifact but are NOT served).
_SERVED_LINES = {
    "L1 UTG-limp/BB-check": "UTG",
    "L2 BTN-limp/BB-check": "BTN",
    "L3 SB-complete/BB-check": "SB",
}


def _limp_line(limper: str):
    return [PreflopAction(limper, "call"), PreflopAction("BB", "check")]


@lru_cache(maxsize=1)
def _artifact():
    ps_path = Path(os.environ.get(
        "POKER_SOLVER_PATH",
        str(Path(__file__).resolve().parent.parent.parent / "poker-solver"),
    ))
    artifact = ps_path / "scripts" / "limp_shapes_captured.json"
    if not artifact.exists():
        return None
    return json.loads(artifact.read_text())


def _walk_or_skip(tag: str):
    data = _artifact()
    if data is None:
        pytest.skip("limp_shapes_captured.json not found (EC2 capture artifact)")
    for w in data["trainer_walks"]:
        if w["tag"] == tag:
            return w
    pytest.skip(f"no trainer walk for {tag!r} in limp_shapes_captured.json")


def _length_walk_or_skip(kind: str, tag: str):
    data = _artifact()
    if data is None:
        pytest.skip("limp_shapes_captured.json not found (EC2 capture artifact)")
    nodes = [w for w in data["length_walks"] if w["tag"] == f"{kind} {tag}"]
    if not nodes:
        pytest.skip(f"no {kind} length walk for {tag!r} in the artifact")
    return nodes


def _bucket(blueprint, board, street, hero):
    bd = list(board)
    if street == 1:
        return blueprint.hand_to_bucket(bd[:3], *hero)
    if street == 2:
        return blueprint.hand_to_bucket_turn(bd[:4], *hero)
    return blueprint.hand_to_bucket_river(bd[:5], *hero)


# ── Flop root: constructed key HITs trained, chips == trainer walk ────────


@pytest.mark.parametrize("tag", sorted(_SERVED_LINES))
def test_limp_flop_root_hits_trained(blueprint, tag):
    w = _walk_or_skip(tag)
    limper = _SERVED_LINES[tag]
    prefix = build_preflop_prefix(_limp_line(limper))
    assert prefix == w["prefix"], "constructed prefix != trainer-C capture"
    st = new_postflop_state(
        seats=list(w["live_seats"]), preflop_prefix=prefix, board=BOARD[:3],
    )
    # Derived chips must equal the literal trainer-C walk.
    assert (st.pot, list(st.stacks)) == (w["flop_pot"], w["flop_stacks"])
    key = build_key(st)
    for hero_name, hero in HEROES.items():
        bucket = _bucket(blueprint, st.board, key.street, hero)
        strat = blueprint.get_strategy(
            list(st.board), key.action_history, key.player, bucket, street=1,
        )
        assert strat is not None, (
            f"MISS at {key.as_tuple()} for {tag} hero {hero_name}"
        )
        arr = np.asarray(strat, dtype=float)
        assert arr.sum() > 0
        assert st.num_legal_actions() == len(arr)
        arr = arr / arr.sum()
        uniform = np.full(len(arr), 1.0 / len(arr))
        assert np.abs(arr - uniform).max() > 0.005, (
            f"untrained sigma at {key.as_tuple()} for {tag}: {arr}"
        )


# ── σ̄-length walks: replay the artifact's chosen sequence st1-st3 ─────────


@pytest.mark.parametrize("kind", ["BETLINE", "WAR"])
@pytest.mark.parametrize("tag", sorted(_SERVED_LINES))
def test_limp_length_walk_replay(blueprint, kind, tag):
    nodes = _length_walk_or_skip(kind, tag)
    w = _walk_or_skip(tag)
    prefix = build_preflop_prefix(_limp_line(_SERVED_LINES[tag]))
    st = new_postflop_state(
        seats=list(w["live_seats"]), preflop_prefix=prefix, board=BOARD,
    )
    for node in nodes:
        assert not st.terminal, f"line ended before the captured walk did ({tag})"
        key = build_key(st)
        # The agent's key must equal the capture's node identity exactly.
        assert list(key.action_history) == node["history"], (
            f"history mismatch at {tag} {kind}: agent {key.action_history} "
            f"vs capture {node['history']}"
        )
        assert key.player == node["player"]
        assert key.street == node["street"]
        assert st.pot == node["pot"]
        slot = st.to_act_slot
        to_call = max(0, st.current_bet - st.street_bets[slot])
        assert to_call == node["to_call"]
        assert st.num_legal_actions() == node["na_c"], (
            f"vocab length {st.num_legal_actions()} != trainer na "
            f"{node['na_c']} at {key.as_tuple()}"
        )
        for hero_name, hero_info in node["heroes"].items():
            hero = HEROES[hero_name]
            bucket = _bucket(blueprint, st.board, key.street, hero)
            strat = blueprint.get_strategy(
                list(st.board), key.action_history, key.player, bucket,
                street=key.street,
            )
            assert (strat is not None) == hero_info["present"], (
                f"presence mismatch at {key.as_tuple()} hero {hero_name}"
            )
            if strat is not None:
                assert len(strat) == hero_info["na_bp"], (
                    f"sigma length {len(strat)} != captured na_bp "
                    f"{hero_info['na_bp']} at {key.as_tuple()}"
                )
        st = st.advance(node["chosen"]["idx"])


# ── Canary: the wrong seat at the L1 flop root must MISS ──────────────────


def test_limp_flop_root_wrong_seat_misses(blueprint):
    w = _walk_or_skip("L1 UTG-limp/BB-check")
    prefix = build_preflop_prefix(_limp_line("UTG"))
    st = new_postflop_state(
        seats=list(w["live_seats"]), preflop_prefix=prefix, board=BOARD[:3],
    )
    key = build_key(st)
    assert key.player == 1  # BB acts first at the limped flop root
    hero = HEROES["QhJh"]
    bucket = _bucket(blueprint, st.board, 1, hero)
    # UTG (2) is NOT first to act: the same history at p=2 must MISS
    # (matches the capture's wrong-seat(2) discriminator).
    assert blueprint.get_strategy(
        list(st.board), key.action_history, 2, bucket, street=1,
    ) is None
