"""TDD: key_builder (plan §3.2, §4e ANCHOR).

The ANCHOR (load-bearing): replay GameState forward along realistic 2p + 3p
lines and assert the blueprint HITs present+trained at EVERY (player,
action_history, street) key key_builder emits. The blueprint only hits the
CORRECT node — a wrong seat or wrong order → MISS → test fails. This uses the
real product (the blueprint) as the oracle; it is stronger than the
instrumented-rollout fixture (which was contaminated by mislabeled positions
and rollout's scan-from-slot0 order) and GPU-free.

Pure-unit key_builder tests (no blueprint) run always; the oracle replay
SKIPS if the 8.05B isn't present.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.agent.game_state import new_postflop_state
from app.agent.key_builder import NodeKey, build_key

# ── Pure unit: assembly correctness (no blueprint) ───────────────────────


def test_build_key_flop_root_3p():
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[], pot=720, stacks=[9660, 9780, 9780],
    )
    key = build_key(st)
    assert isinstance(key, NodeKey)
    assert key.action_history == [2, 0, 0, 1, 0, 1]
    assert key.player == 1  # BB acts first postflop
    assert key.street == 1  # flop


def test_build_key_appends_postflop_history():
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[], pot=720, stacks=[9660, 9780, 9780],
    )
    st = st.advance(0)  # BB check
    key = build_key(st)
    assert key.action_history == [2, 0, 0, 1, 0, 1, 0]
    assert key.player == 2  # UTG to act
    assert key.street == 1


def test_build_key_player_is_seat_not_slot():
    # 3-way, after BB+UTG check, BTN (SEAT 5, slot 2) acts. The key's player
    # MUST be the seat (5), never the slot (2).
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[], pot=720, stacks=[9660, 9780, 9780],
    )
    st = st.advance(0).advance(0)  # BB check, UTG check
    key = build_key(st)
    assert key.player == 5  # BTN seat, NOT slot 2
    assert key.action_history == [2, 0, 0, 1, 0, 1, 0, 0]


def test_build_key_street_advances():
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[10, 20, 30], pot=720, stacks=[9660, 9780, 9780],
    )
    st = st.advance(0).advance(0).advance(0)  # check-check-check -> turn
    key = build_key(st)
    assert key.street == 2  # turn
    assert key.player == 1  # BB first on turn


def test_build_key_raises_on_terminal_state():
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[], pot=440, stacks=[9780, 9780],
    )
    # BB bet, UTG fold -> one player left, terminal.
    st = st.advance(1)  # BB bet
    st = st.advance(0)  # UTG fold -> terminal
    assert st.terminal
    with pytest.raises(ValueError):
        build_key(st)


def test_node_key_as_tuple_is_hashable_and_immutable():
    # Pure-unit so key_builder hits 100% WITHOUT the blueprint (CI-friendly).
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[], pot=440, stacks=[9780, 9780],
    )
    key = build_key(st)
    t = key.as_tuple()
    assert t == ((2, 0, 0, 0, 0, 1), 1, 1)
    assert hash(t)  # usable as a dict/cache key


# ── Blueprint-oracle replay ANCHOR ───────────────────────────────────────


def _replay_assert_hits(blueprint, board_ints, hero, seats, prefix, action_seq):
    """Replay a line; assert the blueprint HITs (present, non-degenerate) at
    each emitted key.

    The ANCHOR is HIT/MISS: a wrong seat/order/street lands on a different
    (or absent) node → MISS → fail. We do NOT assert "trained" per node — the
    8.05B carries no visit-count signal (vc_data absent), so trained-balanced
    vs barely-trained is indistinguishable from the σ̄ vector alone (see the
    decode_resolve memory). We reject only the EXACTLY-uniform degenerate as a
    guard. Strategy QUALITY is checked separately on decisive nodes.

    Returns the list of (key, top_action, mix) for targeted sanity asserts.
    """
    c0, c1 = hero
    st = new_postflop_state(
        seats=seats, preflop_prefix=prefix, board=board_ints, pot=720,
        stacks=[9660] + [9780] * (len(seats) - 1),
    )
    out = []
    for act in action_seq:
        if st.terminal:
            break
        key = build_key(st)
        bd = list(st.board)
        if key.street == 1:
            bucket = blueprint.hand_to_bucket(bd[:3], c0, c1)
        elif key.street == 2:
            bucket = blueprint.hand_to_bucket_turn(bd[:4], c0, c1)
        else:
            bucket = blueprint.hand_to_bucket_river(bd[:5], c0, c1)
        strat = blueprint.get_strategy(
            bd, key.action_history, key.player, bucket, street=key.street,
        )
        assert strat is not None, (
            f"MISS at key {key} — wrong seat/order/street would land here"
        )
        arr = np.asarray(strat, dtype=float)
        assert arr.sum() > 0, f"zero-sum strat at {key}"
        # Per-node VOCAB invariant: the GameState's legal-action count must
        # equal the blueprint's stored σ̄ length at EVERY node. Catches vocab
        # mismatch along the whole line, not just hand-checked nodes.
        assert st.num_legal_actions() == len(arr), (
            f"action-count mismatch at {key}: state says "
            f"{st.num_legal_actions()}, blueprint stores {len(arr)}"
        )
        arr = arr / arr.sum()
        # Degenerate guard: reject EXACTLY uniform (untrained-by-construction).
        uniform = np.full(len(arr), 1.0 / len(arr))
        assert not np.allclose(arr, uniform, atol=1e-6), (
            f"exactly-uniform (degenerate) node at {key}: {arr}"
        )
        out.append((key, int(np.argmax(arr)), arr))
        st = st.advance(act)
    return out


def test_anchor_3p_checkdown_to_river(blueprint):
    # 3p line UTG raise / BTN call / BB call. Check down flop, turn, river.
    # Hero = BB QhJh. Board must extend to 5 cards across streets.
    board = [
        # AhKsTd  Qc (turn)  2s (river) — pick cards that don't conflict QhJh
        _ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s"),
    ]
    hero = (_ci("Qh"), _ci("Jh"))
    # 3 checks per street -> closes each street -> 9 actions to walk the line.
    seq = [0] * 9
    results = _replay_assert_hits(
        blueprint, board, hero, ["BB", "UTG", "BTN"], [2, 0, 0, 1, 0, 1], seq,
    )
    assert len(results) >= 6  # at least flop(3)+turn(3) decisions hit+trained


def test_anchor_3p_faced_bet_who_acts_next(blueprint):
    # THE historical bug as an anchor: BB bets flop, UTG faces (seat2), BTN
    # faces (seat5), wrap to BB. Each must HIT at the right seat.
    board = [_ci("Ah"), _ci("Ks"), _ci("Td")]
    hero = (_ci("Qh"), _ci("Jh"))
    # BB bet(1), UTG call(1), BTN call(1) -> flop closes. Walk just the flop.
    seq = [1, 1, 1]
    results = _replay_assert_hits(
        blueprint, board, hero, ["BB", "UTG", "BTN"], [2, 0, 0, 1, 0, 1], seq,
    )
    # First decision = BB (seat1), then UTG (seat2 facing bet), then BTN (seat5)
    seats_hit = [k.player for k, _, _ in results]
    assert seats_hit[:3] == [1, 2, 5]


def test_anchor_2p_checkdown(blueprint):
    # 2p line UTG raise, fold to BB, BB call. BB QhJh. Check down.
    board = [_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s")]
    hero = (_ci("Qh"), _ci("Jh"))
    seq = [0] * 6  # 2 checks/street x 3 streets
    results = _replay_assert_hits(
        blueprint, board, hero, ["BB", "UTG"], [2, 0, 0, 0, 0, 1], seq,
    )
    assert len(results) >= 4


# ── Targeted poker-SANITY on decisive nodes (strategy quality, not key) ──
#
# HIT proves the key is structurally right (above). These assert the σ̄ at a
# few nodes where poker theory predicts a clear answer — a wrong-but-present
# key would carry the wrong action and fail here. Seeded from the played-line
# probe (playing_agent_v1 memory): faced-bet flop is call-heavy (mixed), and a
# strong made hand on a dry board favors value-betting.


def test_sanity_3p_faced_bet_is_call_heavy(blueprint):
    # 3p: BB bets, UTG (QhJh, a gutshot on AhKsTd) faces it. The captured
    # node top=1 (call) — a genuine mixed call/raise decision, NOT fold-100.
    board = [_ci("Ah"), _ci("Ks"), _ci("Td")]
    hero = (_ci("Qh"), _ci("Jh"))
    st = new_postflop_state(
        seats=["BB", "UTG", "BTN"], preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=board, pot=720, stacks=[9660, 9780, 9780],
    )
    st = st.advance(1)  # BB bet -> UTG faces it
    key = build_key(st)
    assert key.player == 2 and key.action_history == [2, 0, 0, 1, 0, 1, 1]
    bucket = blueprint.hand_to_bucket(board[:3], *hero)
    arr = np.asarray(
        blueprint.get_strategy(board, key.action_history, key.player, bucket,
                               street=1), dtype=float)
    arr = arr / arr.sum()
    # facing-bet vocab [fold0, call1, raise2]. Should not be fold-dominated;
    # call should carry meaningful mass (the captured top action).
    assert len(arr) == 3
    assert arr[1] > 0.25, f"call mass too low: {arr}"
    assert arr[0] < 0.6, f"fold-dominated (suspicious): {arr}"


def test_sanity_strong_madehand_values_on_dry_board(blueprint):
    # BB with a set-ish strong hand on a dry river should bet (not check 100%).
    # Use a board where hero has the nut-ish hand: board Ah Ks Td 2c 7d,
    # hero AdAc? blocked by Ah. Use KdKc (set of kings) — top set on K-high.
    board = [_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("2c"), _ci("7d")]
    hero = (_ci("Kd"), _ci("Kc"))
    st = new_postflop_state(
        seats=["BB", "UTG"], preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=board, pot=720, stacks=[9660, 9780],
    )
    # walk to the river root: check down flop + turn (4 checks).
    for _ in range(4):
        st = st.advance(0)
    assert st.street == 3  # river
    key = build_key(st)
    bucket = blueprint.hand_to_bucket_river(board[:5], *hero)
    arr = np.asarray(
        blueprint.get_strategy(board, key.action_history, key.player, bucket,
                               street=3), dtype=float)
    arr = arr / arr.sum()
    # no-bet vocab [check0, bet1]. Strong made hand should carry real bet mass.
    assert len(arr) == 2
    assert arr[1] > 0.15, f"strong hand barely bets (suspicious): {arr}"


def _ci(card: str) -> int:
    ranks = "AKQJT98765432"
    so = ("c", "d", "h", "s")
    return (12 - ranks.index(card[0].upper())) * 4 + so.index(card[1].lower())


# ── 3a oracle: committed line replayed to TERMINAL ───────────────────────
#
# After the stack-commitment fix, lines continue PAST commitment for the first
# time (the bench used to stop at the crash). This walks a maxed raise war ALL
# THE WAY to terminal and asserts the load-bearing invariant at EVERY present
# node: GameState's chip-aware num_legal_actions() == the blueprint's stored σ̄
# length. A residual chip-math error would surface here as a length mismatch
# (the loud invariant), never a silent wrong-node serve.


def _bucket_for(blueprint, board, street, hero):
    bd = list(board)
    if street == 1:
        return blueprint.hand_to_bucket(bd[:3], *hero)
    if street == 2:
        return blueprint.hand_to_bucket_turn(bd[:4], *hero)
    return blueprint.hand_to_bucket_river(bd[:5], *hero)


def _walk_committed_to_terminal(blueprint, seats, prefix, board, hero, recipe):
    """Walk `recipe` (clamped to the live vocab) to terminal; assert at every
    present+trained node that num_legal_actions == σ̄ length. Returns node count.
    """
    st = new_postflop_state(seats=seats, preflop_prefix=prefix, board=board)
    checked = 0
    guard = 0
    for _ in range(60):
        if st.terminal:
            break
        guard += 1
        n = st.num_legal_actions()
        key = build_key(st)
        bucket = _bucket_for(blueprint, st.board, key.street, hero)
        strat = blueprint.get_strategy(
            list(st.board), key.action_history, key.player, bucket,
            street=key.street,
        )
        if strat is not None:
            arr = np.asarray(strat, dtype=float)
            if arr.sum() > 0:
                # THE invariant: chip-aware vocab matches stored σ̄ length.
                assert n == len(arr), (
                    f"length mismatch at {key.as_tuple()}: num_legal={n}, "
                    f"σ̄ len={len(arr)} (chip-commitment prune disagreement)"
                )
                checked += 1
        # advance: clamp the recipe action to the live vocab (committed nodes
        # may have pruned the raise the recipe wanted).
        want = recipe[(guard - 1) % len(recipe)]
        act = want if want < n else (n - 1)
        st = st.advance(act)
    return checked


def test_oracle_committed_raise_war_3p_invariant_to_terminal(blueprint):
    # The exact raise war that used to CRASH, now walked to terminal. Every
    # present node must satisfy num_legal == σ̄ length (no crash, no mismatch).
    board = [_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s")]
    hero = (_ci("Qh"), _ci("Jh"))
    recipe = [0, 0, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1]
    checked = _walk_committed_to_terminal(
        blueprint, ["BB", "UTG", "BTN"], [2, 0, 0, 1, 0, 1], board, hero, recipe,
    )
    assert checked >= 5  # multiple committed nodes verified, including the deep ones


def test_oracle_committed_raise_war_2p_invariant_to_terminal(blueprint):
    board = [_ci("Ah"), _ci("Ks"), _ci("Td"), _ci("Qc"), _ci("2s")]
    hero = (_ci("Qh"), _ci("Jh"))
    recipe = [1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1]
    checked = _walk_committed_to_terminal(
        blueprint, ["BB", "UTG"], [2, 0, 0, 0, 0, 1], board, hero, recipe,
    )
    assert checked >= 4
