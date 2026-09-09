"""Per-leaf opponent-range conditioning for the 2p-flop cont-strat-leaf solve.

WHY (root-caused + verified 2026-06-17, docs/CPU_ES_CHANCE_NODE_FINDINGS_2026_06_14.md):
the depth-limited cont-strat-leaf flop value samples each villain hand from the
STATIC preflop-grid range, IDENTICALLY at every leaf. The in-tree flop CFR
controls only the REACH magnitude into a leaf (solver.c consumes the leaf scalar
as opp_total * my_vals[h]); it never conditions the opponent HAND distribution
baked into the leaf value. So at the BET-call leaf hero is valued as if the
ENTIRE preflop range called the flop bet — no "only strong hands continue"
filtering — which OVER-REWARDS betting marginal hands (QJs gutshot+overcards bets
~0.99 vs the blueprint's confident CHECK 0.957). The reach-CFR full extension,
which conditions the range in-tree, CHECKS QJs and matches the blueprint.

THE FIX (this module): per leaf, multiply each player's per-hand sampling weight
by the blueprint probability that THAT player, holding that hand, took the flop
actions on the leaf's line:

    cond_weight[leaf][p][h] = static_weight[p][h] * P_continue(leaf, p, h)
    P_continue = product over flop decision nodes d on the line where seat p acts
                 of  sigma_bar(d, bucket_p(h))[action_p_took_at_d]

At the BET-call leaf the only villain factor is sigma_bar(call | h): hands the
blueprint folds get ~0 weight and drop out of the sampled villain range, so
betting a draw is punished by the strong continuing range. At the CHECK-CHECK
root leaf P_continue is all-ones (no continue/fold discriminating action) so the
range is unchanged. The leaf-table value's relative villain weights are what
matter (rollout_sample_opp_hand renormalizes by total), so no global
normalization is needed here.

SCOPE / GATING: 2p FLOP and TURN only (street=1 flop / street=2 turn). Multiway
acting-order attribution is NOT trusted here, and the 5-6p external-sampling path
and the 3p+ flop path call the UNCHANGED sampled construct (they never reach this
module). Fail-loud on misuse.

A blueprint MISS (no bucket / no strategy / action index out of range) yields a
multiplier of EXACTLY 1.0 for that node — a strict no-op, mirroring the rollout's
own tolerate-miss policy. Conditioning can only SHARPEN the villain range toward
the continuing hands; it can never corrupt a hand the blueprint has no opinion on.

HERO-ROW CONTRACT (2026-06-18): the per-leaf list this returns is reused by the C
all-hands construct to enumerate EACH seat as hero for its own table row, and the
hero zero-gate (rollout.c:1250) treats weight<=0 as "skip -> 0.0 EV". So a
conditioned weight is FLOORED to _HERO_ENUM_FLOOR whenever the static weight was
positive: as hero the hand keeps a true EV (contract: "hero keeps the full static
range"); as a sampled opponent the floored hand is ~0 (conditioning preserved).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# A per-leaf, per-player range: list[leaf] -> list[player] -> list[(c0,c1,weight)].
PerLeafRanges = List[List[List[Tuple[int, int, float]]]]

# Below this total conditioned weight for a seat at a leaf, fall back to the
# static weights for that seat+leaf (never ship a silent all-zero villain range —
# a long product of small probabilities could underflow). Bumps a counter.
_MIN_TOTAL_COND_WEIGHT = 1e-9

# HERO-ROW FLOOR (2026-06-18, fixes the review's MEDIUM finding). The per-leaf
# list is consumed by the C all-hands construct in BOTH roles: it samples the
# OPPONENT from each seat's list (rollout_sample_opp_hand, which skips weight<=0
# and renormalizes — so conditioning works) AND it enumerates each seat as HERO
# for that seat's own table row (rollout.c:1250 zero-gates `if (hw<=0) return
# 0.0`). The C export's contract is "HERO rows keep the full static range, so
# hero is still enumerated over all hands" (rollout.c:1657). A conditioning
# multiplier of EXACTLY 0.0 (the blueprint plays a hand's leaf action with pure-0
# probability) would zero that hand's weight, tripping the hero zero-gate and
# handing hero a SPURIOUS 0.0 leaf EV instead of its true rollout value — biasing
# hero's solved strategy precisely on the off-blueprint deviations a live
# re-solve exists to evaluate. FIX: floor every conditioned weight whose STATIC
# weight was positive to this epsilon. As an opponent the hand is then sampled
# with ~eps/total ≈ 0 probability (sampling-equivalent to dropped — conditioning
# preserved); as hero it has weight>0 so the zero-gate does NOT fire and the row
# gets the hand's true EV (contract honored). eps is far below any real σ̄ mass so
# it never perturbs the relative villain weights that drive opponent sampling.
_HERO_ENUM_FLOOR = 1e-12


class _CondStats:
    """Lightweight observability for a conditioning pass (not thread-shared —
    one instance per build call)."""

    __slots__ = ("nodes_conditioned", "bucket_miss", "strat_miss",
                 "idx_oor", "idx_oor_proxied", "underflow_fallback",
                 "hero_floor")

    def __init__(self) -> None:
        self.nodes_conditioned = 0
        self.bucket_miss = 0
        self.strat_miss = 0
        self.idx_oor = 0
        # Off-vocabulary aggressive actions (an explicit all-in or an added
        # size the blueprint node has no column for) conditioned on the
        # node's most aggressive blueprint action instead of being skipped.
        self.idx_oor_proxied = 0
        self.underflow_fallback = 0
        self.hero_floor = 0  # hands lifted off a pure-0 multiplier (hero-row fix)

    def as_dict(self) -> dict:
        return {
            "nodes_conditioned": self.nodes_conditioned,
            "bucket_miss": self.bucket_miss,
            "strat_miss": self.strat_miss,
            "idx_oor": self.idx_oor,
            "idx_oor_proxied": self.idx_oor_proxied,
            "underflow_fallback": self.underflow_fallback,
            "hero_floor": self.hero_floor,
        }


def build_conditioned_per_leaf_ranges(
    *,
    leaf_infos: Sequence,
    player_ranges: List[List[Tuple[int, int, float]]],
    board: List[int],
    player_seats: List[int],
    game_action_history: List[int],
    blueprint,
    n_players: int,
    street: int = 1,
) -> Tuple[PerLeafRanges, dict]:
    """Build a conditioned villain range per leaf for a 2p FLOP or TURN solve.

    Returns (per_leaf_ranges, stats). per_leaf_ranges[leaf][player] is a NEW list
    of (c0, c1, weight) tuples (immutable — the inputs are not mutated). For the
    BET-call leaf the villain's weak/folding hands are down-weighted toward 0; at
    the CHECK-check root leaf the weights equal the static range.

    Fail-loud (ValueError) if n_players != 2 — this conditioning is gated to
    heads-up by design (multiway actor attribution is not validated here).

    Args:
        leaf_infos: the LeafInfo list from extract_leaf_info_from_tree; each must
            carry flop_history_train (the trainer-encoded this-street action
            indices on the line) and flop_actor_train (the parallel acting seat
            per step). The fields are named "flop_*" for historical reasons but
            hold the SUBGAME-ROOT street's actions (flop for a flop root, turn for
            a turn root).
        player_ranges: static per-player [(c0, c1, weight), ...] (the grid range).
        board: the subgame-root board ints — 3 cards for a flop root (street=1),
            4 for a turn root (street=2). The bucketizer + blueprint street are
            dispatched on this length.
        player_seats: full blueprint seat id per compact player slot (so the
            blueprint query keys by the SAME seat the rollout uses).
        game_action_history: the action-index prefix UP TO the subgame-root street
            — preflop for a FLOP root, but preflop ++ flop actions for a TURN root
            (the caller passes full_history minus this street's actions). The
            trainer key for a node on the leaf's line is this prefix ++
            this-street-actions-so-far; the conditioner concatenates them the same
            way the C rollout does, so it keys the SAME node. (Named "preflop"
            locally for the flop case; do NOT trim it to only-preflop or turn
            conditioning will miss every node.)
        blueprint: a BPSReaderC with get_strategy(board, action_history, player,
            bucket, street), hand_to_bucket(board, c0, c1) [flop] and
            hand_to_bucket_turn(board, c0, c1) [turn].
        n_players: must be 2.
        street: blueprint street int — 1 (flop, 3-card board) or 2 (turn, 4-card).
    """
    if n_players != 2:
        raise ValueError(
            f"build_conditioned_per_leaf_ranges is heads-up only, got "
            f"n_players={n_players} (gate this off for 3p+)"
        )
    if street not in (1, 2):
        raise ValueError(
            f"conditioning supports flop (street=1) / turn (street=2), got {street}")
    _need_cards = 3 if street == 1 else 4
    if len(board) < _need_cards:
        raise ValueError(
            f"street={street} needs a {_need_cards}-card board, got {len(board)}")
    if len(player_ranges) != n_players:
        raise ValueError(
            f"player_ranges has {len(player_ranges)} entries, expected "
            f"{n_players}"
        )
    if len(player_seats) != n_players:
        raise ValueError(
            f"player_seats has {len(player_seats)} entries, expected "
            f"{n_players}"
        )

    stats = _CondStats()
    # Diagnostic toggle (default OFF -> floor ENABLED). NEXUS_HERO_FLOOR_DISABLE=1
    # reverts to the pre-fix behavior (conditioned weight can reach exactly 0,
    # re-introducing the hero zero-gate bug) so a measurement harness can A/B the
    # exploitability of the fixed vs bug-masked tables in one run WITHOUT reverting
    # the commit. Production never sets this. See docs/HERO_FLOOR_DIAGNOSIS.md.
    import os as _os
    _hero_floor_on = _os.environ.get(
        "NEXUS_HERO_FLOOR_DISABLE", "")[:1] not in ("1", "t", "T", "y", "Y")
    board = list(board[:_need_cards])
    # Action prefix up to the subgame-root street (preflop for a flop root;
    # preflop ++ flop for a turn root). NOT necessarily preflop-only — see the
    # game_action_history arg doc. The node key on the line is root_prefix ++
    # this-street-actions-before-the-step.
    root_prefix = list(game_action_history or [])

    # Street-dispatched bucketizer (the conditioning MUST use the same bucket the
    # trainer/rollout used for that street, or the query keys the wrong node).
    def _bucket(c0: int, c1: int) -> int:
        if street == 1:
            return int(blueprint.hand_to_bucket(board, c0, c1))
        return int(blueprint.hand_to_bucket_turn(board, c0, c1))

    # Cache bucket per (player_seat, hand) — the bucket is board-only, identical
    # across leaves, so compute it once per player per hand.
    bucket_cache: List[List[int]] = []
    for p in range(n_players):
        row: List[int] = []
        for (c0, c1, w) in player_ranges[p]:
            if w <= 0.0:
                row.append(-1)
                continue
            row.append(_bucket(int(c0), int(c1)))
        bucket_cache.append(row)

    # Cache strategy per (player_seat, prefix_key, bucket): many leaves share the
    # same decision-node prefix, and many hands share a bucket.
    strat_cache: dict = {}

    def _strat(seat: int, prefix: Tuple[int, ...], bucket: int):
        key = (seat, prefix, bucket)
        if key not in strat_cache:
            strat_cache[key] = blueprint.get_strategy(
                board=board, action_history=list(prefix), player=seat,
                bucket=bucket, street=street,
            )
        return strat_cache[key]

    per_leaf: PerLeafRanges = []
    for li in leaf_infos:
        history = tuple(getattr(li, "flop_history_train", ()) or ())
        actors = tuple(getattr(li, "flop_actor_train", ()) or ())
        # Defensive: if the parallel actor tuple is missing/short (legacy
        # LeafInfo), we cannot attribute actions — leave the range unconditioned
        # for this leaf (strict no-op) rather than guess.
        attributable = len(actors) == len(history)

        leaf_ranges: List[List[Tuple[int, int, float]]] = []
        for p in range(n_players):
            seat = int(player_seats[p])
            static = player_ranges[p]
            if not attributable:
                leaf_ranges.append([(int(c0), int(c1), float(w))
                                    for (c0, c1, w) in static])
                continue

            # The flop decision steps on this line where seat p acted, with the
            # incremental prefix = preflop ++ flop actions BEFORE that step (the
            # exact key the rollout/trainer builds, NOT the full closed line).
            steps: List[Tuple[Tuple[int, ...], int]] = []
            for i, (a_idx, actor) in enumerate(zip(history, actors)):
                if int(actor) != p:
                    continue
                prefix = tuple(root_prefix) + tuple(int(x) for x in history[:i])
                steps.append((prefix, int(a_idx)))

            new_hands: List[Tuple[int, int, float]] = []
            total = 0.0
            for h, (c0, c1, w) in enumerate(static):
                if w <= 0.0:
                    new_hands.append((int(c0), int(c1), 0.0))
                    continue
                bucket = bucket_cache[p][h]
                mult = 1.0
                if bucket < 0:
                    stats.bucket_miss += 1  # no-op for every step of this hand
                else:
                    for (prefix, a_idx) in steps:
                        strat = _strat(seat, prefix, bucket)
                        if strat is None or len(strat) == 0:
                            stats.strat_miss += 1
                            continue  # blueprint has no opinion -> x1.0
                        if a_idx < 0:
                            stats.idx_oor += 1
                            continue  # malformed index -> x1.0
                        if a_idx >= len(strat):
                            # OFF-VOCABULARY AGGRESSION (2026-09-09): the trainer
                            # vocab is [0.75] first / [1.0] subsequent plus an
                            # explicit all-in, so a jam (or an added off-tree
                            # size) has NO column in the blueprint node.
                            # Skipping it (x1.0) left the villain's all-in
                            # leaf range equal to his BET range: on Kh Qh 5h 9s
                            # ace-high called a 190 jam at 1.00 because it had
                            # 37% against a betting range. The closest honest
                            # proxy is the node's most aggressive blueprint
                            # action (its last column). Only when the node
                            # has at least two columns (a check/bet or
                            # fold/call/raise node); a one-column node stays a
                            # no-op.
                            if len(strat) >= 2:
                                s = float(strat[len(strat) - 1])
                                stats.idx_oor_proxied += 1
                            else:
                                stats.idx_oor += 1
                                continue
                        else:
                            s = float(strat[a_idx])
                        ssum = float(sum(strat))
                        # Normalize over the node's own returned support (a guard;
                        # bps strat is already a distribution).
                        mult *= (s / ssum) if ssum > 0.0 else 1.0
                        stats.nodes_conditioned += 1
                new_w = float(w) * mult
                # HERO-ROW FLOOR: a hand that was in the static range (w>0) must
                # never reach EXACTLY 0 — the C all-hands construct reuses this
                # list to enumerate the seat as HERO and zero-gates hw<=0 to a
                # spurious 0.0 leaf EV (see _HERO_ENUM_FLOOR). Floor keeps the
                # hand enumerable as hero (true EV) while leaving it ~0 for
                # opponent sampling (conditioning preserved). Only lifts hands the
                # blueprint drove to a pure-0 action; all others are untouched.
                if _hero_floor_on and new_w < _HERO_ENUM_FLOOR:
                    new_w = _HERO_ENUM_FLOOR
                    stats.hero_floor += 1
                new_hands.append((int(c0), int(c1), new_w))
                total += new_w

            # Underflow guard: never hand the sampler an all-zero villain range.
            if steps and total < _MIN_TOTAL_COND_WEIGHT:
                stats.underflow_fallback += 1
                new_hands = [(int(c0), int(c1), float(w))
                             for (c0, c1, w) in static]
            leaf_ranges.append(new_hands)
        per_leaf.append(leaf_ranges)

    return per_leaf, stats.as_dict()
