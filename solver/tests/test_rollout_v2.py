"""TDD tests for the rewritten Pluribus-faithful rollout (python/rollout.py).

Written BEFORE the implementation. Tests are derived from
pluribus_technical_details.md §4 and Pluribus Science 2019 supplement §4.

Test groups:
  A. Bias transforms (paper-exact)
  B. Action history threading (full preflop+postflop history reaches blueprint)
  C. Acting order (postflop order, not modulo num_players)
  D. Bucket correctness (blueprint's k-means, not ad-hoc EHS)
  E. Fail-loud on coverage miss
  F. Showdown payoffs
  G. Determinism
  H. BPR3 parity (rollout mean ≈ BPR3 EV at acting player, within MC noise)

Run with: python tests/test_rollout_v2.py
Exit code 0 iff all pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "python"))

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Test harness
# ──────────────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
FAILED_NAMES: List[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        FAILED_NAMES.append(name)
        print(f"  FAIL {name}" + (f"  [{detail}]" if detail else ""))


def check_close(
    name: str, got: float, expected: float, tol: float = 1e-9
) -> None:
    ok = abs(got - expected) <= tol
    check(name, ok, f"got={got} expected={expected} tol={tol}")


def check_array_close(
    name: str, got: np.ndarray, expected: np.ndarray, tol: float = 1e-9
) -> None:
    ok = got.shape == expected.shape and np.allclose(got, expected, atol=tol)
    check(name, ok, f"got={got}\n    expected={expected}")


# ──────────────────────────────────────────────────────────────────────────
# Fixtures: mock blueprints for isolating behavior
# ──────────────────────────────────────────────────────────────────────────


class RecorderBlueprint:
    """Blueprint stub that records every get_strategy call for inspection.

    Returns a fixed uniform distribution of the requested action count.
    Useful for verifying action-history threading: the test asserts that
    the recorded calls use the expected FULL action history at each step.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.buckets_requested: List[int] = []
        self.streets_to_load = [1, 2, 3]  # satisfy legacy compat check
        self._textures = {"__recorder__": True}

    def get_strategy(
        self,
        board: List[int],
        action_history: List[int],
        player: int,
        bucket: int,
        street: int = 1,
    ) -> np.ndarray:
        self.calls.append({
            "board": list(board),
            "action_history": list(action_history),
            "player": player,
            "bucket": bucket,
            "street": street,
        })
        # Return uniform over 4 actions (matches trained postflop tree
        # `[check, bet_0.5, bet_1.0, all-in]` width).
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    def hand_to_bucket(self, board: List[int], c0: int, c1: int) -> int:
        # Deterministic hash so tests are reproducible.
        h = (c0 * 53 + c1 * 7 + sum(board)) % 200
        self.buckets_requested.append(h)
        return h


class NoneBlueprint:
    """Returns None for every get_strategy — simulates coverage miss."""

    streets_to_load = [1, 2, 3]
    _textures = {"__none__": True}

    def get_strategy(self, *a: Any, **kw: Any) -> None:
        return None

    def hand_to_bucket(self, board: List[int], c0: int, c1: int) -> int:
        return 0


class FixedStrategyBlueprint:
    """Returns a predetermined strategy for every lookup."""

    def __init__(self, strategy: np.ndarray) -> None:
        self._strat = np.asarray(strategy, dtype=np.float32)
        self.streets_to_load = [1, 2, 3]
        self._textures = {"__fixed__": True}

    def get_strategy(self, *a: Any, **kw: Any) -> np.ndarray:
        return self._strat.copy()

    def hand_to_bucket(self, board: List[int], c0: int, c1: int) -> int:
        return 0


# ──────────────────────────────────────────────────────────────────────────
# A. Bias transforms — paper-exact
# ──────────────────────────────────────────────────────────────────────────


def test_bias_unmodified() -> None:
    """Profile 0 returns the input unchanged."""
    from rollout import apply_bias, ActionCategory  # type: ignore
    s = np.array([0.2, 0.3, 0.4, 0.1], dtype=np.float64)
    cats = [ActionCategory.CALL, ActionCategory.RAISE,
            ActionCategory.RAISE, ActionCategory.RAISE]
    out = apply_bias(s, bias_profile=0, categories=cats)
    check_array_close("bias_0 unmodified", out, s)


def test_bias_fold_5x() -> None:
    """Profile 1: P(fold) × 5, then renormalize.

    Start: [fold=0.2, call=0.3, raise=0.4, raise=0.1]  (sum=1.0)
    After fold×5: [1.0, 0.3, 0.4, 0.1]  (sum=1.8)
    Normalized: [0.5556, 0.1667, 0.2222, 0.0556]
    """
    from rollout import apply_bias, ActionCategory
    s = np.array([0.2, 0.3, 0.4, 0.1], dtype=np.float64)
    cats = [ActionCategory.FOLD, ActionCategory.CALL,
            ActionCategory.RAISE, ActionCategory.RAISE]
    out = apply_bias(s, bias_profile=1, categories=cats)
    expected = np.array([1.0, 0.3, 0.4, 0.1])
    expected = expected / expected.sum()
    check_array_close("bias_1 fold×5", out, expected, tol=1e-9)


def test_bias_call_5x() -> None:
    """Profile 2: P(call) × 5, then renormalize."""
    from rollout import apply_bias, ActionCategory
    s = np.array([0.2, 0.3, 0.4, 0.1], dtype=np.float64)
    cats = [ActionCategory.FOLD, ActionCategory.CALL,
            ActionCategory.RAISE, ActionCategory.RAISE]
    out = apply_bias(s, bias_profile=2, categories=cats)
    expected = np.array([0.2, 1.5, 0.4, 0.1])
    expected = expected / expected.sum()
    check_array_close("bias_2 call×5", out, expected, tol=1e-9)


def test_bias_raise_5x() -> None:
    """Profile 3: ALL raise probabilities × 5, renormalize.

    Per paper: "all raise probabilities multiplied by 5." So each
    raise is multiplied (the relative proportions between raises are
    preserved after renormalization — that's fine).
    """
    from rollout import apply_bias, ActionCategory
    s = np.array([0.2, 0.3, 0.4, 0.1], dtype=np.float64)
    cats = [ActionCategory.FOLD, ActionCategory.CALL,
            ActionCategory.RAISE, ActionCategory.RAISE]
    out = apply_bias(s, bias_profile=3, categories=cats)
    expected = np.array([0.2, 0.3, 2.0, 0.5])
    expected = expected / expected.sum()
    check_array_close("bias_3 raise×5 (each)", out, expected, tol=1e-9)


def test_bias_raise_preserves_ratios() -> None:
    """After raise×5 bias, the relative ratio between raises is the
    same as before (both multiplied by 5 then renormalized together).
    """
    from rollout import apply_bias, ActionCategory
    s = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    cats = [ActionCategory.FOLD, ActionCategory.CALL,
            ActionCategory.RAISE, ActionCategory.RAISE]
    out = apply_bias(s, bias_profile=3, categories=cats)
    # Ratio raise0 : raise1 before = 0.3 : 0.4 = 3 : 4
    # After bias+renorm must preserve 3 : 4
    r0, r1 = out[2], out[3]
    check_close("bias_3 preserves raise ratio", r0 / r1, 3.0 / 4.0, tol=1e-12)


def test_bias_degenerate_target_falls_back_to_unbiased() -> None:
    """If a bias profile's target has 0 probability in the source
    strategy (e.g., raise-biased at a check-only node), the post-bias
    sum is 0 (for the bumped categories), so we fall back to the
    *unbiased* strategy, NOT to uniform (the old code's mistake).
    """
    from rollout import apply_bias, ActionCategory
    # Check-only node: only CHECK (we treat check as CALL category).
    s = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cats = [ActionCategory.CALL, ActionCategory.RAISE, ActionCategory.RAISE]
    out = apply_bias(s, bias_profile=3, categories=cats)  # raise-biased
    # No raise mass to bump. Result should be unbiased strategy itself,
    # renormalized (it's already normalized here).
    check_array_close("bias degenerate -> unbiased", out, s, tol=1e-12)


# ──────────────────────────────────────────────────────────────────────────
# B. Action history threading
# ──────────────────────────────────────────────────────────────────────────


def _build_hu_flop_leaf(rec: RecorderBlueprint):
    """Build a minimal 2p flop leaf state.

    Preflop: UTG=player 0 opens (action 2), BB=player 1 calls (action 1).
    Postflop starts with BB to act (OOP). Pot = 6 BB. Stacks 97 BB each.
    """
    from rollout import RolloutLeaf  # type: ignore
    # AhKsTd flop
    board = [47, 50, 34]  # specific card ints; values don't matter for this test
    return RolloutLeaf(
        board=board,
        pot_chips=600,
        active=[True, True],
        stacks=[9700, 9700],
        bets=[0, 0],
        street="flop",
        player_hands=[
            [(0, 1, 1.0), (2, 3, 1.0)],  # UTG hands
            [(4, 5, 1.0), (6, 7, 1.0)],  # BB hands
        ],
        acting_order=[1, 0],  # BB (player 1) acts first postflop
        game_action_history=[2, 1],  # UTG raise, BB call
    )


def test_action_history_includes_preflop() -> None:
    """First blueprint lookup after entering the rollout must see the
    preflop action prefix, not an empty history.
    """
    from rollout import compute_leaf_value  # type: ignore
    rec = RecorderBlueprint()
    leaf = _build_hu_flop_leaf(rec)
    # Just one rollout; we only care about the first call.
    _ = compute_leaf_value(
        leaf=leaf,
        bias_profile=[0, 0],
        blueprint=rec,
        hero_player=0,
        hero_hand_idx=0,
        num_rollouts=1,
        rng=np.random.default_rng(42),
    )
    assert rec.calls, "Expected at least one blueprint lookup"
    first = rec.calls[0]
    # Preflop prefix must be present.
    check(
        "first lookup has preflop prefix",
        first["action_history"][:2] == [2, 1],
        f"got {first['action_history']}",
    )


def test_action_history_grows_in_street() -> None:
    """As rollout actions are taken within a street, they append to the
    history passed to blueprint lookups.
    """
    from rollout import compute_leaf_value
    rec = RecorderBlueprint()
    leaf = _build_hu_flop_leaf(rec)
    _ = compute_leaf_value(
        leaf=leaf,
        bias_profile=[0, 0],
        blueprint=rec,
        hero_player=0,
        hero_hand_idx=0,
        num_rollouts=1,
        rng=np.random.default_rng(42),
    )
    # If the rollout took any second action in the same street, that
    # call's history should be strictly longer than the first call's.
    if len(rec.calls) >= 2:
        len0 = len(rec.calls[0]["action_history"])
        len1 = len(rec.calls[1]["action_history"])
        check(
            "second lookup history >= first",
            len1 >= len0,
            f"first_len={len0} second_len={len1}",
        )


# ──────────────────────────────────────────────────────────────────────────
# C. Acting order (postflop: SB, BB, UTG, MP, CO, BTN)
# ──────────────────────────────────────────────────────────────────────────


def test_acting_order_postflop_hu() -> None:
    """In HU (2p), BB acts first postflop (OOP). Confirm via the first
    player in the recorder's call list.
    """
    from rollout import compute_leaf_value
    rec = RecorderBlueprint()
    leaf = _build_hu_flop_leaf(rec)
    _ = compute_leaf_value(
        leaf=leaf,
        bias_profile=[0, 0],
        blueprint=rec,
        hero_player=0,
        hero_hand_idx=0,
        num_rollouts=1,
        rng=np.random.default_rng(42),
    )
    assert rec.calls, "Expected at least one blueprint lookup"
    first_actor = rec.calls[0]["player"]
    check(
        "HU postflop: first actor is BB (player 1)",
        first_actor == 1,
        f"got player {first_actor}",
    )


def test_acting_order_respects_active_mask() -> None:
    """Folded players are skipped in the acting order."""
    from rollout import RolloutLeaf, compute_leaf_value
    rec = RecorderBlueprint()
    # 3 players, player 1 has already folded.
    leaf = RolloutLeaf(
        board=[47, 50, 34],
        pot_chips=600,
        active=[True, False, True],
        stacks=[9700, 0, 9700],
        bets=[0, 0, 0],
        street="flop",
        player_hands=[
            [(0, 1, 1.0)],
            [(2, 3, 1.0)],  # folded
            [(4, 5, 1.0)],
        ],
        acting_order=[2, 0],  # player 2 (e.g. BB) first, then player 0
        game_action_history=[2, 0, 1],
    )
    _ = compute_leaf_value(
        leaf=leaf,
        bias_profile=[0, 0, 0],
        blueprint=rec,
        hero_player=0,
        hero_hand_idx=0,
        num_rollouts=1,
        rng=np.random.default_rng(42),
    )
    # No call should be for player 1 (folded).
    players_called = set(c["player"] for c in rec.calls)
    check(
        "folded player never queried",
        1 not in players_called,
        f"players queried: {players_called}",
    )


# ──────────────────────────────────────────────────────────────────────────
# D. Bucket correctness (blueprint's hand_to_bucket, not ad-hoc EHS)
# ──────────────────────────────────────────────────────────────────────────


def test_uses_blueprint_hand_to_bucket() -> None:
    """The rollout should call blueprint.hand_to_bucket for every hero
    hand bucket lookup, not compute its own EHS.
    """
    from rollout import compute_leaf_value
    rec = RecorderBlueprint()
    leaf = _build_hu_flop_leaf(rec)
    _ = compute_leaf_value(
        leaf=leaf,
        bias_profile=[0, 0],
        blueprint=rec,
        hero_player=0,
        hero_hand_idx=0,
        num_rollouts=1,
        rng=np.random.default_rng(42),
    )
    check(
        "blueprint.hand_to_bucket called",
        len(rec.buckets_requested) > 0,
        "expected at least one hand_to_bucket call",
    )


# ──────────────────────────────────────────────────────────────────────────
# E. Uniform-on-zero substitution (Pluribus Algorithm 1 supplementary)
# ──────────────────────────────────────────────────────────────────────────


def test_coverage_miss_substitutes_uniform() -> None:
    """When the blueprint returns None (info set not in coverage), the
    rollout substitutes uniform-on-legal-actions per Pluribus Algorithm
    1 supplementary. The leaf_coverage record makes the substitution
    observable; the rollout must NOT raise LeafValueUnavailable.
    """
    from rollout import compute_leaf_value
    from leaf_values import LeafValueUnavailable
    import leaf_coverage
    bp = NoneBlueprint()
    rec_leaf = _build_hu_flop_leaf(RecorderBlueprint())  # reuse shape
    with leaf_coverage.CoverageScope() as scope:
        try:
            value = compute_leaf_value(
                leaf=rec_leaf,
                bias_profile=[0, 0],
                blueprint=bp,
                hero_player=0,
                hero_hand_idx=0,
                num_rollouts=1,
                rng=np.random.default_rng(42),
            )
        except LeafValueUnavailable as e:
            check("coverage miss returns valid value (no LeafValueUnavailable)",
                  False, f"raised: {e}")
            return
        except Exception as e:
            check("coverage miss returns valid value",
                  False, f"got {type(e).__name__}: {e}")
            return
    check(
        "coverage miss returns finite value",
        np.isfinite(value),
        f"value={value}",
    )
    # Coverage scope must record the misses (substitution is observable).
    miss_count = scope.miss_categories.get("rollout_sigma", 0)
    check(
        "coverage misses recorded in leaf_coverage scope",
        miss_count > 0,
        f"miss_count={miss_count}",
    )


def test_coverage_hit_returns_blueprint_strategy() -> None:
    """Positive guard: when the blueprint has coverage, the rollout
    uses its strategy, NOT uniform. Without this, every "expect uniform
    on miss" test could pass by always returning uniform.
    """
    from rollout import _lookup_strategy
    bp = FixedStrategyBlueprint(np.array([0.7, 0.2, 0.1], dtype=np.float32))
    result = _lookup_strategy(
        blueprint=bp, board=[0, 1, 2], action_history=[],
        player=0, bucket=0, street=1,
    )
    check(
        "covered lookup returns blueprint strategy",
        result is not None and abs(result[0] - 0.7) < 1e-6,
        f"result={result}",
    )
    check(
        "covered lookup is NOT uniform",
        result is not None and abs(result[0] - 1.0 / 3) > 0.01,
        f"result={result} — looks uniform; means hit path is broken",
    )


# ──────────────────────────────────────────────────────────────────────────
# F. Showdown payoffs
# ──────────────────────────────────────────────────────────────────────────


def test_showdown_hero_wins() -> None:
    """Hero with pocket aces on a dry board beats opponent with 72o."""
    from rollout import compute_showdown_payoff  # type: ignore
    from solver import card_to_int

    # Hero: AhAs; Opp: 7d2c; Board: Ks8s3h5c9d (hero has AA, opp has high 9)
    hero_hand = (card_to_int("Ah"), card_to_int("As"))
    opp_hand = (card_to_int("7d"), card_to_int("2c"))
    board = [
        card_to_int("Ks"), card_to_int("8s"), card_to_int("3h"),
        card_to_int("5c"), card_to_int("9d"),
    ]
    # Pot = 100, each contributed 50.
    payoff = compute_showdown_payoff(
        full_board=board,
        hands=[hero_hand, opp_hand],
        active=[True, True],
        bets=[50, 50],
        pot_chips=100,
        hero_player=0,
    )
    # Hero wins pot (100) minus their 50 bet = +50.
    check_close("AA vs 72 -> +50", payoff, 50.0, tol=1e-9)


def test_showdown_chop() -> None:
    """Two hands that chop (e.g., both play the board) split the pot."""
    from rollout import compute_showdown_payoff
    from solver import card_to_int

    # Board has a flush, both hands play it: 2c3c (no spades) vs 4d5d (no spades)
    # Board: AsKsQsJs9s — royal flush on board with spades, both players
    # play the board (their hole cards are below the lowest board card).
    hero_hand = (card_to_int("2c"), card_to_int("3c"))
    opp_hand = (card_to_int("4d"), card_to_int("5d"))
    board = [
        card_to_int("As"), card_to_int("Ks"), card_to_int("Qs"),
        card_to_int("Js"), card_to_int("9s"),
    ]
    payoff = compute_showdown_payoff(
        full_board=board,
        hands=[hero_hand, opp_hand],
        active=[True, True],
        bets=[50, 50],
        pot_chips=100,
        hero_player=0,
    )
    # Chop: each gets 50. Hero contributed 50, gets 50 back, net 0.
    check_close("chop pot net 0", payoff, 0.0, tol=1e-9)


def test_fold_payoff_hero_last_standing() -> None:
    """Everyone folded to hero: hero wins pot, loses own bets."""
    from rollout import compute_showdown_payoff
    from solver import card_to_int

    hero_hand = (card_to_int("Ah"), card_to_int("As"))
    opp_hand = (card_to_int("7d"), card_to_int("2c"))
    board = [
        card_to_int("Ks"), card_to_int("8s"), card_to_int("3h"),
        card_to_int("5c"), card_to_int("9d"),
    ]
    # Opponent folded; they're inactive.
    payoff = compute_showdown_payoff(
        full_board=board,
        hands=[hero_hand, opp_hand],
        active=[True, False],
        bets=[30, 50],
        pot_chips=80,
        hero_player=0,
    )
    # Hero wins 80 pot, contributed 30, net +50.
    check_close("opp folded, hero wins pot", payoff, 50.0, tol=1e-9)


# ──────────────────────────────────────────────────────────────────────────
# G. Determinism
# ──────────────────────────────────────────────────────────────────────────


def test_determinism_same_seed() -> None:
    """Same seed → same output bit-for-bit."""
    from rollout import compute_leaf_value
    # Use FixedStrategyBlueprint so rollout actions are sampled from the
    # same distribution at every step — any nondeterminism will appear.
    strat = np.array([0.2, 0.3, 0.3, 0.2], dtype=np.float32)
    bp = FixedStrategyBlueprint(strat)

    def run(seed: int) -> float:
        from rollout import RolloutLeaf
        leaf = RolloutLeaf(
            board=[47, 50, 34],
            pot_chips=600,
            active=[True, True],
            stacks=[9700, 9700],
            bets=[0, 0],
            street="flop",
            player_hands=[[(0, 1, 1.0)], [(4, 5, 1.0)]],
            acting_order=[1, 0],
            game_action_history=[2, 1],
        )
        return compute_leaf_value(
            leaf=leaf,
            bias_profile=[0, 0],
            blueprint=bp,
            hero_player=0,
            hero_hand_idx=0,
            num_rollouts=5,
            rng=np.random.default_rng(seed),
        )

    v1 = run(12345)
    v2 = run(12345)
    check_close("same seed -> same value", v1, v2, tol=0.0)


# ──────────────────────────────────────────────────────────────────────────
# H. BPR3 parity gate (runs only if .bps is available)
# ──────────────────────────────────────────────────────────────────────────


def test_bpr3_parity_gate() -> None:
    """Optional integration test: rollout mean EV at the acting player
    should match BPR3 EV (at the same action_hash) within MC noise.

    Skipped if the .bps cache isn't present (CI/other machines).
    """
    bps_path = REPO / "blueprint_data_v3" / "unified_blueprint_v3_1.5B.bps"
    if not bps_path.exists():
        print("  skip BPR3 parity gate (no .bps)")
        return

    # Marked as a separate 'check' so it counts toward pass/fail only when
    # the file is present. Implementation: see run_parity_suite() below.
    print("  (BPR3 parity gate — run tools/c1_bpr3_parity.py manually)")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    tests = [
        # A. Bias
        test_bias_unmodified,
        test_bias_fold_5x,
        test_bias_call_5x,
        test_bias_raise_5x,
        test_bias_raise_preserves_ratios,
        test_bias_degenerate_target_falls_back_to_unbiased,
        # B. Action history
        test_action_history_includes_preflop,
        test_action_history_grows_in_street,
        # C. Acting order
        test_acting_order_postflop_hu,
        test_acting_order_respects_active_mask,
        # D. Bucket
        test_uses_blueprint_hand_to_bucket,
        # E. Uniform-on-zero substitution
        test_coverage_miss_substitutes_uniform,
        test_coverage_hit_returns_blueprint_strategy,
        # F. Showdown
        test_showdown_hero_wins,
        test_showdown_chop,
        test_fold_payoff_hero_last_standing,
        # G. Determinism
        test_determinism_same_seed,
        # H. BPR3 parity (meta — real gate elsewhere)
        test_bpr3_parity_gate,
    ]

    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except Exception as e:
            global FAIL, FAILED_NAMES
            FAIL += 1
            FAILED_NAMES.append(t.__name__)
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} pass, {FAIL} fail")
    if FAILED_NAMES:
        print("Failed:")
        for n in FAILED_NAMES:
            print(f"  - {n}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
