"""TDD: action_vocab — metadata-driven legal-action counts (plan §5).

The 8.05B trained with postflop_bet_sizes=[0.75], so a no-bet node has 2
actions (check, bet), not the 3 rollout's (0.5,1.0) constant implies. These
pin the count for the 8.05B vocab and a 2-size vocab, mirroring training.
"""

from __future__ import annotations

from app.engine.agent.action_vocab import (
    DEFAULT_ADD_EXPLICIT_ALLIN,
    DEFAULT_POSTFLOP_BET_SIZES,
    DEFAULT_SUBSEQUENT_BET_SIZES,
    legal_categories,
    num_legal,
)

# 8.05B vocab.
P = DEFAULT_POSTFLOP_BET_SIZES      # (0.75,)
S = DEFAULT_SUBSEQUENT_BET_SIZES    # (1.0,)
A = DEFAULT_ADD_EXPLICIT_ALLIN      # False


def test_8_05b_no_bet_has_two_actions():
    # [check, bet0.75]
    assert num_legal(False, 0, P, S, A) == 2


def test_8_05b_facing_first_bet_has_three_actions():
    # [fold, call, raise(0.75 first-raise vocab)]
    assert num_legal(True, 0, P, S, A) == 3


def test_8_05b_facing_reraise_uses_subsequent_vocab():
    # raises_so_far>0 -> subsequent sizes (1.0,) -> [fold, call, raise] = 3
    assert num_legal(True, 1, P, S, A) == 3


def test_two_size_vocab_no_bet_has_three_actions():
    # v3-style [0.5, 1.0]: [check, bet0.5, bet1.0]
    assert num_legal(False, 0, (0.5, 1.0), (1.0,), False) == 3


def test_two_size_vocab_facing_bet_has_four_actions():
    # [fold, call, raise0.5, raise1.0]
    assert num_legal(True, 0, (0.5, 1.0), (1.0,), False) == 4


def test_explicit_allin_adds_a_slot():
    assert num_legal(False, 0, P, S, True) == 3  # check, bet, allin
    assert num_legal(True, 0, P, S, True) == 4   # fold, call, raise, allin


def test_tree_closed_at_max_raises():
    from app.engine.agent.action_vocab import POSTFLOP_MAX_RAISES
    # facing a bet at the raise cap -> [fold, call] only
    assert num_legal(True, POSTFLOP_MAX_RAISES, P, S, A) == 2
    # no bet at the cap -> [check] only
    assert num_legal(False, POSTFLOP_MAX_RAISES, P, S, A) == 1


def test_categories_have_correct_semantics():
    from engine.rollout import ActionCategory  # type: ignore[import-untyped]
    cats = legal_categories(True, 0, P, S, A)
    assert cats[0] == ActionCategory.FOLD
    assert cats[1] == ActionCategory.CALL
    assert cats[2] == ActionCategory.RAISE
    cats2 = legal_categories(False, 0, P, S, A)
    assert cats2[0] == ActionCategory.CALL  # check
    assert cats2[1] == ActionCategory.RAISE  # bet


# ── Chip-aware commitment prune (3a) ─────────────────────────────────────
#
# legal_categories_chips mirrors the TRAINER's generate_actions
# (mccfr_blueprint.c:938-973) BIT-EXACT: it prunes a raise slot when the
# raise-to amount can't exceed the call (ba <= to_call = committed) and dedups
# to a single all-in when ba >= stack. This is what makes the agent's vocab match
# the blueprint's stored σ̄ length on committed lines (no more crash). The
# legacy legal_categories is the deep-stack (no-prune) special case.

from app.engine.agent.action_vocab import (  # noqa: E402
    legal_categories_chips,
    num_legal_chips,
)

_DEEP = 10_000_000  # stack so deep no raise is ever pruned


def test_chips_vocab_deep_stack_equals_legacy_no_bet():
    # Deep stack, no bet (to_call=0): chip-aware == legacy == [check, bet].
    chips = legal_categories_chips(
        facing_bet=False, raises_so_far=0, pot=600, stack=_DEEP, to_call=0,
        postflop_bet_sizes=P, subsequent_bet_sizes=S, add_explicit_allin=A,
    )
    assert chips == legal_categories(False, 0, P, S, A)
    assert len(chips) == 2


def test_chips_vocab_deep_stack_equals_legacy_facing_bet():
    # Deep stack, facing a real bet: chip-aware == legacy == [fold, call, raise].
    chips = legal_categories_chips(
        facing_bet=True, raises_so_far=0, pot=600, stack=_DEEP, to_call=450,
        postflop_bet_sizes=P, subsequent_bet_sizes=S, add_explicit_allin=A,
    )
    assert chips == legal_categories(True, 0, P, S, A)
    assert len(chips) == 3


def test_commitment_drops_raise_facing_bet_sigma_is_fold_call():
    # σ̄=[0,1] case: facing a bet where the only raise would be <= the call
    # (committed) -> raise pruned -> [fold, call] (len 2). Concrete ints, by the
    # formula: ba = to_call + int(1.0*(pot+to_call)) with subsequent size 1.0,
    # but ba clamped to stack; if stack barely covers the call, ba<=to_call.
    # Use a short stack: stack=120, to_call=120 (all-in just to call). The raise
    # ba = 120 + int(1.0*(pot+120)); clamp ba>=stack(120) -> ba=120; 120<=120 ->
    # DROP. Result [fold, call].
    from engine.rollout import ActionCategory
    chips = legal_categories_chips(
        facing_bet=True, raises_so_far=1, pot=1380, stack=120, to_call=120,
        postflop_bet_sizes=P, subsequent_bet_sizes=S, add_explicit_allin=A,
    )
    assert [c for c in chips] == [ActionCategory.FOLD, ActionCategory.CALL]
    assert len(chips) == 2


def test_commitment_drops_bet_no_bet_node_sigma_is_check_only():
    # σ̄=[1.0] case: all-in actor (stack 0) at a no-bet node. to_call=0,
    # ba = int(0.75*pot) clamped to stack(0) -> 0; 0<=0 -> DROP -> [check] (len 1).
    from engine.rollout import ActionCategory
    chips = legal_categories_chips(
        facing_bet=False, raises_so_far=0, pot=30050, stack=0, to_call=0,
        postflop_bet_sizes=P, subsequent_bet_sizes=S, add_explicit_allin=A,
    )
    assert [c for c in chips] == [ActionCategory.CALL]  # check only
    assert len(chips) == 1


def test_allin_dedup_single_slot_when_ba_ge_stack():
    # Two bet sizes both >= stack -> a SINGLE all-in raise slot (deduped), not
    # two. v3-style [0.5,1.0] sizes, short stack so both clamp to all-in.
    from engine.rollout import ActionCategory
    chips = legal_categories_chips(
        facing_bet=False, raises_so_far=0, pot=10000, stack=500, to_call=0,
        postflop_bet_sizes=(0.5, 1.0), subsequent_bet_sizes=(1.0,),
        add_explicit_allin=False,
    )
    # check + ONE all-in (both 0.5*pot=5000 and 1.0*pot=10000 clamp to 500).
    assert chips == [ActionCategory.CALL, ActionCategory.RAISE]
    assert len(chips) == 2


def test_int_truncation_matches_c_not_round():
    # ba uses C-style int() truncation, not rounding. Pick pot so 0.75*pot has
    # a fraction that would round UP but truncates DOWN, and a to_call that sits
    # between the truncated and rounded value so the slot-survival flips.
    # pot=441, to_call=0 -> 0.75*441 = 330.75. int()=330 (truncate), round()=331.
    # Make stack large so no all-in clamp. Then ba=330 > to_call=0 -> survives.
    # The decisive case: to_call=330. Then for a FIRST raise (size 0.75):
    # ba = 330 + int(0.75*(pot+330)). With pot=1, ba=330+int(0.75*331)=330+248=578>330 survives.
    # Direct truncation check on the no-bet bet amount via the helper:
    from app.engine.agent.action_vocab import raise_to_amount
    assert raise_to_amount(0.75, pot=441, to_call=0, stack=_DEEP) == 330  # not 331
    # facing-bet form: ba = to_call + int(frac*(pot+to_call))
    assert raise_to_amount(0.75, pot=100, to_call=330, stack=_DEEP) == 330 + int(0.75 * 430)


def test_chips_add_explicit_allin_off_no_extra_slot():
    # With add_explicit_allin=False (8.05B), a deep-stack node gets no extra
    # all-in slot beyond the sized raises.
    chips = legal_categories_chips(
        facing_bet=False, raises_so_far=0, pot=600, stack=_DEEP, to_call=0,
        postflop_bet_sizes=P, subsequent_bet_sizes=S, add_explicit_allin=False,
    )
    assert len(chips) == 2  # check, bet — no extra allin


def test_chips_max_raises_cap_ignores_chips():
    from engine.rollout import ActionCategory
    # At the raise cap, chips are irrelevant: facing -> [fold,call], else [check].
    facing = legal_categories_chips(
        facing_bet=True, raises_so_far=3, pot=600, stack=_DEEP, to_call=450,
        postflop_bet_sizes=P, subsequent_bet_sizes=S, add_explicit_allin=A,
    )
    assert facing == [ActionCategory.FOLD, ActionCategory.CALL]
    nobet = legal_categories_chips(
        facing_bet=False, raises_so_far=3, pot=600, stack=_DEEP, to_call=0,
        postflop_bet_sizes=P, subsequent_bet_sizes=S, add_explicit_allin=A,
    )
    assert nobet == [ActionCategory.CALL]


def test_num_legal_chips_matches_categories_len():
    assert num_legal_chips(
        facing_bet=True, raises_so_far=0, pot=600, stack=_DEEP, to_call=450,
        postflop_bet_sizes=P, subsequent_bet_sizes=S, add_explicit_allin=A,
    ) == 3
