"""HU limp-check preflop construction + chips (Lane C2, capture-verified).

Ground truth: poker-solver scripts/limp_shapes_captured.json (2026-06-11,
EC2 vs the real 8.05B + the literal trainer C function bp_generate_actions;
ALL PROBES CONSISTENT):
  * three HU limp-check lines verified end-to-end (flop roots GOLDEN with
    wrong-seat / truncated / extended+check / out-of-vocab discriminators
    MISSING; σ̄-length walks clean st1-st3):
      L1 UTG-limp     prefix [1,0,0,0,0,0]  flop pot 250  live BB,UTG
      L2 BTN-limp     prefix [0,0,0,1,0,0]  flop pot 250  live BB,BTN
      L3 SB-complete  prefix [0,0,0,0,1,0]  flop pot 200  live SB,BB
  * index semantics: LIMP = the call action index 1 at the facing-the-blind
    node; the BB's closing CHECK = index 0 at the BB-option node (trainer
    layout na=3 [check, 0.7x, 1.0x], check_idx=0 — NOT the facing-bet call
    index 1);
  * the 3-way limp lines (L4 UTG+BTN, L5 UTG+SB) are ROOT-CAPTURED ONLY:
    their trainer chips are pinned here but the CONSTRUCTOR rejects them
    (postflop walks unverified — rejected rather than served wrong).

These tests pin the constructed prefixes and derived chips BIT-EXACT to the
captured tuples. Pure-local (no blueprint); the blueprint-as-oracle replay
lives in tests/test_agent_preflop_limp_blueprint.py.
"""

from __future__ import annotations

import pytest

from app.engine.agent.game_state import (
    CanonicalChipsUnsupported,
    canonical_flop_chips,
)
from app.engine.agent.preflop_constructor import (
    PreflopAction,
    PreflopConstructionUnsupported,
    build_preflop_prefix,
)

# ── Capture-verified ground truth (limp_shapes_captured.json) ─────────────
# limper -> (prefix, flop_pot, per-slot stacks SB-first, live seats SB-first)
CAPTURED_LIMP = {
    "UTG": ([1, 0, 0, 0, 0, 0], 250, (9900, 9900), ["BB", "UTG"]),
    "BTN": ([0, 0, 0, 1, 0, 0], 250, (9900, 9900), ["BB", "BTN"]),
    "SB": ([0, 0, 0, 0, 1, 0], 200, (9900, 9900), ["SB", "BB"]),
}


def _limp_line(limper: str):
    return [
        PreflopAction(limper, "call"),
        PreflopAction("BB", "check"),
    ]


# ── Constructed prefixes == captured prefixes (bit-exact) ─────────────────


@pytest.mark.parametrize("limper", sorted(CAPTURED_LIMP))
def test_limp_prefix_matches_capture(limper):
    prefix, _, _, _ = CAPTURED_LIMP[limper]
    assert build_preflop_prefix(_limp_line(limper)) == prefix


def test_limp_prefix_with_explicit_folds():
    # Explicit folds (chronological) must not change the emitted prefix.
    actions = [
        PreflopAction("UTG", "call"),   # limp
        PreflopAction("MP", "fold"),
        PreflopAction("CO", "fold"),
        PreflopAction("BTN", "fold"),
        PreflopAction("SB", "fold"),
        PreflopAction("BB", "check"),
    ]
    assert build_preflop_prefix(actions) == [1, 0, 0, 0, 0, 0]


def test_sb_complete_prefix_with_explicit_folds():
    actions = [
        PreflopAction("UTG", "fold"),
        PreflopAction("MP", "fold"),
        PreflopAction("CO", "fold"),
        PreflopAction("BTN", "fold"),
        PreflopAction("SB", "call"),    # complete
        PreflopAction("BB", "check"),
    ]
    assert build_preflop_prefix(actions) == [0, 0, 0, 0, 1, 0]


# ── Chips: derived == trainer C ground truth (bit-exact) ──────────────────


@pytest.mark.parametrize("limper", sorted(CAPTURED_LIMP))
def test_limp_chips_match_trainer(limper):
    prefix, pot, stacks, live = CAPTURED_LIMP[limper]
    assert canonical_flop_chips(prefix, live) == (pot, stacks)


def test_threeway_limp_chips_pinned_but_serving_rejected():
    # L4 (UTG+BTN limp) and L5 (UTG limp + SB complete): trainer chips are
    # capture-pinned (root-captured walks), but the CONSTRUCTOR refuses the
    # lines — σ̄-length walks were never verified for 3-way limped pots.
    assert canonical_flop_chips(
        [1, 0, 0, 1, 0, 0], ["UTG", "BTN", "BB"],
    ) == (350, (9900, 9900, 9900))
    assert canonical_flop_chips(
        [1, 0, 0, 0, 1, 0], ["UTG", "SB", "BB"],
    ) == (300, (9900, 9900, 9900))


# ── Unsupported shapes still fail LOUD (constructor) ──────────────────────


def test_threeway_limp_line_unsupported():
    # L4: UTG limp + BTN limp + BB check — root-captured only, NOT served.
    actions = [
        PreflopAction("UTG", "call"),
        PreflopAction("BTN", "call"),
        PreflopAction("BB", "check"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_limp_plus_sb_complete_unsupported():
    # L5: UTG limp + SB complete + BB check — root-captured only.
    actions = [
        PreflopAction("UTG", "call"),
        PreflopAction("SB", "call"),
        PreflopAction("BB", "check"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


@pytest.mark.parametrize("limper", ["MP", "CO"])
def test_mp_co_limp_unsupported(limper):
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(_limp_line(limper))


def test_limp_raise_unsupported():
    # Limp-behind-then-raise (isolation raise over a limper): the line has a
    # raise AND a pre-raise call -> the single-open handler rejects the limp.
    actions = [
        PreflopAction("UTG", "call"),   # limp
        PreflopAction("BTN", "raise"),  # iso-raise
        PreflopAction("UTG", "call"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_limp_reraise_still_unsupported():
    actions = [
        PreflopAction("UTG", "call"),   # limp
        PreflopAction("MP", "raise"),
        PreflopAction("UTG", "raise"),  # limp-reraise
        PreflopAction("MP", "call"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_bb_raise_over_limper_unsupported():
    # BB raises the option instead of checking: a limp-raise pot.
    actions = [
        PreflopAction("UTG", "call"),
        PreflopAction("BB", "raise"),
        PreflopAction("UTG", "call"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_limp_without_bb_check_unsupported():
    # The line is UNCLOSED without the BB's explicit check.
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix([PreflopAction("UTG", "call")])


def test_bb_call_instead_of_check_unsupported():
    # The BB has nothing to call at the option node — must be "check".
    actions = [
        PreflopAction("UTG", "call"),
        PreflopAction("BB", "call"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_non_bb_check_unsupported():
    actions = [
        PreflopAction("UTG", "check"),
        PreflopAction("BB", "check"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_check_in_raised_line_unsupported():
    # A preflop check only exists as the BB's limped-pot option-closure.
    actions = [
        PreflopAction("UTG", "raise"),
        PreflopAction("BB", "check"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_limp_actions_out_of_order_unsupported():
    actions = [
        PreflopAction("BB", "check"),
        PreflopAction("UTG", "call"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_limp_duplicate_actor_unsupported():
    actions = [
        PreflopAction("UTG", "call"),
        PreflopAction("UTG", "call"),
        PreflopAction("BB", "check"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_bb_fold_in_limped_pot_unsupported():
    actions = [
        PreflopAction("UTG", "call"),
        PreflopAction("BB", "fold"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_threebet_size_idx_rejected_on_limp_line():
    with pytest.raises(ValueError):
        build_preflop_prefix(_limp_line("UTG"), threebet_size_idx=1)


# ── Unsupported prefixes still fail LOUD (chips layer) ────────────────────


def test_chips_limp_raise_prefix_unsupported():
    # Index 1 at the BB-option node = the 0.7x limp-RAISE — not verified.
    with pytest.raises(CanonicalChipsUnsupported):
        canonical_flop_chips([1, 0, 0, 0, 0, 1], ["BB", "UTG"])


def test_chips_truncated_limp_prefix_unsupported():
    # BB never acted — the limped pot is unclosed.
    with pytest.raises(CanonicalChipsUnsupported):
        canonical_flop_chips([1, 0, 0, 0, 0], ["BB", "UTG"])


def test_chips_overlong_limp_prefix_unsupported():
    # Trailing action past the closed round (the extended+check
    # discriminator MISSED on EC2 — an overlong prefix is a wrong key).
    with pytest.raises(CanonicalChipsUnsupported):
        canonical_flop_chips([1, 0, 0, 0, 0, 0, 0], ["BB", "UTG"])


def test_chips_fold_around_unsupported():
    # Everyone folds, the BB "checks" a walk — no flop entry exists.
    with pytest.raises(CanonicalChipsUnsupported):
        canonical_flop_chips([0, 0, 0, 0, 0, 0], ["BB"])


def test_chips_limp_seats_mismatch_is_value_error():
    with pytest.raises(ValueError):
        canonical_flop_chips([1, 0, 0, 0, 0, 0], ["BB", "BTN"])


# ── Existing shapes: chips regressions unchanged ──────────────────────────


def test_single_open_chips_regression_unchanged():
    assert canonical_flop_chips([2, 0, 0, 0, 0, 1], ["BB", "UTG"]) == (
        600, (9725, 9725),
    )


def test_threebet_chips_regression_unchanged():
    assert canonical_flop_chips([0, 0, 0, 2, 0, 2, 1], ["BB", "BTN"]) == (
        1440, (9305, 9305),
    )
