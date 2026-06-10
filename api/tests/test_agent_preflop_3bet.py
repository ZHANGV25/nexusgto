"""3-bet HEADS-UP preflop construction + chips (Lane 2, capture-verified).

Ground truth: poker-solver scripts/preflop_shapes_captured.json (2026-06-10,
bp_generate_actions):
  * all 15 (opener, 3-bettor) flop-root keys present + trained, with
    wrong-seat / truncated / extended / out-of-vocab discriminators MISSING;
  * trainer chip walks — FLOAT32 arithmetic confirmed: the cold-3-bet node
    (int)(0.7f*700) == 490 in C while Python doubles give 489 (UTG-open
    lines: pot 1680/stacks 9235, NOT the double-math 1678/9236);
  * sigma-length walks: blueprint sigma length == trainer na at every node
    through the river all-in clamp (chips-discriminating: single-open chips
    would serve na=3 at the river faced node; the 3-bet chips serve na=2).

These tests pin the constructed prefixes and derived chips BIT-EXACT to the
captured tuples. Pure-local (no blueprint); the blueprint-as-oracle replay
lives in tests/test_agent_preflop_3bet_blueprint.py.
"""

from __future__ import annotations

import pytest

from app.agent.game_state import (
    CanonicalChipsUnsupported,
    canonical_flop_chips,
    preflop_raise_amount_f32,
)
from app.agent.preflop_constructor import (
    PreflopAction,
    PreflopConstructionUnsupported,
    build_preflop_prefix,
)

# ── Capture-verified ground truth (preflop_shapes_captured.json) ──────────
# pair -> (prefix, flop_pot, per-slot stacks SB-first, live seats SB-first)
CAPTURED = {
    ("UTG", "MP"): ([2, 2, 0, 0, 0, 0, 1], 1680, (9235, 9235), ["UTG", "MP"]),
    ("UTG", "CO"): ([2, 0, 2, 0, 0, 0, 1], 1680, (9235, 9235), ["UTG", "CO"]),
    ("UTG", "BTN"): ([2, 0, 0, 2, 0, 0, 1], 1680, (9235, 9235), ["UTG", "BTN"]),
    ("UTG", "SB"): ([2, 0, 0, 0, 2, 0, 1], 1560, (9270, 9270), ["SB", "UTG"]),
    ("UTG", "BB"): ([2, 0, 0, 0, 0, 2, 1], 1440, (9305, 9305), ["BB", "UTG"]),
    ("MP", "CO"): ([0, 2, 2, 0, 0, 0, 1], 1680, (9235, 9235), ["MP", "CO"]),
    ("MP", "BTN"): ([0, 2, 0, 2, 0, 0, 1], 1680, (9235, 9235), ["MP", "BTN"]),
    ("MP", "SB"): ([0, 2, 0, 0, 2, 0, 1], 1560, (9270, 9270), ["SB", "MP"]),
    ("MP", "BB"): ([0, 2, 0, 0, 0, 2, 1], 1440, (9305, 9305), ["BB", "MP"]),
    ("CO", "BTN"): ([0, 0, 2, 2, 0, 0, 1], 1680, (9235, 9235), ["CO", "BTN"]),
    ("CO", "SB"): ([0, 0, 2, 0, 2, 0, 1], 1560, (9270, 9270), ["SB", "CO"]),
    ("CO", "BB"): ([0, 0, 2, 0, 0, 2, 1], 1440, (9305, 9305), ["BB", "CO"]),
    ("BTN", "SB"): ([0, 0, 0, 2, 2, 0, 1], 1560, (9270, 9270), ["SB", "BTN"]),
    ("BTN", "BB"): ([0, 0, 0, 2, 0, 2, 1], 1440, (9305, 9305), ["BB", "BTN"]),
    ("SB", "BB"): ([0, 0, 0, 0, 2, 2, 1], 1152, (9424, 9424), ["SB", "BB"]),
}


def _threebet_line(opener: str, threebettor: str):
    return [
        PreflopAction(opener, "raise"),
        PreflopAction(threebettor, "raise"),
        PreflopAction(opener, "call"),
    ]


# ── Constructed prefixes == captured prefixes (bit-exact) ─────────────────


def test_threebet_prefix_btn_open_bb_3bet():
    # The probe's line A: flop root GOLDEN at player=1 (BB first to act).
    assert build_preflop_prefix(_threebet_line("BTN", "BB")) == [
        0, 0, 0, 2, 0, 2, 1,
    ]


def test_threebet_prefix_utg_open_btn_3bet():
    # The probe's line B (folds between AND after the 3-bettor).
    assert build_preflop_prefix(_threebet_line("UTG", "BTN")) == [
        2, 0, 0, 2, 0, 0, 1,
    ]


@pytest.mark.parametrize("pair", sorted(CAPTURED), ids=lambda p: f"{p[0]}-{p[1]}")
def test_threebet_prefix_all_15_pairs_match_capture(pair):
    prefix, _, _, _ = CAPTURED[pair]
    assert build_preflop_prefix(_threebet_line(*pair)) == prefix


def test_threebet_prefix_with_explicit_folds():
    # Explicit folds (chronological) must not change the emitted prefix.
    actions = [
        PreflopAction("UTG", "fold"),
        PreflopAction("MP", "fold"),
        PreflopAction("CO", "fold"),
        PreflopAction("BTN", "raise"),
        PreflopAction("SB", "fold"),
        PreflopAction("BB", "raise"),
        PreflopAction("BTN", "call"),
    ]
    assert build_preflop_prefix(actions) == [0, 0, 0, 2, 0, 2, 1]


# ── Chips: derived == trainer C ground truth (bit-exact) ──────────────────


@pytest.mark.parametrize("pair", sorted(CAPTURED), ids=lambda p: f"{p[0]}-{p[1]}")
def test_threebet_chips_all_15_pairs_match_trainer(pair):
    prefix, pot, stacks, live = CAPTURED[pair]
    got_pot, got_stacks = canonical_flop_chips(prefix, live)
    assert (got_pot, got_stacks) == (pot, stacks)


def test_threebet_chips_float32_divergent_cold_3bet():
    # THE float32 case: UTG opens to 275, BTN cold-3-bets. The trainer's
    # (int)(0.7f * 700) == 490 -> 3-bet raise-to 765 -> pot 1680. Python
    # DOUBLE arithmetic gives 489 -> 764 -> pot 1678 (silently wrong).
    pot, stacks = canonical_flop_chips([2, 0, 0, 2, 0, 0, 1], ["UTG", "BTN"])
    assert pot == 1680, "double-arithmetic regression (1678 = wrong chips)"
    assert stacks == (9235, 9235)


def test_preflop_raise_amount_f32_matches_trainer():
    # Open: to_call 100 into pot 150 -> 100 + (int)(0.7f*250)=175 -> 275.
    assert preflop_raise_amount_f32(0.7, pot=150, to_call=100, stack=10000) == 275
    # BB 3-bet vs BTN open: to_call 175, pot 425 -> 175 + 420 = 595.
    assert preflop_raise_amount_f32(0.7, pot=425, to_call=175, stack=9900) == 595
    # Cold 3-bet vs an open (the divergent case): to_call 275, pot 425 ->
    # float32 product 490 (doubles: 489).
    assert preflop_raise_amount_f32(0.7, pot=425, to_call=275, stack=10000) == 765
    # SB 3-bet vs BTN open (divergent too): to_call 225, pot 425 ->
    # float32 product 455 (doubles: 454).
    assert preflop_raise_amount_f32(0.7, pot=425, to_call=225, stack=9950) == 680
    # Stack clamp.
    assert preflop_raise_amount_f32(0.7, pot=425, to_call=275, stack=400) == 400


# ── Single-open regression: previously verified values unchanged ──────────


def test_single_open_chips_2p_regression():
    assert canonical_flop_chips([2, 0, 0, 0, 0, 1], ["BB", "UTG"]) == (
        600, (9725, 9725),
    )


def test_single_open_chips_3p_regression():
    assert canonical_flop_chips([2, 0, 0, 1, 0, 1], ["BB", "UTG", "BTN"]) == (
        875, (9725, 9725, 9725),
    )


# ── Unsupported shapes still fail LOUD ─────────────────────────────────────


def test_fourbet_line_unsupported():
    actions = [
        PreflopAction("UTG", "raise"),
        PreflopAction("BTN", "raise"),
        PreflopAction("UTG", "raise"),  # 4-bet
        PreflopAction("BTN", "call"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_squeeze_unsupported():
    # Open + cold-caller + 3-bet = squeeze shape, not capture-verified.
    actions = [
        PreflopAction("UTG", "raise"),
        PreflopAction("MP", "call"),
        PreflopAction("BB", "raise"),
        PreflopAction("UTG", "call"),
        PreflopAction("MP", "call"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_limp_reraise_unsupported():
    actions = [
        PreflopAction("UTG", "call"),   # limp
        PreflopAction("MP", "raise"),
        PreflopAction("UTG", "raise"),  # limp-reraise
        PreflopAction("MP", "call"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_opener_folds_to_3bet_unsupported():
    # No flop exists — no flop key to build.
    actions = [
        PreflopAction("UTG", "raise"),
        PreflopAction("BTN", "raise"),
        PreflopAction("UTG", "fold"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_threebettor_cannot_be_earlier_seat():
    # A re-raise from an earlier seat would be a limp-reraise line.
    actions = [
        PreflopAction("BTN", "raise"),
        PreflopAction("UTG", "raise"),
        PreflopAction("BTN", "call"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_threebet_actions_out_of_order_unsupported():
    # The opener's call listed BEFORE the 3-bet is not chronological.
    actions = [
        PreflopAction("BTN", "raise"),
        PreflopAction("BTN", "call"),
        PreflopAction("BB", "raise"),
    ]
    with pytest.raises(PreflopConstructionUnsupported):
        build_preflop_prefix(actions)


def test_chips_fourbet_prefix_unsupported():
    # Three raises in the prefix (4-bet war) — chips not capture-verified.
    with pytest.raises(CanonicalChipsUnsupported):
        canonical_flop_chips([2, 0, 0, 2, 0, 0, 2, 1], ["UTG", "BTN"])


def test_chips_non_smallest_raise_index_unsupported():
    # Index 3 = the 1.0x size: trained in the blueprint but not wired in v1.
    with pytest.raises(CanonicalChipsUnsupported):
        canonical_flop_chips([0, 0, 0, 2, 0, 3, 1], ["BB", "BTN"])


def test_chips_multiway_3bet_pot_unsupported():
    # Squeeze-ish: MP flats the open, BB 3-bets, both call -> 3 live players.
    with pytest.raises(CanonicalChipsUnsupported):
        canonical_flop_chips(
            [2, 1, 0, 0, 0, 2, 1, 1], ["UTG", "MP", "BB"],
        )


def test_chips_unclosed_3bet_prefix_unsupported():
    # Opener hasn't responded to the 3-bet — not a flop-entry prefix.
    with pytest.raises(CanonicalChipsUnsupported):
        canonical_flop_chips([0, 0, 0, 2, 0, 2], ["BB", "BTN"])


def test_chips_seats_mismatch_is_value_error():
    # Caller passes seats inconsistent with the replayed live players.
    with pytest.raises(ValueError):
        canonical_flop_chips([0, 0, 0, 2, 0, 2, 1], ["SB", "BTN"])
