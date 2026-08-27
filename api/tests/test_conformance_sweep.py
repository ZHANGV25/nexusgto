"""Gap 6 + Gap 4(1) conformance-sweep tests (CONFORMANCE_SCOREBOARD).

Gap 4(1): the Ganzfried-Sandholm PSEUDO-HARMONIC off-tree boundary
    f_lo(x) = ((B - x)(1 + A)) / ((B - A)(1 + x)),  boundary (A+B+2AB)/(A+B+2)
replaces the abs/log-distance snap at both live call sites, kill-switched by
NEXUS_OFFTREE_SNAP=legacy. The 2026-07-13 merge review caught the first draft
shipping the plain HARMONIC (reciprocal) mapping — several fixtures below
exist specifically to keep the G&S form locked in. Gap 6: PLAY (final-
iterate) vs DISPLAY (sigma-bar) config split.
Spec: PLURIBUS_EXACT_CONFORMANCE_SPEC_2026-06-26.md.
"""

import pytest

from app.engine.agent.pseudoharmonic import (
    pseudoharmonic_map,
    pseudoharmonic_pick,
    use_pseudoharmonic_snap,
)
from app.engine.agent.action_translation import (
    OFF_TREE_PSEUDOHARMONIC,
    translate_action,
)
from app.engine.engine_constants import (
    DEFAULT_LIVE_CONFIG_BY_N,
    PLAY_LIVE_CONFIG_BY_N,
    live_config_for,
)


# ── pseudoharmonic_map / pick unit behavior ────────────────────────────────


def test_map_exact_match_single_pair():
    assert pseudoharmonic_map(0.75, [0.5, 0.75, 1.0]) == [(1, 1.0)]


def test_map_below_and_above_range_saturate():
    assert pseudoharmonic_map(0.2, [0.5, 1.0]) == [(0, 1.0)]
    assert pseudoharmonic_map(1.6, [0.5, 1.0]) == [(1, 1.0)]


def test_map_weights_are_ganzfried_sandholm():
    # x=0.72 in [0.5, 1.0]: f_lo = ((1-0.72)(1.5)) / ((0.5)(1.72)) = 0.48837
    mapping = dict(pseudoharmonic_map(0.72, [0.5, 1.0]))
    assert mapping[0] == pytest.approx(0.48837, abs=1e-4)
    assert mapping[1] == pytest.approx(0.51163, abs=1e-4)


def test_formula_is_pseudoharmonic_not_harmonic():
    # THE formula-regression fixture (merge review 2026-07-13): at x=0.70 on
    # [0.5, 1.0], G&S gives f_lo = (0.30*1.5)/(0.5*1.7) = 0.5294 -> pick LO,
    # while the plain harmonic mapping (the bug the review caught) gives
    # w_hi = (2 - 1/0.7)/(2 - 1) = 0.5714 -> pick HI. This test fails on the
    # harmonic form.
    assert pseudoharmonic_pick(0.70, [0.5, 1.0]) == 0


def test_pick_boundary_at_gs_threshold():
    # Boundary x* = (A+B+2AB)/(A+B+2) = 2.5/3.5 = 0.714285... Probe BOTH
    # sides with a margin — x* is 5/7, not float-representable, so an
    # exact-tie assertion is an ULP coin-flip (the first tests box failed
    # on exactly that: w_lo computed 0.49999999999999994). The tie->smaller
    # rule still holds for exact w equality (strict '>' in the pick).
    b = 2.5 / 3.5
    assert pseudoharmonic_pick(b - 1e-4, [0.5, 1.0]) == 0
    assert pseudoharmonic_pick(b + 1e-4, [0.5, 1.0]) == 1


def test_pick_differs_from_abs_distance_snap():
    # x=0.73: abs distance picks 0.5 (|0.73-0.5| = 0.23 < 0.27) but G&S
    # f_lo = (0.27*1.5)/(0.5*1.73) = 0.468 -> picks 1.0.
    assert pseudoharmonic_pick(0.73, [0.5, 1.0]) == 1


def test_pick_is_not_scale_invariant_chips_must_be_normalized():
    # The G&S "(1+x)" terms are pot-relative: feeding raw CHIPS uses a wrong
    # boundary (as A,B -> inf it degenerates to the harmonic mean). At
    # pot=1000: 700 chips on [500, 1000] raw -> HI, but the correct
    # pot-normalized 0.70 on [0.5, 1.0] -> LO. This is WHY the game_state
    # call site divides by the pot (merge review 2026-07-13).
    assert pseudoharmonic_pick(700.0, [500.0, 1000.0]) == 1
    assert pseudoharmonic_pick(0.70, [0.5, 1.0]) == 0


def test_pick_unsorted_tree_values_returns_input_position():
    # Survivor lists are not guaranteed sorted; positions must index the
    # INPUT list.
    assert pseudoharmonic_pick(0.73, [1.0, 0.5]) == 0


def test_pick_empty_fails_loud():
    with pytest.raises(ValueError):
        pseudoharmonic_pick(0.5, [])


# ── env kill-switch ────────────────────────────────────────────────────────


def test_snap_mode_default_on(monkeypatch):
    monkeypatch.delenv("NEXUS_OFFTREE_SNAP", raising=False)
    assert use_pseudoharmonic_snap() is True


def test_snap_mode_legacy_off(monkeypatch):
    monkeypatch.setenv("NEXUS_OFFTREE_SNAP", "legacy")
    assert use_pseudoharmonic_snap() is False


# ── translate_action wiring (site A) ───────────────────────────────────────


def test_translate_off_tree_uses_pseudoharmonic_boundary(monkeypatch):
    monkeypatch.delenv("NEXUS_OFFTREE_SNAP", raising=False)
    res = translate_action(
        "raise", amount=73.0, pot=100.0, bet_sizes=(0.5, 1.0),
        add_explicit_allin=False, facing_bet=False, raises_so_far=0,
    )
    assert res.kind == OFF_TREE_PSEUDOHARMONIC
    assert res.snapped_label == "Bet 100%"  # abs snap would say "Bet 50%"


def test_translate_off_tree_legacy_switch_restores_abs_snap(monkeypatch):
    monkeypatch.setenv("NEXUS_OFFTREE_SNAP", "legacy")
    res = translate_action(
        "raise", amount=73.0, pot=100.0, bet_sizes=(0.5, 1.0),
        add_explicit_allin=False, facing_bet=False, raises_so_far=0,
    )
    assert res.kind == OFF_TREE_PSEUDOHARMONIC  # provenance tag unchanged
    assert res.snapped_label == "Bet 50%"


def test_translate_huge_raise_keeps_legacy_allin_route(monkeypatch):
    # frac >= 5.0 must stay on the legacy label path (all-in detection in
    # _snap_amount_to_tree) — the pseudo-harmonic pick can only produce
    # bet-size labels and would mis-slot an explicit-allin vocab (merge
    # review 2026-07-13).
    monkeypatch.delenv("NEXUS_OFFTREE_SNAP", raising=False)
    res = translate_action(
        "raise", amount=600.0, pot=100.0, bet_sizes=(0.5, 1.0),
        add_explicit_allin=False, facing_bet=False, raises_so_far=0,
    )
    assert res.snapped_label == "All-in"


def test_translate_on_tree_unchanged(monkeypatch):
    monkeypatch.delenv("NEXUS_OFFTREE_SNAP", raising=False)
    res = translate_action(
        "raise", amount=50.0, pot=100.0, bet_sizes=(0.5, 1.0),
        add_explicit_allin=False, facing_bet=False, raises_so_far=0,
    )
    assert res.kind == "on_tree_exact"
    assert res.snapped_label == "Bet 50%"


# ── Gap 6: PLAY vs DISPLAY config split ────────────────────────────────────


def test_play_table_final_readout_every_n():
    for n, cfg in PLAY_LIVE_CONFIG_BY_N.items():
        assert cfg.readout == "final", f"{n}p play config must be final"


def test_display_table_sigma_bar_2p3p_has_per_hand_coverage_budget():
    # The /v1 frequency-display contract: 2p/3p stay sigma-bar.
    assert DEFAULT_LIVE_CONFIG_BY_N[2].readout == "avg"
    assert DEFAULT_LIVE_CONFIG_BY_N[3].readout == "avg"
    assert DEFAULT_LIVE_CONFIG_BY_N[2].iterations == 4_000
    assert DEFAULT_LIVE_CONFIG_BY_N[3].iterations == 4_000


def test_play_rows_differ_only_in_readout_and_note():
    for n in (2, 3):
        play, disp = PLAY_LIVE_CONFIG_BY_N[n], DEFAULT_LIVE_CONFIG_BY_N[n]
        assert (play.iterations, play.num_rollouts, play.vec_mode) == (
            disp.iterations, disp.num_rollouts, disp.vec_mode,
        )
    for n in (4, 5, 6):
        assert PLAY_LIVE_CONFIG_BY_N[n] is DEFAULT_LIVE_CONFIG_BY_N[n]


def test_live_config_for_purpose_routing():
    assert live_config_for(2).readout == "avg"
    assert live_config_for(2, purpose="play").readout == "final"
    assert live_config_for(6, purpose="play").readout == "final"
    with pytest.raises(ValueError):
        live_config_for(2, purpose="agent")
