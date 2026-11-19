"""Pure tests for the off-tree add: event capture, planning, and row projection."""

from __future__ import annotations

import struct
from dataclasses import replace

import pytest

from app.engine.agent.game_state import GameState, new_postflop_state
from app.engine.agent.offtree_add import (
    FIRST_RAISE,
    SKIP_ADDED_SIZE_CAP,
    SKIP_ALL_IN_FAMILY,
    SKIP_DUPLICATE_ADDED,
    SKIP_DUPLICATE_BASE,
    SKIP_DUPLICATE_STEP,
    SKIP_INVALID_DENOMINATOR,
    SKIP_INVALID_STEP,
    SKIP_NON_POSITIVE,
    SKIP_UNSORTED_BASE_SIZES,
    SUBSEQUENT_RAISE,
    _builder_amount,
    _fraction,
    build_offtree_extension,
    merge_added_action_mass,
    offtree_add_enabled,
)


def _two_player_state() -> GameState:
    return new_postflop_state(
        seats=["BB", "UTG"],
        preflop_prefix=[2, 0, 0, 0, 0, 1],
        board=[10, 20, 30],
    )


def _three_player_state() -> GameState:
    return new_postflop_state(
        seats=["BB", "UTG", "BTN"],
        preflop_prefix=[2, 0, 0, 1, 0, 1],
        board=[10, 20, 30],
    )


def _synthetic_state(
    history: tuple[int, ...],
    events: tuple[tuple[int, int, int, int, int], ...],
) -> GameState:
    state = _two_player_state()
    return replace(
        state,
        postflop_history=history,
        street_history=history,
        street_offtree_raises=events,
    )


# Event-channel recording and lifecycle.


def test_offtree_genuine_raise_records_actor_view() -> None:
    state = _two_player_state()

    result = state.advance_observed("raise", amount=900)

    assert result.off_tree is True
    assert result.applied_amount == 900
    assert result.state.street_history == (1,)
    assert result.state.street_offtree_raises == ((0, 900, 600, 0, 9725),)
    assert state.street_offtree_raises == ()


def test_observed_raise_records_post_stack_clamp_increment() -> None:
    state = _two_player_state()

    result = state.advance_observed("raise", amount=20_000)

    assert result.applied_amount == 9725
    assert result.state.street_offtree_raises == (
        (0, 9725, 600, 0, 9725),
    )


def test_exact_on_tree_raise_does_not_record_event() -> None:
    state = _two_player_state()

    result = state.advance_observed("raise", amount=450)

    assert result.off_tree is False
    assert result.state.street_offtree_raises == ()


def test_under_raise_as_call_does_not_record_event() -> None:
    state = _three_player_state().advance(1)  # BB bets 656.

    result = state.advance_observed("raise", amount=500)

    assert result.off_tree is True
    assert result.action_idx == 1
    assert result.state.street_offtree_raises == ()


def test_raise_at_capped_node_does_not_record_event() -> None:
    state = _two_player_state()
    state = state.advance(1)  # BB bet.
    state = state.advance(2)  # UTG raise.
    state = state.advance(2)  # BB raise: cap reached.

    result = state.advance_observed("raise", amount=5000)

    assert result.off_tree is True
    assert result.action_idx == 1
    assert result.state.street_offtree_raises == ()


def test_ordinary_advance_carries_event_until_street_close() -> None:
    state = _three_player_state().advance_observed(
        "raise", amount=900,
    ).state
    event_channel = state.street_offtree_raises

    state = state.advance(1)  # UTG calls; BTN still owes action.

    assert state.street_offtree_raises == event_channel
    assert state.street == 1


def test_street_close_resets_event_channel_with_street_history() -> None:
    state = _three_player_state().advance_observed(
        "raise", amount=900,
    ).state
    state = state.advance(1)  # UTG calls.
    state = state.advance(1)  # BTN calls; flop closes.

    assert state.street == 2
    assert state.street_history == ()
    assert state.street_offtree_raises == ()


# Planning and disclosure.


def test_offtree_add_env_gate_is_read_per_call(monkeypatch) -> None:
    monkeypatch.delenv("NEXUS_OFFTREE_ADD", raising=False)
    assert offtree_add_enabled() is False

    for truthy in ("1", "true", "True", "yes", "Yes"):
        monkeypatch.setenv("NEXUS_OFFTREE_ADD", truthy)
        assert offtree_add_enabled() is True

    monkeypatch.setenv("NEXUS_OFFTREE_ADD", "0")
    assert offtree_add_enabled() is False


def test_zero_events_returns_none() -> None:
    assert build_offtree_extension(
        _two_player_state(), (0.75,), (1.0,),
    ) is None


def test_unfaced_event_derives_first_raise_fraction() -> None:
    state = _two_player_state().advance_observed(
        "raise", amount=900,
    ).state

    analysis = build_offtree_extension(state, (0.75,), (1.0,))
    expected_fraction = (900.0 + 0.5) / 600.0

    assert analysis is not None
    assert analysis.plan is not None
    assert analysis.events[0].raise_level == FIRST_RAISE
    assert analysis.events[0].stack_before == 9725
    assert analysis.events[0].fraction == pytest.approx(expected_fraction)
    assert analysis.plan.added_first == pytest.approx((expected_fraction,))
    assert analysis.plan.added_subsequent == ()
    assert analysis.plan.bet_sizes == pytest.approx((0.75, expected_fraction))
    assert set(analysis.plan.event_fractions) == {0}
    assert analysis.plan.event_fractions[0] == pytest.approx(expected_fraction)


def test_faced_event_derives_subsequent_raise_fraction() -> None:
    state = _two_player_state().advance(1)  # pot=1050, to_call=450.
    state = state.advance_observed("raise", amount=1500).state

    analysis = build_offtree_extension(state, (0.75,), (1.0,))
    expected_fraction = (1500.0 - 450.0 + 0.5) / (1050.0 + 450.0)

    assert analysis is not None
    assert analysis.plan is not None
    event = analysis.events[0]
    assert (event.step, event.pot_before, event.to_call_before) == (1, 1050, 450)
    assert event.raise_level == SUBSEQUENT_RAISE
    assert event.fraction == pytest.approx(expected_fraction)
    assert analysis.plan.added_first == ()
    assert analysis.plan.added_subsequent == pytest.approx((expected_fraction,))
    assert analysis.plan.subsequent_bet_sizes == pytest.approx(
        (expected_fraction, 1.0)
    )


def test_one_chip_from_base_child_is_skipped_and_disclosed() -> None:
    state = _synthetic_state((1,), ((0, 751, 1000, 0, 10_000),))

    analysis = build_offtree_extension(state, (0.75,), (1.0,))

    assert analysis is not None
    assert analysis.plan is None
    assert [skip.reason for skip in analysis.skipped] == [SKIP_DUPLICATE_BASE]


def test_added_child_chip_duplicate_is_skipped() -> None:
    accepted_fraction = (1228.0 + 0.5) / 1024.0
    state = _synthetic_state(
        (1, 2, 2),
        (
            (1, 1228, 1024, 0, 10_000),
            (2, 2457, 2048, 0, 10_000),
        ),
    )

    analysis = build_offtree_extension(state, (0.75,), (1.0,))

    assert analysis is not None
    assert analysis.plan is not None
    assert analysis.plan.added_subsequent == pytest.approx((accepted_fraction,))
    assert analysis.plan.event_fractions == pytest.approx({
        1: accepted_fraction,
        2: accepted_fraction,
    })
    assert [skip.reason for skip in analysis.skipped] == [SKIP_DUPLICATE_ADDED]


def test_added_child_three_chip_difference_is_not_deduped() -> None:
    state = _synthetic_state(
        (1, 2, 2),
        (
            (1, 1228, 1024, 0, 10_000),
            (2, 2460, 2048, 0, 10_000),
        ),
    )

    analysis = build_offtree_extension(state, (0.75,), (1.0,))

    assert analysis is not None
    assert analysis.plan is not None
    assert len(analysis.plan.added_subsequent) == 2
    assert analysis.skipped == ()


def test_per_list_cap_keeps_first_three_observed() -> None:
    state = _synthetic_state(
        (1, 2, 2, 2, 2),
        (
            (1, 1400, 1000, 0, 10_000),
            (2, 1200, 1000, 0, 10_000),
            (3, 1300, 1000, 0, 10_000),
            (4, 1500, 1000, 0, 10_000),
        ),
    )

    analysis = build_offtree_extension(state, (0.75,), (1.0,))

    assert analysis is not None
    assert analysis.plan is not None
    assert analysis.plan.added_subsequent == pytest.approx(
        (1.2005, 1.3005, 1.4005)
    )
    assert set(analysis.plan.event_fractions) == {1, 2, 3}
    assert [skip.reason for skip in analysis.skipped] == [SKIP_ADDED_SIZE_CAP]


def test_allin_and_nonpositive_events_return_analysis_without_plan() -> None:
    state = _synthetic_state(
        (1, 2),
        (
            (0, 5000, 1000, 0, 10_000),
            (1, 999, 1000, 1000, 10_000),
        ),
    )

    analysis = build_offtree_extension(state, (0.75,), (1.0,))

    assert analysis is not None
    assert analysis.plan is None
    assert [skip.reason for skip in analysis.skipped] == [
        SKIP_ALL_IN_FAMILY,
        SKIP_NON_POSITIVE,
    ]
    assert len(analysis.events) == 2


def test_stack_shove_is_allin_family_at_small_pot_fraction() -> None:
    state = _synthetic_state(
        (1,),
        ((0, 900, 1000, 0, 900),),
    )

    analysis = build_offtree_extension(state, (0.75,), (1.0,))

    assert analysis is not None
    assert analysis.events[0].fraction < 1.0
    assert analysis.plan is None
    assert [skip.reason for skip in analysis.skipped] == [SKIP_ALL_IN_FAMILY]


def test_base_dedup_applies_observed_stack_clamp() -> None:
    state = _synthetic_state(
        (1,),
        ((0, 999, 2000, 0, 1000),),
    )

    analysis = build_offtree_extension(state, (1.0,), (1.0,))

    assert analysis is not None
    assert analysis.plan is None
    assert [skip.reason for skip in analysis.skipped] == [SKIP_DUPLICATE_BASE]


def test_invalid_event_step_is_skipped_and_disclosed() -> None:
    state = _synthetic_state(
        (1,),
        ((1, 900, 600, 0, 10_000),),
    )

    analysis = build_offtree_extension(state, (0.75,), (1.0,))

    assert analysis is not None
    assert analysis.plan is None
    assert [skip.reason for skip in analysis.skipped] == [SKIP_INVALID_STEP]


def test_duplicate_event_step_is_skipped_and_disclosed() -> None:
    state = _synthetic_state(
        (1,),
        (
            (0, 900, 600, 0, 10_000),
            (0, 1000, 600, 0, 10_000),
        ),
    )

    analysis = build_offtree_extension(state, (0.75,), (1.0,))

    assert analysis is not None
    assert analysis.plan is not None
    assert [skip.reason for skip in analysis.skipped] == [SKIP_DUPLICATE_STEP]


def test_invalid_observation_denominator_is_skipped_and_disclosed() -> None:
    state = _synthetic_state(
        (1,),
        ((0, 900, 0, 0, 10_000),),
    )

    analysis = build_offtree_extension(state, (0.75,), (1.0,))

    assert analysis is not None
    assert analysis.plan is None
    assert [skip.reason for skip in analysis.skipped] == [
        SKIP_INVALID_DENOMINATOR
    ]


@pytest.mark.parametrize(
    ("bet_sizes", "subsequent_bet_sizes"),
    (
        ((1.0, 0.75), (1.0,)),
        ((0.75,), (1.25, 1.0)),
    ),
)
def test_unsorted_base_lists_fail_safe_with_disclosure(
    bet_sizes,
    subsequent_bet_sizes,
) -> None:
    state = _synthetic_state(
        (1,),
        ((0, 900, 600, 0, 10_000),),
    )

    analysis = build_offtree_extension(
        state, bet_sizes, subsequent_bet_sizes,
    )

    assert analysis is not None
    assert analysis.plan is None
    assert [skip.reason for skip in analysis.skipped] == [
        SKIP_UNSORTED_BASE_SIZES
    ]


def test_margin_fraction_survives_float32_0_7_times_700_seam() -> None:
    def _float32(value: float) -> float:
        return struct.unpack("!f", struct.pack("!f", value))[0]

    def _float32_product(left: float, right: float) -> float:
        return _float32(_float32(left) * _float32(right))

    assert int(0.7 * 700.0) == 489
    assert int(_float32_product(0.7, 700.0)) == 490

    fraction, valid = _fraction(
        increment=590,
        pot_before=600,
        to_call_before=100,
    )

    assert valid is True
    assert fraction == pytest.approx(490.5 / 700.0)
    assert _builder_amount(fraction, 600, 100) == 590
    assert 100 + int(_float32_product(fraction, 700.0)) == 590


# Per-row merge back to the base label universe.


def test_merge_added_mass_uses_pseudoharmonic_nearest_base_raise() -> None:
    result = merge_added_action_mass(
        (0.1, 0.2, 0.3),
        (0.5, 1.0),
        (1, 2),
        ((0.7, 0.15), (1.3, 0.25)),
    )

    assert result.masses == pytest.approx((0.1, 0.35, 0.55))
    assert result.merged_mass == pytest.approx(0.4)
    assert result.dropped_mass == 0.0
    assert result.dropped is False
    assert result.drop_reason is None


def test_merge_is_deterministic_at_pseudoharmonic_lower_choice() -> None:
    args = (
        (0.2, 0.3, 0.1),
        (0.5, 1.0),
        (1, 2),
        ((0.7, 0.4),),
    )

    first = merge_added_action_mass(*args)
    second = merge_added_action_mass(*args)

    assert first == second
    assert first.masses == pytest.approx((0.2, 0.7, 0.1))


def test_no_base_raise_drops_added_mass_and_renormalizes() -> None:
    result = merge_added_action_mass(
        (0.2, 0.3),
        (),
        (),
        ((0.75, 0.5),),
    )

    assert result.masses == pytest.approx((0.4, 0.6))
    assert result.merged_mass == 0.0
    assert result.dropped_mass == pytest.approx(0.5)
    assert result.dropped is True
    assert result.drop_reason == "no_base_raise_child"


def test_no_remaining_base_mass_uses_deterministic_uniform_mix() -> None:
    result = merge_added_action_mass(
        (0.0, 0.0),
        (),
        (),
        ((0.75, 1.0),),
    )

    assert result.masses == pytest.approx((0.5, 0.5))
    assert result.dropped_mass == 1.0


def test_zero_added_mass_is_not_disclosed_as_dropped() -> None:
    result = merge_added_action_mass(
        (0.4, 0.6),
        (),
        (),
        ((0.75, 0.0),),
    )

    assert result.masses == pytest.approx((0.4, 0.6))
    assert result.merged_mass == 0.0
    assert result.dropped_mass == 0.0
    assert result.dropped is False
    assert result.drop_reason is None


def test_merge_rejects_raise_position_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        merge_added_action_mass(
            (0.5, 0.5),
            (0.75,),
            (),
            ((1.5, 0.1),),
        )


# LiveResolveSource seam, mocked below the solve boundary (no C library).


class _AnyRow(dict):
    def __init__(self, labels):
        super().__init__()
        self._labels = labels

    def get(self, key, default=None):
        return self._labels


class _CaptureEngine:
    def __init__(
        self,
        *,
        unsafe_first: bool = False,
        replay_unsupported: bool = False,
        zero_mass_drop: bool = False,
        positive_mass_drop: bool = False,
    ):
        self.calls = []
        self.unsafe_first = unsafe_first
        self.replay_unsupported = replay_unsupported
        self.zero_mass_drop = zero_mass_drop
        self.positive_mass_drop = positive_mass_drop
        self._last_solve_stats = {}

    def solve(self, **kwargs):
        from app.engine.cpu_es_solver import (
            OffTreeExtensionUnsafe,
            ReplayUnsupported,
        )

        self.calls.append(kwargs)
        is_extended = "offtree" in kwargs["hero_replay"]
        if is_extended and self.unsafe_first:
            self.unsafe_first = False
            raise OffTreeExtensionUnsafe("synthetic child-width overflow")
        if is_extended and self.replay_unsupported:
            raise ReplayUnsupported("synthetic amount-match failure")
        hero_idx = int(kwargs["hero_replay"]["player_idx"])
        rows = range(len(kwargs["player_ranges"][hero_idx]))
        self._last_solve_stats = {
            "offtree_replayed_through": is_extended,
            "offtree_hero_merged_mass_by_row": {
                row: (
                    0.0
                    if self.zero_mass_drop or self.positive_mass_drop
                    else (row + 1) / 1000.0
                )
                for row in rows
            } if is_extended else {},
            "offtree_hero_dropped_mass_by_row": {
                row: (0.2 if self.positive_mass_drop else 0.0)
                for row in rows
            } if is_extended and (
                self.zero_mass_drop or self.positive_mass_drop
            ) else {},
            "offtree_hero_drop_reason_by_row": {
                row: "no_base_raise_child" for row in rows
            } if is_extended and (
                self.zero_mass_drop or self.positive_mass_drop
            ) else {},
            "offtree_added_child_chips": {0: 900} if is_extended else {},
        }
        if int(kwargs["hero_replay"]["expected_actions"]) == 2:
            labels = {"Fold": 0.4, "Call": 0.6}
        else:
            labels = {"Fold": 0.2, "Call": 0.5, "Raise 100%": 0.3}
        return {0: _AnyRow(labels), 1: _AnyRow(labels)}


def _offtree_source(engine, enabled=True):
    from app.engine.agent.modes import ModeConfig
    from app.engine.agent.strategy_source import LiveResolveSource

    return LiveResolveSource(
        ModeConfig(mode="LIVE_RESOLVE"),
        cpu_es_solver=engine,
        seed_ranges=False,
        offtree_add=enabled,
    )


@pytest.mark.parametrize("grid_k", ["1", None])
def test_live_source_extends_solve_and_discloses_served_row(
    monkeypatch, grid_k,
) -> None:
    from app.engine.agent.live_resolve_adapter import hero_grid_row

    # K=1 serves the hero_grid_row class row; the default grid serves the
    # hero's exact combo row. The disclosure reports the served row.
    if grid_k is None:
        monkeypatch.delenv("NEXUS_GRID_K", raising=False)
    else:
        monkeypatch.setenv("NEXUS_GRID_K", grid_k)
    engine = _CaptureEngine()
    source = _offtree_source(engine)
    state = _two_player_state().advance_observed(
        "raise", amount=900,
    ).state
    expected_fraction = (900.0 + 0.5) / 600.0

    mix = source.get_mix(state, (40, 36))

    call = engine.calls[0]
    assert call["bet_sizes"] == pytest.approx([0.75, expected_fraction])
    assert call["subsequent_bet_sizes"] == [1.0]
    assert call["hero_replay"]["offtree"] == {
        "base_bet_sizes": [0.75],
        "base_subsequent_bet_sizes": [1.0],
        "event_fractions": {0: expected_fraction},
    }
    assert set(mix.freqs) == set(range(state.num_legal_actions()))
    if grid_k == "1":
        served_row = hero_grid_row(
            call["player_ranges"], call["hero_replay"]["player_idx"], (40, 36)
        )
    else:
        assert source.last_hero_row_exact is True
        served_row = call["hero_actual_hand"]["hand_idx"]
    disclosure = source.last_offtree_add
    assert disclosure["replayed_through_added"] is True
    assert disclosure["hero_merged_mass"] == pytest.approx(
        (served_row + 1) / 1000.0
    )
    assert disclosure["dropped"] is False


def test_live_source_unsafe_extension_resolves_once_with_base_vocab() -> None:
    engine = _CaptureEngine(unsafe_first=True)
    source = _offtree_source(engine)
    state = _two_player_state().advance_observed(
        "raise", amount=900,
    ).state

    source.get_mix(state, (40, 36))

    assert len(engine.calls) == 2
    assert "offtree" in engine.calls[0]["hero_replay"]
    assert "offtree" not in engine.calls[1]["hero_replay"]
    assert engine.calls[1]["bet_sizes"] == [0.75]
    assert source.last_offtree_add["dropped"] is True
    assert "synthetic child-width overflow" in (
        source.last_offtree_add["drop_reason"]
    )


def test_live_source_replay_fallback_failure_is_disclosed() -> None:
    from app.engine.agent.live_resolve_adapter import LiveResolveUnsupported

    engine = _CaptureEngine(replay_unsupported=True)
    source = _offtree_source(engine)
    state = _two_player_state().advance_observed(
        "raise", amount=900,
    ).state

    with pytest.raises(LiveResolveUnsupported, match="amount-match failure"):
        source.get_mix(state, (40, 36))

    assert len(engine.calls) == 1
    assert source.last_offtree_add["dropped"] is True
    assert source.last_offtree_add["drop_reason"] == (
        "synthetic amount-match failure"
    )


def test_live_source_does_not_disclose_zero_mass_as_dropped() -> None:
    engine = _CaptureEngine(zero_mass_drop=True)
    source = _offtree_source(engine)
    state = _two_player_state().advance_observed(
        "raise", amount=900,
    ).state

    source.get_mix(state, (40, 36))

    assert source.last_offtree_add["hero_merged_mass"] == 0.0
    assert source.last_offtree_add["dropped"] is False
    assert source.last_offtree_add["drop_reason"] is None


def test_live_source_discloses_positive_structural_drop() -> None:
    engine = _CaptureEngine(positive_mass_drop=True)
    source = _offtree_source(engine)
    state = _two_player_state().advance_observed(
        "raise", amount=900,
    ).state

    source.get_mix(state, (40, 36))

    assert source.last_offtree_add["hero_merged_mass"] == 0.0
    assert source.last_offtree_add["dropped"] is True
    assert source.last_offtree_add["drop_reason"] == "no_base_raise_child"


def test_live_source_skip_only_analysis_uses_plain_solve_and_discloses() -> None:
    engine = _CaptureEngine()
    source = _offtree_source(engine)
    base = _two_player_state()
    state = base.advance_observed(
        "raise", amount=base.stacks[base.to_act_slot],
    ).state

    source.get_mix(state, (40, 36))

    assert len(engine.calls) == 1
    assert "offtree" not in engine.calls[0]["hero_replay"]
    disclosure = source.last_offtree_add
    assert disclosure["added_first"] == []
    assert disclosure["skipped"][0]["reason"] == SKIP_ALL_IN_FAMILY
    assert disclosure["dropped"] is False


def test_live_source_flag_off_keeps_disclosure_none_and_plain_solve() -> None:
    engine = _CaptureEngine()
    source = _offtree_source(engine, enabled=False)
    state = _two_player_state().advance_observed(
        "raise", amount=900,
    ).state

    source.get_mix(state, (40, 36))

    assert source.last_offtree_add is None
    assert "offtree" not in engine.calls[0]["hero_replay"]
