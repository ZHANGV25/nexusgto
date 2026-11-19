"""Pure tests for trainer-generic preflop play and street-0 lookup."""

from __future__ import annotations

import os

import pytest

from app.engine.agent.agent import PokerAgent
from app.engine.agent.game_state import (
    CanonicalChipsUnsupported,
    canonical_flop_chips,
)
from app.engine.agent.modes import ModeConfig
from app.engine.agent.preflop_play import (
    FULL_SEAT,
    PreflopError,
    PreflopReplayError,
    PreflopRules,
    SHIPPED_PREFLOP_RULES,
    advance_preflop,
    new_preflop_state,
    play_preflop,
    replay_preflop,
    settle_preflop_terminal,
)
from app.engine.agent.preflop_range import hand_class_169
from app.engine.agent.strategy_source import (
    ActionMix,
    LookupSource,
    StrategySource,
)


FIVE_WAY_MP = [0, 2, 1, 1, 1, 1]
HU_THREEBET = [0, 0, 0, 2, 0, 2, 1]
SQUEEZE_THREE_WAY = [2, 1, 0, 0, 0, 2, 1, 1]
FOURBET_HU = [2, 0, 0, 2, 0, 0, 2, 1]
LIMP_CHECK = [1, 0, 0, 0, 0, 0]
LIMP_ISOLATION = [1, 0, 2, 0, 0, 0, 1]


def _six_hands():
    return tuple((2 * seat, 2 * seat + 1) for seat in range(6))


def test_known_mp_open_five_way_exact_chips_and_canonical_regression():
    state = replay_preflop(FIVE_WAY_MP, require_terminal=True)
    assert state.flop_ready
    assert state.pot == 1375
    assert state.live_seat_names == ("SB", "BB", "MP", "CO", "BTN")
    assert state.live_stacks == (9725,) * 5

    seats = ["MP", "CO", "BTN", "SB", "BB"]
    # The legacy capture-limited path stays byte-for-byte compatible.
    assert canonical_flop_chips(FIVE_WAY_MP, seats) == (1375, (9725,) * 5)
    # The opt-in generic validator reaches the same trainer result.
    assert canonical_flop_chips(
        FIVE_WAY_MP, seats, rules=SHIPPED_PREFLOP_RULES
    ) == (1375, (9725,) * 5)


def test_fold_around_is_terminal_and_settles_blinds():
    state = replay_preflop([0, 0, 0, 0, 0], require_terminal=True)
    assert state.uncontested
    assert not state.flop_ready
    assert state.live_seats == (FULL_SEAT["BB"],)
    assert state.pot == 150
    assert state.contributions == (50, 100, 0, 0, 0, 0)
    assert (state.small_blind, state.big_blind, state.initial_stack) == (
        50,
        100,
        10_000,
    )

    result = settle_preflop_terminal(state, _six_hands())
    assert result.awards == (0, 150, 0, 0, 0, 0)
    assert result.deltas == (-50, 50, 0, 0, 0, 0)
    assert sum(result.deltas) == 0


def test_one_point_zero_open_is_supported_only_by_opt_in_generic_replay():
    prefix = [3, 0, 0, 0, 0, 1]
    state = replay_preflop(prefix, require_terminal=True)
    assert state.pot == 750
    assert state.live_seat_names == ("BB", "UTG")
    assert state.live_stacks == (9650, 9650)
    with pytest.raises(CanonicalChipsUnsupported):
        canonical_flop_chips(prefix, ["BB", "UTG"])
    assert canonical_flop_chips(
        prefix, ["BB", "UTG"], rules=SHIPPED_PREFLOP_RULES
    ) == (750, (9650, 9650))


def test_hu_threebet_matches_existing_capture_pinned_result():
    state = replay_preflop(HU_THREEBET, require_terminal=True)
    assert state.pot == 1440
    assert state.live_seat_names == ("BB", "BTN")
    assert state.live_stacks == (9305, 9305)
    assert canonical_flop_chips(HU_THREEBET, ["BB", "BTN"]) == (
        1440,
        (9305, 9305),
    )
    assert canonical_flop_chips(
        HU_THREEBET, ["BB", "BTN"], rules=SHIPPED_PREFLOP_RULES
    ) == (1440, (9305, 9305))


def test_squeeze_multiway_threebet_generic_replay_exact_chips():
    state = replay_preflop(SQUEEZE_THREE_WAY, require_terminal=True)
    assert state.pot == 2711
    assert state.live_seat_names == ("BB", "UTG", "MP")
    assert state.live_stacks == (9113, 9113, 9113)
    with pytest.raises(CanonicalChipsUnsupported):
        canonical_flop_chips(
            SQUEEZE_THREE_WAY, ["UTG", "MP", "BB"]
        )
    assert canonical_flop_chips(
        SQUEEZE_THREE_WAY,
        ["UTG", "MP", "BB"],
        rules=SHIPPED_PREFLOP_RULES,
    ) == (2711, (9113, 9113, 9113))


def test_fourbet_uses_last_tier_then_caps_further_raises():
    before_call = replay_preflop(FOURBET_HU[:-1])
    assert before_call.num_raises == 3
    assert before_call.to_act_seat() == FULL_SEAT["BTN"]
    assert [action.kind for action in before_call.legal_actions()] == [
        "fold",
        "call",
    ]

    state = advance_preflop(before_call, FOURBET_HU[-1])
    assert state.flop_ready
    assert state.pot == 5040
    assert state.live_seat_names == ("UTG", "BTN")
    assert state.live_stacks == (7555, 7555)
    with pytest.raises(CanonicalChipsUnsupported):
        canonical_flop_chips(FOURBET_HU, ["UTG", "BTN"])
    assert canonical_flop_chips(
        FOURBET_HU, ["UTG", "BTN"], rules=SHIPPED_PREFLOP_RULES
    ) == (5040, (7555, 7555))


def test_limp_check_matches_existing_canonical_result():
    state = replay_preflop(LIMP_CHECK, require_terminal=True)
    assert state.pot == 250
    assert state.live_seat_names == ("BB", "UTG")
    assert state.live_stacks == (9900, 9900)
    assert canonical_flop_chips(LIMP_CHECK, ["BB", "UTG"]) == (
        250,
        (9900, 9900),
    )
    assert canonical_flop_chips(
        LIMP_CHECK, ["BB", "UTG"], rules=SHIPPED_PREFLOP_RULES
    ) == (250, (9900, 9900))


def test_limp_isolation_preserves_legacy_undocumented_acceptance():
    state = replay_preflop(LIMP_ISOLATION, require_terminal=True)
    assert state.pot == 840
    assert state.live_seat_names == ("UTG", "CO")
    assert state.live_stacks == (9655, 9655)
    assert canonical_flop_chips(LIMP_ISOLATION, ["UTG", "CO"]) == (
        840,
        (9655, 9655),
    )
    assert canonical_flop_chips(
        LIMP_ISOLATION,
        ["UTG", "CO"],
        rules=SHIPPED_PREFLOP_RULES,
    ) == (840, (9655, 9655))


@pytest.mark.parametrize("bad", [4, 1.5, True, "2"])
def test_replay_rejects_out_of_vocab_or_non_integer_action(bad):
    with pytest.raises(PreflopReplayError):
        replay_preflop([bad])


def test_advance_rejects_bool_action_index():
    with pytest.raises(PreflopReplayError):
        advance_preflop(new_preflop_state(), False)


def test_public_game_loop_boundaries_reject_non_shipped_rules():
    custom = PreflopRules(tiers=((0.5,), (0.75,), (1.0,)))
    root = new_preflop_state(custom)
    terminal = replay_preflop([0, 0, 0, 0, 0], custom, require_terminal=True)
    assert terminal.uncontested  # low-level generic replay remains available

    with pytest.raises(PreflopError, match="SHIPPED_PREFLOP_RULES"):
        play_preflop((object(),) * 6, _six_hands(), state=root)
    with pytest.raises(PreflopError, match="SHIPPED_PREFLOP_RULES"):
        settle_preflop_terminal(terminal, _six_hands())
    with pytest.raises(CanonicalChipsUnsupported, match="SHIPPED_PREFLOP_RULES"):
        canonical_flop_chips(
            LIMP_CHECK, ["BB", "UTG"], rules=custom
        )


class _ScriptSource(StrategySource):
    def __init__(self, prefix):
        self._prefix = tuple(prefix)

    def get_mix(self, state, hero_cards):
        action_idx = self._prefix[len(state.action_history)]
        assert action_idx < state.num_legal_actions()
        return ActionMix({action_idx: 1.0})


def test_driver_uses_real_poker_agent_duck_type_and_replay_validates():
    prefix = [0, 0, 0, 0, 0]
    source = _ScriptSource(prefix)
    agents = tuple(
        PokerAgent(source, ModeConfig(mode="LOOKUP", seed=seat))
        for seat in range(6)
    )
    state = play_preflop(agents, _six_hands())
    assert state.action_history == tuple(prefix)
    assert state.uncontested
    assert settle_preflop_terminal(state, _six_hands()).deltas == (
        -50,
        50,
        0,
        0,
        0,
        0,
    )


class _StreetZeroReader:
    def __init__(self, strategy=(0.1, 0.2, 0.3, 0.4), metadata=None):
        self.strategy = strategy
        self.calls = []
        self.metadata = metadata or {
            "preflop_tiers": {"0": [0.7, 1.0], "1": [0.7, 1.0], "2": [1.0]},
            "preflop_max_raises": 3,
            "add_explicit_allin": 0,
            "blinds": [50, 100],
            "initial_stack": 10_000,
            "num_players": 6,
        }

    def get_metadata(self):
        return self.metadata

    def hand_to_bucket(self, *args):
        raise AssertionError("street 0 must not use the flop bucketizer")

    def hand_to_bucket_turn(self, *args):
        raise AssertionError("street 0 must not use the turn bucketizer")

    def hand_to_bucket_river(self, *args):
        raise AssertionError("street 0 must not use the river bucketizer")

    def get_strategy(self, board, history, player, bucket, street=None):
        self.calls.append((board, history, player, bucket, street))
        return self.strategy


def test_lookup_source_street_zero_uses_169_class_and_exact_key():
    reader = _StreetZeroReader()
    source = LookupSource(ModeConfig(mode="LOOKUP"), reader=reader)
    state = new_preflop_state()
    hero = (48, 49)
    mix = source.get_mix(state, hero)
    assert reader.calls == [([], [], FULL_SEAT["UTG"], hand_class_169(*hero), 0)]
    assert mix.freqs == pytest.approx({0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4})
    assert not mix.is_uniform_fallback


def test_poker_agent_samples_lookup_source_on_preflop_state():
    reader = _StreetZeroReader(strategy=(0.0, 0.0, 1.0, 0.0))
    agent = PokerAgent(
        LookupSource(ModeConfig(mode="LOOKUP"), reader=reader),
        ModeConfig(mode="LOOKUP", seed=17),
    )
    sampled = agent.act(new_preflop_state(), (48, 49))
    assert sampled.seat == FULL_SEAT["UTG"]
    assert sampled.action_idx == 2


def test_lookup_source_preflop_action_count_mismatch_fails_loud():
    source = LookupSource(
        ModeConfig(mode="LOOKUP"), reader=_StreetZeroReader(strategy=(0.2, 0.3, 0.5))
    )
    with pytest.raises(ValueError, match="preflop action-count mismatch"):
        source.get_mix(new_preflop_state(), (48, 49))


def test_lookup_source_preflop_zero_mass_wrong_length_still_fails_loud():
    source = LookupSource(
        ModeConfig(mode="LOOKUP"),
        reader=_StreetZeroReader(strategy=(0.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="preflop action-count mismatch"):
        source.get_mix(new_preflop_state(), (48, 49))


def test_lookup_source_preflop_metadata_mismatch_fails_loud():
    source = LookupSource(ModeConfig(mode="LOOKUP"), reader=_StreetZeroReader())
    other_rules = PreflopRules(tiers=((0.5,), (1.0,), (1.0,)))
    with pytest.raises(ValueError, match="PreflopState vocab does not match"):
        source.get_mix(new_preflop_state(other_rules), (48, 49))


@pytest.mark.parametrize(
    "tier_metadata",
    [
        {"1": [0.7, 1.0], "2": [0.7, 1.0], "3": [1.0]},
        {"0": [0.7, 1.0], "2": [0.7, 1.0], "3": [1.0]},
    ],
)
def test_lookup_source_rejects_shifted_or_noncontiguous_tier_indices(
    tier_metadata,
):
    metadata = dict(_StreetZeroReader().metadata)
    metadata["preflop_tiers"] = tier_metadata
    source = LookupSource(
        ModeConfig(mode="LOOKUP"),
        reader=_StreetZeroReader(metadata=metadata),
    )
    with pytest.raises(ValueError, match="PreflopState vocab does not match"):
        source.get_mix(new_preflop_state(), (48, 49))


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("blinds", [25, 100]),
        ("blinds", [50.5, 100]),
        ("initial_stack", 20_000),
        ("initial_stack", 10_000.5),
        ("num_players", 5),
        ("num_players", 6.5),
    ],
)
def test_lookup_source_preflop_chip_or_seat_metadata_mismatch_fails_loud(
    field, wrong_value
):
    metadata = dict(_StreetZeroReader().metadata)
    metadata[field] = wrong_value
    source = LookupSource(
        ModeConfig(mode="LOOKUP"),
        reader=_StreetZeroReader(metadata=metadata),
    )
    with pytest.raises(ValueError, match="PreflopState vocab does not match"):
        source.get_mix(new_preflop_state(), (48, 49))


def test_real_blueprint_street_zero_root_hit(fixture_blueprint_path):
    source = LookupSource(
        ModeConfig(
            mode="LOOKUP",
            blueprint_path=fixture_blueprint_path,
            uniform_on_miss=False,
        )
    )
    mix = source.get_mix(new_preflop_state(), (48, 49))
    assert set(mix.freqs) == {0, 1, 2, 3}
    assert not mix.is_uniform_fallback


@pytest.mark.skipif(
    not os.environ.get("AGENT_BLUEPRINT_PATH"),
    reason="requires the opt-in 8.05B blueprint",
)
@pytest.mark.parametrize(
    "prefix",
    [
        [3, 0, 0, 0, 0, 1],
        SQUEEZE_THREE_WAY,
        FOURBET_HU,
    ],
)
def test_real_blueprint_hits_every_node_on_widened_preflop_lines(prefix):
    source = LookupSource(
        ModeConfig(
            mode="LOOKUP",
            blueprint_path=os.environ["AGENT_BLUEPRINT_PATH"],
            uniform_on_miss=False,
        )
    )
    state = new_preflop_state()
    hero = (48, 49)  # AA: stable strong preflop class for key-existence walks
    for action_idx in prefix:
        mix = source.get_mix(state, hero)
        assert not mix.is_uniform_fallback
        assert len(mix.freqs) == state.num_legal_actions()
        assert action_idx in mix.freqs
        state = state.advance(action_idx)
    assert state.terminal
