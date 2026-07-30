"""Pure full-hand and duplicate/rotation tests for scripts/run_match.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_match.py"


def _load_runner():
    name = "run_match_test_module"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so Python 3.9 dataclasses can resolve __module__.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class _LabelPolicy:
    def __init__(self, label: str) -> None:
        self.label = label


class _CheckCallPolicy:
    """Limp/call preflop and check/call postflop; never loads a blueprint."""

    def act(self, state, _hero_cards):
        return runner.ActionChoice(1 if state.facing_bet else 0)


class _AlwaysFoldPolicy:
    def act(self, _state, _hero_cards):
        return runner.ActionChoice(0)


class _SeededPassivePolicy:
    """Stochastic fold/call policy with no raises, safe on both state types."""

    def __init__(self, seed: int, fold_probability: float) -> None:
        self._rng = np.random.default_rng(seed)
        self._fold_probability = fold_probability

    def act(self, state, _hero_cards):
        if state.facing_bet and state.num_legal_actions() >= 2:
            action = 0 if self._rng.random() < self._fold_probability else 1
        else:
            action = 0
        return runner.ActionChoice(action)


class _FallbackMix:
    is_uniform_fallback = True


class _FallbackAction:
    def __init__(self, action_idx: int) -> None:
        self.action_idx = action_idx
        self.mix = _FallbackMix()


class _FallbackCheckCallPolicy:
    def act(self, state, _hero_cards):
        return _FallbackAction(1 if state.facing_bet else 0)


class _AggressiveLinePolicy:
    """All seats limp; SB bets flop, selected seats fold/call, survivors check down."""

    def __init__(self, seat: int) -> None:
        self.seat = seat

    def act(self, state, _hero_cards):
        if state.street == 0:
            action = 1 if state.facing_bet else 0
        elif state.street == 1 and self.seat == 0 and not state.facing_bet:
            action = 1  # SB bets 75% pot.
        elif state.facing_bet:
            action = 1 if self.seat in (2, 4) else 0  # UTG/CO call; others fold.
        else:
            action = 0
        return runner.ActionChoice(action)


def _fixed_deal():
    return runner.Deal(
        board=(0, 5, 10, 15, 20),
        hole_cards=((1, 2), (3, 4), (6, 7), (8, 9), (11, 12), (13, 14)),
    )


def test_ab_group_is_hero_minus_mean_field_and_rotates_all_seats():
    seen_hero_seats = []

    def play(_deal, policies):
        hero = next(seat for seat, policy in enumerate(policies) if policy.label == "A")
        seen_hero_seats.append(hero)
        deltas = [-10] * 6
        deltas[hero] = 50
        return runner.PlayedHand(
            deltas=tuple(deltas),
            action_count=1,
            terminal="synthetic",
            went_to_showdown=False,
        )

    group = runner.play_deal_group(
        _fixed_deal(),
        group_index=0,
        mode="ab",
        master_seed=7,
        a_factory=lambda _seed: _LabelPolicy("A"),
        b_factory=lambda _seed: _LabelPolicy("B"),
        hand_player=play,
    )

    assert seen_hero_seats == [0, 1, 2, 3, 4, 5]
    assert group.paired_differential_chips == 60.0  # 50 - mean([-10] * 5)
    assert all(rotation.seat_differential_chips == 60.0 for rotation in group.rotations)
    # Headline metric: hero's own net chips, exactly the differential / 1.2.
    assert group.hero_net_chips_per_hand == 50.0


def test_flag_off_omits_all_adjusted_serialization_fields():
    def play(_deal, _policies):
        return runner.PlayedHand(
            deltas=(5, -1, -1, -1, -1, -1),
            adjusted_deltas=(2.5, -0.5, -0.5, -0.5, -0.5, -0.5),
            action_count=1,
            terminal="synthetic",
            went_to_showdown=False,
        )

    group = runner.play_deal_group(
        _fixed_deal(),
        group_index=0,
        mode="aa",
        master_seed=0,
        a_factory=lambda _seed: _LabelPolicy("A"),
        hand_player=play,
    )
    record = group.as_dict(mode="aa", master_seed=0)
    summary = runner.summarize_groups(
        [group], mode="aa", master_seed=0, epsilon=0.05
    )

    assert group.allin_equity_adjust is False
    assert all(
        "adjusted" not in key
        for key in record
    )
    assert all(
        "adjusted" not in key
        for rotation in record["rotations"]
        for key in rotation
    )
    assert all("adjusted" not in key for key in summary)


def test_session_rotation_order_advances_with_group_index():
    def play(_deal, _policies):
        return runner.PlayedHand(
            deltas=(0, 0, 0, 0, 0, 0),
            action_count=0,
            terminal="synthetic",
            went_to_showdown=False,
        )

    orders = []
    for group_index in (0, 1, 5, 6):
        group = runner.play_deal_group(
            _fixed_deal(),
            group_index=group_index,
            mode="ab",
            master_seed=1,
            a_factory=lambda _seed: _LabelPolicy("A"),
            b_factory=lambda _seed: _LabelPolicy("B"),
            hand_player=play,
        )
        orders.append([rotation.hero_seat for rotation in group.rotations])
        assert [rotation.session_hand_index for rotation in group.rotations] == [group_index] * 6

    assert orders == [
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 0],
        [5, 0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5],
    ]


def test_aa_identical_stochastic_policies_have_exact_zero_group_differential():
    run = runner.run_match(
        deals=4,
        mode="aa",
        master_seed=12345,
        a_factory=lambda seed: _SeededPassivePolicy(seed, fold_probability=0.25),
    )

    assert all(group.paired_numerator == 0 for group in run.groups)
    assert all(group.paired_differential_chips == 0.0 for group in run.groups)
    assert run.summary["mean_bb_per_100"] == 0.0
    assert run.summary["stderr_bb_per_100"] == 0.0
    assert run.summary["ci95_bb_per_100"] == [0.0, 0.0]


def test_master_seed_reproduces_decks_actions_and_summary():
    kwargs = dict(
        deals=3,
        mode="ab",
        master_seed=911,
        a_factory=lambda seed: _SeededPassivePolicy(seed, fold_probability=0.05),
        b_factory=lambda seed: _SeededPassivePolicy(seed, fold_probability=0.65),
    )
    first = runner.run_match(**kwargs)
    second = runner.run_match(**kwargs)
    different_seed = runner.run_match(**{**kwargs, "master_seed": 912})

    assert first == second
    assert first.summary == second.summary
    assert first.groups[0].deal != different_seed.groups[0].deal


def test_epsilon_policy_always_selects_a_different_action_when_possible():
    class _State:
        def num_legal_actions(self):
            return 3

    class _AlwaysOne:
        def act(self, _state, _cards):
            return runner.ActionChoice(1)

    policy = runner.EpsilonPolicy(_AlwaysOne(), epsilon=1.0, seed=3)
    actions = [policy.act(_State(), (0, 1)).action_idx for _ in range(20)]

    assert set(actions) <= {0, 2}
    assert 1 not in actions
    assert policy.perturbations == 20


def test_epsilon_mode_has_negative_known_sign_in_synthetic_power_check():
    class _TwoActions:
        def num_legal_actions(self):
            return 2

    class _AlwaysZero:
        def act(self, _state, _cards):
            return runner.ActionChoice(0)

    def play(_deal, policies):
        actions = [policy.act(_TwoActions(), (0, 1)).action_idx for policy in policies]
        hero = actions.index(1)
        assert actions.count(1) == 1
        deltas = [10] * 6
        deltas[hero] = -50
        return runner.PlayedHand(
            deltas=tuple(deltas),
            action_count=6,
            terminal="synthetic_power_check",
            went_to_showdown=False,
        )

    run = runner.run_match(
        deals=3,
        mode="epsilon",
        master_seed=20260711,
        epsilon=1.0,
        a_factory=lambda _seed: _AlwaysZero(),
        hand_player=play,
    )

    assert all(group.paired_differential_chips == -60.0 for group in run.groups)
    # Headline bb/100 is hero NET (-50 chips/hand), not the -60 differential.
    assert all(group.hero_net_chips_per_hand == -50.0 for group in run.groups)
    assert run.summary["mean_bb_per_100"] == -50.0
    assert run.summary["ci95_bb_per_100"] == [-50.0, -50.0]


@pytest.mark.parametrize("epsilon", [float("nan"), float("inf"), float("-inf")])
def test_epsilon_rejects_non_finite_values(epsilon):
    with pytest.raises(ValueError, match="finite"):
        runner.EpsilonPolicy(_CheckCallPolicy(), epsilon=epsilon, seed=0)
    with pytest.raises(ValueError, match="finite"):
        runner.run_match(
            deals=1,
            mode="aa",
            master_seed=0,
            epsilon=epsilon,
            a_factory=lambda _seed: _CheckCallPolicy(),
        )


def test_epsilon_propagates_underlying_fallback_mix():
    class _State:
        def num_legal_actions(self):
            return 2

    class _FallbackZero:
        def act(self, _state, _cards):
            return _FallbackAction(0)

    result = runner.EpsilonPolicy(
        _FallbackZero(), epsilon=1.0, seed=1
    ).act(_State(), (0, 1))
    assert result.mix.is_uniform_fallback is True


def test_fallback_observability_covers_preflop_and_postflop_decisions():
    run = runner.run_match(
        deals=1,
        mode="aa",
        master_seed=8,
        a_factory=lambda _seed: _FallbackCheckCallPolicy(),
    )

    assert run.summary["decisions"] > 0
    assert run.summary["fallback_count"] == run.summary["decisions"]
    assert run.summary["fallback_rate"] == 1.0
    assert all(
        rotation.hand.fallback_count == rotation.hand.action_count
        for rotation in run.groups[0].rotations
    )


def test_aggressive_real_hand_merges_contributions_and_folded_statuses():
    hand = runner.play_full_hand(
        _fixed_deal(), tuple(_AggressiveLinePolicy(seat) for seat in range(6))
    )

    assert hand.terminal == "showdown"
    assert hand.contributions == (550, 100, 550, 100, 550, 100)
    assert hand.folded == (False, True, False, True, False, True)
    assert sum(hand.contributions) == 1950
    assert sum(hand.deltas) == 0


def test_fold_and_river_closed_hands_copy_raw_deltas_when_adjusted():
    fold_hand = runner.play_full_hand(
        _fixed_deal(),
        tuple(_AlwaysFoldPolicy() for _ in range(6)),
        allin_equity_adjust=True,
    )
    river_hand = runner.play_full_hand(
        _fixed_deal(),
        tuple(_CheckCallPolicy() for _ in range(6)),
        allin_equity_adjust=True,
    )

    assert fold_hand.terminal == "preflop_fold"
    assert fold_hand.adjusted_deltas == tuple(
        float(delta) for delta in fold_hand.deltas
    )
    assert river_hand.terminal == "showdown"
    assert river_hand.adjusted_deltas == tuple(
        float(delta) for delta in river_hand.deltas
    )


def test_flop_allin_adjustment_equals_independent_completion_mean():
    from app.engine.agent.settlement import settle_hand

    board = (0, 5, 10, 44, 45)
    hole_cards = (
        (48, 49),
        (46, 47),
        (1, 2),
        (3, 4),
        (6, 7),
        (8, 9),
    )
    contributions = (1000, 1000, 0, 0, 0, 0)
    folded = (False, False, True, True, True, True)

    production = runner._allin_equity_adjusted_deltas(
        full_board=board,
        hole_cards=hole_cards,
        contributions=contributions,
        folded=folded,
        live_board_card_count=3,
    )

    # Independent enumeration: exclude every dealt hole card, including all
    # four folded hands, plus only the three board cards live at the all-in.
    excluded = set(board[:3])
    for hand in hole_cards:
        excluded.update(hand)
    remaining = tuple(card for card in range(52) if card not in excluded)
    assert len(remaining) == 37
    totals = [0] * 6
    completion_count = 0
    for turn, river in combinations(remaining, 2):
        settled = settle_hand(
            [*board[:3], turn, river],
            hole_cards,
            contributions,
            folded,
            button=runner.BUTTON_SEAT,
        )
        for seat, delta in enumerate(settled.deltas):
            totals[seat] += delta
        completion_count += 1
    expected = tuple(
        total / float(completion_count) for total in totals
    )
    realized = settle_hand(
        board,
        hole_cards,
        contributions,
        folded,
        button=runner.BUTTON_SEAT,
    ).deltas

    assert production == pytest.approx(expected)
    assert sum(production) == pytest.approx(0.0, abs=1e-9)
    assert any(
        abs(adjusted - float(delta)) > 1e-9
        for adjusted, delta in zip(production, realized)
    )


def test_aa_adjusted_telescoping_is_exact_zero_across_rotations():
    def play(
        _deal, _policies, *, allin_equity_adjust=False
    ):
        assert allin_equity_adjust is True
        return runner.PlayedHand(
            deltas=(5, -1, -1, -1, -1, -1),
            adjusted_deltas=(2.5, -0.5, -0.5, -0.5, -0.5, -0.5),
            action_count=1,
            terminal="synthetic_adjusted",
            went_to_showdown=True,
        )

    run = runner.run_match(
        deals=2,
        mode="aa",
        master_seed=17,
        a_factory=lambda _seed: _LabelPolicy("A"),
        hand_player=play,
        allin_equity_adjust=True,
    )

    assert all(
        group.hero_net_chips_per_hand_adjusted == 0.0
        for group in run.groups
    )
    assert all(
        group.as_dict(
            mode="aa", master_seed=17
        )["hero_net_chips_per_hand_adjusted"] == 0.0
        for group in run.groups
    )
    adjusted_record = run.groups[0].as_dict(
        mode="aa", master_seed=17
    )
    assert (
        adjusted_record["paired_differential_chips_adjusted"] == 0.0
    )
    assert all(
        "adjusted_deltas" in rotation
        and "hero_delta_chips_adjusted" in rotation
        and "field_mean_delta_chips_adjusted" in rotation
        and "seat_differential_chips_adjusted" in rotation
        for rotation in adjusted_record["rotations"]
    )
    assert run.summary["mean_bb_per_100_adjusted"] == 0.0
    assert run.summary["stderr_bb_per_100_adjusted"] == 0.0
    assert run.summary["ci95_bb_per_100_adjusted"] == [0.0, 0.0]


def test_stub_agents_play_at_least_100_full_showdown_hands_through_real_states():
    # 17 independent groups x 6 duplicate rotations = 102 complete 6-max hands.
    # Snapshot-compare sys.modules: earlier blueprint-gated tests legitimately
    # load the reader on EC2 (AGENT_BLUEPRINT_PATH set), so assert only that
    # THIS stub run loads no blueprint or native-solver machinery itself.
    def _heavy_modules():
        return {
            name
            for name in sys.modules
            if "bps_reader" in name or "street_solver_gpu" in name
        }

    heavy_before = _heavy_modules()
    run = runner.run_match(
        deals=17,
        mode="aa",
        master_seed=20260711,
        a_factory=lambda _seed: _CheckCallPolicy(),
    )

    hands = [rotation.hand for group in run.groups for rotation in group.rotations]
    assert len(hands) == 102
    assert all(hand.terminal == "showdown" for hand in hands)
    assert all(hand.went_to_showdown for hand in hands)
    assert all(sum(hand.deltas) == 0 for hand in hands)
    assert all(hand.action_count > 6 for hand in hands)
    assert all(group.paired_differential_chips == 0.0 for group in run.groups)
    assert _heavy_modules() == heavy_before


def test_jsonl_group_records_and_summary_are_deterministic(tmp_path):
    run = runner.run_match(
        deals=2,
        mode="aa",
        master_seed=44,
        a_factory=lambda _seed: _CheckCallPolicy(),
    )
    groups_path, summary_path = runner.write_match(run, tmp_path)
    first_groups = groups_path.read_bytes()
    first_summary = summary_path.read_bytes()

    lines = groups_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(records) == 2
    assert [record["group_index"] for record in records] == [0, 1]
    assert all(len(record["rotations"]) == 6 for record in records)
    assert all("seat_differential_chips" in arm for arm in records[0]["rotations"])
    assert all("paired_differential_chips" not in arm for arm in records[0]["rotations"])
    assert summary["n_deal_groups"] == 2
    assert summary["n_full_hands"] == 12
    assert summary["fallback_count"] == 0
    assert "solve_wall_s_by_source" not in records[0]
    assert "solve_wall_s_by_source" not in summary
    assert "truncated_by_wall_budget" not in summary
    assert "seats" not in summary

    runner.write_match(run, tmp_path)
    assert groups_path.read_bytes() == first_groups
    assert summary_path.read_bytes() == first_summary


def test_cli_boundary_writes_artifacts_and_final_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        runner,
        "_lookup_factories",
        lambda _args: (
            lambda _seed: _CheckCallPolicy(),
            None,
            {"blueprint_a": "stub-a.bps", "blueprint_b": None},
        ),
    )
    out = tmp_path / "cli"
    exit_code = runner.main(
        [
            "--mode", "aa", "--deals", "1", "--master-seed", "19",
            "--out", str(out), "--allin-equity-adjust",
        ]
    )

    stdout = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert (out / "deal_groups.jsonl").is_file()
    assert (out / "summary.json").is_file()
    assert not (out / "deal_groups.jsonl.partial").exists()
    assert stdout[-1].startswith("MATCH_SUMMARY ")
    printed = json.loads(stdout[-1].removeprefix("MATCH_SUMMARY "))
    assert printed == json.loads((out / "summary.json").read_text(encoding="utf-8"))
    # The summary must record which blueprints played, and a single deal
    # group must not report a zero-width confidence interval.
    assert printed["blueprint_a"] == "stub-a.bps"
    assert printed["blueprint_b"] is None
    assert printed["stderr_bb_per_100"] is None
    assert printed["ci95_bb_per_100"] is None
    assert printed["mean_bb_per_100_adjusted"] == 0.0
    assert printed["stderr_bb_per_100_adjusted"] is None
    assert printed["ci95_bb_per_100_adjusted"] is None
    assert "no variance estimate" in printed["ci_note"]


def test_ab_mode_refuses_to_default_blueprint_b_to_a(monkeypatch, tmp_path):
    blueprint_a = tmp_path / "a.bps"
    blueprint_a.write_bytes(b"stub")
    monkeypatch.delenv("AGENT_BLUEPRINT_B_PATH", raising=False)
    args = runner._parse_args(
        [
            "--mode", "ab", "--deals", "1", "--master-seed", "0",
            "--out", str(tmp_path / "out"),
            "--blueprint-a", str(blueprint_a),
        ]
    )

    with pytest.raises(ValueError, match="refusing to silently fall back"):
        runner._lookup_factories(args)


def test_cli_source_defaults_and_validation(capsys, tmp_path):
    args = runner._parse_args(
        ["--mode", "aa", "--out", str(tmp_path / "default")]
    )
    assert args.seats == 6
    assert args.hero_source == "lookup"
    assert args.field_source == "lookup"
    assert args.live_iterations == 250_000
    assert args.live_num_rollouts is None
    assert args.max_wall_s is None
    assert args.allin_equity_adjust is False

    adjusted = runner._parse_args(
        [
            "--mode", "aa", "--out", str(tmp_path / "adjusted"),
            "--allin-equity-adjust",
        ]
    )
    assert adjusted.allin_equity_adjust is True

    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--mode", "aa", "--out", str(tmp_path / "hu"),
                "--seats", "2",
            ]
        )
    assert "full-hand preflop transition" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--mode", "aa", "--out", str(tmp_path / "bad-source"),
                "--hero-source", "live_eager",
                "--field-source", "live_sampled",
            ]
        )
    assert "requires identical hero/field sources" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--mode", "aa", "--out", str(tmp_path / "aa-live"),
                "--hero-source", "live_sampled",
                "--field-source", "live_sampled",
            ]
        )
    assert "exact deterministic lookup plumbing null" in capsys.readouterr().err

    mixed = runner._parse_args(
        [
            "--mode", "ab", "--out", str(tmp_path / "mixed"),
            "--hero-source", "live_eager",
            "--field-source", "live_sampled",
        ]
    )
    assert (mixed.hero_source, mixed.field_source) == (
        "live_eager", "live_sampled",
    )

    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--mode", "ab", "--out", str(tmp_path / "unknown"),
                "--hero-source", "not-a-source",
            ]
        )
    with pytest.raises(SystemExit):
        runner._parse_args(
            ["--mode", "ab", "--out", str(tmp_path / "bad-seats"),
             "--seats", "3"]
        )
    with pytest.raises(SystemExit):
        runner._parse_args(
            ["--mode", "ab", "--out", str(tmp_path / "bad-wall"),
             "--max-wall-s", "0"]
        )
    live_override = runner._parse_args(
        [
            "--mode", "ab", "--out", str(tmp_path / "live-override"),
            "--live-num-rollouts", "2",
        ]
    )
    assert live_override.live_num_rollouts == 2
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--mode", "ab", "--out", str(tmp_path / "bad-rollouts"),
                "--live-num-rollouts", "0",
            ]
        )


def test_leaf_mode_scope_sets_exact_values_and_restores_on_exception(
    monkeypatch,
):
    assert runner._cli.effective_leaf_mode_env("live_sampled") == {
        "NEXUS_LEAF_SAMPLED": "1",
        "NEXUS_LEAF_SAMPLED_POLICY": "1",
        "NEXUS_LEAF_RANGE_CONDITIONING": "0",
    }
    assert runner._cli.effective_leaf_mode_env("live_eager") == {
        "NEXUS_LEAF_SAMPLED": "0 (unset)",
        "NEXUS_LEAF_SAMPLED_POLICY": "1 (inert default)",
        "NEXUS_LEAF_RANGE_CONDITIONING": "1 (implicit default)",
    }
    controlled = tuple(runner._cli._CONTROLLED_LEAF_ENVS)
    for index, name in enumerate(controlled):
        if index % 2:
            monkeypatch.setenv(name, f"prior-{index}")
        else:
            monkeypatch.delenv(name, raising=False)
    before = {name: os.environ.get(name) for name in controlled}

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with runner._scoped_leaf_mode("live_sampled"):
            assert os.environ["NEXUS_LEAF_SAMPLED"] == "1"
            assert os.environ["NEXUS_LEAF_SAMPLED_POLICY"] == "1"
            assert os.environ["NEXUS_LEAF_RANGE_CONDITIONING"] == "0"
            for name in controlled:
                if name not in {
                    "NEXUS_LEAF_SAMPLED",
                    "NEXUS_LEAF_SAMPLED_POLICY",
                    "NEXUS_LEAF_RANGE_CONDITIONING",
                }:
                    assert name not in os.environ
            raise RuntimeError("synthetic failure")

    assert {name: os.environ.get(name) for name in controlled} == before

    with runner._scoped_leaf_mode("live_eager"):
        assert all(name not in os.environ for name in controlled)
    assert {name: os.environ.get(name) for name in controlled} == before


def test_timed_solver_restores_environment_and_records_failed_solve(
    monkeypatch,
):
    controlled = tuple(runner._cli._CONTROLLED_LEAF_ENVS)
    for index, name in enumerate(controlled):
        monkeypatch.setenv(name, f"prior-{index}")
    before = {name: os.environ.get(name) for name in controlled}
    tracker = runner._cli.SolveWallTracker({"hero": "live_sampled"})
    tracker.begin_group()
    clock_values = iter((10.0, 10.25))

    class FailingSolver:
        def solve(self):
            assert os.environ["NEXUS_LEAF_SAMPLED"] == "1"
            assert os.environ["NEXUS_LEAF_SAMPLED_POLICY"] == "1"
            assert os.environ["NEXUS_LEAF_RANGE_CONDITIONING"] == "0"
            raise RuntimeError("synthetic solve failure")

    timed = runner._cli._ScopedTimedSolver(
        FailingSolver(),
        role="hero",
        source="live_sampled",
        tracker=tracker,
        clock=lambda: next(clock_values),
    )
    with pytest.raises(RuntimeError, match="synthetic solve failure"):
        timed.solve()

    assert tracker.finish_group() == {"hero": (0.25,)}
    assert {name: os.environ.get(name) for name in controlled} == before


def test_live_router_uses_lookup_preflop_and_street_entry_player_count():
    class Source:
        def __init__(self, value):
            self.value = value

        def get_mix(self, _state, _hero_cards):
            return self.value

    router = runner._cli._LiveByPlayerCountSource(
        Source("lookup"),
        {2: Source("two"), 3: Source("three")},
    )
    preflop = type("State", (), {"street": 0})()
    three_way = type(
        "State", (),
        {
            "street": 1,
            "active": (True, False, True, False),
            "acted": (False, True, False, False),
        },
    )()

    assert router.get_mix(preflop, (0, 1)) == "lookup"
    assert router.get_mix(three_way, (0, 1)) == "three"


@pytest.fixture
def stub_live_factory_modules(monkeypatch):
    config_calls = []
    live_sources = []

    class Stub:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.__dict__.update(kwargs)

    class LiveResolveSource:
        def __init__(self, config, **kwargs):
            self.config = config
            self.__dict__.update(kwargs)
            live_sources.append(self)

    def live_config_for(n_players, **kwargs):
        config_calls.append((n_players, kwargs))
        num_rollouts = kwargs.get("num_rollouts_override")
        if num_rollouts is not None and not 3 <= num_rollouts <= 30:
            raise ValueError(
                f"num_rollouts override {num_rollouts} outside the measured "
                "bounds [3, 30]"
            )
        return types.SimpleNamespace(
            iterations=kwargs["iterations_override"],
            num_rollouts=3 if num_rollouts is None else num_rollouts,
            vec_mode=False,
            readout="final",
        )

    fake_modules = {
        "app.engine.agent.agent": types.SimpleNamespace(
            PokerAgent=Stub
        ),
        "app.engine.agent.modes": types.SimpleNamespace(
            ModeConfig=Stub
        ),
        "app.engine.agent.strategy_source": types.SimpleNamespace(
            HybridSource=Stub,
            LiveResolveSource=LiveResolveSource,
            LookupSource=Stub,
        ),
        "app.engine.cpu_es_solver": types.SimpleNamespace(
            CpuEsSolver=Stub
        ),
        "app.engine.engine_constants": types.SimpleNamespace(
            live_config_for=live_config_for
        ),
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return config_calls, live_sources


def test_live_num_rollouts_override_reaches_configs_and_metadata(
    stub_live_factory_modules, tmp_path,
):
    blueprint_a = tmp_path / "a.bps"
    blueprint_b = tmp_path / "b.bps"
    blueprint_a.write_bytes(b"stub")
    blueprint_b.write_bytes(b"stub")
    config_calls, live_sources = stub_live_factory_modules

    args = runner._parse_args(
        [
            "--mode", "ab", "--out", str(tmp_path / "out"),
            "--blueprint-a", str(blueprint_a),
            "--blueprint-b", str(blueprint_b),
            "--hero-source", "live_sampled",
            "--live-num-rollouts", "2",
        ]
    )
    _, _, metadata = runner._lookup_factories(args)

    assert [call[0] for call in config_calls] == [2, 3, 4, 5, 6]
    assert all(
        "num_rollouts_override" not in call[1]
        for call in config_calls
    )
    assert all(source.num_rollouts == 2 for source in live_sources)
    assert metadata["live_num_rollouts"] == 2
    assert (
        metadata["leaf_mode_envs"]["hero"]["live_num_rollouts_bounds"]
        == "sampled_direct"
    )


def test_live_sampled_nr_one_bypasses_eager_config_bounds(
    stub_live_factory_modules, tmp_path,
):
    blueprint = tmp_path / "blueprint.bps"
    blueprint.write_bytes(b"stub")
    config_calls, live_sources = stub_live_factory_modules

    sampled_args = runner._parse_args(
        [
            "--mode", "ab", "--out", str(tmp_path / "sampled"),
            "--blueprint-a", str(blueprint),
            # mode 'ab' refuses to fall back to blueprint A silently; the real
            # match invocation always passes both paths (identical here, as in
            # the hero-vs-own-blueprint benches).
            "--blueprint-b", str(blueprint),
            "--hero-source", "live_sampled",
            "--live-num-rollouts", "1",
        ]
    )
    _, _, metadata = runner._lookup_factories(sampled_args)

    assert all(
        "num_rollouts_override" not in kwargs
        for _, kwargs in config_calls
    )
    assert all(source.num_rollouts == 1 for source in live_sources)
    assert (
        metadata["leaf_mode_envs"]["hero"]["live_num_rollouts_bounds"]
        == "sampled_direct"
    )

    eager_args = runner._parse_args(
        [
            "--mode", "ab", "--out", str(tmp_path / "eager"),
            "--blueprint-a", str(blueprint),
            "--blueprint-b", str(blueprint),
            "--hero-source", "live_eager",
            "--live-num-rollouts", "1",
        ]
    )
    with pytest.raises(ValueError, match=r"bounds \[3, 30\]"):
        runner._lookup_factories(eager_args)
    assert config_calls[-1][1]["num_rollouts_override"] == 1


def test_group_and_summary_aggregate_tracked_solve_walls():
    tracker = runner._cli.SolveWallTracker({
        "hero": "live_eager", "field": "live_sampled",
    })
    calls = {"n": 0}

    def play(_deal, _policies):
        calls["n"] += 1
        tracker.record("hero", float(calls["n"]))
        tracker.record("field", 0.5)
        return runner.PlayedHand(
            deltas=(0, 0, 0, 0, 0, 0),
            action_count=0,
            terminal="synthetic",
            went_to_showdown=False,
        )

    group = runner.play_deal_group(
        _fixed_deal(),
        group_index=0,
        mode="ab",
        master_seed=0,
        a_factory=lambda _seed: _LabelPolicy("A"),
        b_factory=lambda _seed: _LabelPolicy("B"),
        hand_player=play,
        solve_wall_tracker=tracker,
    )
    group_walls = group.as_dict(
        mode="ab", master_seed=0,
    )["solve_wall_s_by_source"]
    assert group_walls["hero"]["count"] == 6
    assert group_walls["hero"]["p50"] == 3.5
    assert group_walls["field"]["count"] == 6
    assert group_walls["field"]["mean"] == 0.5

    summary = runner.summarize_groups(
        [group], mode="ab", master_seed=0, epsilon=0.05,
    )
    assert summary["solve_wall_s_by_source"] == group_walls


def test_wall_budget_stops_after_a_completed_group_and_flags_summary():
    clock_values = iter((10.0, 12.0))

    def clock():
        return next(clock_values)

    def play(_deal, _policies):
        return runner.PlayedHand(
            deltas=(0, 0, 0, 0, 0, 0),
            action_count=0,
            terminal="synthetic",
            went_to_showdown=False,
        )

    run = runner.run_match(
        deals=3,
        mode="aa",
        master_seed=0,
        a_factory=lambda _seed: _LabelPolicy("A"),
        hand_player=play,
        max_wall_s=1.0,
        clock=clock,
    )

    assert len(run.groups) == 1
    assert run.summary["n_deal_groups"] == 1
    assert run.summary["truncated_by_wall_budget"] is True


def test_solve_wall_summaries_pin_linear_percentiles():
    summary = runner._solve_wall_summaries(
        {"hero": (1.0, 2.0, 4.0, 8.0)},
        {"hero": "live_sampled", "field": "lookup"},
    )

    hero = summary["hero"]
    assert hero["source"] == "live_sampled"
    assert hero["count"] == 4
    assert hero["mean"] == 3.75
    assert hero["p50"] == 3.0
    assert hero["p95"] == pytest.approx(7.4)
    assert hero["max"] == 8.0
    assert summary["field"] == {
        "source": "lookup",
        "count": 0,
        "mean": None,
        "p50": None,
        "p95": None,
        "max": None,
    }


def test_summary_statistics_pin_nonzero_variance_and_bb_conversion(monkeypatch):
    # Hero nets +50/+100/+150 chips per hand in three groups; the five field
    # seats split the loss evenly, so every hand conserves chips exactly.
    hero_values = (50, 100, 150)
    calls = {"n": 0}

    def play(_deal, policies):
        group = calls["n"] // 6
        calls["n"] += 1
        hero = next(seat for seat, policy in enumerate(policies) if policy.label == "A")
        value = hero_values[group]
        deltas = [-(value // 5)] * 6
        deltas[hero] = value
        return runner.PlayedHand(
            deltas=tuple(deltas),
            action_count=1,
            terminal="synthetic",
            went_to_showdown=False,
        )

    groups = [
        runner.play_deal_group(
            _fixed_deal(),
            group_index=group_index,
            mode="ab",
            master_seed=1,
            a_factory=lambda _seed: _LabelPolicy("A"),
            b_factory=lambda _seed: _LabelPolicy("B"),
            hand_player=play,
        )
        for group_index in range(3)
    ]

    summary = runner.summarize_groups(groups, mode="ab", master_seed=1, epsilon=0.05)

    # Hand-computed: mean 100 chips, std(ddof=1) of (50, 100, 150) = 50,
    # stderr = 50/sqrt(3) = 28.867513...; BIG_BLIND=100 makes bb/100 = chips.
    assert summary["mean_hero_net_chips_per_hand"] == 100.0
    assert summary["mean_bb_per_100"] == 100.0
    assert summary["stderr_bb_per_100"] == 28.8675
    assert summary["ci95_bb_per_100"] == [43.4197, 156.5803]
    assert summary["mean_paired_differential_chips_per_hand"] == 120.0

    # Mutation-test the bb conversion itself: halving BIG_BLIND must double
    # the reported bb/100 (a dropped or inverted conversion cannot pass).
    monkeypatch.setattr(runner, "BIG_BLIND", 50)
    rescaled = runner.summarize_groups(groups, mode="ab", master_seed=1, epsilon=0.05)
    assert rescaled["mean_bb_per_100"] == 200.0
    assert rescaled["stderr_bb_per_100"] == 57.735
