"""Pure full-hand and duplicate/rotation tests for scripts/run_match.py."""

from __future__ import annotations

import importlib.util
import json
import sys
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
    assert run.summary["mean_bb_per_100"] == -60.0
    assert run.summary["ci95_bb_per_100"] == [-60.0, -60.0]


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


def test_stub_agents_play_at_least_100_full_showdown_hands_through_real_states():
    # 17 independent groups x 6 duplicate rotations = 102 complete 6-max hands.
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
    assert "engine.bps_reader_py" not in sys.modules
    assert "street_solver_gpu" not in sys.modules


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

    runner.write_match(run, tmp_path)
    assert groups_path.read_bytes() == first_groups
    assert summary_path.read_bytes() == first_summary


def test_cli_boundary_writes_artifacts_and_final_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        runner,
        "_lookup_factories",
        lambda _args: (lambda _seed: _CheckCallPolicy(), None),
    )
    out = tmp_path / "cli"
    exit_code = runner.main(
        [
            "--mode", "aa", "--deals", "1", "--master-seed", "19",
            "--out", str(out),
        ]
    )

    stdout = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert (out / "deal_groups.jsonl").is_file()
    assert (out / "summary.json").is_file()
    assert stdout[-1].startswith("MATCH_SUMMARY ")
    printed = json.loads(stdout[-1].removeprefix("MATCH_SUMMARY "))
    assert printed == json.loads((out / "summary.json").read_text(encoding="utf-8"))
