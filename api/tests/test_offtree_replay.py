"""Pure tests for Gap 4(2)'s amount-matched extended-tree replay."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from app.engine.cpu_es_solver import CpuEsSolver
from app.engine.offtree_replay import (
    OffTreeExtensionUnsafe,
    ReplayUnsupported,
    _sizes_for_node,
    build_node_projection,
    replay_offtree_street_node,
    validate_extended_tree,
)


def _node(
    *,
    idx: int,
    player: int,
    bets: tuple[int, int],
    children: tuple[int, ...] = (),
    active: tuple[bool, bool] = (True, True),
    ntype: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        idx=idx,
        type=ntype,
        player=player,
        bets=bets,
        children=children,
        num_children=len(children),
        active=active,
    )


def _extended_tree() -> SimpleNamespace:
    # Root: check, base 0.75 bet, added 1.40 bet. The added branch reaches
    # node 3, where player 1 has fold/call, added 0.50 raise, base 1.00 raise.
    nodes = [
        _node(idx=0, player=0, bets=(0, 0), children=(1, 2, 3)),
        _node(idx=1, player=1, bets=(0, 0)),
        _node(idx=2, player=1, bets=(750, 0)),
        _node(idx=3, player=1, bets=(1400, 0), children=(4, 5, 6, 7)),
        _node(
            idx=4, player=-1, bets=(1400, 0),
            active=(True, False), ntype=4,
        ),
        _node(idx=5, player=0, bets=(1400, 1400), ntype=4),
        _node(idx=6, player=0, bets=(1400, 3300), ntype=4),
        _node(idx=7, player=0, bets=(1400, 5200), ntype=4),
    ]
    return SimpleNamespace(
        nodes=nodes,
        num_nodes=len(nodes),
        starting_pot=1000,
        effective_stack=10_000,
        decision_node_indices=(0, 3),
    )


def _repeated_added_tree() -> SimpleNamespace:
    # Root base bet -> added 0.50 raise -> the same global added 0.50
    # re-raise. This models two observations deduped to one added size.
    nodes = [
        _node(idx=0, player=0, bets=(0, 0), children=(1, 2)),
        _node(idx=1, player=1, bets=(0, 0), ntype=4),
        _node(idx=2, player=1, bets=(750, 0), children=(3, 4, 5, 6)),
        _node(
            idx=3, player=-1, bets=(750, 0),
            active=(True, False), ntype=4,
        ),
        _node(idx=4, player=0, bets=(750, 750), ntype=4),
        _node(idx=5, player=0, bets=(750, 2000), children=(7, 8, 9, 10)),
        _node(idx=6, player=0, bets=(750, 3250), ntype=4),
        _node(
            idx=7, player=-1, bets=(750, 2000),
            active=(False, True), ntype=4,
        ),
        _node(idx=8, player=1, bets=(2000, 2000), ntype=4),
        _node(idx=9, player=1, bets=(4500, 2000), ntype=4),
        _node(idx=10, player=1, bets=(7000, 2000), ntype=4),
    ]
    return SimpleNamespace(
        nodes=nodes,
        num_nodes=len(nodes),
        starting_pot=1000,
        effective_stack=10_000,
        decision_node_indices=(0, 2, 5),
    )


def _tree(
    nodes: list[SimpleNamespace],
    *,
    starting_pot: int = 1000,
    effective_stack: int = 10_000,
) -> SimpleNamespace:
    return SimpleNamespace(
        nodes=nodes,
        num_nodes=len(nodes),
        starting_pot=starting_pot,
        effective_stack=effective_stack,
        decision_node_indices=tuple(
            node.idx for node in nodes
            if node.type == 0 and node.num_children > 0
        ),
    )


def _walk(
    tree: SimpleNamespace,
    street_actions: tuple[int, ...],
    *,
    event_fractions: dict[int, float] | None = None,
    base_bet_sizes: tuple[float, ...] = (0.75,),
    base_subsequent_bet_sizes: tuple[float, ...] = (1.0,),
    bet_sizes: tuple[float, ...] = (0.75,),
    subsequent_bet_sizes: tuple[float, ...] = (1.0,),
    expected_event_amounts: dict[int, int] | None = None,
):
    return replay_offtree_street_node(
        tree,
        street_actions,
        n_players=2,
        base_bet_sizes=base_bet_sizes,
        base_subsequent_bet_sizes=base_subsequent_bet_sizes,
        bet_sizes=bet_sizes,
        subsequent_bet_sizes=subsequent_bet_sizes,
        add_explicit_allin=False,
        event_fractions=event_fractions or {},
        expected_event_amounts=expected_event_amounts,
    )


def test_v2_replay_walks_added_child_by_exact_integer_amount() -> None:
    walked = replay_offtree_street_node(
        _extended_tree(),
        (1,),  # recorded base-vocabulary bet index
        n_players=2,
        base_bet_sizes=(0.75,),
        base_subsequent_bet_sizes=(1.0,),
        bet_sizes=(0.75, 1.4),
        subsequent_bet_sizes=(0.5, 1.0),
        add_explicit_allin=False,
        event_fractions={0: 1.4},
        expected_event_amounts={0: 1400},
    )

    assert walked.node_index == 3
    assert walked.replayed_through_added is True
    assert dict(walked.added_child_chips) == {0: 1400}


def test_strict_replay_rejects_scalar_stack_amount_clamp() -> None:
    tree = _extended_tree()
    tree.effective_stack = 1000

    with pytest.raises(
        ReplayUnsupported,
        match="generated raise amount 1000.*observed increment 1400",
    ):
        _walk(
            tree,
            (1,),
            event_fractions={0: 1.4},
            bet_sizes=(0.75, 1.4),
            expected_event_amounts={0: 1400},
        )


def test_v2_replay_reuses_one_added_fraction_for_later_deduped_event() -> None:
    walked = replay_offtree_street_node(
        _repeated_added_tree(),
        (1, 2, 2),
        n_players=2,
        base_bet_sizes=(0.75,),
        base_subsequent_bet_sizes=(1.0,),
        bet_sizes=(0.75,),
        subsequent_bet_sizes=(0.5, 1.0),
        add_explicit_allin=False,
        event_fractions={1: 0.5, 2: 0.5},
    )

    assert walked.node_index == 9
    assert walked.replayed_through_added is True
    assert dict(walked.added_child_chips) == {1: 2000, 2: 3750}


def test_v2_replay_rejects_ambiguous_raise_amount_match() -> None:
    tree = _tree([
        _node(idx=0, player=0, bets=(0, 0), children=(1, 2, 3)),
        _node(idx=1, player=1, bets=(0, 0), ntype=4),
        _node(idx=2, player=1, bets=(750, 0), ntype=4),
        _node(idx=3, player=1, bets=(750, 0), ntype=4),
    ])

    with pytest.raises(ReplayUnsupported, match="matched 2 children"):
        _walk(tree, (1,))


def test_v2_replay_rejects_missing_raise_amount_match() -> None:
    tree = _tree([
        _node(idx=0, player=0, bets=(0, 0), children=(1, 2)),
        _node(idx=1, player=1, bets=(0, 0), ntype=4),
        _node(idx=2, player=1, bets=(600, 0), ntype=4),
    ])

    with pytest.raises(ReplayUnsupported, match="matched 0 children"):
        _walk(
            tree,
            (1,),
            event_fractions={0: 0.5},
            base_bet_sizes=(),
            bet_sizes=(0.5,),
        )


def test_v2_replay_rejects_event_on_nonraise_action() -> None:
    tree = _tree([
        _node(idx=0, player=0, bets=(0, 0), children=(1, 2)),
        _node(idx=1, player=1, bets=(0, 0), ntype=4),
        _node(idx=2, player=1, bets=(750, 0), ntype=4),
    ])

    with pytest.raises(ReplayUnsupported, match="maps to non-raise"):
        _walk(tree, (0,), event_fractions={0: 0.75})


def test_v2_replay_rejects_unconsumed_event_step() -> None:
    with pytest.raises(ReplayUnsupported, match=r"event steps \[2\]"):
        _walk(_extended_tree(), (), event_fractions={2: 1.4})


def test_v2_replay_rejects_raise_at_capped_node() -> None:
    tree = _tree([
        _node(idx=0, player=0, bets=(0, 0), children=(1,)),
        _node(idx=1, player=1, bets=(0, 0), ntype=4),
    ])

    with pytest.raises(ReplayUnsupported, match="raise-capped node 0"):
        _walk(tree, (1,))


def test_v2_replay_rejects_nondecision_node_mid_walk() -> None:
    tree = _tree([
        _node(idx=0, player=0, bets=(0, 0), children=(1,)),
        _node(idx=1, player=1, bets=(0, 0), ntype=4),
    ])

    with pytest.raises(ReplayUnsupported, match="not a decision node"):
        _walk(tree, (0, 0))


@pytest.mark.parametrize(
    "reached",
    [
        _node(idx=0, player=1, bets=(0, 0), children=(1,), ntype=4),
        _node(idx=0, player=1, bets=(0, 0), children=(1,)),
    ],
)
def test_replayed_hero_target_shape_is_typed_unsupported(
    reached: SimpleNamespace,
) -> None:
    tree = SimpleNamespace(nodes=[reached])

    with pytest.raises(ReplayUnsupported, match="not a decision node") as exc_info:
        CpuEsSolver._require_replayed_hero_node(tree, 0, hero_idx=0)

    assert type(exc_info.value) is ReplayUnsupported


def test_solve_routes_both_replay_target_guards_through_typed_helper() -> None:
    source = inspect.getsource(CpuEsSolver.solve)

    assert source.count("self._require_replayed_hero_node(") == 2
    assert "replayed street actions lead to node" not in source


@pytest.mark.parametrize(
    ("base_bet_sizes", "base_subsequent_bet_sizes", "label"),
    [
        ((1.0, 0.75), (1.0,), "base_bet_sizes"),
        ((0.75,), (1.0, 0.5), "base_subsequent_bet_sizes"),
    ],
)
def test_v2_replay_rejects_unsorted_base_size_order(
    base_bet_sizes: tuple[float, ...],
    base_subsequent_bet_sizes: tuple[float, ...],
    label: str,
) -> None:
    with pytest.raises(ReplayUnsupported, match=label):
        _walk(
            _extended_tree(),
            (),
            base_bet_sizes=base_bet_sizes,
            base_subsequent_bet_sizes=base_subsequent_bet_sizes,
        )


def test_subsequent_empty_list_falls_back_to_first_sizes() -> None:
    node = _node(idx=0, player=1, bets=(750, 0))

    assert _sizes_for_node(node, (0.75,), ()) == (0.75,)


def test_v2_replay_fold_shape_drift_fails_loud() -> None:
    tree = _tree([
        _node(idx=0, player=1, bets=(750, 0), children=(1, 2)),
        _node(idx=1, player=0, bets=(750, 0), ntype=4),
        _node(idx=2, player=0, bets=(750, 750), ntype=4),
    ])

    with pytest.raises(ValueError, match="not a structural fold"):
        _walk(tree, (0,))


def test_v2_replay_call_shape_drift_fails_loud() -> None:
    tree = _tree([
        _node(idx=0, player=1, bets=(750, 0), children=(1, 2)),
        _node(
            idx=1,
            player=0,
            bets=(750, 0),
            active=(True, False),
            ntype=4,
        ),
        _node(idx=2, player=0, bets=(750, 749), ntype=4),
    ])

    with pytest.raises(ValueError, match="not a structural check/call"):
        _walk(tree, (1,))


def test_v2_replay_raise_shape_drift_fails_loud() -> None:
    tree = _tree([
        _node(idx=0, player=0, bets=(0, 0), children=(1, 2)),
        _node(idx=1, player=1, bets=(0, 0), ntype=4),
        _node(
            idx=2,
            player=1,
            bets=(750, 0),
            active=(False, True),
            ntype=4,
        ),
    ])

    with pytest.raises(ValueError, match="not structurally a raise"):
        _walk(tree, (1,))


def test_hero_projection_keeps_base_arity_and_merges_every_row() -> None:
    tree = _extended_tree()
    projection = build_node_projection(
        tree,
        3,
        n_players=2,
        base_bet_sizes=(0.75,),
        base_subsequent_bet_sizes=(1.0,),
        bet_sizes=(0.75, 1.4),
        subsequent_bet_sizes=(0.5, 1.0),
        add_explicit_allin=False,
    )

    assert projection.base_child_indices == (0, 1, 3)
    assert projection.base_raise_targets == ((2, 1.0),)
    assert projection.base_action_labels == ("Fold", "Call", "Raise 100%")
    assert [(child.child_index, child.fraction)
            for child in projection.added_children] == [(2, 0.5)]

    solver = CpuEsSolver.__new__(CpuEsSolver)
    sigma = np.asarray([
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.1, 0.4],
    ], dtype=np.float32)
    projected, merged, dropped, reasons = (
        solver._project_offtree_strategy(sigma, projection)
    )

    np.testing.assert_allclose(projected, np.asarray([
        [0.1, 0.2, 0.7],
        [0.2, 0.3, 0.5],
    ], dtype=np.float32))
    assert merged == pytest.approx({0: 0.3, 1: 0.1})
    assert dropped == {}
    assert reasons == {}


def test_projection_rejects_ambiguous_added_fraction_owner() -> None:
    tree = _tree(
        [
            _node(idx=0, player=0, bets=(0, 0), children=(1, 2)),
            _node(idx=1, player=1, bets=(0, 0), ntype=4),
            _node(idx=2, player=1, bets=(5, 0), ntype=4),
        ],
        starting_pot=10,
        effective_stack=100,
    )

    with pytest.raises(ReplayUnsupported, match="maps to 2 added fractions"):
        build_node_projection(
            tree,
            0,
            n_players=2,
            base_bet_sizes=(),
            base_subsequent_bet_sizes=(),
            bet_sizes=(0.51, 0.59),
            subsequent_bet_sizes=(),
            add_explicit_allin=False,
        )


def test_projection_rejects_added_fraction_with_two_children() -> None:
    tree = _tree([
        _node(idx=0, player=0, bets=(0, 0), children=(1, 2, 3)),
        _node(idx=1, player=1, bets=(0, 0), ntype=4),
        _node(idx=2, player=1, bets=(500, 0), ntype=4),
        _node(idx=3, player=1, bets=(500, 0), ntype=4),
    ])

    with pytest.raises(ReplayUnsupported, match="exactly one child"):
        build_node_projection(
            tree,
            0,
            n_players=2,
            base_bet_sizes=(),
            base_subsequent_bet_sizes=(),
            bet_sizes=(0.5,),
            subsequent_bet_sizes=(),
            add_explicit_allin=False,
        )


def test_projection_rejects_added_amount_with_no_child() -> None:
    tree = _tree([
        _node(idx=0, player=0, bets=(0, 0), children=(1,)),
        _node(idx=1, player=1, bets=(0, 0), ntype=4),
    ])

    with pytest.raises(ReplayUnsupported, match="matched no child"):
        build_node_projection(
            tree,
            0,
            n_players=2,
            base_bet_sizes=(),
            base_subsequent_bet_sizes=(),
            bet_sizes=(0.5,),
            subsequent_bet_sizes=(),
            add_explicit_allin=False,
        )


def test_projection_treats_clamp_collided_added_allin_as_base_child() -> None:
    tree = _tree(
        [
            _node(idx=0, player=0, bets=(0, 0), children=(1, 2)),
            _node(idx=1, player=1, bets=(0, 0), ntype=4),
            _node(idx=2, player=1, bets=(100, 0), ntype=4),
        ],
        effective_stack=100,
    )

    projection = build_node_projection(
        tree,
        0,
        n_players=2,
        base_bet_sizes=(1.0,),
        base_subsequent_bet_sizes=(),
        bet_sizes=(0.5, 1.0),
        subsequent_bet_sizes=(),
        add_explicit_allin=False,
    )

    assert projection.base_child_indices == (0, 1)
    assert projection.base_raise_targets == ((1, 1.0),)
    assert projection.added_children == ()
    assert projection.base_action_labels == ("Check", "All-in")


def test_real_projection_drops_added_mass_without_base_raise_child() -> None:
    tree = _tree([
        _node(idx=0, player=0, bets=(0, 0), children=(1, 2)),
        _node(idx=1, player=1, bets=(0, 0), ntype=4),
        _node(idx=2, player=1, bets=(500, 0), ntype=4),
    ])
    projection = build_node_projection(
        tree,
        0,
        n_players=2,
        base_bet_sizes=(),
        base_subsequent_bet_sizes=(),
        bet_sizes=(0.5,),
        subsequent_bet_sizes=(),
        add_explicit_allin=False,
    )
    solver = CpuEsSolver.__new__(CpuEsSolver)
    sigma = np.asarray([
        [0.25, 0.75],
        [1.0, 0.0],
    ], dtype=np.float32)

    projected, merged, dropped, reasons = (
        solver._project_offtree_strategy(sigma, projection)
    )

    np.testing.assert_allclose(
        projected,
        np.asarray([[1.0], [1.0]], dtype=np.float32),
    )
    assert merged == {0: 0.0, 1: 0.0}
    assert dropped == pytest.approx({0: 0.75})
    assert reasons == {0: "no_base_raise_child"}


def test_extended_tree_child_width_is_rejected_before_solver() -> None:
    tree = SimpleNamespace(
        num_nodes=1,
        nodes=[SimpleNamespace(type=0, num_children=9)],
    )

    with pytest.raises(OffTreeExtensionUnsafe, match="SS_MAX_ACTIONS=8"):
        validate_extended_tree(tree)


def test_extended_tree_width_validation_skips_nondecision_nodes() -> None:
    tree = SimpleNamespace(
        num_nodes=2,
        nodes=[
            SimpleNamespace(type=1, num_children=99),
            SimpleNamespace(type=0, num_children=8),
        ],
    )

    assert validate_extended_tree(tree) is None


def test_extended_tree_width_validation_requires_num_nodes() -> None:
    tree = SimpleNamespace(
        nodes=[SimpleNamespace(type=0, num_children=1)],
    )

    with pytest.raises(AttributeError, match="num_nodes"):
        validate_extended_tree(tree)


def test_solve_time_catalog_guard_rejects_offtree_descriptor(monkeypatch) -> None:
    solver = CpuEsSolver.__new__(CpuEsSolver)
    solver._ready = True
    monkeypatch.setenv("CONT_EV_CATALOG_MANIFEST", "/unused/catalog.json")

    with pytest.raises(RuntimeError, match="off-tree add.*incompatible"):
        solver.solve(
            board_strs=["Ah", "Kh", "2c"],
            board_ints=[48, 44, 1],
            player_ranges=[[], []],
            positions_sorted=["BB", "UTG"],
            player_seats=[1, 2],
            pot_bb=10.0,
            stack_bb=90.0,
            bet_sizes=[0.75, 1.4],
            subsequent_bet_sizes=[1.0],
            add_explicit_allin=False,
            game_action_indices=[],
            iterations=1,
            hero_replay={
                "player_idx": 0,
                "street_actions": [1],
                "expected_actions": 3,
                "offtree": {
                    "base_bet_sizes": [0.75],
                    "base_subsequent_bet_sizes": [1.0],
                    "event_fractions": {0: 1.4},
                },
            },
        )


def test_non_offtree_single_size_guard_text_remains_present() -> None:
    source = inspect.getsource(CpuEsSolver.solve)
    assert "street replay is validated only for single-size" in source
    assert "if (add_explicit_allin or len(bet_sizes) > 1" in source


def _chance_extended_tree() -> SimpleNamespace:
    # Player 1 bets 0.75 x 500 = 375; player 0 faces it: fold / call. The
    # call closes the turn on a river-extended tree, so the call child is a
    # CHANCE node (type 3) whose street bets are reset by the builder.
    nodes = [
        _node(idx=0, player=1, bets=(0, 0), children=(1, 2)),
        _node(idx=1, player=1, bets=(0, 0)),
        _node(idx=2, player=0, bets=(0, 375), children=(3, 4)),
        _node(idx=3, player=-1, bets=(0, 375), active=(False, True), ntype=1),
        _node(idx=4, player=-1, bets=(0, 0), ntype=3),
    ]
    return SimpleNamespace(
        nodes=nodes, num_nodes=len(nodes), starting_pot=500,
        effective_stack=10000,
    )


def test_structural_call_accepts_chance_child_on_extended_tree() -> None:
    from app.engine.offtree_replay import _structural_nonraise_children

    tree = _chance_extended_tree()
    assert _structural_nonraise_children(tree, 2, 2) == (0, 1)


def test_replay_walk_records_path_and_hero_frozen_steps() -> None:
    tree = _chance_extended_tree()
    walk = replay_offtree_street_node(
        tree, [1, 1], n_players=2,
        base_bet_sizes=(0.75,), base_subsequent_bet_sizes=(1.0,),
        bet_sizes=(0.75,), subsequent_bet_sizes=(1.0,),
        add_explicit_allin=False, event_fractions={},
        expected_event_amounts={},
    )
    # step 0: player 1 bets (node 0 -> child 1 -> node 2); step 1: player 0
    # calls (node 2 -> child 1 -> chance node 4).
    assert walk.path == ((0, 1, 1), (2, 0, 1))
    assert walk.node_index == 4
    # Player 0's only decision on the path is node 2; player 1's is node 0.
    assert walk.hero_frozen_steps(0) == ((2, 1),)
    assert walk.hero_frozen_steps(1) == ((0, 1),)


def test_hero_frozen_steps_never_includes_the_served_node() -> None:
    tree = _chance_extended_tree()
    walk = replay_offtree_street_node(
        tree, [1], n_players=2,
        base_bet_sizes=(0.75,), base_subsequent_bet_sizes=(1.0,),
        bet_sizes=(0.75,), subsequent_bet_sizes=(1.0,),
        add_explicit_allin=False, event_fractions={},
        expected_event_amounts={},
    )
    assert walk.node_index == 2
    assert walk.path == ((0, 1, 1),)
    assert walk.hero_frozen_steps(0) == ()
    assert walk.hero_frozen_steps(1) == ((0, 1),)


def test_solve_freezes_hero_line_before_running_cfr() -> None:
    """Hero's earlier street decisions are frozen from the replay walk BEFORE
    the CFR runs (Pluribus safe re-solving), gated by NEXUS_FREEZE_HERO_LINE,
    and reported in the solve stats."""
    source = inspect.getsource(CpuEsSolver.solve)
    i_freeze = source.index("gpu.set_frozen_action_indices(_frozen_steps)")
    i_run = source.index("cs.run(iterations")
    assert i_freeze < i_run
    assert "hero_frozen_steps(" in source
    assert "NEXUS_FREEZE_HERO_LINE" in source
    assert '"hero_frozen_steps": int(len(_frozen_steps))' in source


def test_solve_exposes_chance_sampling_knob() -> None:
    source = inspect.getsource(CpuEsSolver.solve)
    assert "NEXUS_CHANCE_SAMPLE" in source
    assert "cs.set_chance_mode(0 if _chance_sampled else 1)" in source
