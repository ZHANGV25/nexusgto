"""Amount-faithful replay and base-vocabulary projection for off-tree solves.

The normal CpuEsSolver replay is intentionally index-positional and remains
the default path.  This module is used only when a solve carries an explicit
``hero_replay["offtree"]`` descriptor: the solve tree then has street-global
added sizes, so recorded base-vocabulary indices can no longer be used as raw
child indices.

All chip arithmetic here mirrors ``ss_tree_builder`` / the agent trainer:
integer truncation, remaining-stack clamp, ``<= to_call`` pruning, and the
first sorted size owning a stack-clamped all-in.  A missing or ambiguous
amount match is unsupported; guessing a child would serve the wrong node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


class ReplayUnsupported(Exception):
    """The extended tree cannot faithfully represent the recorded replay."""


class OffTreeExtensionUnsafe(Exception):
    """An extended tree violates a pre-solver memory-layout invariant."""


@dataclass(frozen=True)
class RaiseAction:
    """One surviving builder raise entry at a particular decision node."""

    fraction: Optional[float]
    amount: int


@dataclass(frozen=True)
class AddedChild:
    """An added-size child and the fraction that generated it."""

    child_index: int
    fraction: float


@dataclass(frozen=True)
class NodeProjection:
    """Column projection from an extended node onto the base vocabulary.

    ``base_child_indices`` is already in agent/base action order.  Each
    ``base_raise_target`` is ``(output_column, fraction)``.  Added children
    are merged into those targets (or dropped if there is no base fraction).
    ``base_action_labels`` describes that projected order directly; deriving
    labels from extended-tree child positions would mislabel faced nodes.
    """

    base_child_indices: Tuple[int, ...]
    base_raise_targets: Tuple[Tuple[int, float], ...]
    added_children: Tuple[AddedChild, ...]
    base_action_labels: Tuple[str, ...]


@dataclass(frozen=True)
class ReplayWalk:
    """Result of walking the recorded street inside an extended tree."""

    node_index: int
    replayed_through_added: bool
    added_child_chips: Tuple[Tuple[int, int], ...]


def validate_extended_tree(tree: object, max_actions: int = 8) -> None:
    """Reject a built tree whose fixed-width solver layout would overflow."""

    # ``num_nodes`` is an ABI-bearing tree field.  Missing it is convention
    # drift, not an empty tree that can safely bypass validation.
    num_nodes = int(tree.num_nodes)
    nodes = getattr(tree, "nodes")
    for node_index in range(num_nodes):
        node = nodes[node_index]
        # Chance/terminal nodes do not expose solver action columns.  Their
        # child fanout is therefore unrelated to SS_MAX_ACTIONS.
        if int(node.type) != 0:
            continue
        num_children = int(node.num_children)
        if num_children > max_actions:
            raise OffTreeExtensionUnsafe(
                "off-tree extension produced node "
                f"{node_index} with {num_children} children; "
                f"SS_MAX_ACTIONS={max_actions} would be exceeded"
            )


def _node_terms(
    tree: object, node: object, n_players: int,
) -> Tuple[int, int, int, int, int]:
    actor = int(node.player)
    max_bet = max(int(node.bets[q]) for q in range(n_players))
    actor_bet = int(node.bets[actor])
    to_call = max_bet - actor_bet
    pot = int(getattr(tree, "starting_pot")) + sum(
        int(node.bets[q]) for q in range(n_players)
    )
    remaining = int(getattr(tree, "effective_stack")) - actor_bet
    return actor, max_bet, to_call, pot if pot > 0 else 0, remaining


def _raise_actions(
    tree: object,
    node: object,
    n_players: int,
    sizes: Sequence[float],
    add_explicit_allin: bool,
) -> Tuple[RaiseAction, ...]:
    """Enumerate surviving sizes with the builder's exact prune/order rules."""

    _actor, _max_bet, to_call, pot, remaining = _node_terms(
        tree, node, n_players
    )
    if remaining <= to_call:
        return ()

    actions: List[RaiseAction] = []
    added_allin = False
    for raw_fraction in sorted(float(value) for value in sizes):
        fraction = float(raw_fraction)
        if to_call == 0:
            amount = int(fraction * pot)
        else:
            amount = to_call + int(fraction * (pot + to_call))
        if amount >= remaining:
            amount = remaining
        if amount <= to_call:
            continue
        if amount >= remaining:
            if added_allin:
                continue
            added_allin = True
        actions.append(RaiseAction(fraction=fraction, amount=int(amount)))

    if add_explicit_allin and not added_allin and remaining > to_call:
        actions.append(RaiseAction(fraction=None, amount=int(remaining)))
    return tuple(actions)


def _structural_nonraise_children(
    tree: object, node_index: int, n_players: int
) -> Tuple[int, ...]:
    node = tree.nodes[node_index]
    actor, max_bet, to_call, _pot, _remaining = _node_terms(
        tree, node, n_players
    )
    n_nonraise = 2 if to_call > 0 else 1
    if int(node.num_children) < n_nonraise:
        raise ValueError(
            f"node {node_index} has {node.num_children} children but needs "
            f"{n_nonraise} structural non-raise children"
        )

    result: List[int] = []
    for child_index in range(n_nonraise):
        child = tree.nodes[node.children[child_index]]
        if to_call > 0 and child_index == 0:
            if child.active[actor] or int(child.bets[actor]) != int(node.bets[actor]):
                raise ValueError(
                    f"node {node_index} child 0 is not a structural fold"
                )
        else:
            if (not child.active[actor]
                    or int(child.bets[actor]) != max_bet):
                raise ValueError(
                    f"node {node_index} child {child_index} is not a "
                    "structural check/call"
                )
        result.append(child_index)
    return tuple(result)


def _raise_child_matches(
    tree: object,
    node_index: int,
    n_players: int,
    amount: int,
) -> Tuple[int, ...]:
    node = tree.nodes[node_index]
    actor, max_bet, to_call, _pot, _remaining = _node_terms(
        tree, node, n_players
    )
    n_nonraise = 2 if to_call > 0 else 1
    matches: List[int] = []
    for child_index in range(n_nonraise, int(node.num_children)):
        child = tree.nodes[node.children[child_index]]
        delta = int(child.bets[actor]) - int(node.bets[actor])
        if delta != int(amount):
            continue
        if not child.active[actor] or int(child.bets[actor]) <= max_bet:
            raise ValueError(
                f"node {node_index} child {child_index} amount-matches "
                "a raise but is not structurally a raise"
            )
        matches.append(child_index)
    return tuple(matches)


def _unique_raise_child(
    tree: object,
    node_index: int,
    n_players: int,
    amount: int,
    *,
    context: str,
) -> int:
    matches = _raise_child_matches(tree, node_index, n_players, amount)
    if len(matches) != 1:
        raise ReplayUnsupported(
            f"{context}: target raise amount {amount} at node {node_index} "
            f"matched {len(matches)} children (expected exactly one)"
        )
    return matches[0]


def _sizes_for_node(
    node: object,
    first_sizes: Sequence[float],
    subsequent_sizes: Sequence[float],
) -> Sequence[float]:
    if max(int(value) for value in node.bets) <= 0:
        return first_sizes
    # The C builder treats an empty/NULL subsequent list as the first-level
    # list (ss_tree_builder.c:195-202).
    return subsequent_sizes if subsequent_sizes else first_sizes


def replay_offtree_street_node(
    tree: object,
    street_actions: Sequence[int],
    *,
    n_players: int,
    base_bet_sizes: Sequence[float],
    base_subsequent_bet_sizes: Sequence[float],
    bet_sizes: Sequence[float],
    subsequent_bet_sizes: Sequence[float],
    add_explicit_allin: bool,
    event_fractions: Mapping[int, float],
) -> ReplayWalk:
    """Walk a street by base-index/observed-amount matching."""

    # Recorded agent raise indices follow the state's list order, while the
    # amount mirror below sorts sizes.  Require the ordering invariant rather
    # than silently translating a base index through a different order.
    for label, values in (
        ("base_bet_sizes", base_bet_sizes),
        ("base_subsequent_bet_sizes", base_subsequent_bet_sizes),
    ):
        normalized = tuple(float(value) for value in values)
        if normalized != tuple(sorted(normalized)):
            raise ReplayUnsupported(
                f"{label} must be ascending-sorted for amount replay"
            )

    node_index = 0
    replayed_added = False
    added_child_chips: Dict[int, int] = {}
    consumed_event_steps: set[int] = set()

    for step, raw_index in enumerate(street_actions):
        node = tree.nodes[node_index]
        if int(node.type) != 0 or int(node.num_children) <= 0:
            raise ReplayUnsupported(
                f"street replay step {step}: node {node_index} is not a "
                f"decision node (type={node.type})"
            )

        actor, max_bet, to_call, _pot, _remaining = _node_terms(
            tree, node, n_players
        )
        n_nonraise = 2 if to_call > 0 else 1
        action_index = int(raw_index)
        if action_index < 0:
            raise ReplayUnsupported(
                f"street replay step {step}: negative action index {action_index}"
            )

        if action_index < n_nonraise:
            if step in event_fractions:
                raise ReplayUnsupported(
                    f"street replay step {step}: off-tree event maps to "
                    f"non-raise action index {action_index}"
                )
            _structural_nonraise_children(tree, node_index, n_players)
            child_index = action_index
        else:
            if int(node.num_children) == n_nonraise:
                raise ReplayUnsupported(
                    f"street replay step {step}: recorded raise "
                    f"(index {action_index}) at raise-capped node "
                    f"{node_index}"
                )
            if step in event_fractions:
                target_fraction = float(event_fractions[step])
                extended_sizes = _sizes_for_node(
                    node, bet_sizes, subsequent_bet_sizes
                )
                candidates = _raise_actions(
                    tree, node, n_players, extended_sizes,
                    add_explicit_allin,
                )
                owned = [
                    action for action in candidates
                    if action.fraction == target_fraction
                ]
                if len(owned) != 1:
                    raise ReplayUnsupported(
                        f"street replay step {step}: added fraction "
                        f"{target_fraction!r} survived {len(owned)} times at "
                        f"node {node_index}"
                    )
                target_amount = owned[0].amount
                child_index = _unique_raise_child(
                    tree, node_index, n_players, target_amount,
                    context=f"street replay step {step} added action",
                )
                replayed_added = True
                consumed_event_steps.add(int(step))
                added_child_chips[int(step)] = int(target_amount)
            else:
                base_sizes = _sizes_for_node(
                    node, base_bet_sizes, base_subsequent_bet_sizes
                )
                survivors = _raise_actions(
                    tree, node, n_players, base_sizes,
                    add_explicit_allin,
                )
                raise_position = action_index - n_nonraise
                if not (0 <= raise_position < len(survivors)):
                    if int(node.num_children) == n_nonraise:
                        raise ReplayUnsupported(
                            f"street replay step {step}: recorded raise "
                            f"(index {action_index}) at raise-capped node "
                            f"{node_index}"
                        )
                    raise ReplayUnsupported(
                        f"street replay step {step}: base raise position "
                        f"{raise_position} outside {len(survivors)} surviving "
                        "base raises"
                    )
                target_amount = survivors[raise_position].amount
                child_index = _unique_raise_child(
                    tree, node_index, n_players, target_amount,
                    context=f"street replay step {step} base action",
                )

        child = tree.nodes[node.children[child_index]]
        if action_index >= n_nonraise:
            if not child.active[actor] or int(child.bets[actor]) <= max_bet:
                raise ValueError(
                    f"street replay step {step}: matched child is not a raise"
                )
        node_index = int(node.children[child_index])

    missing_event_steps = sorted(
        set(int(step) for step in event_fractions) - consumed_event_steps
    )
    if missing_event_steps:
        raise ReplayUnsupported(
            "off-tree replay did not consume event steps "
            f"{missing_event_steps}"
        )

    return ReplayWalk(
        node_index=node_index,
        replayed_through_added=bool(replayed_added),
        added_child_chips=tuple(sorted(added_child_chips.items())),
    )


def build_node_projection(
    tree: object,
    node_index: int,
    *,
    n_players: int,
    base_bet_sizes: Sequence[float],
    base_subsequent_bet_sizes: Sequence[float],
    bet_sizes: Sequence[float],
    subsequent_bet_sizes: Sequence[float],
    add_explicit_allin: bool,
) -> NodeProjection:
    """Describe how one extended node maps back to base action columns."""

    node = tree.nodes[node_index]
    nonraise = list(_structural_nonraise_children(tree, node_index, n_players))
    base_sizes = tuple(float(value) for value in _sizes_for_node(
        node, base_bet_sizes, base_subsequent_bet_sizes
    ))
    extended_sizes = tuple(float(value) for value in _sizes_for_node(
        node, bet_sizes, subsequent_bet_sizes
    ))
    base_actions = _raise_actions(
        tree, node, n_players, base_sizes, add_explicit_allin
    )
    extended_actions = _raise_actions(
        tree, node, n_players, extended_sizes, add_explicit_allin
    )

    _actor, _max_bet, to_call, _pot, remaining = _node_terms(
        tree, node, n_players
    )
    base_children: List[int] = nonraise
    base_labels: List[str] = (
        ["Fold", "Call"] if to_call > 0 else ["Check"]
    )
    base_targets: List[Tuple[int, float]] = []
    used_children = set(nonraise)
    for action in base_actions:
        matches = [
            child_index
            for child_index in _raise_child_matches(
                tree, node_index, n_players, action.amount
            )
            if child_index not in used_children
        ]
        if not matches and int(node.num_children) == len(nonraise):
            # Raise-capped node: a base raise may legitimately be absent.
            # Any planned added action is still required below.
            continue
        if len(matches) != 1:
            raise ReplayUnsupported(
                f"base projection at node {node_index}: amount "
                f"{action.amount} matched {len(matches)} unused children"
            )
        child_index = matches[0]
        used_children.add(child_index)
        output_column = len(base_children)
        base_children.append(child_index)
        if action.fraction is not None:
            base_targets.append((output_column, float(action.fraction)))
        if action.fraction is None or action.amount >= remaining:
            base_labels.append("All-in")
        else:
            verb = "Raise" if to_call > 0 else "Bet"
            base_labels.append(
                f"{verb} {int(round(action.fraction * 100))}%"
            )

    n_nonraise = 2 if to_call > 0 else 1
    added_children: List[AddedChild] = []
    matched_added_actions: Dict[Tuple[float, int], int] = {}
    base_amounts = {int(action.amount) for action in base_actions}
    expected_added_actions = tuple(
        action for action in extended_actions
        if action.fraction is not None
        and action.fraction not in base_sizes
        # A street-global added size can clamp/dedup to a base all-in at a
        # node other than the observation point.  That one real child was
        # intentionally consumed as the base action above; it is not a
        # second missing added child.
        and int(action.amount) not in base_amounts
    )
    for child_index in range(n_nonraise, int(node.num_children)):
        if child_index in used_children:
            continue
        child = tree.nodes[node.children[child_index]]
        actor = int(node.player)
        amount = int(child.bets[actor]) - int(node.bets[actor])
        owners = [
            action for action in extended_actions
            if action.fraction is not None
            and action.fraction not in base_sizes
            and int(action.amount) not in base_amounts
            and action.amount == amount
        ]
        if len(owners) != 1:
            raise ReplayUnsupported(
                f"added projection at node {node_index}: child {child_index} "
                f"amount {amount} maps to {len(owners)} added fractions"
            )
        added_children.append(AddedChild(
            child_index=child_index,
            fraction=float(owners[0].fraction),
        ))
        owner_key = (float(owners[0].fraction), int(owners[0].amount))
        matched_added_actions[owner_key] = (
            matched_added_actions.get(owner_key, 0) + 1
        )

    missing_added_actions = [
        (float(action.fraction), int(action.amount))
        for action in expected_added_actions
        if (float(action.fraction), int(action.amount))
        not in matched_added_actions
    ]
    if missing_added_actions:
        raise ReplayUnsupported(
            f"added projection at node {node_index}: added actions "
            f"{missing_added_actions!r} matched no child"
        )
    duplicate_added_actions = [
        (fraction, amount, count)
        for (fraction, amount), count in matched_added_actions.items()
        if count != 1
    ]
    if duplicate_added_actions:
        raise ReplayUnsupported(
            f"added projection at node {node_index}: added actions "
            f"{duplicate_added_actions!r} did not match exactly one child"
        )

    return NodeProjection(
        base_child_indices=tuple(base_children),
        base_raise_targets=tuple(base_targets),
        added_children=tuple(added_children),
        base_action_labels=tuple(base_labels),
    )
