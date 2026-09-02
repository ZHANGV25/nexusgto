"""Unit tests for the lazy and sampled leaf-oracle serving modes.

Mocked at the seam — NO blueprint, NO C libraries. app.engine.cpu_es_solver
imports only stdlib + numpy at module scope (the C/blueprint loading is inside
CpuEsSolver._init), so these tests bypass _init via __new__ and drive solve()
against fake collaborators.

Coverage:
  * flag-off default identity: the EAGER fill
    (compute_subgame_leaf_values_sampled_c) is what runs, from_table constructs,
    from_real_oracle is NEVER touched, leaf_mode=="table_eager".
  * flag-on routing: the eager fill is SKIPPED, from_real_oracle constructs with
    faithfully-mapped arguments (bp_handle.value, street code, stack, rollouts,
    cache_k, master_seed, per-leaf arrays), leaf_mode=="rollout_lazy".
  * per-leaf state array mapping (_build_real_oracle_leaf_state).
  * interaction guards fail loud: rollout + armed cont_ev / bias-action /
    miss-fallback / 2p range-conditioning.
  * stats recording from real_oracle_stats().
  * sampled-mode identity, routing, scalar precedence, deadline/stats/guards.
  * solve-level belief-stash hook success + signature-drift fail-safe.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.engine import cpu_es_solver as ces
from app.engine.cpu_es_solver import CpuEsSolver


# ── Fakes ────────────────────────────────────────────────────────────────
class _FakeNode:
    def __init__(
        self, ntype, leaf_idx=-1, active=None, *, idx=-1, player=-1,
        children=(), pot=0, bets=None, num_board=3, num_players=0,
    ):
        self.type = ntype
        self.leaf_idx = leaf_idx
        self.active_players = active if active is not None else []
        self.idx = idx
        self.player = player
        self.children = tuple(children)
        self.num_children = len(self.children)
        self.pot = pot
        self.bets = tuple(bets if bets is not None else [0] * num_players)
        self.active = tuple(bool(value) for value in self.active_players)
        self.num_board = num_board
        self.fold_player = -1
        self.num_players = num_players


class _FakeTree:
    """Minimal ss_tree stand-in. Empty decision_node_indices makes the
    strategy-extraction tail trivial (node_idx None -> raw={})."""

    def __init__(
        self, n_players, n_leaves=2, n_nodes=50, has_chance=False,
        with_decision_root=False,
    ):
        self.num_leaf_nodes = n_leaves
        self.num_leaves = n_leaves
        self.num_nodes = n_nodes
        self.num_chance_nodes = 1 if has_chance else 0
        self.num_decision_nodes = 1 if with_decision_root else 4
        self.max_depth = 3
        self.num_players = n_players
        self.num_board = 3
        self.board = (0, 1, 2)
        self.starting_pot = 2000
        self.effective_stack = 9500
        self.player_hands = [
            [(0, 1, 1.0), (2, 3, 1.0)] for _ in range(n_players)
        ]
        self.num_hands = [len(hands) for hands in self.player_hands]
        self.max_hands = max(self.num_hands)
        active = [1] * n_players
        bets = [0] * n_players
        if with_decision_root:
            # The stash hook needs a PlainTree-shaped root decision and leaf
            # boundaries. Disconnected padding below proves the packer prunes.
            self.decision_node_indices = [0]
            self.nodes = [
                _FakeNode(
                    0, active=active, idx=0, player=0, children=(1, 2),
                    pot=2000, bets=bets, num_board=3,
                    num_players=n_players,
                ),
                _FakeNode(
                    4, leaf_idx=0, active=active, idx=1, pot=2000,
                    bets=bets, num_board=3, num_players=n_players,
                ),
                _FakeNode(
                    4, leaf_idx=1, active=active, idx=2, pot=2000,
                    bets=bets, num_board=3, num_players=n_players,
                ),
            ]
        else:
            self.decision_node_indices = []  # trivial extraction tail
            # One LEAF node (type==4) per leaf_idx carrying an active vector,
            # so _build_real_oracle_leaf_state's per-leaf read finds them.
            self.nodes = [
                _FakeNode(
                    4, leaf_idx=li, active=active, idx=li, pot=2000,
                    bets=bets, num_board=3, num_players=n_players,
                )
                for li in range(n_leaves)
            ]
        # pad with non-leaf nodes up to num_nodes
        while len(self.nodes) < n_nodes:
            idx = len(self.nodes)
            self.nodes.append(
                _FakeNode(
                    0, active=active, idx=idx, pot=2000, bets=bets,
                    num_board=3, num_players=n_players,
                )
            )


class _FakeGpu:
    def __init__(self, tree, max_hands=6):
        self._tree = tree
        self._max_hands = max_hands
        self._freed = False


class _FakeLeafInfo:
    def __init__(self, leaf_idx, pot, bets, hist):
        self.leaf_idx = leaf_idx
        self.pot = pot
        self.bets = bets
        self.flop_history_train = hist
        self.flop_actor_train = []


class _FakeSolver:
    """Records construction + exposes the run/read surface solve() calls."""

    last_from_table = None
    last_from_bias_table = None
    last_from_real_oracle = None
    last_from_sampled_oracle = None
    last_from_table_with_root_pots = None
    avg_calls = []
    final_calls = []
    cfr_calls = []
    vec_calls = []
    chance_calls = []
    opp_calls = []
    reach_calls = []
    deadline_calls = []
    actual_hand_calls = []
    strategy_sum_mass_calls = []
    strategy_sum_mass_value = 321.0
    iterations_run_calls = 0
    sampled_stats_calls = 0
    bias_stats_calls = 0
    bias_stats = (13, 0)
    events = []
    iterations_completed = 1
    avg_sigma = np.asarray([[0.8, 0.2], [0.35, 0.65]], dtype=np.float32)
    final_sigma = np.asarray([[0.1, 0.9], [0.6, 0.4]], dtype=np.float32)

    def __init__(self):
        self._reach = False

    # ── construction classmethods (recorded on the class) ──
    @classmethod
    def from_table(cls, tree, table, n_players, max_h, seed=1):
        cls.last_from_table = {
            "tree": tree, "table": table, "n_players": n_players,
            "max_h": max_h, "seed": seed}
        return cls()

    @classmethod
    def from_table_with_root_pots(
        cls,
        tree,
        table,
        n_players,
        max_h,
        root_pot_amounts,
        root_pot_eligible_masks,
        seed=1,
    ):
        cls.last_from_table_with_root_pots = {
            "tree": tree,
            "table": table,
            "n_players": n_players,
            "max_h": max_h,
            "root_pot_amounts": tuple(root_pot_amounts),
            "root_pot_eligible_masks": tuple(root_pot_eligible_masks),
            "seed": seed,
        }
        return cls()

    @classmethod
    def from_bias_table(cls, tree, values, **kw):
        cls.last_from_bias_table = {"tree": tree, "values": values, **kw}
        return cls()

    @classmethod
    def from_real_oracle(cls, tree, bp_handle, **kw):
        cls.last_from_real_oracle = {"tree": tree, "bp_handle": bp_handle, **kw}
        return cls()

    @classmethod
    def from_sampled_oracle(cls, tree, bp_handle, **kw):
        cls.last_from_sampled_oracle = {
            "tree": tree, "bp_handle": bp_handle, **kw}
        return cls()

    # ── solve surface ──
    def set_cfr_mode(self, m):
        type(self).cfr_calls.append(m)

    def set_vec_mode(self, m):
        type(self).vec_calls.append(m)

    def set_chance_mode(self, m):
        type(self).chance_calls.append(m)

    def set_opp_mode(self, m):
        type(self).opp_calls.append(m)

    def set_reach_mode(self, m):
        type(self).reach_calls.append(m)

    def set_deadline_ms(self, ms):
        type(self).deadline_calls.append(ms)
        type(self).events.append(("deadline", ms))

    def set_traverser_actual_hand(self, player_idx, hand_idx, fraction):
        type(self).actual_hand_calls.append(
            (player_idx, hand_idx, fraction)
        )

    def strategy_sum_mass(self, node_idx, hand_idx):
        type(self).strategy_sum_mass_calls.append((node_idx, hand_idx))
        return type(self).strategy_sum_mass_value

    def run(self, iterations, num_threads=0):
        type(self).events.append(("run", iterations, num_threads))

    def iterations_run(self):
        type(self).iterations_run_calls += 1
        type(self).events.append(("iterations_run",))
        return type(self).iterations_completed

    def avg_strategy(self, ni):
        type(self).avg_calls.append(ni)
        return type(self).avg_sigma

    def final_strategy(self, ni):
        type(self).final_calls.append(ni)
        return type(self).final_sigma

    def real_oracle_stats(self):
        return (7, 3, 21, 0)

    def sampled_oracle_stats(self):
        type(self).sampled_stats_calls += 1
        return (11, 33, 2, 4, 12, 7, 21)

    def bias_oracle_stats(self):
        type(self).bias_stats_calls += 1
        return type(self).bias_stats


class _FakeRollout:
    """Fake isolated rollout_c. Records the eager-fill call; provides the
    cont_ev setters as no-ops (unused when unarmed)."""

    def __init__(self):
        self.eager_fill_calls = 0
        self.sampled_fill_calls = 0
        self.sampled_perleaf_fill_calls = 0
        self.bias4_fill_calls = 0
        self.bias4_perleaf_fill_calls = 0
        self.joint_fill_calls = 0
        self.last_fill_kwargs = None

    def compute_subgame_leaf_values_sampled_c(self, **kw):
        self.eager_fill_calls += 1
        self.sampled_fill_calls += 1
        self.last_fill_kwargs = kw
        n_players = len(kw["player_hands"])
        return np.zeros((2, n_players, 6), dtype=np.float32)

    def compute_subgame_leaf_values_sampled_perleaf_c(self, **kw):
        self.eager_fill_calls += 1
        self.sampled_perleaf_fill_calls += 1
        self.last_fill_kwargs = kw
        n_players = len(kw["player_hands"])
        return np.zeros((2, n_players, 6), dtype=np.float32)

    def compute_subgame_leaf_values_bias4(self, **kw):
        self.bias4_fill_calls += 1
        self.last_fill_kwargs = kw
        n_players = len(kw["player_hands"])
        return np.zeros((2, 4, n_players, 6), dtype=np.float32)

    def compute_subgame_leaf_values_bias4_perleaf(self, **kw):
        self.bias4_perleaf_fill_calls += 1
        self.last_fill_kwargs = kw
        n_players = len(kw["player_hands"])
        return np.zeros((2, 4, n_players, 6), dtype=np.float32)

    def compute_subgame_leaf_values_c(self, **kw):
        self.joint_fill_calls += 1
        self.last_fill_kwargs = kw
        n_players = len(kw["player_hands"])
        return np.zeros((2 * 16, n_players, 6), dtype=np.float32)


class _FakeBpHandle:
    def __init__(self, addr=0xDEADBEEF):
        self.value = addr


def _make_solver(n_players, tree, rc, cev_holder=None):
    """CpuEsSolver with _init bypassed and the seam wired to fakes."""
    s = CpuEsSolver.__new__(CpuEsSolver)
    s._ready = True
    s._rollout_c = rc
    s._CSolver = _FakeSolver
    s._es_lib = type(
        "_FakeEsLib", (), {"es_create_with_root_pots": object()}
    )()
    s._get_street_solver_lib = lambda: type(
        "_FakeStreetLib", (), {"ss_build_tree_stacks": object()}
    )()
    s._bp_handle = _FakeBpHandle()
    s._blueprint = object()
    s._cev_holder = cev_holder
    s._train_postflop_sizes = [0.75]
    s._train_subsequent_sizes = [1.0]
    s._cev_bias_semantics = "uniform"
    s._last_cont_ev_stats = None
    s._last_cont_ev_turn_stats = None
    s._last_cont_ev_bias_action_stats = None
    s._last_br_gap_objs = None
    s._last_leaf_rollout_stats = None
    s._last_sampled_stats = None
    s._last_bias_oracle_stats = None
    s._last_solve_stats = None

    tree_for_snapshot = tree
    s._StreetSolverGPU = lambda **kw: _FakeGpu(tree)
    s._snapshot_tree = lambda t, ranges: tree_for_snapshot
    s._build_labels = lambda t, ni, root_stacks_chips=None: []

    def _extract(t, num_players, postflop_bet_sizes, subsequent_bet_sizes):
        if int(t.num_leaf_nodes) == 0:
            return []
        return [
            _FakeLeafInfo(0, pot=200, bets=[100] * num_players, hist=[1, 2]),
            _FakeLeafInfo(1, pot=400, bets=[200] * num_players, hist=[3]),
        ]
    s._extract_leaf_info = _extract
    return s


def _solve_kwargs(n_players, street="flop"):
    board = {"flop": ["Ah", "Kh", "2c"],
             "turn": ["Ah", "Kh", "2c", "7d"],
             "river": ["Ah", "Kh", "2c", "7d", "9s"]}[street]
    ranges = [[(0, 1, 1.0), (2, 3, 1.0)] for _ in range(n_players)]
    return dict(
        board_strs=board, board_ints=list(range(len(board))),
        player_ranges=ranges,
        positions_sorted=["p%d" % i for i in range(n_players)],
        player_seats=list(range(n_players)),
        pot_bb=20.0, stack_bb=95.0, bet_sizes=[0.75],
        subsequent_bet_sizes=[1.0], add_explicit_allin=False,
        game_action_indices=None, iterations=1,
        cache_k=1, num_rollouts=3, master_seed=0xE5,
        vec_mode=0, readout="avg",
    )


_ISOLATED_ENV_KEYS = (
    "NEXUS_LEAF_ROLLOUT",
    "NEXUS_LEAF_CACHE_K",
    "NEXUS_LEAF_SAMPLED",
    "NEXUS_LEAF_SAMPLED_POLICY",
    "NEXUS_SAMPLED_DEADLINE_MS",
    "NEXUS_LEAF_BIAS_CFR",
    "NEXUS_BIAS_CFR_ROUNDS",
    "NEXUS_BIAS_CFR_NR",
    "NEXUS_A3_CONT_EV",
    "NEXUS_A3_CONT_EV_TURN",
    "NEXUS_CONT_EV_BIAS_ACTION",
    "NEXUS_CONT_EV_MISS_FALLBACK",
    "CONT_EV_CATALOG_MANIFEST",
    "NEXUS_LEAF_RANGE_CONDITIONING",
    "NEXUS_ALLOW_CHANCE_SOLVE",
    "NEXUS_ES_REACH",
    "NEXUS_BR_GAP_EXPOSE",
)


@pytest.fixture(autouse=True)
def _reset_fake_solver(monkeypatch):
    _FakeSolver.last_from_table = None
    _FakeSolver.last_from_bias_table = None
    _FakeSolver.last_from_real_oracle = None
    _FakeSolver.last_from_sampled_oracle = None
    _FakeSolver.last_from_table_with_root_pots = None
    _FakeSolver.avg_calls = []
    _FakeSolver.final_calls = []
    _FakeSolver.cfr_calls = []
    _FakeSolver.vec_calls = []
    _FakeSolver.chance_calls = []
    _FakeSolver.opp_calls = []
    _FakeSolver.reach_calls = []
    _FakeSolver.deadline_calls = []
    _FakeSolver.actual_hand_calls = []
    _FakeSolver.strategy_sum_mass_calls = []
    _FakeSolver.strategy_sum_mass_value = 321.0
    _FakeSolver.iterations_run_calls = 0
    _FakeSolver.sampled_stats_calls = 0
    _FakeSolver.bias_stats_calls = 0
    _FakeSolver.bias_stats = (13, 0)
    _FakeSolver.events = []
    _FakeSolver.iterations_completed = 1
    # Measurement/serving env from the invoking box must not select a mode or
    # trip a guard in these pure fake-backed tests. monkeypatch restores every
    # original value after the test, including values set by the test itself.
    for key in _ISOLATED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEXUS_LEAF_RANGE_CONDITIONING", "0")


# ── flag parsing ──────────────────────────────────────────────────────────
def test_unequal_stack_river_uses_both_additive_factories():
    tree = _FakeTree(3, n_leaves=0, n_nodes=3)
    tree.starting_pot = 700
    rollout = _FakeRollout()
    solver = _make_solver(3, tree, rollout)
    captured_tree_kwargs = {}

    def build_tree(**kwargs):
        captured_tree_kwargs.update(kwargs)
        gpu = _FakeGpu(tree)
        gpu.root_stacks_chips = tuple(
            int(round(float(value) * 100.0))
            for value in kwargs["root_stacks_bb"]
        )
        return gpu

    solver._StreetSolverGPU = build_tree
    kwargs = _solve_kwargs(3, street="river")
    kwargs.update(
        pot_bb=7.0,
        stack_bb=0.0,
        root_stacks_bb=[0.0, 2.0, 3.0],
        root_pot_amounts=[300, 400],
        root_pot_eligible_masks=[0b111, 0b110],
    )

    solver.solve(**kwargs)

    root_call = _FakeSolver.last_from_table_with_root_pots
    assert captured_tree_kwargs["root_stacks_bb"] == (0.0, 2.0, 3.0)
    assert captured_tree_kwargs["use_cont_strats"] is False
    assert _FakeSolver.last_from_table is None
    assert root_call is not None
    assert root_call["tree"] is tree
    assert root_call["table"].shape == (0, 3, 6)
    assert root_call["n_players"] == 3
    assert root_call["max_h"] == 6
    assert root_call["root_pot_amounts"] == (300, 400)
    assert root_call["root_pot_eligible_masks"] == (0b111, 0b110)
    assert root_call["seed"] == 1
    assert tree.root_stacks_chips == (0, 200, 300)
    assert tree.root_pot_amounts == (300, 400)
    assert tree.root_pot_eligible_masks == (0b111, 0b110)


def test_unequal_stack_capability_probe_requires_both_native_symbols():
    tree = _FakeTree(3, n_leaves=0, n_nodes=3)
    solver = _make_solver(3, tree, _FakeRollout())
    assert solver.supports_unequal_stack_abi() is True

    solver._es_lib = object()
    assert solver.supports_unequal_stack_abi() is False

    solver._es_lib = type(
        "_FakeEsLib", (), {"es_create_with_root_pots": object()}
    )()
    solver._CSolver = object()
    assert solver.supports_unequal_stack_abi() is False

    solver._CSolver = _FakeSolver
    solver._get_street_solver_lib = lambda: object()
    assert solver.supports_unequal_stack_abi() is False

    solver._get_street_solver_lib = lambda: type(
        "_FakeStreetLib", (), {"ss_build_tree_stacks": object()}
    )()
    solver._StreetSolverGPU = lambda board: None
    assert solver.supports_unequal_stack_abi() is False

    solver._StreetSolverGPU = lambda **kwargs: None
    solver._build_labels = lambda tree, node_idx: []
    assert solver.supports_unequal_stack_abi() is False

    solver._build_labels = (
        lambda tree, node_idx, root_stacks_chips=None: []
    )

    class _StaleCSolver:
        @classmethod
        def from_table_with_root_pots(cls, tree, table, n_players, max_h):
            return cls()

    solver._CSolver = _StaleCSolver
    assert solver.supports_unequal_stack_abi() is False

    def failed_street_probe():
        raise OSError("synthetic load failure")

    solver._CSolver = _FakeSolver
    solver._get_street_solver_lib = failed_street_probe
    assert solver.supports_unequal_stack_abi() is False


def test_scalar_river_keeps_historical_tree_and_solver_factories():
    tree = _FakeTree(3, n_leaves=0, n_nodes=3)
    solver = _make_solver(3, tree, _FakeRollout())
    captured_tree_kwargs = {}

    def build_tree(**kwargs):
        captured_tree_kwargs.update(kwargs)
        return _FakeGpu(tree)

    solver._StreetSolverGPU = build_tree

    solver.solve(**_solve_kwargs(3, street="river"))

    assert "root_stacks_bb" not in captured_tree_kwargs
    assert captured_tree_kwargs["use_cont_strats"] is True
    assert _FakeSolver.last_from_table is not None
    assert _FakeSolver.last_from_table_with_root_pots is None


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("NEXUS_LEAF_ROLLOUT", raising=False)
    assert ces._leaf_rollout_enabled() is False


@pytest.mark.parametrize("val", ["1", "t", "T", "y", "Y", "true", "yes"])
def test_flag_on_values(monkeypatch, val):
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", val)
    assert ces._leaf_rollout_enabled() is True


@pytest.mark.parametrize("val", ["0", "f", "n", "N", "", "off"])
def test_flag_off_values(monkeypatch, val):
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", val)
    assert ces._leaf_rollout_enabled() is False


# ── default-off identity: eager fill runs, from_real_oracle untouched ──────
def test_flag_off_uses_eager_fill(monkeypatch):
    monkeypatch.delenv("NEXUS_LEAF_ROLLOUT", raising=False)
    # 3p flop avoids the 2p range-conditioning branch entirely.
    monkeypatch.setenv("NEXUS_LEAF_RANGE_CONDITIONING", "0")
    n = 3
    tree = _FakeTree(n)
    rc = _FakeRollout()
    s = _make_solver(n, tree, rc)

    s.solve(**_solve_kwargs(n))

    assert rc.eager_fill_calls == 1, "eager fill MUST run when flag off"
    assert _FakeSolver.last_from_table is not None, "from_table constructs"
    assert _FakeSolver.last_from_real_oracle is None, "rollout path untouched"
    assert s._last_solve_stats["leaf_mode"] == "table_eager"
    assert s._last_solve_stats["leaf_rollout_stats"] is None


def test_actual_hero_row_config_and_coverage_mass_are_plumbed(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_RANGE_CONDITIONING", "0")
    tree = _FakeTree(2, with_decision_root=True)
    s = _make_solver(2, tree, _FakeRollout())
    s._build_labels = lambda _tree, _node: ["Check", "Bet 75%"]
    kwargs = _solve_kwargs(2)
    kwargs["player_ranges"][0][1] = (2, 3, 0.0)
    kwargs["hero_actual_hand"] = {
        "player_idx": 0,
        "hand_idx": 1,
        "traverser_fraction": 0.5,
    }

    result = s.solve(**kwargs)

    assert _FakeSolver.actual_hand_calls == [(0, 1, 0.5)]
    assert _FakeSolver.strategy_sum_mass_calls == [(0, 1)]
    assert s._last_solve_stats["hero_row_strategy_sum_mass"] == 321.0
    assert s._last_solve_stats["hero_row_strategy_sum_mass_by_row"] == {
        1: 321.0,
    }
    assert 1 in result[0], "the configured actual row bypasses weight-0 filtering"


# ── flag-on: eager fill SKIPPED, from_real_oracle with mapped args ─────────
def test_flag_on_uses_real_oracle(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", "1")
    n = 3
    tree = _FakeTree(n)
    rc = _FakeRollout()
    s = _make_solver(n, tree, rc)

    kwargs = _solve_kwargs(n)
    s.solve(**kwargs)

    assert rc.eager_fill_calls == 0, "eager fill MUST be skipped when flag on"
    assert _FakeSolver.last_from_table is None, "from_table NOT used"
    call = _FakeSolver.last_from_real_oracle
    assert call is not None, "from_real_oracle constructs"
    # faithful argument mapping
    assert call["bp_handle"] == 0xDEADBEEF          # bp_handle.value
    assert call["street"] == 1                        # flop -> 1
    assert call["stack_chips"] == int(round(95.0 * 100))
    assert call["num_rollouts"] == 3
    assert call["cache_k"] == 1                        # SAME as eager num_samples_k
    assert call["master_seed"] == 0xE5
    assert call["max_h"] == 6
    assert call["n_leaves"] == 2
    # per-leaf arrays present + correctly typed
    assert call["pot"].tolist() == [200, 400]
    assert call["player_seats"].tolist() == [0, 1, 2]
    assert s._last_solve_stats["leaf_mode"] == "rollout_lazy"


def test_flag_on_turn_street_code(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", "1")
    n = 3
    tree = _FakeTree(n)
    s = _make_solver(n, tree, _FakeRollout())
    s.solve(**_solve_kwargs(n, street="turn"))
    assert _FakeSolver.last_from_real_oracle["street"] == 2  # turn -> 2


def test_flag_on_records_oracle_stats(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", "1")
    n = 3
    s = _make_solver(n, _FakeTree(n), _FakeRollout())
    s.solve(**_solve_kwargs(n))
    st = s._last_leaf_rollout_stats
    assert st == {"leaf_fills": 7, "cache_hits": 3,
                  "rollout_sims": 21, "oracle_errors": 0}
    assert s._last_solve_stats["leaf_rollout_stats"] == st


# ── fresh per-visit sampled leaf oracle ─────────────────────────────────────
def test_sampled_off_is_eager_noop_even_with_sampled_knobs(monkeypatch):
    """Sampled-only sub-knobs are inert until the mode flag is enabled."""
    n = 3
    baseline_rc = _FakeRollout()
    baseline = _make_solver(n, _FakeTree(n), baseline_rc)
    baseline_result = baseline.solve(**_solve_kwargs(n))

    # Deliberately invalid values prove neither parser nor sampled CSolver API
    # is touched while the default-off mode flag is absent.
    monkeypatch.setenv("NEXUS_LEAF_SAMPLED_POLICY", "not-an-int")
    monkeypatch.setenv("NEXUS_SAMPLED_DEADLINE_MS", "also-not-an-int")
    probed_rc = _FakeRollout()
    probed = _make_solver(n, _FakeTree(n), probed_rc)
    probed_result = probed.solve(**_solve_kwargs(n))

    assert probed_result == baseline_result
    assert baseline_rc.eager_fill_calls == 1
    assert probed_rc.eager_fill_calls == 1
    assert _FakeSolver.last_from_table is not None
    assert _FakeSolver.last_from_real_oracle is None
    assert _FakeSolver.last_from_sampled_oracle is None
    assert _FakeSolver.deadline_calls == []
    assert _FakeSolver.iterations_run_calls == 0
    assert _FakeSolver.sampled_stats_calls == 0
    assert probed._last_sampled_stats is None
    assert probed._last_solve_stats["leaf_mode"] == "table_eager"
    assert probed._last_solve_stats["sampled_stats"] is None
    assert probed._last_solve_stats["sampled_policy"] is None
    assert probed._last_solve_stats["sampled_deadline_ms"] is None
    assert probed._last_solve_stats["iterations_run_observed"] is False


def test_eager_measurement_observes_deadline_iterations_and_phases():
    n = 3
    rc = _FakeRollout()
    solver = _make_solver(n, _FakeTree(n), rc)
    kwargs = _solve_kwargs(n)
    _FakeSolver.iterations_completed = kwargs["iterations"]
    phases = []

    solver.solve(
        **kwargs, deadline_ms=0,
        progress_callback=lambda phase, payload: phases.append((phase, payload)),
    )

    assert _FakeSolver.deadline_calls == [0]
    assert _FakeSolver.iterations_run_calls == 1
    assert solver._last_solve_stats["iterations_run"] == kwargs["iterations"]
    assert solver._last_solve_stats["iterations_run_observed"] is True
    assert solver._last_solve_stats["deadline_ms"] == 0
    assert [phase for phase, _payload in phases] == [
        "pre_fill", "post_fill", "pre_es_run", "post_es_run", "post_readout",
    ]


def test_eager_measurement_memory_guard_fails_before_fill():
    n = 3
    rc = _FakeRollout()
    solver = _make_solver(n, _FakeTree(n), rc)
    phases = []

    with pytest.raises(ces.ProjectedFillMemoryExceeded):
        solver.solve(
            **_solve_kwargs(n), max_projected_fill_bytes=1,
            progress_callback=lambda phase, payload: phases.append((phase, payload)),
        )

    assert rc.eager_fill_calls == 0
    assert [phase for phase, _payload in phases] == ["pre_fill"]
    assert solver._last_preflight_stats["projected_fill_bytes"] > 1


@pytest.mark.parametrize(
    ("policy_value", "expected_policy"),
    [(None, 1), ("0", 0)],
)
def test_sampled_routes_with_policy_seed_and_leaf_arrays(
    monkeypatch, policy_value, expected_policy,
):
    monkeypatch.setenv("NEXUS_LEAF_SAMPLED", "1")
    if policy_value is not None:
        monkeypatch.setenv("NEXUS_LEAF_SAMPLED_POLICY", policy_value)
    n = 3
    tree = _FakeTree(n)
    rc = _FakeRollout()
    s = _make_solver(n, tree, rc)
    kwargs = _solve_kwargs(n)

    s.solve(**kwargs)

    assert rc.eager_fill_calls == 0
    assert _FakeSolver.last_from_table is None
    assert _FakeSolver.last_from_real_oracle is None
    call = _FakeSolver.last_from_sampled_oracle
    assert call is not None
    assert call["tree"] is tree
    assert call["bp_handle"] == 0xDEADBEEF
    assert call["street"] == 1
    assert call["stack_chips"] == int(round(95.0 * 100))
    assert call["n_leaves"] == 2
    assert call["max_h"] == 6
    assert call["num_rollouts"] == 3
    assert call["bias_policy"] == expected_policy
    # The sampled oracle has one RNG: the resolver seed. The public solve seed
    # maps to it; cached-oracle-only arguments must never leak into this call.
    assert call["seed"] == kwargs["master_seed"]
    assert "cache_k" not in call
    assert "master_seed" not in call
    assert call["pot"].tolist() == [200, 400]
    assert call["bets"].tolist() == [100, 100, 100, 200, 200, 200]
    assert call["active"].tolist() == [1, 1, 1, 1, 1, 1]
    assert call["hist_len"].tolist() == [2, 1]
    assert call["player_seats"].tolist() == [0, 1, 2]
    assert s._last_solve_stats["leaf_mode"] == "sampled_per_visit"
    assert s._last_solve_stats["sampled_policy"] == expected_policy
    assert s._last_solve_stats["sampled_deadline_ms"] == 0


def test_sampled_records_all_oracle_stats(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_SAMPLED", "1")
    n = 3
    s = _make_solver(n, _FakeTree(n), _FakeRollout())

    s.solve(**_solve_kwargs(n))

    expected = {
        "leaf_visits": 11,
        "rollouts": 33,
        "oracle_errors": 2,
        "uniform_visits": 4,
        "uniform_rollouts": 12,
        "regret_visits": 7,
        "regret_rollouts": 21,
    }
    assert _FakeSolver.sampled_stats_calls == 1
    assert s._last_sampled_stats == expected
    assert s._last_solve_stats["sampled_stats"] == expected


def test_sampled_deadline_wraps_run_and_records_actual_iterations(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_SAMPLED", "1")
    monkeypatch.setenv("NEXUS_SAMPLED_DEADLINE_MS", "250")
    _FakeSolver.iterations_completed = 23
    n = 3
    s = _make_solver(n, _FakeTree(n), _FakeRollout())
    kwargs = _solve_kwargs(n)
    kwargs["iterations"] = 50

    s.solve(**kwargs)

    # num_threads=0 (all cores): sampled mode is EXEMPT from the small-tree
    # single-thread force — per-iteration work is rollout-dominated, so
    # wall-clock scales with threads (2026-07-17 fix; 3p sampled 50k iters
    # measured 836s when forced single-thread).
    assert _FakeSolver.events == [
        ("deadline", 250),
        ("run", 50, 0),
        ("iterations_run",),
    ]
    assert _FakeSolver.deadline_calls == [250]
    assert _FakeSolver.iterations_run_calls == 1
    assert s._last_solve_stats["iterations"] == 50
    assert s._last_solve_stats["iterations_run"] == 23
    assert s._last_solve_stats["sampled_deadline_ms"] == 250


def test_sampled_vec_avg_request_runs_scalar_without_forcing_readout(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_SAMPLED", "1")
    n = 3
    s = _make_solver(n, _FakeTree(n), _FakeRollout())
    kwargs = _solve_kwargs(n)
    kwargs.update(vec_mode=1, readout="avg")

    s.solve(**kwargs)

    assert _FakeSolver.vec_calls == []
    assert _FakeSolver.opp_calls == [0]
    assert _FakeSolver.reach_calls == []
    assert s._last_solve_stats["vec_mode"] == 0
    assert s._last_solve_stats["readout"] == "avg"
    assert s._last_solve_stats["use_reach"] is False


@pytest.mark.parametrize("n_players", [2, 3])
def test_sampled_chance_tree_suppresses_vec_opp_and_reach(
    monkeypatch, n_players,
):
    monkeypatch.setenv("NEXUS_LEAF_SAMPLED", "1")
    monkeypatch.setenv("NEXUS_ALLOW_CHANCE_SOLVE", "1")
    if n_players == 2:
        monkeypatch.setenv("NEXUS_LEAF_RANGE_CONDITIONING", "0")
    tree = _FakeTree(n_players, has_chance=True)
    s = _make_solver(n_players, tree, _FakeRollout())
    kwargs = _solve_kwargs(n_players)
    kwargs.update(vec_mode=1, readout="final")

    s.solve(**kwargs)

    assert _FakeSolver.chance_calls == [1]
    assert _FakeSolver.vec_calls == []
    assert _FakeSolver.opp_calls == [0]
    assert _FakeSolver.reach_calls == []
    assert s._last_solve_stats["vec_mode"] == 0
    assert s._last_solve_stats["use_reach"] is False


@pytest.mark.parametrize(
    ("env_name", "env_value", "n_players"),
    [
        ("NEXUS_LEAF_ROLLOUT", "1", 3),
        ("NEXUS_A3_CONT_EV", "1", 3),
        ("NEXUS_A3_CONT_EV_TURN", "1", 3),
        ("NEXUS_CONT_EV_BIAS_ACTION", "1", 3),
        ("NEXUS_CONT_EV_MISS_FALLBACK", "1", 3),
        ("CONT_EV_CATALOG_MANIFEST", "/unused/catalog.json", 3),
        ("NEXUS_LEAF_RANGE_CONDITIONING", "1", 3),
    ],
)
def test_sampled_rejects_every_forbidden_combination(
    monkeypatch, env_name, env_value, n_players,
):
    monkeypatch.setenv("NEXUS_LEAF_SAMPLED", "1")
    monkeypatch.setenv(env_name, env_value)
    rc = _FakeRollout()
    s = _make_solver(n_players, _FakeTree(n_players), rc)

    with pytest.raises(RuntimeError, match=env_name):
        s.solve(**_solve_kwargs(n_players))

    assert rc.eager_fill_calls == 0
    assert _FakeSolver.last_from_table is None
    assert _FakeSolver.last_from_real_oracle is None
    assert _FakeSolver.last_from_sampled_oracle is None


def test_sampled_conflict_rejected_before_zero_leaf_bypass(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_SAMPLED", "1")
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", "1")
    monkeypatch.setenv("NEXUS_ALLOW_CHANCE_SOLVE", "1")
    n = 3
    tree = _FakeTree(n, n_leaves=0, has_chance=True)
    s = _make_solver(n, tree, _FakeRollout())

    with pytest.raises(RuntimeError, match="NEXUS_LEAF_ROLLOUT"):
        s.solve(**_solve_kwargs(n, street="turn"))


# ── per-leaf state builder ─────────────────────────────────────────────────
def test_build_real_oracle_leaf_state_mapping():
    n = 3
    tree = _FakeTree(n, n_leaves=2)
    s = CpuEsSolver.__new__(CpuEsSolver)
    leaf_infos = [
        _FakeLeafInfo(0, pot=200, bets=[100, 50, 0], hist=[1, 2, 3]),
        _FakeLeafInfo(1, pot=400, bets=[200, 200, 0], hist=[]),
    ]
    (n_leaves, pot, bets, active, hist, hist_len, seats) = (
        s._build_real_oracle_leaf_state(tree, leaf_infos, n, [4, 5, 6]))
    assert n_leaves == 2
    assert pot.tolist() == [200, 400]
    assert bets.tolist() == [100, 50, 0, 200, 200, 0]
    assert active.tolist() == [1, 1, 1, 1, 1, 1]
    assert hist_len.tolist() == [3, 0]
    assert hist[0:3].tolist() == [1, 2, 3]
    assert seats.tolist() == [4, 5, 6]
    # buffer width = ESRO_MAX_FLOP_HIST per leaf
    assert hist.shape[0] == 2 * ces._ESRO_MAX_FLOP_HIST


def test_build_real_oracle_seat_count_mismatch_raises():
    n = 3
    s = CpuEsSolver.__new__(CpuEsSolver)
    leaf_infos = [_FakeLeafInfo(0, 200, [100, 50, 0], [1])]
    with pytest.raises(RuntimeError, match="seat map mismatch"):
        s._build_real_oracle_leaf_state(_FakeTree(n), leaf_infos, n, [0, 1])


# -- solve-level belief stash hook ------------------------------------------
def test_solve_packs_belief_stash_from_avg_when_serving_final(monkeypatch):
    monkeypatch.delenv("CONT_EV_CATALOG_MANIFEST", raising=False)
    n = 3
    tree = _FakeTree(n, with_decision_root=True)
    s = _make_solver(n, tree, _FakeRollout())
    stash = {"entry_slots": [2, 0, 1], "hist_start": 4}
    kwargs = _solve_kwargs(n)
    kwargs["readout"] = "final"

    s.solve(**kwargs, belief_stash_out=stash)

    assert set(stash) == {
        "status", "tree", "sigma", "entry_slots", "player_seats",
        "street", "board", "hist_start",
    }
    assert stash["status"] == "ok"
    assert stash["entry_slots"] == [2, 0, 1]
    assert stash["hist_start"] == 4
    assert stash["player_seats"] == [0, 1, 2]
    assert stash["street"] == 1
    assert stash["board"] == (0, 1, 2)

    pruned = stash["tree"]
    assert pruned is not tree
    assert len(tree.nodes) == 50
    assert len(pruned.nodes) == 3
    assert pruned.decision_node_indices == (0,)
    assert pruned.nodes[0].children == (1, 2)
    assert [node.type for node in pruned.nodes] == [0, 4, 4]
    assert [node.num_children for node in pruned.nodes] == [2, 0, 0]
    assert set(stash["sigma"]) == {0}
    np.testing.assert_array_equal(stash["sigma"][0], _FakeSolver.avg_sigma)
    assert not np.array_equal(stash["sigma"][0], _FakeSolver.final_sigma)
    assert _FakeSolver.avg_calls == [0]
    assert _FakeSolver.final_calls == [0]


def test_solve_belief_stash_wrong_packer_signature_is_fail_safe(monkeypatch):
    from app.engine.agent import belief_stash as belief_stash_module

    def _wrong_signature(out):
        raise AssertionError("wrong-signature stub must not be entered")

    monkeypatch.delenv("CONT_EV_CATALOG_MANIFEST", raising=False)
    monkeypatch.setattr(
        belief_stash_module, "pack_belief_stash", _wrong_signature,
    )
    n = 3
    tree = _FakeTree(n, with_decision_root=True)
    s = _make_solver(n, tree, _FakeRollout())
    stash = {"entry_slots": [2, 0, 1], "hist_start": 7}

    s.solve(**_solve_kwargs(n), belief_stash_out=stash)

    assert stash["status"] == "empty"
    assert stash["reason"].startswith(
        "belief-stash extraction failed: TypeError:"
    )
    assert "unexpected keyword argument" in stash["reason"]
    assert stash["entry_slots"] == [2, 0, 1]
    assert stash["hist_start"] == 7
    assert stash["street"] == 1
    assert stash["board"] == (0, 1, 2)
    assert "tree" not in stash and "sigma" not in stash


# ── interaction guards fail loud ───────────────────────────────────────────
def test_guard_belief_stash_plus_catalog(monkeypatch):
    monkeypatch.setenv("NEXUS_BELIEF_CARRY", "1")
    monkeypatch.setenv("CONT_EV_CATALOG_MANIFEST", "/unused/catalog.json")
    n = 3
    s = _make_solver(n, _FakeTree(n), _FakeRollout())
    stash = {"entry_slots": list(range(n)), "hist_start": 0}

    with pytest.raises(RuntimeError, match="belief carry.*incompatible"):
        s.solve(**_solve_kwargs(n), belief_stash_out=stash)


def test_guard_rollout_plus_armed_cont_ev_flop(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", "1")
    n = 3  # flop + n>=3 + cev_holder present -> a3 would arm
    tree = _FakeTree(n)
    s = _make_solver(n, tree, _FakeRollout(), cev_holder=object())
    with pytest.raises(RuntimeError, match="cont_ev boundary"):
        s.solve(**_solve_kwargs(n))


def test_guard_rollout_plus_armed_cont_ev_turn(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", "1")
    monkeypatch.setenv("NEXUS_A3_CONT_EV_TURN", "1")
    n = 3
    tree = _FakeTree(n)
    s = _make_solver(n, tree, _FakeRollout(), cev_holder=object())
    try:
        with pytest.raises(RuntimeError, match="cont_ev boundary"):
            s.solve(**_solve_kwargs(n, street="turn"))
    finally:
        monkeypatch.delenv("NEXUS_A3_CONT_EV_TURN", raising=False)


def test_guard_rollout_plus_bias_action(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", "1")
    monkeypatch.setenv("NEXUS_CONT_EV_BIAS_ACTION", "1")
    n = 3
    s = _make_solver(n, _FakeTree(n), _FakeRollout())  # no cev_holder
    try:
        with pytest.raises(RuntimeError, match="BIAS_ACTION"):
            s.solve(**_solve_kwargs(n))
    finally:
        monkeypatch.delenv("NEXUS_CONT_EV_BIAS_ACTION", raising=False)


def test_guard_rollout_plus_miss_fallback(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", "1")
    monkeypatch.setenv("NEXUS_CONT_EV_MISS_FALLBACK", "1")
    n = 3
    s = _make_solver(n, _FakeTree(n), _FakeRollout())
    try:
        with pytest.raises(RuntimeError, match="MISS_FALLBACK"):
            s.solve(**_solve_kwargs(n))
    finally:
        monkeypatch.delenv("NEXUS_CONT_EV_MISS_FALLBACK", raising=False)


def test_guard_rollout_plus_2p_conditioning_default(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", "1")
    monkeypatch.delenv("NEXUS_LEAF_RANGE_CONDITIONING", raising=False)  # default ON
    n = 2
    tree = _FakeTree(n)
    s = _make_solver(n, tree, _FakeRollout())
    with pytest.raises(RuntimeError, match="LEAF_RANGE_CONDITIONING=0"):
        s.solve(**_solve_kwargs(n))


def test_rollout_2p_with_conditioning_off_ok(monkeypatch):
    monkeypatch.setenv("NEXUS_LEAF_ROLLOUT", "1")
    monkeypatch.setenv("NEXUS_LEAF_RANGE_CONDITIONING", "0")
    n = 2
    tree = _FakeTree(n)
    rc = _FakeRollout()
    s = _make_solver(n, tree, rc)
    s.solve(**_solve_kwargs(n))
    assert rc.eager_fill_calls == 0
    assert _FakeSolver.last_from_real_oracle is not None
    monkeypatch.delenv("NEXUS_LEAF_RANGE_CONDITIONING", raising=False)
