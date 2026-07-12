"""Unit tests for the LAZY leaf-rollout serving mode (NEXUS_LEAF_ROLLOUT).

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
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from app.engine import cpu_es_solver as ces
from app.engine.cpu_es_solver import CpuEsSolver


# ── Fakes ────────────────────────────────────────────────────────────────
class _FakeNode:
    def __init__(self, ntype, leaf_idx=-1, active=None):
        self.type = ntype
        self.leaf_idx = leaf_idx
        self.active_players = active if active is not None else []
        self.player = -1
        self.num_children = 0


class _FakeTree:
    """Minimal ss_tree stand-in. Empty decision_node_indices makes the
    strategy-extraction tail trivial (node_idx None -> raw={})."""

    def __init__(self, n_players, n_leaves=2, n_nodes=50, has_chance=False):
        self.num_leaf_nodes = n_leaves
        self.num_leaves = n_leaves
        self.num_nodes = n_nodes
        self.num_chance_nodes = 1 if has_chance else 0
        self.num_decision_nodes = 4
        self.max_depth = 3
        self.decision_node_indices = []  # trivial extraction tail
        # One LEAF node (type==4) per leaf_idx carrying an active vector, so
        # _build_real_oracle_leaf_state's per-leaf active read finds them.
        self.nodes = []
        for li in range(n_leaves):
            self.nodes.append(
                _FakeNode(4, leaf_idx=li, active=[1] * n_players))
        # pad with non-leaf nodes up to num_nodes
        while len(self.nodes) < n_nodes:
            self.nodes.append(_FakeNode(0))


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
    last_from_real_oracle = None

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
    def from_real_oracle(cls, tree, bp_handle, **kw):
        cls.last_from_real_oracle = {"tree": tree, "bp_handle": bp_handle, **kw}
        return cls()

    # ── solve surface ──
    def set_cfr_mode(self, m): pass
    def set_vec_mode(self, m): pass
    def run(self, iterations, num_threads=0): pass
    def avg_strategy(self, ni): return None
    def final_strategy(self, ni): return None
    def real_oracle_stats(self):
        return (7, 3, 21, 0)


class _FakeRollout:
    """Fake isolated rollout_c. Records the eager-fill call; provides the
    cont_ev setters as no-ops (unused when unarmed)."""

    def __init__(self):
        self.eager_fill_calls = 0

    def compute_subgame_leaf_values_sampled_c(self, **kw):
        self.eager_fill_calls += 1
        n_players = len(kw["player_hands"])
        return np.zeros((2, n_players, 6), dtype=np.float32)

    def compute_subgame_leaf_values_sampled_perleaf_c(self, **kw):
        self.eager_fill_calls += 1
        n_players = len(kw["player_hands"])
        return np.zeros((2, n_players, 6), dtype=np.float32)


class _FakeBpHandle:
    def __init__(self, addr=0xDEADBEEF):
        self.value = addr


def _make_solver(n_players, tree, rc, cev_holder=None):
    """CpuEsSolver with _init bypassed and the seam wired to fakes."""
    s = CpuEsSolver.__new__(CpuEsSolver)
    s._ready = True
    s._rollout_c = rc
    s._CSolver = _FakeSolver
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
    s._last_solve_stats = None

    tree_for_snapshot = tree
    s._StreetSolverGPU = lambda **kw: _FakeGpu(tree)
    s._snapshot_tree = lambda t, ranges: tree_for_snapshot
    s._build_labels = lambda t, ni: []

    def _extract(t, num_players, postflop_bet_sizes, subsequent_bet_sizes):
        return [
            _FakeLeafInfo(0, pot=200, bets=[100] * num_players, hist=[1, 2]),
            _FakeLeafInfo(1, pot=400, bets=[200] * num_players, hist=[3]),
        ]
    s._extract_leaf_info = _extract
    return s


def _solve_kwargs(n_players, street="flop"):
    board = {"flop": ["Ah", "Kh", "2c"],
             "turn": ["Ah", "Kh", "2c", "7d"]}[street]
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


@pytest.fixture(autouse=True)
def _reset_fake_solver():
    _FakeSolver.last_from_table = None
    _FakeSolver.last_from_real_oracle = None
    # never leave NEXUS_LEAF_ROLLOUT set across tests
    yield
    os.environ.pop("NEXUS_LEAF_ROLLOUT", None)


# ── flag parsing ──────────────────────────────────────────────────────────
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


# ── interaction guards fail loud ───────────────────────────────────────────
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
