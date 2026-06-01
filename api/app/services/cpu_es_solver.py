"""CPU external-sampling ES re-solver — production engine for 5p flop/turn.

The HYBRID architecture (see docs/MULTIWAY_5_6P_DECISION_MEMO.md): 2-3p (+4p
flop/turn) stay on the GPU; 5p flop/turn move here — a CPU external-sampling
MCCFR re-solver that runs OFF the GPU lock, freeing the single GPU for the common
interactive 2-3p solves. Benchmarked head-to-head (CPU_ES_RESOLVER_BENCHMARK.md):
with Linear CFR + multi-thread Hogwild it reaches the GPU's 1600-iter ship-budget
exploitability ~1.65× faster at 5p AND concurrently with GPU play.

ISOLATION (load-bearing): this path uses its OWN ctypes bindings into
es_resolver_real.dll and its OWN BPSReaderC loaded through that same DLL. It does
NOT touch the GPU provider's `rollout_c` / `bps_reader_py` module globals — the
benchmark's monkeypatch of those would silently repoint the GPU provider's leaf
construction to a different DLL and feed it a cross-DLL BPSReader* handle. A
dedicated `rollout_c` module instance (via importlib) keeps the construct binding
isolated.

SCOPE (validated regime only): 5p FLOP/TURN — cont-strat depth-limited leaf-table
terminals, the regime every quality gate was measured on. 5p RIVER is NOT served
here (its terminals are showdowns; es_showdown_value is single-pot chop validated
only at 2p, vs the GPU's side-pot showdown — wrong for 5-way unequal all-ins).
River stays GPU-async. 6p stays GPU-async until its iso-quality T is measured.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# poker-solver locations (the provider side-loads python/ onto sys.path at init;
# we resolve the DLL + python dir from the same env the provider uses).
_POKER_SOLVER = os.environ.get(
    "POKER_SOLVER_PATH",
)
# Bare name (no extension); c_solver resolves .dll on Windows / .so on Linux.
_ES_DLL_NAME = "es_resolver_real"


class CpuEsSolver:
    """Isolated CPU-ES engine. Constructed once at provider init (loads a second
    ~12GB blueprint through es_resolver_real.dll); thread-safe per-solve (each
    solve builds its own tree + table + CSolver; the only shared state is the
    read-only blueprint handle)."""

    def __init__(self, blueprint_path: str) -> None:
        self._ready = False
        self._blueprint = None
        self._bp_handle = None
        self._rollout_c = None  # isolated module instance
        self._CSolver = None
        self._snapshot_tree = None
        self._extract_leaf_info = None
        self._build_labels = None
        self._StreetSolverGPU = None
        self._init(blueprint_path)

    def _init(self, blueprint_path: str) -> None:
        py_dir = os.path.join(_POKER_SOLVER, "python")
        if py_dir not in sys.path:
            sys.path.insert(0, py_dir)

        # Force c_solver to load es_resolver_real.dll (rollout + bps + ES in one).
        os.environ["ES_DLL"] = _ES_DLL_NAME
        from es_resolver.c_solver import _lib as es_lib, _HAS_REAL, CSolver  # type: ignore
        if not _HAS_REAL:
            raise RuntimeError(
                "es_resolver_real.dll lacks the real oracle/construct — rebuild it"
            )
        self._CSolver = CSolver
        self._es_lib = es_lib

        # ── Isolated rollout_c bound to es_resolver_real.dll ──
        # Load a SECOND copy of the rollout_c module under a private name so its
        # `_LIB` global points at es_resolver_real.dll WITHOUT mutating the
        # shared rollout_c the GPU provider uses.
        spec = importlib.util.spec_from_file_location(
            "rollout_c_es", os.path.join(py_dir, "rollout_c.py")
        )
        rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rc)
        rc._LIB = es_lib
        rc._setup_ctypes(es_lib)
        self._rollout_c = rc

        # ── Isolated BPSReaderC loaded THROUGH es_resolver_real.dll ──
        # Load a second copy of bps_reader_py whose _find_lib returns es_lib, so
        # the BPSReader* + the construct + the ES all live in one DLL (the
        # validated config) and the shared bps_reader_py is untouched.
        spec2 = importlib.util.spec_from_file_location(
            "bps_reader_py_es", os.path.join(py_dir, "bps_reader_py.py")
        )
        bpm = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(bpm)
        bpm._find_lib = lambda: es_lib  # type: ignore
        t0 = time.time()
        self._blueprint = bpm.BPSReaderC(blueprint_path)
        self._bp_handle = self._blueprint._reader
        logger.info("CpuEsSolver: blueprint loaded through %s in %.1fs",
                    _ES_DLL_NAME, time.time() - t0)

        # Tree builder + helpers (shared, read-only use).
        from street_solver_gpu import StreetSolverGPU, _build_labels_at_node  # type: ignore
        from es_resolver.es_solver import snapshot_tree  # type: ignore
        from leaf_values import extract_leaf_info_from_tree  # type: ignore
        self._StreetSolverGPU = StreetSolverGPU
        self._build_labels = _build_labels_at_node
        self._snapshot_tree = snapshot_tree
        self._extract_leaf_info = extract_leaf_info_from_tree
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def solve(
        self,
        *,
        board_strs: List[str],
        board_ints: List[int],
        player_ranges: List[List[Tuple[int, int, float]]],
        positions_sorted: List[str],
        player_seats: List[int],
        pot_bb: float,
        stack_bb: float,
        bet_sizes: List[float],
        subsequent_bet_sizes: List[float],
        add_explicit_allin: bool,
        game_action_indices: Optional[List[int]],
        iterations: int,
        num_threads: int = 0,
        cache_k: int = 1,
        num_rollouts: int = 3,
        master_seed: int = 0xE5,
    ) -> Dict[int, Dict[int, Dict[str, float]]]:
        """Run a 5p flop/turn CPU-ES solve. Returns all_by_player:
        {player_idx: {hand_idx: {action_label: freq}}} — the SAME structure the
        GPU provider's get_all_avg_strategies produces, so the provider's
        _build_response_strategies maps it identically.

        Builds the tree (CPU; GPU tree-builder, no GPU solve), constructs the
        leaf table (parallel batch = the GPU's exact construct floor, off-lock),
        runs the C ES (Linear CFR, multi-thread Hogwild), extracts σ̄ at each
        player's first decision node with the weight-0 filter the GPU applies.
        """
        if not self._ready:
            raise RuntimeError("CpuEsSolver not initialized")
        rc = self._rollout_c
        n_players = len(player_ranges)
        scale = 100.0
        stack_chips = int(round(stack_bb * scale))
        street = "flop" if len(board_strs) == 3 else ("turn" if len(board_strs) == 4 else "river")
        if street == "river":
            raise ValueError("CpuEsSolver does not serve river (showdowns) — route to GPU-async")

        # ── Build the tree (CPU tree-builder; no GPU solve / no leaf_value_fn) ──
        gpu = self._StreetSolverGPU(
            board=board_strs, pot_bb=pot_bb, stack_bb=stack_bb,
            player_ranges=player_ranges, bet_sizes=bet_sizes,
            subsequent_bet_sizes=subsequent_bet_sizes,
            use_cont_strats=True, leaf_value_fn=None,
            add_explicit_allin=add_explicit_allin,
        )
        try:
            tree = self._snapshot_tree(gpu._tree, player_ranges)
            max_h = gpu._max_hands
            leaf_infos = self._extract_leaf_info(gpu._tree, num_players=n_players)
            if not leaf_infos:
                raise RuntimeError("no leaf infos extracted from tree")

            # ── Construct the leaf table (parallel batch, off-lock) ──
            t0 = time.time()
            table = rc.compute_subgame_leaf_values_sampled_c(
                leaf_infos=leaf_infos, board=board_ints, street=street,
                player_hands=player_ranges, active=[True] * n_players,
                acting_order=list(range(n_players)), player_seats=list(player_seats),
                game_action_history=list(game_action_indices or []),
                blueprint=self._blueprint,
                starting_pot_chips=int(round(pot_bb * scale)),
                stack_chips=stack_chips, max_hands=max_h,
                num_rollouts=num_rollouts, num_samples_k=cache_k,
                master_seed=master_seed,
            )
            construct_s = time.time() - t0

            # ── Solve: C ES, Linear CFR, multi-thread Hogwild ──
            cs = self._CSolver.from_table(gpu._tree, table, n_players, max_h, seed=1)
            cs.set_cfr_mode(1)  # Linear CFR (matches the GPU, ~3x more iter-efficient)
            t1 = time.time()
            cs.run(iterations, num_threads=num_threads)
            solve_s = time.time() - t1

            # ── Extract σ̄ → all_by_player, with the weight-0 / never-visited
            #    filter the GPU's get_all_avg_strategies applies (freq>0.001;
            #    skip uniform-default rows so out-of-range hands stay fold-100%
            #    placeholders in _build_response_strategies). ──
            all_by_player: Dict[int, Dict[int, Dict[str, float]]] = {}
            for p_idx in range(n_players):
                node_idx = self._first_decision_node_for(tree, p_idx)
                raw: Dict[int, Dict[str, float]] = {}
                if node_idx is not None:
                    labels = self._build_labels(gpu._tree, node_idx)
                    sigma = cs.avg_strategy(node_idx)  # [nh, na] or None
                    na = sigma.shape[1] if sigma is not None else 0
                    nh = len(player_ranges[p_idx])
                    for h in range(min(nh, sigma.shape[0] if sigma is not None else 0)):
                        # weight-0 hand → never visited → skip (fold-100% placeholder)
                        if player_ranges[p_idx][h][2] <= 0.0:
                            continue
                        strat = {}
                        for a in range(min(na, len(labels))):
                            freq = float(sigma[h, a])
                            if freq > 0.001:
                                strat[labels[a]] = freq
                        if strat:
                            raw[h] = strat
                all_by_player[p_idx] = raw

            logger.info(
                "CpuEsSolver solve: %dp %s construct=%.1fs solve=%.1fs "
                "(%d iters, %s threads, off-lock)",
                n_players, street, construct_s, solve_s, iterations,
                num_threads or "all",
            )
            return all_by_player
        finally:
            gpu._freed = True  # native ss_free_tree hangs on this box; skip it

    def _first_decision_node_for(self, tree, player_idx: int) -> Optional[int]:
        """The first decision node owned by `player_idx` (BFS order), matching
        the GPU's get_all_avg_strategies which returns the first such node."""
        for ni in tree.decision_node_indices:
            nd = tree.nodes[ni]
            if nd.type == 0 and nd.player == player_idx and nd.num_children > 0:
                return ni
        return None
