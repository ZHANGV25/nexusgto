"""CPU external-sampling ES re-solver — production engine for postflop streets.

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

SCOPE: FLOP/TURN — cont-strat depth-limited leaf-table terminals, the regime
every quality gate was measured on — and RIVER (2026-06-11). A river tree
terminates in SHOWDOWN/FOLD terminals only (StreetSolverGPU disables cont-strat
leaves at is_river), and the C ES settles showdowns natively via
es_showdown_value — side-pot validated against an exact Python reference
(poker-solver validate_sidepot.py: literal units + 100k-case fuzz < 1e-9 +
bit-exact single-pot regression + the nh>1 vectorized gate). River solves skip
the leaf-table construct entirely (no depth-limit leaves -> num_rollouts has no
effect, no leaf-sampling noise); the bypass is fail-loud (see solve()).
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ReplayUnsupported(Exception):
    """The solved tree cannot faithfully represent this mid-street entry.

    Raised (instead of ValueError) for KNOWN abstraction-surface limits —
    the multiway raise cap (the tree caps street raises at
    SS_MULTIWAY_MAX_RAISES=2 while the agent/trainer vocab allows 3), an
    action-count divergence between the reached node and the agent's legal
    set (real-chips commitment prune vs canonical tree chips), or an
    unvalidated vocab shape (multi-size / explicit-allin) on the replay
    path. LiveResolveSource converts this to LiveResolveUnsupported so the
    hybrid source falls back to blueprint LOOKUP for the decision (the
    blueprint vocab covers these lines). Genuine drift/bugs stay ValueError
    and fail loud."""

# poker-solver locations (the provider side-loads python/ onto sys.path at init;
# we resolve the DLL + python dir from the same env the provider uses).
_POKER_SOLVER = os.environ.get(
    "POKER_SOLVER_PATH",
)
# Bare name (no extension); c_solver resolves .dll on Windows / .so on Linux.
_ES_DLL_NAME = "es_resolver_real"


def _a3_cont_ev_enabled() -> bool:
    """A3 EXPERIMENT flag — DEFAULT OFF. When unset/0 the cont_ev boundary
    consumer is never loaded or enabled and this engine behaves byte-identically
    to the shipped path (the now-green E2E contract). Enable ONLY on the A3
    measurement box via NEXUS_A3_CONT_EV=1 + NEXUS_A3_CEV_PATH=<...>.cev.
    See docs/A3_CONT_EV_CONSUMER_BRIEF.md."""
    v = os.environ.get("NEXUS_A3_CONT_EV", "")
    return v[:1] in ("1", "t", "T", "y", "Y")


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
        self._cev_holder = None  # A3: kept alive iff NEXUS_A3_CONT_EV=1
        self._last_cont_ev_stats = None  # A3: {hits, attempts} from last solve
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

        # ── A3 (default OFF): load the cont_ev boundary table THROUGH es_lib ──
        # The table pointer must be valid in the SAME DLL that runs the construct
        # rollouts (es_resolver_real), so we bind bp_cont_ev_load_standalone on
        # self._es_lib — NOT via rollout_c.load_cont_ev_table_standalone, which
        # uses _find_blueprint_lib() (the separate mccfr_blueprint.so → a pointer
        # the es_resolver_real consumer cannot read). es_resolver_real compiles
        # mccfr_cont_ev_table.c, so it exports the loader + shares the struct.
        self._cev_holder = None
        if _a3_cont_ev_enabled():
            self._load_cont_ev_table(rc, es_lib)

        self._ready = True

    def _load_cont_ev_table(self, rc, es_lib) -> None:
        """A3: load the standalone .cev artifact through es_resolver_real and
        hand its pointer to the isolated rollout consumer. Fail-loud — if the
        flag is on, a missing/unloadable artifact is an error, never a silent
        skip (the experiment must not run against a half-wired engine)."""
        import ctypes

        cev_path = os.environ.get("NEXUS_A3_CEV_PATH", "")
        if not cev_path or not os.path.exists(cev_path):
            raise RuntimeError(
                f"NEXUS_A3_CONT_EV=1 but NEXUS_A3_CEV_PATH is missing/not found: "
                f"{cev_path!r}"
            )
        # Reuse rollout_c's struct mirror so the layout matches the C side
        # exactly (drift here scrambles every slot — see rollout_c.py).
        table = rc._BPContEVTableStandalone()
        ctypes.memset(ctypes.byref(table), 0, ctypes.sizeof(table))
        es_lib.bp_cont_ev_load_standalone.argtypes = [ctypes.c_void_p,
                                                      ctypes.c_char_p]
        es_lib.bp_cont_ev_load_standalone.restype = ctypes.c_int
        t0 = time.time()
        rcode = es_lib.bp_cont_ev_load_standalone(
            ctypes.byref(table), cev_path.encode("utf-8"))
        if rcode != 0:
            raise RuntimeError(
                f"bp_cont_ev_load_standalone({cev_path}) failed rc={rcode} "
                f"(bad magic/version/checksum/truncation)")
        n_slots = int(table.num_entries)
        if n_slots == 0:
            raise RuntimeError(
                f"cont_ev artifact {cev_path} loaded 0 slots — empty-state save")
        # Keep the struct alive for the solver's lifetime (the pointer the
        # consumer reads lives inside it).
        self._cev_holder = table
        table_ptr = ctypes.addressof(table)
        rc.set_rollout_cont_ev_table(table_ptr)
        logger.info(
            "CpuEsSolver A3: cont_ev table loaded through %s in %.1fs "
            "(%d slots) — boundary consumer ARMED (default-off flag is ON)",
            _ES_DLL_NAME, time.time() - t0, n_slots)

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
        vec_mode: int = 0,
        readout: str = "avg",
        hero_replay: Optional[Dict[str, object]] = None,
    ) -> Dict[int, Dict[int, Dict[str, float]]]:
        """Run an N-player flop/turn/river CPU-ES solve. Returns all_by_player:
        {player_idx: {hand_idx: {action_label: freq}}} — the SAME structure the
        GPU provider's get_all_avg_strategies produces, so the provider's
        _build_response_strategies maps it identically.

        Builds the tree (CPU; GPU tree-builder, no GPU solve), constructs the
        leaf table (parallel batch = the GPU's exact construct floor, off-lock),
        runs the C ES (Linear CFR, multi-thread Hogwild), extracts each
        player's strategy with the weight-0 filter the GPU applies.

        RIVER: terminals are exact showdown/fold (no depth-limit leaves), so
        the leaf-table construct is skipped (fail-loud bypass below) and
        num_rollouts / cache_k / master_seed have NO effect on the solve —
        there is no leaf-sampling noise on river.

        vec_mode: 0 = scalar traverser (the validated 5p-display reference);
            1 = vectorized traverser. vec-Linear is Gate-1-validated for the
            FINAL-iterate readout ONLY (vec-Linear-AVERAGE is seed-unstable at
            3+ players — VECTOR_TRAVERSER_DESIGN.md). Enforced below.
        readout: "avg" = σ̄ (the frequency-display contract) or "final" =
            final iterate (the playing-agent contract; what the EC2 wall-clock
            gate measured: 6p ~0.15%/pot floor in ~22s solve at 96 cores).
        hero_replay: optional {"player_idx": int, "street_actions": [int,...]}.
            When given, that player's strategy is read at the node reached by
            REPLAYING this street's action indices from the street root —
            the mid-street / faced-bet entry the live game loop needs — not
            at the first-BFS (no-bet) decision node. Fail-loud on any
            replay mismatch (off-tree action / vocab drift).
        """
        if vec_mode and readout != "final":
            raise ValueError(
                "vec_mode=1 requires readout='final' — vec-Linear-AVERAGE is "
                "seed-unstable at 3+ players (Gate-1, VECTOR_TRAVERSER_DESIGN.md)"
            )
        if readout not in ("avg", "final"):
            raise ValueError(f"readout must be 'avg' or 'final', got {readout!r}")
        if not self._ready:
            raise RuntimeError("CpuEsSolver not initialized")
        rc = self._rollout_c
        n_players = len(player_ranges)
        scale = 100.0
        stack_chips = int(round(stack_bb * scale))
        if len(board_strs) not in (3, 4, 5):
            raise ValueError(
                f"board must be 3 (flop), 4 (turn) or 5 (river) cards, got "
                f"{len(board_strs)}"
            )
        street = {3: "flop", 4: "turn", 5: "river"}[len(board_strs)]

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

            # IN-TREE CHANCE NODE GATE (2026-06-14, narrowed 2026-06-15).
            # Background: the OP3 §4 depth rule once extended a HEADS-UP (2p) flop
            # tree THROUGH a chance node, enumerating the full turn+river runout
            # (873k nodes). The CPU-ES chance handling was first a degenerate stub,
            # then fixed with full-per-hand-REACH vector CFR (correct) — but the
            # 873k-node flop tree is 20-90 min on CPU (not interactive), and that
            # full extension is NOT what Pluribus does.
            #
            # PLURIBUS-FAITHFUL (2026-06-15, paper-verified, Science 2019 supp):
            # the FLOP subgame terminates at a continuation-strategy LEAF at the
            # turn chance boundary — NO in-tree chance node — for ALL player
            # counts. poker-solver ss_tree_builder now depth-limits the 2p flop by
            # DEFAULT (POKER_FLOP_DEPTH_LIMIT=1): the 2p flop is a ~21-node
            # cont-strat-leaf betting tree (sub-ms), IDENTICAL in shape to the
            # already-shipping 3p+ flop path → it solves LIVE here with NO chance
            # node and NO gate. (POKER_FLOP_DEPTH_LIMIT=0 restores the 873k
            # extension for research.)
            #
            # So this gate now ONLY fires for trees that STILL contain a chance
            # node — the 2p TURN subgame (turn extends to river per the paper's
            # "all other cases"). The turn reach-CFR solve is correct + interactive
            # (~5k nodes, few seconds) but stays gated behind NEXUS_ALLOW_CHANCE_SOLVE
            # until the ship decision is made; until then a 2p turn decision is
            # served from the blueprint LOOKUP (honest fallback). 2p flop/river and
            # 3p+ build no chance node and are unaffected.
            _allow_chance = os.environ.get(
                "NEXUS_ALLOW_CHANCE_SOLVE", "")[:1] in ("1", "t", "T", "y", "Y")
            if (not _allow_chance
                    and int(getattr(gpu._tree, "num_chance_nodes", 0)) > 0):
                raise ReplayUnsupported(
                    "live re-solve tree contains an in-tree chance node (the "
                    "heads-up TURN->river extension); chance-node solving is gated "
                    "off (set NEXUS_ALLOW_CHANCE_SOLVE=1 to enable the reach-CFR "
                    "turn solve), so this decision is served from the blueprint "
                    "lookup instead"
                )

            n_leaf_nodes = int(gpu._tree.num_leaf_nodes)
            n_leaves = int(gpu._tree.num_leaves)
            # ZERO-LEAF trees terminate in SHOWDOWN/FOLD only and need NO leaf
            # table — the C ES settles those natively. Two cases:
            #   (a) RIVER: the builder disables cont-strat leaves at is_river.
            #   (b) A 2p FLOP/TURN tree EXTENDED through an in-tree CHANCE node to
            #       the next street (the OP3 depth rule): the extension deals the
            #       remaining board and terminates at showdown/fold, so it ALSO
            #       has zero leaves (verified: flop 873k-node + turn 5349-node 2p
            #       chance trees both have 0 leaf nodes). Before reach mode this
            #       path raised "no leaf infos extracted" (the chance tree has no
            #       leaves to extract) — so a 2p chance solve crashed in construct
            #       BEFORE the solver even ran (EC2 2026-06-14). Route any zero-leaf
            #       tree through the empty-table path; the table oracle exists but
            #       is provably unreachable (no leaf node references it).
            tree_has_chance0 = int(getattr(gpu._tree, "num_chance_nodes", 0)) > 0
            if street == "river" or (n_leaf_nodes == 0 and n_leaves == 0
                                     and tree_has_chance0):
                # FAIL-LOUD on the river contract: a depth-limit leaf in a river
                # tree would read garbage from the empty table. (For the chance
                # branch we already gated on n_leaf_nodes==0, so it cannot fire.)
                if street == "river" and (n_leaf_nodes != 0 or n_leaves != 0):
                    raise RuntimeError(
                        f"river tree has {n_leaf_nodes} leaf nodes / "
                        f"{n_leaves} leaves — river terminals must all be "
                        "showdown/fold (builder contract violated; refusing "
                        "to solve with an empty leaf table)"
                    )
                table = np.zeros((0, n_players, max_h), dtype=np.float32)
                construct_s = 0.0
            else:
                leaf_infos = self._extract_leaf_info(
                    gpu._tree, num_players=n_players)
                if not leaf_infos:
                    raise RuntimeError("no leaf infos extracted from tree")

                # ── A3 (default OFF): arm the cont_ev boundary consumer for
                # THIS construct iff the flag is on, the table is loaded, and
                # this is a flop multiway spot (the consumer's own predicate).
                # When off, none of these globals are touched. ──
                a3_armed = (
                    self._cev_holder is not None
                    and street == "flop"
                    and n_players >= 3
                )
                if a3_armed:
                    rc.reset_cont_ev_boundary_hit_count()
                    rc.set_use_cont_ev_at_flop_boundary(True)

                # ── Construct the leaf table (parallel batch, off-lock) ──
                t0 = time.time()
                try:
                    table = rc.compute_subgame_leaf_values_sampled_c(
                        leaf_infos=leaf_infos, board=board_ints, street=street,
                        player_hands=player_ranges, active=[True] * n_players,
                        acting_order=list(range(n_players)),
                        player_seats=list(player_seats),
                        game_action_history=list(game_action_indices or []),
                        blueprint=self._blueprint,
                        starting_pot_chips=int(round(pot_bb * scale)),
                        stack_chips=stack_chips, max_hands=max_h,
                        num_rollouts=num_rollouts, num_samples_k=cache_k,
                        master_seed=master_seed,
                    )
                finally:
                    if a3_armed:
                        # Disable the global flag so it can't leak into other
                        # solves, then snapshot the counters for this solve.
                        rc.set_use_cont_ev_at_flop_boundary(False)
                        hits = rc.get_cont_ev_boundary_hit_count()
                        attempts = rc.get_cont_ev_boundary_attempt_count()
                        self._last_cont_ev_stats = {
                            "hits": int(hits),
                            "attempts": int(attempts),
                            "hit_rate": (hits / attempts) if attempts else 0.0,
                        }
                        logger.info(
                            "CpuEsSolver A3: cont_ev boundary hits=%d attempts=%d "
                            "hit_rate=%.4f", hits, attempts,
                            self._last_cont_ev_stats["hit_rate"])
                construct_s = time.time() - t0

            # DIAG (NEXUS_ES_DIAG=1): dump leaf-table stats — isolate the 2p-flop
            # uniform-0.5 mystery (is the construct producing action-blind/zero
            # leaf values?). Cheap; off by default.
            if os.environ.get("NEXUS_ES_DIAG", "")[:1] in ("1", "t", "T", "y", "Y"):
                try:
                    import numpy as _np
                    _t = _np.asarray(table)
                    print(
                        f"CpuEsSolver DIAG: {n_players}p {street} table shape={_t.shape} "
                        f"min={float(_t.min()) if _t.size else 0.0:.4f} "
                        f"max={float(_t.max()) if _t.size else 0.0:.4f} "
                        f"mean={float(_t.mean()) if _t.size else 0.0:.4f} "
                        f"nonzero={int(_np.count_nonzero(_t)) if _t.size else 0}/{_t.size} "
                        f"iters={iterations}", file=sys.stderr, flush=True)
                except Exception as _e:
                    print(f"CpuEsSolver DIAG: table dump failed: {_e}", file=sys.stderr, flush=True)

            # ── Solve: C ES, Linear CFR, multi-thread Hogwild ──
            cs = self._CSolver.from_table(gpu._tree, table, n_players, max_h, seed=1)
            cs.set_cfr_mode(1)  # Linear CFR (matches the GPU, ~3x more iter-efficient)

            # In-tree CHANCE handling. A 2p flop/turn tree extends through a chance
            # node (OP3 turn-extension). Single-card-per-iter sampling is high-
            # variance + cross-line-incoherent -> degenerate/oscillating solve
            # (EC2 2026-06-14). For a CORRECT chance solve: ENUMERATE the card
            # outcomes (zero chance-variance, the GPU's full-reach equivalent),
            # ENUMERATE opponent actions (full CFR — removes opp-action sampling
            # variance; only tractable at small player counts), and use the VEC
            # traverser (updates all traverser hands per walk -> no traverser-hand
            # sampling variance). Residual variance = opp-hand sampling only.
            # 3p+ has no chance node so leave its (validated) sampling untouched —
            # opponent enumeration would blow up at high N anyway.
            tree_has_chance = int(getattr(gpu._tree, "num_chance_nodes", 0)) > 0
            # REACH MODE (2026-06-14): for a 2p in-tree-chance tree, prefer the
            # full-per-hand-REACH vector CFR (the GPU's own math, ported to CPU) —
            # it converges to the CORRECT mixed strategy deterministically and is
            # NOT degenerate, unlike the sampling+enum path (which still floors at
            # a pure over-bet on the chance-tree fixed point: turn toy spot 36% of
            # pot exploitable vs reach's 0.84% — validate_reach_cfr.py). Gated to
            # 2p (np==2): full reach is O(nodes*nh) per iter with an O(nh*nh_opp)
            # showdown term, tractable only at small player counts; 5-6p MUST keep
            # sampling and 3p+ trees have no chance node anyway. Off by default
            # (NEXUS_ES_REACH unset/0 falls back to the prior sampling+enum path so
            # this is a no-op until explicitly enabled for the validated 2p case).
            _reach = os.environ.get("NEXUS_ES_REACH", "1")[:1] in (
                "1", "t", "T", "y", "Y")
            use_reach = (
                tree_has_chance and n_players == 2 and _reach
                and hasattr(cs, "set_reach_mode")
            )
            if use_reach:
                cs.set_reach_mode(1)    # full reach; ignores cfr/chance/opp/vec
                readout = "avg"         # σ̄ readout (the frequency-display contract)
                vec_mode = 0
            elif tree_has_chance and hasattr(cs, "set_chance_mode"):
                cs.set_chance_mode(1)   # enumerate card outcomes (correct)
                if n_players <= 3 and hasattr(cs, "set_opp_mode"):
                    cs.set_opp_mode(1)  # full CFR over opponent actions (2-3p)
                    cs.set_vec_mode(1)  # all traverser hands per walk
                    vec_mode = 1        # force vec for the readout below
            if vec_mode:
                cs.set_vec_mode(1)
            # SMALL-TREE -> SINGLE-THREAD (2026-06-16). Multi-thread Hogwild on a
            # tiny tree is dominated by race noise: the Pluribus-faithful 2p flop
            # is only ~21 nodes / 8 decision nodes, and at num_threads=0 (all
            # cores) racing on those few shared regret arrays produced a DEGENERATE
            # uniform 0.500 mix, while single-thread gave a sensible check ~0.92
            # (EC2 2026-06-16). Hogwild buys nothing on a tree this small (the
            # per-iter work is trivial), so force single-thread below a node
            # threshold — deterministic AND correct. Large trees (873k flop ext,
            # 5-6p) keep multi-thread. Reach mode already runs single-thread.
            _nn = int(getattr(gpu._tree, "num_nodes", 0))
            _eff_threads = 1 if (0 < _nn <= 2000 and not use_reach) else num_threads
            t1 = time.time()
            cs.run(iterations, num_threads=_eff_threads)
            solve_s = time.time() - t1
            read_strategy = cs.final_strategy if readout == "final" else cs.avg_strategy

            # DIAG (NEXUS_ES_DIAG=1): dump the RAW first-decision-node strategy for
            # each player (unfiltered by the freq>0.001 display) + per-hand spread,
            # to see whether the solver produced action-discriminating mixes or a
            # uniform 0.5 (the 2p-flop mystery). Off by default.
            if os.environ.get("NEXUS_ES_DIAG", "")[:1] in ("1", "t", "T", "y", "Y"):
                try:
                    import numpy as _np
                    for _p in range(n_players):
                        _ni = None
                        for _cand in gpu._tree.decision_node_indices:
                            _nd = gpu._tree.nodes[_cand]
                            if _nd.type == 0 and _nd.player == _p and _nd.num_children > 0:
                                _ni = _cand
                                break
                        if _ni is None:
                            continue
                        _sig = read_strategy(_ni)
                        if _sig is None:
                            logger.info("CpuEsSolver DIAG: p%d node %d sig=None", _p, _ni)
                            continue
                        _sig = _np.asarray(_sig)
                        print(
                            f"CpuEsSolver DIAG: p{_p} node={_ni} na={_sig.shape[1]} "
                            f"nh={_sig.shape[0]} colmean={_np.round(_sig.mean(axis=0), 4).tolist()} "
                            f"action0_spread={float(_sig[:, 0].min()):.3f}..{float(_sig[:, 0].max()):.3f}",
                            file=sys.stderr, flush=True)
                except Exception as _e:
                    print(f"CpuEsSolver DIAG: strategy dump failed: {_e}", file=sys.stderr, flush=True)

            # The hero's replayed node (mid-street entry), if requested.
            hero_idx: Optional[int] = None
            hero_node: Optional[int] = None
            if hero_replay is not None and hero_replay.get("street_actions"):
                # The replay path is validated for the single-size,
                # no-explicit-allin vocab only (the 8.05B shape). With
                # multi-size / explicit-allin vocabs the agent's surviving-
                # raise count can silently diverge from the tree's child
                # count (different all-in baselines) and the structural
                # check cannot distinguish raise children — route to the
                # lookup fallback rather than risk a wrong-node read
                # (review 2026-06-10).
                if (add_explicit_allin or len(bet_sizes) > 1
                        or len(subsequent_bet_sizes) > 1):
                    raise ReplayUnsupported(
                        "street replay is validated only for single-size, "
                        "no-explicit-allin vocabs (got bet_sizes="
                        f"{list(bet_sizes)}, subsequent="
                        f"{list(subsequent_bet_sizes)}, allin="
                        f"{add_explicit_allin})"
                    )
                hero_idx = int(hero_replay["player_idx"])  # type: ignore[arg-type]
                hero_node = self._replay_street_node(
                    tree,
                    list(hero_replay["street_actions"]),  # type: ignore[arg-type]
                    n_players=n_players,
                )
                reached = tree.nodes[hero_node]
                if reached.type != 0 or reached.player != hero_idx:
                    raise ValueError(
                        f"replayed street actions lead to node {hero_node} "
                        f"(type={reached.type}, player={reached.player}) — not a "
                        f"decision node owned by hero player {hero_idx}. The "
                        "recorded street history does not match the solved tree."
                    )
                expected = hero_replay.get("expected_actions")
                if expected is not None and reached.num_children != int(expected):  # type: ignore[arg-type]
                    # The tree's action set at the hero's node differs from
                    # the agent's legal set: either the multiway raise cap
                    # (tree capped at 2 raises, vocab allows a 3rd — the mix
                    # would silently lack it) or a real-chips commitment
                    # prune the canonical tree chips don't have. Both are
                    # KNOWN abstraction limits -> lookup fallback.
                    raise ReplayUnsupported(
                        f"hero node has {reached.num_children} tree actions "
                        f"but the agent's legal set has {expected} — the "
                        "solved tree cannot represent this decision"
                    )

            # ── Extract strategies → all_by_player, with the weight-0 /
            #    never-visited filter the GPU's get_all_avg_strategies applies
            #    (freq>0.001; skip uniform-default rows so out-of-range hands
            #    stay fold-100% placeholders in _build_response_strategies). ──
            all_by_player: Dict[int, Dict[int, Dict[str, float]]] = {}
            for p_idx in range(n_players):
                if hero_idx is not None and p_idx == hero_idx:
                    node_idx = hero_node
                else:
                    node_idx = self._first_decision_node_for(tree, p_idx)
                raw: Dict[int, Dict[str, float]] = {}
                if node_idx is not None:
                    if hero_idx is not None and p_idx == hero_idx:
                        # Replayed (possibly faced-bet) node: synthesize
                        # labels from the tree's own chip deltas —
                        # _build_labels is wrong off the street root.
                        labels = self._node_action_labels(tree, node_idx, n_players)
                    else:
                        labels = self._build_labels(gpu._tree, node_idx)
                    sigma = read_strategy(node_idx)  # [nh, na] or None
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

    def _replay_street_node(
        self,
        tree,
        street_actions: List[int],
        *,
        n_players: int,
    ) -> int:
        """Walk the solved tree from the street root through this street's
        recorded agent action indices; return the reached node index.

        The tree builder (poker-solver src/ss_tree_builder.c:272-370) emits
        children in the TRAINER's action order — [fold iff facing a bet]
        [check/call][raises in size order][explicit all-in last] — which is
        the SAME convention the agent's action indices use
        (legal_categories_chips / _solver_label_to_action_index). So the
        recorded agent index IS the child index. Each step is STRUCTURALLY
        verified against the child's chips/active state (fold → actor
        inactive; check/call → actor matched the max bet; raise → actor
        exceeds it) so a convention drift fails loud instead of silently
        walking the wrong line. (Do NOT map via _build_labels_at_node here:
        its labels are chip-amount-only and wrong at faced-bet nodes —
        found empirically 2026-06-10.)

        Fail-loud (ValueError) on: a non-decision node mid-replay, an
        out-of-range action index (off-tree action — bet-size translation
        is out of scope), or a structural-verification mismatch.
        """
        ni = 0  # street root
        for step, want_idx in enumerate(street_actions):
            nd = tree.nodes[ni]
            if nd.type != 0 or nd.num_children <= 0:
                raise ValueError(
                    f"street replay step {step}: node {ni} is not a decision "
                    f"node (type={nd.type}) — recorded history does not match "
                    "the solved tree"
                )
            want = int(want_idx)
            actor = nd.player
            mx = max(nd.bets[q] for q in range(n_players))
            to_call = mx - nd.bets[actor]
            n_nonraise = 2 if to_call > 0 else 1
            if not (0 <= want < nd.num_children):
                if nd.num_children == n_nonraise and want >= n_nonraise:
                    # The recorded action is a RAISE but this tree node has
                    # no raise children: the tree's raise cap
                    # (SS_MULTIWAY_MAX_RAISES=2 multiway vs the agent/trainer
                    # cap of 3) — a known abstraction limit, NOT drift.
                    # Surface as ReplayUnsupported so the hybrid falls back
                    # to LOOKUP, whose vocab covers the 3rd raise.
                    raise ReplayUnsupported(
                        f"street replay step {step}: recorded raise (index "
                        f"{want}) at a raise-capped tree node {ni} "
                        f"({nd.num_children} children) — tree raise cap is "
                        "below the agent vocab's"
                    )
                raise ValueError(
                    f"street replay step {step}: action index {want} out of "
                    f"range [0,{nd.num_children}) at node {ni} — vocab drift "
                    "or off-tree action"
                )
            child = tree.nodes[nd.children[want]]
            if to_call > 0 and want == 0:
                # fold: actor leaves the hand, chips unchanged.
                if child.active[actor] or child.bets[actor] != nd.bets[actor]:
                    raise ValueError(
                        f"street replay step {step}: child {want} of node {ni} "
                        "does not look like a FOLD — tree child-order "
                        "convention drift (fail loud)"
                    )
            elif want == n_nonraise - 1:
                # check/call: actor matches the outstanding max bet.
                if not child.active[actor] or child.bets[actor] != mx:
                    raise ValueError(
                        f"street replay step {step}: child {want} of node {ni} "
                        "does not look like a CHECK/CALL — tree child-order "
                        "convention drift (fail loud)"
                    )
            else:
                # bet/raise: actor exceeds the prior max bet.
                if not child.active[actor] or child.bets[actor] <= mx:
                    raise ValueError(
                        f"street replay step {step}: child {want} of node {ni} "
                        "does not look like a BET/RAISE — tree child-order "
                        "convention drift (fail loud)"
                    )
            ni = nd.children[want]
        return ni

    def _node_action_labels(self, tree, node_idx: int, n_players: int) -> List[str]:
        """Synthesize correct action labels for a decision node from the
        tree's own chip/active deltas — the exact inverse of the tree
        BUILDER's STORAGE convention.

        The production _build_labels_at_node is chip-amount-only and labels
        faced-bet nodes wrongly (a fold reads "Check", a call reads as a
        pot-fraction "Raise") — fine for the display path, which only reads
        street roots, but wrong for a replayed mid-street node.

        Builder convention (review 2026-06-10 — NOT the trainer's): at a
        faced-bet node ss_tree_builder.c:326 computes bet_amount = to_call +
        size*(pot+to_call) and :340 STORES the child level as
        mx + bet_amount (on top of the max bet, not the actor's own bet), so
        the child's delta vs the actor's bet = 2*to_call + size*(pot+to_call)
        and size = (delta - 2*to_call)/(pot + to_call). The all-in clamp is
        applied to bet_amount (= delta - to_call) against the CONSTANT
        street-entry effective stack (:328).
        """
        nd = tree.nodes[node_idx]
        actor = nd.player
        mx = max(nd.bets[q] for q in range(n_players))
        to_call = mx - nd.bets[actor]
        pot = float(nd.pot)
        stack = int(tree.effective_stack)
        labels: List[str] = []
        for c in range(nd.num_children):
            ch = tree.nodes[nd.children[c]]
            if to_call > 0 and c == 0:
                labels.append("Fold")
                continue
            if c == (1 if to_call > 0 else 0):
                labels.append("Call" if to_call > 0 else "Check")
                continue
            delta = int(ch.bets[actor]) - int(nd.bets[actor])
            bet_amount = delta - to_call  # the builder's clamped quantity
            if bet_amount >= stack:
                labels.append("All-in")
                continue
            if to_call > 0:
                frac = (delta - 2 * to_call) / max(pot + to_call, 1.0)
                labels.append(f"Raise {int(round(frac * 100))}%")
            else:
                frac = delta / max(pot, 1.0)
                labels.append(f"Bet {int(round(frac * 100))}%")
        return labels
