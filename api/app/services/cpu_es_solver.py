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

        # ── 2p-TURN reach-CFR enablement (2026-06-17, OPT-IN, scoped) ──
        # The in-tree chance node (turn->river extension) that the reach-CFR turn
        # solve needs is built ONLY when POKER_SEARCH_DEPTH=full is set at C
        # tree-build time (ss_tree_api.c ss_build_tree2). We set it for THIS build
        # only, ONLY for an n_players==2 turn, and ONLY when explicitly opted in
        # via NEXUS_ALLOW_CHANCE_SOLVE (restored in finally so it never leaks).
        #
        # DEFAULT OFF (2026-06-17, EC2 probe verdict): reach-CFR is a correct,
        # CONVERGING solver (es_exploitability_2p dropped 5028->21 pot-chips/hand
        # over 50->2000 iters on the real turn tree), BUT at the INTERACTIVE iter
        # budget (50-1000) it is far from converged: it gives a muddy ~0.65-check
        # strategy that does NOT discriminate by hand strength like the blueprint
        # (e.g. AsAh nut overpair: reach 0.5 check vs blueprint bet 1.0; Td9d
        # gutshot: reach 0.66 vs blueprint 0.004). An under-converged turn solve is
        # WORSE than the blueprint lookup. So a 2p turn defaults to the blueprint
        # LOOKUP; reach-CFR stays behind NEXUS_ALLOW_CHANCE_SOLVE for research /
        # a future higher-iter path. (Turn full convergence needs far more than the
        # interactive budget — the 5349-node tree's BR gap was still ~3.5% pot at
        # 2000 iters; matches the findings doc's "reach correct but iter-hungry".)
        _allow_chance_turn = os.environ.get(
            "NEXUS_ALLOW_CHANCE_SOLVE", "")[:1] in ("1", "t", "T", "y", "Y")
        _want_turn_reach = (n_players == 2 and street == "turn"
                            and _allow_chance_turn)
        _depth_prev = os.environ.get("POKER_SEARCH_DEPTH")
        _depth_set = False
        if _want_turn_reach and _depth_prev is None:
            os.environ["POKER_SEARCH_DEPTH"] = "full"
            _depth_set = True
        try:
            # ── Build the tree (CPU tree-builder; no GPU solve / no leaf_value_fn) ──
            gpu = self._StreetSolverGPU(
                board=board_strs, pot_bb=pot_bb, stack_bb=stack_bb,
                player_ranges=player_ranges, bet_sizes=bet_sizes,
                subsequent_bet_sizes=subsequent_bet_sizes,
                use_cont_strats=True, leaf_value_fn=None,
                add_explicit_allin=add_explicit_allin,
            )
        finally:
            # Restore immediately after the build (the env only matters during
            # ss_build_tree2); keep the window minimal.
            if _depth_set:
                if _depth_prev is None:
                    os.environ.pop("POKER_SEARCH_DEPTH", None)
                else:
                    os.environ["POKER_SEARCH_DEPTH"] = _depth_prev
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
            # node. The ONLY such tree the engine builds by default is the 2p TURN
            # subgame (we set POKER_SEARCH_DEPTH=full for it above; the turn
            # extends to river per the paper's "all other cases"). The 2p turn
            # reach-CFR solve is correct + converging (es_exploitability_2p drops
            # monotonically) BUT iter-hungry: at the interactive budget it is
            # UNDER-converged and gives a muddy non-polarized strategy worse than
            # the blueprint lookup (EC2 probe 2026-06-17). So it is OPT-IN only
            # (NEXUS_ALLOW_CHANCE_SOLVE=1 sets POKER_SEARCH_DEPTH=full above to
            # build the chance node); without the opt-in NO chance node is built
            # for the 2p turn and we route it to the blueprint LOOKUP (below).
            #
            # A chance node here therefore means EITHER the 2p-turn opt-in fired
            # (n_players==2 + reach available -> served by reach-CFR), OR an
            # operator forced POKER_SEARCH_DEPTH=full at a player count with NO
            # validated reach path -> gated to lookup. _chance_is_reach mirrors the
            # use_reach predicate below.
            _has_chance = int(getattr(gpu._tree, "num_chance_nodes", 0)) > 0
            _reach_default = os.environ.get("NEXUS_ES_REACH", "1")[:1] in (
                "1", "t", "T", "y", "Y")
            _chance_is_reach = (_has_chance and n_players == 2 and _reach_default)
            _allow_chance = os.environ.get(
                "NEXUS_ALLOW_CHANCE_SOLVE", "")[:1] in ("1", "t", "T", "y", "Y")
            if _has_chance and not _chance_is_reach and not _allow_chance:
                raise ReplayUnsupported(
                    "live re-solve tree contains an in-tree chance node at a "
                    "player count with no validated reach-CFR path (reach is "
                    "heads-up only); chance-node solving is gated off for it (set "
                    "NEXUS_ALLOW_CHANCE_SOLVE=1 to force the sampling+enum chance "
                    "path), so this decision is served from the blueprint lookup "
                    "instead"
                )

            # 2p TURN, NOT opted into reach-CFR (the default): no chance node was
            # built, so the tree is a cont-strat-LEAF turn tree whose sampled solve
            # has the SAME un-conditioned-range over-bet as the flop (and the flop
            # conditioning fix is flop-gated, not validated for the turn). Rather
            # than serve a known-over-betting turn solve, route to the blueprint
            # LOOKUP (the honest fallback) — matches the probe verdict that the
            # turn live-solve is not yet ship-quality.
            if (n_players == 2 and street == "turn" and not _has_chance
                    and not _allow_chance):
                raise ReplayUnsupported(
                    "2p turn live-solve is gated to the blueprint lookup (the "
                    "cont-strat-leaf turn solve has an un-conditioned-range "
                    "over-bet, and reach-CFR is under-converged at interactive "
                    "iters — EC2 probe 2026-06-17); set NEXUS_ALLOW_CHANCE_SOLVE=1 "
                    "to opt into the research reach-CFR turn solve"
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

                # DIAG (NEXUS_ES_DIAG=1): reset the rollout blueprint-miss counters
                # before the construct so we can read the miss-rate after. A high
                # blueprint miss-rate -> the turn/river CONTINUATION falls back to
                # UNIFORM random play, which over-rewards BETTING (betting into a
                # randomly-playing villain looks great) -> the QJs over-bet. This
                # is the suspected cause (memory solve_overbet_rootcause_turn_centroids).
                _diag = os.environ.get("NEXUS_ES_DIAG", "")[:1] in ("1","t","T","y","Y")
                if _diag and hasattr(rc, "reset_rollout_observability"):
                    rc.reset_rollout_observability()

                # ── 2p-FLOP OPPONENT-RANGE CONDITIONING (2026-06-17, gated) ──
                # The depth-limited cont-strat-leaf values the BET leaf against
                # the UN-conditioned full preflop range -> betting marginal hands
                # (QJs) is under-punished -> over-bet vs the blueprint's confident
                # check (root cause: docs/CPU_ES_CHANCE_NODE_FINDINGS). Condition
                # each leaf's villain sampling weights on the blueprint probability
                # that the seat took the flop actions on the leaf's line, so the
                # BET leaf is valued vs the CALLING subset. Gated to 2p flop ONLY:
                # the 3p+ flop and 5-6p external-sampling paths take the UNCHANGED
                # shared-range construct below (provably no regression).
                #
                # DEFAULT ON (2026-06-17, EC2-validated, user flip): QJs check
                # OFF=0.002 (over-bet) -> ON=0.97-0.99 (matches blueprint 0.957),
                # clean (nodes_conditioned=1498, zero miss/fallback), faster
                # (1.5s vs 2.7s). Explicit NEXUS_LEAF_RANGE_CONDITIONING=0 opts out.
                _cond_on = os.environ.get(
                    "NEXUS_LEAF_RANGE_CONDITIONING", "1")[:1] not in (
                        "0", "f", "F", "n", "N")
                _use_cond = (_cond_on and n_players == 2 and street == "flop")
                per_leaf_ranges = None
                if _use_cond:
                    from leaf_villain_conditioning import (  # type: ignore
                        build_conditioned_per_leaf_ranges)
                    per_leaf_ranges, _cond_stats = build_conditioned_per_leaf_ranges(
                        leaf_infos=leaf_infos, player_ranges=player_ranges,
                        flop_board=board_ints, player_seats=list(player_seats),
                        game_action_history=list(game_action_indices or []),
                        blueprint=self._blueprint, n_players=n_players,
                    )
                    if _diag:
                        print(f"CpuEsSolver DIAG: range-conditioning stats="
                              f"{_cond_stats}", file=sys.stderr, flush=True)

                # ── Construct the leaf table (parallel batch, off-lock) ──
                t0 = time.time()
                try:
                    if per_leaf_ranges is not None:
                        table = rc.compute_subgame_leaf_values_sampled_perleaf_c(
                            leaf_infos=leaf_infos, board=board_ints, street=street,
                            player_hands=player_ranges,
                            per_leaf_player_hands=per_leaf_ranges,
                            active=[True] * n_players,
                            acting_order=list(range(n_players)),
                            player_seats=list(player_seats),
                            game_action_history=list(game_action_indices or []),
                            blueprint=self._blueprint,
                            starting_pot_chips=int(round(pot_bb * scale)),
                            stack_chips=stack_chips, max_hands=max_h,
                            num_rollouts=num_rollouts, num_samples_k=cache_k,
                            master_seed=master_seed,
                        )
                    else:
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
                    # blueprint MISS-RATE during the construct rollout — the key
                    # over-bet suspect (miss -> uniform continuation -> bet looks
                    # too good).
                    if hasattr(rc, "get_rollout_observability"):
                        _miss, _zs = rc.get_rollout_observability()
                        _bm = rc.get_rollout_bucket_miss_count() if hasattr(
                            rc, "get_rollout_bucket_miss_count") else -1
                        _ps = rc.get_rollout_per_street_miss_counts() if hasattr(
                            rc, "get_rollout_per_street_miss_counts") else None
                        print(f"CpuEsSolver DIAG: rollout blueprint_miss={_miss} "
                              f"zerosum={_zs} bucket_miss={_bm} per_street={_ps}",
                              file=sys.stderr, flush=True)
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
                # NOTE (2026-06-17): reach mode only RUNS for the 2p turn when the
                # operator opted in (NEXUS_ALLOW_CHANCE_SOLVE -> POKER_SEARCH_DEPTH
                # =full -> chance node built). It is correct + converging but
                # iter-hungry; the default 2p turn never reaches here (gated to
                # lookup above). Research / future higher-iter path.
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
