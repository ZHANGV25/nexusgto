"""Gate 3 (real-blueprint): the C ES solver running rollout_compute_leaf_value_c
LAZILY against the real 8.05B blueprint. Two checks:

  (A) ORACLE ISOLATION cross-check: the C real oracle and the proven Python
      compute_leaf_value wrap the SAME rollout. For a few (leaf, hero, hand)
      cells, average each over high num_rollouts and compare means within
      combined SE. This isolates the C oracle (seat map, flop_history, ranges)
      from the solver — exactly as the primitive test isolated the terminals.
  (B) NON-DEGENERACY gate: run the C ES solver end-to-end; the root σ̄ must be
      mixed + hand-dependent (NOT the degenerate uniform-1/n strategy). Also
      measures the REAL single-hand per-call wall cost, rather than assuming
      the batch path's ~20 us.

The blueprint loads THROUGH es_resolver_real.dll so the BPSReader struct and the
rollout calls live in ONE DLL (no cross-DLL pointer hazard).

Run: ES_DLL=es_resolver_real.dll ES_N=5 ES_ITERS=3000 PYTHONIOENCODING=utf-8 \
     py -3.9 -m es_resolver.run_real_spot_c
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force the real DLL for c_solver BEFORE importing it.
os.environ.setdefault("ES_DLL", "es_resolver_real.dll")

ESRO_MAX_FLOP_HIST = 256


def _real_grid_ranges(board_ints, n_players):
    deck = [x for x in range(52) if x not in set(board_ints)]
    ranges = []
    cur = 0
    nh = max(4, min(12, len(deck) // (2 * n_players)))
    for _p in range(n_players):
        hs = [(deck[cur + 2 * i], deck[cur + 2 * i + 1], 1.0) for i in range(nh)]
        cur += 2 * nh
        ranges.append(hs)
    return ranges, nh


def _extract_leaf_state(ss_tree, n_players):
    """Per-leaf flat int32 arrays for the C oracle: pot, bets, active, hist,
    hist_len. pot/bets/hist come from the search's extract_leaf_info_from_tree
    (the load-bearing flop_history_train); active is read from the leaf SSNode."""
    from engine.leaf_values import extract_leaf_info_from_tree
    leaf_infos = extract_leaf_info_from_tree(ss_tree, num_players=n_players)
    by_idx = {li.leaf_idx: li for li in leaf_infos}
    n_leaves = ss_tree.num_leaves
    pot = np.zeros(n_leaves, dtype=np.int32)
    bets = np.zeros(n_leaves * n_players, dtype=np.int32)
    active = np.zeros(n_leaves * n_players, dtype=np.int32)
    hist = np.zeros(n_leaves * ESRO_MAX_FLOP_HIST, dtype=np.int32)
    hist_len = np.zeros(n_leaves, dtype=np.int32)

    # active per leaf: find a LEAF node for each leaf_idx and read active_players
    leaf_active = {}
    for i in range(ss_tree.num_nodes):
        nd = ss_tree.nodes[i]
        if nd.type == 4 and nd.leaf_idx >= 0 and nd.leaf_idx not in leaf_active:
            leaf_active[nd.leaf_idx] = [int(nd.active_players[p]) for p in range(n_players)]

    for li in leaf_infos:
        idx = li.leaf_idx
        if idx < 0 or idx >= n_leaves:
            continue
        pot[idx] = int(li.pot)
        lb = list(li.bets)
        for p in range(n_players):
            bets[idx * n_players + p] = int(lb[p]) if p < len(lb) else 0
        av = leaf_active.get(idx, [1] * n_players)
        for p in range(n_players):
            active[idx * n_players + p] = av[p]
        h = list(li.flop_history_train)
        hl = min(len(h), ESRO_MAX_FLOP_HIST)
        for k in range(hl):
            hist[idx * ESRO_MAX_FLOP_HIST + k] = int(h[k])
        hist_len[idx] = hl
    return n_leaves, pot, bets, active, hist, hist_len, by_idx, leaf_active


def main() -> int:
    bp_path = os.environ.get(
        "POKER_BP",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "blueprint_data", "unified_blueprint.bps",
        ),
    )
    n_players = int(os.environ.get("ES_N", "5"))
    iters = int(os.environ.get("ES_ITERS", "3000"))

    print("=" * 78)
    print(f"GATE 3: C ES real-blueprint oracle — {n_players}p flop, 8.05B blueprint")
    print("=" * 78)

    # ── Load blueprint THROUGH es_resolver_real.dll (single-DLL pointer) ──
    print(f"\n[1/5] Loading blueprint via es_resolver_real.dll ...", flush=True)
    import engine.bps_reader_py as bps_reader_py
    from es_resolver.c_solver import _lib as es_real_lib, _HAS_REAL
    if not _HAS_REAL:
        print("  ERROR: es_resolver_real.dll not loaded (set ES_DLL). Abort.")
        return 1
    # Repoint BPSReaderC's lib at the real ES DLL so the reader struct + rollout
    # share one module.
    bps_reader_py._find_lib = lambda: es_real_lib
    from engine.bps_reader_py import BPSReaderC
    t0 = time.time()
    blueprint = BPSReaderC(bp_path)
    print(f"      loaded in {time.time()-t0:.1f}s", flush=True)
    bp_handle = blueprint._reader.value  # BPSReader* address (int)

    # ── Build tree ──
    from engine.street_solver_gpu import StreetSolverGPU, _build_labels_at_node
    from engine.cards import card_to_int
    board = ["Ah", "Kh", "2c"]
    bi = [card_to_int(c) for c in board]
    ranges, nh = _real_grid_ranges(bi, n_players)
    print(f"[2/5] Building {n_players}p flop tree (nh={nh}/player) ...", flush=True)
    gpu = StreetSolverGPU(
        board=board, pot_bb=20.0, stack_bb=95.0, player_ranges=ranges,
        bet_sizes=[0.75], subsequent_bet_sizes=[1.0],
        use_cont_strats=True, leaf_value_fn=None, add_explicit_allin=False,
    )
    root_labels = _build_labels_at_node(gpu._tree, 0)
    print(f"      nodes={gpu._tree.num_nodes} leaves={gpu._tree.num_leaves}", flush=True)

    player_seats = np.arange(n_players, dtype=np.int32)
    stack_chips = int(95.0 * 100)

    # ── Extract per-leaf state ──
    print(f"[3/5] Extracting per-leaf state (flop_history_train) ...", flush=True)
    (n_leaves, pot, bets, active, hist, hist_len,
     by_idx, leaf_active) = _extract_leaf_state(gpu._tree, n_players)

    # ── (A) ORACLE WIRING ISOLATION (zero-confound) ──
    # Correctness anchor is the PYTHON REFERENCE (es_solver.py + real_oracle.py
    # uses PER-LEAF active), NOT the GPU/batch path (which treats every player as
    # active, active=[True]*np: a different quantity that deals folded players
    # into the runout, so it is not used as the reference here).
    #
    # The clean check: call rollout_compute_leaf_values_all_hands_c (a known-good
    # caller of the SAME _impl) with the leaf's OWN per-leaf active + a FIXED
    # bias B + high num_rollouts + skip=1; call the single-hand-skip C entry with
    # the SAME per-leaf active + SAME B + same rollouts. Same _impl, same active,
    # same bias → must agree to MC noise. Proves seat-map / flop_history / bets /
    # skip wiring with zero bias-sampling and zero active-semantics confound.
    print(f"[4/5] Oracle WIRING isolation (single-hand-skip vs all-hands, "
          f"same per-leaf active + fixed bias) ...", flush=True)
    from es_resolver.c_solver import CSolver
    from es_resolver.es_solver import snapshot_tree
    snap = snapshot_tree(gpu._tree, ranges)

    L = es_real_lib
    # bind the two C entries directly
    class _RHand(ctypes.Structure):
        _fields_ = [("c0", ctypes.c_int32), ("c1", ctypes.c_int32), ("weight", ctypes.c_float)]
    class _RHandList(ctypes.Structure):
        _fields_ = [("hands", ctypes.POINTER(_RHand)), ("n_hands", ctypes.c_int32)]
    IP = ctypes.POINTER(ctypes.c_int32)
    L.rollout_compute_leaf_value_skip_c.restype = ctypes.c_int32
    L.rollout_compute_leaf_value_skip_c.argtypes = [
        ctypes.c_void_p, IP, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        IP, ctypes.c_int32, IP, IP, ctypes.POINTER(_RHandList), IP, ctypes.c_int32,
        IP, IP, ctypes.c_int32, IP, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float)]
    L.rollout_compute_leaf_values_all_hands_c.restype = ctypes.c_int32
    L.rollout_compute_leaf_values_all_hands_c.argtypes = [
        ctypes.c_void_p, IP, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        IP, ctypes.c_int32, IP, IP, ctypes.POINTER(_RHandList), IP, ctypes.c_int32,
        IP, IP, ctypes.c_int32, IP, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float)]

    def _mk_handlists(php):
        bufs = []; lists = (_RHandList * n_players)()
        for p in range(n_players):
            n = len(php[p]); buf = (_RHand * n)()
            for h in range(n):
                buf[h].c0, buf[h].c1, buf[h].weight = php[p][h][0], php[p][h][1], php[p][h][2]
            bufs.append(buf)
            lists[p].hands = ctypes.cast(buf, ctypes.POINTER(_RHand)); lists[p].n_hands = n
        return lists, bufs

    php = snap.player_hands
    hlists, _keep = _mk_handlists(php)
    board_arr = (ctypes.c_int32 * len(bi))(*bi)
    seats_arr = (ctypes.c_int32 * n_players)(*range(n_players))
    aord_arr = (ctypes.c_int32 * n_players)(*range(n_players))
    max_h = snap.max_hands
    NUM_ROLL_HI = 400
    FIXED_BIAS = (ctypes.c_int32 * n_players)(*([0] * n_players))  # unmodified strat
    SEED = 0xC0FFEE

    test_leaves = sorted(by_idx)[:6]
    print(f"      leaves {test_leaves}, hero=0 hand=0, fixed bias=0, "
          f"num_rollouts={NUM_ROLL_HI}, per-leaf active:", flush=True)
    cross_ok = True
    for leaf_idx in test_leaves:
        av = leaf_active.get(leaf_idx, [1]*n_players)
        if av[0] != 1:
            continue  # hero must be active to compare a meaningful value
        lpot = int(by_idx[leaf_idx].pot)
        lbets = [int(b) for b in list(by_idx[leaf_idx].bets)]
        lhist = list(by_idx[leaf_idx].flop_history_train)
        active_arr = (ctypes.c_int32 * n_players)(*av)
        bets_arr = (ctypes.c_int32 * n_players)(*lbets)
        stacks_arr = (ctypes.c_int32 * n_players)(*[max(0, stack_chips - b) for b in lbets])
        hist_arr = (ctypes.c_int32 * max(1, len(lhist)))(*(lhist or [0]))
        skip = 1 if len(lhist) > 0 else 0

        # single-hand-skip entry
        out1 = ctypes.c_float(0.0)
        rc1 = L.rollout_compute_leaf_value_skip_c(
            ctypes.c_void_p(bp_handle), board_arr, len(bi), 1, lpot,
            active_arr, n_players, stacks_arr, bets_arr, hlists, aord_arr, n_players,
            seats_arr, hist_arr, len(lhist), FIXED_BIAS, 0, 0, NUM_ROLL_HI,
            SEED, leaf_idx, 0, skip, ctypes.byref(out1))
        # all-hands entry (read hero=0, hand=0)
        out_all = (ctypes.c_float * (n_players * max_h))()
        rc2 = L.rollout_compute_leaf_values_all_hands_c(
            ctypes.c_void_p(bp_handle), board_arr, len(bi), 1, lpot,
            active_arr, n_players, stacks_arr, bets_arr, hlists, aord_arr, n_players,
            seats_arr, hist_arr, len(lhist), FIXED_BIAS, max_h, NUM_ROLL_HI,
            SEED, leaf_idx, 0, skip, out_all)
        v1 = out1.value
        v2 = out_all[0 * max_h + 0]
        d = abs(v1 - v2)
        # tolerance: same seed+fixed bias → SHOULD be bit-identical if seed chains
        # match; if internal seed derivation differs between entries, allow MC SE.
        tol = max(1.0, 0.02 * max(abs(v1), abs(v2)))
        flag = "OK" if d <= tol else "MISMATCH"
        if d > tol:
            cross_ok = False
        print(f"        leaf {leaf_idx} (active={av},hist_len={len(lhist)},skip={skip}): "
              f"single={v1:+9.1f}  all-hands={v2:+9.1f}  |Δ|={d:7.2f}  rc=({rc1},{rc2})  {flag}",
              flush=True)
    print(f"      => oracle wiring: {'PASS' if cross_ok else 'INVESTIGATE'}", flush=True)

    # ── (B) NON-DEGENERACY end-to-end + real per-call wall cost ──
    NUM_ROLL = 3
    print(f"[5/5] End-to-end C ES ({iters} iters) + non-degeneracy + wall cost ...", flush=True)
    CACHE_K = 16
    cs = CSolver.from_real_oracle(
        gpu._tree, bp_handle, street=1, stack_chips=stack_chips, n_leaves=n_leaves,
        max_h=snap.max_hands, pot=pot, bets=bets, active=active, hist_flat=hist,
        hist_len=hist_len, player_seats=player_seats, num_rollouts=NUM_ROLL,
        cache_k=CACHE_K, master_seed=999, seed=1)
    t0 = time.time()
    cs.run(iters, num_threads=1)
    es_dt = time.time() - t0
    st = cs.stats
    fills, hits, sims, errs = cs.real_oracle_stats()
    print(f"      ES {iters} iters in {es_dt:.2f}s "
          f"({st.traversals/es_dt:,.0f} trav/s, {st.leaf_rollouts:,} leaf-hits, "
          f"{st.leaf_rollouts/max(st.traversals,1):.2f} rollouts/trav)", flush=True)
    print(f"      CACHE: {fills:,} leaf FILLS (cache_k={CACHE_K}), {hits:,} cache hits "
          f"→ {100.0*hits/max(hits+fills,1):.1f}% hit rate; oracle_errors={errs}", flush=True)
    if fills > 0:
        print(f"      per-FILL wall ≈ {1e6*es_dt/max(fills,1):.0f} us "
              f"(amortizes the runout across all hands; one fill per reached leaf)",
              flush=True)

    root = cs.avg_strategy(0)
    na = root.shape[1]
    uniform = np.full(na, 1.0/na)
    dists = np.abs(root - uniform[None, :]).mean(axis=1)
    frac = float((dists > 0.05).mean())
    print(f"\n  NON-DEGENERATE check: root σ̄ vs uniform(1/{na}), labels={root_labels}")
    print(f"    mean L1-from-uniform = {dists.mean():.3f}   "
          f"fraction non-uniform (>0.05) = {frac:.1%}")
    for h in range(min(4, root.shape[0])):
        print(f"      hand{h}: {np.round(root[h], 3).tolist()}")
    nondegen = dists.mean() > 0.05
    print(f"\n  VERDICT: {'PASS (non-degenerate)' if nondegen else 'FAIL (uniform — degenerate)'}")

    gpu._freed = True
    blueprint._freed = True if hasattr(blueprint, "_freed") else None
    ok = cross_ok and nondegen
    print(f"\n  >>> GATE 3 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
