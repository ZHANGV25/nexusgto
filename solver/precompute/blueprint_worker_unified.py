#!/usr/bin/env python3
"""Unified blueprint worker: 6-player preflop-through-river MCCFR.

Matches Pluribus exactly: ONE unified solve over the entire game tree.
All 6 players, all 1326 hands, 169 lossless preflop buckets, 200 k-means
postflop buckets. Preflop → flop → turn → river → showdown in a single MCCFR.

This is the Pluribus approach (Brown & Sandholm, Science 2019):
  - 64 cores, 8 days, 12,400 CPU core hours, <512GB RAM
  - ~665M action sequences, ~414M encountered

Usage (local test):
    python blueprint_worker_unified.py --iterations 10000 --num-threads 1

Usage (full scale; every artifact is written under --output-dir):
    python blueprint_worker_unified.py --iterations 0 --time-limit-hours 192 \
        --num-threads 64 --hash-size 536870912 \
        --output-dir ./blueprint_out

The solver runs until --time-limit-hours is reached (Pluribus ran for 8 days
= 192 hours). Iteration count is not pre-specified — training runs continuously
and convergence is measured by exploitability and strategy stability.
"""

import argparse
import ctypes
import json
import os
import signal
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

# ── Constants (Pluribus-matched) ──────────────────────────────────

NUM_PLAYERS = 6
SMALL_BLIND = 50     # $50
BIG_BLIND = 100      # $100
INITIAL_STACK = 10000 # $10,000 = 100BB

# Pluribus bet sizes: up to 14 for preflop, 3 for later rounds
# Flat preflop sizes (used as fallback if tiered not set)
PREFLOP_BET_SIZES = [0.7, 1.0]
# Pruned abstraction: single postflop bet size (all-in auto-added).
POSTFLOP_BET_SIZES = [0.75]

# Tiered preflop sizing, pruned: no 0.5x open size. Dropping it cuts
# end-of-flop boundary action_hashes by about 89% (1.8M to 205K), which
# lifts trainer visits per slot from ~5 (too few to learn from) to ~43
# static / ~100-130 effective with Pluribus regret-based dynamic pruning
# at 10B iters.
PREFLOP_TIERS = {
    0: [0.7, 1.0],                   # open raise: 2 sizes
    1: [0.7, 1.0],                   # 3-bet: 2 sizes
    2: [1.0],                        # 4-bet: 1 size (all-in auto-added)
}
PREFLOP_MAX_RAISES = 3

# ── Canonical 169-class preflop bucket map (matches mccfr_blueprint.c) ──
# Built programmatically so probes display the right hand under each
# bucket index. Probe code uses this map rather than hardcoded indices,
# which are easy to get wrong. Single source.
_RANKS = "AKQJT98765432"


def _build_c_bucket_map():
    """Replicate mccfr_blueprint.c init_unified() class_map iteration.
    The C code:
        for r0 = 12 down to 0:
            for r1 = r0 down to 0:
                if r0 == r1: pair → 1 class
                else:        suited then offsuit → 2 classes
    Yields AA=0, AKs=1, AKo=2, ..., 22=168.
    """
    bm = {}
    n = 0
    for r0 in range(12, -1, -1):
        for r1 in range(r0, -1, -1):
            r0c = _RANKS[12 - r0]
            r1c = _RANKS[12 - r1]
            if r0 == r1:
                bm[r0c + r1c] = n
                n += 1
            else:
                bm[r0c + r1c + "s"] = n
                n += 1
                bm[r0c + r1c + "o"] = n
                n += 1
    assert n == 169, f"Expected 169, got {n}"
    # Spot-check sentinels — fail loud at module load if any drift.
    assert bm["AA"] == 0
    assert bm["AKs"] == 1
    assert bm["AKo"] == 2
    assert bm["KK"] == 25
    assert bm["JJ"] == 69
    assert bm["TT"] == 88
    assert bm["22"] == 168
    return bm


_C_BUCKET_MAP = _build_c_bucket_map()

# ── Config hash (stale-checkpoint guard) ──────────────────────────
# A run that loads a checkpoint trained under a different abstraction
# (for example flat preflop sizes under tiered sizing) trains garbage
# without any error. Embedding a SHA256 of the config-relevant fields in
# the checkpoint meta and refusing to load on mismatch prevents that.

def _read_checkpoint_meta(output_dir):
    """Return checkpoint_meta.json from ``output_dir``, or exit with status 2.

    A resume needs this file. Its iteration count keeps the discount and
    pruning schedule in step, and its config_hash is checked against this
    run's settings. A missing or unreadable file stops the run; the
    checkpoint is left in place.
    """
    path = os.path.join(output_dir, "checkpoint_meta.json")
    try:
        with open(path) as f:
            meta = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[BP][FATAL] cannot read {path}: {exc}. Refusing to resume "
              f"without the checkpoint's iteration count and config hash. "
              f"The checkpoint is kept.", flush=True)
        sys.exit(2)
    iterations = meta.get("iterations") if isinstance(meta, dict) else None
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
        print(f"[BP][FATAL] {path} has no valid 'iterations' count. Refusing "
              f"to resume. The checkpoint is kept.", flush=True)
        sys.exit(2)
    return meta


def _compute_config_hash(add_explicit_allin=None):
    """Hash of everything that shapes the regret table.
    A mismatch between current config and checkpoint meta means the
    regret table's action_hash/bucket layout is incompatible and the
    checkpoint must be rejected.

    add_explicit_allin is part of the layout: gate=0 produces one fewer
    slot per deep-stack infoset. Pass None to omit (legacy
    behavior, equivalent to add_explicit_allin=1).
    """
    import hashlib
    # Note: river_ochs and river_ochs_clusters below are read from the
    # environment (BPS_RIVER_OCHS, BPS_RIVER_OCHS_CLUSTERS), not from the
    # --river-ochs / --river-ochs-clusters flags. A run that sets only the
    # flags trains with OCHS but hashes as river_ochs "0". To reproduce a
    # published config_hash, set the environment variables (the flags then
    # default to them). Left as is so existing checkpoints keep their hash.
    payload = {
        "num_players": NUM_PLAYERS,
        "blinds": [SMALL_BLIND, BIG_BLIND],
        "initial_stack": INITIAL_STACK,
        "preflop_tiers": {str(k): v for k, v in sorted(PREFLOP_TIERS.items())},
        "preflop_max_raises": PREFLOP_MAX_RAISES,
        "postflop_bet_sizes": POSTFLOP_BET_SIZES,
        "preflop_buckets": 169,
        "postflop_buckets": 200,
        # DCFR params affect only regret trajectory (not table layout), but
        # they DO affect convergence — resuming a Linear CFR checkpoint
        # under DCFR (or vice versa) is a correctness hazard. Include them.
        "dcfr_alpha": os.environ.get("BPS_DCFR_ALPHA", ""),
        "dcfr_beta": os.environ.get("BPS_DCFR_BETA", ""),
        "dcfr_gamma": os.environ.get("BPS_DCFR_GAMMA", ""),
        # River OCHS is a card-abstraction change: river_bucket differs
        # between OCHS and EHS-percentile, so info-set keys differ. A v3
        # OCHS checkpoint cannot be resumed under the legacy path (or
        # vice versa) without producing garbage.
        "river_ochs": os.environ.get("BPS_RIVER_OCHS", "0"),
        "river_ochs_clusters": os.environ.get("BPS_RIVER_OCHS_CLUSTERS", ""),
        # Hash table size is NOT included: bp_load_regrets rehashes each
        # entry via info_table_find_or_create into the current table_size,
        # so checkpoints are portable across hash table sizes.
    }
    if add_explicit_allin is not None:
        payload["add_explicit_allin"] = int(add_explicit_allin)
    canon = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canon.encode('utf-8')).hexdigest()

# ── DLL loading ───────────────────────────────────────────────────

class BPConfig(ctypes.Structure):
    _fields_ = [
        ("discount_stop_iter", ctypes.c_int64),
        ("discount_interval", ctypes.c_int64),
        ("prune_start_iter", ctypes.c_int64),
        ("snapshot_start_iter", ctypes.c_int64),
        ("snapshot_interval", ctypes.c_int64),
        ("strategy_interval", ctypes.c_int64),
        ("num_threads", ctypes.c_int),
        ("hash_table_size", ctypes.c_int64),
        ("snapshot_dir", ctypes.c_char_p),
        ("include_preflop", ctypes.c_int),
        ("postflop_num_buckets", ctypes.c_int),
        # Opt-in OCHS river abstraction (Johanson 2013 / Pluribus). 0 = legacy
        # EHS-percentile path; 1 = OCHS with river_ochs_num_clusters opponent
        # clusters (0 → default of 8).
        ("river_ochs_enabled", ctypes.c_int),
        ("river_ochs_num_clusters", ctypes.c_int),
    ]


def load_bp_dll(build_dir):
    """Load the MCCFR blueprint DLL."""
    for ext in ['so', 'dll']:
        path = os.path.join(build_dir, f'mccfr_blueprint.{ext}')
        if os.path.exists(path):
            bp = ctypes.CDLL(path)
            bp.bp_default_config.restype = None
            bp.bp_init_unified.restype = ctypes.c_int
            bp.bp_set_buckets.restype = ctypes.c_int
            bp.bp_solve.restype = ctypes.c_int
            bp.bp_solve.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            bp.bp_get_strategy.restype = ctypes.c_int
            bp.bp_num_info_sets.restype = ctypes.c_int64
            bp.bp_num_info_sets.argtypes = [ctypes.c_void_p]
            # Hash-table health stats. All four out-pointers may be NULL.
            # Without these in checkpoint_meta.json a saturated table drops
            # info sets silently, and the loss shows only at the next
            # checkpoint's info-set count.
            if hasattr(bp, "bp_get_table_stats"):
                bp.bp_get_table_stats.restype = None
                bp.bp_get_table_stats.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                ]
            bp.bp_export_strategies.restype = ctypes.c_int
            bp.bp_export_strategies.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t)
            ]
            # Streaming variant (no 12 GiB Python alloc). Gated on
            # availability so older .so builds still load: `_stream_bps_export`
            # falls back to the legacy buffer path if this symbol isn't
            # present.
            if hasattr(bp, "bp_export_strategies_streaming"):
                # int (*emit)(const void *, size_t, void *)
                bp.BP_EMIT_FN = ctypes.CFUNCTYPE(
                    ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                    ctypes.c_void_p)
                bp.bp_export_strategies_streaming.restype = ctypes.c_int
                bp.bp_export_strategies_streaming.argtypes = [
                    ctypes.c_void_p, bp.BP_EMIT_FN, ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_size_t)
                ]
            bp.bp_set_preflop_tier.restype = ctypes.c_int
            bp.bp_set_preflop_tier.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_int
            ]
            bp.bp_get_regrets.restype = ctypes.c_int
            bp.bp_save_regrets.restype = ctypes.c_int
            bp.bp_save_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            bp.bp_load_regrets.restype = ctypes.c_int64
            bp.bp_load_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            # Explicit all-in gate hooks (no-op if older C lib doesn't have them).
            if hasattr(bp, "bp_set_add_explicit_allin"):
                bp.bp_set_add_explicit_allin.restype = ctypes.c_int
                bp.bp_set_add_explicit_allin.argtypes = [
                    ctypes.c_void_p, ctypes.c_int,
                ]
            if hasattr(bp, "bp_drop_trailing_allin_slot"):
                bp.bp_drop_trailing_allin_slot.restype = ctypes.c_int64
                bp.bp_drop_trailing_allin_slot.argtypes = [ctypes.c_void_p]
            # Depth-limit hooks (no-op if older C lib doesn't have them).
            if hasattr(bp, "bp_set_depth_limit_cont_strats"):
                bp.bp_set_depth_limit_cont_strats.restype = ctypes.c_int
                bp.bp_set_depth_limit_cont_strats.argtypes = [
                    ctypes.c_void_p, ctypes.c_int,
                ]
            if hasattr(bp, "bp_set_cont_strat_prob"):
                bp.bp_set_cont_strat_prob.restype = ctypes.c_int
                bp.bp_set_cont_strat_prob.argtypes = [
                    ctypes.c_void_p, ctypes.c_float,
                ]
            # Mean-field opponent-bias-as-action mode
            # (no-op if an older C lib doesn't have it).
            if hasattr(bp, "bp_set_cont_ev_bias_mode"):
                bp.bp_set_cont_ev_bias_mode.restype = ctypes.c_int
                bp.bp_set_cont_ev_bias_mode.argtypes = [
                    ctypes.c_void_p, ctypes.c_int,
                ]
            bp.bp_save_texture_cache.restype = ctypes.c_int
            bp.bp_save_texture_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            bp.bp_load_texture_cache.restype = ctypes.c_int
            bp.bp_load_texture_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            bp.bp_free.restype = None
            # Hash table health stats (optional symbol)
            try:
                bp.bp_get_table_stats.restype = None
                bp.bp_get_table_stats.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                ]
            except AttributeError:
                # Older .so without the new symbol — silently disable.
                bp.bp_get_table_stats = None
            return bp
    raise FileNotFoundError(f"mccfr_blueprint not found in {build_dir}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified 6-player blueprint (Pluribus-style)")
    parser.add_argument("--iterations", type=int, default=0,
                        help="Max iterations (0 = run until time limit)")
    parser.add_argument("--time-limit-hours", type=float, default=0,
                        help="Run for this many hours (Pluribus: 192 = 8 days)")
    parser.add_argument("--num-threads", type=int, default=0,
                        help="OpenMP threads (0=auto)")
    parser.add_argument("--hash-size", type=int, default=0,
                        help="Hash table slots (0=auto, Pluribus=536870912)")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--output-dir", default="blueprint_out",
                        help="Where checkpoints, the exported .bps and the "
                             "abstraction caches go (default: blueprint_out "
                             "in the working directory). The trainer's cache "
                             "directory, BP_CACHE_DIR, defaults to it.")
    parser.add_argument("--checkpoint-interval", type=int, default=1000000,
                        help="Export checkpoint every N iterations")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest regret checkpoint in "
                             "--output-dir")
    parser.add_argument("--resume-from", default="",
                        help="Path (relative to --output-dir) of the regret "
                             "file to resume from. Default: "
                             "regrets_latest.bin.")
    parser.add_argument("--no-explicit-allin", action="store_true",
                        default=(os.environ.get("BPS_NO_EXPLICIT_ALLIN", "0") == "1"),
                        help="Gate the implicit-allin block in "
                             "generate_actions(). Default off. When on, "
                             "deep-stack nodes have no allin in action set "
                             "(short-stack clamp still applies). When "
                             "combined with --resume from a legacy "
                             "checkpoint, runs the post-load fixup that "
                             "drops the trailing allin slot from preflop/"
                             "postflop infosets saved at max-with-allin "
                             "size.")
    parser.add_argument("--river-ochs", action="store_true",
                        default=(os.environ.get("BPS_RIVER_OCHS", "0") == "1"),
                        help="Enable OCHS river abstraction (Johanson 2013, "
                             "Pluribus-faithful). Defaults to env BPS_RIVER_OCHS.")
    parser.add_argument("--river-ochs-clusters", type=int,
                        default=int(os.environ.get("BPS_RIVER_OCHS_CLUSTERS", "0") or 0),
                        help="Number of opponent clusters for OCHS (0=default 8).")
    # Depth-limit + cont-strat-sampling dispatch at the multiway-flop
    # boundary (mccfr_phase_r.c). The boundary dispatches via a HEADS
    # rollout, populates the cont-ev table, and round-trips through BPR5
    # storage.
    parser.add_argument("--depth-limit-cont-strats", action="store_true",
                        default=(os.environ.get("BPS_DEPTH_LIMIT_CONT_STRATS",
                                                "0") == "1"),
                        help="Off by default: terminate the trainer recursion at "
                             "multiway-flop boundaries with probability "
                             "--cont-strat-prob, sample joint cont-bias, "
                             "rollout to showdown, accumulate boundary EV "
                             "slots. Default OFF for backward compatibility "
                             "with existing BPR4 checkpoints. The BPR5 "
                             "trailer write fires automatically on "
                             "bp_save_regrets when cont-strat dispatch has "
                             "populated >=1 boundary slot.")
    parser.add_argument("--cont-strat-prob", type=float,
                        default=float(os.environ.get(
                            "BPS_CONT_STRAT_PROB", "0.5")),
                        help="Probability of HEADS (terminate at "
                             "multiway-flop boundary) vs TAILS (continue "
                             "legacy recursion). Used only when "
                             "--depth-limit-cont-strats is set. Must be in "
                             "[0, 1]. Default 0.5.")
    # Mean-field opponent-bias-as-action mode. Default 'uniform' =
    # legacy (byte-identical); 'mean_field' makes opponents sample their
    # continuation bias by regret matching. Only meaningful together with
    # --depth-limit-cont-strats (the boundary must be armed to sample biases).
    parser.add_argument("--cont-ev-bias-mode",
                        choices=["uniform", "mean_field"],
                        default=os.environ.get("BPS_CONT_EV_BIAS_MODE",
                                               "uniform"),
                        help="Opponent continuation-bias semantics at "
                             "the boundary HEADS branch. 'uniform' (default, "
                             "legacy): opponents' bias drawn uniformly. "
                             "'mean_field': opponents sample their bias by "
                             "regret matching over a per-slot sidecar. Requires "
                             "--depth-limit-cont-strats. env BPS_CONT_EV_BIAS_MODE.")
    parser.add_argument("--checkpoint-prefix", type=str, default="",
                        help="Run label recorded in every manifest this "
                             "worker writes (checkpoint_prefix field). "
                             "Required for per-chunk strategy snapshots, "
                             "so a snapshot always names its run. "
                             "Trailing slashes are stripped.")
    parser.add_argument("--snapshot-bps-every-chunk", action="store_true",
                        default=(os.environ.get(
                            "BPS_SNAPSHOT_BPS_EVERY_CHUNK", "0") == "1"),
                        help="At every mini_chunk boundary "
                             "(BPS_MINI_CHUNK iters), export a "
                             "strategy-only .bps snapshot to "
                             "<prefix>/strategy_snapshots/blueprint_<NNN>M.bps. "
                             "Used to compute reach-weighted exploitability "
                             "at every probe checkpoint instead of just the "
                             "final state. Cost: ~3 min wall + ~5 GB written "
                             "per snapshot on 2B-info-set blueprint. "
                             "Requires --checkpoint-prefix to be set "
                             "(belt-and-braces guard against accidentally "
                             "publishing experimental snapshots).")
    args = parser.parse_args()
    # The C trainer reads and writes its abstraction caches (texture,
    # turn/river centroids, OCHS clusters) in $BP_CACHE_DIR, else the working
    # directory. Keep them with this run's output unless the caller chose.
    os.environ.setdefault("BP_CACHE_DIR", os.path.abspath(args.output_dir))
    # Normalise checkpoint prefix: strip surrounding slashes; empty stays empty.
    args.checkpoint_prefix = args.checkpoint_prefix.strip("/").strip()

    print(f"=== Unified 6-Player Blueprint (Pluribus-style) ===")
    print(f"Players: {NUM_PLAYERS}")
    print(f"Blinds: {SMALL_BLIND}/{BIG_BLIND}")
    print(f"Stack: {INITIAL_STACK} ({INITIAL_STACK//BIG_BLIND}BB)")
    print(f"Preflop tiers: {PREFLOP_TIERS}")
    print(f"Preflop max raises: {PREFLOP_MAX_RAISES}")
    print(f"Postflop bet sizes: {POSTFLOP_BET_SIZES}")
    print(f"Threads: {args.num_threads or 'auto'}")
    print()

    bp_lib = load_bp_dll(args.build_dir)
    print("DLL loaded.")

    # Config — scale to Pluribus timing
    config = BPConfig()
    bp_lib.bp_default_config(ctypes.byref(config))
    config.num_threads = args.num_threads
    config.include_preflop = 1
    config.river_ochs_enabled = 1 if args.river_ochs else 0
    config.river_ochs_num_clusters = args.river_ochs_clusters

    if args.hash_size > 0:
        config.hash_table_size = args.hash_size
    else:
        config.hash_table_size = 3000000000  # 3B slots (~180GB meta; fits in 376GB of RAM)

    # If running by time limit, estimate iterations based on Pluribus core-hours.
    # Pluribus: 12,400 core-hours on 64 cores. The estimate matches total
    # core-hours. Burst throughput is ~280K iter/s (pruned) / ~140K (unpruned), average ~200K.
    # With large checkpoint intervals, overhead is negligible.
    if args.time_limit_hours > 0 and args.iterations == 0:
        effective_threads = args.num_threads if args.num_threads > 0 else (os.cpu_count() or 1)
        # Target: Pluribus's 12,400 core-hours scaled to this core count
        target_hours = 12400 / effective_threads  # wall-clock hours of solving
        # Use conservative average throughput (mix of pruned/unpruned phases)
        est_iter_per_sec = 200000  # ~200K iter/s on 96 threads
        args.iterations = int(target_hours * 3600 * est_iter_per_sec)
        print(f"Target: {12400} core-hours / {effective_threads} cores = {target_hours:.1f}h solving")
        print(f"Estimated throughput: ~{est_iter_per_sec:,} iter/s → {args.iterations:,} iterations")

    if args.iterations <= 0:
        args.iterations = 1000000  # default 1M

    # Scale Pluribus timing parameters proportionally (int64, no overflow concern).
    config.discount_stop_iter = max(args.iterations * 35 // 1000, 1000)
    config.discount_interval = max(config.discount_stop_iter // 40, 100)
    config.prune_start_iter = max(args.iterations * 17 // 1000, 500)
    config.snapshot_start_iter = max(args.iterations * 7 // 100, 10000)
    config.snapshot_interval = max(args.iterations * 17 // 1000, 5000)
    config.strategy_interval = 10000  # Pluribus: every 10K iterations

    print(f"Iterations: {args.iterations:,}")
    print(f"Discount: first {config.discount_stop_iter:,} iters, every {config.discount_interval:,}")
    print(f"Pruning: after iter {config.prune_start_iter:,}")
    print(f"Snapshots: after iter {config.snapshot_start_iter:,}, every {config.snapshot_interval:,}")
    print(f"River OCHS: {'ENABLED' if config.river_ochs_enabled else 'disabled'} "
          f"(clusters={config.river_ochs_num_clusters or 8})")
    print()

    # Reuse a texture cache from a previous run in --output-dir (saves the
    # ~65 min flop-bucket precompute).
    texture_cache_local = os.path.join(args.output_dir, "texture_cache.bin")
    texture_loaded = os.path.exists(texture_cache_local)
    if texture_loaded:
        print(f"Found texture cache at {texture_cache_local}, will load after init")
    else:
        print("No texture cache found, will precompute")

    # Initialize
    buf = (ctypes.c_char * 524288)()
    solver = ctypes.cast(buf, ctypes.c_void_p)

    c_postflop = (ctypes.c_float * len(POSTFLOP_BET_SIZES))(*POSTFLOP_BET_SIZES)
    c_preflop = (ctypes.c_float * len(PREFLOP_BET_SIZES))(*PREFLOP_BET_SIZES)

    # Load texture cache BEFORE init so bp_init_unified skips the 65-min precompute.
    # bp_load_texture_cache allocates the cache array and sets num_cached_textures > 0.
    # bp_init_unified checks this and skips if already loaded.
    if texture_loaded:
        n = bp_lib.bp_load_texture_cache(solver, texture_cache_local.encode('utf-8'))
        if n > 0:
            print(f"Loaded texture cache: {n} textures")
        else:
            print("Failed to load texture cache, will precompute")
            texture_loaded = False

    ret = bp_lib.bp_init_unified(
        solver, NUM_PLAYERS,
        SMALL_BLIND, BIG_BLIND, INITIAL_STACK,
        c_postflop, len(POSTFLOP_BET_SIZES),
        c_preflop, len(PREFLOP_BET_SIZES),
        ctypes.byref(config)
    )
    if ret != 0:
        print(f"FATAL: bp_init_unified returned {ret}")
        sys.exit(1)
    print("Solver initialized.")

    # Gate the implicit-allin block in generate_actions().
    if args.no_explicit_allin:
        if not hasattr(bp_lib, "bp_set_add_explicit_allin"):
            print("FATAL: --no-explicit-allin requested but the loaded "
                  "mccfr_blueprint.so doesn't export "
                  "bp_set_add_explicit_allin. Rebuild the C library "
                  "from this tree.")
            sys.exit(1)
        bp_lib.bp_set_add_explicit_allin(solver, 0)

    # Depth-limit + cont-strat-sampling dispatch (mccfr_phase_r.c).
    if args.depth_limit_cont_strats:
        if not hasattr(bp_lib, "bp_set_depth_limit_cont_strats"):
            print("FATAL: --depth-limit-cont-strats requested but the "
                  "loaded mccfr_blueprint.so doesn't export "
                  "bp_set_depth_limit_cont_strats. Rebuild with the "
                  "depth-limit code (mccfr_phase_r.c).")
            sys.exit(1)
        if not (0.0 <= args.cont_strat_prob <= 1.0):
            print(f"FATAL: --cont-strat-prob {args.cont_strat_prob} out "
                  "of [0, 1]")
            sys.exit(1)
        bp_lib.bp_set_cont_strat_prob(
            solver, ctypes.c_float(args.cont_strat_prob))
        # Memory note: arming this callocs the continuation-EV table up
        # front, 2^26 slots of 72 bytes, so about 4.6 GB of virtual memory
        # (the log prints ~4608 MB). Pages are touched only as boundary
        # slots fill, so resident memory stays small (about 200 MB on the
        # fixture run). A strict overcommit setting or a ulimit -v can
        # refuse it; bp_set_cont_ev_table_size picks a smaller table.
        bp_lib.bp_set_depth_limit_cont_strats(solver, 1)
        print(f"[depth-limit] cont-strat dispatch ACTIVE "
              f"(cont_strat_prob={args.cont_strat_prob}). HEADS-branch "
              f"rollouts will populate boundary EV slots; "
              f"bp_save_regrets writes BPR5 trailer on completion.")

        # Mean-field opponent-bias-as-action mode.
        # Only applied when a boundary is armed (bias sampling happens at the
        # HEADS branch). Default 'uniform' is a no-op (byte-identical legacy).
        if args.cont_ev_bias_mode == "mean_field":
            if not hasattr(bp_lib, "bp_set_cont_ev_bias_mode"):
                print("FATAL: --cont-ev-bias-mode mean_field requested but "
                      "the loaded mccfr_blueprint.so doesn't export "
                      "bp_set_cont_ev_bias_mode. Rebuild with the "
                      "mean-field bias mode.")
                sys.exit(1)
            if bp_lib.bp_set_cont_ev_bias_mode(solver, 1) != 0:
                print("FATAL: bp_set_cont_ev_bias_mode(MEANFIELD) failed")
                sys.exit(1)
            print("[mean-field] opponent-bias mode ACTIVE — "
                  "opponents sample continuation bias by regret matching.")
    elif args.cont_ev_bias_mode == "mean_field":
        print("WARNING: --cont-ev-bias-mode mean_field has no effect without "
              "--depth-limit-cont-strats (no boundary armed to sample biases "
              "at). Ignoring.")

    # Set tiered preflop sizing (overrides flat sizes)
    for level, sizes in sorted(PREFLOP_TIERS.items()):
        c_sizes = (ctypes.c_float * len(sizes))(*sizes)
        ret = bp_lib.bp_set_preflop_tier(
            solver, level, c_sizes, len(sizes), PREFLOP_MAX_RAISES)
        if ret != 0:
            print(f"FATAL: bp_set_preflop_tier level {level} returned {ret}")
            sys.exit(1)
    print(f"Tiered preflop: {len(PREFLOP_TIERS)} levels, max {PREFLOP_MAX_RAISES} raises")

    # Hard preflight: verify the loaded/precomputed texture cache's mixer
    # matches the runtime mixer. If it doesn't,
    # bp_compute_flop_bucket_for_test returns -1 (texture_index_lookup
    # fails on every flop), which means the in-tree flop info-set keying
    # falls through to bucket_map[1] = identity = hand_idx, silently
    # wrong for every flop. Abort BEFORE the long training rather than
    # finding it at the BPR5 verifier days later.
    if hasattr(bp_lib, "bp_compute_flop_bucket_for_test"):
        bp_lib.bp_compute_flop_bucket_for_test.restype = ctypes.c_int
        bp_lib.bp_compute_flop_bucket_for_test.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.c_int]
        # 3 known-distinct-texture flops covering rainbow/paired/monotone.
        # rank<<2 | suit. card 0 = 2c, etc.
        smoke_flops = [
            ([0*4+0, 5*4+1, 11*4+2], "2c-7d-Kh rainbow"),
            ([12*4+3, 12*4+0, 7*4+1], "As-Ac-9d paired"),
            ([8*4+2, 4*4+2, 1*4+2],  "Th-6h-3h monotone"),
        ]
        smoke_fail = 0
        for flop_cards, name in smoke_flops:
            arr = (ctypes.c_int * 3)(*flop_cards)
            b = bp_lib.bp_compute_flop_bucket_for_test(solver, arr, 0, 0)
            ok = (0 <= b < 200)
            print(f"[texture preflight] flop {name} hand_idx=0 "
                  f"-> bucket={b} {'PASS' if ok else 'FAIL'}", flush=True)
            if not ok:
                smoke_fail += 1
        if smoke_fail > 0:
            print(f"[FATAL] texture preflight FAILED on "
                  f"{smoke_fail}/{len(smoke_flops)} flops. The texture "
                  f"cache's stored hashes do NOT match the runtime mixer "
                  f"(boost vs splitmix). Abort before training: the "
                  f"in-tree flop info-set keying would fall through to "
                  f"bucket_map[1] identity (hand_idx, not flop bucket). "
                  f"Recovery: rebuild texture_cache.bin under the runtime's "
                  f"actual mixer, or call bp_set_legacy_hash_mixer() to "
                  f"match the cache.", flush=True)
            sys.exit(2)
        print("[texture preflight] PASS — texture lookup "
              "returns valid buckets in [0, 200) for all 3 test flops.",
              flush=True)
    else:
        print("[texture preflight] bp_compute_flop_bucket_for_test "
              "not present in DLL; skipping smoke test (older binary).",
              flush=True)

    # Save texture cache for future runs (skips 65-min precompute next time)
    if not texture_loaded and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        bp_lib.bp_save_texture_cache(solver, texture_cache_local.encode('utf-8'))
        print(f"Texture cache saved to {texture_cache_local}")
    # Postflop buckets (200 EHS percentile) are precomputed inside bp_init_unified
    # for all 1,755 flop textures. During traversal, the dealt flop is canonicalized
    # and the precomputed bucket mapping is looked up in O(1755) per deal.

    # Resume from checkpoint if requested
    iters_already_done = 0
    if args.resume:
        resume_key = args.resume_from or "regrets_latest.bin"
        regret_local = os.path.join(args.output_dir, resume_key)
        print(f"Checking for checkpoint at {regret_local}...")
        if os.path.exists(regret_local):
            n_loaded = bp_lib.bp_load_regrets(
                solver, regret_local.encode('utf-8'))
            if n_loaded <= 0:
                # A checkpoint that exists but does not load is not a fresh
                # start. Stop, and keep the file for inspection.
                print(f"[BP][FATAL] {regret_local} did not load "
                      f"(bp_load_regrets returned {n_loaded}). The file is "
                      f"kept; move it away to start fresh.", flush=True)
                sys.exit(2)

            # Post-load load-factor sanity check. A 1.5B-entry checkpoint
            # loaded into a 2B-slot table runs near saturation, and the
            # table then drops entries silently. Refuse to proceed if the
            # loaded count brings load factor above 0.85; the operator must
            # bump --hash-size first.
            if n_loaded > 0 and hasattr(bp_lib, "bp_get_table_stats"):
                _e = ctypes.c_int64(0); _s = ctypes.c_int64(0)
                _f = ctypes.c_int64(0); _p = ctypes.c_int64(0)
                bp_lib.bp_get_table_stats(
                    solver,
                    ctypes.byref(_e), ctypes.byref(_s),
                    ctypes.byref(_f), ctypes.byref(_p),
                )
                lf = (_e.value / _s.value) if _s.value > 0 else 0.0
                print(f"[BP] post-load: entries={_e.value:,} table_size={_s.value:,} "
                      f"load={lf*100:.1f}% max_probe={_p.value} "
                      f"insertion_failures={_f.value}", flush=True)
                if _f.value > 0:
                    print(f"[BP][FATAL] {_f.value} insertion failures during load — "
                          f"checkpoint partially dropped. Bump --hash-size and retry.",
                          flush=True)
                    sys.exit(2)
                if lf > 0.85:
                    print(f"[BP][FATAL] load factor {lf*100:.1f}% exceeds 0.85 "
                          f"safety bound. Risk of probe-cap exhaustion during "
                          f"training. Bump --hash-size to at least "
                          f"{int(_e.value / 0.50):,} and retry.", flush=True)
                    sys.exit(2)

            if n_loaded > 0 and args.no_explicit_allin:
                # Lazy shrink: when --no-explicit-allin is set, the
                # next find_or_create call for each infoset will shrink the
                # slot's num_actions to match the new (smaller) action set.
                # No explicit fixup needed; lazy shrink handles preflop
                # tiers + postflop subsequent-raise cases uniformly.
                # See mccfr_blueprint.c::info_table_find_or_create.
                print("[explicit-allin] --no-explicit-allin active. Slots will "
                      "lazily shrink on first visit.")
            if n_loaded > 0:
                # Read back iterations_run from the solver struct
                # (bp_load_regrets sets s->iterations_run)
                # Python can't easily read it, but the C code printed it.
                # Parse it from the checkpoint metadata instead.
                # _export_checkpoint writes checkpoint_meta.json next to
                # regrets_latest.bin in --output-dir. A missing or unreadable
                # file stops the run (_read_checkpoint_meta).
                cmeta = _read_checkpoint_meta(args.output_dir)
                iters_already_done = cmeta["iterations"]
                # Stale-checkpoint guard: refuse to continue if the
                # checkpoint's config hash doesn't match the current
                # config (for example a tiered run loading a
                # flat-sizing checkpoint).
                ckpt_hash = cmeta.get("config_hash")
                # The config hash optionally includes
                # add_explicit_allin. Legacy v3 checkpoints were
                # saved BEFORE this field existed in the hash, so
                # their hash was computed sans field. Detect this
                # by absence of "add_explicit_allin" in meta and
                # use a sans-field comparison hash.
                ckpt_allin_raw = cmeta.get("add_explicit_allin")
                if ckpt_allin_raw is None:
                    # Legacy meta: hash was computed without the
                    # add_explicit_allin field. Reproduce by
                    # passing None.
                    ckpt_compat_hash = _compute_config_hash(
                        add_explicit_allin=None)
                    ckpt_allin = 1  # legacy behavior
                else:
                    ckpt_allin = int(ckpt_allin_raw)
                    ckpt_compat_hash = _compute_config_hash(
                        add_explicit_allin=ckpt_allin)
                cur_allin = 0 if args.no_explicit_allin else 1
                cur_hash = _compute_config_hash(add_explicit_allin=cur_allin)
                if (args.no_explicit_allin
                        and ckpt_allin == 1
                        and ckpt_hash == ckpt_compat_hash
                        and os.environ.get("BPS_PHASE3_ALLOW_LEGACY_CKPT", "0") == "1"):
                    print(f"[explicit-allin] Resuming legacy (allin=1) "
                          f"checkpoint under --no-explicit-allin "
                          f"gate. Lazy shrink will trim trailing "
                          f"allin slots on first visit. "
                          f"(ckpt_hash={ckpt_hash[:16]}..., "
                          f"compat={ckpt_compat_hash[:16]}...)")
                    ckpt_hash = None  # skip mismatch path below
                if ckpt_hash and ckpt_hash != cur_hash:
                    # The config hash no longer includes hash_table_size,
                    # but old checkpoints were saved with it. Try matching
                    # with the old hash_table_size value to distinguish
                    # a benign resize from a real config incompatibility.
                    import hashlib as _hl
                    old_ht = os.environ.get("BPS_OLD_HASH_SIZE", "750000000")
                    _p = json.loads(json.dumps({
                        "num_players": NUM_PLAYERS,
                        "blinds": [SMALL_BLIND, BIG_BLIND],
                        "initial_stack": INITIAL_STACK,
                        "preflop_tiers": {str(k): v for k, v in sorted(PREFLOP_TIERS.items())},
                        "preflop_max_raises": PREFLOP_MAX_RAISES,
                        "postflop_bet_sizes": POSTFLOP_BET_SIZES,
                        "preflop_buckets": 169,
                        "postflop_buckets": 200,
                        "dcfr_alpha": os.environ.get("BPS_DCFR_ALPHA", ""),
                        "dcfr_beta": os.environ.get("BPS_DCFR_BETA", ""),
                        "dcfr_gamma": os.environ.get("BPS_DCFR_GAMMA", ""),
                        "river_ochs": os.environ.get("BPS_RIVER_OCHS", "0"),
                        "river_ochs_clusters": os.environ.get("BPS_RIVER_OCHS_CLUSTERS", ""),
                        "hash_table_size": old_ht,
                    }))
                    _c = json.dumps(_p, sort_keys=True, separators=(',', ':'))
                    legacy_hash = _hl.sha256(_c.encode()).hexdigest()
                    if ckpt_hash == legacy_hash:
                        print(f"[INFO] Config hash matched via legacy "
                              f"format (hash_table_size resize). "
                              f"Resuming.")
                    else:
                        print(f"[FATAL] Config hash mismatch.\n"
                              f"  checkpoint: {ckpt_hash}\n"
                              f"  current:    {cur_hash}\n"
                              f"Refusing to resume — action/bucket "
                              f"layout may be incompatible.")
                        sys.exit(2)
                if not ckpt_hash:
                    print("[WARN] Checkpoint has no config_hash "
                          "(pre-v3 format). Resuming at your risk.")
                print(f"Resumed from checkpoint: {n_loaded} info sets, "
                      f"{iters_already_done:,} iterations already done")
            # Only a checkpoint that loaded and passed the checks above gets
            # here. When training continues, the next checkpoint writes
            # regrets_latest.bin again, so removing the loaded file now frees
            # its disk space. A run that is already complete keeps it.
            if iters_already_done < args.iterations:
                os.remove(regret_local)
        else:
            print("No checkpoint found, starting fresh.")

    # Solve in chunks with periodic checkpoints.
    # Each checkpoint writes the regret table into --output-dir.
    # If the process dies, at most one chunk of work is lost.
    # The C solver's hash table accumulates regrets across bp_solve calls,
    # so calling bp_solve(N) then bp_solve(M) equals bp_solve(N+M).
    total_iters = args.iterations
    iters_done = iters_already_done
    iters_remaining = total_iters - iters_done

    if iters_remaining <= 0:
        print(f"Already completed {iters_done:,} >= {total_iters:,} target. Done.")
        bp_lib.bp_free(solver)
        return

    print(f"\nStarting solve: {iters_remaining:,} iterations remaining "
          f"(of {total_iters:,}), adaptive checkpoints...")
    # Crash debugging: flush on every print and catch signals
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    t0 = time.time()

    # Mutable state for the signal handler to checkpoint on SIGTERM.
    # A preempted machine usually gets SIGTERM with a short warning, enough
    # time to flush regrets to disk and save up to 48 min of work.
    _sig_state = {"iters_done": iters_done, "solver": solver}

    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        if signum == signal.SIGTERM:
            print(f"\n[SIGTERM] Preemption or shutdown — saving emergency checkpoint...",
                  flush=True)
            try:
                elapsed = time.time() - t0
                n_is = bp_lib.bp_num_info_sets(_sig_state["solver"])
                _export_checkpoint(bp_lib, _sig_state["solver"], args,
                                   _sig_state["iters_done"], elapsed, n_is,
                                   label="emergency")
                print("[SIGTERM] Emergency checkpoint saved.", flush=True)
            except Exception as e:
                print(f"[SIGTERM] Checkpoint save failed: {e}", flush=True)
        else:
            print(f"\n[FATAL] Caught signal {signum} ({sig_name})", flush=True)
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass

    # Checkpoint cadence: every 100M iters. At ~110K iter/s, 100M = ~15min
    # wall-clock, so a preemption loses at most ~15min of work. Each checkpoint = 76 GB regret file + ~50 KB meta.
    # Saving takes ~3 min at 1 GB/s disk throughput. Overhead: ~20% of
    # wall time.
    #
    checkpoint_interval_iters = 100_000_000
    checkpoint_milestones = [
        (i + 1) * checkpoint_interval_iters
        for i in range(max(1, int(total_iters // checkpoint_interval_iters)))
    ]
    # After last explicit milestone, keep the same cadence
    checkpoint_ramp_interval = checkpoint_interval_iters
    last_milestone = checkpoint_milestones[-1] if checkpoint_milestones else 0

    def _next_checkpoint(current):
        """Return the next checkpoint iteration after current."""
        for m in checkpoint_milestones:
            if m > current:
                return m
        # Past all milestones: use ramp interval
        base = last_milestone
        while base <= current:
            base += checkpoint_ramp_interval
        return base

    next_checkpoint_at = _next_checkpoint(iters_done)

    # bp_solve takes int32 max_iterations for the C call.
    INT32_MAX = 2_147_483_647

    # Solve in mini-chunks (50M) with lightweight strategy probes between each.
    # Full checkpoint (regret save) only at adaptive milestones.
    # Probes and experiments can override via BPS_MINI_CHUNK env to get a
    # finer-grained convergence trajectory (e.g. 10M for 5 points across a
    # 50M-iter run). Capped to avoid pathological 0/negative values.
    mini_chunk = max(
        1_000_000,
        int(os.environ.get("BPS_MINI_CHUNK", "50000000")))

    while iters_done < total_iters:
        # Solve one mini-chunk
        call_size = min(mini_chunk, total_iters - iters_done, INT32_MAX)
        call_size_int = int(call_size)
        print(f"[Solve] {call_size_int:,} iters from {iters_done:,}...", flush=True)
        ret = bp_lib.bp_solve(solver, ctypes.c_int64(call_size_int))
        if ret != 0:
            print(f"[Solve] bp_solve returned {ret}", flush=True)
        iters_done += call_size
        _sig_state["iters_done"] = iters_done

        elapsed = time.time() - t0
        n_is = bp_lib.bp_num_info_sets(solver)
        ips = iters_done / elapsed if elapsed > 0 else 0

        # Hash table health stats. Optional symbol; gated on availability so
        # the script still works against older .so builds.
        if getattr(bp_lib, 'bp_get_table_stats', None) is not None:
            ht_entries = ctypes.c_int64(0)
            ht_size = ctypes.c_int64(0)
            ht_fails = ctypes.c_int64(0)
            ht_max_probe = ctypes.c_int64(0)
            bp_lib.bp_get_table_stats(
                solver,
                ctypes.byref(ht_entries),
                ctypes.byref(ht_size),
                ctypes.byref(ht_fails),
                ctypes.byref(ht_max_probe),
            )
            if ht_size.value > 0:
                load_pct = 100.0 * ht_entries.value / ht_size.value
                print(
                    f"[Table] entries={ht_entries.value:,} "
                    f"({load_pct:.2f}% load), "
                    f"ins_fails={ht_fails.value}, "
                    f"max_probe={ht_max_probe.value}",
                    flush=True,
                )
                if ht_fails.value > 0:
                    print(
                        f"[Table] WARNING: {ht_fails.value} insertion failures — "
                        f"hash table too small or pathologically clustered. "
                        f"Bump --hash-size for next run.",
                        flush=True,
                    )

        # Optional per-mini_chunk strategy snapshot for reach-weighted
        # exploitability trajectory. Gated on the CLI flag + a non-empty
        # checkpoint_prefix (the helper itself enforces the prefix guard;
        # the gate here just avoids the wasted print on the legacy path).
        if (args.snapshot_bps_every_chunk
                and args.checkpoint_prefix
                and iters_done < total_iters):
            # Skip the snapshot at the FINAL chunk — _export_checkpoint
            # writes the canonical-equivalent (prefixed) BPS at iters_done
            # == total_iters anyway, so the snapshot would be redundant.
            _export_strategy_snapshot(
                bp_lib, solver, args, iters_done, n_is, elapsed)

        # Lightweight strategy probe: extract strategies + raw regrets
        # directly from memory via bp_get_strategy/bp_get_regrets (no disk I/O).
        probe = _probe_preflop(bp_lib, solver)
        if probe:
            header = f"[Probe {iters_done:,}]"
            for line in probe.split("\n"):
                print(f"{header} {line}", flush=True)
            if args.output_dir:
                probe_path = os.path.join(args.output_dir, "probe_latest.txt")
                with open(probe_path, 'w') as pf:
                    pf.write(f"iteration: {iters_done}\n{probe}\n")

        # Full checkpoint at adaptive milestones (200M, 400M, 600M, 2B, 4B, ...)
        if iters_done >= next_checkpoint_at or iters_done >= total_iters:
            remaining_h = (total_iters - iters_done) / ips / 3600 if ips > 0 else 0
            print(f"[Checkpoint] {iters_done:,}/{total_iters:,} iters, "
                  f"{n_is:,} IS, {elapsed/3600:.1f}h elapsed, "
                  f"~{remaining_h:.1f}h remaining, {ips:.0f} iter/s")
            if args.output_dir:
                _export_checkpoint(bp_lib, solver, args, iters_done, elapsed, n_is)
            next_checkpoint_at = _next_checkpoint(iters_done)

    elapsed = time.time() - t0
    n_is = bp_lib.bp_num_info_sets(solver)
    print(f"\nDone: {iters_done:,} iterations, {n_is:,} info sets")
    print(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Speed: {iters_done/elapsed:.0f} iter/s")
    print(f"CPU-hours: {elapsed * max(args.num_threads, 1) / 3600:.0f}")

    # Final export (same as last checkpoint, but labeled as final)
    if args.output_dir:
        _export_checkpoint(bp_lib, solver, args, iters_done, elapsed, n_is,
                           label="final")

    bp_lib.bp_free(solver)
    print("\nTraining complete.")

    print("\nAll done.")


def _probe_preflop(bp_lib, solver):
    """Extract richer preflop convergence diagnostics.

    Three sections:

      OPEN — open-raise strategy at every position (UTG, MP, CO, BTN, SB, BB).
        Per-raise-size frequencies (not aggregated R%) to show whether
        r0.5x / r0.7x / r1.0x are converging differentially. UTG = no
        prior action; MP = UTG folded; CO = UTG+MP folded; BTN = UTG+MP+CO
        folded; SB = UTG+MP+CO+BTN folded; BB = UTG+MP+CO+BTN+SB called
        (BB defends). Format:
          OPEN UTG AA: F0 C5 R0.5x:30 R0.7x:35 R1.0x:30
          OPEN BTN 22: F60 C20 R0.5x:5 R0.7x:5 R1.0x:10

      VS_OPEN — defending strategy when villain (UTG) opens to 3bb.
        BB and BTN as defenders. Shows fold/call/3bet ranges.
        Format:
          VS_OPEN BB AA: F2 C20 R3bet=78
          VS_OPEN BTN AKs: F0 C40 R3bet=60

      VS_3BET — 4bet response when villain (BB) 3bets the UTG opener.
        Detects whether 5-bet jam line is live. Format:
          VS_3BET UTG AA: F2 C5 R4bet=93
          VS_3BET UTG TT: F30 C50 R4bet=20

      REGRETS — raw int regrets for diagnostic hands at all 6 positions.
        Per-raise-size regrets to show which sizes are saturated
        vs growing. Pruning indicator (`pruned=N/M`) shows how many
        raise actions are below the prune threshold (-300M).

      TABLE — hash table health: entries, load factor, insertion failures,
        max probe distance. Critical for catching saturation early.
    """
    try:
        # PRUNE_THRESHOLD in C is -300M; mirror here for "is this raise
        # currently pruned?" reporting. Don't conflate with REGRET_FLOOR.
        PRUNE_THRESH = -300_000_000

        # Strategically diverse hand spread (37 hands).
        #
        # Bucket indices come from _C_BUCKET_MAP — built programmatically
        # from the C class_map iteration in mccfr_blueprint.c:init_unified.
        # Verified at module load time; if this assertion ever fires the
        # C side has changed and probes will mislabel.
        diag_hand_names = [
            # Premium pairs + big aces
            "AA", "AKs", "AKo", "AQs", "AQo", "AJs", "ATs", "A5s", "A2s",
            # K-range
            "KK", "KQs", "KQo", "KJs", "KTs", "K9s",
            # Q-range
            "QQ", "QJs", "QTs",
            # J-range
            "JJ", "JTs",
            # T+ pairs
            "TT", "99", "88", "77", "66", "55", "44", "33", "22",
            # Suited connectors
            "T9s", "98s", "87s", "76s", "65s", "54s",
            # Trash benchmarks
            "T2o", "72o", "32o",
        ]
        diag_hands = [(n, _C_BUCKET_MAP[n]) for n in diag_hand_names]

        # All 6 preflop positions with prefix action sequences. The C tree
        # uses traversal order (UTG, MP, CO, BTN, SB, BB) with action
        # indices: 0=fold, 1=call/check, 2..=raise sizes.
        # "Open" position: every prior actor folded (action 0 each).
        open_positions = [
            ("UTG", 2, []),                  # first to act
            ("MP",  3, [0]),                 # UTG folded
            ("CO",  4, [0, 0]),              # UTG, MP folded
            ("BTN", 5, [0, 0, 0]),           # UTG, MP, CO folded
            ("SB",  0, [0, 0, 0, 0]),        # all folded to SB
            ("BB",  1, [0, 0, 0, 0, 1]),     # UTG-SB folded (1=SB call)
        ]

        # UTG's preflop actions at the open (na=5):
        #   0=fold, 1=call, 2=r0.5x, 3=r0.7x, 4=r1.0x   (tier-0 sizes)
        #   plus auto-allin only if r1.0x doesn't already make UTG allin.
        # vs-open: UTG opens r0.7x (action 3), then 0/0/0 fold across MP/CO.
        vs_open_positions = [
            ("BB",  1, [3, 0, 0, 0, 0]),     # UTG r0.7x; MP/CO/BTN/SB fold
            ("BTN", 5, [3, 0, 0]),           # UTG r0.7x; MP/CO fold
        ]

        # BB's response to UTG's r0.7x has na ≈ 5:
        #   0=fold, 1=call, 2=r0.7x (3bet), 3=r1.0x (3bet),
        #   4=allin (auto-added when r1.0x doesn't already shove)
        # vs-3bet: UTG opens r0.7x, MP/CO/BTN/SB fold, BB 3bets r1.0x.
        # BB action 3 = r1.0x 3bet (NOT 4 — that would be BB shove).
        vs_3bet_positions = [
            ("UTG", 2, [3, 0, 0, 0, 0, 3]),  # last 3 = BB r1.0x 3bet
        ]

        strat_buf = (ctypes.c_float * 16)()
        regret_buf = (ctypes.c_int * 16)()
        empty_board = (ctypes.c_int * 1)(0)

        lines = []

        def probe_strat(player, action_seq, bucket):
            """Return (na, [strategy_freqs])  or (0, None) if not found."""
            if action_seq:
                c_seq = (ctypes.c_int * len(action_seq))(*action_seq)
                seq_len = len(action_seq)
            else:
                c_seq = (ctypes.c_int * 1)(0)
                seq_len = 0
            na = bp_lib.bp_get_strategy(solver, player, empty_board, 0,
                                         c_seq, seq_len, strat_buf, bucket)
            if na <= 0:
                return 0, None
            return na, [strat_buf[a] for a in range(na)]

        def probe_regrets(player, action_seq, bucket):
            """Return (na, [regret_ints]) or (0, None) if not found."""
            if action_seq:
                c_seq = (ctypes.c_int * len(action_seq))(*action_seq)
                seq_len = len(action_seq)
            else:
                c_seq = (ctypes.c_int * 1)(0)
                seq_len = 0
            na = bp_lib.bp_get_regrets(solver, player, empty_board, 0,
                                        c_seq, seq_len, regret_buf, bucket)
            if na <= 0:
                return 0, None
            return na, [regret_buf[a] for a in range(na)]

        # ── OPEN strategies — per-raise-size frequencies ──
        # Tier 0 sizes: [0.5, 0.7, 1.0]  → raise actions at index 2,3,4.
        # Some positions have a fold action (0) only when there's
        # something to fold to — UTG has no fold (open spot), so na may
        # be 4 (check, r0.5, r0.7, r1.0). BB has fold + check + raises.
        # The probe detects na and labels by index dynamically.
        for pos_name, player, action_seq in open_positions:
            for hand_name, bucket in diag_hands:
                na, strat = probe_strat(player, action_seq, bucket)
                if not strat:
                    lines.append(f"OPEN {pos_name} {hand_name}: not_found")
                    continue
                # Pretty-print: pct for each action
                # First decide which actions exist:
                #   UTG/MP/CO/BTN: na=4 (check, r0.5, r0.7, r1.0) typically
                #   BB: na=5 (fold, check, r0.5, r0.7, r1.0)
                #   SB: na=5 (fold, call, r0.5, r0.7, r1.0)
                # The C side doesn't expose labels; use heuristic:
                # if to_call > 0 there's fold + call, otherwise check.
                # Record raw (action_idx, freq) pairs.
                pcts = [int(round(p * 100)) for p in strat]
                # Skip near-zero entries to keep the line compact.
                tail = " ".join(
                    f"a{i}={pcts[i]}" for i in range(len(pcts))
                    if pcts[i] >= 1
                )
                lines.append(
                    f"OPEN {pos_name} {hand_name} "
                    f"(na={na}): {tail}"
                )

        # ── VS_OPEN — defender strategies after UTG opens r0.7x ──
        for pos_name, player, action_seq in vs_open_positions:
            for hand_name, bucket in diag_hands:
                na, strat = probe_strat(player, action_seq, bucket)
                if not strat:
                    lines.append(f"VS_OPEN {pos_name} {hand_name}: not_found")
                    continue
                pcts = [int(round(p * 100)) for p in strat]
                tail = " ".join(
                    f"a{i}={pcts[i]}" for i in range(len(pcts))
                    if pcts[i] >= 1
                )
                lines.append(
                    f"VS_OPEN {pos_name} {hand_name} "
                    f"(na={na}): {tail}"
                )

        # ── VS_3BET — UTG facing BB's 3bet ──
        for pos_name, player, action_seq in vs_3bet_positions:
            for hand_name, bucket in diag_hands:
                na, strat = probe_strat(player, action_seq, bucket)
                if not strat:
                    lines.append(f"VS_3BET {pos_name} {hand_name}: not_found")
                    continue
                pcts = [int(round(p * 100)) for p in strat]
                tail = " ".join(
                    f"a{i}={pcts[i]}" for i in range(len(pcts))
                    if pcts[i] >= 1
                )
                lines.append(
                    f"VS_3BET {pos_name} {hand_name} "
                    f"(na={na}): {tail}"
                )

        # ── REGRETS — raw int regrets at OPEN positions for all hands ──
        # Per-raise-size regrets shows which sizes are pruned vs growing.
        for pos_name, player, action_seq in open_positions:
            for hand_name, bucket in diag_hands:
                na, regrets = probe_regrets(player, action_seq, bucket)
                if not regrets:
                    lines.append(f"REGRETS {pos_name} {hand_name}: not_found")
                    continue
                # Show all action regrets so fold vs call vs each raise
                # size is visible individually.
                regret_str = "/".join(f"{r:+d}" for r in regrets)
                n_pruned = sum(1 for r in regrets if r < PRUNE_THRESH)
                lines.append(
                    f"REGRETS {pos_name} {hand_name} "
                    f"(na={na}): [{regret_str}] pruned={n_pruned}/{na}"
                )

        # ── TABLE — hash health stats ──
        if getattr(bp_lib, 'bp_get_table_stats', None) is not None:
            ht_entries = ctypes.c_int64(0)
            ht_size = ctypes.c_int64(0)
            ht_fails = ctypes.c_int64(0)
            ht_max_probe = ctypes.c_int64(0)
            bp_lib.bp_get_table_stats(
                solver,
                ctypes.byref(ht_entries),
                ctypes.byref(ht_size),
                ctypes.byref(ht_fails),
                ctypes.byref(ht_max_probe),
            )
            if ht_size.value > 0:
                load = 100.0 * ht_entries.value / ht_size.value
                lines.append(
                    f"TABLE entries={ht_entries.value} size={ht_size.value} "
                    f"load={load:.2f}% ins_fails={ht_fails.value} "
                    f"max_probe={ht_max_probe.value}"
                )

        return "\n".join(lines)
    except Exception as e:
        return f"probe_error:{e}"


def _stream_bps_export(bp_lib, solver, bps_path, meta_dict):
    """OOM-safe streaming export of strategies to a .bps file.

    Used by both the final-checkpoint export (`_export_checkpoint`) and
    per-mini-chunk strategy snapshots (`_export_strategy_snapshot`).

    Drives `bp_export_strategies_streaming` so the raw strat blob is
    consumed in 4 MB chunks emitted by the C side, rather than materializing
    the full (~12 GiB) strat_buf in Python via
    `(ctypes.c_char * strat_size)()`. On top of the live regret table, that
    single allocation can exhaust a 384 GB machine around 1B iterations.

    Returns the on-disk compressed_size in bytes, or 0 if the solver had
    no strategies to export (caller should treat that as a no-op).
    """
    import lzma
    import struct

    # Size query first — both APIs accept emit/buf == NULL for this.
    needed = ctypes.c_size_t(0)
    if hasattr(bp_lib, "bp_export_strategies_streaming"):
        bp_lib.bp_export_strategies_streaming(
            solver, ctypes.cast(None, bp_lib.BP_EMIT_FN), None,
            ctypes.byref(needed))
    else:
        bp_lib.bp_export_strategies(solver, None, 0, ctypes.byref(needed))
    strat_size = needed.value
    if strat_size <= 0:
        return 0

    print(f"  Streaming strat export: "
          f"{strat_size / 1024 / 1024 / 1024:.2f} GiB raw",
          flush=True)

    meta_bytes = json.dumps(meta_dict, separators=(',', ':')).encode('utf-8')
    out_tmp = bps_path + ".tmp"
    compressor = lzma.LZMACompressor(preset=1)

    # Per-call state for the emit callback — mutable via list to avoid
    # `nonlocal` (which only binds enclosing-function locals; lambdas /
    # callbacks are easier to reason about with explicit boxes).
    state = {
        "bytes_in": 0,
        "compressed_size": 0,
        "last_pct_logged": -1,
        "file": None,
        "err": None,
    }

    def _emit_callback(buf_ptr, n, _ctx):
        try:
            # Copy n bytes out of the C-owned scratch buffer. The copy is
            # 4 MB max (BP_EXPORT_STREAM_BUF_SIZE), so Python never holds
            # the entire 12 GiB strat blob in memory.
            chunk = ctypes.string_at(buf_ptr, n)
            out_chunk = compressor.compress(chunk)
            if out_chunk:
                state["file"].write(out_chunk)
                state["compressed_size"] += len(out_chunk)
            state["bytes_in"] += n
            pct = int(100 * state["bytes_in"] / strat_size)
            if pct >= state["last_pct_logged"] + 10:
                print(f"    compress {pct}% "
                      f"(in={state['bytes_in'] / 1024 / 1024 / 1024:.1f} GiB, "
                      f"out={state['compressed_size'] / 1024 / 1024 / 1024:.2f} GiB)",
                      flush=True)
                state["last_pct_logged"] = pct
            return 0
        except Exception as e:
            state["err"] = e
            return -1

    if hasattr(bp_lib, "bp_export_strategies_streaming"):
        cb = bp_lib.BP_EMIT_FN(_emit_callback)
    else:
        cb = None  # legacy fallback path below allocates strat_buf

    with open(out_tmp, 'wb') as f:
        f.write(b'BPS3')
        header_compressed_size_offset = f.tell()
        f.write(struct.pack('<QI', 0, len(meta_bytes)))
        state["file"] = f

        if cb is not None:
            written = ctypes.c_size_t(0)
            rc = bp_lib.bp_export_strategies_streaming(
                solver, cb, None, ctypes.byref(written))
            if rc != 0:
                if state["err"] is not None:
                    raise state["err"]
                raise RuntimeError(
                    f"bp_export_strategies_streaming returned {rc}")
            if written.value != strat_size:
                raise RuntimeError(
                    f"strat export size mismatch: queried {strat_size}, "
                    f"streamed {written.value}")
        else:
            # Legacy fallback: only kept so a brand-new code build that
            # somehow ships an older .so still works. The OOM risk is
            # back in this path — explicit so a future reader sees it.
            strat_buf = (ctypes.c_char * strat_size)()
            written = ctypes.c_size_t(0)
            bp_lib.bp_export_strategies(solver, strat_buf, strat_size,
                                        ctypes.byref(written))
            CHUNK = 64 * 1024 * 1024
            mv = memoryview(strat_buf)[:written.value]
            pos = 0
            while pos < written.value:
                chunk_end = min(pos + CHUNK, written.value)
                _emit_callback(
                    ctypes.addressof(strat_buf) + pos, chunk_end - pos, None)
                pos = chunk_end

        tail = compressor.flush()
        if tail:
            f.write(tail)
            state["compressed_size"] += len(tail)
        f.write(meta_bytes)
        f.seek(header_compressed_size_offset)
        f.write(struct.pack('<Q', state["compressed_size"]))

    os.replace(out_tmp, bps_path)
    return state["compressed_size"]


def _export_strategy_snapshot(bp_lib, solver, args, iters_done, n_is,
                              elapsed):
    """Export a strategy-only snapshot at a probe checkpoint.

    Skips the regret save (~105 GB) and writes only the .bps, to
    <output-dir>/strategy_snapshots/blueprint_<iters>.bps. This is the path
    that produced the published 8.05B blueprint. --checkpoint-prefix must be
    set so every snapshot manifest names its run.
    """
    if not args.checkpoint_prefix:
        # Refuse to write strategy snapshots without a prefix — the
        # `unified_blueprint_<iter_tag>.bps` versioned key is stable and
        # safe, but explicit prefix-gating is a belt-and-braces guard
        # against accidentally publishing experimental snapshots to
        # canonical-adjacent keys (an audit-friendly invariant: snapshot
        # path always contains the run-id).
        print(f"  [snapshot] skipping — --checkpoint-prefix required for "
              f"strategy snapshots", flush=True)
        return

    iter_tag = (
        f"{iters_done // 1_000_000_000}B"
        if iters_done >= 1_000_000_000 and iters_done % 1_000_000_000 == 0
        else f"{iters_done // 1_000_000}M")

    snapshot_dir = os.path.join(args.output_dir, "strategy_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    bps_path = os.path.join(snapshot_dir, f"blueprint_{iter_tag}.bps")

    snapshot_meta = {
        "type": "phase_r_strategy_snapshot",
        "num_players": NUM_PLAYERS,
        "iterations": iters_done,
        "num_info_sets": n_is,
        "time_seconds": round(elapsed),
        "checkpoint": f"snapshot_{iter_tag}",
        "config_hash": _compute_config_hash(
            add_explicit_allin=(0 if args.no_explicit_allin else 1)),
        "add_explicit_allin": (0 if args.no_explicit_allin else 1),
        "depth_limit_cont_strats": int(args.depth_limit_cont_strats),
        "cont_strat_prob": args.cont_strat_prob,
        "cont_ev_bias_mode": args.cont_ev_bias_mode,
        "checkpoint_prefix": args.checkpoint_prefix,
        "snapshot_only": True,
        # Self-describing trainer-config block. Downstream consumers (the
        # search's blueprint loader, BPSReaderC's mixer selection) read
        # these off the .bps trailing JSON, so a snapshot needs no
        # post-hoc metadata patch. Pulled from the same module-level
        # constants the trainer itself uses.
        "hash_mixer": "splitmix64",
        "preflop_buckets": 169,        # hardcoded throughout trainer
        "postflop_buckets": 200,
        "preflop_bet_sizes": PREFLOP_BET_SIZES,
        "preflop_tiers": {str(k): v for k, v in sorted(PREFLOP_TIERS.items())},
        "preflop_max_raises": PREFLOP_MAX_RAISES,
        "postflop_bet_sizes": POSTFLOP_BET_SIZES,
        # SUBSEQUENT_BET_SIZES is fixed on the C side in
        # mccfr_blueprint.c (=[1.0] invariantly at every
        # num_raises>=1 postflop site, no opp-history conditional).
        # Stamp the literal so consumers can read it off metadata.
        "subsequent_bet_sizes": [1.0],
        "blinds": [SMALL_BLIND, BIG_BLIND],
        "initial_stack": INITIAL_STACK,
    }
    t_export = time.time()
    compressed_size = _stream_bps_export(bp_lib, solver, bps_path,
                                         snapshot_meta)
    if compressed_size <= 0:
        print(f"  [snapshot {iter_tag}] no strategies to export "
              f"(empty arena)", flush=True)
        return

    print(f"  [snapshot {iter_tag}] "
          f"{compressed_size / 1024 / 1024:.0f} MB compressed, "
          f"export {(time.time() - t_export):.0f}s, "
          f"path={bps_path}", flush=True)


def _export_checkpoint(bp_lib, solver, args, iters_done, elapsed, n_is,
                       label=None):
    """Export strategy + regret checkpoint into --output-dir.

    Saves two things:
    1. Strategy .bps file (for use by the runtime GPU solver)
    2. Regret table .bin file (for resuming training after interruption)

    Uses atomic writes (write to temp file, then rename) to prevent
    corrupted checkpoints if the process is killed mid-write.
    """
    import lzma
    import struct

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Save regret table (for resume) — atomic write ──
    regret_path = os.path.join(args.output_dir, "regrets_latest.bin")
    regret_tmp = regret_path + ".tmp"
    # Pre-flight disk space check. The save needs ~80 GB on a 1.5B-IS run;
    # at 5B target with 2.4B IS, peak is ~120 GB. Require 2x headroom
    # to leave room for the in-progress .tmp + the existing regrets_latest.bin
    # that won't be replaced until os.replace below. If the disk fills
    # (for example from an unrelated copy job), bp_save_regrets hits ENOSPC
    # and the snapshot comes out truncated.
    try:
        import shutil
        free_bytes = shutil.disk_usage(args.output_dir).free
        free_gb = free_bytes / 1024**3
        # Estimate current save size: ~50 bytes/entry × num info sets
        est_save_gb = (n_is * 50) / 1024**3
        needed_gb = est_save_gb * 2.0  # 2x for old + new file during atomic swap
        if free_gb < needed_gb:
            raise RuntimeError(
                f"Pre-save disk-space check failed: need ~{needed_gb:.1f} GB "
                f"free in {args.output_dir} (estimated save {est_save_gb:.1f} "
                f"GB × 2 for atomic-swap headroom), have {free_gb:.1f} GB. "
                f"Aborting save to prevent silent ENOSPC truncation. "
                f"Free disk space and retry on next checkpoint.")
    except RuntimeError:
        raise
    except Exception as e:
        print(f"  [WARN] disk-space pre-flight check failed: {e} — proceeding "
              f"anyway (C code now has fwrite checks as safety net)",
              flush=True)

    save_rc = bp_lib.bp_save_regrets(solver, regret_tmp.encode('utf-8'))
    if save_rc != 0:
        # The C code signals -2 on short fwrite (ENOSPC). The .tmp is
        # already deleted by the C code. Don't os.replace; retain prior
        # regrets_latest.bin so resume still has a valid recovery point.
        raise RuntimeError(
            f"bp_save_regrets returned {save_rc} — likely ENOSPC. "
            f"Prior regrets_latest.bin retained. "
            f"Free disk space ({shutil.disk_usage(args.output_dir).free/1024**3:.1f} "
            f"GB available) before next checkpoint.")
    os.replace(regret_tmp, regret_path)

    # Hash-table health stats (None if older C lib lacks bp_get_table_stats)
    ht_entries = ctypes.c_int64(0)
    ht_table_size = ctypes.c_int64(0)
    ht_failures = ctypes.c_int64(0)
    ht_max_probe = ctypes.c_int64(0)
    if hasattr(bp_lib, "bp_get_table_stats"):
        bp_lib.bp_get_table_stats(
            solver,
            ctypes.byref(ht_entries),
            ctypes.byref(ht_table_size),
            ctypes.byref(ht_failures),
            ctypes.byref(ht_max_probe),
        )

    # Save checkpoint metadata — atomic write
    meta = {
        "type": "unified_blueprint",
        "num_players": NUM_PLAYERS,
        "blinds": [SMALL_BLIND, BIG_BLIND],
        "initial_stack": INITIAL_STACK,
        "preflop_bet_sizes": PREFLOP_BET_SIZES,
        "preflop_tiers": {str(k): v for k, v in PREFLOP_TIERS.items()},
        "preflop_max_raises": PREFLOP_MAX_RAISES,
        "postflop_bet_sizes": POSTFLOP_BET_SIZES,
        "iterations": iters_done,
        "num_info_sets": n_is,
        "time_seconds": round(elapsed),
        "time_hours": round(elapsed / 3600, 1),
        "preflop_buckets": 169,
        "postflop_buckets": 200,
        "checkpoint": label or f"iter_{iters_done}",
        "config_hash": _compute_config_hash(
            add_explicit_allin=(0 if args.no_explicit_allin else 1)),
        "add_explicit_allin": (0 if args.no_explicit_allin else 1),
        "hash_mixer": "splitmix64",
        # Hash-table health, so silently dropped info sets show up in the
        # meta. load_factor > 0.85 or insertion_failures > 0 means the
        # table must grow (--hash-size) before any further training.
        "hash_table_size": ht_table_size.value,
        "hash_load_factor": (ht_entries.value / ht_table_size.value
                              if ht_table_size.value > 0 else 0.0),
        "hash_insertion_failures": ht_failures.value,
        "hash_max_probe_observed": ht_max_probe.value,
        "dcfr": {
            "alpha": os.environ.get("BPS_DCFR_ALPHA", ""),
            "beta": os.environ.get("BPS_DCFR_BETA", ""),
            "gamma": os.environ.get("BPS_DCFR_GAMMA", ""),
        },
    }
    meta_path = os.path.join(args.output_dir, "checkpoint_meta.json")
    meta_tmp = meta_path + ".tmp"
    with open(meta_tmp, 'w') as f:
        json.dump(meta, f)
    os.replace(meta_tmp, meta_path)

    # ── Save strategy .bps (for runtime use) ──
    # Only export .bps on final checkpoint — intermediate full-checkpoint
    # exports are too expensive (300M+ info sets × LZMA compress = minutes
    # per checkpoint). The regret checkpoint is sufficient for resume;
    # canonical .bps is only needed for the runtime solver after training
    # completes. Per-mini-chunk experimental snapshots are emitted via
    # `_export_strategy_snapshot`; those write under strategy_snapshots/,
    # never to the canonical `unified_blueprint.bps`.
    bps_path = os.path.join(args.output_dir, "unified_blueprint.bps")
    regret_mb = os.path.getsize(regret_path) / 1024 / 1024

    if label == "final":
        # OOM-safe streaming export. Materializing the entire raw strat
        # (~23 GB), a `bytes()` copy of it and the full LZMA output in
        # Python memory at once, with the regret hash table still alive
        # plus OS write buffers, can exceed 384 GB and get the process
        # OOM-killed mid-write, truncating the output. See
        # `_stream_bps_export` for the streaming pattern.
        compressed_size = _stream_bps_export(bp_lib, solver, bps_path, meta)
        if compressed_size > 0:
            comp_mb = compressed_size / 1024 / 1024
            print(f"  Final checkpoint: strategy={comp_mb:.0f}MB compressed, "
                  f"regrets={regret_mb:.0f}MB", flush=True)
        else:
            print("  [WARN] No strategies to export")
    else:
        print(f"  Checkpoint: regrets={regret_mb:.0f}MB (strategy export deferred to final)")


if __name__ == "__main__":
    main()
