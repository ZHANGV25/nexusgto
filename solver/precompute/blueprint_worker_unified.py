#!/usr/bin/env python3
"""Unified blueprint worker: 6-player preflop-through-river MCCFR.

Matches Pluribus exactly: ONE unified solve over the entire game tree.
All 6 players, all 1326 hands, 169 lossless preflop buckets, 200 k-means
postflop buckets. Preflop → flop → turn → river → showdown in a single MCCFR.

This is the Pluribus approach (Brown & Sandholm, Science 2019):
  - 64 cores, 8 days, 12,400 CPU core hours, <512GB RAM
  - ~665M action sequences, ~414M encountered
  - Our target: 64-core c5.18xlarge ($3.06/hr) for 8 days ≈ $587

Usage (local test):
    python blueprint_worker_unified.py --iterations 10000 --num-threads 1

Usage (EC2, full Pluribus-scale):
    python blueprint_worker_unified.py --iterations 0 --time-limit-hours 192 \
        --num-threads 64 --hash-size 536870912 \

The solver runs until --time-limit-hours is reached (Pluribus ran for 8 days
= 192 hours). Iteration count is not pre-specified — training runs continuously
and convergence is measured by exploitability and strategy stability.
"""

import argparse
import ctypes
import json
import os
import signal
import subprocess
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
PREFLOP_BET_SIZES = [0.5, 0.7, 1.0]
# Postflop first raise: Pluribus turn/river = {0.5x, 1x, all-in}
# (all-in added automatically by generate_actions; subsequent raises use {1x, all-in})
POSTFLOP_BET_SIZES = [0.5, 1.0]

# Tiered preflop sizing (Pluribus uses up to 14 sizes per decision; we ship a
# coarser tier but cover the bands humans actually open/3-bet/4-bet at).
#
# Enumerated tree (6p, 100BB): 380K preflop decision nodes × 169 buckets
# = 64M preflop IS, 381K flop lines. Est ~350M total IS, ~220M encountered
# with -300M pruning. 750M hash at 44% load = 42 GB metadata.
# Hand-picked from 2B checkpoint regret analysis: r0.7x had highest
# positive rate (29%), all 3 open sizes within 25-29%. 5-bet shove
# dropped (0-6% positive = dead weight).
PREFLOP_TIERS = {
    0: [0.5, 0.7, 1.0],              # open raise: 3 sizes (all viable per regret data)
    1: [0.7, 1.0],                   # 3-bet: 2 sizes
    2: [1.0],                        # 4-bet: 1 size (all-in auto-added)
}
PREFLOP_MAX_RAISES = 3

# ── Canonical 169-class preflop bucket map (matches mccfr_blueprint.c) ──
# Built programmatically so probes display the right hand under each
# bucket index. Probe code uses this map; hardcoded indices have been
# wrong twice in the past (Bug A in EXTRACTOR_BUGS.md). Single source.
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
# Prior incident: a new tiered-sizing run loaded an OLD flat-8 checkpoint
# and produced garbage for hours before detection. Embedding a SHA256 of
# the config-relevant fields in the checkpoint meta and refusing to load
# on mismatch prevents the entire failure class.

def _compute_config_hash(add_explicit_allin=None):
    """Hash of everything that shapes the regret table.
    A mismatch between current config and checkpoint meta means the
    regret table's action_hash/bucket layout is incompatible and the
    checkpoint must be rejected.

    Phase 3: add_explicit_allin is part of the layout — gate=0 produces
    one fewer slot per deep-stack infoset. Pass None to omit (legacy
    behavior, equivalent to add_explicit_allin=1).
    """
    import hashlib
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
            bp.bp_export_strategies.restype = ctypes.c_int
            bp.bp_export_strategies.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
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
            # Phase 3 hooks (no-op if older C lib doesn't have them).
            if hasattr(bp, "bp_set_add_explicit_allin"):
                bp.bp_set_add_explicit_allin.restype = ctypes.c_int
                bp.bp_set_add_explicit_allin.argtypes = [
                    ctypes.c_void_p, ctypes.c_int,
                ]
            if hasattr(bp, "bp_drop_trailing_allin_slot"):
                bp.bp_drop_trailing_allin_slot.restype = ctypes.c_int64
                bp.bp_drop_trailing_allin_slot.argtypes = [ctypes.c_void_p]
            bp.bp_save_texture_cache.restype = ctypes.c_int
            bp.bp_save_texture_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            bp.bp_load_texture_cache.restype = ctypes.c_int
            bp.bp_load_texture_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            bp.bp_free.restype = None
            # Hash table health stats (added for v2)
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
    parser.add_argument("--output-dir", default="/opt/blueprint_unified")
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--checkpoint-interval", type=int, default=1000000,
                        help="Export checkpoint every N iterations")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest regret checkpoint on S3")
    parser.add_argument("--resume-from", default="",
                        help="S3 key (relative to bucket) of the regret "
                             "file to resume from. Default: "
                             "checkpoints/regrets_latest.bin. Used by "
                             "Phase 3 warm-restart launcher to point at "
                             "regrets_3400M.bin instead.")
    parser.add_argument("--no-explicit-allin", action="store_true",
                        default=(os.environ.get("BPS_NO_EXPLICIT_ALLIN", "0") == "1"),
                        help="Phase 3: gate the implicit-allin block in "
                             "generate_actions(). Default off. When on, "
                             "deep-stack nodes have no allin in action set "
                             "(short-stack clamp still applies). When "
                             "combined with --resume from a legacy "
                             "checkpoint, runs the post-load fixup that "
                             "drops the trailing allin slot from preflop/"
                             "postflop infosets saved at max-with-allin "
                             "size. See docs/PHASE3_RETRAIN_STATUS.md.")
    parser.add_argument("--river-ochs", action="store_true",
                        default=(os.environ.get("BPS_RIVER_OCHS", "0") == "1"),
                        help="Enable OCHS river abstraction (Johanson 2013, "
                             "Pluribus-faithful). Defaults to env BPS_RIVER_OCHS.")
    parser.add_argument("--river-ochs-clusters", type=int,
                        default=int(os.environ.get("BPS_RIVER_OCHS_CLUSTERS", "0") or 0),
                        help="Number of opponent clusters for OCHS (0=default 8).")
    args = parser.parse_args()

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
        config.hash_table_size = 3000000000  # 3B slots (~180GB meta, fits c7a.metal 376GB)

    # If running by time limit, estimate iterations based on Pluribus core-hours.
    # Pluribus: 12,400 core-hours on 64 cores. We match total core-hours.
    # Burst throughput is ~280K iter/s (pruned) / ~140K (unpruned), average ~200K.
    # With large checkpoint intervals, overhead is negligible.
    if args.time_limit_hours > 0 and args.iterations == 0:
        effective_threads = args.num_threads if args.num_threads > 0 else (os.cpu_count() or 1)
        # Target: Pluribus's 12,400 core-hours scaled to our core count
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

    # Try to load cached texture buckets from S3 (saves ~65 min precompute)
    texture_cache_local = os.path.join(args.output_dir, "texture_cache.bin")
    texture_loaded = False
    if args.s3_bucket:
        os.makedirs(args.output_dir, exist_ok=True)
        ret = subprocess.run(
            ["aws", "s3", "cp", texture_s3, texture_cache_local, "--quiet"],
            capture_output=True
        )
        if ret.returncode == 0 and os.path.exists(texture_cache_local):
            print(f"Found texture cache on S3, will load after init")
            texture_loaded = True
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

    # Phase 3: gate the implicit-allin block in generate_actions().
    # See docs/PHASE3_RETRAIN_STATUS.md.
    if args.no_explicit_allin:
        if not hasattr(bp_lib, "bp_set_add_explicit_allin"):
            print("FATAL: --no-explicit-allin requested but the loaded "
                  "mccfr_blueprint.so doesn't export "
                  "bp_set_add_explicit_allin. Rebuild with the Phase 3 "
                  "patch.")
            sys.exit(1)
        bp_lib.bp_set_add_explicit_allin(solver, 0)

    # Set tiered preflop sizing (overrides flat sizes)
    for level, sizes in sorted(PREFLOP_TIERS.items()):
        c_sizes = (ctypes.c_float * len(sizes))(*sizes)
        ret = bp_lib.bp_set_preflop_tier(
            solver, level, c_sizes, len(sizes), PREFLOP_MAX_RAISES)
        if ret != 0:
            print(f"FATAL: bp_set_preflop_tier level {level} returned {ret}")
            sys.exit(1)
    print(f"Tiered preflop: {len(PREFLOP_TIERS)} levels, max {PREFLOP_MAX_RAISES} raises")

    # Save texture cache for future runs (skips 65-min precompute next time)
    if not texture_loaded and args.s3_bucket:
        os.makedirs(args.output_dir, exist_ok=True)
        bp_lib.bp_save_texture_cache(solver, texture_cache_local.encode('utf-8'))
        try:
            subprocess.run(
                ["aws", "s3", "cp", texture_cache_local,
                check=True, capture_output=True
            )
            print("Texture cache saved to S3")
        except subprocess.CalledProcessError:
            print("Warning: failed to upload texture cache to S3")
    # Postflop buckets (200 EHS percentile) are precomputed inside bp_init_unified
    # for all 1,755 flop textures. During traversal, the dealt flop is canonicalized
    # and the precomputed bucket mapping is looked up in O(1755) per deal.

    # Resume from checkpoint if requested
    iters_already_done = 0
    if args.resume and args.s3_bucket:
        resume_key = args.resume_from or "checkpoints/regrets_latest.bin"
        regret_local = os.path.join(
            args.output_dir, os.path.basename(resume_key))
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Checking for checkpoint at {regret_s3}...")
        ret = subprocess.run(
            ["aws", "s3", "cp", regret_s3, regret_local, "--quiet"],
            capture_output=True
        )
        if ret.returncode == 0 and os.path.exists(regret_local):
            n_loaded = bp_lib.bp_load_regrets(
                solver, regret_local.encode('utf-8'))
            if n_loaded > 0 and args.no_explicit_allin:
                # Phase 3 lazy-shrink: when --no-explicit-allin is set, the
                # next find_or_create call for each infoset will shrink the
                # slot's num_actions to match the new (smaller) action set.
                # No explicit fixup needed; lazy shrink handles preflop
                # tiers + postflop subsequent-raise cases uniformly.
                # See mccfr_blueprint.c::info_table_find_or_create.
                print("[Phase 3] --no-explicit-allin active. Slots will "
                      "lazily shrink on first visit.")
            if n_loaded > 0:
                # Read back iterations_run from the solver struct
                # (bp_load_regrets sets s->iterations_run)
                # We can't easily read it from Python, but the C code printed it.
                # Parse it from the checkpoint metadata instead.
                try:
                    meta_local = os.path.join(args.output_dir, "checkpoint_meta.json")
                    subprocess.run(
                        ["aws", "s3", "cp", meta_s3, meta_local, "--quiet"],
                        capture_output=True
                    )
                    if os.path.exists(meta_local):
                        import json as _json
                        with open(meta_local) as mf:
                            cmeta = _json.load(mf)
                        iters_already_done = cmeta.get("iterations", 0)
                        # Stale-checkpoint guard: refuse to continue if the
                        # checkpoint's config hash doesn't match the current
                        # config. Prevents the prior-incident class where a
                        # new tiered run loaded a flat-8 checkpoint.
                        ckpt_hash = cmeta.get("config_hash")
                        # Phase 3: the config hash now optionally includes
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
                            print(f"[Phase 3] Resuming legacy (allin=1) "
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
                except Exception:
                    pass
                print(f"Resumed from checkpoint: {n_loaded} info sets, "
                      f"{iters_already_done:,} iterations already done")
            os.remove(regret_local)
        else:
            print("No checkpoint found, starting fresh.")

    # Solve in chunks with periodic checkpoints.
    # Each checkpoint exports the full strategy to .bps and uploads to S3.
    # If the instance dies, we lose at most one chunk of work.
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
    # On spot reclaim AWS sends SIGTERM with 2-min warning — enough time
    # to flush regrets to disk and upload to S3, saving up to 48 min of
    # work that would otherwise be lost.
    _sig_state = {"iters_done": iters_done, "solver": solver}

    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        if signum == signal.SIGTERM:
            print(f"\n[SIGTERM] Spot reclaim or shutdown — saving emergency checkpoint...",
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

    # Checkpoint cadence: every 100M iters. At ~110K iter/s on c6a.metal,
    # 100M = ~15min wall-clock. Spot reclaim loses at most ~15min of work.
    # Each checkpoint = 76 GB regret file + ~50 KB meta. Save+upload takes
    # ~3 min on gp3 1GB/s throughput. Overhead: ~20% of wall time.
    #
    # Versioned snapshots: every checkpoint also creates a versioned
    # `regrets_NNNM.bin` copy in S3 so a truncated upload of regrets_latest
    # never destroys a known-good prior checkpoint. Lifecycle policy on
    # the bucket prunes old versioned snapshots after retention window.
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
    # Full checkpoint (regret save + S3 upload) only at adaptive milestones.
    mini_chunk = 50_000_000  # strategy probe every 50M iterations

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

        # Hash table health stats. New API as of v2; gated on availability so
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

        # Lightweight strategy probe: extract strategies + raw regrets
        # directly from memory via bp_get_strategy/bp_get_regrets (no disk I/O).
        probe = _probe_preflop(bp_lib, solver)
        if probe:
            header = f"[Probe {iters_done:,}]"
            for line in probe.split("\n"):
                print(f"{header} {line}", flush=True)
            # Upload probe file to S3
            if args.s3_bucket and args.output_dir:
                iter_tag = f"{iters_done // 1_000_000}M"
                probe_path = os.path.join(args.output_dir, "probe_latest.txt")
                with open(probe_path, 'w') as pf:
                    pf.write(f"iteration: {iters_done}\n{probe}\n")
                try:
                    subprocess.run(
                        ["aws", "s3", "cp", probe_path,
                         "--quiet"], capture_output=True, timeout=30)
                    subprocess.run(
                        ["aws", "s3", "cp", probe_path,
                         "--quiet"], capture_output=True, timeout=30)
                except Exception:
                    pass

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

    # ── Auto-chain EV walk ──────────────────────────────────────────
    # The .bps is on local disk. Load it into a BPSReaderC and run the
    # 500M EV walk in-process. On c7a.metal-48xl (192 vCPU), this takes
    # ~15 min. The EV walk populates per-action EVs needed by the runtime
    # solver's leaf-value computation (rollout.c).
    if args.output_dir and os.environ.get("BPS_AUTO_EV_WALK", "1") == "1":
        bps_path = os.path.join(args.output_dir, "unified_blueprint.bps")
        if os.path.exists(bps_path):
            print("\n" + "=" * 60)
            print("  AUTO EV WALK — 500M iterations")
            print("=" * 60)
            try:
                sys.path.insert(0, os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "python"))
                from bps_reader_py import BPSReaderC

                ev_t0 = time.time()
                print(f"Loading final blueprint from {bps_path}...", flush=True)
                reader = BPSReaderC(bps_path)
                print(f"Loaded in {time.time() - ev_t0:.1f}s", flush=True)

                ev_iters = int(os.environ.get("BPS_EV_WALK_ITERS", "500000000"))
                ev_threads = max(args.num_threads, 1)
                ev_checkpoint_interval = 50_000_000  # 50M
                ev_cache_path = bps_path + ".cache"

                print(f"Starting {ev_iters:,} EV walk iterations on "
                      f"{ev_threads} threads (checkpoint every "
                      f"{ev_checkpoint_interval:,})...", flush=True)

                reader.compute_evs(
                    num_iterations=ev_iters,
                    num_threads=ev_threads,
                    num_players=NUM_PLAYERS,
                    small_blind=SMALL_BLIND,
                    big_blind=BIG_BLIND,
                    initial_stack=INITIAL_STACK,
                    checkpoint_interval=ev_checkpoint_interval,
                    checkpoint_path=ev_cache_path,
                )
                ev_elapsed = time.time() - ev_t0
                print(f"EV walk complete in {ev_elapsed:.0f}s "
                      f"({ev_elapsed / 60:.1f} min)", flush=True)

                # Save the final cache with EVs
                print(f"Saving EV cache to {ev_cache_path}...", flush=True)
                reader.save_cache(ev_cache_path)

                # Upload .bps.cache to S3 (.bps already uploaded by _export_checkpoint)
                s3_bucket = os.environ.get("S3_BUCKET")
                if s3_bucket:
                    fname = os.path.basename(ev_cache_path)
                    print(f"Uploading {fname} to {s3_dest}...", flush=True)
                    subprocess.run(
                        ["aws", "s3", "cp", ev_cache_path, s3_dest],
                        check=True
                    )
                print("EV walk + upload done.")
            except Exception as e:
                print(f"[WARN] Auto EV walk failed: {e}", flush=True)
                print("Manual EV walk will be needed.", flush=True)
        else:
            print(f"[WARN] No .bps at {bps_path}, skipping auto EV walk")

    print("\nAll done.")


def _probe_preflop(bp_lib, solver):
    """Extract richer preflop convergence diagnostics.

    Three sections:

      OPEN — open-raise strategy at every position (UTG, MP, CO, BTN, SB, BB).
        Per-raise-size frequencies (not aggregated R%) so we can see if
        r0.5x / r0.7x / r1.0x are converging differentially. UTG = no
        prior action; MP = UTG folded; CO = UTG+MP folded; BTN = UTG+MP+CO
        folded; SB = UTG+MP+CO+BTN folded; BB = UTG+MP+CO+BTN+SB called
        (BB defends). Format:
          OPEN UTG AA: F0 C5 R0.5x:30 R0.7x:35 R1.0x:30
          OPEN BTN 22: F60 C20 R0.5x:5 R0.7x:5 R1.0x:10

      VS_OPEN — defending strategy when villain (UTG) opens to 3bb.
        BB and BTN as defenders. Lets us see fold/call/3bet ranges.
        Format:
          VS_OPEN BB AA: F2 C20 R3bet=78
          VS_OPEN BTN AKs: F0 C40 R3bet=60

      VS_3BET — 4bet response when villain (BB) 3bets us (UTG opener).
        Detects whether 5-bet jam line is live. Format:
          VS_3BET UTG AA: F2 C5 R4bet=93
          VS_3BET UTG TT: F30 C50 R4bet=20

      REGRETS — raw int regrets for diagnostic hands at all 6 positions.
        Per-raise-size regrets so we can see which sizes are saturated
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
        # We detect na and label by index dynamically.
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
                # We don't know labels from the C side; use heuristic:
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
                # Show all action regrets so we can see fold vs call vs each
                # raise size individually (per-size visibility was the gap
                # in the prior probe).
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


def _s3_head_size(s3_uri):
    """Return ContentLength of an S3 object, or None if not found / error.

    Uses `aws s3api head-object` because `aws s3 ls` returns sizes only as
    a count of bytes via parsing — head-object returns a structured JSON
    that's safer to parse and includes an ETag we can also cross-check.
    """
        return None
    rest = s3_uri[5:]
    if "/" not in rest:
        return None
    bucket, key = rest.split("/", 1)
    try:
        out = subprocess.run(
            ["aws", "s3api", "head-object",
             "--bucket", bucket, "--key", key,
             "--output", "json"],
            check=True, capture_output=True, timeout=30,
        )
        meta = json.loads(out.stdout.decode())
        return int(meta.get("ContentLength", 0))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            ValueError, KeyError):
        return None


def _s3_upload_with_retry(local_path, s3_uri, max_attempts=3, delay=5,
                          verify=True):
    """Upload a file to S3 with retry + size-verification on each attempt.

    The 3B run on 2026-04-27 silently truncated a 76 GB regret upload to
    57 GB on a multipart cancellation. The aws CLI returned 0 (or 1, the
    log line was suppressed by `--quiet`), and the S3 head_object size
    came back smaller than the local file — but nothing checked that.

    This function:
      1. Uploads via `aws s3 cp`
      2. Calls `head-object` to confirm bytes-on-S3 == bytes-on-disk
      3. Treats a size mismatch as a failed attempt (retries)

    Set verify=False ONLY for files where mid-stream interruption is
    survivable (e.g. periodic log uploads). For checkpoint regrets and
    .bps files, ALWAYS verify.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)
    expected_size = os.path.getsize(local_path)

    for attempt in range(1, max_attempts + 1):
        try:
            # Note: `--expected-size` is for stdin streams only, doesn't
            # apply here. The size check via head-object after upload
            # is the actual integrity guarantee.
            subprocess.run(
                ["aws", "s3", "cp", local_path, s3_uri, "--quiet"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            if attempt < max_attempts:
                stderr = e.stderr.decode().strip() if e.stderr else ""
                print(f"  [WARN] S3 upload failed "
                      f"(attempt {attempt}/{max_attempts}): {stderr}, "
                      f"retrying in {delay}s...", flush=True)
                time.sleep(delay)
                continue
            stderr = e.stderr.decode().strip() if e.stderr else ""
            print(f"  [ERROR] S3 upload failed after {max_attempts} "
                  f"attempts: {local_path} -> {s3_uri}: {stderr}",
                  flush=True)
            raise

        if not verify:
            return  # success, no integrity check

        actual_size = _s3_head_size(s3_uri)
        if actual_size == expected_size:
            return  # success and verified

        # Size mismatch — uploaded data is corrupted/truncated.
        if attempt < max_attempts:
            print(f"  [WARN] S3 size mismatch "
                  f"(attempt {attempt}/{max_attempts}): "
                  f"local={expected_size}, s3={actual_size} "
                  f"({s3_uri}), retrying...", flush=True)
            time.sleep(delay)
        else:
            print(f"  [ERROR] S3 upload size mismatch after "
                  f"{max_attempts} attempts: local={expected_size}, "
                  f"s3={actual_size} ({s3_uri})", flush=True)
            raise RuntimeError(
                f"S3 upload integrity check failed: "
                f"{s3_uri} expected {expected_size} bytes, "
                f"got {actual_size}"
            )


def _export_checkpoint(bp_lib, solver, args, iters_done, elapsed, n_is,
                       label=None):
    """Export strategy + regret checkpoint and upload to S3.

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
    # at 5B target with 2.4B IS, peak is ~120 GB. Require 1.5x headroom
    # to leave room for the in-progress .tmp + the existing regrets_latest.bin
    # that won't be replaced until os.replace below. 2026-04-29 incident:
    # /opt filled silently from accidental aws s3 sync of blueprint_data_v3,
    # bp_save_regrets hit ENOSPC and produced a truncated 3.3B snapshot.
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
        # The C code now signals -2 on short fwrite (ENOSPC). The .tmp is
        # already deleted by the C code. Don't os.replace; retain prior
        # regrets_latest.bin so resume still has a valid recovery point.
        raise RuntimeError(
            f"bp_save_regrets returned {save_rc} — likely ENOSPC. "
            f"Prior regrets_latest.bin retained. "
            f"Free disk space ({shutil.disk_usage(args.output_dir).free/1024**3:.1f} "
            f"GB available) before next checkpoint.")
    os.replace(regret_tmp, regret_path)

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
    # Only export .bps on final checkpoint — intermediate exports are too
    # expensive (300M+ info sets × LZMA compress = minutes per checkpoint).
    # The regret checkpoint is sufficient for resume; .bps is only needed
    # for the runtime solver after training completes.
    bps_path = os.path.join(args.output_dir, "unified_blueprint.bps")
    regret_mb = os.path.getsize(regret_path) / 1024 / 1024

    if label == "final":
        # OOM-safe streaming export.
        #
        # Prior pattern materialized the entire raw strat (~23 GB) AND a
        # `bytes()` copy of it AND the full LZMA output in Python memory
        # simultaneously. With the regret hash table still alive (~75 GB
        # table arrays + ~25 GB arena) plus OS write buffers, peak crossed
        # the c6a.metal 384 GB ceiling under load and got OOM-killed mid
        # multipart upload — the truncation that cost the 3B run.
        #
        # New pattern:
        #   1. Single `bp_export_strategies` call into one C-side buffer
        #      (Python sees it via ctypes; no `bytes()` copy).
        #   2. Compress the buffer slice-by-slice with `LZMACompressor`,
        #      streaming each chunk straight into the output file. Only
        #      a small (~1 MB) compressor working set lives in RAM.
        #   3. Outer .bps wrapper is written with a placeholder header,
        #      then we patch in the real compressed_size after streaming.
        needed = ctypes.c_size_t(0)
        bp_lib.bp_export_strategies(solver, None, 0, ctypes.byref(needed))
        strat_size = needed.value

        if strat_size > 0:
            print(f"  Streaming strat export: {strat_size / 1024 / 1024 / 1024:.2f} GiB raw",
                  flush=True)
            strat_buf = (ctypes.c_char * strat_size)()
            written = ctypes.c_size_t(0)
            bp_lib.bp_export_strategies(solver, strat_buf, strat_size,
                                        ctypes.byref(written))
            written_bytes = written.value
            meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')

            # Stream LZMA directly to file. Use `memoryview` over strat_buf
            # so we don't allocate a Python `bytes` copy of the whole buffer.
            CHUNK = 64 * 1024 * 1024  # 64 MB compress chunks
            out_tmp = bps_path + ".tmp"

            # Strategy: write header with placeholder compressed_size = 0,
            # stream-compress to disk past the header, fseek back and patch
            # the header. Atomically rename at the end.
            compressor = lzma.LZMACompressor(preset=1)
            compressed_size = 0

            with open(out_tmp, 'wb') as f:
                # Header placeholder. compressed_size=0 will be patched.
                f.write(b'BPS3')
                header_compressed_size_offset = f.tell()
                f.write(struct.pack('<QI', 0, len(meta_bytes)))

                # Stream raw strat_buf in 64 MB chunks through the compressor.
                mv = memoryview(strat_buf)[:written_bytes]
                pos = 0
                last_pct_logged = -1
                while pos < written_bytes:
                    chunk_end = min(pos + CHUNK, written_bytes)
                    # memoryview slice avoids the per-chunk bytes() copy
                    # (lzma.compress accepts buffer-protocol objects directly).
                    out_chunk = compressor.compress(mv[pos:chunk_end])
                    if out_chunk:
                        f.write(out_chunk)
                        compressed_size += len(out_chunk)
                    pos = chunk_end
                    pct = int(100 * pos / written_bytes)
                    if pct >= last_pct_logged + 10:
                        print(f"    compress {pct}% "
                              f"(in={pos / 1024 / 1024 / 1024:.1f} GiB, "
                              f"out={compressed_size / 1024 / 1024 / 1024:.2f} GiB)",
                              flush=True)
                        last_pct_logged = pct
                tail = compressor.flush()
                if tail:
                    f.write(tail)
                    compressed_size += len(tail)

                # Now write meta footer.
                f.write(meta_bytes)

                # Patch the compressed_size field in the header.
                f.seek(header_compressed_size_offset)
                f.write(struct.pack('<Q', compressed_size))

            os.replace(out_tmp, bps_path)

            comp_mb = compressed_size / 1024 / 1024
            print(f"  Final checkpoint: strategy={comp_mb:.0f}MB compressed, "
                  f"regrets={regret_mb:.0f}MB", flush=True)
        else:
            print("  [WARN] No strategies to export")
    else:
        print(f"  Checkpoint: regrets={regret_mb:.0f}MB (strategy export deferred to final)")

    # ── Upload to S3 (with retries) ──
    if args.s3_bucket:
        # Always upload latest (for resume)
        uploads = [
            (regret_path, "checkpoints/regrets_latest.bin"),
            (meta_path, "checkpoint_meta.json"),
        ]
        # Also save a timestamped copy of every checkpoint for later analysis.
        # Use "XB" for exact billions, "XM" otherwise (avoids collisions
        # where e.g. 1.5B and 1B both mapped to "1B" via integer division).
        if iters_done >= 1_000_000_000 and iters_done % 1_000_000_000 == 0:
            iter_tag = f"{iters_done // 1_000_000_000}B"
        else:
            iter_tag = f"{iters_done // 1_000_000}M"
        uploads.append((regret_path, f"checkpoints/regrets_{iter_tag}.bin"))
        uploads.append((meta_path, f"checkpoints/meta_{iter_tag}.json"))

        if label == "final" and os.path.exists(bps_path):
            uploads.append((bps_path, "unified_blueprint.bps"))

        # ── Run extract_roots for convergence monitoring ──
        extract_bin = os.path.join(args.build_dir, "extract_roots")
        summary_path = None
        if os.path.exists(extract_bin):
            summary_path = os.path.join(args.output_dir, f"strategy_{iter_tag}.txt")
            try:
                result = subprocess.run(
                    [extract_bin, regret_path],
                    capture_output=True, text=True, timeout=1800
                )
                with open(summary_path, 'w') as sf:
                    sf.write(result.stdout)
                # Print a compact UTG summary to the log
                for line in result.stdout.splitlines():
                    if "Summary:" in line or "SANITY" in line or "CHECK:" in line or "OK:" in line:
                        print(f"  {line.strip()}")
            except Exception as e:
                print(f"  [WARN] extract_roots failed: {e}")
                summary_path = None

        # Upload small files FIRST (summary + meta) so results are visible
        # immediately, then upload the large regret file once + server-side copy.
        small_uploads = [(meta_path, "checkpoint_meta.json"),
                         (meta_path, f"checkpoints/meta_{iter_tag}.json")]
        if summary_path and os.path.exists(summary_path):
            small_uploads.append((summary_path, f"summaries/strategy_{iter_tag}.txt"))
        if label == "final" and os.path.exists(bps_path):
            small_uploads.append((bps_path, "unified_blueprint.bps"))

        for local, s3key in small_uploads:
            if os.path.exists(local):
                try:
                    _s3_upload_with_retry(
                except subprocess.CalledProcessError:
                    pass

        # Upload regret file with integrity verification. Two-step pattern:
        #   1. Upload to checkpoints/regrets_NNNM.bin (versioned snapshot)
        #      VERIFIED first — this is the durable artifact.
        #   2. Server-side copy to regrets_latest.bin (resume entry point).
        #
        # Why versioned-first: if step 1 truncates, regrets_latest stays as
        # the previous good copy. The 3B-run failure mode (truncated upload
        # overwrote regrets_latest, destroying the prior good 2.75B copy)
        # is impossible under this order. Lifecycle policy on the bucket
        # prunes old versioned snapshots after retention window.
        versioned_s3 = (
        )
        # Soft-fail policy: log loud on upload errors, but don't raise.
        # A transient S3 hiccup shouldn't kill a 5-day run. The next
        # checkpoint (in 100M iters / ~15 min) will retry with fresher
        # regrets, and any prior versioned snapshot is still safe on S3.
        # If something is structurally broken (creds, bucket policy),
        # multiple consecutive failures will surface in the log.
        try:
            _s3_upload_with_retry(regret_path, versioned_s3, verify=True)
            print(f"  Regrets versioned snapshot uploaded ({iter_tag}) "
                  f"— size verified", flush=True)
            # Update regrets_latest by uploading the local file a second
            # time. We previously used `aws s3 cp` for s3-to-s3 copy, but
            # AWS CLI requires --expected-size for objects > 5 GB and
            # silently fails the simple form. Re-uploading the local file
            # uses the same multipart machinery (auto-triggered at 64 MB
            # via the AWS_CLI configure call earlier) and has identical
            # cost since the egress is from the same EC2-S3 region pair
            # (free intra-region bandwidth). The file is already on disk
            # from the versioned upload moments earlier — no extra I/O.
            _s3_upload_with_retry(regret_path, latest_s3, verify=True)
            ls = _s3_head_size(latest_s3)
            local_size = os.path.getsize(regret_path)
            if ls != local_size:
                print(f"  [ERROR] regrets_latest size mismatch: "
                      f"latest={ls}, local={local_size}. Continuing — "
                      f"versioned snapshot is intact, next checkpoint "
                      f"will retry.", flush=True)
            else:
                print(f"  regrets_latest updated ({iter_tag}, "
                      f"{ls / 1024 / 1024 / 1024:.1f} GiB verified)",
                      flush=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                RuntimeError) as e:
            print(f"  [ERROR] regret upload/copy failed: {e}. "
                  f"Continuing — next checkpoint will retry.", flush=True)


if __name__ == "__main__":
    main()
