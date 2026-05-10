/**
 * mccfr_blueprint.h — Production N-player external-sampling MCCFR
 *
 * Matches the exact Pluribus blueprint training algorithm:
 *   - N-player (2-6) external-sampling MCCFR
 *   - Multi-street: flop -> turn -> river -> showdown
 *   - Linear CFR discount: every DISCOUNT_INTERVAL iters for first DISCOUNT_PERIOD iters
 *   - Regret-based pruning: skip actions with regret < PRUNE_THRESHOLD (95% of iters)
 *   - Integer regrets (int32) with floor at REGRET_FLOOR
 *   - Card abstraction: hand_to_bucket mapping (200 buckets per street)
 *   - OpenMP parallel iterations with thread-local RNG (Hogwild-style)
 *   - Strategy snapshots to disk (not continuous accumulation for rounds 2-4)
 *
 * Pluribus parameters (from supplementary):
 *   - 64 cores, 8 days, 12400 CPU-hours, <512GB RAM
 *   - Discount: d = (T/interval)/(T/interval+1) every 10 min for first 400 min
 *   - Pruning: after 200 min, threshold -300M, 95% of iterations
 *   - Regret floor: -310M
 *   - 665M action sequences, 414M encountered
 *   - Regrets as int32, lazy allocation
 */
#ifndef MCCFR_BLUEPRINT_H
#define MCCFR_BLUEPRINT_H

#include <stdint.h>
#include <stddef.h>
#include "mccfr_cont_ev_table.h"

/* ── Limits ──────────────────────────────────────────────────────────── */

#define BP_MAX_PLAYERS    6
#define BP_MAX_HANDS      1326   /* max hands per player (all possible 2-card combos) */
#define BP_MAX_ACTIONS    16
#define BP_MAX_BOARD      5

/* Action-type tags. Mirrors the static ACT_* defines inside
 * mccfr_blueprint.c — promoted to the header so Phase R helpers in
 * mccfr_phase_r.c can classify actions for cont-bias reweighting. */
#define BP_ACT_FOLD     0
#define BP_ACT_CHECK    1
#define BP_ACT_CALL     2
#define BP_ACT_BET      3

/* BPAction — single emitted action from the trainer's
 * generate_actions(). Promoted to the header in Phase R Block B.3
 * (Step 6) so the encoding-alignment test can drive a pure-C
 * trainer-walker against the same emission contract the production
 * traverse() uses. */
typedef struct {
    int type;     /* BP_ACT_FOLD/CHECK/CALL/BET */
    int bet_idx;  /* -1 for non-bet; index into bet_sizes[] for bet/raise */
    int amount;   /* total chips committed by this action (call=to_call, raise=to_call+raise) */
} BPAction;

#define BP_MAX_GEN_ACTIONS 16

/* Hash table sizing.
 * Per-texture: ~5-50M info sets depending on player count and bet sizes.
 * BP_HASH_SIZE_MEDIUM (64M slots) fits ~45M info sets at 70% load.
 * Memory: 64M * (key=20 + set=16 + occupied=4) = ~2.5GB metadata.
 * For small test runs, use BP_HASH_SIZE_SMALL. */
#define BP_HASH_SIZE_3B     ((int64_t)3000000000) /* 3B slots (~180GB metadata) — full unified solve */
#define BP_HASH_SIZE_LARGE  (1 << 30)   /* 1.07B slots (~44GB metadata) — bucket-in-key needs more */
#define BP_HASH_SIZE_MEDIUM (1 << 26)   /* 64M slots (~2.5GB metadata) — per-texture */
#define BP_HASH_SIZE_SMALL  (1 << 22)   /* 4M slots (~160MB metadata) — testing only */

/* Pluribus-matched constants */
#define BP_REGRET_FLOOR      (-310000000)   /* minimum regret per action */
#define BP_REGRET_CEILING     2000000000    /* ~int32 max. Pluribus has no explicit ceiling (only
                                            * the implicit int32 max). The old 310M ceiling caused
                                            * dominant actions to saturate 7x too early, losing
                                            * ordering information when multiple actions competed.
                                            * int64 intermediate arithmetic prevents overflow. */
#define BP_PRUNE_THRESHOLD   (-300000000)   /* Pluribus standard: Linear CFR regrets hit this naturally */
#define BP_PRUNE_PROB        0.95f          /* fraction of iters that prune */

/* ── Data structures ─────────────────────────────────────────────────── */

/* Info set key — includes bucket (Pluribus-style: one info set per bucket).
 * This means each hash table slot stores regrets[num_actions] only,
 * reducing memory ~200x vs storing regrets[num_actions * num_buckets]. */
typedef struct {
    int player;
    int street;              /* 0=preflop, 1=flop, 2=turn, 3=river */
    int bucket;              /* card abstraction bucket index */
    uint64_t board_hash;
    uint64_t action_hash;
} BPInfoKey;

/* Info set data — integer regrets for memory efficiency.
 * Pluribus stores regrets as int32, strategies derived from regrets.
 * With bucket-in-key, each info set stores regrets for ONE bucket only.
 *
 * strategy_sum lifecycle (NOTE: this differs from Pluribus — see SOLVER_CONFIG.md §10):
 *   - Round 1: accumulated continuously by traverse() at mccfr_blueprint.c:1283-1289
 *     on every traverser visit (full regret-matched distribution added).
 *   - Rounds 2-4: accumulated in-memory by accumulate_snapshot() at
 *     mccfr_blueprint.c:1343-1358, called from bp_solve at snapshot barriers.
 *     Note that accumulate_snapshot does NOT filter by street, so round 1 also
 *     gets snapshot-accumulated in addition to per-visit accumulation.
 *
 * Pluribus uses a sparse sampled UPDATE-STRATEGY for round 1 (every 10K iters,
 * one path sampled, single phi[I,a] += 1) and on-disk snapshots for rounds 2-4
 * to halve memory. We use full-distribution per-visit + in-memory snapshots,
 * which is algorithmically defensible (lower variance, more memory). */
typedef struct {
    int num_actions;
    int *regrets;          /* [num_actions], int32 */
    float *strategy_sum;   /* [num_actions] or NULL (lazy) */
    float *action_evs;     /* [num_actions] or NULL (populated by bp_compute_action_evs).
                            * Phase 1.3: sum of per-action expected values observed during
                            * the σ̄-sampled EV walk. Average EV = action_evs[a] / ev_visit_count.
                            * Arena-allocated lazily on first traverser visit during the EV
                            * walk; NULL until then. */
    int ev_visit_count;    /* Phase 1.3: number of traverser visits that contributed to
                            * action_evs during the EV walk. Used to normalize the
                            * accumulated EVs into averages at export time. Relaxed atomic. */
} BPInfoSet;

/* Hash table (open addressing, linear probing) */
typedef struct {
    BPInfoKey *keys;
    BPInfoSet *sets;
    int *occupied;
    int64_t num_entries;
    int64_t table_size;             /* actual hash table size (supports up to 4B+) */
    int64_t insertion_failures;     /* atomic, lifetime count of probe-cap-exceeded returns */
    int64_t max_probe_observed;     /* atomic, lifetime maximum probe distance */
} BPInfoTable;

/* Blueprint solver configuration */
typedef struct {
    /* Timing-based thresholds (converted to iteration counts).
     * Pluribus used wall-clock minutes; we convert to iterations
     * based on estimated iterations/minute on the target hardware.
     * Uses int64 because full Pluribus-equivalent compute on fast hardware
     * requires ~100B+ iterations, exceeding INT32_MAX. */
    int64_t discount_stop_iter;     /* stop Linear CFR discount after this (Pluribus: 400 min) */
    int64_t discount_interval;      /* apply discount every N iterations (Pluribus: ~10 min) */
    int64_t prune_start_iter;       /* start pruning after this (Pluribus: 200 min) */
    int64_t snapshot_start_iter;    /* start saving snapshots after this (Pluribus: 800 min) */
    int64_t snapshot_interval;      /* save snapshot every N iterations (Pluribus: ~200 min) */
    /* DEPRECATED — set by ~30 ctypes callers but never read by
     * mccfr_blueprint.c. The original Pluribus design uses Strategy Interval
     * = 10K to gate sampled UPDATE-STRATEGY for round 1; our implementation
     * accumulates strategy_sum on every visit (SOLVER_CONFIG.md §10), so
     * this field is dead code. Kept in the struct for ABI safety with the
     * 30+ ctypes callers; ignore in new code. See STATUS.md §E3 and
     * BLUEPRINT_BUGS.md (Bug 6 explains why the gated path is unsafe). */
    int64_t strategy_interval;
    int num_threads;            /* OpenMP threads (0 = auto) */
    int64_t hash_table_size;    /* 0 = auto. Supports >2B (e.g. 3B for 376GB instance) */
    const char *snapshot_dir;   /* directory for strategy snapshots (NULL = no snapshots) */
    int include_preflop;        /* 1 = start from preflop (unified Pluribus-style), 0 = start from flop */
    int postflop_num_buckets;   /* 0 = default (200). Reduce to shrink game tree. */
    int river_ochs_enabled;     /* 0 = EHS-percentile (legacy); 1 = OCHS (Johanson 2013, Pluribus-faithful) */
    int river_ochs_num_clusters;/* 0 = default (8) opponent clusters */
} BPConfig;

/* Blueprint solver state */
typedef struct {
    /* Players */
    int num_players;
    int hands[BP_MAX_PLAYERS][BP_MAX_HANDS][2]; /* actual cards */
    float weights[BP_MAX_PLAYERS][BP_MAX_HANDS];
    int num_hands[BP_MAX_PLAYERS];

    /* Card abstraction: hand -> bucket mapping per street.
     * bucket_map[street][player][hand_idx] = bucket index.
     * street: 0=preflop, 1=flop, 2=turn, 3=river.
     * When use_buckets=0, bucket = hand_idx (identity mapping). */
    int use_buckets;
    int bucket_map[4][BP_MAX_PLAYERS][BP_MAX_HANDS];
    int num_buckets[4][BP_MAX_PLAYERS]; /* buckets per street per player */

    /* Board */
    int flop[3];

    /* Bet sizing (postflop first raise: Pluribus turn/river = 0.5x, 1x, all-in) */
    float bet_sizes[BP_MAX_ACTIONS];
    int num_bet_sizes;

    /* Postflop subsequent raises (Pluribus turn/river = 1x, all-in) */
    float subsequent_bet_sizes[BP_MAX_ACTIONS];
    int num_subsequent_bet_sizes;

    /* Preflop bet sizing (Pluribus: 1-14 sizes per decision point) */
    float preflop_bet_sizes[BP_MAX_ACTIONS];
    int num_preflop_bet_sizes;

    /* Tiered preflop sizing: per-raise-level arrays (Pluribus-style).
     * Level 0 = open raise, 1 = 3-bet, 2 = 4-bet, 3 = 5-bet.
     * If num_preflop_tiers > 0, these override preflop_bet_sizes. */
    float preflop_tiered_sizes[4][BP_MAX_ACTIONS];
    int num_preflop_tiered_sizes[4];  /* sizes per level */
    int num_preflop_tiers;            /* 0 = use flat preflop_bet_sizes */
    int preflop_max_raises;           /* 0 = default (4) */

    /* Phase 3: gate the implicit-allin block in generate_actions().
     * 1 = legacy behavior (always append explicit all-in raise when no
     *     bet_size already clamped to stack). v3 trained with this = 1.
     * 0 = no implicit all-in. Used by the Phase 3 warm-restart trainer
     *     so allin is never an option at deep-stack nodes (where stack
     *     >> any bet_size-derived raise). Short-stack scenarios where
     *     a bet_size genuinely clamps to all-in (mccfr_blueprint.c:913)
     *     are unaffected. See docs/PHASE3_RETRAIN_STATUS.md. */
    int add_explicit_allin;

    /* Phase R: depth-limit dispatch + cont-strat sampling at multiway-flop
     * boundary (mirror of street_solve.cu:159-186 §4 dispatch).
     *
     * depth_limit_cont_strats:
     *   0 = legacy full-game-depth MCCFR (default; existing checkpoints
     *       carry forward unchanged).
     *   1 = at end-of-multiway-flop, with probability cont_strat_prob,
     *       terminate the recursion and accumulate to the cont-strat EV
     *       slot table; otherwise recurse legacy. Block A.1 wires the
     *       dispatch site; Block A.2 fills in the HEADS-branch rollout.
     *
     * cont_strat_prob:
     *   p_boundary in [0, 1]; default 0.5. Probability that a multiway-
     *   flop iter terminates at the boundary (HEADS) vs. recurses past
     *   it (TAILS). TAILS keeps multiway turn/river σ̄ co-evolving;
     *   HEADS accumulates cont-bias EVs against the running σ̄. See
     *   docs/PHASE_R_TRAINER_SPEC.md §3.
     *
     * INVARIANT (A.1 byte-identical regression): when
     * depth_limit_cont_strats == 0, the dispatch helper returns 0
     * deterministically without consuming RNG. The new field's
     * presence in BPSolver does NOT change traversal behavior unless
     * the flag is explicitly enabled. */
    int depth_limit_cont_strats;
    float cont_strat_prob;

    /* Phase R Block A.2 — boundary EV slot table.
     *
     * Allocated lazily on bp_set_depth_limit_cont_strats(s, 1) using
     * cont_ev_table_size_hint (see below) as the requested capacity.
     * Stays NULL when the feature is OFF.
     *
     * Storage: float ev_sum[4] (Linear-CFR-discounted via the same
     * periodic-discount tick that scales the regret table) + int64
     * ev_count[4] (unweighted visit count) per slot, keyed on
     * (action_hash_at_boundary, traverser, bucket).
     *
     * Free'd by bp_free. */
    BPContEVTable cont_ev_table;

    /* Requested capacity for cont_ev_table at allocation time.
     * 0 → use the spec §4 default (1<<26 ≈ 67M slots, ~5GB at 72B/slot).
     * Tests override via bp_set_cont_ev_table_size to keep the unit-test
     * footprint small (~MB). */
    int64_t cont_ev_table_size_hint;

    /* Pot, stacks, and blinds */
    int starting_pot;       /* pot at start of postflop (when include_preflop=0) */
    int effective_stack;    /* stack at start of postflop (when include_preflop=0) */
    int small_blind;        /* SB amount in chips (e.g. 50 for $50/$100) */
    int big_blind;          /* BB amount in chips (e.g. 100) */
    int initial_stack;      /* starting stack per player (e.g. 10000 for 100BB) */

    /* Info set storage */
    BPInfoTable info_table;

    /* Config */
    BPConfig config;

    /* RNG state — one per thread for OpenMP */
    uint64_t *rng_states;    /* [num_threads] */
    int num_rng_states;

    /* Postflop bucket config for unified solver.
     * When include_preflop=1, postflop buckets are precomputed for all 1,755
     * flop textures at init time. During traversal, the canonical flop hash
     * is used to look up the precomputed 200-bucket mapping. */
    int postflop_num_buckets;  /* target buckets per street (Pluribus: 200) */
    int postflop_ehs_samples;  /* MC samples for EHS precompute (Pluribus: ~500) */

    /* Precomputed flop texture bucket cache.
     * Key: canonical flop hash (from 3 sorted ranks + suit pattern).
     * Value: bucket_map[1326] for that texture.
     * Allocated at init, freed by bp_free. */
    #define BP_MAX_TEXTURES 1760
    /* Bug 7 fix: open-addressing hashmap on top of texture_hash_keys[].
     * Sized to next power of 2 above 2 * BP_MAX_TEXTURES → 4096 buckets,
     * load factor ≤43% so probe chains stay under 3-4 on average. The hash
     * mixer is splitmix64-style. Replaces the previous O(N) linear scan
     * (~1755 comparisons per flop deal). Populated by
     * `texture_index_rebuild()` after the texture cache is loaded or
     * precomputed. */
    #define BP_TEXTURE_INDEX_SIZE 4096
    int *texture_bucket_cache;     /* [BP_MAX_TEXTURES * BP_MAX_HANDS] flat array */
    uint64_t texture_hash_keys[BP_MAX_TEXTURES];  /* hash key per texture */
    int texture_hash_index[BP_TEXTURE_INDEX_SIZE]; /* open-addressed: -1 = empty, else idx into texture_hash_keys */
    int num_cached_textures;

    /* Precomputed turn k-means centroids for bucketing.
     * 200 centroids in [EHS, PPot, NPot] feature space.
     * During traversal, each hand's features are computed inline and
     * mapped to nearest centroid. Matches Pluribus bucketing approach. */
    float turn_centroids[200][3];
    int turn_centroids_k;

    /* River OCHS abstraction (Johanson 2013, AAMAS). When
     * config.river_ochs_enabled == 1, the river bucket comes from nearest
     * centroid in OCHS feature space instead of EHS percentile.
     *
     * ochs_cluster_of_169: preflop class -> opponent cluster. Fixed at init.
     * river_centroids: flat [river_centroids_k * river_ochs_num_clusters].
     *   Per-run single global centroid set (option ii from the plan).
     *   Per-board-texture centroids are a planned extension; for now the
     *   same centroid set is reused across boards with on-the-fly per-hand
     *   feature compute at deal time (cached in river_bucket[p]). */
    int ochs_enabled;            /* mirror of config.river_ochs_enabled */
    int ochs_num_clusters;       /* k for opponent clusters, typically 8 */
    int ochs_cluster_of_169[169];
    float *river_centroids;      /* [river_centroids_k * ochs_num_clusters] */
    int river_centroids_k;

    /* Stats */
    int64_t iterations_run;
    int64_t snapshots_saved;

    /* Phase R Step 11 — zero-strategy-drift warm-start mode.
     *
     * When set (via BP_BOUNDARY_EV_ONLY=1 at solver init), traverse()
     * skips ALL writes to is->regrets[] and is->strategy_sum[], and
     * bp_solve() skips apply_discount() + accumulate_snapshot(). The
     * cont-strat HEADS-branch path remains active so cont_ev_table
     * keeps accumulating boundary EV slots — that is the whole point
     * of EV-only mode: regenerate the BPR5 trailer from a frozen
     * blueprint without drifting the underlying regrets/strategies.
     *
     * Default 0 (legacy CFR with regret + strategy_sum updates). */
    int boundary_ev_only;
} BPSolver;

/* ── Public API ──────────────────────────────────────────────────────── */

#ifdef _WIN32
#define BP_EXPORT __declspec(dllexport)
#else
#define BP_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize with default Pluribus-matched config.
 */
BP_EXPORT void bp_default_config(BPConfig *config);

/**
 * Initialize the blueprint solver.
 */
BP_EXPORT int bp_init(BPSolver *s, int num_players,
                      const int *flop,
                      const int hands[][BP_MAX_HANDS][2],
                      const float weights[][BP_MAX_HANDS],
                      const int *num_hands,
                      int starting_pot, int effective_stack,
                      const float *bet_sizes, int num_bet_sizes);

/**
 * Initialize with explicit config.
 */
BP_EXPORT int bp_init_ex(BPSolver *s, int num_players,
                          const int *flop,
                          const int hands[][BP_MAX_HANDS][2],
                          const float weights[][BP_MAX_HANDS],
                          const int *num_hands,
                          int starting_pot, int effective_stack,
                          const float *bet_sizes, int num_bet_sizes,
                          const BPConfig *config);

/**
 * Set card abstraction buckets for a given street.
 * Must be called after bp_init, before bp_solve.
 *
 * street: 0=preflop, 1=flop, 2=turn, 3=river
 * bucket_map[player][hand_idx] = bucket index
 * num_buckets[player] = number of buckets for this player
 */
BP_EXPORT int bp_set_buckets(BPSolver *s, int street,
                              const int bucket_map[][BP_MAX_HANDS],
                              const int *num_buckets);

/**
 * Set tiered preflop bet sizes (Pluribus-style: fewer sizes at higher raise levels).
 * level: 0 = open raise, 1 = 3-bet, 2 = 4-bet, 3 = 5-bet.
 * Call once per level after bp_init_unified, before bp_solve.
 * max_raises: total preflop raise cap (set on first call, e.g. 4).
 */
BP_EXPORT int bp_set_preflop_tier(BPSolver *s, int level,
                                   const float *sizes, int num_sizes,
                                   int max_raises);

/**
 * Phase 3: gate the implicit-allin block in generate_actions().
 * Pass 0 to disable (Phase 3 warm-restart trainer); default is 1
 * (legacy v3 training behavior — appended an explicit all-in raise to
 * every betting node when no bet_size already clamped to stack).
 *
 * See docs/PHASE3_RETRAIN_STATUS.md.
 */
BP_EXPORT int bp_set_add_explicit_allin(BPSolver *s, int enabled);

/**
 * Phase R: enable/disable the depth-limit + cont-strat-sampling dispatch
 * at multiway-flop boundary. See docs/PHASE_R_TRAINER_SPEC.md.
 *
 * enabled = 0 → legacy full-game-depth MCCFR (default).
 * enabled = 1 → boundary dispatch active; A.1 wires the site, A.2 fills
 *               in the HEADS-branch rollout.
 *
 * With enabled=0, the trainer is byte-identical to legacy on every iter:
 * the dispatch helper short-circuits before consuming RNG.
 */
BP_EXPORT int bp_set_depth_limit_cont_strats(BPSolver *s, int enabled);

/**
 * Phase R: set the boundary-coin-flip probability `p_boundary`.
 *
 * prob = probability of HEADS (terminate) vs TAILS (continue legacy)
 * at the multiway-flop boundary, used only when
 * depth_limit_cont_strats == 1. Default 0.5. Must be in [0, 1].
 *
 * Returns 0 on success, -1 if prob is out of range.
 */
BP_EXPORT int bp_set_cont_strat_prob(BPSolver *s, float prob);

/**
 * Phase R: set the cont-ev table size hint.
 *
 * Must be called BEFORE bp_set_depth_limit_cont_strats(s, 1) for the
 * setting to take effect — the table is allocated lazily on the first
 * call to bp_set_depth_limit_cont_strats with enabled=1, using this
 * hint as the requested capacity (prime-promoted internally).
 *
 * Pass `size_hint = 0` to fall back to the spec default (1<<26).
 *
 * Returns 0 on success, -1 on invalid args.
 */
BP_EXPORT int bp_set_cont_ev_table_size(BPSolver *s, int64_t size_hint);

/**
 * Phase R helper: pure dispatch decision for the depth-limit boundary.
 *
 * Returns 1 if the trainer should TERMINATE at this end-of-street (HEADS
 * branch — sample joint cont-bias and accumulate EV slot). Returns 0 if
 * the trainer should continue legacy recursion (TAILS branch or non-
 * boundary site).
 *
 * Pure function (no global state). Exposed so tests can verify the
 * dispatch matrix without exercising the full traverse() path.
 *
 * Invariant: when flag == 0, returns 0 without touching `*rng_state`.
 * (Required by the bit-identical regression test — see spec §7.)
 *
 * Called at end-of-street sites where round_done() has fired:
 *   - num_board: current board count (3 = end-of-flop, 4 = end-of-turn,
 *                5 = end-of-river → caller already has showdown branch).
 *   - n_active: number of non-folded players at this site.
 *   - flag: 0 or 1 (BPSolver::depth_limit_cont_strats).
 *   - prob: BPSolver::cont_strat_prob, ignored when flag == 0.
 *   - rng_state: per-thread xorshift state; consumed only on multiway
 *                flop with flag == 1.
 */
BP_EXPORT int bp_should_dispatch_to_boundary(int num_board, int n_active,
                                              int flag, float prob,
                                              uint64_t *rng_state);

/**
 * Phase R Block A.2 — pure-math cont-bias reweighting helper.
 *
 * Applies one of four Pluribus §C2 cont-bias reweightings to an
 * already-computed average strategy:
 *   bias 0: baseline (no reweight) — out = avg_strategy
 *   bias 1: fold-bias — ×5 weight on actions with type BP_ACT_FOLD
 *   bias 2: call-bias — ×5 weight on actions with BP_ACT_CHECK or BP_ACT_CALL
 *   bias 3: raise-bias — ×5 weight on actions with type BP_ACT_BET
 *
 * The reweighted distribution is L1-normalized into `out[]`. If the
 * reweighted sum is zero (e.g., all input probabilities are zero AND
 * the bias targets a non-existent action class), `out[]` falls back
 * to a uniform distribution.
 *
 * Pure function — no allocation, no global state, no RNG. Spec:
 * docs/PHASE_R_TRAINER_SPEC.md §3.
 *
 * Args:
 *   avg_strategy   [num_actions]   probabilities (need not sum to 1; will be
 *                                  used as relative weights then renormalized)
 *   action_types   [num_actions]   BP_ACT_FOLD / CHECK / CALL / BET
 *   num_actions                    1..BP_MAX_ACTIONS
 *   bias                           0..3 — out-of-range produces baseline
 *   out            [num_actions]   reweighted, L1-normalized distribution
 */
BP_EXPORT void bp_apply_cont_bias(const float *avg_strategy,
                                   const int *action_types,
                                   int num_actions, int bias,
                                   float *out);

/**
 * Phase R Block A.2 — pure action-history hash (mirrors trainer-internal
 * `compute_action_hash` at mccfr_blueprint.c:526).
 *
 * Exposed in the header so a regression test can pin the invariant
 * documented in spec §4: cont-bias does NOT enter the action_hash
 * computation. The function signature takes only `(actions, n_actions)`
 * — by construction cont_bias cannot affect the output. A test asserts
 * that the returned hash is deterministic and unaffected by Phase R
 * state (flag, prob, cont_bias_per_player) by re-deriving from the
 * raw action history alone.
 *
 * Pure function: no global state read or written beyond the legacy
 * mixer mode flag (set via bp_set_legacy_hash_mixer).
 */
BP_EXPORT uint64_t bp_compute_action_hash(const int *actions, int num_actions);

/* Phase R Block B.3 (Step 6) — exposed wrapper for the static
 * generate_actions inside mccfr_blueprint.c. Mirrors the production
 * action-emission contract byte-for-byte (same indices, same
 * fold/check/call/raise ordering, same allin saturation + dedup,
 * same explicit-allin gate). The encoding-alignment test drives a
 * pure-C trainer-walker against this wrapper to cross-validate the
 * kernel-built tree.
 *
 * Args mirror generate_actions:
 *   out, max_out         — output buffer for emitted BPAction[].
 *   pot, stack, to_call  — current decision-node state.
 *   num_raises, max_raises — raise-cap state.
 *   bet_sizes, num_bet_sizes — current per-level bet-size fractions.
 *     (Caller selects between *_first / *_subsequent via num_raises,
 *     same as the trainer's traverse() does at mccfr_blueprint.c:
 *     1581-1602.)
 *   add_explicit_allin   — global flag (s->add_explicit_allin).
 *
 * Returns: number of actions emitted (≥ 1, ≤ max_out, ≤ BP_MAX_GEN_ACTIONS).
 */
BP_EXPORT int bp_generate_actions(BPAction *out, int max_out,
                                   int pot, int stack, int to_call,
                                   int num_raises, int max_raises,
                                   const float *bet_sizes, int num_bet_sizes,
                                   int add_explicit_allin);

/**
 * Phase 3 post-load fixup: drop the trailing allin slot from infosets
 * saved by the legacy trainer (where the implicit-allin block fired).
 * Call after bp_load_regrets when add_explicit_allin = 0 and resuming
 * from a checkpoint that was trained with add_explicit_allin = 1.
 *
 * Returns the number of infosets dropped.
 */
BP_EXPORT int64_t bp_drop_trailing_allin_slot(BPSolver *s);

/**
 * Run external-sampling MCCFR with Pluribus optimizations.
 * Supports OpenMP parallelism, pruning, Linear CFR discount.
 */
BP_EXPORT int bp_solve(BPSolver *s, int64_t max_iterations);

/**
 * Extract weighted-average strategy at a specific info set.
 */
BP_EXPORT int bp_get_strategy(const BPSolver *s, int player,
                               const int *board, int num_board,
                               const int *action_seq, int seq_len,
                               float *strategy_out, int bucket);

/**
 * Extract raw integer regrets at a specific info set.
 * regrets_out must have space for at least BP_MAX_ACTIONS ints.
 * Returns num_actions, or 0 if info set not found.
 */
BP_EXPORT int bp_get_regrets(const BPSolver *s, int player,
                              const int *board, int num_board,
                              const int *action_seq, int seq_len,
                              int *regrets_out, int bucket);

BP_EXPORT int64_t bp_num_info_sets(const BPSolver *s);
BP_EXPORT void bp_free(BPSolver *s);

/**
 * Hash table health stats. All four out-pointers may be NULL to skip.
 * Safe to call concurrently with bp_solve (counters use relaxed atomics).
 *
 * insertion_failures: lifetime count of info_table_find_or_create calls that
 *   exhausted the 4096-probe cap and silently returned -1. Should always be 0
 *   in a properly-sized table. Nonzero indicates strategy_sum corruption.
 * max_probe_observed: lifetime maximum probe distance for any successful
 *   insertion. Approaching 4096 indicates the table is too full.
 */
BP_EXPORT void bp_get_table_stats(const BPSolver *s,
                                   int64_t *out_entries,
                                   int64_t *out_table_size,
                                   int64_t *out_insertion_failures,
                                   int64_t *out_max_probe_observed);

/**
 * Enable the legacy boost-style hash_combine mixer. Must be called BEFORE
 * bp_load_regrets when loading a v2 checkpoint (trained before commit
 * 48da71b on 2026-04-08 01:51 UTC). v2's stored action_hash values were
 * computed with the boost-style mixer; if Phase 1.3's traverse_ev runs
 * with splitmix64, every non-root info set lookup fails because the
 * recomputed action_hashes don't match the stored ones.
 */
BP_EXPORT void bp_set_legacy_hash_mixer(int enabled);

/**
 * Save/load precomputed texture bucket cache to skip the 65-90 min
 * precompute on subsequent launches. Format: TXC1 header + hash keys
 * + bucket mappings. ~9.3 MB file for 1755 textures × 1326 hands.
 */
BP_EXPORT int bp_save_texture_cache(const BPSolver *s, const char *path);
BP_EXPORT int bp_load_texture_cache(BPSolver *s, const char *path);

/**
 * Initialize a unified preflop-through-river solver (Pluribus-style).
 *
 * All 6 players get all 1326 hands. Preflop uses 169 lossless classes.
 * Postflop uses card abstraction via bp_set_buckets().
 *
 * Traversal starts from preflop with SB/BB posting, then UTG→MP→CO→BTN→SB→BB
 * acting in order. After preflop round, 3 flop cards are dealt, then
 * flop→turn→river betting rounds proceed as in the standard postflop solver.
 *
 * This matches Pluribus Algorithm 1 exactly: one unified MCCFR solve.
 *
 * Args:
 *   s: solver state (caller allocates)
 *   num_players: 6 for full table (2-6 supported)
 *   small_blind, big_blind: blind amounts in chips (e.g. 50, 100)
 *   initial_stack: starting stack per player in chips (e.g. 10000 for 100BB)
 *   postflop_bet_sizes, num_postflop_bet_sizes: bet fractions for flop/turn/river
 *   preflop_bet_sizes, num_preflop_bet_sizes: bet fractions for preflop raises
 *   config: solver configuration
 */
BP_EXPORT int bp_init_unified(BPSolver *s, int num_players,
                                int small_blind, int big_blind, int initial_stack,
                                const float *postflop_bet_sizes, int num_postflop_bet_sizes,
                                const float *preflop_bet_sizes, int num_preflop_bet_sizes,
                                const BPConfig *config);

/**
 * Export ALL info set strategies to a binary buffer.
 *
 * For each occupied slot in the hash table, writes:
 *   BPInfoKey (player, board_hash, action_hash)  — 20 bytes
 *   uint8 num_actions                             — 1 byte
 *   uint8 num_hands                               — 1 byte (actually num_buckets)
 *   uint8[num_actions * num_hands] strategy       — quantized to 0-255
 *
 * Strategies are derived from regrets via regret matching (not raw regrets).
 * Quantization: strategy[a] * 255, rounded, clamped to [0, 255].
 *
 * Args:
 *   s: solved state
 *   buf: output buffer (must be pre-allocated)
 *   buf_size: size of buf in bytes
 *   bytes_written: output, actual bytes written
 *
 * Returns 0 on success, -1 if buffer too small.
 * Call with buf=NULL to query required buffer size via bytes_written.
 */
BP_EXPORT int bp_export_strategies(const BPSolver *s,
                                    unsigned char *buf, size_t buf_size,
                                    size_t *bytes_written);

/**
 * Save the full regret table to a binary file for checkpoint/resume.
 *
 * Writes all occupied hash table slots (keys + regrets + strategy_sum)
 * to a binary file that can be reloaded via bp_load_regrets().
 *
 * Format:
 *   Header: "BPRG" (4B) + table_size (4B) + num_entries (4B) + iterations_run (4B)
 *   For each occupied slot:
 *     BPInfoKey (player 4B, street 4B, board_hash 8B, action_hash 8B) = 24B
 *     num_actions (4B), num_hands (4B) = 8B
 *     regrets[num_actions * num_hands] (int32) = variable
 *     has_strategy_sum (4B)
 *     if has_strategy_sum: strategy_sum[num_actions * num_hands] (float32) = variable
 *
 * Returns 0 on success, -1 on write error.
 */
BP_EXPORT int bp_save_regrets(const BPSolver *s, const char *path);

/**
 * Load a previously saved regret table from a checkpoint file.
 *
 * Must be called AFTER bp_init_unified or bp_init_ex (hash table must exist).
 * Loaded entries are inserted into the existing hash table. If the table is
 * too small for the saved entries, excess entries are silently dropped.
 *
 * Returns number of entries loaded, or -1 on error.
 */
BP_EXPORT int64_t bp_load_regrets(BPSolver *s, const char *path);

/**
 * Phase 1.3: Compute per-action expected values under the average strategy σ̄.
 *
 * Runs num_iterations of a modified MCCFR traversal where:
 *   - Strategies at all decision nodes are sampled from strategy_sum (the
 *     average strategy), not regret-matched from current regrets.
 *   - At traverser nodes, per-action EVs `action_values[a]` are accumulated
 *     into `is->action_evs[a]` and `is->ev_visit_count` is incremented.
 *   - Regrets and strategy_sum are NOT updated (read-only with respect to
 *     the underlying blueprint).
 *
 * After this runs, `action_evs[a] / ev_visit_count` is an unbiased estimator
 * of `v̄(I, a)` — the counterfactual value of action `a` at info set `I`
 * under the average strategy. Math: action_evs[a] / ev_visit_count is an
 * unbiased σ̄-weighted counterfactual value estimator.
 *
 * Must be called AFTER bp_load_regrets or bp_solve so that strategy_sum is
 * populated. Must be called BEFORE bp_export_action_evs.
 *
 * Multi-threaded via OpenMP (same thread count as bp_solve).
 *
 * Returns 0 on success.
 */
BP_EXPORT int bp_compute_action_evs(BPSolver *s, int64_t num_iterations);

/**
 * Phase 1.3: Export per-action EVs to a binary buffer.
 *
 * For each occupied slot with ev_visit_count > 0, writes:
 *   player(1) + street(1) + bucket(2) + action_hash(8) + num_actions(1)
 *   + float32[num_actions] avg_ev (= action_evs[a] / ev_visit_count)
 *
 * Header: "BPR3" (4) + u32 num_entries
 *
 * Info sets with ev_visit_count == 0 are skipped. Consumer should fall back
 * to equity approximation at those info sets.
 *
 * Args:
 *   s: solver state (after bp_compute_action_evs)
 *   buf: output buffer (pre-allocated; pass NULL to query required size)
 *   buf_size: size of buf in bytes
 *   bytes_written: output, actual bytes written (or required size if buf=NULL)
 *
 * Returns 0 on success, -1 if buffer too small.
 */
BP_EXPORT int bp_export_action_evs(const BPSolver *s,
                                    unsigned char *buf, size_t buf_size,
                                    size_t *bytes_written);

/**
 * Phase 1.3: Aggregate visit count statistics across all info sets that
 * the EV walk touched (ev_visit_count > 0).
 *
 * The exported BPR3 section only writes the NORMALIZED average EV per
 * action, not the raw visit count. Downstream sentinels need the visit
 * distribution to judge EV confidence and flag low-traffic info sets.
 * This function reports the aggregate stats cheaply at export time.
 *
 * Fills the caller-provided struct with:
 *   total_visited: number of info sets with ev_visit_count > 0
 *   min/p10/p50/p90/p99/max: percentile visit counts across those sets
 *   below_5: count with ev_visit_count < 5 (low-confidence)
 *   below_100: count with ev_visit_count < 100
 *   above_1000: count with ev_visit_count >= 1000 (high-confidence)
 *
 * Safe to call after bp_compute_action_evs. Iterates the hash table once.
 */
typedef struct {
    int64_t total_visited;
    int64_t min_visits;
    int64_t p10_visits;
    int64_t p50_visits;
    int64_t p90_visits;
    int64_t p99_visits;
    int64_t max_visits;
    int64_t below_5;
    int64_t below_100;
    int64_t above_1000;
} BPEVVisitStats;

BP_EXPORT int bp_get_ev_visit_stats(const BPSolver *s, BPEVVisitStats *out);

/**
 * Export EHS values and bucket assignments for all hands.
 *
 * Writes per-hand: float32 EHS, int32 bucket_index
 * Total: num_hands * 8 bytes
 *
 * Args:
 *   s: solver state (after bp_set_buckets)
 *   street: which street's bucketing (1=flop, 2=turn, 3=river)
 *   player: which player
 *   bucket_out: [num_hands] bucket assignments
 *
 * Returns num_hands.
 */
BP_EXPORT int bp_export_buckets(const BPSolver *s, int street, int player,
                                 int *bucket_out);

/**
 * Phase R Step 11 — test wrapper for the multiway-flop boundary
 * bucket-lookup helper. Exposed because the static helper takes a
 * private TraversalState; the wrapper accepts the raw inputs needed
 * to drive the same logic from a unit test.
 *
 * Mirrors compute_boundary_flop_bucket() inside mccfr_blueprint.c:
 *   1. Canonicalize the 3 dealt flop cards (rank-descending, suit-pattern
 *      → canonical suit assignment).
 *   2. compute_board_hash on the canonical board.
 *   3. texture_index_lookup → texture_idx.
 *   4. Read s->texture_bucket_cache[texture_idx * BP_MAX_HANDS + hand_idx].
 *
 * Args:
 *   s       — initialized BPSolver with texture cache loaded.
 *   board   — 3 dealt flop cards (raw, NOT canonicalized).
 *   player  — only used for bounds-checking; bucket lookup is player-
 *             agnostic at the postflop layer.
 *   hand_idx — full-deck pair index in [0, BP_MAX_HANDS).
 *
 * Returns:
 *   ≥0  — flop bucket in [0, postflop_num_buckets).
 *   -1  — texture cache empty or texture not found (defensive).
 */
BP_EXPORT int bp_compute_flop_bucket_for_test(const BPSolver *s,
                                                const int *board,
                                                int player,
                                                int hand_idx);

#ifdef __cplusplus
}
#endif

#endif /* MCCFR_BLUEPRINT_H */
