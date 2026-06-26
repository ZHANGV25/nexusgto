/**
 * rollout.c — Pluribus-faithful depth-limited rollout leaf values (C port).
 *
 * Port of python/rollout.py, function-for-function. See
 * docs/C2_CPU_ROLLOUT_PLAN.md for module layout, RNG seed derivation,
 * known parity-with-Python limitations, and the test plan.
 *
 * Algorithm: Brown, Sandholm, "Superhuman AI for multiplayer poker"
 * (Science 2019, supplementary §4 / Algorithm 2).
 */
#include "rollout.h"
#include "rollout_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "hand_eval.h"
#include "mccfr_cont_ev_table.h"  /* Phase F: cont_ev_table consumer */

/* External declarations for the trainer-side action_hash function. We
 * avoid including mccfr_blueprint.h because that pulls in the entire
 * trainer header surface (card_abstraction, OMP types, etc.) which is
 * out of scope for the rollout TU's link line. The single function we
 * need is exported via BP_EXPORT — see src/mccfr_blueprint.h:521. */
extern uint64_t bp_compute_action_hash(const int *actions, int num_actions);

/* ── Compile-time caps ───────────────────────────────────────────────── */

#define ROLLOUT_MAX_HANDS          1326
/* ROLLOUT_MAX_ACTION_HISTORY now lives in rollout.h — it sizes a public
 * field on RolloutLeafInfo. */
#define ROLLOUT_DECK               52

/* ── splitmix64 (matches src/bps_reader.c:18-23) ─────────────────────── */
/* rollout_splitmix64_next and rollout_derive_seed are defined as static inline
 * in rollout_internal.h (shared with rollout_sampled.c). */

/* Uniform [0.0, 1.0) from a 64-bit state. */
static inline double rollout_rand_double(uint64_t *state) {
    /* Use top 53 bits for a mantissa-exact double in [0, 1). */
    uint64_t x = rollout_splitmix64_next(state);
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);  /* 2^53 */
}

/* Weighted discrete sample. weights[:n] need not sum to 1; total > 0.
 * Returns index in [0, n). */
static inline int32_t rollout_rand_weighted(
    uint64_t *state, const double *weights, int32_t n, double total
) {
    double u = rollout_rand_double(state) * total;
    double acc = 0.0;
    for (int32_t i = 0; i < n; i++) {
        acc += weights[i];
        if (u < acc) return i;
    }
    return n - 1;  /* numerical slack guard */
}

/* ── Phase F: cont_ev_table consumer state ────────────────────────────
 *
 * Process-global handle + flag. Set once at startup before workers
 * fork (so each worker inherits the same pointer/flag via COW); the
 * cont_ev_table itself is read-only at consume time so concurrent
 * reads across workers are safe.
 *
 * The hit-count counter is per-thread (matches the existing
 * observability pattern in this file). Workers' counts can be folded
 * back via the existing add_observability pathway if needed; for the
 * Phase 1 smoke we just sum across the parent thread.
 */
static const BPContEVTable *g_rollout_cont_ev_table = NULL;
static int g_rollout_use_cont_ev_at_flop_boundary = 0;

/* SYNTHETIC mode (DEBUG / MEASUREMENT ONLY — default OFF). When enabled, a
 * boundary query MISS/EMPTY_BIAS does NOT fail loud; instead the consumer
 * find_or_creates the slot with a dummy finite EV (ev_sum=0, ev_count=1) using
 * the SAME key the query used, so this and every subsequent query for that key
 * returns OK. This lets the armed-RESIDUAL latency probe measure the
 * cont_ev-armed decision wall-clock (independent of the EV VALUES — only of
 * hit-vs-miss) WITHOUT a populated walk artifact. INVALID (bias out of range)
 * is a real bug and still fails loud. The real (synthetic OFF) path is
 * UNCHANGED and still fail-loud. Applies to BOTH the flop and turn boundaries.
 * A separate MUTABLE table handle is kept because the normal consumer is a
 * pure reader (const) and only synthetic mode inserts. Both are set by the
 * same rollout_set_cont_ev_table call. */
static BPContEVTable *g_rollout_cont_ev_table_mut = NULL;
static int g_rollout_cont_ev_synthetic = 0;

/* Dual-boundary build (2026-06-25): independent TURN-boundary enable flag,
 * parallel to the flop flag above. Default OFF — the served path does NOT
 * change. The SAME g_rollout_cont_ev_table holds both boundaries' slots
 * (street disambiguation is in the key-hash, bp_cont_ev_boundary_key_hash),
 * so no separate table handle is needed. The two flags are orthogonal:
 * either, both, or neither boundary may be armed. */
static int g_rollout_use_cont_ev_at_turn_boundary = 0;

/* PROCESS-GLOBAL atomic counters (NOT thread-local). The construct runs the
 * rollouts under `#pragma omp parallel for`, so a per-thread counter is
 * invisible to the parent thread that reads it from Python afterward (each
 * OMP worker has its own _Thread_local copy; the parent's stays 0). The
 * earlier per-thread design note in this file ("sum across the parent
 * thread") does not hold for OMP workers — these must be shared + atomically
 * updated. hits = successful all-4-bias short-circuits; attempts = every time
 * the flop-boundary predicate fired (the hit-rate denominator).
 *
 * Updated via __atomic builtins (gcc/clang) / Interlocked-style fallback so
 * concurrent worker increments don't race; read with a plain relaxed load
 * from the parent after the omp region joins. */
static volatile uint64_t g_rollout_cont_ev_boundary_hits = 0;
static volatile uint64_t g_rollout_cont_ev_boundary_attempts = 0;

/* Dual-boundary build: SEPARATE turn-boundary hit/attempt counters so a
 * turn hit-rate isn't aliased into the flop counters (and vice versa). A
 * turn-decision construct queries ONLY turn boundaries; a flop-decision
 * construct queries ONLY flop boundaries — but keeping them distinct makes
 * the per-boundary hit-rate unambiguous in mixed harnesses. */
static volatile uint64_t g_rollout_cont_ev_turn_boundary_hits = 0;
static volatile uint64_t g_rollout_cont_ev_turn_boundary_attempts = 0;

BPS_EXPORT void rollout_set_cont_ev_table(const BPContEVTable *t) {
    g_rollout_cont_ev_table = t;
    /* The underlying table object is mutable (the loader populates it); the
     * read-only handle is the consumer's pure-reader view. Synthetic mode uses
     * the mutable handle to insert dummy slots on miss. Casting away const is
     * sound: the caller owns a non-const BPContEVTable. */
    g_rollout_cont_ev_table_mut = (BPContEVTable *)t;
}

BPS_EXPORT const BPContEVTable *rollout_get_cont_ev_table(void) {
    return g_rollout_cont_ev_table;
}

/* SYNTHETIC mode toggle (DEBUG / MEASUREMENT ONLY — default OFF). See the
 * g_rollout_cont_ev_synthetic declaration comment. Applies to both boundaries. */
BPS_EXPORT void rollout_set_cont_ev_synthetic(int enabled) {
    g_rollout_cont_ev_synthetic = enabled ? 1 : 0;
}

BPS_EXPORT int rollout_get_cont_ev_synthetic(void) {
    return g_rollout_cont_ev_synthetic;
}

/* SYNTHETIC-mode slot fill, shared by the flop and turn consumer loops.
 * On a MISS/EMPTY_BIAS (synthetic ON, mutable table set), create-or-fill the
 * slot with a dummy finite EV (0.0) keyed IDENTICALLY to the query (the caller
 * passes the already-salted key), so this and every subsequent query for that
 * key returns OK. Returns 1 on success (treat as OK), 0 if synthetic cannot
 * satisfy (caller should fail loud exactly as the real path would). Never
 * called for INVALID (a real bug). */
static int rollout_cont_ev_synth_fill(uint64_t key_ah, int player,
                                      int bucket, int bias) {
    if (!g_rollout_cont_ev_synthetic || g_rollout_cont_ev_table_mut == NULL)
        return 0;
    BPContEVEntry *e = bp_cont_ev_find_or_create(
        g_rollout_cont_ev_table_mut, key_ah, player, bucket);
    if (e == NULL) return 0;  /* table saturated/unallocated — cannot synthesize */
    if (e->ev_count[bias] == 0) {
        e->ev_sum[bias] = 0.0f;
        e->ev_count[bias] = 1;
    }
    return 1;
}

BPS_EXPORT void rollout_set_use_cont_ev_at_flop_boundary(int enabled) {
    g_rollout_use_cont_ev_at_flop_boundary = enabled ? 1 : 0;
}

BPS_EXPORT int rollout_get_use_cont_ev_at_flop_boundary(void) {
    return g_rollout_use_cont_ev_at_flop_boundary;
}

BPS_EXPORT void rollout_get_cont_ev_boundary_hit_count(uint64_t *out_hits) {
    if (out_hits)
        *out_hits = __atomic_load_n(&g_rollout_cont_ev_boundary_hits,
                                    __ATOMIC_RELAXED);
}

BPS_EXPORT void rollout_reset_cont_ev_boundary_hit_count(void) {
    __atomic_store_n(&g_rollout_cont_ev_boundary_hits, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_rollout_cont_ev_boundary_attempts, 0, __ATOMIC_RELAXED);
}

/* A3: total flop-boundary query attempts (the hit-rate denominator). */
BPS_EXPORT void rollout_get_cont_ev_boundary_attempt_count(uint64_t *out_attempts) {
    if (out_attempts)
        *out_attempts = __atomic_load_n(&g_rollout_cont_ev_boundary_attempts,
                                        __ATOMIC_RELAXED);
}

/* ── Dual-boundary build (2026-06-25): TURN-boundary consumer API ─────── */

BPS_EXPORT void rollout_set_use_cont_ev_at_turn_boundary(int enabled) {
    g_rollout_use_cont_ev_at_turn_boundary = enabled ? 1 : 0;
}

BPS_EXPORT int rollout_get_use_cont_ev_at_turn_boundary(void) {
    return g_rollout_use_cont_ev_at_turn_boundary;
}

/* Turn-boundary successful all-4-bias short-circuits. */
BPS_EXPORT void rollout_get_cont_ev_turn_boundary_hit_count(uint64_t *out_hits) {
    if (out_hits)
        *out_hits = __atomic_load_n(&g_rollout_cont_ev_turn_boundary_hits,
                                    __ATOMIC_RELAXED);
}

/* Turn-boundary query attempts (the turn hit-rate denominator). */
BPS_EXPORT void rollout_get_cont_ev_turn_boundary_attempt_count(uint64_t *out_attempts) {
    if (out_attempts)
        *out_attempts = __atomic_load_n(&g_rollout_cont_ev_turn_boundary_attempts,
                                        __ATOMIC_RELAXED);
}

/* Reset BOTH turn counters (parallel to rollout_reset_cont_ev_boundary_hit_count
 * for the flop). Kept separate so a harness can reset/measure one boundary
 * without disturbing the other's counts. */
BPS_EXPORT void rollout_reset_cont_ev_turn_boundary_hit_count(void) {
    __atomic_store_n(&g_rollout_cont_ev_turn_boundary_hits, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_rollout_cont_ev_turn_boundary_attempts, 0, __ATOMIC_RELAXED);
}

/* Mirror of bps_reader.c::hand_index_for (static there).
 * Returns the canonical full-deck pair index for the traverser's hole
 * cards relative to a 3-card flop, in [0, 1326). The trainer's BPR5
 * trailer encodes the boundary slot's `bucket` field as this value
 * (per Phase R Step 11/12 — texture-cache mixer mismatch caused
 * the trainer's bucket lookup to fall back to identity = hand_idx;
 * the verifier accepts only this encoding, see
 * tools/phase_r_probe_analysis/verify_bpr5_roundtrip.c).
 *
 * Inputs c0/c1 are 0..51 deck indices. The board is the 3 flop cards
 * (only board[0..2] used). Returns -1 if the hero's hole cards are
 * blocked by the flop (caller surfaces as fail-loud).
 */
static int rollout_hand_index_for_flop(const int32_t *board,
                                        int32_t c0, int32_t c1) {
    int blocked[52] = {0};
    for (int i = 0; i < 3; i++) {
        if (board[i] < 0 || board[i] >= 52) return -1;
        blocked[board[i]] = 1;
    }
    if (c0 < 0 || c0 >= 52 || c1 < 0 || c1 >= 52) return -1;
    if (blocked[c0] || blocked[c1]) return -1;
    if (c0 > c1) { int tmp = c0; c0 = c1; c1 = tmp; }
    return c0 * 51 - c0 * (c0 - 1) / 2 + c1 - c0 - 1;
}

/* Dual-boundary build (2026-06-25) — TURN analogue. Same canonical
 * full-deck pair index in [0, 1326), but the blocked set is the 4-card
 * TURN board (board[0..3]). The index VALUE is identical to the flop
 * helper for the same hole cards (the formula depends only on c0,c1) —
 * the turn card only adds a blocking constraint. This MUST match the
 * producer's bp_boundary_slot_hand_index_turn exactly (mccfr_blueprint.c)
 * or every turn-boundary query MISSes. Returns -1 if hero's hole cards
 * are blocked by the 4-card board (caller surfaces as fail-loud). */
static int rollout_hand_index_for_turn(const int32_t *board,
                                        int32_t c0, int32_t c1) {
    int blocked[52] = {0};
    for (int i = 0; i < 4; i++) {
        if (board[i] < 0 || board[i] >= 52) return -1;
        blocked[board[i]] = 1;
    }
    if (c0 < 0 || c0 >= 52 || c1 < 0 || c1 >= 52) return -1;
    if (blocked[c0] || blocked[c1]) return -1;
    if (c0 > c1) { int tmp = c0; c0 = c1; c1 = tmp; }
    return c0 * 51 - c0 * (c0 - 1) / 2 + c1 - c0 - 1;
}

/* Exported test hook for the TURN hand-index. Exposes the EXACT static
 * function the live consumer calls at the turn boundary, so the
 * key-alignment test can assert producer-write == consumer-read against
 * the REAL consumer code (not a re-typed copy of the formula — a re-typed
 * copy could drift from the live path and give false confidence). board
 * is the 4 turn cards (board[0..3]). */
BPS_EXPORT int rollout_hand_index_for_turn_test(const int32_t *board,
                                                int32_t c0, int32_t c1) {
    return rollout_hand_index_for_turn(board, c0, c1);
}

/* ── Thread-local error surface ──────────────────────────────────────── */

typedef struct {
    char    reason[256];
    int32_t street;
    int32_t num_players;
} RolloutLastError;

#if defined(_MSC_VER)
static __declspec(thread) RolloutLastError g_rollout_last_error;
#else
static _Thread_local RolloutLastError g_rollout_last_error;
#endif

void rollout_set_error(int32_t street, int32_t num_players,
                              const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_rollout_last_error.reason,
              sizeof(g_rollout_last_error.reason), fmt, ap);
    va_end(ap);
    g_rollout_last_error.street = street;
    g_rollout_last_error.num_players = num_players;
}

BPS_EXPORT void rollout_get_last_error(
    const char **reason, int32_t *street, int32_t *num_players
) {
    if (reason)      *reason      = g_rollout_last_error.reason;
    if (street)      *street      = g_rollout_last_error.street;
    if (num_players) *num_players = g_rollout_last_error.num_players;
}

BPS_EXPORT int32_t rollout_leaf_info_sizeof(void) {
    return (int32_t)sizeof(RolloutLeafInfo);
}

/* ── Observability counters ──────────────────────────────────────────── */

/* Pluribus Algorithm 1 supplementary substitutes uniform-on-legal-actions
 * for missing or zero-sum blueprint σ̄. The two cases mean different
 * things and are counted separately:
 *   blueprint_miss_count: σ̄ entry not in blueprint (coverage gap; expected
 *                         outside the trained tree).
 *   zerosum_count:        σ̄ entry present but the truncated strategy sums
 *                         to zero (data integrity issue; should be rare). */
typedef struct {
    uint64_t blueprint_miss_count;
    uint64_t zerosum_count;
    uint64_t bucket_miss_count;
     * Index by street: 0=preflop, 1=flop, 2=turn, 3=river. Distinguishes
     * action_history hash failure (high miss on every street) from
     * blueprint coverage sparseness (miss skewed to street≥1). */
    uint64_t blueprint_miss_by_street[4];
} RolloutObservability;

#if defined(_MSC_VER)
static __declspec(thread) RolloutObservability g_rollout_obs;
#else
static _Thread_local RolloutObservability g_rollout_obs;
#endif

BPS_EXPORT void rollout_reset_observability(void) {
    g_rollout_obs.blueprint_miss_count = 0;
    g_rollout_obs.zerosum_count = 0;
    g_rollout_obs.bucket_miss_count = 0;
    for (int i = 0; i < 4; i++) g_rollout_obs.blueprint_miss_by_street[i] = 0;
}

BPS_EXPORT void rollout_get_observability(
    uint64_t *out_miss, uint64_t *out_zerosum
) {
    if (out_miss)    *out_miss    = g_rollout_obs.blueprint_miss_count;
    if (out_zerosum) *out_zerosum = g_rollout_obs.zerosum_count;
}

BPS_EXPORT void rollout_get_bucket_miss_count(uint64_t *out_bucket_miss) {
    if (out_bucket_miss) *out_bucket_miss = g_rollout_obs.bucket_miss_count;
}

 * into out[4] (street: 0=preflop, 1=flop, 2=turn, 3=river). */
BPS_EXPORT void rollout_get_per_street_miss_counts(uint64_t out[4]) {
    if (!out) return;
    for (int i = 0; i < 4; i++) out[i] = g_rollout_obs.blueprint_miss_by_street[i];
}

/* Add to the per-thread observability counters. Used by the Python
 * orchestration to fold worker-process counter totals back into the
 * parent thread's view after a multiprocessing.Pool fan-out — workers
 * each have their own thread-local copy, and the parent's view would
 * otherwise stay at zero despite real misses. */
BPS_EXPORT void rollout_add_observability(
    uint64_t add_miss, uint64_t add_zerosum, uint64_t add_bucket_miss
) {
    g_rollout_obs.blueprint_miss_count += add_miss;
    g_rollout_obs.zerosum_count        += add_zerosum;
    g_rollout_obs.bucket_miss_count    += add_bucket_miss;
}

BPS_EXPORT void rollout_add_per_street_miss_counts(const uint64_t add[4]) {
    if (!add) return;
    for (int i = 0; i < 4; i++) g_rollout_obs.blueprint_miss_by_street[i] += add[i];
}

/* ── Tolerate-bucket-miss gate ───────────────────────────────────────── */

/* -1 = uninitialized (lazy env read on first hot-path use); 0 = off; 1 = on.
 * Process-wide (NOT thread-local): the toggle is a configuration choice
 * for the whole rollout campaign, not a per-thread setting. The lazy-read
 * race is benign — concurrent first-callers all observe the same env
 * value and write the same int. */
static int g_rollout_tolerate_bucket_miss = -1;

static int rollout_should_tolerate_bucket_miss(void) {
    int v = g_rollout_tolerate_bucket_miss;
    if (v < 0) {
        const char *e = getenv("ROLLOUT_TOLERATE_BUCKET_MISS");
        v = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T')) ? 1 : 0;
        g_rollout_tolerate_bucket_miss = v;
    }
    return v;
}

BPS_EXPORT void rollout_set_tolerate_bucket_miss(int enabled) {
    g_rollout_tolerate_bucket_miss = enabled ? 1 : 0;
}

BPS_EXPORT int rollout_get_tolerate_bucket_miss(void) {
    return rollout_should_tolerate_bucket_miss();
}

/* ── apply_bias ──────────────────────────────────────────────────────── */

/* Matches python/rollout.py::apply_bias.
 *
 * Semantics:
 *  - strategy sum <= 0 → uniform(n)
 *  - bias_profile == 0 → strategy / sum(strategy)
 *  - bias_profile ∈ {1,2,3} → multiply entries matching target category
 *    by BIAS_MULTIPLIER, renormalize. If post-bias sum is 0 (no mass on
 *    target, e.g. raise-biased at check-only node) → fall back to
 *    unbiased strategy/sum, NOT uniform.
 */
BPS_EXPORT void rollout_apply_bias_c(
    const double *strategy, int32_t n,
    int32_t bias_profile,
    const int32_t *categories,
    double *out
) {
    if (n <= 0 || !out) return;

    double s_sum = 0.0;
    for (int32_t i = 0; i < n; i++) s_sum += strategy[i];

    if (s_sum <= 0.0) {
        double u = 1.0 / (double)n;
        for (int32_t i = 0; i < n; i++) out[i] = u;
        return;
    }

    /* Python uses x / sum(x), not x * (1/sum). We mirror that to stay
     * bit-exact — (1/sum) introduces a sub-ULP reciprocal error that
     * breaks bit-for-bit parity on the apply_bias unit tests. */
    if (bias_profile == 0) {
        for (int32_t i = 0; i < n; i++) out[i] = strategy[i] / s_sum;
        return;
    }

    int32_t target;
    switch (bias_profile) {
        case 1:  target = ROLLOUT_CAT_FOLD;  break;
        case 2:  target = ROLLOUT_CAT_CALL;  break;
        case 3:  target = ROLLOUT_CAT_RAISE; break;
        default: {
            for (int32_t i = 0; i < n; i++) out[i] = strategy[i] / s_sum;
            return;
        }
    }

    double biased[ROLLOUT_MAX_ACTIONS_PER_NODE];
    if (n > ROLLOUT_MAX_ACTIONS_PER_NODE) {
        for (int32_t i = 0; i < n; i++) out[i] = strategy[i] / s_sum;
        return;
    }
    double total = 0.0;
    for (int32_t i = 0; i < n; i++) {
        biased[i] = strategy[i];
        if (categories && i < n && categories[i] == target) {
            biased[i] *= ROLLOUT_BIAS_MULTIPLIER;
        }
        total += biased[i];
    }

    if (total <= 0.0) {
        for (int32_t i = 0; i < n; i++) out[i] = strategy[i] / s_sum;
        return;
    }

    for (int32_t i = 0; i < n; i++) out[i] = biased[i] / total;
}

/* ── classify_postflop_actions ───────────────────────────────────────── */

/* Read once: whether to append an explicit all-in slot to postflop action
 * sets. MUST match the training-time generate_actions() add_explicit_allin
 * gate — otherwise the rollout's action-history indices contain a value the
 * trainer never used at that position, and every subsequent blueprint query
 * misses (cascading miss). phase3 trained with add_explicit_allin=0, so the
 * default here is 0 too. Override via env BPS_ROLLOUT_ADD_EXPLICIT_ALLIN=1
 * if you load a blueprint that WAS trained with the gate on. */
static int g_rollout_add_explicit_allin = -1;

static int rollout_should_add_explicit_allin(void) {
    int v = __atomic_load_n(&g_rollout_add_explicit_allin, __ATOMIC_RELAXED);
    if (v >= 0) return v;
    const char *e = getenv("BPS_ROLLOUT_ADD_EXPLICIT_ALLIN");
    int parsed = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T' ||
                        e[0] == 'y' || e[0] == 'Y' || e[0] == 'o' || e[0] == 'O'))
                  ? 1 : 0;
    __atomic_store_n(&g_rollout_add_explicit_allin, parsed, __ATOMIC_RELAXED);
    return parsed;
}

/* Matches python/rollout.py::classify_postflop_actions.
 *
 * Returns the number of actions. Layout (depending on
 * BPS_ROLLOUT_ADD_EXPLICIT_ALLIN; phase3 default is OFF):
 *   tree-closed + facing_bet: [fold, call]
 *   tree-closed + !facing_bet: [check]
 *   open + facing_bet (first raise):
 *     allin off (default): [fold, call, raise×0.5, raise×1.0]
 *     allin on:            [fold, call, raise×0.5, raise×1.0, allin]
 *   open + facing_bet (subsequent):
 *     allin off (default): [fold, call, raise×1.0]
 *     allin on:            [fold, call, raise×1.0, allin]
 *   open + !facing_bet:
 *     allin off (default): [check, bet×0.5, bet×1.0]
 *     allin on:            [check, bet×0.5, bet×1.0, allin]
 */
BPS_EXPORT int32_t rollout_classify_postflop_actions_c(
    int32_t facing_bet, int32_t raises_so_far,
    int32_t *out_cats, int32_t cap
) {
    int32_t n = 0;
    int32_t tree_closed = (raises_so_far >= ROLLOUT_POSTFLOP_MAX_RAISES);
    int32_t add_allin = rollout_should_add_explicit_allin();

    if (tree_closed) {
        if (facing_bet) {
            if (cap >= 2 && out_cats) {
                out_cats[0] = ROLLOUT_CAT_FOLD;
                out_cats[1] = ROLLOUT_CAT_CALL;
            }
            return 2;
        } else {
            if (cap >= 1 && out_cats) out_cats[0] = ROLLOUT_CAT_CALL;  /* check */
            return 1;
        }
    }

    if (facing_bet) {
        int32_t n_raises = (raises_so_far > 0) ? ROLLOUT_SUBSEQUENT_RAISE_N
                                               : ROLLOUT_FIRST_RAISE_N;
        /* [fold, call, raise×…, (allin)?] */
        n = 2 + n_raises + (add_allin ? 1 : 0);
        if (cap >= n && out_cats) {
            out_cats[0] = ROLLOUT_CAT_FOLD;
            out_cats[1] = ROLLOUT_CAT_CALL;
            for (int32_t i = 0; i < n_raises; i++) {
                out_cats[2 + i] = ROLLOUT_CAT_RAISE;
            }
            if (add_allin) out_cats[n - 1] = ROLLOUT_CAT_RAISE;  /* allin */
        }
        return n;
    }

    /* not facing bet: [check, bet×0.5, bet×1.0, (allin)?] */
    n = 1 + ROLLOUT_FIRST_RAISE_N + (add_allin ? 1 : 0);
    if (cap >= n && out_cats) {
        out_cats[0] = ROLLOUT_CAT_CALL;  /* check */
        for (int32_t i = 0; i < ROLLOUT_FIRST_RAISE_N; i++) {
            out_cats[1 + i] = ROLLOUT_CAT_RAISE;
        }
        if (add_allin) out_cats[n - 1] = ROLLOUT_CAT_RAISE;  /* allin */
    }
    return n;
}

/* ── action_amount_chips ─────────────────────────────────────────────── */

/* Matches python/rollout.py::action_amount_chips. See that function's
 * docstring for (pot, to_call, stack) semantics; raise_to formula is
 * Pluribus's: raise_to = to_call + frac * (pot + to_call). */
BPS_EXPORT void rollout_action_amount_chips_c(
    int32_t action_idx,
    int32_t facing_bet, int32_t raises_so_far,
    int32_t pot, int32_t to_call, int32_t stack,
    int32_t *out_kind, int32_t *out_amount
) {
    int32_t cats_n = rollout_classify_postflop_actions_c(
        facing_bet, raises_so_far, NULL, 0);
    int32_t tree_closed = (raises_so_far >= ROLLOUT_POSTFLOP_MAX_RAISES);
    int32_t add_allin = rollout_should_add_explicit_allin();

    if (action_idx < 0 || action_idx >= cats_n) {
        /* Python raises ValueError; we return a conservative check. */
        if (out_kind)   *out_kind   = ROLLOUT_KIND_CHECK;
        if (out_amount) *out_amount = 0;
        return;
    }

    if (facing_bet) {
        if (action_idx == 0) {
            if (out_kind)   *out_kind   = ROLLOUT_KIND_FOLD;
            if (out_amount) *out_amount = 0;
            return;
        }
        if (action_idx == 1) {
            int32_t a = to_call < stack ? to_call : stack;
            if (out_kind)   *out_kind   = ROLLOUT_KIND_CALL;
            if (out_amount) *out_amount = a;
            return;
        }
        if (tree_closed) {
            int32_t a = to_call < stack ? to_call : stack;
            if (out_kind)   *out_kind   = ROLLOUT_KIND_CALL;
            if (out_amount) *out_amount = a;
            return;
        }
        if (add_allin && action_idx == cats_n - 1) {
            if (out_kind)   *out_kind   = ROLLOUT_KIND_ALLIN;
            if (out_amount) *out_amount = stack;
            return;
        }
        /* Sized raise */
        int32_t frac_slot = action_idx - 2;
        double frac;
        if (raises_so_far > 0) {
            /* SUBSEQUENT_BET_FRACTIONS = (1.0,) */
            frac = 1.0;
        } else {
            /* POSTFLOP_BET_FRACTIONS = (0.5, 1.0) */
            frac = (frac_slot == 0) ? 0.5 : 1.0;
        }
        /* raise_to = to_call + int(frac * (pot + to_call)) — Python int() truncates toward 0 */
        int32_t raise_to = to_call + (int32_t)(frac * (double)(pot + to_call));
        int32_t amount = raise_to < stack ? raise_to : stack;
        /* Python: max(amount, to_call + 1) if stack > to_call + 1 else stack */
        int32_t kind_amt;
        if (stack > to_call + 1) {
            kind_amt = amount > (to_call + 1) ? amount : (to_call + 1);
        } else {
            kind_amt = stack;
        }
        if (out_kind)   *out_kind   = ROLLOUT_KIND_RAISE;
        if (out_amount) *out_amount = kind_amt;
        return;
    }

    /* Not facing bet */
    if (action_idx == 0) {
        if (out_kind)   *out_kind   = ROLLOUT_KIND_CHECK;
        if (out_amount) *out_amount = 0;
        return;
    }
    if (tree_closed) {
        if (out_kind)   *out_kind   = ROLLOUT_KIND_CHECK;
        if (out_amount) *out_amount = 0;
        return;
    }
    if (add_allin && action_idx == cats_n - 1) {
        if (out_kind)   *out_kind   = ROLLOUT_KIND_ALLIN;
        if (out_amount) *out_amount = stack;
        return;
    }
    int32_t frac_slot = action_idx - 1;
    double frac = (frac_slot == 0) ? 0.5 : 1.0;
    int32_t bet_amount = (int32_t)(frac * (double)pot);
    /* Python: min(max(bet_amount, 1), stack) */
    int32_t a = bet_amount > 1 ? bet_amount : 1;
    if (a > stack) a = stack;
    if (out_kind)   *out_kind   = ROLLOUT_KIND_BET;
    if (out_amount) *out_amount = a;
}

/* ── Showdown payoff ─────────────────────────────────────────────────── */

/* Matches python/rollout.py::compute_showdown_payoff — winner-takes /
 * chop, no side pots. */
BPS_EXPORT float rollout_compute_showdown_payoff_c(
    const int *full_board,
    const int *hands_c0, const int *hands_c1,
    int32_t num_players,
    const int32_t *active,
    const int32_t *bets,
    int32_t pot_chips,
    int32_t hero_player
) {
    /* Live players at showdown */
    int32_t live[ROLLOUT_MAX_PLAYERS];
    int32_t n_live = 0;
    for (int32_t i = 0; i < num_players; i++) {
        if (active[i]) live[n_live++] = i;
    }
    if (n_live == 0) {
        return -(float)bets[hero_player];
    }

    /* Strength per live player */
    uint32_t strengths[ROLLOUT_MAX_PLAYERS];
    uint32_t best = 0;
    for (int32_t k = 0; k < n_live; k++) {
        int32_t p = live[k];
        int seven[7] = {
            full_board[0], full_board[1], full_board[2],
            full_board[3], full_board[4],
            hands_c0[p], hands_c1[p]
        };
        uint32_t s = eval7(seven);
        strengths[k] = s;
        if (s > best) best = s;
    }

    int32_t n_winners = 0;
    int32_t hero_wins = 0;
    for (int32_t k = 0; k < n_live; k++) {
        if (strengths[k] == best) {
            n_winners++;
            if (live[k] == hero_player) hero_wins = 1;
        }
    }

    double hero_share = 0.0;
    if (hero_wins) {
        hero_share = (double)pot_chips / (double)n_winners;
    }
    return (float)(hero_share - (double)bets[hero_player]);
}

/* ── Blueprint lookup ────────────────────────────────────────────────── */

/* Returns num_actions written to out_strat (normalized), or 0 on miss.
 * Matches python/rollout.py::_lookup_strategy.
 *
 * Note: bps_query_strategy already normalizes; zero-sum inputs are
 * turned into uniform by C. Caller substitutes uniform-on-legal-actions
 * on miss (return value 0) per Pluribus Algorithm 1 supplementary; the
 * substitution count is exposed via rollout_get_observability. */
static int rollout_lookup_strategy(
    const BPSReader *r,
    const int32_t *action_history, int32_t ah_len,
    int32_t player, int32_t bucket, int32_t street,
    float *out_strat, int32_t cap
) {
    if (cap < BPS_MAX_ACTIONS) {
        /* Should never happen — callers pass a big-enough buffer. */
        return 0;
    }
    /* bps_query_strategy signature: const int *action_history, int action_len.
     * Our ah values are int32_t which matches int on ILP32/LP64/LLP64. */
    int na = bps_query_strategy(r, (const int *)action_history,
                                 (int)ah_len,
                                 (int)player, (int)street, (int)bucket,
                                 out_strat);
    return na;  /* 0 means node not found */
}

/* ── Chance / dealing ────────────────────────────────────────────────── */

/* Uniform random card from the 52 not in used_mask. Returns -1 on
 * exhausted deck. */
static int32_t rollout_deal_one_card(uint64_t *rng, uint64_t used_mask) {
    /* Count unused */
    int32_t unused[ROLLOUT_DECK];
    int32_t n_unused = 0;
    for (int32_t c = 0; c < ROLLOUT_DECK; c++) {
        if (!(used_mask & (1ULL << c))) unused[n_unused++] = c;
    }
    if (n_unused == 0) return -1;
    uint64_t r = rollout_splitmix64_next(rng);
    int32_t idx = (int32_t)(r % (uint64_t)n_unused);
    return unused[idx];
}

/* Sample one opponent hand weighted by reach, avoiding conflicts.
 * Returns 0 on success, 1 if no valid hand. */
static int rollout_sample_opp_hand(
    uint64_t *rng,
    const RolloutHand *candidates, int32_t n_cand,
    uint64_t used_mask,
    int32_t *out_c0, int32_t *out_c1
) {
    /* Gather valid candidates + weights. */
    double weights[ROLLOUT_MAX_HANDS];
    int32_t valid_idx[ROLLOUT_MAX_HANDS];
    int32_t n_valid = 0;
    double total = 0.0;

    int32_t cap = n_cand;
    if (cap > ROLLOUT_MAX_HANDS) cap = ROLLOUT_MAX_HANDS;
    for (int32_t i = 0; i < cap; i++) {
        if (candidates[i].weight <= 0.0f) continue;
        int32_t c0 = candidates[i].c0;
        int32_t c1 = candidates[i].c1;
        if (c0 < 0 || c0 >= 52 || c1 < 0 || c1 >= 52) continue;
        if (used_mask & (1ULL << c0)) continue;
        if (used_mask & (1ULL << c1)) continue;
        weights[n_valid] = (double)candidates[i].weight;
        valid_idx[n_valid] = i;
        total += weights[n_valid];
        n_valid++;
    }
    if (n_valid == 0 || total <= 0.0) return 1;

    int32_t pick = rollout_rand_weighted(rng, weights, n_valid, total);
    int32_t idx = valid_idx[pick];
    *out_c0 = candidates[idx].c0;
    *out_c1 = candidates[idx].c1;
    return 0;
}

/* ── _simulate_one ───────────────────────────────────────────────────── */

/* Streets: 1=flop 2=turn 3=river. Returns 0 on success (payoff written
 * to *out_payoff), -1 on LeafValueUnavailable (last-error set).
 *
 * skip_first_betting: when 1, skip the betting loop on leaf_street and
 * advance straight to the next street's chance/deal step. Used by the
 * subgame leaf paths when leaf->flop_history_len > 0 — in that case the
 * leaf already represents end-of-street state and the betting we'd
 * fictitiously redo here would (a) corrupt blueprint queries on the
 * NEXT street by appending fictitious actions to game_history, and
 * (b) miss training storage on the CURRENT street because training
 * stored end-of-flop info sets at street=2 (turn), not street=1 (flop).
 * See PHASE0_STATUS.md / advisor analysis 2026-04-30. */
int rollout_simulate_one(
    const BPSReader *r,
    /* immutable inputs */
    const int *board_in, int32_t n_board_in,
    int32_t leaf_street,
    int32_t pot_in,
    const int32_t *active_in,
    const int32_t *stacks_in,
    const int32_t *bets_in,
    int32_t num_players,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_history_in, int32_t gah_len_in,
    const int32_t *bias_profile,
    int32_t hero_player,
    const int32_t *sampled_hands_c0,
    const int32_t *sampled_hands_c1,
    const int32_t *has_sample,  /* 1 if sampled_hands[p] is set */
    int32_t skip_first_betting,
    uint64_t *rng,
    /* out */
    float *out_payoff
) {
    /* Local copies (mutable during rollout) */
    int32_t active[ROLLOUT_MAX_PLAYERS];
    int32_t stacks[ROLLOUT_MAX_PLAYERS];
    int32_t bets[ROLLOUT_MAX_PLAYERS];
    int32_t board[5];
    int32_t n_board = n_board_in;
    int32_t game_history[ROLLOUT_MAX_ACTION_HISTORY];
    int32_t gah_len = gah_len_in;

    for (int32_t i = 0; i < num_players; i++) {
        active[i] = active_in[i];
        stacks[i] = stacks_in[i];
        bets[i]   = bets_in[i];
    }
    for (int32_t i = 0; i < n_board_in; i++) board[i] = board_in[i];
    if (gah_len > ROLLOUT_MAX_ACTION_HISTORY) {
        rollout_set_error(leaf_street, num_players,
            "game_action_history too long (%d > %d)",
            gah_len, ROLLOUT_MAX_ACTION_HISTORY);
        return -1;
    }
    for (int32_t i = 0; i < gah_len; i++) game_history[i] = game_history_in[i];

    int32_t pot = pot_in;
    int32_t street = leaf_street;
    int32_t first_iter = 1;

    while (1) {
        /* If every live player has gone all-in (no chips left), there
         * is no more betting action possible. Skip straight to chance /
         * showdown. Without this check, a post-allin river node would
         * query σ̄ at a closed betting line (ah_len overflow into a
         * terminal info set), producing spurious blueprint misses. */
        {
            int32_t n_live_can_act = 0;
            for (int32_t i = 0; i < num_players; i++) {
                if (active[i] && stacks[i] > 0) n_live_can_act++;
            }
            if (n_live_can_act < 2) goto street_closed;
        }

        /* Skip the betting loop on the leaf's street when the leaf
         * already represents end-of-street state (its flop_history was
         * threaded into game_history via the subgame leaf path). The
         * current-street betting was already accounted for by the
         * subgame tree; redoing it here would either miss (training
         * stored end-of-flop info sets at street=NEXT) or pollute
         * subsequent queries with fictitious actions. */
        if (first_iter && skip_first_betting) {
            first_iter = 0;
            goto street_closed;
        }
        first_iter = 0;

        /* Betting round for current street */
        int32_t street_bets[ROLLOUT_MAX_PLAYERS] = {0};
        int32_t current_bet = 0;
        int32_t raises_this_street = 0;
        int32_t acted[ROLLOUT_MAX_PLAYERS] = {0};
        int32_t safety = num_players * (ROLLOUT_POSTFLOP_MAX_RAISES + 2) + 4;

        for (int32_t iter_idx = 0; iter_idx < safety; iter_idx++) {
            /* Find next active player owing action, in acting_order */
            int32_t chosen = -1;
            for (int32_t k = 0; k < acting_order_len; k++) {
                int32_t p = acting_order[k];
                if (!active[p]) continue;
                if (stacks[p] <= 0) continue;  /* allin player can't act */
                if (!acted[p] || street_bets[p] < current_bet) {
                    chosen = p;
                    break;
                }
            }
            if (chosen < 0) break;

            int32_t n_active = 0;
            for (int32_t i = 0; i < num_players; i++) if (active[i]) n_active++;
            if (n_active <= 1) break;

            int32_t p = chosen;
            if (!has_sample[p]) {
                /* No hand sampled — treat as silent check/fold step.
                 * Matches Python: "treat as check/fold conservatively". */
                acted[p] = 1;
                continue;
            }

            int32_t c0 = sampled_hands_c0[p];
            int32_t c1 = sampled_hands_c1[p];
            /* Dispatch bucket lookup by current street / board size.
             * Flop uses the precomputed texture cache; turn uses
             * k-means centroids; river uses deterministic 200-sample
             * EHS percentile. All three match training-time bucketing
             * exactly (bps_ev_walk.c:855-898 is the precedent). */
            int bucket;
            if (n_board == 3) {
                bucket = bps_hand_to_bucket(r, board, c0, c1);
            } else if (n_board == 4) {
                bucket = bps_hand_to_bucket_turn(r, board, c0, c1);
            } else if (n_board == 5) {
                bucket = bps_hand_to_bucket_river(r, board, c0, c1);
            } else {
                bucket = -1;
            }
            int32_t bucket_miss = 0;
            if (bucket < 0) {
                if (rollout_should_tolerate_bucket_miss()) {
                    /* Coverage gap in the texture/bucket precompute
                     * (e.g. paired-flushdraw flop textures missing from
                     * texture_cache.bin: the precompute enumeration in
                     * mccfr_blueprint.c emits only one of two variants
                     * when r0==r1 or r1==r2). The .bps has no
                     * strategies for these textures either, so the
                     * principled response is the same as for any
                     * coverage gap: Pluribus Alg.1 supplementary
                     * uniform-on-legal-actions. We jump straight to
                     * that branch by skipping the strategy lookup;
                     * calling ca_assign_buckets_kmeans inline is not
                     * an option (batch precompute, ~hundreds of ms
                     * per board, and the bucket numbers it produces
                     * index into other textures' trained buckets —
                     * functionally arbitrary). */
                    g_rollout_obs.bucket_miss_count++;
                    bucket_miss = 1;
                } else {
                    char board_desc[64] = {0};
                    int off = 0;
                    for (int32_t bi = 0; bi < n_board && off < 60; bi++) {
                        off += snprintf(board_desc + off, 64 - off,
                                        "%d%s", board[bi], bi + 1 < n_board ? "," : "");
                    }
                    rollout_set_error(street, num_players,
                        "hand_to_bucket returned %d for (%d,%d) on %d-card board "
                        "[%s] (street=%d, player=%d); turn centroids loaded=%d",
                        bucket, c0, c1, n_board, board_desc, street, p,
                        /* r is NULL-tolerated everywhere else on this path
                         * (the bucketizers NULL-guard and return -1); guard
                         * the diagnostic deref too so a NULL-reader miss
                         * reports cleanly instead of segfaulting. */
                        r ? r->turn_centroids_k : -1);
                    return -1;
                }
            }

            int32_t facing_bet = (current_bet > street_bets[p]);
            int32_t to_call = facing_bet ? (current_bet - street_bets[p]) : 0;
            int32_t cats[ROLLOUT_MAX_ACTIONS_PER_NODE];
            int32_t n_actions = rollout_classify_postflop_actions_c(
                facing_bet, raises_this_street, cats,
                ROLLOUT_MAX_ACTIONS_PER_NODE);

            /* Blueprint lookup — full seat for the blueprint's
             * seat-indexed storage. Skipped on bucket_miss (no valid
             * key to query); falls through to the same uniform fallback
             * as a strategy miss. */
            int32_t full_seat = player_seats[p];
            float strat_full[BPS_MAX_ACTIONS];
            int bp_na = 0;
            if (!bucket_miss) {
                bp_na = rollout_lookup_strategy(
                    r, game_history, gah_len,
                    full_seat, bucket, street,
                    strat_full, BPS_MAX_ACTIONS);
            }
            if (bp_na <= 0) {
                /* Pluribus Algorithm 1 supplementary: σ̄ unavailable →
                 * uniform over the rollout tree's legal action set. The
                 * counter is exposed via rollout_get_observability so the
                 * substitution remains visible to callers (e.g., API
                 * logs an alert if miss-rate exceeds a threshold).
                 * Bucket-miss cases already bumped bucket_miss_count
                 * upstream; don't double-count them here. */
                if (!bucket_miss) {
                    g_rollout_obs.blueprint_miss_count++;
                     * action_history hash mismatch (high miss everywhere) from
                     * sparse coverage (miss skewed to deeper streets). */
                    int32_t st = (street >= 0 && street < 4) ? street : 0;
                    g_rollout_obs.blueprint_miss_by_street[st]++;
                }
                bp_na = n_actions;
                float u = 1.0f / (float)n_actions;
                for (int32_t i = 0; i < n_actions; i++) strat_full[i] = u;
            }

            /* Truncate/pad to rollout tree's action count — matches
             * python/rollout.py:485-503. */
            double strat[ROLLOUT_MAX_ACTIONS_PER_NODE];
            int32_t cats_active_n = n_actions;
            if (bp_na >= n_actions) {
                for (int32_t i = 0; i < n_actions; i++)
                    strat[i] = (double)strat_full[i];
            } else {
                for (int32_t i = 0; i < n_actions; i++) strat[i] = 0.0;
                for (int32_t i = 0; i < bp_na; i++)
                    strat[i] = (double)strat_full[i];
                /* Python: cats = cats[:max(len(strat_full), 1)] */
                cats_active_n = bp_na > 0 ? bp_na : 1;
                if (cats_active_n > n_actions) cats_active_n = n_actions;
            }

            /* Pluribus Algorithm 1 supplementary: post-truncate zero-sum
             * → uniform. This catches both bp_na < n_actions degenerates
             * (blueprint had values outside the rollout tree's action
             * range) and bp_na >= n_actions cases where the first
             * n_actions slots are zero (probability mass was entirely in
             * the truncated tail). bps_query_strategy already smooths
             * raw zero-regret entries to uniform, so this counter
             * primarily catches truncation edge cases. */
            double strat_sum = 0.0;
            for (int32_t i = 0; i < n_actions; i++) strat_sum += strat[i];
            if (strat_sum <= 0.0) {
                g_rollout_obs.zerosum_count++;
                double u = 1.0 / (double)n_actions;
                for (int32_t i = 0; i < n_actions; i++) strat[i] = u;
                cats_active_n = n_actions;
            }

            /* apply_bias — note Python passes `cats` (possibly truncated)
             * but strat has n_actions entries. apply_bias iterates to
             * min(len(biased), len(categories)) via the `if i < len(biased)`
             * guard. Entries beyond cats_active_n simply aren't category-
             * matched (category doesn't match target → not multiplied),
             * so effectively contribute raw strategy mass. Match that. */
            double biased[ROLLOUT_MAX_ACTIONS_PER_NODE];
            /* Build an effective category array that matches Python's
             * truncation: indices >= cats_active_n get a sentinel that
             * won't match any target. */
            int32_t eff_cats[ROLLOUT_MAX_ACTIONS_PER_NODE];
            for (int32_t i = 0; i < n_actions; i++) {
                eff_cats[i] = (i < cats_active_n) ? cats[i] : -1;
            }
            rollout_apply_bias_c(strat, n_actions, bias_profile[p],
                                 eff_cats, biased);

            /* Defensive re-normalize (python/rollout.py:506-510) */
            double bsum = 0.0;
            for (int32_t i = 0; i < n_actions; i++) bsum += biased[i];
            if (bsum <= 0.0) {
                double u = 1.0 / (double)n_actions;
                for (int32_t i = 0; i < n_actions; i++) biased[i] = u;
                bsum = 1.0;
            }

            int32_t action_idx = rollout_rand_weighted(rng, biased, n_actions, bsum);

            /* Execute action */
            int32_t kind, amt;
            rollout_action_amount_chips_c(
                action_idx, facing_bet, raises_this_street,
                pot, to_call, stacks[p], &kind, &amt);

            switch (kind) {
                case ROLLOUT_KIND_FOLD:
                    active[p] = 0;
                    acted[p] = 1;
                    break;
                case ROLLOUT_KIND_CALL:
                case ROLLOUT_KIND_CHECK:
                    stacks[p]      -= amt;
                    street_bets[p] += amt;
                    bets[p]        += amt;
                    pot            += amt;
                    acted[p] = 1;
                    break;
                case ROLLOUT_KIND_BET:
                case ROLLOUT_KIND_RAISE:
                case ROLLOUT_KIND_ALLIN:
                    stacks[p]      -= amt;
                    street_bets[p] += amt;
                    bets[p]        += amt;
                    pot            += amt;
                    if (street_bets[p] > current_bet) {
                        current_bet = street_bets[p];
                    }
                    raises_this_street++;
                    acted[p] = 1;
                    /* Reopen action for other live players */
                    for (int32_t q = 0; q < num_players; q++) {
                        if (q != p && active[q]) acted[q] = 0;
                    }
                    break;
                default:
                    rollout_set_error(street, num_players,
                        "unknown action kind %d", kind);
                    return -1;
            }

            /* Append to game history */
            if (gah_len >= ROLLOUT_MAX_ACTION_HISTORY) {
                rollout_set_error(street, num_players,
                    "game_action_history overflow during rollout");
                return -1;
            }
            game_history[gah_len++] = action_idx;

            int32_t n_active2 = 0;
            for (int32_t i = 0; i < num_players; i++) if (active[i]) n_active2++;
            if (n_active2 <= 1) break;
        }

street_closed:
        /* End of street */
        ;
        int32_t n_live_at_close = 0;
        for (int32_t i = 0; i < num_players; i++) if (active[i]) n_live_at_close++;
        if (n_live_at_close <= 1) break;

        /* Phase F: cont_ev_table boundary short-circuit.
         *
         * At end-of-flop in a multiway pot (and only there), the trainer
         * accumulated the boundary EV slot. If the consumer is enabled,
         * query the slot for hero's (action_hash, hand_idx) at all 4
         * traverser-bias indices, take the uniform mean, and return that
         * as the rollout's leaf value — short-circuiting the legacy
         * turn + river simulation that produced the §4b degeneracy
         * (uniform-on-legal fallback at 99% blueprint_miss).
         *
         * Bias-combination semantics: uniform mean. Pluribus §C2 calls
         * for cont-strat to be a CFR-selectable action, but the trainer
         * deliberately collapsed opponents' bias dimension via uniform
         * MCCFR sampling (PHASE_R_TRAINER_SPEC §3-4) — the producer's
         * σ̄ values were optimized against opps using uniform-mixed
         * biases. Consuming with uniform mean preserves the same
         * distribution: E_{b~uniform}[ev[b]] is the matching consumer.
         *
         * MISS / EMPTY_BIAS / INVALID is fail-loud per PHASE_R_TRAINER_SPEC
         * §5. No silent NaN sentinel, no heuristic fallback. If the
         * smoke surfaces coverage gaps at trafficked leaves, that's a
         * Phase 2c retraining decision, not a runtime concern. */
        if (g_rollout_use_cont_ev_at_flop_boundary
            && street == ROLLOUT_STREET_FLOP
            && n_live_at_close >= 3
            && g_rollout_cont_ev_table != NULL
            && hero_player >= 0 && hero_player < num_players
            && active[hero_player]
            && has_sample[hero_player]
            && n_board >= 3) {
            /* A3: count this query ATTEMPT before the lookup, so the hit-rate
             * denominator includes MISS/EMPTY_BIAS/hand-index failures. */
            __atomic_add_fetch(&g_rollout_cont_ev_boundary_attempts, 1,
                               __ATOMIC_RELAXED);
            uint64_t bah = bp_compute_action_hash(game_history, gah_len);
            int hidx = rollout_hand_index_for_flop(
                board,
                sampled_hands_c0[hero_player],
                sampled_hands_c1[hero_player]);
            if (hidx < 0) {
                rollout_set_error(street, num_players,
                    "cont_ev_table boundary: hand_index_for failed "
                    "(hero c0=%d c1=%d on flop %d,%d,%d)",
                    (int)sampled_hands_c0[hero_player],
                    (int)sampled_hands_c1[hero_player],
                    (int)board[0], (int)board[1], (int)board[2]);
                return -1;
            }
            float ev_acc = 0.0f;
            int n_ok = 0;
            for (int bias = 0; bias < BP_CONT_EV_NUM_BIASES; bias++) {
                float ev = 0.0f;
                BPBoundaryEVStatus st = bp_query_boundary_ev(
                    g_rollout_cont_ev_table, bah,
                    hero_player, hidx, bias, &ev);
                if (st == BP_BOUNDARY_EV_OK) {
                    ev_acc += ev;
                    n_ok++;
                } else if ((st == BP_BOUNDARY_EV_MISS
                            || st == BP_BOUNDARY_EV_EMPTY_BIAS)
                           && rollout_cont_ev_synth_fill(bah, hero_player,
                                                         hidx, bias)) {
                    /* SYNTHETIC (DEBUG/MEASUREMENT): dummy 0.0 filled; treat OK. */
                    ev_acc += 0.0f;
                    n_ok++;
                } else {
                    const char *st_name =
                        (st == BP_BOUNDARY_EV_EMPTY_BIAS) ? "EMPTY_BIAS" :
                        (st == BP_BOUNDARY_EV_MISS)       ? "MISS" :
                        (st == BP_BOUNDARY_EV_INVALID)    ? "INVALID" :
                                                            "UNKNOWN";
                    rollout_set_error(street, num_players,
                        "cont_ev_table boundary: status=%s ah=0x%016llx "
                        "player=%d hand_idx=%d bias=%d",
                        st_name, (unsigned long long)bah,
                        (int)hero_player, hidx, bias);
                    return -1;
                }
            }
            /* All 4 biases OK — uniform mean integrates over the
             * producer's MCCFR-uniform bias sampling distribution. */
            *out_payoff = ev_acc / (float)n_ok;
            __atomic_add_fetch(&g_rollout_cont_ev_boundary_hits, 1,
                               __ATOMIC_RELAXED);
            return 0;
        }

        /* ── Dual-boundary build (2026-06-25): TURN-boundary short-circuit ──
         *
         * Structurally identical to the FLOP block above, one street later:
         * at end-of-multiway-TURN (street==TURN, n_board==4, >=3 live), if
         * the turn consumer is enabled, query hero's (turn-salted action_hash,
         * turn hand_idx) at all 4 biases, take the uniform mean, and return it
         * as the leaf value — short-circuiting the legacy RIVER simulation.
         *
         * INDEPENDENT enable flag (g_rollout_use_cont_ev_at_turn_boundary) +
         * SEPARATE turn counters. The key uses the TURN street salt
         * (bp_cont_ev_boundary_key_hash) and the 4-card turn hand-index
         * (rollout_hand_index_for_turn) — EXACTLY what the producer wrote at
         * the turn boundary. Same fail-loud contract on MISS/EMPTY_BIAS/
         * INVALID. Flop default is untouched. */
        if (g_rollout_use_cont_ev_at_turn_boundary
            && street == ROLLOUT_STREET_TURN
            && n_live_at_close >= 3
            && g_rollout_cont_ev_table != NULL
            && hero_player >= 0 && hero_player < num_players
            && active[hero_player]
            && has_sample[hero_player]
            && n_board >= 4) {
            __atomic_add_fetch(&g_rollout_cont_ev_turn_boundary_attempts, 1,
                               __ATOMIC_RELAXED);
            uint64_t bah_t = bp_cont_ev_boundary_key_hash(
                bp_compute_action_hash(game_history, gah_len),
                BP_CONT_EV_STREET_TURN);
            int hidx_t = rollout_hand_index_for_turn(
                board,
                sampled_hands_c0[hero_player],
                sampled_hands_c1[hero_player]);
            if (hidx_t < 0) {
                rollout_set_error(street, num_players,
                    "cont_ev_table TURN boundary: hand_index_for failed "
                    "(hero c0=%d c1=%d on board %d,%d,%d,%d)",
                    (int)sampled_hands_c0[hero_player],
                    (int)sampled_hands_c1[hero_player],
                    (int)board[0], (int)board[1], (int)board[2],
                    (int)board[3]);
                return -1;
            }
            float ev_acc_t = 0.0f;
            int n_ok_t = 0;
            for (int bias = 0; bias < BP_CONT_EV_NUM_BIASES; bias++) {
                float ev = 0.0f;
                BPBoundaryEVStatus st = bp_query_boundary_ev(
                    g_rollout_cont_ev_table, bah_t,
                    hero_player, hidx_t, bias, &ev);
                if (st == BP_BOUNDARY_EV_OK) {
                    ev_acc_t += ev;
                    n_ok_t++;
                } else if ((st == BP_BOUNDARY_EV_MISS
                            || st == BP_BOUNDARY_EV_EMPTY_BIAS)
                           && rollout_cont_ev_synth_fill(bah_t, hero_player,
                                                         hidx_t, bias)) {
                    /* SYNTHETIC (DEBUG/MEASUREMENT): dummy 0.0 filled; treat OK.
                     * Keyed with the TURN-salted bah_t so synthetic turn slots
                     * stay disjoint from flop slots. */
                    ev_acc_t += 0.0f;
                    n_ok_t++;
                } else {
                    const char *st_name =
                        (st == BP_BOUNDARY_EV_EMPTY_BIAS) ? "EMPTY_BIAS" :
                        (st == BP_BOUNDARY_EV_MISS)       ? "MISS" :
                        (st == BP_BOUNDARY_EV_INVALID)    ? "INVALID" :
                                                            "UNKNOWN";
                    rollout_set_error(street, num_players,
                        "cont_ev_table TURN boundary: status=%s ah=0x%016llx "
                        "player=%d hand_idx=%d bias=%d",
                        st_name, (unsigned long long)bah_t,
                        (int)hero_player, hidx_t, bias);
                    return -1;
                }
            }
            *out_payoff = ev_acc_t / (float)n_ok_t;
            __atomic_add_fetch(&g_rollout_cont_ev_turn_boundary_hits, 1,
                               __ATOMIC_RELAXED);
            return 0;
        }

        /* Deal next street */
        int32_t next_street;
        if      (street == ROLLOUT_STREET_FLOP)  next_street = ROLLOUT_STREET_TURN;
        else if (street == ROLLOUT_STREET_TURN)  next_street = ROLLOUT_STREET_RIVER;
        else                                      break;  /* river done → showdown */

        /* Build used mask: board + all sampled player hands */
        uint64_t used = 0ULL;
        for (int32_t i = 0; i < n_board; i++) used |= (1ULL << board[i]);
        for (int32_t i = 0; i < num_players; i++) {
            if (has_sample[i]) {
                used |= (1ULL << sampled_hands_c0[i]);
                used |= (1ULL << sampled_hands_c1[i]);
            }
        }
        int32_t new_card = rollout_deal_one_card(rng, used);
        if (new_card < 0) {
            rollout_set_error(street, num_players, "deck exhausted during deal");
            return -1;
        }
        if (n_board >= 5) {
            rollout_set_error(street, num_players, "board overflow during deal");
            return -1;
        }
        board[n_board++] = new_card;
        street = next_street;
    }

    /* Terminal */
    int32_t n_live = 0;
    int32_t last_live = -1;
    for (int32_t i = 0; i < num_players; i++) {
        if (active[i]) { n_live++; last_live = i; }
    }
    if (n_live <= 1) {
        if (n_live == 1 && last_live == hero_player) {
            *out_payoff = (float)pot - (float)bets[hero_player];
        } else {
            *out_payoff = -(float)bets[hero_player];
        }
        return 0;
    }

    /* Showdown — deal remaining board if needed */
    while (n_board < 5) {
        uint64_t used = 0ULL;
        for (int32_t i = 0; i < n_board; i++) used |= (1ULL << board[i]);
        for (int32_t i = 0; i < num_players; i++) {
            if (has_sample[i]) {
                used |= (1ULL << sampled_hands_c0[i]);
                used |= (1ULL << sampled_hands_c1[i]);
            }
        }
        int32_t new_card = rollout_deal_one_card(rng, used);
        if (new_card < 0) {
            rollout_set_error(street, num_players, "deck exhausted pre-showdown");
            return -1;
        }
        board[n_board++] = new_card;
    }

    /* Fill hands_c0/c1 arrays for showdown; inactive get (0,1) sentinel. */
    int full_board[5];
    for (int i = 0; i < 5; i++) full_board[i] = board[i];

    int hands_c0[ROLLOUT_MAX_PLAYERS];
    int hands_c1[ROLLOUT_MAX_PLAYERS];
    for (int32_t i = 0; i < num_players; i++) {
        if (has_sample[i]) {
            hands_c0[i] = sampled_hands_c0[i];
            hands_c1[i] = sampled_hands_c1[i];
        } else {
            hands_c0[i] = 0;
            hands_c1[i] = 1;
        }
    }

    *out_payoff = rollout_compute_showdown_payoff_c(
        full_board, hands_c0, hands_c1, num_players,
        active, bets, pot, hero_player);
    return 0;
}

/* ── Public entry points ─────────────────────────────────────────────── */

/* Common args → compute mean payoff for one hero hand. See
 * rollout_simulate_one for the skip_first_betting semantics. */
static int rollout_compute_leaf_value_impl(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    int32_t pot_chips,
    const int32_t *active, int32_t num_players,
    const int32_t *stacks,
    const int32_t *bets,
    const RolloutHandList *player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    const int32_t *bias_profile,
    int32_t hero_player,
    int32_t hero_hand_idx,
    int32_t num_rollouts,
    uint64_t master_seed,
    int32_t leaf_idx_for_seed,
    int32_t combo_for_seed,
    int32_t skip_first_betting,
    float *out_value
) {
    if (num_players <= 0 || num_players > ROLLOUT_MAX_PLAYERS) {
        rollout_set_error(street, num_players, "bad num_players=%d", num_players);
        return -1;
    }
    if (hero_player < 0 || hero_player >= num_players) {
        rollout_set_error(street, num_players,
            "bad hero_player=%d", hero_player);
        return -1;
    }
    if (!active[hero_player]) {
        /* Hero folded — pay -bets[hero]. Matches Python. */
        *out_value = -(float)bets[hero_player];
        return 0;
    }
    const RolloutHandList *hh = &player_hands[hero_player];
    if (hero_hand_idx < 0 || hero_hand_idx >= hh->n_hands) {
        rollout_set_error(street, num_players,
            "hero_hand_idx %d out of range (%d)",
            hero_hand_idx, hh->n_hands);
        return -1;
    }
    int32_t hc0 = hh->hands[hero_hand_idx].c0;
    int32_t hc1 = hh->hands[hero_hand_idx].c1;
    float hw   = hh->hands[hero_hand_idx].weight;

    if (hw <= 0.0f) { *out_value = 0.0f; return 0; }

    /* Blocked by board? */
    uint64_t board_mask = 0ULL;
    for (int32_t i = 0; i < n_board_cards; i++) board_mask |= (1ULL << board[i]);
    if ((board_mask & (1ULL << hc0)) || (board_mask & (1ULL << hc1))) {
        *out_value = 0.0f;
        return 0;
    }

    double payoff_sum = 0.0;
    int32_t payoffs_recorded = 0;

    for (int32_t r_idx = 0; r_idx < num_rollouts; r_idx++) {
        uint64_t rng = rollout_derive_seed(
            master_seed, leaf_idx_for_seed, combo_for_seed,
            hero_player, hero_hand_idx, r_idx);

        /* Sample opponent hands */
        int32_t sampled_c0[ROLLOUT_MAX_PLAYERS];
        int32_t sampled_c1[ROLLOUT_MAX_PLAYERS];
        int32_t has_sample[ROLLOUT_MAX_PLAYERS];
        for (int32_t i = 0; i < num_players; i++) {
            sampled_c0[i] = 0; sampled_c1[i] = 0; has_sample[i] = 0;
        }
        sampled_c0[hero_player] = hc0;
        sampled_c1[hero_player] = hc1;
        has_sample[hero_player] = 1;

        uint64_t used = board_mask | (1ULL << hc0) | (1ULL << hc1);
        int ok = 1;
        for (int32_t p = 0; p < num_players; p++) {
            if (p == hero_player) continue;
            if (!active[p]) continue;
            int32_t c0, c1;
            if (rollout_sample_opp_hand(&rng,
                    player_hands[p].hands, player_hands[p].n_hands,
                    used, &c0, &c1) != 0) {
                ok = 0;
                break;
            }
            sampled_c0[p] = c0;
            sampled_c1[p] = c1;
            has_sample[p] = 1;
            used |= (1ULL << c0);
            used |= (1ULL << c1);
        }
        if (!ok) continue;

        float payoff;
        int rc = rollout_simulate_one(
            r, board, n_board_cards, street, pot_chips,
            active, stacks, bets, num_players,
            acting_order, acting_order_len,
            player_seats, game_action_history, gah_len,
            bias_profile, hero_player,
            sampled_c0, sampled_c1, has_sample,
            skip_first_betting,
            &rng, &payoff);
        if (rc != 0) {
            /* Propagate LeafValueUnavailable. */
            return -1;
        }
        payoff_sum += (double)payoff;
        payoffs_recorded++;
    }

    if (payoffs_recorded == 0) { *out_value = 0.0f; return 0; }
    *out_value = (float)(payoff_sum / (double)payoffs_recorded);
    return 0;
}

BPS_EXPORT int32_t rollout_compute_leaf_value_c(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    int32_t pot_chips,
    const int32_t *active, int32_t num_players,
    const int32_t *stacks,
    const int32_t *bets,
    const RolloutHandList *player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    const int32_t *bias_profile,
    int32_t hero_player,
    int32_t hero_hand_idx,
    int32_t num_rollouts,
    uint64_t master_seed,
    int32_t leaf_idx_for_seed,
    int32_t combo_for_seed,
    float *out_value
) {
    /* Public single-hand entry preserves its old signature; callers
     * outside the subgame leaf path don't deal with skip_first_betting,
     * so they always start the rollout with a fresh first-street round. */
    return rollout_compute_leaf_value_impl(
        r, board, n_board_cards, street, pot_chips,
        active, num_players, stacks, bets,
        player_hands, acting_order, acting_order_len,
        player_seats, game_action_history, gah_len,
        bias_profile, hero_player, hero_hand_idx,
        num_rollouts, master_seed,
        leaf_idx_for_seed, combo_for_seed,
        /*skip_first_betting=*/0,
        out_value);
}

/* Single-hand entry WITH the skip_first_betting flag (subgame leaf path).
 * Purely additive — mirrors rollout_compute_leaf_values_all_hands_c's flag.
 * A depth-limited subgame leaf already represents end-of-leaf-street betting
 * state (its game_action_history carries the closed flop_history); the rollout
 * must NOT re-run that betting round (which would double-count chips and
 * corrupt subsequent-street blueprint-query hashing). The ES re-solver oracle
 * (es_real_oracle.c) calls this with skip_first_betting=1 whenever the leaf's
 * flop_history is non-empty. See rollout.h:182-188. */
BPS_EXPORT int32_t rollout_compute_leaf_value_skip_c(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    int32_t pot_chips,
    const int32_t *active, int32_t num_players,
    const int32_t *stacks,
    const int32_t *bets,
    const RolloutHandList *player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    const int32_t *bias_profile,
    int32_t hero_player,
    int32_t hero_hand_idx,
    int32_t num_rollouts,
    uint64_t master_seed,
    int32_t leaf_idx_for_seed,
    int32_t combo_for_seed,
    int32_t skip_first_betting,
    float *out_value
) {
    return rollout_compute_leaf_value_impl(
        r, board, n_board_cards, street, pot_chips,
        active, num_players, stacks, bets,
        player_hands, acting_order, acting_order_len,
        player_seats, game_action_history, gah_len,
        bias_profile, hero_player, hero_hand_idx,
        num_rollouts, master_seed,
        leaf_idx_for_seed, combo_for_seed,
        skip_first_betting,
        out_value);
}

BPS_EXPORT int32_t rollout_compute_leaf_values_all_hands_c(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    int32_t pot_chips,
    const int32_t *active, int32_t num_players,
    const int32_t *stacks,
    const int32_t *bets,
    const RolloutHandList *player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    const int32_t *bias_profile,
    int32_t max_hands,
    int32_t num_rollouts,
    uint64_t master_seed,
    int32_t leaf_idx_for_seed,
    int32_t combo_for_seed,
    int32_t skip_first_betting,
    float *out
) {
    /* Zero-init */
    memset(out, 0, (size_t)num_players * (size_t)max_hands * sizeof(float));

    uint64_t board_mask = 0ULL;
    for (int32_t i = 0; i < n_board_cards; i++) board_mask |= (1ULL << board[i]);

    for (int32_t p = 0; p < num_players; p++) {
        if (!active[p]) continue;
        const RolloutHandList *hl = &player_hands[p];
        int32_t n = hl->n_hands < max_hands ? hl->n_hands : max_hands;
        for (int32_t h = 0; h < n; h++) {
            const RolloutHand *hd = &hl->hands[h];
            if (hd->weight <= 0.0f) continue;
            if (board_mask & (1ULL << hd->c0)) continue;
            if (board_mask & (1ULL << hd->c1)) continue;

            float v;
            int rc = rollout_compute_leaf_value_impl(
                r, board, n_board_cards, street, pot_chips,
                active, num_players, stacks, bets,
                player_hands, acting_order, acting_order_len,
                player_seats, game_action_history, gah_len,
                bias_profile, p, h, num_rollouts, master_seed,
                leaf_idx_for_seed, combo_for_seed,
                skip_first_betting, &v);
            if (rc != 0) return -1;
            out[p * max_hands + h] = v;
        }
    }
    return 0;
}

BPS_EXPORT int32_t rollout_compute_subgame_leaf_values_c(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    const RolloutLeafInfo *leaf_infos, int32_t n_leaves,
    const int32_t *active, int32_t num_players,
    const RolloutHandList *player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    int32_t stack_chips,
    int32_t max_hands,
    int32_t num_rollouts,
    uint64_t master_seed,
    float *out, int32_t out_size
) {
    /* Compute cont_per_leaf = NUM_BIAS_PROFILES^num_players */
    int64_t cont_per_leaf = 1;
    for (int32_t i = 0; i < num_players; i++) {
        cont_per_leaf *= ROLLOUT_NUM_BIAS_PROFILES;
    }

    /* total = (max_leaf_idx + 1) * cont_per_leaf */
    int32_t max_leaf_idx = 0;
    for (int32_t i = 0; i < n_leaves; i++) {
        if (leaf_infos[i].leaf_idx > max_leaf_idx) {
            max_leaf_idx = leaf_infos[i].leaf_idx;
        }
    }
    int64_t total = (int64_t)(max_leaf_idx + 1) * cont_per_leaf;
    int64_t floats_needed = total * (int64_t)num_players * (int64_t)max_hands;

    if (floats_needed > (int64_t)out_size) {
        rollout_set_error(street, num_players,
            "out buffer too small: need %lld floats, got %d",
            (long long)floats_needed, out_size);
        return -1;
    }

    memset(out, 0, (size_t)floats_needed * sizeof(float));

    /* OpenMP parallelization over (leaf × combo). The RNG is per-tuple
     * splitmix64 with no shared state; `out` is written to disjoint
     * per-(leaf_idx, combo) regions by construction (flat indexing in
     * rollout_compute_leaf_values_all_hands_c).
     *
     * Error handling: rollout_set_error writes to thread-local storage
     * inside worker threads, so we cannot surface the reason through the
     * caller's TLS slot. First-failure capture pattern:
     *   - `g_fail` is set atomically on the first rc != 0.
     *   - Remaining iterations short-circuit via the flag check.
     *   - The worker that captured the first failure copies its TLS error
     *     into a shared buffer before exiting, so the caller's
     *     rollout_get_last_error sees a meaningful reason.
     */
    volatile int g_fail = 0;
    /* Error-capture buffer, protected by a critical section. */
    char cap_reason[512] = {0};
    int32_t cap_street = street;
    int32_t cap_num_players = num_players;

    /* Parallelize across the full (leaf × combo) cartesian product.
     * collapse(2) requires perfectly-nested loops, so stack computation
     * moves inside. `g_fail` guards short-circuit; break is illegal in
     * OpenMP `for` body so we use continue.
     *
     * Typical workloads at 2p have n_leaves small (1-few) but
     * cont_per_leaf = 16; at 3p cont_per_leaf = 64; at 6p it's 4096.
     * Inner combo loop carries the real fanout — collapse is essential. */
    #pragma omp parallel for collapse(2) schedule(dynamic, 1) default(none) \
        shared(out, leaf_infos, n_leaves, cont_per_leaf, stack_chips, \
                num_players, max_hands, num_rollouts, master_seed, \
                r, board, n_board_cards, street, active, player_hands, \
                acting_order, acting_order_len, player_seats, \
                game_action_history, gah_len, g_fail, \
                cap_reason, cap_street, cap_num_players)
    for (int32_t li_idx = 0; li_idx < n_leaves; li_idx++) {
        for (int64_t combo = 0; combo < cont_per_leaf; combo++) {
            if (g_fail) continue;

            const RolloutLeafInfo *li = &leaf_infos[li_idx];
            int32_t stacks[ROLLOUT_MAX_PLAYERS];
            for (int32_t p = 0; p < num_players; p++) {
                int32_t s = stack_chips - li->bets[p];
                stacks[p] = s < 0 ? 0 : s;
            }

            int32_t bias_profile[ROLLOUT_MAX_PLAYERS];
            int64_t c = combo;
            for (int32_t p = 0; p < num_players; p++) {
                bias_profile[p] = (int32_t)(c % ROLLOUT_NUM_BIAS_PROFILES);
                c /= ROLLOUT_NUM_BIAS_PROFILES;
            }

            int64_t flat = (int64_t)li->leaf_idx * cont_per_leaf + combo;
            float *slot = out + flat * (int64_t)num_players * (int64_t)max_hands;

            /* Per-leaf game history: [passed_in_preflop, leaf->flop_history].
             * Without the leaf-specific flop slice the rollout's blueprint
             * queries during turn/river miss 100% (training stored those
             * nodes with [preflop, real_flop_actions, ...]) → uniform-on-
             * zero substitution → identical leaf values across hands → CFR
             * collapses to uniform 1/n. See PHASE0_STATUS.md (2026-04-30
             * diagnosis). */
            int32_t per_leaf_history[ROLLOUT_MAX_ACTION_HISTORY];
            int32_t fh_len = li->flop_history_len;
            if (fh_len < 0) fh_len = 0;
            if ((int64_t)gah_len + (int64_t)fh_len > ROLLOUT_MAX_ACTION_HISTORY) {
                #pragma omp critical(rollout_parallel_fail)
                {
                    if (!g_fail) {
                        snprintf(cap_reason, sizeof(cap_reason) - 1,
                                 "per-leaf history overflow: preflop=%d + flop=%d > %d",
                                 gah_len, fh_len, ROLLOUT_MAX_ACTION_HISTORY);
                        cap_street = street;
                        cap_num_players = num_players;
                        g_fail = 1;
                    }
                }
                continue;
            }
            for (int32_t k = 0; k < gah_len; k++) per_leaf_history[k] = game_action_history[k];
            for (int32_t k = 0; k < fh_len; k++) per_leaf_history[gah_len + k] = li->flop_history[k];
            int32_t per_leaf_history_len = gah_len + fh_len;

            /* Skip the leaf's first betting round when the leaf already
             * carries a closed flop_history — see rollout_simulate_one. */
            int32_t skip_first = (fh_len > 0) ? 1 : 0;
            int rc = rollout_compute_leaf_values_all_hands_c(
                r, board, n_board_cards, street, li->pot_chips,
                active, num_players, stacks, li->bets,
                player_hands, acting_order, acting_order_len,
                player_seats, per_leaf_history, per_leaf_history_len,
                bias_profile, max_hands, num_rollouts, master_seed,
                li->leaf_idx, (int32_t)combo, skip_first, slot);
            if (rc != 0) {
                #pragma omp critical(rollout_parallel_fail)
                {
                    if (!g_fail) {
                        const char *reason = NULL;
                        int32_t s_st = 0, s_np = 0;
                        rollout_get_last_error(&reason, &s_st, &s_np);
                        if (reason) {
                            strncpy(cap_reason, reason,
                                    sizeof(cap_reason) - 1);
                            cap_reason[sizeof(cap_reason) - 1] = '\0';
                            cap_street = s_st;
                            cap_num_players = s_np;
                        }
                        g_fail = 1;
                    }
                }
            }
        }
    }

    if (g_fail) {
        /* Re-raise into the caller's TLS slot so rollout_get_last_error
         * continues to work on the main thread. */
        rollout_set_error(cap_street, cap_num_players, "%s",
                          cap_reason[0] ? cap_reason
                                        : "rollout worker failed");
        return -1;
    }
    return 0;
}
