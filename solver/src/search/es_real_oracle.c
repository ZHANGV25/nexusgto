/*
 * es_real_oracle.c — REAL blueprint-rollout leaf oracle for the ES re-solver.
 *
 * The production hot-path leaf oracle: each ES traversal that reaches a betting
 * leaf computes a LAZY value via rollout_compute_leaf_value_c (the SAME C path
 * the GPU's leaf construct floor uses — but evaluated lazily on sampled lines,
 * not eagerly over all leaves). NO Python in the hot path (advisor).
 *
 * Per-leaf state (pot, bets, active, flop_history_train) is iteration-
 * independent, so it is extracted ONCE in Python via the proven
 * leaf_values.extract_leaf_info_from_tree (the load-bearing flop_history_train
 * mapping — re-porting it to C buys nothing and is the README's "100% miss ->
 * uniform 1/n" trap) and handed in as flat arrays. The board, per-player hand
 * lists, and num_players come from the SSTreeData the resolver already holds.
 *
 * FIDELITY (advisor, load-bearing for GPU-equivalence): hero = the traverser's
 * sampled hand; OPPONENTS = full ranges (the rollout marginalizes them
 * internally). sampled_hands is IGNORED at the leaf. This matches
 * real_oracle.py and is the definition of the depth-limited leaf value the GPU
 * leaf_values[leaf,player,hand] table holds. Fixing opponents to sampled hands
 * would compute a different quantity and move the equilibrium off the GPU's.
 */

#include "es_resolver.h"
#include "rollout.h"
#include "bps_reader.h"

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#define ESRO_MAX_FLOP_HIST ROLLOUT_MAX_ACTION_HISTORY

typedef struct {
    const BPSReader *bp;
    const SSTreeData *tree;       /* for hands + board */
    int np;
    int street;                   /* ROLLOUT_STREET_FLOP / TURN */
    int stack_chips;
    int num_rollouts;
    int num_samples_k;            /* cont-strat bias profiles averaged per call */
    uint64_t master_seed;

    /* per-leaf state, indexed by leaf_idx (length n_leaves) */
    int n_leaves;
    int *pot;                     /* [n_leaves] */
    int *bets;                    /* [n_leaves * np] */
    int *active;                  /* [n_leaves * np] */
    int *hist;                    /* [n_leaves * ESRO_MAX_FLOP_HIST] */
    int *hist_len;                /* [n_leaves] */

    int *player_seats;            /* [np] compact slot -> blueprint full seat */
    int *acting_order;            /* [np] identity 0..np-1 (rollout recomputes) */

    /* ── Leaf-value CACHE (the decisive perf fix) ─────────────────────────
     * The leaf value is STRATEGY-INDEPENDENT: opponents are marginalized over
     * the FIXED full ranges, bias is drawn from the uniform {0,1,2,3}
     * distribution, and the runout queries the FIXED blueprint — none of it
     * depends on the evolving ES regrets. So a leaf's value is a CONSTANT for
     * the whole solve. We compute it ONCE on first hit (all hands at once, via
     * the all-hands entry, averaged over cache_k bias samples × num_rollouts
     * for low variance) and cache value[leaf*np*max_h + p*max_h + h]; every
     * later visit is an O(1) read. Still LAZY (only reached leaves, once each —
     * no global construct floor over all leaves), EXACT in expectation (the
     * cached bias-averaged value is the E_bias[...] the per-visit-fresh-bias
     * path converges to, lower variance), and recovers the per-hand runout
     * amortization the single-hand path threw away.
     *
     * Thread-safe (Hogwild): cache_state[leaf] is 0=empty 1=computing 2=ready,
     * advanced via CAS. A loser of the CAS spins until ready (the runout is the
     * expensive part; the spin is brief and rare — each leaf computed once). */
    int max_h;
    int cache_k;                  /* bias samples averaged when filling a leaf */
    float *cache_val;             /* [n_leaves * np * max_h] */
    _Atomic(int) *cache_state;    /* [n_leaves]: 0 empty, 1 computing, 2 ready */

    /* observability + wall-cost instrumentation. atomic so Hogwild threads can
     * bump without a lock. */
    _Atomic(uint64_t) rollout_calls;    /* leaf FILLS (cache misses) */
    _Atomic(uint64_t) cache_hits;       /* O(1) reads */
    _Atomic(uint64_t) rollout_sims;     /* fills * cache_k * num_rollouts * np-hands */
    _Atomic(uint64_t) oracle_errors;
} ESRealOracleCtx;

/* Fill the WHOLE leaf (all players × all hands) into the cache exactly once.
 * Uses the all-hands rollout entry (amortizes the per-leaf runout across hands)
 * averaged over cache_k uniform-bias samples (the cont-strat chance node is
 * integrated by averaging the uniform bias distribution — the same E_bias[...]
 * the per-visit fresh-bias path converges to, with lower variance). Returns 0
 * on success. Caller holds the "computing" CAS for this leaf. */
static int esro_fill_leaf(ESRealOracleCtx *c, int leaf_idx) {
    const SSTreeData *t = c->tree;
    int np = c->np;
    int max_h = c->max_h;

    /* RolloutHand is 12 bytes (two int32_t plus one float), so this is
     * 6 * 1400 * 12 = 100,800 bytes (98.4 KiB), below the 256 KiB threshold;
     * keep it on the stack for this probe. */
    RolloutHand hand_buf[SS_MAX_PLAYERS][SS_MAX_HANDS];
    RolloutHandList hand_lists[SS_MAX_PLAYERS];
    int32_t active32[SS_MAX_PLAYERS], bets32[SS_MAX_PLAYERS], stacks32[SS_MAX_PLAYERS];
    for (int p = 0; p < np; p++) {
        int nh = t->num_hands[p];
        for (int h = 0; h < nh; h++) {
            hand_buf[p][h].c0 = t->hands[p][h][0];
            hand_buf[p][h].c1 = t->hands[p][h][1];
            hand_buf[p][h].weight = t->weights[p][h];
        }
        hand_lists[p].hands = hand_buf[p];
        hand_lists[p].n_hands = nh;
        active32[p] = c->active[leaf_idx * np + p];
        bets32[p] = c->bets[leaf_idx * np + p];
        int b = bets32[p];
        stacks32[p] = (c->stack_chips - b > 0) ? (c->stack_chips - b) : 0;
    }
    int board[SS_MAX_BOARD];
    for (int b = 0; b < t->num_board; b++) board[b] = t->board[b];
    const int32_t *hist = (const int32_t *)&c->hist[leaf_idx * ESRO_MAX_FLOP_HIST];
    int32_t hist_len = c->hist_len[leaf_idx];
    int32_t pot = c->pot[leaf_idx];
    int32_t skip_first = (hist_len > 0) ? 1 : 0;  /* see rollout.h:182-188 */

    /* accumulate all-hands output over cache_k bias samples */
    float *acc = (float *)calloc((size_t)np * max_h, sizeof(float));
    if (!acc) return -1;
    float *scratch = (float *)malloc((size_t)np * max_h * sizeof(float));
    if (!scratch) { free(acc); return -1; }
    int got = 0;
    /* deterministic per-leaf bias stream (cache fill is strategy-independent;
     * a fixed per-leaf seed keeps the cached value reproducible). */
    uint64_t prng = c->master_seed
                    ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(leaf_idx + 1));
    for (int k = 0; k < c->cache_k; k++) {
        int32_t bias[SS_MAX_PLAYERS];
        for (int p = 0; p < np; p++) {
            prng ^= prng << 13; prng ^= prng >> 7; prng ^= prng << 17;
            bias[p] = (int32_t)(prng % (uint64_t)ROLLOUT_NUM_BIAS_PROFILES);
        }
        prng ^= prng << 13; prng ^= prng >> 7; prng ^= prng << 17;
        uint64_t rseed = prng ^ (0x100000001B3ULL * (uint64_t)(k + 1));
        memset(scratch, 0, (size_t)np * max_h * sizeof(float));
        int rc = rollout_compute_leaf_values_all_hands_c(
            c->bp, board, t->num_board, c->street, pot,
            active32, np, stacks32, bets32, hand_lists,
            c->acting_order, np, c->player_seats, hist, hist_len,
            bias, max_h, c->num_rollouts, rseed,
            leaf_idx, (int32_t)k, skip_first, scratch);
        if (rc == 0) {
            for (int i = 0; i < np * max_h; i++) acc[i] += scratch[i];
            got++;
        } else {
            atomic_fetch_add(&c->oracle_errors, 1);
        }
    }
    float inv = (got > 0) ? (1.0f / (float)got) : 0.0f;
    float *dst = &c->cache_val[(size_t)leaf_idx * np * max_h];
    for (int i = 0; i < np * max_h; i++) dst[i] = acc[i] * inv;
    free(acc); free(scratch);
    atomic_fetch_add(&c->rollout_calls, 1);  /* one leaf FILL (cache miss) */
    atomic_fetch_add(&c->rollout_sims,
                     (uint64_t)c->cache_k * (uint64_t)c->num_rollouts);
    return (got > 0) ? 0 : -1;
}

static double esro_oracle(void *vctx, int leaf_idx, int traverser,
                          int traverser_hand, const int *sampled_hands,
                          const SSNode *leaf_node, uint64_t *rng) {
    (void)sampled_hands;  /* IGNORED at leaf — opponents marginalized (fidelity) */
    (void)leaf_node; (void)rng;  /* cached value is strategy-independent */
    ESRealOracleCtx *c = (ESRealOracleCtx *)vctx;
    int np = c->np;

    if (leaf_idx < 0 || leaf_idx >= c->n_leaves) return 0.0;
    /* hero not active at this leaf -> they folded; lose their bet. */
    if (!c->active[leaf_idx * np + traverser])
        return -(double)c->bets[leaf_idx * np + traverser];

    /* Cache lookup (the value is a constant for the whole solve). */
    int st = atomic_load(&c->cache_state[leaf_idx]);
    if (st != 2) {
        int expected = 0;
        if (atomic_compare_exchange_strong(&c->cache_state[leaf_idx], &expected, 1)) {
            /* we own the fill */
            esro_fill_leaf(c, leaf_idx);
            atomic_store(&c->cache_state[leaf_idx], 2);
        } else {
            /* another thread is filling (or already filled) — spin to ready.
             * The fill is the expensive part; this spin is brief and each leaf
             * is filled exactly once. */
            while (atomic_load(&c->cache_state[leaf_idx]) != 2) { /* busy-wait */ }
        }
    } else {
        atomic_fetch_add(&c->cache_hits, 1);
    }
    return (double)c->cache_val[((size_t)leaf_idx * np + traverser) * c->max_h
                                + traverser_hand];
}

/* ── C-ABI: build / free / query ─────────────────────────────────────────── */

ES_EXPORT void *es_make_real_oracle(
    const void *bp_void,
    const SSTreeData *tree,
    int street,                    /* ROLLOUT_STREET_FLOP=1 / TURN=2 */
    int stack_chips,
    int num_rollouts,
    int cache_k,                   /* bias samples averaged per cached leaf fill */
    uint64_t master_seed,
    int n_leaves,
    int max_h,                     /* cache stride: max hands per player */
    const int *pot,                /* [n_leaves] */
    const int *bets,               /* [n_leaves * np] */
    const int *active,             /* [n_leaves * np] */
    const int *hist_flat,          /* [n_leaves * ESRO_MAX_FLOP_HIST] */
    const int *hist_len,           /* [n_leaves] */
    const int *player_seats        /* [np] */
) {
    ESRealOracleCtx *c = (ESRealOracleCtx *)calloc(1, sizeof(ESRealOracleCtx));
    if (!c) return NULL;
    int np = tree->num_players;
    c->bp = (const BPSReader *)bp_void; c->tree = tree; c->np = np;
    c->street = street; c->stack_chips = stack_chips;
    c->num_rollouts = num_rollouts > 0 ? num_rollouts : ROLLOUT_NUM_ROLLOUTS_DEFAULT;
    c->cache_k = cache_k > 0 ? cache_k : 1;
    c->num_samples_k = c->cache_k;  /* kept for stats compatibility */
    c->master_seed = master_seed ? master_seed : 0xABCDEF0123456789ULL;
    c->n_leaves = n_leaves;
    c->max_h = max_h;

    /* max_h is the cache stride (hands per player). If it were < num_hands[p]
     * for any player, the all-hands fill clamps to max_h (rollout.c) and the
     * leaf-table readout's [.., :num_hands] slice would read past the stride
     * into the next player's row → silently wrong BR-gap (review 2026-07-12 5b).
     * The caller contract is max_h >= max_p num_hands[p]; enforce it. */
    for (int _p = 0; _p < np; _p++) {
        if (max_h < tree->num_hands[_p]) {
            free(c);
            return NULL;  /* caller: pass max_h >= max num_hands per player */
        }
    }

    c->pot = (int *)malloc((size_t)n_leaves * sizeof(int));
    c->bets = (int *)malloc((size_t)n_leaves * np * sizeof(int));
    c->active = (int *)malloc((size_t)n_leaves * np * sizeof(int));
    c->hist = (int *)malloc((size_t)n_leaves * ESRO_MAX_FLOP_HIST * sizeof(int));
    c->hist_len = (int *)malloc((size_t)n_leaves * sizeof(int));
    c->player_seats = (int *)malloc((size_t)np * sizeof(int));
    c->acting_order = (int *)malloc((size_t)np * sizeof(int));
    c->cache_val = (float *)calloc((size_t)n_leaves * np * max_h, sizeof(float));
    c->cache_state = (_Atomic(int) *)calloc((size_t)n_leaves, sizeof(_Atomic(int)));
    if (!c->pot || !c->bets || !c->active || !c->hist || !c->hist_len
        || !c->player_seats || !c->acting_order || !c->cache_val || !c->cache_state) {
        es_free_real_oracle(c);
        return NULL;
    }
    memcpy(c->pot, pot, (size_t)n_leaves * sizeof(int));
    memcpy(c->bets, bets, (size_t)n_leaves * np * sizeof(int));
    memcpy(c->active, active, (size_t)n_leaves * np * sizeof(int));
    memcpy(c->hist, hist_flat, (size_t)n_leaves * ESRO_MAX_FLOP_HIST * sizeof(int));
    memcpy(c->hist_len, hist_len, (size_t)n_leaves * sizeof(int));
    memcpy(c->player_seats, player_seats, (size_t)np * sizeof(int));
    for (int p = 0; p < np; p++) c->acting_order[p] = p;
    for (int i = 0; i < n_leaves; i++) atomic_store(&c->cache_state[i], 0);
    atomic_store(&c->rollout_calls, 0);
    atomic_store(&c->cache_hits, 0);
    atomic_store(&c->rollout_sims, 0);
    atomic_store(&c->oracle_errors, 0);
    return c;
}

ES_EXPORT ESLeafOracle es_real_oracle_fn(void) { return esro_oracle; }

ES_EXPORT void es_real_oracle_stats(void *ctx, uint64_t *calls,
                                    uint64_t *sims, uint64_t *errors) {
    ESRealOracleCtx *c = (ESRealOracleCtx *)ctx;
    if (!c) return;
    if (calls) *calls = atomic_load(&c->rollout_calls);
    if (sims) *sims = atomic_load(&c->rollout_sims);
    if (errors) *errors = atomic_load(&c->oracle_errors);
}

ES_EXPORT void es_free_real_oracle(void *ctx) {
    ESRealOracleCtx *c = (ESRealOracleCtx *)ctx;
    if (!c) return;
    free(c->pot); free(c->bets); free(c->active);
    free(c->hist); free(c->hist_len);
    free(c->player_seats); free(c->acting_order);
    free(c->cache_val); free((void *)c->cache_state);
    free(c);
}

/* Extended stats: also expose cache hits (calls = leaf fills/misses). */
ES_EXPORT void es_real_oracle_stats2(void *ctx, uint64_t *fills,
                                     uint64_t *cache_hits, uint64_t *sims,
                                     uint64_t *errors) {
    ESRealOracleCtx *c = (ESRealOracleCtx *)ctx;
    if (!c) return;
    if (fills) *fills = atomic_load(&c->rollout_calls);
    if (cache_hits) *cache_hits = atomic_load(&c->cache_hits);
    if (sims) *sims = atomic_load(&c->rollout_sims);
    if (errors) *errors = atomic_load(&c->oracle_errors);
}

/* Cache dimensions, so a caller can size a [n_leaves, np, max_h] buffer before
 * reading the leaf-value table out. Any out-param may be NULL. */
ES_EXPORT void es_real_oracle_dims(void *ctx, int *n_leaves, int *np,
                                   int *max_h) {
    ESRealOracleCtx *c = (ESRealOracleCtx *)ctx;
    if (!c) return;
    if (n_leaves) *n_leaves = c->n_leaves;
    if (np)       *np       = c->np;
    if (max_h)    *max_h    = c->max_h;
}

/* Read the DENSE leaf-value table out as a flat float[n_leaves*np*max_h] array
 * in the exact [leaf, player, hand] layout `cache_val` already holds — the same
 * layout the BR-gap best-responder (python/es_resolver/exploitability_c.py)
 * indexes as table[leaf, player, hand]. Returns `need` (the element count) on
 * success, or `need` (WITHOUT copying) when out==NULL so the caller can size
 * its buffer first, or -1 if out is non-NULL but out_len < need.
 *
 * ⭐ FILL-ALL (the whole reason this exists): during the solve the cache is
 * filled LAZILY — only leaves the sampled CFR walk VISITED are cache_state==2;
 * unvisited leaves are still 0. But the BR-gap walk reads EVERY leaf, so a raw
 * memcpy would hand back 0 for every unreached leaf and corrupt the gap. So we
 * first fill any not-ready leaf here, exactly as esro_oracle's lazy path does
 * (esro_fill_leaf is deterministic + strategy-independent — bias-averaged over
 * the fixed {0..3} distribution — so a post-solve fill yields the identical
 * value a visited leaf would hold). This runs ONLY at readout (expose-gated,
 * post-solve, single-threaded — no Hogwild contention), NEVER in the hot path,
 * so the shipped lazy-once-per-visit behavior is unchanged.
 *
 * Thread-safety: intended for a post-solve, single-threaded call. We still use
 * the CAS/spin fill discipline so a stray concurrent reader can't double-fill;
 * a leaf already ready (state 2) is left untouched. */
ES_EXPORT int es_real_oracle_get_leaf_table(void *ctx, float *out,
                                            int out_len) {
    ESRealOracleCtx *c = (ESRealOracleCtx *)ctx;
    if (!c || !c->cache_val) return -1;
    /* size in elements; guard the int multiply against overflow on absurd trees */
    long long need_ll = (long long)c->n_leaves * (long long)c->np
                        * (long long)c->max_h;
    if (need_ll <= 0 || need_ll > 0x7fffffffLL) return -1;
    int need = (int)need_ll;
    if (out == NULL) return need;          /* size query */
    if (out_len < need) return -1;         /* caller buffer too small */

    /* FILL-ALL: bring every not-ready leaf to state 2 before the copy.
     * ⭐ Unlike the hot-path lazy fill (which ignores the rc — a rare failed
     * leaf there is a tolerable degraded serve), the READOUT must NOT hand the
     * best-responder a leaf that failed to fill: esro_fill_leaf returns -1 on
     * OOM or all-rollouts-errored, in which case dst[]==0.0 and a zeroed leaf is
     * indistinguishable from a real zero value → silently biases the BR-gap
     * toward 0. So track any fill failure and fail the WHOLE readout loud
     * (return -2) rather than copy a partially-zeroed table (review 2026-07-12
     * 5a). A leaf already ready (state 2) from the solve is trusted. */
    int fill_failed = 0;
    for (int leaf_idx = 0; leaf_idx < c->n_leaves; leaf_idx++) {
        int st = atomic_load(&c->cache_state[leaf_idx]);
        if (st != 2) {
            int expected = 0;
            if (atomic_compare_exchange_strong(
                    &c->cache_state[leaf_idx], &expected, 1)) {
                if (esro_fill_leaf(c, leaf_idx) != 0) fill_failed = 1;
                atomic_store(&c->cache_state[leaf_idx], 2);
            } else {
                /* someone else is filling (or already did) — wait for ready */
                while (atomic_load(&c->cache_state[leaf_idx]) != 2) { /* spin */ }
            }
        }
    }
    if (fill_failed) return -2;  /* a leaf failed to fill; do NOT return zeros */
    memcpy(out, c->cache_val, (size_t)need * sizeof(float));
    return need;
}
