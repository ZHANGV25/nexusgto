/*
 * es_vec_sampled_oracle.c - vec-coherent per-visit sampled leaf oracle.
 *
 * One vector leaf visit = one world: the opponents' continuation biases are
 * sampled ONCE from their persistent regret rows, the turn/river runout is
 * (optionally) drawn ONCE, and every live traverser hand is valued in that
 * world with its own regret-matched bias. The world is a deterministic
 * function of the persistent store and the traversal RNG stream, so a
 * single-thread solve is bit-reproducible and all hands of one visit share
 * the same opponent realization. Contrast es_sampled_oracle.c, which draws a
 * fresh world per hand and is therefore scalar-only.
 *
 * Per hand, the traverser's four continuation biases are either all
 * evaluated (ENUMERATE: full external-sampling regret update, 4 rollout
 * batches, common random numbers across the four) or one is sampled from
 * the epsilon-explored regret-matched mixture (SAMPLE: importance-weighted
 * outcome-sampling regret update, 1 rollout batch). Both estimators are
 * unbiased for the regret and for the returned mixture value.
 */

#include "es_vec_sampled_oracle.h"
#include "rollout_internal.h"
#include "rollout_bucket_memo.h"

#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <time.h>
#endif

#define ESVS_MAX_HIST ROLLOUT_MAX_ACTION_HISTORY
#define ESVS_NB ROLLOUT_NUM_BIAS_PROFILES
#define ESVS_PINNED_CARDS 4   /* turn + river + one spare per hero hole card */

typedef struct {
    const BPSReader *bp;
    const SSTreeData *tree;
    int np;
    int street;
    int stack_chips;
    int num_rollouts;
    int hero_bias_mode;
    int runout_mode;
    float explore_eps;
    int opp_bias_mode;     /* ES_VS_OPP_BIAS_*; 0 (REGRET) after calloc */
    int hero_bias_fixed;   /* 1: traverser plays bias 0, no regret update */

    int n_leaves;
    int max_h;
    int *pot;             /* [n_leaves] */
    int *bets;            /* [n_leaves * np] */
    int *active;          /* [n_leaves * np] */
    int *hist;            /* [n_leaves * ESVS_MAX_HIST] */
    int *hist_len;        /* [n_leaves] */
    int *player_seats;    /* [np] */
    int *acting_order;    /* [np], identity */

    /* Dense [leaf][player][hand][bias] continuation-bias regrets. Plain
     * float Hogwild additions, exactly like the resolver's own regrets. */
    float *bias_regret;

    _Atomic(uint64_t) leaf_visits;
    _Atomic(uint64_t) hand_visits;
    _Atomic(uint64_t) rollouts;
    _Atomic(uint64_t) oracle_errors;
    _Atomic(uint64_t) oracle_ns;
} ESVecSampledCtx;

/* One visit's fixed world, shared by every hand valued in that visit. */
typedef struct {
    int leaf_idx;
    int traverser;
    int32_t active[SS_MAX_PLAYERS];
    int32_t bets[SS_MAX_PLAYERS];
    int32_t stacks[SS_MAX_PLAYERS];
    int32_t c0[SS_MAX_PLAYERS];
    int32_t c1[SS_MAX_PLAYERS];
    int32_t has_sample[SS_MAX_PLAYERS];
    int32_t bias_profile[SS_MAX_PLAYERS];
    int32_t pinned[ESVS_PINNED_CARDS];
    int32_t pinned_len;
    uint64_t visit_seed;
} ESVSVisit;

static inline uint64_t esvs_now_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    if (!QueryPerformanceFrequency(&f) || !QueryPerformanceCounter(&c))
        return 0;
    return (uint64_t)((double)c.QuadPart * 1e9 / (double)f.QuadPart);
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/* Same xorshift64 step as the scalar ES resolver: every draw here advances
 * the caller's per-traversal stream. */
static inline uint64_t esvs_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline double esvs_rng_double(uint64_t *state) {
    return (double)(esvs_rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

/* Per-hand rollout stream: a splitmix mix of the visit seed and the hand
 * index, so hand h's draws never depend on how many draws hands < h used.
 * Never zero (xorshift64 would stick at zero). */
static inline uint64_t esvs_hand_stream(uint64_t visit_seed, int hand,
                                        uint64_t salt) {
    uint64_t s = visit_seed ^ salt
                 ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(hand + 1));
    uint64_t z = rollout_splitmix64_next(&s);
    return z ? z : 0xdeadbeefcafebabeULL;
}

static void esvs_regret_match(const float *row, float out[ESVS_NB]) {
    float sum = 0.0f;
    for (int b = 0; b < ESVS_NB; b++) {
        float positive = row[b] > 0.0f ? row[b] : 0.0f;
        out[b] = positive;
        sum += positive;
    }
    if (sum > 0.0f) {
        for (int b = 0; b < ESVS_NB; b++) out[b] /= sum;
    } else {
        for (int b = 0; b < ESVS_NB; b++) out[b] = 1.0f / (float)ESVS_NB;
    }
}

static int esvs_sample_from(const float strategy[ESVS_NB], double u) {
    double cumulative = 0.0;
    for (int b = 0; b < ESVS_NB; b++) {
        cumulative += (double)strategy[b];
        if (u < cumulative) return b;
    }
    return ESVS_NB - 1;
}

static inline float *esvs_regret_row(ESVecSampledCtx *c, int leaf_idx,
                                     int player, int hand) {
    size_t row = ((size_t)leaf_idx * (size_t)c->np + (size_t)player)
                 * (size_t)c->max_h + (size_t)hand;
    return &c->bias_regret[row * ESVS_NB];
}

/* Draw the visit's world: leaf state, opponents' cards + regret-sampled
 * biases, the pinned runout. Consumes the traversal RNG. Returns -1 on a
 * malformed input (counted as an oracle error by the caller). */
static int esvs_visit_setup(ESVecSampledCtx *c, int leaf_idx, int traverser,
                            const int *opp_sampled, uint64_t *rng,
                            ESVSVisit *v) {
    const SSTreeData *t = c->tree;
    size_t base = (size_t)leaf_idx * (size_t)c->np;
    v->leaf_idx = leaf_idx;
    v->traverser = traverser;
    uint64_t used = 0ULL;
    for (int b = 0; b < t->num_board; b++) used |= 1ULL << t->board[b];
    for (int p = 0; p < c->np; p++) {
        v->active[p] = c->active[base + (size_t)p];
        v->bets[p] = c->bets[base + (size_t)p];
        v->stacks[p] = c->stack_chips > v->bets[p]
                       ? c->stack_chips - v->bets[p] : 0;
        v->c0[p] = 0; v->c1[p] = 0; v->has_sample[p] = 0;
        v->bias_profile[p] = 0;
        if (p == traverser) continue;
        int hand = opp_sampled ? opp_sampled[p] : -1;
        if (hand < 0) continue;
        if (hand >= t->num_hands[p]) return -1;
        v->c0[p] = t->hands[p][hand][0];
        v->c1[p] = t->hands[p][hand][1];
        v->has_sample[p] = 1;
        used |= 1ULL << v->c0[p];
        used |= 1ULL << v->c1[p];
        /* Opponent continuation bias: one regret-matched sample per visit,
         * held fixed for every traverser hand valued in this visit. */
        if (v->active[p]) {
            if (c->opp_bias_mode == ES_VS_OPP_BIAS_UNBIASED) {
                v->bias_profile[p] = 0;
            } else if (c->opp_bias_mode == ES_VS_OPP_BIAS_UNIFORM) {
                v->bias_profile[p] =
                    (int32_t)(esvs_rng_next(rng) % (uint64_t)ESVS_NB);
            } else {
                float strategy[ESVS_NB];
                esvs_regret_match(esvs_regret_row(c, leaf_idx, p, hand),
                                  strategy);
                v->bias_profile[p] =
                    esvs_sample_from(strategy, esvs_rng_double(rng));
            }
        }
    }
    v->pinned_len = 0;
    if (c->runout_mode == ES_VS_RUNOUT_SHARED) {
        /* Partial Fisher-Yates over the unused deck: a uniform ordered
         * 4-card sequence from (deck - board - opponents' cards). */
        int32_t unused[52];
        int n = 0;
        for (int card = 0; card < 52; card++)
            if (!(used & (1ULL << card))) unused[n++] = card;
        for (int k = 0; k < ESVS_PINNED_CARDS && k < n; k++) {
            int j = k + (int)(esvs_rng_next(rng) % (uint64_t)(n - k));
            int32_t tmp = unused[k]; unused[k] = unused[j]; unused[j] = tmp;
            v->pinned[k] = unused[k];
            v->pinned_len++;
        }
    }
    v->visit_seed = esvs_rng_next(rng);
    return 0;
}

/* num_rollouts playouts with a fixed joint bias profile in the visit's world
 * for the traverser's hand (already installed in v->c0/c1/has_sample). */
static int esvs_batch(ESVecSampledCtx *c, ESVSVisit *v, uint64_t stream,
                      float *out_value) {
    const SSTreeData *t = c->tree;
    int board[SS_MAX_BOARD];
    for (int b = 0; b < t->num_board; b++) board[b] = t->board[b];
    const int32_t *hist =
        (const int32_t *)&c->hist[(size_t)v->leaf_idx * ESVS_MAX_HIST];
    int32_t hist_len = c->hist_len[v->leaf_idx];
    int32_t skip_first = hist_len > 0 ? 1 : 0;
    uint64_t rng = stream;
    double sum = 0.0;
    for (int r = 0; r < c->num_rollouts; r++) {
        float payoff = 0.0f;
        atomic_fetch_add_explicit(&c->rollouts, 1, memory_order_relaxed);
        int rc = rollout_simulate_one_ex(
            c->bp, board, t->num_board, c->street, c->pot[v->leaf_idx],
            v->active, v->stacks, v->bets, c->np,
            (const int32_t *)c->acting_order, c->np,
            (const int32_t *)c->player_seats,
            hist, hist_len, v->bias_profile, v->traverser,
            v->c0, v->c1, v->has_sample, skip_first,
            v->pinned_len > 0 ? v->pinned : NULL, v->pinned_len,
            &rng, &payoff);
        if (rc != 0) {
            atomic_fetch_add_explicit(&c->oracle_errors, 1,
                                      memory_order_relaxed);
            return -1;
        }
        sum += (double)payoff;
    }
    *out_value = (float)(sum / (double)c->num_rollouts);
    return 0;
}

/* Value one live traverser hand in the visit's world and update its bias
 * regret row. Returns the mixture (ENUMERATE) or its unbiased estimate
 * (SAMPLE); 0.0 with an error count on a rollout failure. */
static double esvs_eval_hand(ESVecSampledCtx *c, ESVSVisit *v, int hand) {
    const SSTreeData *t = c->tree;
    int trav = v->traverser;
    atomic_fetch_add_explicit(&c->hand_visits, 1, memory_order_relaxed);
    if (!v->active[trav]) return -(double)v->bets[trav];
    v->c0[trav] = t->hands[trav][hand][0];
    v->c1[trav] = t->hands[trav][hand][1];
    v->has_sample[trav] = 1;

    if (c->hero_bias_fixed) {
        /* Research A/B: the unbiased continuation, no bias regret update. */
        v->bias_profile[trav] = 0;
        float fixed_value = 0.0f;
        uint64_t fixed_stream = esvs_hand_stream(v->visit_seed, hand, 0x1234ULL);
        if (esvs_batch(c, v, fixed_stream, &fixed_value) != 0) return 0.0;
        return (double)fixed_value;
    }

    float *row = esvs_regret_row(c, v->leaf_idx, trav, hand);
    float sigma[ESVS_NB];
    esvs_regret_match(row, sigma);

    if (c->hero_bias_mode == ES_VS_HERO_BIAS_ENUMERATE) {
        /* Common random numbers: every bias replays the same stream, so
         * the four values differ only where the bias changes an action. */
        uint64_t stream = esvs_hand_stream(v->visit_seed, hand, 0x1234ULL);
        float values[ESVS_NB];
        for (int b = 0; b < ESVS_NB; b++) {
            v->bias_profile[trav] = b;
            if (esvs_batch(c, v, stream, &values[b]) != 0) return 0.0;
        }
        float mixture = 0.0f;
        for (int b = 0; b < ESVS_NB; b++) mixture += sigma[b] * values[b];
        for (int b = 0; b < ESVS_NB; b++) row[b] += values[b] - mixture;
        return (double)mixture;
    }

    /* SAMPLE: b ~ q = (1-eps) sigma + eps/4; one batch; importance-weighted
     * outcome-sampling estimates. E[value] = sum_b sigma_b v_b and
     * E[regret(a)] = v_a - sum_b sigma_b v_b, both exact over the draw. */
    float q[ESVS_NB];
    float eps = c->explore_eps;
    for (int b = 0; b < ESVS_NB; b++)
        q[b] = (1.0f - eps) * sigma[b] + eps / (float)ESVS_NB;
    uint64_t pick = esvs_hand_stream(v->visit_seed, hand, 0x5678ULL);
    double u = (double)(pick >> 11) * (1.0 / 9007199254740992.0);
    int b = esvs_sample_from(q, u);
    v->bias_profile[trav] = b;
    float value = 0.0f;
    uint64_t stream = esvs_hand_stream(v->visit_seed, hand, 0x1234ULL);
    if (esvs_batch(c, v, stream, &value) != 0) return 0.0;
    float weighted = q[b] > 0.0f ? value / q[b] : 0.0f;
    for (int a = 0; a < ESVS_NB; a++)
        row[a] += ((a == b ? 1.0f : 0.0f) - sigma[b]) * weighted;
    return (double)(sigma[b] * weighted);
}

static int esvs_check_args(const ESVecSampledCtx *c, int leaf_idx,
                           int traverser) {
    return c && leaf_idx >= 0 && leaf_idx < c->n_leaves
           && traverser >= 0 && traverser < c->np;
}

static void esvs_oracle_vec(void *vctx, int leaf_idx, int traverser,
                            const int *opp_sampled, const SSNode *leaf_node,
                            uint64_t *rng, int num_hands, const int *dead,
                            double *out_vals) {
    (void)leaf_node;
    ESVecSampledCtx *c = (ESVecSampledCtx *)vctx;
    for (int h = 0; h < num_hands; h++) out_vals[h] = 0.0;
    if (!c) return;
    uint64_t t0 = esvs_now_ns();
    atomic_fetch_add_explicit(&c->leaf_visits, 1, memory_order_relaxed);
    if (!rng || !esvs_check_args(c, leaf_idx, traverser)
        || num_hands > c->tree->num_hands[traverser]) {
        atomic_fetch_add_explicit(&c->oracle_errors, 1, memory_order_relaxed);
        return;
    }
    ESVSVisit v;
    if (esvs_visit_setup(c, leaf_idx, traverser, opp_sampled, rng, &v) != 0) {
        atomic_fetch_add_explicit(&c->oracle_errors, 1, memory_order_relaxed);
        return;
    }
    for (int h = 0; h < num_hands; h++) {
        if (dead && dead[h]) continue;
        out_vals[h] = esvs_eval_hand(c, &v, h);
    }
    atomic_fetch_add_explicit(&c->oracle_ns, esvs_now_ns() - t0,
                              memory_order_relaxed);
}

static double esvs_oracle_scalar(void *vctx, int leaf_idx, int traverser,
                                 int traverser_hand, const int *sampled_hands,
                                 const SSNode *leaf_node, uint64_t *rng) {
    (void)leaf_node;
    ESVecSampledCtx *c = (ESVecSampledCtx *)vctx;
    if (!c) return 0.0;
    uint64_t t0 = esvs_now_ns();
    atomic_fetch_add_explicit(&c->leaf_visits, 1, memory_order_relaxed);
    if (!rng || !esvs_check_args(c, leaf_idx, traverser)
        || traverser_hand < 0
        || traverser_hand >= c->tree->num_hands[traverser]) {
        atomic_fetch_add_explicit(&c->oracle_errors, 1, memory_order_relaxed);
        return 0.0;
    }
    ESVSVisit v;
    if (esvs_visit_setup(c, leaf_idx, traverser, sampled_hands, rng, &v) != 0) {
        atomic_fetch_add_explicit(&c->oracle_errors, 1, memory_order_relaxed);
        return 0.0;
    }
    double out = esvs_eval_hand(c, &v, traverser_hand);
    atomic_fetch_add_explicit(&c->oracle_ns, esvs_now_ns() - t0,
                              memory_order_relaxed);
    return out;
}

ES_EXPORT ESLeafOracleVec es_vec_sampled_oracle_vec_fn(void) {
    return esvs_oracle_vec;
}

ES_EXPORT ESLeafOracle es_vec_sampled_oracle_scalar_fn(void) {
    return esvs_oracle_scalar;
}

ES_EXPORT void *es_make_vec_sampled_oracle(
    const void *bp_void, const SSTreeData *tree, int street, int stack_chips,
    int num_rollouts, int hero_bias_mode, int runout_mode, double explore_eps,
    int n_leaves, int max_h, const int *pot, const int *bets, const int *active,
    const int *hist_flat, const int *hist_len, const int *player_seats
) {
    if (!tree || !pot || !bets || !active || !hist_flat || !hist_len
        || !player_seats || n_leaves <= 0 || max_h <= 0
        || tree->num_players < 2 || tree->num_players > SS_MAX_PLAYERS
        || tree->num_leaves < 0 || n_leaves < tree->num_leaves
        || tree->num_board < 0 || tree->num_board > SS_MAX_BOARD
        || (street != ROLLOUT_STREET_FLOP && street != ROLLOUT_STREET_TURN)
        || (hero_bias_mode != ES_VS_HERO_BIAS_ENUMERATE
            && hero_bias_mode != ES_VS_HERO_BIAS_SAMPLE)
        || (runout_mode != ES_VS_RUNOUT_FRESH
            && runout_mode != ES_VS_RUNOUT_SHARED)
        || !(explore_eps >= 0.0 && explore_eps <= 0.5))
        return NULL;

    int np = tree->num_players;
    for (int p = 0; p < np; p++) {
        if (tree->num_hands[p] < 0 || tree->num_hands[p] > SS_MAX_HANDS
            || max_h < tree->num_hands[p])
            return NULL;
    }
    for (int leaf = 0; leaf < n_leaves; leaf++) {
        if (hist_len[leaf] < 0 || hist_len[leaf] > ESVS_MAX_HIST) return NULL;
    }

    size_t leaf_elems = (size_t)n_leaves;
    if (leaf_elems > SIZE_MAX / (size_t)np) return NULL;
    size_t leaf_player_elems = leaf_elems * (size_t)np;
    if (leaf_elems > SIZE_MAX / ESVS_MAX_HIST) return NULL;
    size_t hist_elems = leaf_elems * ESVS_MAX_HIST;
    if (hist_elems > SIZE_MAX / sizeof(int)) return NULL;
    size_t regret_elems = leaf_player_elems;
    if (regret_elems > SIZE_MAX / (size_t)max_h) return NULL;
    regret_elems *= (size_t)max_h;
    if (regret_elems > SIZE_MAX / ESVS_NB) return NULL;
    regret_elems *= ESVS_NB;
    if (regret_elems > SIZE_MAX / sizeof(float)) return NULL;

    ESVecSampledCtx *c = (ESVecSampledCtx *)calloc(1, sizeof(ESVecSampledCtx));
    if (!c) return NULL;
    c->bp = (const BPSReader *)bp_void;
    c->tree = tree;
    c->np = np;
    c->street = street;
    c->stack_chips = stack_chips;
    c->num_rollouts = num_rollouts > 0 ? num_rollouts : 1;
    c->hero_bias_mode = hero_bias_mode;
    c->runout_mode = runout_mode;
    c->explore_eps = (float)explore_eps;
    c->n_leaves = n_leaves;
    c->max_h = max_h;
    atomic_init(&c->leaf_visits, 0);
    atomic_init(&c->hand_visits, 0);
    atomic_init(&c->rollouts, 0);
    atomic_init(&c->oracle_errors, 0);
    atomic_init(&c->oracle_ns, 0);

    c->pot = (int *)malloc(leaf_elems * sizeof(int));
    c->bets = (int *)malloc(leaf_player_elems * sizeof(int));
    c->active = (int *)malloc(leaf_player_elems * sizeof(int));
    c->hist = (int *)malloc(hist_elems * sizeof(int));
    c->hist_len = (int *)malloc(leaf_elems * sizeof(int));
    c->player_seats = (int *)malloc((size_t)np * sizeof(int));
    c->acting_order = (int *)malloc((size_t)np * sizeof(int));
    c->bias_regret = (float *)calloc(regret_elems, sizeof(float));
    if (!c->pot || !c->bets || !c->active || !c->hist || !c->hist_len
        || !c->player_seats || !c->acting_order || !c->bias_regret) {
        es_free_vec_sampled_oracle(c);
        return NULL;
    }
    memcpy(c->pot, pot, leaf_elems * sizeof(int));
    memcpy(c->bets, bets, leaf_player_elems * sizeof(int));
    memcpy(c->active, active, leaf_player_elems * sizeof(int));
    memcpy(c->hist, hist_flat, hist_elems * sizeof(int));
    memcpy(c->hist_len, hist_len, leaf_elems * sizeof(int));
    memcpy(c->player_seats, player_seats, (size_t)np * sizeof(int));
    for (int p = 0; p < np; p++) c->acting_order[p] = p;
    /* Single-threaded setup point: arm (or keep warm) the per-solve
     * turn/river bucket memo for this flop before any Hogwild worker runs. */
    if (tree->num_board >= 3) {
        int flop3[3] = { tree->board[0], tree->board[1], tree->board[2] };
        rollout_bucket_memo_reset(bp_void, flop3);
    }
    return c;
}

ES_EXPORT int es_vec_sampled_oracle_set_bias_modes(
    void *ctx, int opp_bias_mode, int hero_bias_fixed
) {
    ESVecSampledCtx *c = (ESVecSampledCtx *)ctx;
    if (!c
        || (opp_bias_mode != ES_VS_OPP_BIAS_REGRET
            && opp_bias_mode != ES_VS_OPP_BIAS_UNIFORM
            && opp_bias_mode != ES_VS_OPP_BIAS_UNBIASED)
        || (hero_bias_fixed != 0 && hero_bias_fixed != 1))
        return -1;
    c->opp_bias_mode = opp_bias_mode;
    c->hero_bias_fixed = hero_bias_fixed;
    return 0;
}

ES_EXPORT void es_vec_sampled_oracle_stats(
    void *ctx, uint64_t *leaf_visits, uint64_t *hand_visits,
    uint64_t *rollouts, uint64_t *errors, uint64_t *oracle_ns
) {
    ESVecSampledCtx *c = (ESVecSampledCtx *)ctx;
    if (!c) return;
    if (leaf_visits)
        *leaf_visits = atomic_load_explicit(&c->leaf_visits,
                                            memory_order_relaxed);
    if (hand_visits)
        *hand_visits = atomic_load_explicit(&c->hand_visits,
                                            memory_order_relaxed);
    if (rollouts)
        *rollouts = atomic_load_explicit(&c->rollouts, memory_order_relaxed);
    if (errors)
        *errors = atomic_load_explicit(&c->oracle_errors,
                                       memory_order_relaxed);
    if (oracle_ns)
        *oracle_ns = atomic_load_explicit(&c->oracle_ns,
                                          memory_order_relaxed);
}

ES_EXPORT void es_free_vec_sampled_oracle(void *ctx) {
    ESVecSampledCtx *c = (ESVecSampledCtx *)ctx;
    if (!c) return;
    free(c->pot);
    free(c->bets);
    free(c->active);
    free(c->hist);
    free(c->hist_len);
    free(c->player_seats);
    free(c->acting_order);
    free(c->bias_regret);
    free(c);
}
