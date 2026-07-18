/*
 * es_sampled_oracle.c - Pluribus-style per-visit sampled leaf oracle.
 *
 * One ES leaf visit rolls forward from that visit's GIVEN joint sampled
 * hands.  There is no cache, all-hands fill, marginalization, or leaf table.
 * Policy 0 draws every active player's continuation bias uniformly and fresh.
 * Policy 1 samples opponents from per-(leaf, player, hand) bias regrets while
 * enumerating the traverser's four biases and updating that traverser row.
 */

#include "es_sampled_oracle.h"
#include "rollout_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define ESSO_MAX_FLOP_HIST ROLLOUT_MAX_ACTION_HISTORY

typedef struct {
    /* Load-bearing first-member ABI: es_sampled_oracle_fn() returns a tagged
     * resolver-local adapter which reads this immutable function pointer and
     * dispatches through it.  This lets es_create identify sampled mode without
     * making the lightweight resolver target depend on rollout.c. */
    ESLeafOracle invoke;

    const BPSReader *bp;
    const SSTreeData *tree;
    int np;
    int street;
    int stack_chips;
    int num_rollouts;
    int bias_policy;

    int n_leaves;
    int max_h;
    int *pot;             /* [n_leaves] */
    int *bets;            /* [n_leaves * np] */
    int *active;          /* [n_leaves * np] */
    int *hist;            /* [n_leaves * ESSO_MAX_FLOP_HIST] */
    int *hist_len;        /* [n_leaves] */
    int *player_seats;    /* [np], compact player -> blueprint seat */
    int *acting_order;    /* [np], identity */

    /* Dense [leaf][player][hand][bias] continuation-bias regrets.  Updates
     * are intentionally plain float Hogwild additions: concurrent visits may
     * race exactly as the resolver's main regret table does.  No atomics or
     * locks are needed for this convergence-tolerant transient state. */
    float *bias_regret;

    _Atomic(uint64_t) leaf_visits;
    _Atomic(uint64_t) rollouts;
    _Atomic(uint64_t) uniform_visits;
    _Atomic(uint64_t) uniform_rollouts;
    _Atomic(uint64_t) regret_visits;
    _Atomic(uint64_t) regret_rollouts;
    _Atomic(uint64_t) oracle_errors;
} ESSampledOracleCtx;

_Static_assert(offsetof(ESSampledOracleCtx, invoke) == 0,
               "sampled-oracle dispatch must be the first context member");

/* Same xorshift64 step as the scalar ES resolver.  All oracle randomness is an
 * advancement of the caller's per-traversal stream; there is no local/global
 * seed or random state. */
static inline uint64_t esso_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline double esso_rng_double(uint64_t *state) {
    return (double)(esso_rng_next(state) >> 11)
           * (1.0 / 9007199254740992.0);
}

static void esso_regret_match(const float *row, float out[4]) {
    float sum = 0.0f;
    for (int b = 0; b < ROLLOUT_NUM_BIAS_PROFILES; b++) {
        float positive = row[b] > 0.0f ? row[b] : 0.0f;
        out[b] = positive;
        sum += positive;
    }
    if (sum > 0.0f) {
        for (int b = 0; b < ROLLOUT_NUM_BIAS_PROFILES; b++) out[b] /= sum;
    } else {
        for (int b = 0; b < ROLLOUT_NUM_BIAS_PROFILES; b++)
            out[b] = 1.0f / (float)ROLLOUT_NUM_BIAS_PROFILES;
    }
}

static int esso_sample_bias(const float strategy[4], uint64_t *rng) {
    double u = esso_rng_double(rng);
    double cumulative = 0.0;
    for (int b = 0; b < ROLLOUT_NUM_BIAS_PROFILES; b++) {
        cumulative += (double)strategy[b];
        if (u < cumulative) return b;
    }
    return ROLLOUT_NUM_BIAS_PROFILES - 1;
}

static inline float *esso_regret_row(ESSampledOracleCtx *c, int leaf_idx,
                                     int player, int hand) {
    size_t row = ((size_t)leaf_idx * (size_t)c->np + (size_t)player)
                 * (size_t)c->max_h + (size_t)hand;
    return &c->bias_regret[row * ROLLOUT_NUM_BIAS_PROFILES];
}

/* Run one num_rollouts batch with a fixed joint bias profile and the visit's
 * fixed joint hands. */
static int esso_rollout_batch(
    ESSampledOracleCtx *c, int leaf_idx, int traverser,
    const int32_t *active, const int32_t *stacks, const int32_t *bets,
    const int32_t *sampled_c0, const int32_t *sampled_c1,
    const int32_t *has_sample, const int32_t *bias_profile,
    uint64_t *rng, float *out_value
) {
    const SSTreeData *t = c->tree;
    int board[SS_MAX_BOARD];
    for (int b = 0; b < t->num_board; b++) board[b] = t->board[b];
    const int32_t *hist =
        (const int32_t *)&c->hist[(size_t)leaf_idx * ESSO_MAX_FLOP_HIST];
    int32_t hist_len = c->hist_len[leaf_idx];
    int32_t skip_first = hist_len > 0 ? 1 : 0;

    double sum = 0.0;
    for (int r = 0; r < c->num_rollouts; r++) {
        float payoff = 0.0f;
        atomic_fetch_add_explicit(&c->rollouts, 1, memory_order_relaxed);
        if (c->bias_policy == ES_SAMPLED_BIAS_UNIFORM_FRESH)
            atomic_fetch_add_explicit(&c->uniform_rollouts, 1,
                                      memory_order_relaxed);
        else
            atomic_fetch_add_explicit(&c->regret_rollouts, 1,
                                      memory_order_relaxed);
        int rc = rollout_simulate_one(
            c->bp, board, t->num_board, c->street, c->pot[leaf_idx],
            active, stacks, bets, c->np,
            (const int32_t *)c->acting_order, c->np,
            (const int32_t *)c->player_seats,
            hist, hist_len, bias_profile, traverser,
            sampled_c0, sampled_c1, has_sample, skip_first, rng, &payoff);
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

ES_EXPORT int32_t es_sampled_debug_simulate_fixed_c(
    const void *bp_void,
    const int *board, int32_t n_board_cards,
    int32_t street, int32_t pot_chips,
    const int32_t *active, int32_t num_players,
    const int32_t *stacks, const int32_t *bets,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    const int32_t *bias_profile, int32_t hero_player,
    const int32_t *sampled_hands_c0,
    const int32_t *sampled_hands_c1,
    const int32_t *has_sample,
    int32_t skip_first_betting,
    uint64_t rng_seed,
    float *out_payoff, uint64_t *out_rng_state
) {
    if (!bp_void || !board || !active || !stacks || !bets || !acting_order
        || !player_seats || !game_action_history || !bias_profile
        || !sampled_hands_c0 || !sampled_hands_c1 || !has_sample
        || !out_payoff || n_board_cards < 0 || n_board_cards > 5
        || num_players < 2 || num_players > SS_MAX_PLAYERS
        || acting_order_len < 0 || acting_order_len > num_players
        || gah_len < 0 || gah_len > 256
        || hero_player < 0 || hero_player >= num_players)
        return -1;
    uint64_t rng = rng_seed ? rng_seed : 0xdeadbeefcafebabeULL;
    int32_t rc = rollout_simulate_one(
        (const BPSReader *)bp_void,
        board, n_board_cards, street, pot_chips,
        active, stacks, bets, num_players,
        acting_order, acting_order_len, player_seats,
        game_action_history, gah_len, bias_profile, hero_player,
        sampled_hands_c0, sampled_hands_c1, has_sample,
        skip_first_betting, &rng, out_payoff);
    if (out_rng_state) *out_rng_state = rng;
    return rc;
}

static double esso_oracle(void *vctx, int leaf_idx, int traverser,
                          int traverser_hand, const int *sampled_hands,
                          const SSNode *leaf_node, uint64_t *rng) {
    (void)leaf_node;
    ESSampledOracleCtx *c = (ESSampledOracleCtx *)vctx;
    if (!c) return 0.0;

    atomic_fetch_add_explicit(&c->leaf_visits, 1, memory_order_relaxed);
    if (c->bias_policy == ES_SAMPLED_BIAS_UNIFORM_FRESH)
        atomic_fetch_add_explicit(&c->uniform_visits, 1, memory_order_relaxed);
    else
        atomic_fetch_add_explicit(&c->regret_visits, 1, memory_order_relaxed);

    if (!rng) {
        atomic_fetch_add_explicit(&c->oracle_errors, 1, memory_order_relaxed);
        return 0.0;
    }

    if (leaf_idx < 0 || leaf_idx >= c->n_leaves
        || traverser < 0 || traverser >= c->np || !sampled_hands) {
        atomic_fetch_add_explicit(&c->oracle_errors, 1, memory_order_relaxed);
        return 0.0;
    }

    int32_t active[SS_MAX_PLAYERS];
    int32_t bets[SS_MAX_PLAYERS];
    int32_t stacks[SS_MAX_PLAYERS];
    int32_t sampled_c0[SS_MAX_PLAYERS];
    int32_t sampled_c1[SS_MAX_PLAYERS];
    int32_t has_sample[SS_MAX_PLAYERS];
    int32_t bias_profile[SS_MAX_PLAYERS];
    size_t leaf_player_base = (size_t)leaf_idx * (size_t)c->np;

    for (int p = 0; p < c->np; p++) {
        active[p] = c->active[leaf_player_base + (size_t)p];
        bets[p] = c->bets[leaf_player_base + (size_t)p];
        stacks[p] = c->stack_chips > bets[p] ? c->stack_chips - bets[p] : 0;
        sampled_c0[p] = 0;
        sampled_c1[p] = 0;
        has_sample[p] = 0;
        bias_profile[p] = 0;

        /* Hero is explicitly the interface's traverser_hand; every opponent
         * comes from sampled_hands.  Inactive players are still mapped so
         * their folded cards remain dead during future board deals. */
        int hand = p == traverser ? traverser_hand : sampled_hands[p];
        if (hand >= 0) {
            if (hand >= c->tree->num_hands[p]) {
                atomic_fetch_add_explicit(&c->oracle_errors, 1,
                                          memory_order_relaxed);
                return 0.0;
            }
            sampled_c0[p] = c->tree->hands[p][hand][0];
            sampled_c1[p] = c->tree->hands[p][hand][1];
            has_sample[p] = 1;
        }
    }

    /* A folded traverser has a known terminal contribution and does not make a
     * continuation-bias choice or execute a rollout. */
    if (!active[traverser]) return -(double)bets[traverser];
    if (!has_sample[traverser]) {
        atomic_fetch_add_explicit(&c->oracle_errors, 1, memory_order_relaxed);
        return 0.0;
    }

    if (c->bias_policy == ES_SAMPLED_BIAS_UNIFORM_FRESH) {
        for (int p = 0; p < c->np; p++) {
            if (active[p] && has_sample[p])
                bias_profile[p] =
                    (int32_t)(esso_rng_next(rng) % ROLLOUT_NUM_BIAS_PROFILES);
        }
        float value = 0.0f;
        if (esso_rollout_batch(c, leaf_idx, traverser, active, stacks, bets,
                               sampled_c0, sampled_c1, has_sample,
                               bias_profile, rng, &value) != 0)
            return 0.0;
        return (double)value;
    }

    /* Regret-matched opponents: sample once and hold their choices fixed while
     * the traverser's four biases are externally enumerated. */
    for (int p = 0; p < c->np; p++) {
        if (p == traverser || !active[p] || !has_sample[p]) continue;
        float strategy[ROLLOUT_NUM_BIAS_PROFILES];
        esso_regret_match(esso_regret_row(c, leaf_idx, p, sampled_hands[p]),
                          strategy);
        bias_profile[p] = esso_sample_bias(strategy, rng);
    }

    float traverser_strategy[ROLLOUT_NUM_BIAS_PROFILES];
    float *traverser_regret =
        esso_regret_row(c, leaf_idx, traverser, traverser_hand);
    esso_regret_match(traverser_regret, traverser_strategy);

    float values[ROLLOUT_NUM_BIAS_PROFILES];
    for (int b = 0; b < ROLLOUT_NUM_BIAS_PROFILES; b++) {
        bias_profile[traverser] = b;
        if (esso_rollout_batch(c, leaf_idx, traverser, active, stacks, bets,
                               sampled_c0, sampled_c1, has_sample,
                               bias_profile, rng, &values[b]) != 0)
            return 0.0;
    }

    float mixture = 0.0f;
    for (int b = 0; b < ROLLOUT_NUM_BIAS_PROFILES; b++)
        mixture += traverser_strategy[b] * values[b];

    /* Deliberately Hogwild-racy, matching the main ES regret accumulation. */
    for (int b = 0; b < ROLLOUT_NUM_BIAS_PROFILES; b++)
        traverser_regret[b] += values[b] - mixture;

    return (double)mixture;
}

ES_EXPORT void *es_make_sampled_oracle(
    const void *bp_void, const SSTreeData *tree, int street, int stack_chips,
    int num_rollouts, int bias_policy,
    int n_leaves, int max_h, const int *pot, const int *bets, const int *active,
    const int *hist_flat, const int *hist_len, const int *player_seats
) {
    if (!tree || !pot || !bets || !active || !hist_flat || !hist_len
        || !player_seats || n_leaves <= 0 || max_h <= 0
        || tree->num_players < 2 || tree->num_players > SS_MAX_PLAYERS
        || tree->num_leaves < 0 || n_leaves < tree->num_leaves
        || tree->num_board < 0 || tree->num_board > SS_MAX_BOARD
        || (street != ROLLOUT_STREET_FLOP
            && street != ROLLOUT_STREET_TURN)
        || (bias_policy != ES_SAMPLED_BIAS_UNIFORM_FRESH
            && bias_policy != ES_SAMPLED_BIAS_REGRET_MATCHED))
        return NULL;

    int np = tree->num_players;
    for (int p = 0; p < np; p++) {
        if (tree->num_hands[p] < 0 || tree->num_hands[p] > SS_MAX_HANDS
            || max_h < tree->num_hands[p])
            return NULL;
    }
    for (int leaf = 0; leaf < n_leaves; leaf++) {
        if (hist_len[leaf] < 0 || hist_len[leaf] > ESSO_MAX_FLOP_HIST)
            return NULL;
    }

    /* Guard every allocation-size multiplication. */
    size_t leaf_elems = (size_t)n_leaves;
    if (leaf_elems > SIZE_MAX / sizeof(int)) return NULL;
    if (leaf_elems > SIZE_MAX / (size_t)np) return NULL;
    size_t leaf_player_elems = leaf_elems * (size_t)np;
    if (leaf_player_elems > SIZE_MAX / sizeof(int)) return NULL;
    if (leaf_elems > SIZE_MAX / ESSO_MAX_FLOP_HIST) return NULL;
    size_t hist_elems = leaf_elems * ESSO_MAX_FLOP_HIST;
    if (hist_elems > SIZE_MAX / sizeof(int)) return NULL;

    size_t regret_elems = leaf_player_elems;
    if (regret_elems > SIZE_MAX / (size_t)max_h) return NULL;
    regret_elems *= (size_t)max_h;
    if (regret_elems > SIZE_MAX / ROLLOUT_NUM_BIAS_PROFILES) return NULL;
    regret_elems *= ROLLOUT_NUM_BIAS_PROFILES;
    if (regret_elems > SIZE_MAX / sizeof(float)) return NULL;

    ESSampledOracleCtx *c =
        (ESSampledOracleCtx *)calloc(1, sizeof(ESSampledOracleCtx));
    if (!c) return NULL;
    c->invoke = esso_oracle;
    c->bp = (const BPSReader *)bp_void;
    c->tree = tree;
    c->np = np;
    c->street = street;
    c->stack_chips = stack_chips;
    c->num_rollouts = num_rollouts > 0
                          ? num_rollouts : ROLLOUT_NUM_ROLLOUTS_DEFAULT;
    c->bias_policy = bias_policy;
    c->n_leaves = n_leaves;
    c->max_h = max_h;
    atomic_init(&c->leaf_visits, 0);
    atomic_init(&c->rollouts, 0);
    atomic_init(&c->uniform_visits, 0);
    atomic_init(&c->uniform_rollouts, 0);
    atomic_init(&c->regret_visits, 0);
    atomic_init(&c->regret_rollouts, 0);
    atomic_init(&c->oracle_errors, 0);

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
        es_free_sampled_oracle(c);
        return NULL;
    }

    memcpy(c->pot, pot, leaf_elems * sizeof(int));
    memcpy(c->bets, bets, leaf_player_elems * sizeof(int));
    memcpy(c->active, active, leaf_player_elems * sizeof(int));
    memcpy(c->hist, hist_flat, hist_elems * sizeof(int));
    memcpy(c->hist_len, hist_len, leaf_elems * sizeof(int));
    memcpy(c->player_seats, player_seats, (size_t)np * sizeof(int));
    for (int p = 0; p < np; p++) c->acting_order[p] = p;
    return c;
}

ES_EXPORT void es_sampled_oracle_stats(
    void *ctx, uint64_t *leaf_visits, uint64_t *rollouts, uint64_t *errors
) {
    ESSampledOracleCtx *c = (ESSampledOracleCtx *)ctx;
    if (!c) return;
    if (leaf_visits)
        *leaf_visits = atomic_load_explicit(&c->leaf_visits,
                                            memory_order_relaxed);
    if (rollouts)
        *rollouts = atomic_load_explicit(&c->rollouts, memory_order_relaxed);
    if (errors)
        *errors = atomic_load_explicit(&c->oracle_errors,
                                       memory_order_relaxed);
}

ES_EXPORT void es_sampled_oracle_stats2(
    void *ctx,
    uint64_t *uniform_visits, uint64_t *uniform_rollouts,
    uint64_t *regret_visits, uint64_t *regret_rollouts
) {
    ESSampledOracleCtx *c = (ESSampledOracleCtx *)ctx;
    if (!c) return;
    if (uniform_visits)
        *uniform_visits = atomic_load_explicit(&c->uniform_visits,
                                               memory_order_relaxed);
    if (uniform_rollouts)
        *uniform_rollouts = atomic_load_explicit(&c->uniform_rollouts,
                                                 memory_order_relaxed);
    if (regret_visits)
        *regret_visits = atomic_load_explicit(&c->regret_visits,
                                              memory_order_relaxed);
    if (regret_rollouts)
        *regret_rollouts = atomic_load_explicit(&c->regret_rollouts,
                                                memory_order_relaxed);
}

ES_EXPORT void es_free_sampled_oracle(void *ctx) {
    ESSampledOracleCtx *c = (ESSampledOracleCtx *)ctx;
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
