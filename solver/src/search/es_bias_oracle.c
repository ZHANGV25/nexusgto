/*
 * es_bias_oracle.c - table-backed continuation-bias CFR leaf oracle.
 *
 * Per-(leaf, player, hand) continuation-bias regrets are regret-matched on
 * every leaf visit. MARGINAL reads the traverser's four precomputed per-bias
 * values.
 * JOINT2P folds the full 16-combo table against the opponent's current
 * regret-matched bias mixture. No oracle randomness is consumed.
 */

#include "es_bias_oracle.h"

#include <limits.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>

typedef struct {
    const float *values;  /* Borrowed; caller keeps alive. */
    int mode;
    int np;
    int max_h;
    int n_leaves;

    /* Dense [leaf][player][hand][bias] continuation-bias regrets. Updates
     * are intentionally plain float Hogwild additions: concurrent visits may
     * race exactly as the resolver's main regret table does. No atomics or
     * locks are needed for this convergence-tolerant transient state. */
    float *bias_regret;

    _Atomic(uint64_t) leaf_visits;
    _Atomic(uint64_t) oracle_errors;
} ESBiasOracleCtx;

static void esbo_regret_match(const float *row, float out[4]) {
    float sum = 0.0f;
    for (int b = 0; b < ES_BIAS_NUM_ACTIONS; b++) {
        float positive = row[b] > 0.0f ? row[b] : 0.0f;
        out[b] = positive;
        sum += positive;
    }
    if (sum > 0.0f) {
        for (int b = 0; b < ES_BIAS_NUM_ACTIONS; b++) out[b] /= sum;
    } else {
        for (int b = 0; b < ES_BIAS_NUM_ACTIONS; b++)
            out[b] = 1.0f / (float)ES_BIAS_NUM_ACTIONS;
    }
}

static inline float *esbo_regret_row(ESBiasOracleCtx *c, int leaf_idx,
                                     int player, int hand) {
    size_t row = ((size_t)leaf_idx * (size_t)c->np + (size_t)player)
                 * (size_t)c->max_h + (size_t)hand;
    return &c->bias_regret[row * ES_BIAS_NUM_ACTIONS];
}

static inline size_t esbo_marginal_index(const ESBiasOracleCtx *c,
                                         int leaf_idx, int bias,
                                         int player, int hand) {
    return (((size_t)leaf_idx * ES_BIAS_NUM_ACTIONS + (size_t)bias)
             * (size_t)c->np + (size_t)player)
           * (size_t)c->max_h + (size_t)hand;
}

static inline size_t esbo_joint_index(const ESBiasOracleCtx *c,
                                      int leaf_idx, int combo,
                                      int player, int hand) {
    return (((size_t)leaf_idx * 16U + (size_t)combo) * (size_t)c->np
             + (size_t)player)
           * (size_t)c->max_h + (size_t)hand;
}

static void esbo_build_values(const ESBiasOracleCtx *c, const float *values,
                              int leaf_idx, int traverser, int hand,
                              const float opponent_strategy[4],
                              float val[4]) {
    if (c->mode == ES_BIAS_MARGINAL) {
        for (int b = 0; b < ES_BIAS_NUM_ACTIONS; b++) {
            val[b] = values[
                esbo_marginal_index(c, leaf_idx, b, traverser, hand)];
        }
        return;
    }

    /* Existing JOINT rollout decode order is player 0 first / low digit:
     * combo = player0_bias + 4 * player1_bias. */
    for (int b = 0; b < ES_BIAS_NUM_ACTIONS; b++) {
        float sum = 0.0f;
        for (int opponent_bias = 0;
             opponent_bias < ES_BIAS_NUM_ACTIONS; opponent_bias++) {
            int combo = traverser == 0
                            ? b + ES_BIAS_NUM_ACTIONS * opponent_bias
                            : opponent_bias + ES_BIAS_NUM_ACTIONS * b;
            sum += opponent_strategy[opponent_bias]
                   * values[esbo_joint_index(
                       c, leaf_idx, combo, traverser, hand)];
        }
        val[b] = sum;
    }
}

static double es_oracle_bias(void *vctx, int leaf_idx, int traverser,
                             int hand, const int *sampled_hands,
                             const SSNode *leaf_node, uint64_t *rng) {
    (void)leaf_node;
    (void)rng;
    ESBiasOracleCtx *c = (ESBiasOracleCtx *)vctx;
    if (!c) return 0.0;

    atomic_fetch_add_explicit(&c->leaf_visits, 1, memory_order_relaxed);
    if (leaf_idx < 0 || leaf_idx >= c->n_leaves
        || traverser < 0 || traverser >= c->np
        || hand < 0 || hand >= c->max_h || !sampled_hands) {
        atomic_fetch_add_explicit(&c->oracle_errors, 1,
                                  memory_order_relaxed);
        return 0.0;
    }

    float opponent_strategy[ES_BIAS_NUM_ACTIONS] = {
        0.25f, 0.25f, 0.25f, 0.25f
    };
    if (c->mode == ES_BIAS_JOINT2P) {
        int opponent = 1 - traverser;
        int opponent_hand = sampled_hands[opponent];
        if (opponent_hand >= c->max_h) {
            atomic_fetch_add_explicit(&c->oracle_errors, 1,
                                      memory_order_relaxed);
            return 0.0;
        }
        if (opponent_hand >= 0) {
            esbo_regret_match(
                esbo_regret_row(c, leaf_idx, opponent, opponent_hand),
                opponent_strategy);
        }
    }

    float val[ES_BIAS_NUM_ACTIONS];
    esbo_build_values(c, c->values, leaf_idx, traverser, hand,
                      opponent_strategy, val);

    float *traverser_regret =
        esbo_regret_row(c, leaf_idx, traverser, hand);
    float traverser_strategy[ES_BIAS_NUM_ACTIONS];
    esbo_regret_match(traverser_regret, traverser_strategy);

    float mixture = 0.0f;
    for (int b = 0; b < ES_BIAS_NUM_ACTIONS; b++)
        mixture += traverser_strategy[b] * val[b];

    /* Deliberately Hogwild-racy, matching the main ES regret accumulation. */
    for (int b = 0; b < ES_BIAS_NUM_ACTIONS; b++)
        traverser_regret[b] += val[b] - mixture;

    return (double)mixture;
}

ES_EXPORT void *es_make_bias_oracle(
    const float *values, int mode, int np, int max_h, int n_leaves
) {
    if (!values
        || (mode != ES_BIAS_MARGINAL && mode != ES_BIAS_JOINT2P)
        || np < 2 || np > SS_MAX_PLAYERS
        || max_h <= 0 || max_h > SS_MAX_HANDS
        || n_leaves <= 0 || n_leaves > SS_MAX_TREE_NODES
        || (mode == ES_BIAS_JOINT2P && np != 2))
        return NULL;

    /* Guard every allocation-size multiplication. The values table is
     * borrowed, so only the regret table is allocated here. */
    size_t regret_elems = (size_t)n_leaves;
    if (regret_elems > SIZE_MAX / (size_t)np) return NULL;
    regret_elems *= (size_t)np;
    if (regret_elems > SIZE_MAX / (size_t)max_h) return NULL;
    regret_elems *= (size_t)max_h;
    if (regret_elems > SIZE_MAX / ES_BIAS_NUM_ACTIONS) return NULL;
    regret_elems *= ES_BIAS_NUM_ACTIONS;
    if (regret_elems > SIZE_MAX / sizeof(float)) return NULL;

    ESBiasOracleCtx *c =
        (ESBiasOracleCtx *)calloc(1, sizeof(ESBiasOracleCtx));
    if (!c) return NULL;
    c->bias_regret = (float *)calloc(regret_elems, sizeof(float));
    if (!c->bias_regret) {
        free(c);
        return NULL;
    }

    c->values = values;
    c->mode = mode;
    c->np = np;
    c->max_h = max_h;
    c->n_leaves = n_leaves;
    atomic_init(&c->leaf_visits, 0);
    atomic_init(&c->oracle_errors, 0);
    return c;
}

ES_EXPORT ESLeafOracle es_bias_oracle_fn(void) {
    return es_oracle_bias;
}

ES_EXPORT void es_bias_oracle_stats(
    void *ctx, uint64_t *visits, uint64_t *errors
) {
    ESBiasOracleCtx *c = (ESBiasOracleCtx *)ctx;
    if (!c) return;
    if (visits)
        *visits = atomic_load_explicit(&c->leaf_visits,
                                       memory_order_relaxed);
    if (errors)
        *errors = atomic_load_explicit(&c->oracle_errors,
                                       memory_order_relaxed);
}

ES_EXPORT int32_t es_bias_oracle_get_effective_table(
    void *ctx, const float *values, float *out, int32_t out_size
) {
    ESBiasOracleCtx *c = (ESBiasOracleCtx *)ctx;
    if (!c || !values || values != c->values || !out || out_size < 0)
        return -1;

    size_t needed = (size_t)c->n_leaves;
    if (needed > SIZE_MAX / (size_t)c->np) return -1;
    needed *= (size_t)c->np;
    if (needed > SIZE_MAX / (size_t)c->max_h) return -1;
    needed *= (size_t)c->max_h;
    if (needed > (size_t)INT32_MAX || needed > (size_t)out_size) return -1;

    const float uniform_opponent[ES_BIAS_NUM_ACTIONS] = {
        0.25f, 0.25f, 0.25f, 0.25f
    };
    for (int leaf = 0; leaf < c->n_leaves; leaf++) {
        for (int player = 0; player < c->np; player++) {
            for (int hand = 0; hand < c->max_h; hand++) {
                float val[ES_BIAS_NUM_ACTIONS];
                esbo_build_values(c, values, leaf, player, hand,
                                  uniform_opponent, val);
                float strategy[ES_BIAS_NUM_ACTIONS];
                esbo_regret_match(
                    esbo_regret_row(c, leaf, player, hand), strategy);
                float effective = 0.0f;
                for (int b = 0; b < ES_BIAS_NUM_ACTIONS; b++)
                    effective += strategy[b] * val[b];
                size_t idx =
                    ((size_t)leaf * (size_t)c->np + (size_t)player)
                    * (size_t)c->max_h + (size_t)hand;
                out[idx] = effective;
            }
        }
    }
    return (int32_t)needed;
}

ES_EXPORT void es_free_bias_oracle(void *ctx) {
    ESBiasOracleCtx *c = (ESBiasOracleCtx *)ctx;
    if (!c) return;
    free(c->bias_regret);
    free(c);
}
