/**
 * es_sampled_oracle.h - per-visit sampled-rollout leaf oracle.
 *
 * Unlike es_real_oracle, this oracle never builds or caches an all-hands leaf
 * table.  Each call uses the hands sampled by the current ES traversal and
 * consumes only the resolver-provided RNG stream.
 */
#ifndef ES_SAMPLED_ORACLE_H
#define ES_SAMPLED_ORACLE_H

#include "es_resolver.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ES_SAMPLED_BIAS_UNIFORM_FRESH  0
#define ES_SAMPLED_BIAS_REGRET_MATCHED 1

/* Build a sampled-oracle context.  The tree and blueprint are borrowed and
 * must outlive the context; all flat metadata arrays are copied.  max_h is the
 * dense regret-store stride and must cover every player's hand list.
 *
 * There is deliberately no cache_k or oracle-local seed: one visit performs
 * exactly one batch per required bias evaluation, and every random choice is
 * drawn from the ESLeafOracle call's passed RNG. */
ES_EXPORT void *es_make_sampled_oracle(
    const void *bp, const SSTreeData *tree, int street, int stack_chips,
    int num_rollouts, int bias_policy,
    int n_leaves, int max_h, const int *pot, const int *bets, const int *active,
    const int *hist_flat, const int *hist_len, const int *player_seats);

/* Returns the tagged adapter used by es_create to enforce scalar-only mode. */
ES_EXPORT ESLeafOracle es_sampled_oracle_fn(void);

/* Aggregate counters.  A rollout is one actual rollout_simulate_one call. */
ES_EXPORT void es_sampled_oracle_stats(
    void *ctx, uint64_t *leaf_visits, uint64_t *rollouts,
    uint64_t *errors);

/* Per-policy counter breakdown (a context uses exactly one policy). */
ES_EXPORT void es_sampled_oracle_stats2(
    void *ctx,
    uint64_t *uniform_visits, uint64_t *uniform_rollouts,
    uint64_t *regret_visits, uint64_t *regret_rollouts);

ES_EXPORT void es_free_sampled_oracle(void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* ES_SAMPLED_ORACLE_H */
