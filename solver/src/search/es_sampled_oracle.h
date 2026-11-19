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

/* Diagnostic-only fixed-joint-hand entry.  This mirrors esso_rollout_batch's
 * inner rollout_simulate_one call while allowing the harness to pin the bias
 * profile and RNG state.  It is additive and absent from the solve path. */
ES_EXPORT int32_t es_sampled_debug_simulate_fixed_c(
    const void *bp,
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
    float *out_payoff, uint64_t *out_rng_state);

#ifdef __cplusplus
}
#endif

#endif /* ES_SAMPLED_ORACLE_H */
