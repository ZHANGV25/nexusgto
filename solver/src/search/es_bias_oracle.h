/**
 * es_bias_oracle.h - table-backed continuation-bias CFR leaf oracle.
 *
 * The values buffer is borrowed, never copied, and must remain alive until
 * es_free_bias_oracle. MARGINAL values use
 * [n_leaves][4][np][max_h]. JOINT2P values use the existing rollout joint
 * layout [n_leaves * 16][np][max_h], with player 0 as the low base-4 digit.
 */
#ifndef ES_BIAS_ORACLE_H
#define ES_BIAS_ORACLE_H

#include "es_resolver.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ES_BIAS_MARGINAL 0
#define ES_BIAS_JOINT2P  1
#define ES_BIAS_NUM_ACTIONS 4

/* Build a plain (non-adapter) leaf oracle. JOINT2P requires np == 2.
 * The dense regret store is [n_leaves][np][max_h][4]. */
ES_EXPORT void *es_make_bias_oracle(
    const float *values, int mode, int np, int max_h, int n_leaves);

/* Returns the oracle's own function pointer. It deliberately does not use the
 * sampled-oracle first-member adapter ABI, so the resolver treats it as
 * deterministic and vector-legal. */
ES_EXPORT ESLeafOracle es_bias_oracle_fn(void);

ES_EXPORT void es_free_bias_oracle(void *ctx);

ES_EXPORT void es_bias_oracle_stats(
    void *ctx, uint64_t *visits, uint64_t *errors);

/* Post-solve, single-thread reporting export. Writes
 * [n_leaves][np][max_h] and returns the number of floats written, or -1 for
 * invalid arguments / insufficient out_size. `values` must point to a table
 * with the same mode-specific layout supplied at construction.
 *
 * In JOINT2P solve calls, sampled_hands[opponent] selects the opponent's bias
 * regret row. If that hand is < 0, the live oracle falls back to a uniform
 * opponent-bias mixture over the 16-combo table. This export always folds the
 * opponent with a UNIFORM bias mixture because no sampled opponent hand exists
 * at reporting time; the JOINT2P effective table is reporting-only. */
ES_EXPORT int32_t es_bias_oracle_get_effective_table(
    void *ctx, const float *values, float *out, int32_t out_size);

#ifdef __cplusplus
}
#endif

#endif /* ES_BIAS_ORACLE_H */
