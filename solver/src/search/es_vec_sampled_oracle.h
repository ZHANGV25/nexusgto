/**
 * es_vec_sampled_oracle.h - vec-coherent per-visit sampled leaf oracle.
 *
 * The Pluribus per-visit leaf (fresh rollouts on every visit, continuation
 * biases chosen by regret matching) inside the served VECTOR traverser. The
 * shelved scalar oracle (es_sampled_oracle.c) draws a fresh world per hand and
 * is therefore scalar-only. This oracle draws the world ONCE per leaf visit -
 * the opponents' continuation biases (sampled from their persistent
 * per-(leaf, player, hand) regret rows) and, optionally, the turn/river runout
 * - and values every live traverser hand in that one world. Each traverser
 * hand's own bias is regret-matched from its persistent row, so the hand
 * vector of one visit is coherent and the visit is reproducible given the
 * store and the traversal RNG stream (bit-identical single-thread).
 */
#ifndef ES_VEC_SAMPLED_ORACLE_H
#define ES_VEC_SAMPLED_ORACLE_H

#include "es_resolver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Traverser-bias handling per hand visit. ENUMERATE evaluates all four
 * biases (num_rollouts each, common random numbers across the four) and
 * applies the full external-sampling regret update: literal Pluribus, 4x the
 * rollouts. SAMPLE draws ONE bias from the epsilon-explored regret-matched
 * mixture and applies the importance-weighted (outcome-sampling) regret
 * update; unbiased, 1 rollout batch per hand visit. */
#define ES_VS_HERO_BIAS_ENUMERATE 0
#define ES_VS_HERO_BIAS_SAMPLE    1

/* Runout handling per visit. FRESH lets the simulator deal turn/river per
 * rollout (per hand). SHARED pins one runout sequence per visit so every hand
 * is valued on the same board (hands colliding with a pinned card skip it -
 * unbiased, see rollout_pinned_next_card). */
#define ES_VS_RUNOUT_FRESH  0
#define ES_VS_RUNOUT_SHARED 1

/* Opponent continuation-bias handling per visit (research A/B). REGRET keeps
 * the Pluribus leaf game: each opponent's bias is regret-matched from its
 * persistent row. UNIFORM draws each opponent's bias uniformly per visit.
 * UNBIASED fixes every opponent on the unbiased blueprint continuation
 * (bias profile 0). */
#define ES_VS_OPP_BIAS_REGRET   0
#define ES_VS_OPP_BIAS_UNIFORM  1
#define ES_VS_OPP_BIAS_UNBIASED 2

/* Build a context. The tree and blueprint are borrowed and must outlive the
 * context; all flat per-leaf arrays are copied (same layouts and meaning as
 * es_make_sampled_oracle). explore_eps in [0, 0.5] is only used by SAMPLE
 * mode. Arms the per-solve bucket memo for the tree's flop. */
ES_EXPORT void *es_make_vec_sampled_oracle(
    const void *bp, const SSTreeData *tree, int street, int stack_chips,
    int num_rollouts, int hero_bias_mode, int runout_mode, double explore_eps,
    int n_leaves, int max_h, const int *pot, const int *bets, const int *active,
    const int *hist_flat, const int *hist_len, const int *player_seats);

/* Research A/B: set the opponent-bias mode (ES_VS_OPP_BIAS_*) and whether the
 * traverser plays the unbiased continuation with no bias regret update
 * (hero_bias_fixed = 1, which bypasses hero_bias_mode). Call before solving.
 * A new context starts at (ES_VS_OPP_BIAS_REGRET, 0), the served behavior.
 * Returns 0, or -1 on a NULL context or an invalid mode. */
ES_EXPORT int es_vec_sampled_oracle_set_bias_modes(
    void *ctx, int opp_bias_mode, int hero_bias_fixed);

/* The vector entry (register with es_set_vec_leaf_oracle). */
ES_EXPORT ESLeafOracleVec es_vec_sampled_oracle_vec_fn(void);

/* Single-hand entry for es_create (used by every non-vector resolver path).
 * One call = one visit of one hand: the world is drawn per call. */
ES_EXPORT ESLeafOracle es_vec_sampled_oracle_scalar_fn(void);

/* Counters: leaf_visits = vector calls + scalar calls; hand_visits = live
 * hands valued; rollouts = rollout_simulate_one_ex calls; oracle_ns = wall
 * nanoseconds spent inside the oracle (monotonic clock, summed over threads
 * - divide by thread count for a per-core share). */
ES_EXPORT void es_vec_sampled_oracle_stats(
    void *ctx, uint64_t *leaf_visits, uint64_t *hand_visits,
    uint64_t *rollouts, uint64_t *errors, uint64_t *oracle_ns);

ES_EXPORT void es_free_vec_sampled_oracle(void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* ES_VEC_SAMPLED_ORACLE_H */
