/*
 * es_oracles.c — deterministic leaf oracles for ES-resolver validation.
 *
 * These implement ESLeafOracle (es_resolver.h) entirely in C so the validation
 * harnesses never pay a per-leaf Python callback in the hot path (advisor: a
 * Python callback at T iterations dominates wall-time and makes the benchmark
 * unrepresentative). Each "make" returns an opaque context the caller passes
 * to es_create alongside the matching oracle function pointer.
 *
 * Two oracles:
 *   1. TABLE  — out = table[leaf_idx*NP*max_h + traverser*max_h + hand].
 *               Matches validate_vs_gpu's deterministic shared leaf table and
 *               validate_fixed_profile's random leaf table. Bit-exact, zero
 *               rollout noise → ES and GPU share a fixed point.
 *   2. JOINT  — zero-sum 2p strength showdown: payoff to p0 = sign(s0-s1)*pot;
 *               net to traverser = (win? pot : 0; tie pot/2) - my_bet.
 *               Matches validate_tiny's make_joint_oracle. Genuinely zero-sum,
 *               deterministic in (h0,h1) → BR-gap is a clean convergence signal.
 */

#include "es_resolver.h"

#include <stdlib.h>
#include <string.h>

/* ── TABLE oracle ────────────────────────────────────────────────────────── */
typedef struct {
    const float *table;   /* [num_leaves * np * max_h] */
    int np;
    int max_h;
} ESTableCtx;

static double es_oracle_table(void *vctx, int leaf_idx, int traverser,
                              int traverser_hand, const int *sampled_hands,
                              const SSNode *leaf_node) {
    (void)sampled_hands; (void)leaf_node;
    const ESTableCtx *c = (const ESTableCtx *)vctx;
    if (traverser_hand < 0) return 0.0;
    size_t idx = ((size_t)leaf_idx * c->np + traverser) * c->max_h + traverser_hand;
    return (double)c->table[idx];
}

ES_EXPORT void *es_make_table_oracle(const float *table, int np, int max_h) {
    ESTableCtx *c = (ESTableCtx *)malloc(sizeof(ESTableCtx));
    if (!c) return NULL;
    c->table = table; c->np = np; c->max_h = max_h;
    return c;
}
ES_EXPORT ESLeafOracle es_table_oracle_fn(void) { return es_oracle_table; }
ES_EXPORT void es_free_oracle_ctx(void *ctx) { free(ctx); }

/* ── JOINT zero-sum strength oracle (2p) ─────────────────────────────────────
 * strength[p] points to a per-hand scalar strength array for player p (length
 * num_hands[p]). Payoff to p0 = sign(s0 - s1) * pot. Tie splits the pot.
 */
typedef struct {
    const double *strength0;
    const double *strength1;
} ESJointCtx;

static double es_oracle_joint(void *vctx, int leaf_idx, int traverser,
                              int traverser_hand, const int *sampled_hands,
                              const SSNode *leaf_node) {
    (void)leaf_idx; (void)traverser_hand;
    const ESJointCtx *c = (const ESJointCtx *)vctx;
    int h0 = sampled_hands[0];
    int h1 = sampled_hands[1];
    if (h0 < 0 || h1 < 0) return 0.0;
    double s0 = c->strength0[h0];
    double s1 = c->strength1[h1];
    double pot = (double)leaf_node->pot;
    double my_bet = (double)leaf_node->bets[traverser];
    if (s0 == s1) return pot / 2.0 - my_bet;
    int p0_wins = (s0 > s1);
    int trav_wins = (traverser == 0 && p0_wins) || (traverser == 1 && !p0_wins);
    if (trav_wins) return pot - my_bet;
    return -my_bet;
}

ES_EXPORT void *es_make_joint_oracle(const double *strength0, const double *strength1) {
    ESJointCtx *c = (ESJointCtx *)malloc(sizeof(ESJointCtx));
    if (!c) return NULL;
    c->strength0 = strength0; c->strength1 = strength1;
    return c;
}
ES_EXPORT ESLeafOracle es_joint_oracle_fn(void) { return es_oracle_joint; }
