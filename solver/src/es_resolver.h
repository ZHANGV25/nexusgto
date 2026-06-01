/**
 * es_resolver.h — CPU External-Sampling MCCFR subgame re-solver (production C).
 *
 * C port of python/es_resolver/es_solver.py (the validated correct-but-slow
 * reference). Same algorithm → same fixed point. This is the shipping path for
 * 5-6p postflop subgame re-solves (and benchmarked head-to-head vs the GPU
 * vector solver for all N=2..6).
 *
 * ALGORITHM (es_solver.py is the spec — VANILLA external-sampling CFR):
 *   - At the TRAVERSER's decision nodes: explore ALL actions (branch).
 *   - At every OTHER player's decision node: sample ONE action from their
 *     current regret-matched strategy.
 *   - Regret update at traverser nodes: r(a) += child_val(a) - node_val.
 *     NO t/(t+1) discount, NO reach weight (external sampling realizes the
 *     traverser's cf-reach via the sampled path).
 *   - Strategy-sum update at traverser nodes: ssum(a) += sigma(a). Plain
 *     per-visit accumulation — NO iteration weight (matches
 *     mccfr_blueprint.c:2037 and es_solver.py:341). The `linear_cfr` flag in
 *     the Python reference is DEAD CODE (never read by the update math); this
 *     port is deliberately vanilla. See memory es-resolver-vanilla-not-linear-cfr.
 *   - Leaves are LAZY: a betting-leaf value is one sampled-opponent rollout,
 *     supplied via a caller function pointer (table-lookup oracle for
 *     validation; rollout_compute_leaf_value_c for the real benchmark).
 *
 * TREE: consumes the IDENTICAL SSTreeData that ss_build_tree2 produces (the
 * same struct the GPU solver uses) — no structural-mismatch risk, and a fair
 * benchmark uses the same tree on both sides. The struct is read-only here.
 *
 * STORAGE: dense float64 regret + strategy-sum, keyed (decision_node, hand,
 * action), via per-node offset tables. The subgame is bounded
 * (<= SS_MAX_TREE_NODES nodes, <= SS_MAX_HANDS hands, <= SS_MAX_ACTIONS
 * actions) so a hash table is unnecessary (that exists for the billions of
 * blueprint info sets; here we have thousands). float64 to match the
 * reference numerics.
 *
 * THREADING: optional Hogwild (shared dense tables, per-thread RNG, racy
 * accumulation) like mccfr_blueprint.c. Validated single-threaded FIRST.
 */
#ifndef ES_RESOLVER_H
#define ES_RESOLVER_H

#include <stdint.h>
#include <stddef.h>

#include "cuda/street_solve.cuh"   /* SSTreeData, SSNode, SS_* limits */

#ifdef _WIN32
#define ES_EXPORT __declspec(dllexport)
#else
#define ES_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Leaf oracle interface ───────────────────────────────────────────────
 *
 * Returns the TRAVERSER's net chip payoff at betting leaf `leaf_idx`, holding
 * hand index `traverser_hand` (into that player's hand list), with all players'
 * sampled hand indices in `sampled_hands` (length num_players; -1 = unassigned).
 * `ctx` is the caller-supplied opaque pointer (e.g. a deterministic table, or
 * a struct wrapping a BPSReader + rollout state). LAZY: invoked only when a
 * traversal reaches the leaf.
 *
 * Mirrors es_solver.py's leaf_rollout_fn(leaf_idx, traverser, hand,
 * sampled_hands, leaf_node) -> float. */
typedef double (*ESLeafOracle)(
    void *ctx,
    int leaf_idx,
    int traverser,
    int traverser_hand,
    const int *sampled_hands,
    const SSNode *leaf_node
);

/* ── Solver handle (opaque) ──────────────────────────────────────────────
 * Allocated by es_create, freed by es_free. Holds the dense regret/strat-sum
 * tables and per-node offset metadata derived once from the tree. */
typedef struct ESResolver ESResolver;

/* Create a resolver bound to a tree + leaf oracle. The tree pointer must
 * outlive the resolver (read-only, not copied). `seed` seeds the master RNG.
 * Returns NULL on allocation failure or invalid tree (see es_get_last_error). */
ES_EXPORT ESResolver *es_create(
    const SSTreeData *tree,
    ESLeafOracle leaf_oracle,
    void *oracle_ctx,
    uint64_t seed
);

ES_EXPORT void es_free(ESResolver *r);

/* Run `iterations` ES iterations. Each iteration walks once per player as
 * traverser (matching es_solver.py iterate()). num_threads > 1 enables
 * Hogwild (iterations partitioned across threads; shared tables, racy adds).
 * num_threads <= 0 → use all available cores. num_threads == 1 → deterministic
 * single-thread (the validated reference path).
 *
 * Returns 0 on success, -1 on error. */
ES_EXPORT int es_run(ESResolver *r, int64_t iterations, int num_threads);

/* Forward-only EV estimate for `player` under the UNIFORM profile (no regret
 * updates, every action equiprobable). Mirrors validate_fixed_profile's
 * es_root_ev_uniform — the variant-independent multiway correctness gate.
 * Writes the mean to *out_mean and the standard error to *out_se.
 * Returns 0 on success. */
ES_EXPORT int es_forward_ev_uniform(
    ESResolver *r, int player, int64_t iterations,
    double *out_mean, double *out_se
);

/* ── Strategy readout ────────────────────────────────────────────────────
 *
 * Write σ̄ (normalized strategy-sum per hand) at decision node `node_idx`
 * into the caller buffer `out` of shape [num_hands_of_node_player * num_actions]
 * row-major (out[hand * na + a]). Hands with zero strat-sum get uniform.
 * `out_cap` is the buffer capacity in floats. Returns the number of floats
 * written, or -1 on error (not a decision node / buffer too small).
 *
 * This is the surface the Python validation harnesses call (swap ESSolver ->
 * CSolver: .avg_strategy(node_idx) -> [nh, na]). */
ES_EXPORT int es_avg_strategy(
    const ESResolver *r, int node_idx, double *out, int out_cap
);

/* Final-iteration regret-matched strategy at `node_idx` (Pluribus play-path).
 * Same layout as es_avg_strategy. */
ES_EXPORT int es_final_strategy(
    const ESResolver *r, int node_idx, double *out, int out_cap
);

/* ── Stats (THE invariant: leaf-rollouts-to-converge, advisor) ───────────── */
typedef struct {
    int64_t traversals;
    int64_t leaf_rollouts;
    int64_t fold_terminals;
    int64_t showdown_terminals;
    int64_t decision_visits;
} ESStats;

ES_EXPORT void es_get_stats(const ESResolver *r, ESStats *out);
ES_EXPORT void es_reset_stats(ESResolver *r);

ES_EXPORT const char *es_get_last_error(void);

/* ── Deterministic validation oracles (es_oracles.c) ──────────────────────
 * Each make_* returns an opaque ctx to pass to es_create alongside the
 * matching *_fn() function pointer; free with es_free_oracle_ctx. */

/* TABLE oracle: out = table[leaf_idx*np*max_h + traverser*max_h + hand].
 * `table` is borrowed (caller keeps it alive for the resolver's lifetime). */
ES_EXPORT void *es_make_table_oracle(const float *table, int np, int max_h);
ES_EXPORT ESLeafOracle es_table_oracle_fn(void);

/* JOINT 2p zero-sum strength oracle: payoff to p0 = sign(s0-s1)*pot.
 * strength0/strength1 are per-hand scalar arrays (borrowed). */
ES_EXPORT void *es_make_joint_oracle(const double *strength0, const double *strength1);
ES_EXPORT ESLeafOracle es_joint_oracle_fn(void);

ES_EXPORT void es_free_oracle_ctx(void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* ES_RESOLVER_H */
