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
 * sampled_hands, leaf_node) -> float.
 *
 * `rng` is the CALLER's per-traversal RNG state (per-thread under Hogwild). A
 * sampling oracle (the real blueprint rollout) MUST draw its cont-strat bias
 * profile via this rng so the bias varies FRESH per call — the cont-strat
 * chance node is then integrated across visits (matching real_oracle.py's
 * fresh-per-call bias draw and the GPU SAMPLED mode). A deterministic oracle
 * (table / joint-strength) ignores it. Advancing rng inside the oracle is
 * fine (it is the traversal's own stream). */
typedef double (*ESLeafOracle)(
    void *ctx,
    int leaf_idx,
    int traverser,
    int traverser_hand,
    const int *sampled_hands,
    const SSNode *leaf_node,
    uint64_t *rng
);

/* Resolver-owned trampoline for the sampled-rollout oracle. The sampled
 * oracle factory returns a ctx whose first member is its immutable dispatch
 * callback; this trampoline forwards each visit to that callback. Keeping the
 * trampoline in the resolver core lets es_create identify sampled mode without
 * a link-time dependency on es_sampled_oracle.c (the lightweight resolver does
 * not link rollout code). Callers normally obtain this through the sampled
 * oracle factory rather than invoking it directly. */
ES_EXPORT ESLeafOracle es_sampled_oracle_fn(void);

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

/* CFR variant: 0 = VANILLA (default; the validated reference fixed point),
 * 1 = LINEAR CFR matching the GPU (strat-sum += t·σ, regrets ·= t/(t+1) per
 * traverser-node visit). Linear converges to the same Nash faster — set it for
 * an iso-quality iteration comparison against the GPU (which runs Linear).
 * Set BEFORE es_run. */
ES_EXPORT void es_set_cfr_mode(ESResolver *r, int mode);

/* Traverser vectorization: 0 = SCALAR (default; the validated reference — one
 * walk updates one traverser hand), 1 = VECTORIZED (one walk down a single
 * opponent-sampled path updates ALL traverser hands; lower variance ⇒ fewer
 * iters for the same exploitability; IDENTICAL fixed point). Card-conflicting
 * traverser hands are skipped per walk (the unbiased card-removal estimator).
 * Set BEFORE es_run. Gate 1 (VECTOR_TRAVERSER_DESIGN.md): vec must reach the
 * SAME exploitability as scalar on a 2p/3p spot with OVERLAPPING ranges.
 * Sampled leaf mode is scalar-only: es_run fails if vec mode is enabled on a
 * resolver using es_sampled_oracle_fn(). */
ES_EXPORT void es_set_vec_mode(ESResolver *r, int mode);

/* Chance-node handling: 1 = ENUMERATE (correct, zero chance-variance: average
 * the value over ALL dealt cards — the GPU's full-reach equivalent), 0 = SAMPLE
 * (one card per iter — high variance + cross-line incoherence; the original OP3
 * design), -1 = inherit the process/env default (ENUMERATE unless
 * ES_CHANCE_SAMPLE=1). REQUIRED for a correct in-tree-chance (2p HU flop/turn)
 * solve. Set BEFORE es_run. */
ES_EXPORT void es_set_chance_mode(ESResolver *r, int mode);

/* Opponent-action handling: 1 = ENUMERATE (full CFR — recurse ALL opponent
 * actions sigma-weighted; with chance enumeration this removes opponent-action
 * sampling variance, for SMALL player counts only since the action-product
 * blows up at high N), 0 = SAMPLE (external sampling — mandatory + validated at
 * 5-6p), -1 = inherit the env default (SAMPLE unless ES_OPP_ENUMERATE=1). For
 * the 2p chance-extension solve, set 1 (+ vec mode) to kill the single-sample
 * oscillation. Set BEFORE es_run. */
ES_EXPORT void es_set_opp_mode(ESResolver *r, int mode);

/* Full-per-hand-REACH vector CFR: 1 = ON, 0 = OFF (default). When ON, es_run
 * carries a per-player per-node per-hand reach vector and integrates ALL
 * opponent hands each iteration with NO opponent-hand sampling — mirroring the
 * GPU street_solve.cu. Zero variance over opponent hands AND chance (enumerated),
 * so a 2p in-tree-chance subgame converges to the correct mixed strategy in
 * ~hundreds of iters in seconds, instead of the minutes the sampling vec path
 * needs to average out opponent-hand variance. It is always Linear-CFR + chance-
 * ENUMERATE + full reach by construction (ignores cfr/vec/chance/opp modes) and
 * is DETERMINISTIC (no RNG). For SMALL player counts only — 5-6p MUST keep
 * SAMPLING (set reach_mode=0). Sampled leaf mode is also incompatible with
 * reach mode because its fresh per-visit oracle requires one sampled hand
 * profile and one per-thread RNG stream. Set BEFORE es_run. */
ES_EXPORT void es_set_reach_mode(ESResolver *r, int mode);

/* 2p exploitability (best-response gap) of the CURRENT σ̄, in pot-chip units per
 * hand. Returns gap_0 + gap_1 (each >= 0; written to *out_gap0/1 if non-NULL); a
 * VALIDATED quality computer that reuses the reach machinery's own showdown/fold
 * evaluators (so it grades chance trees, unlike the Python exploitability_c.py
 * which zeros showdowns). DROPS toward 0 as the solve converges — the structural
 * truth garbage can't fake. np==2 only; returns NAN otherwise. */
ES_EXPORT double es_exploitability_2p(ESResolver *r, double *out_gap0,
                                      double *out_gap1);

/* Read + reset reach-mode per-phase wall-clock (seconds): t1=regret-match,
 * t2=reach-propagate, t3=terminal CFV, t4=CFV+update. Accumulated only when
 * ES_REACH_PROFILE=1. For diagnosing where a flop-scale solve spends its time. */
ES_EXPORT void es_reach_profile_read(double *t1, double *t2, double *t3, double *t4);

/* DEBUG: per-node traverser-visit counter (single-thread localization of the
 * vec reachability bug). Enable BEFORE es_run; read with es_get_visit_count
 * (returns -1 if disabled / bad index). Zero overhead when not enabled. */
ES_EXPORT void es_enable_visit_counter(ESResolver *r);
ES_EXPORT int64_t es_get_visit_count(const ESResolver *r, int node_idx);

/* Optional per-es_run monotonic wall-clock budget in milliseconds. `ms` > 0
 * applies a fresh duration budget to each subsequent es_run call; `ms` <= 0
 * disables the deadline (the default). The loops check at most once per 4096
 * outer CFR iterations and always finish an in-flight outer iteration cleanly.
 * Set before es_run; changing it during a run is unsupported. */
ES_EXPORT void es_set_deadline_ms(ESResolver *r, int64_t ms);

/* Number of complete outer CFR iterations performed by the most recent
 * es_run call. This is a last-run count, not cumulative. It is reset to zero at
 * run entry, including no-op and failed runs, and may be less than the requested
 * count when a deadline expires. */
ES_EXPORT int64_t es_iterations_run(const ESResolver *r);

/* Run `iterations` ES iterations. Each iteration walks once per player as
 * traverser (matching es_solver.py iterate()). num_threads > 1 enables
 * Hogwild (iterations partitioned across threads; shared tables, racy adds).
 * num_threads <= 0 → use all available cores. num_threads == 1 → deterministic
 * single-thread (the validated reference path).
 *
 * A resolver using es_sampled_oracle_fn() must keep both vec and reach modes
 * disabled; incompatible modes fail loud before any traversal begins.
 *
 * Returns 0 on success (including a clean deadline stop), -1 on error. Read
 * es_iterations_run() afterward for the completed outer-iteration count. */
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

/* ── TEST-ONLY: side-pot showdown probes (validate_sidepot.py fuzz gate) ───
 * Evaluate the shipped showdown code on a caller-described single showdown
 * node with ONE fixed hand per player. board5: 5 card ints; hand_c0/hand_c1/
 * active/bets: arrays of length np; pot in chips. Returns the traverser's NET
 * value (gross award − own street bet); NAN on invalid arguments. The _vec
 * variant drives the vectorized traverser's inline showdown block; the plain
 * variant drives es_showdown_value (the scalar + forward-walk path). */
ES_EXPORT double es_test_showdown_value(
    int np, const int *board5, const int *hand_c0, const int *hand_c1,
    const int *active, const int *bets, int pot, int traverser);
ES_EXPORT double es_test_showdown_value_vec(
    int np, const int *board5, const int *hand_c0, const int *hand_c1,
    const int *active, const int *bets, int pot, int traverser);

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

/* ── REAL blueprint-rollout oracle (es_real_oracle.c) ─────────────────────
 * The production hot-path oracle: lazy rollout_compute_leaf_value_c per leaf
 * hit. Per-leaf state is extracted ONCE in Python (extract_leaf_info_from_tree)
 * and passed flat. `bp` is an opaque BPSReader* (declared void* here to avoid
 * pulling bps_reader.h into this header). hist_flat is
 * [n_leaves * ROLLOUT_MAX_ACTION_HISTORY (256)] row-major. Build with
 * es_make_real_oracle, pass es_real_oracle_fn() as the oracle, free with
 * es_free_real_oracle.
 *
 * Leaf values are CACHED on first hit (strategy-independent -> constant per
 * solve): each reached leaf is filled once (all hands, cache_k bias-averaged)
 * and reused. Still lazy (no global construct floor), exact in expectation. */
ES_EXPORT void *es_make_real_oracle(
    const void *bp, const SSTreeData *tree, int street, int stack_chips,
    int num_rollouts, int cache_k, uint64_t master_seed,
    int n_leaves, int max_h, const int *pot, const int *bets, const int *active,
    const int *hist_flat, const int *hist_len, const int *player_seats);
ES_EXPORT ESLeafOracle es_real_oracle_fn(void);
ES_EXPORT void es_real_oracle_stats(void *ctx, uint64_t *calls,
                                    uint64_t *sims, uint64_t *errors);
ES_EXPORT void es_real_oracle_stats2(void *ctx, uint64_t *fills,
                                     uint64_t *cache_hits, uint64_t *sims,
                                     uint64_t *errors);
ES_EXPORT void es_free_real_oracle(void *ctx);

/* Diagnostic-only exact sampler surface.  Replays es_sample_hands with its
 * native randomized assignment order and weighted reservoir, retaining only
 * joint draws whose conditioned player received conditioned_hand.  The
 * production resolver never calls this entry point.
 *
 * out_hands is sample-major [num_samples * tree->num_players]; out_rng_states
 * receives the sampler stream immediately after each retained joint draw.
 * Returns the number retained (possibly less than num_samples when
 * max_attempts is exhausted), or -1 for invalid arguments. */
ES_EXPORT int32_t es_debug_sample_hands_conditioned_c(
    const SSTreeData *tree,
    uint64_t stream_seed,
    int32_t conditioned_player,
    int32_t conditioned_hand,
    int32_t num_samples,
    int32_t max_attempts,
    int32_t *out_hands,
    uint64_t *out_rng_states,
    int32_t *out_attempts,
    int32_t *out_incomplete);

#ifdef __cplusplus
}
#endif

#endif /* ES_RESOLVER_H */
