/**
 * mccfr_bias_regrets.h - mean-field opponent-bias regret sidecar.
 *
 * Experimental, off by default: allocated only with --cont-ev-bias-mode
 * mean_field. The published blueprint was trained without it.
 *
 * A SEPARATE open-addressed hash table, keyed EXACTLY like the cont_ev slot
 * table (street-salted action_hash + player + bucket triple — see
 * mccfr_cont_ev_table.h and bp_cont_ev_boundary_key_hash), holding 4 float
 * regrets (+ 4 float strategy-sum + a running value baseline) per slot.
 *
 * PURPOSE: by default the boundary-EV walk draws every player's
 * continuation-strategy bias UNIFORMLY (rng_int(rng, 4)), a chance node,
 * so ev_sum[b] = E[value | traverser bias b, opponents ~ Uniform{biases}].
 * The mean-field target instead has
 * each OPPONENT sample its bias from a regret-matched strategy maintained in
 * THIS sidecar, so ev_sum[b] becomes E[value | my bias b, opponents playing
 * their learned bias mix] — a step from "uniform opponents" toward
 * "equilibrium (minimax) opponents" without a BPR5 wire-format change and
 * without a retrain.
 *
 * This is NOT the literal-joint bias-profile game (its 4^(n-1) table blowup
 * puts it out of reach). It is a mean-field
 * (per-player marginal) approximation: each opponent slot carries its own
 * independent bias mix, and the sampler draws each opponent's bias
 * independently.
 *
 * WIRE FORMAT: none. This table is transient training state, allocated ONLY
 * when mean-field mode is on (bp_set_cont_ev_bias_mode(s, BP_BIAS_MODE_
 * MEANFIELD)). It is never serialized into the .cev / BPR5 artifact — the
 * mean-field semantics it produces are baked into the cont_ev ev_sum values,
 * and recorded out-of-band in the walk's .cev.meta.json sidecar.
 *
 * THREAD-SAFETY (mirrors mccfr_cont_ev_table.c exactly): lock-free
 * open-addressed table with the same 0/1/2 CAS state machine on `occupied`,
 * the same 4096-probe cap, the same pre-publish dedup scan. Float
 * accumulation (regrets, strategy_sum, value baseline) uses a CAS loop
 * (bp_bias_regrets_atomic_add_f32 == the type-punned uint32 CAS from
 * bp_cont_ev_atomic_add_f32). Integer counts use a relaxed fetch_add.
 * The trainer accumulates Hogwild-style (multiple OpenMP threads writing the
 * same slot without a lock); the CAS-loop float add tolerates that the same
 * way the cont_ev ev_sum accumulation does.
 *
 * Self-contained TU — depends only on stdint/stdlib/atomics + the ONE
 * canonical key-hash from mccfr_cont_ev_table.h (so producer-write ==
 * consumer-read per street cannot drift). The unit test links against just
 * this file + mccfr_cont_ev_table.c.
 */
#ifndef MCCFR_BIAS_REGRETS_H
#define MCCFR_BIAS_REGRETS_H

#include <stdint.h>

#include "mccfr_cont_ev_table.h"  /* BP_CONT_EV_NUM_BIASES, key-hash */

#ifdef __cplusplus
extern "C" {
#endif

/* One slot: 4-bias regret vector for a single (street-salted action_hash,
 * player, bucket) opponent infoset. Same CAS state machine on `occupied`
 * as BPContEVEntry:
 *   0 = empty
 *   1 = ready (key fields populated; regret/strategy_sum/value valid)
 *   2 = initializing (some thread is mid-publish; spin-wait) */
typedef struct {
    /* Key — identical triple to BPContEVEntry (the action_hash is ALREADY
     * street-salted by the caller via bp_cont_ev_boundary_key_hash, exactly
     * as the cont_ev slot's key is). */
    uint64_t action_hash;
    int player;
    int bucket;
    /* CAS state machine. */
    int occupied;
    /* Value — regret-matching state, indexed by cont_bias 0..3. */
    float regret[BP_CONT_EV_NUM_BIASES];        /* CFR regret; RM over positives */
    float strategy_sum[BP_CONT_EV_NUM_BIASES];  /* optional avg-strategy accum */
    /* Running value baseline for the outcome-based (variance-reduced)
     * regret estimator — see mccfr_bias_regrets.c bp_bias_regrets_update.
     * value_sum / value_count give the slot's running-mean opponent value,
     * used as the control-variate baseline the sampled-bias outcome is
     * regretted against. */
    float value_sum[BP_CONT_EV_NUM_BIASES];
    int64_t value_count[BP_CONT_EV_NUM_BIASES];
} BPBiasRegretEntry;

typedef struct {
    BPBiasRegretEntry *entries;      /* calloc'd; NULL means "mode OFF / unalloc" */
    int64_t table_size;              /* prime-promoted count of entries */
    int64_t num_entries;             /* relaxed atomic */
    int64_t insertion_failures;      /* relaxed atomic; probe-cap exhaustions */
    int64_t max_probe_observed;      /* relaxed atomic */
} BPBiasRegretTable;

/**
 * Allocate the bias-regret sidecar table. `table_size` is promoted to the
 * next prime (same trick as bp_cont_ev_init). Pass 0 to leave the table
 * un-allocated (entries == NULL), in which case find_or_create returns NULL.
 *
 * Returns 0 on success, -1 on OOM or invalid size.
 */
int bp_bias_regrets_init(BPBiasRegretTable *t, int64_t table_size);

/** Free the entries array and zero the struct. Safe on a zero-init table. */
void bp_bias_regrets_free(BPBiasRegretTable *t);

/** Occupied-slot count (atomic load; safe from any thread). */
int64_t bp_bias_regrets_count(const BPBiasRegretTable *t);

/**
 * Lock-free find-or-create, mirroring bp_cont_ev_find_or_create (identical
 * 0→2→1 CAS state machine, pre-publish dedup, probe cap). `action_hash` is
 * the ALREADY street-salted key hash (caller passes
 * bp_cont_ev_boundary_key_hash(raw_hash, street)).
 *
 * Returns the entry, or NULL if the table is unallocated or the probe cap is
 * exhausted. Concurrent calls with the same key converge on one slot.
 */
BPBiasRegretEntry *bp_bias_regrets_find_or_create(BPBiasRegretTable *t,
                                                  uint64_t action_hash,
                                                  int player, int bucket);

/**
 * Read-only probe. Returns the matching entry, or NULL on miss / unallocated.
 * No mutation, no allocation. Used by tests and the regret-matching sampler.
 */
BPBiasRegretEntry *bp_bias_regrets_find(const BPBiasRegretTable *t,
                                        uint64_t action_hash,
                                        int player, int bucket);

/**
 * Regret-matching sampler — standard CFR regret matching over the slot's
 * positive-regret mass, drawn from the caller's per-thread xorshift RNG.
 *
 *   - Slot found with sum(positive regrets) > 0 → sample proportional to the
 *     positive regrets (biases with regret <= 0 get zero probability).
 *   - Slot found but all regrets <= 0, OR slot NOT in the table (unseen) →
 *     uniform 1/BP_CONT_EV_NUM_BIASES (standard regret-matching fallback,
 *     identical distribution to the legacy uniform bias sampler).
 *
 * RNG DISCIPLINE: consumes EXACTLY ONE rng_next() draw (upper-24-bit uniform
 * float, byte-identical stream to mccfr_blueprint.c::rng_float), same as the
 * single rng_int() the uniform path consumes. The caller substitutes this
 * sampler's result for the uniform rng_int(rng, 4) ONLY in mean-field mode;
 * the uniform path is untouched, preserving byte-identity.
 *
 * `t` may be NULL (unallocated) — then the uniform fallback is used. Returns
 * a bias index in [0, BP_CONT_EV_NUM_BIASES).
 */
int bp_bias_regrets_sample(const BPBiasRegretTable *t,
                           uint64_t action_hash, int player, int bucket,
                           uint64_t *rng_state);

/**
 * Extract the regret-matched strategy (probabilities) for a slot into `out`
 * (length BP_CONT_EV_NUM_BIASES). Same distribution the sampler draws from:
 * positive-regret-proportional, or uniform when all regrets <= 0 / slot
 * unseen. Pure; no RNG, no mutation. Exposed for tests + diagnostics.
 */
void bp_bias_regrets_strategy(const BPBiasRegretTable *t,
                              uint64_t action_hash, int player, int bucket,
                              float *out);

/**
 * Outcome-based regret update for a sampled opponent bias.
 *
 * ESTIMATOR (documented in full in mccfr_bias_regrets.c): the boundary walk
 * gives an outcome value for the SAMPLED bias only — it does NOT provide
 * per-bias counterfactual values without extra rollouts (each rollout fixes
 * one joint bias tuple). So this is a Monte-Carlo external-sampling update,
 * NOT a full-sweep counterfactual one:
 *
 *   1. Accumulate the observed value into the sampled bias's running mean
 *      (value_sum[sampled] += value; value_count[sampled]++).
 *   2. Form a baseline = mean observed value across ALL biases seen so far
 *      for this slot (a control variate that reduces update variance).
 *   3. regret[sampled] += (value - baseline)   — reward the sampled bias
 *      when it beat the slot's running average, penalize it when it lagged.
 *   4. strategy_sum[sampled] += 1 (visit-count average-strategy accumulation;
 *      optional, used only for diagnostics / avg-strategy readout).
 *
 * `value` is the opponent's realized value at this boundary (the caller
 * passes -boundary_ev in the HU-collapsed convention; see the .c file for
 * the bias/variance tradeoff of that convention at multiway boundaries).
 *
 * Thread-safe: all float accumulation via the CAS-loop add; the int count
 * via relaxed fetch_add. `e` must be a live slot from find_or_create.
 * `sampled_bias` out of [0, BP_CONT_EV_NUM_BIASES) is a no-op.
 */
void bp_bias_regrets_update(BPBiasRegretEntry *e, int sampled_bias,
                            float value);

/**
 * Hogwild-safe atomic float add (type-punned uint32 CAS loop). Identical
 * math to bp_cont_ev_atomic_add_f32; re-exported here so this TU is
 * self-contained for the unit test. Used internally by the update.
 */
void bp_bias_regrets_atomic_add_f32(float *dst, float val);

#ifdef __cplusplus
}
#endif

#endif /* MCCFR_BIAS_REGRETS_H */
