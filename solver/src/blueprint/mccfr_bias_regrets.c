/**
 * mccfr_bias_regrets.c — Stage 1 mean-field opponent-bias regret sidecar.
 *
 * See mccfr_bias_regrets.h for the contract and design rationale. This file
 * mirrors the concurrency primitives of mccfr_cont_ev_table.c BYTE-FOR-BYTE
 * in shape:
 *   - splitmix64 hash mixer (compatible with cev_mix64 / hash_combine)
 *   - 4096-probe linear probing
 *   - 0/1/2 CAS state machine on `occupied` with publish + spin-wait
 *   - pre-publish dedup scan to guard multi-thread races
 *   - CAS-loop atomic float add (type-punned uint32) for Hogwild accumulation
 *
 * Keeping the primitives local (rather than sharing cont_ev's static ones)
 * keeps this a self-contained TU: the unit test can build with just this
 * file + mccfr_cont_ev_table.c (for the shared key-hash) and not pull in the
 * trainer's card_abstraction / hand_eval / OMP dependency chain.
 *
 * Self-contained — no dependency on the trainer's heavyweight modules.
 */

#include "mccfr_bias_regrets.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

/* ── Probe / hashing primitives (local copies; math identical to
 * mccfr_cont_ev_table.c::cev_mix64 / cev_hash_combine — if you change one,
 * change all three so producer/consumer keys can't drift). ─────────────── */
#define BP_BIAS_REGRETS_PROBE_LIMIT 4096

static inline uint64_t bre_mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline uint64_t bre_hash_combine(uint64_t a, uint64_t b) {
    return bre_mix64(a ^ bre_mix64(b));
}

/* Slot hash over the (action_hash, player, bucket) triple — byte-identical
 * to cev_compute_slot_hash, so a given key lands in the same probe order in
 * both tables (not required for correctness, but keeps the two tables'
 * behavior symmetric and debuggable). The action_hash passed in is ALREADY
 * street-salted by the caller. */
static inline uint64_t bre_compute_slot_hash(uint64_t action_hash,
                                             int player, int bucket) {
    uint64_t h = bre_hash_combine(action_hash, (uint64_t)player);
    h = bre_hash_combine(h, (uint64_t)bucket);
    return h;
}

static int bre_is_prime_i64(int64_t n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if ((n & 1) == 0) return 0;
    for (int64_t d = 3; d * d <= n; d += 2)
        if (n % d == 0) return 0;
    return 1;
}

static int64_t bre_next_prime_i64(int64_t n) {
    if (n <= 2) return 2;
    if ((n & 1) == 0) n++;
    while (!bre_is_prime_i64(n)) n += 2;
    return n;
}

/* Spin-wait until a state=2 slot finishes initializing (becomes 1).
 * Mirrors cev_spin_until_ready. */
static inline void bre_spin_until_ready(const int *flag) {
    int spins = 0;
    while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) == 2) {
        #ifdef __x86_64__
        __builtin_ia32_pause();
        #elif defined(__aarch64__)
        __asm__ volatile("yield");
        #endif
        if (++spins > 10000000) {
            spins = 0;
            sched_yield();
        }
    }
}

static inline int bre_key_eq(const BPBiasRegretEntry *e,
                             uint64_t action_hash, int player, int bucket) {
    return e->action_hash == action_hash &&
           e->player == player &&
           e->bucket == bucket;
}

/* ── Public API ─────────────────────────────────────────────────── */

int bp_bias_regrets_init(BPBiasRegretTable *t, int64_t table_size) {
    if (t == NULL) return -1;
    memset(t, 0, sizeof(*t));
    if (table_size <= 0) {
        /* Sentinel: mode OFF / un-allocated. find_or_create returns NULL. */
        return 0;
    }
    int64_t prime_size = bre_next_prime_i64(table_size);
    BPBiasRegretEntry *entries = (BPBiasRegretEntry *)calloc(
        (size_t)prime_size, sizeof(BPBiasRegretEntry));
    if (entries == NULL) return -1;
    t->entries = entries;
    t->table_size = prime_size;
    t->num_entries = 0;
    t->insertion_failures = 0;
    t->max_probe_observed = 0;
    return 0;
}

void bp_bias_regrets_free(BPBiasRegretTable *t) {
    if (t == NULL) return;
    if (t->entries != NULL) {
        free(t->entries);
    }
    memset(t, 0, sizeof(*t));
}

int64_t bp_bias_regrets_count(const BPBiasRegretTable *t) {
    if (t == NULL) return 0;
    return __atomic_load_n(&t->num_entries, __ATOMIC_RELAXED);
}

BPBiasRegretEntry *bp_bias_regrets_find(const BPBiasRegretTable *t,
                                        uint64_t action_hash,
                                        int player, int bucket) {
    if (t == NULL || t->entries == NULL || t->table_size <= 0) return NULL;
    uint64_t h = bre_compute_slot_hash(action_hash, player, bucket);
    int64_t slot = (int64_t)(h % (uint64_t)t->table_size);
    for (int probe = 0; probe < BP_BIAS_REGRETS_PROBE_LIMIT; probe++) {
        int64_t idx = (slot + probe) % t->table_size;
        int state = __atomic_load_n(&t->entries[idx].occupied, __ATOMIC_ACQUIRE);
        if (state == 0) return NULL;
        if (state == 2) continue;
        if (bre_key_eq(&t->entries[idx], action_hash, player, bucket))
            return &t->entries[idx];
    }
    return NULL;
}

BPBiasRegretEntry *bp_bias_regrets_find_or_create(BPBiasRegretTable *t,
                                                  uint64_t action_hash,
                                                  int player, int bucket) {
    if (t == NULL || t->entries == NULL || t->table_size <= 0) return NULL;

    uint64_t h = bre_compute_slot_hash(action_hash, player, bucket);
    int64_t slot = (int64_t)(h % (uint64_t)t->table_size);

    for (int probe = 0; probe < BP_BIAS_REGRETS_PROBE_LIMIT; probe++) {
        int64_t idx = (slot + probe) % t->table_size;
        int state = __atomic_load_n(&t->entries[idx].occupied, __ATOMIC_ACQUIRE);

        if (state == 1) {
            if (bre_key_eq(&t->entries[idx], action_hash, player, bucket)) {
                if (probe > 0) {
                    int64_t cur = __atomic_load_n(&t->max_probe_observed,
                                                   __ATOMIC_RELAXED);
                    while ((int64_t)probe > cur &&
                           !__atomic_compare_exchange_n(&t->max_probe_observed,
                                                         &cur, (int64_t)probe,
                                                         1, __ATOMIC_RELAXED,
                                                         __ATOMIC_RELAXED)) { }
                }
                return &t->entries[idx];
            }
            continue;
        }

        if (state == 0) {
            int expected = 0;
            if (__atomic_compare_exchange_n(&t->entries[idx].occupied,
                                             &expected, 2,
                                             0, __ATOMIC_ACQ_REL,
                                             __ATOMIC_ACQUIRE)) {
                /* We won this slot (0 → 2). Pre-publish dedup scan of the
                 * earlier probe chain — mirrors cont_ev_find_or_create:
                 * under contention two threads can CAS-win different slots
                 * in the same chain and both publish before the other's
                 * post-publish scan; a late-comer keyed at the duplicate
                 * would return it and never reach the canonical. Rolling
                 * back to the earlier match eliminates that window. */
                if (probe > 0) {
                    for (int p2 = 0; p2 < probe; p2++) {
                        int64_t idx2 = (slot + p2) % t->table_size;
                        int st2 = __atomic_load_n(&t->entries[idx2].occupied,
                                                   __ATOMIC_ACQUIRE);
                        if (st2 == 2) {
                            bre_spin_until_ready(&t->entries[idx2].occupied);
                            st2 = __atomic_load_n(&t->entries[idx2].occupied,
                                                   __ATOMIC_ACQUIRE);
                        }
                        if (st2 == 1 && bre_key_eq(&t->entries[idx2],
                                                   action_hash, player, bucket)) {
                            __atomic_store_n(&t->entries[idx].occupied, 0,
                                             __ATOMIC_RELEASE);
                            return &t->entries[idx2];
                        }
                    }
                }

                /* No earlier match — we ARE the canonical. Initialize key
                 * fields and publish. regret/strategy_sum/value_* are zeroed
                 * by calloc and stay zero until the first update. */
                t->entries[idx].action_hash = action_hash;
                t->entries[idx].player = player;
                t->entries[idx].bucket = bucket;
                __atomic_store_n(&t->entries[idx].occupied, 1,
                                 __ATOMIC_RELEASE);

                __atomic_fetch_add(&t->num_entries, 1, __ATOMIC_RELAXED);
                if (probe > 0) {
                    int64_t cur = __atomic_load_n(&t->max_probe_observed,
                                                   __ATOMIC_RELAXED);
                    while ((int64_t)probe > cur &&
                           !__atomic_compare_exchange_n(&t->max_probe_observed,
                                                         &cur, (int64_t)probe,
                                                         1, __ATOMIC_RELAXED,
                                                         __ATOMIC_RELAXED)) { }
                }
                return &t->entries[idx];
            }
            /* CAS lost — re-read state and fall through. */
            state = __atomic_load_n(&t->entries[idx].occupied,
                                     __ATOMIC_ACQUIRE);
        }

        if (state == 2) {
            bre_spin_until_ready(&t->entries[idx].occupied);
            if (__atomic_load_n(&t->entries[idx].occupied,
                                 __ATOMIC_ACQUIRE) == 1) {
                if (bre_key_eq(&t->entries[idx], action_hash, player, bucket))
                    return &t->entries[idx];
            }
            continue;
        }
    }
    /* Probe cap exhausted — table too full or pathologically clustered. */
    __atomic_fetch_add(&t->insertion_failures, 1, __ATOMIC_RELAXED);
    {
        int64_t cur = __atomic_load_n(&t->max_probe_observed, __ATOMIC_RELAXED);
        int64_t want = (int64_t)BP_BIAS_REGRETS_PROBE_LIMIT;
        while (want > cur &&
               !__atomic_compare_exchange_n(&t->max_probe_observed, &cur, want,
                                             1, __ATOMIC_RELAXED,
                                             __ATOMIC_RELAXED)) { }
    }
    return NULL;
}

void bp_bias_regrets_atomic_add_f32(float *dst, float val) {
    /* CAS loop; type-pun via uint32_t — identical to
     * bp_cont_ev_atomic_add_f32. */
    _Static_assert(sizeof(float) == sizeof(uint32_t),
                   "bp_bias_regrets_atomic_add_f32: float is not uint32_t-sized");
    uint32_t *as_u32 = (uint32_t *)dst;
    uint32_t old_u32 = __atomic_load_n(as_u32, __ATOMIC_RELAXED);
    for (;;) {
        float old_f;
        memcpy(&old_f, &old_u32, sizeof(float));
        float new_f = old_f + val;
        uint32_t new_u32;
        memcpy(&new_u32, &new_f, sizeof(float));
        if (__atomic_compare_exchange_n(as_u32, &old_u32, new_u32,
                                         0, __ATOMIC_RELEASE,
                                         __ATOMIC_RELAXED)) {
            return;
        }
        /* CAS failed — old_u32 now holds the current value. Retry. */
    }
}

/* ── Regret matching (sample + strategy readout) ────────────────────
 *
 * Standard CFR regret matching over the positive-regret mass. The
 * distribution is IDENTICAL in shape to mccfr_blueprint.c::regret_match
 * (positive-clipped, normalized; uniform fallback when the positive mass is
 * zero). Split into a shared probability builder + a one-draw sampler so the
 * sampler and the strategy readout can't diverge. */

static void bre_build_probs(const BPBiasRegretEntry *e, float *probs) {
    float sum = 0.0f;
    if (e != NULL) {
        for (int b = 0; b < BP_CONT_EV_NUM_BIASES; b++) {
            float pos = (e->regret[b] > 0.0f) ? e->regret[b] : 0.0f;
            probs[b] = pos;
            sum += pos;
        }
    } else {
        /* Unseen slot: sum stays 0 → uniform fallback below. */
        for (int b = 0; b < BP_CONT_EV_NUM_BIASES; b++) probs[b] = 0.0f;
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int b = 0; b < BP_CONT_EV_NUM_BIASES; b++) probs[b] *= inv;
    } else {
        /* All regrets <= 0 (or unseen): uniform. Matches the legacy uniform
         * bias sampler's distribution exactly. */
        float u = 1.0f / (float)BP_CONT_EV_NUM_BIASES;
        for (int b = 0; b < BP_CONT_EV_NUM_BIASES; b++) probs[b] = u;
    }
}

/* xorshift64 — kept identical to mccfr_blueprint.c::rng_next so the sampler
 * consumes the trainer's per-thread RNG stream in lock-step. Inlined here so
 * this TU is self-contained. */
static inline uint64_t bre_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

int bp_bias_regrets_sample(const BPBiasRegretTable *t,
                           uint64_t action_hash, int player, int bucket,
                           uint64_t *rng_state) {
    const BPBiasRegretEntry *e =
        bp_bias_regrets_find(t, action_hash, player, bucket);

    float probs[BP_CONT_EV_NUM_BIASES];
    bre_build_probs(e, probs);

    /* Exactly ONE rng draw — upper-24-bit uniform float, byte-identical to
     * mccfr_blueprint.c::rng_float. Cumulative walk mirrors sample_action. */
    float r = (float)(bre_rng_next(rng_state) & 0xFFFFFF) / (float)0x1000000;
    float cumulative = 0.0f;
    for (int b = 0; b < BP_CONT_EV_NUM_BIASES; b++) {
        cumulative += probs[b];
        if (r <= cumulative) return b;
    }
    return BP_CONT_EV_NUM_BIASES - 1;
}

void bp_bias_regrets_strategy(const BPBiasRegretTable *t,
                              uint64_t action_hash, int player, int bucket,
                              float *out) {
    const BPBiasRegretEntry *e =
        bp_bias_regrets_find(t, action_hash, player, bucket);
    bre_build_probs(e, out);
}

/* ── Outcome-based regret update (item 4 — the one open design point) ──
 *
 * WHY THIS ESTIMATOR (bias / variance tradeoff, stated plainly for review):
 *
 * The mean-field target wants each opponent slot's bias mix to converge, by
 * regret matching, toward the biases that do well FOR THAT OPPONENT against
 * the current mixture of the other players. Full CFR regret matching needs a
 * per-bias COUNTERFACTUAL value v(b) for every b — "what would this opponent
 * have scored had it played bias b, holding everyone else's play fixed."
 *
 * The boundary all-biases sweep does NOT hand us those. Each sweep rollout
 * fixes ONE joint bias tuple (traverser bias forced to the swept index,
 * opponents' biases sampled once) and deals fresh cards, then walks the
 * descendant subtree once. It yields the value of the SAMPLED bias only; it
 * gives no counterfactual for the un-sampled opponent biases without extra
 * rollouts. The HONESTY RULE (Unit-2 item 4) is explicit: do NOT silently add
 * rollouts to manufacture the missing counterfactuals — that would blow the
 * 2.3s/3.8s armed-latency budget the whole cont_ev design exists to protect.
 *
 * So this is a MONTE-CARLO EXTERNAL-SAMPLING regret update, not a full-sweep
 * counterfactual one:
 *
 *   regret[sampled] += (value - baseline)
 *
 * where `value` is the opponent's realized value at this boundary and
 * `baseline` is the slot's running mean value across all biases sampled so
 * far (value_sum / value_count summed over biases). The un-sampled biases'
 * regrets are left UNCHANGED this visit — they get their turn when regret
 * matching samples them, so over many visits every bias is sampled and
 * updated (coupon-collector coverage, same shape as the cont_ev table's own
 * per-bias coverage).
 *
 * Properties:
 *  - Sound direction: E[value | bias b] > slot mean ⇒ regret[b] drifts up ⇒
 *    RM raises P(b). Fixed points are the biases with above-average value —
 *    the mean-field best responses. This is the standard outcome-sampling
 *    MCCFR regret signal (Lanctot et al. 2009), specialized to a 4-action
 *    "which continuation bias" node.
 *  - Baseline = control variate: subtracting the running mean removes the
 *    common additive offset in the boundary value (pot size, position), which
 *    is the dominant variance term. Without it, regrets track the absolute
 *    chip value and RM degenerates toward whichever bias was visited first.
 *  - BIAS of the estimator: outcome sampling gives an UNBIASED estimate of
 *    the sampled action's advantage ONLY in expectation over the sampling
 *    distribution; because we sample from the CURRENT (evolving) RM mix rather
 *    than uniformly, early visits are correlated with the mix and the fixed
 *    point is a mean-field (per-player marginal) equilibrium, NOT the literal
 *    joint-bias equilibrium (that one is permanently walled — 4^(n-1) table).
 *    This is the accepted Stage-1 approximation (STAGE1 plan "mean-field").
 *  - VARIANCE: one sample per visit per bias ⇒ high variance early; the
 *    walk's iteration count (billions) + the baseline keep it usable. The
 *    Stage-2 equity gate is what actually validates whether the resulting
 *    leaf values beat the uniform baseline; this estimator only has to move
 *    the mix in the right direction cheaply.
 *
 * VALUE CONVENTION (`value`): the caller passes the OPPONENT's value at the
 * boundary. In the HU-collapsed sense that is -boundary_ev (the traverser's
 * gain is the opponent's loss). At a true MULTIWAY boundary, -boundary_ev is
 * the opponent's value only in the 2-player residual; with >2 active players
 * it is an APPROXIMATION (it ignores the third-party split). We use it anyway
 * because (a) it is the only opponent-side signal the sweep produces without
 * extra rollouts, and (b) the sign is correct — a bias that lets the traverser
 * profit more is worse for the opponent — which is all RM needs to move the
 * mix. The residual multiway inaccuracy is folded into the estimator's
 * documented bias and is exactly the kind of thing the Stage-2 gate measures.
 *
 * Thread-safety: value_sum/regret/strategy_sum via CAS-loop float add;
 * value_count via relaxed fetch_add. Hogwild-tolerant, same as cont_ev. */
void bp_bias_regrets_update(BPBiasRegretEntry *e, int sampled_bias,
                            float value) {
    if (e == NULL) return;
    if (sampled_bias < 0 || sampled_bias >= BP_CONT_EV_NUM_BIASES) return;

    /* 1. Accumulate the observed value into the sampled bias's running mean. */
    bp_bias_regrets_atomic_add_f32(&e->value_sum[sampled_bias], value);
    __atomic_fetch_add(&e->value_count[sampled_bias], 1, __ATOMIC_RELAXED);

    /* 2. Baseline = mean observed value across ALL biases seen so far.
     * Snapshot loads (RELAXED) — Hogwild slop in the baseline is tolerated
     * exactly like the regret table's slop; the baseline only needs to track
     * the slot's rough average. */
    float total_sum = 0.0f;
    int64_t total_cnt = 0;
    for (int b = 0; b < BP_CONT_EV_NUM_BIASES; b++) {
        uint32_t u = __atomic_load_n((uint32_t *)&e->value_sum[b],
                                     __ATOMIC_RELAXED);
        float f;
        memcpy(&f, &u, sizeof(float));
        total_sum += f;
        total_cnt += __atomic_load_n(&e->value_count[b], __ATOMIC_RELAXED);
    }
    float baseline = (total_cnt > 0) ? (total_sum / (float)total_cnt) : 0.0f;

    /* 3. Outcome-based regret increment for the sampled bias. */
    bp_bias_regrets_atomic_add_f32(&e->regret[sampled_bias], value - baseline);

    /* 4. Average-strategy accumulation (visit count; optional, diagnostics). */
    bp_bias_regrets_atomic_add_f32(&e->strategy_sum[sampled_bias], 1.0f);
}
