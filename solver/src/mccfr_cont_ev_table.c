/**
 * mccfr_cont_ev_table.c — Implementation of the cont-bias boundary EV
 * slot table for Phase R Block A.2.
 *
 * See mccfr_cont_ev_table.h for the contract. This file mirrors the
 * concurrency primitives of mccfr_blueprint.c:
 *   - splitmix64 hash mixer (compatible with hash_combine in trainer)
 *   - 4096-probe linear probing
 *   - 0/1/2 state machine on `occupied` with CAS publish + spin-wait
 *   - de-dup scan on insertion to guard against multi-thread races
 *
 * Self-contained — does not depend on the trainer's heavyweight
 * dependencies (card_abstraction, hand_eval, OpenMP runtime).
 */

#include "mccfr_cont_ev_table.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

/* ── Probe / hashing primitives ──────────────────────────────────────
 *
 * Local copies of trainer's splitmix64 + hash_combine. Keeping them
 * static here (rather than linking against mccfr_blueprint.c) so that
 * the unit test can build with just this TU and not pull in the
 * trainer's full dep chain. The math is byte-identical to
 * mccfr_blueprint.c::bp_mix64 / hash_combine; if you change one,
 * change the other.
 */
#define BP_CONT_EV_PROBE_LIMIT 4096

static inline uint64_t cev_mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline uint64_t cev_hash_combine(uint64_t a, uint64_t b) {
    return cev_mix64(a ^ cev_mix64(b));
}

static inline uint64_t cev_compute_slot_hash(uint64_t action_hash,
                                              int player, int bucket) {
    uint64_t h = cev_hash_combine(action_hash, (uint64_t)player);
    h = cev_hash_combine(h, (uint64_t)bucket);
    return h;
}

/* Round n up to the next prime. Same trick as info_table_init
 * (mccfr_blueprint.c:565). Fast for the size range we care about
 * (1<<10 .. 1<<27); trial-division is fine. */
static int cev_is_prime_i64(int64_t n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if ((n & 1) == 0) return 0;
    for (int64_t d = 3; d * d <= n; d += 2)
        if (n % d == 0) return 0;
    return 1;
}

static int64_t cev_next_prime_i64(int64_t n) {
    if (n <= 2) return 2;
    if ((n & 1) == 0) n++;
    while (!cev_is_prime_i64(n)) n += 2;
    return n;
}

/* Spin-wait until a state=2 slot finishes initializing (becomes 1).
 * Mirrors mccfr_blueprint.c::spin_until_ready. */
static inline void cev_spin_until_ready(const int *flag) {
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

static inline int cev_key_eq(const BPContEVEntry *e,
                              uint64_t action_hash, int player, int bucket) {
    return e->action_hash == action_hash &&
           e->player == player &&
           e->bucket == bucket;
}

/* ── Public API ─────────────────────────────────────────────────── */

int bp_cont_ev_init(BPContEVTable *t, int64_t table_size) {
    if (t == NULL) return -1;
    memset(t, 0, sizeof(*t));
    if (table_size <= 0) {
        /* Sentinel: feature OFF. find_or_create returns NULL. */
        return 0;
    }
    int64_t prime_size = cev_next_prime_i64(table_size);
    BPContEVEntry *entries = (BPContEVEntry *)calloc(
        (size_t)prime_size, sizeof(BPContEVEntry));
    if (entries == NULL) return -1;
    t->entries = entries;
    t->table_size = prime_size;
    t->num_entries = 0;
    t->insertion_failures = 0;
    t->max_probe_observed = 0;
    return 0;
}

void bp_cont_ev_free(BPContEVTable *t) {
    if (t == NULL) return;
    if (t->entries != NULL) {
        free(t->entries);
    }
    memset(t, 0, sizeof(*t));
}

int64_t bp_cont_ev_count(const BPContEVTable *t) {
    if (t == NULL) return 0;
    return __atomic_load_n(&t->num_entries, __ATOMIC_RELAXED);
}

BPContEVEntry *bp_cont_ev_find(const BPContEVTable *t,
                                uint64_t action_hash,
                                int player, int bucket) {
    if (t == NULL || t->entries == NULL || t->table_size <= 0) return NULL;
    uint64_t h = cev_compute_slot_hash(action_hash, player, bucket);
    int64_t slot = (int64_t)(h % (uint64_t)t->table_size);
    for (int probe = 0; probe < BP_CONT_EV_PROBE_LIMIT; probe++) {
        int64_t idx = (slot + probe) % t->table_size;
        int state = __atomic_load_n(&t->entries[idx].occupied, __ATOMIC_ACQUIRE);
        if (state == 0) return NULL;
        if (state == 2) continue;
        if (cev_key_eq(&t->entries[idx], action_hash, player, bucket))
            return &t->entries[idx];
    }
    return NULL;
}

BPContEVEntry *bp_cont_ev_find_or_create(BPContEVTable *t,
                                          uint64_t action_hash,
                                          int player, int bucket) {
    if (t == NULL || t->entries == NULL || t->table_size <= 0) return NULL;

    uint64_t h = cev_compute_slot_hash(action_hash, player, bucket);
    int64_t slot = (int64_t)(h % (uint64_t)t->table_size);

    for (int probe = 0; probe < BP_CONT_EV_PROBE_LIMIT; probe++) {
        int64_t idx = (slot + probe) % t->table_size;
        int state = __atomic_load_n(&t->entries[idx].occupied, __ATOMIC_ACQUIRE);

        if (state == 1) {
            if (cev_key_eq(&t->entries[idx], action_hash, player, bucket)) {
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
                /* We won this slot (state went 0 → 2). Before writing
                 * our key fields and publishing, scan earlier slots in
                 * the probe chain to see if another thread already
                 * published the same key. If yes — we'd be a duplicate;
                 * roll back our CAS (state 2 → 0) and return the
                 * earlier slot, ensuring all concurrent inserters
                 * converge on the same canonical slot.
                 *
                 * Pre-publish dedup matters: under high contention, two
                 * threads can each CAS-win different slots in the same
                 * probe chain and both publish their duplicates before
                 * the post-publish scan runs. A late-comer entering
                 * mid-init at the duplicate would key_eq match it and
                 * return the duplicate, never reaching the canonical.
                 * Pre-publish-rollback eliminates the window. */
                if (probe > 0) {
                    for (int p2 = 0; p2 < probe; p2++) {
                        int64_t idx2 = (slot + p2) % t->table_size;
                        int st2 = __atomic_load_n(&t->entries[idx2].occupied,
                                                   __ATOMIC_ACQUIRE);
                        if (st2 == 2) {
                            cev_spin_until_ready(&t->entries[idx2].occupied);
                            st2 = __atomic_load_n(&t->entries[idx2].occupied,
                                                   __ATOMIC_ACQUIRE);
                        }
                        if (st2 == 1 && cev_key_eq(&t->entries[idx2],
                                                    action_hash, player, bucket)) {
                            /* Earlier match exists. Roll back our CAS so
                             * this slot is reusable, and return the
                             * canonical slot. */
                            __atomic_store_n(&t->entries[idx].occupied, 0,
                                             __ATOMIC_RELEASE);
                            return &t->entries[idx2];
                        }
                    }
                }

                /* No earlier match — we ARE the canonical. Initialize
                 * key fields and publish. */
                t->entries[idx].action_hash = action_hash;
                t->entries[idx].player = player;
                t->entries[idx].bucket = bucket;
                /* ev_sum / ev_count are zeroed by calloc and stay zero
                 * until the trainer accumulates the first sample. */
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
            cev_spin_until_ready(&t->entries[idx].occupied);
            if (__atomic_load_n(&t->entries[idx].occupied,
                                 __ATOMIC_ACQUIRE) == 1) {
                if (cev_key_eq(&t->entries[idx], action_hash, player, bucket))
                    return &t->entries[idx];
            }
            continue;
        }
    }
    /* Probe cap exhausted — table too full or pathologically clustered. */
    __atomic_fetch_add(&t->insertion_failures, 1, __ATOMIC_RELAXED);
    {
        int64_t cur = __atomic_load_n(&t->max_probe_observed, __ATOMIC_RELAXED);
        int64_t want = (int64_t)BP_CONT_EV_PROBE_LIMIT;
        while (want > cur &&
               !__atomic_compare_exchange_n(&t->max_probe_observed, &cur, want,
                                             1, __ATOMIC_RELAXED,
                                             __ATOMIC_RELAXED)) { }
    }
    return NULL;
}

void bp_cont_ev_atomic_add_f32(float *dst, float val) {
    /* CAS loop: load current, compute new, CAS-or-retry. C11 stdatomic
     * doesn't expose float atomics; we type-pun via uint32_t. */
    _Static_assert(sizeof(float) == sizeof(uint32_t),
                   "bp_cont_ev_atomic_add_f32: float is not uint32_t-sized");
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
        /* CAS failed — old_u32 has been updated to the current value.
         * Retry with the fresh load. */
    }
}

void bp_cont_ev_apply_discount(BPContEVTable *t, float discount) {
    if (t == NULL || t->entries == NULL) return;
    /* Sweeps the entire table once. Mirrors apply_discount in
     * mccfr_blueprint.c (line 2276) — the trainer wires this from the
     * same `if (do_discount)` block. We don't add an `omp for` pragma
     * here because this TU is intentionally OMP-free; the caller can
     * wrap the call in their own `#pragma omp for` if they want
     * parallel discount sweeps. */
    for (int64_t i = 0; i < t->table_size; i++) {
        if (__atomic_load_n(&t->entries[i].occupied, __ATOMIC_ACQUIRE) != 1)
            continue;
        for (int b = 0; b < BP_CONT_EV_NUM_BIASES; b++) {
            t->entries[i].ev_sum[b] *= discount;
        }
        /* ev_count is NOT discounted — it's a raw visit count. The
         * `ev_sum / ev_count` ratio at query time gives a Linear-CFR-
         * weighted EV per visit, which is what the kernel reads. */
    }
}
