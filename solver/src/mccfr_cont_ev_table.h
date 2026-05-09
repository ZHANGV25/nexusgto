/**
 * mccfr_cont_ev_table.h — Phase R cont-bias boundary EV slot table.
 *
 * Open-addressed hash table parallel to BPInfoTable, keyed on
 * (action_hash_at_boundary, traverser, bucket). Stores per-bias
 * EV running sums + visit counts for the multiway-flop boundary.
 *
 * Spec: docs/PHASE_R_TRAINER_SPEC.md §4 (slot layout) and §3 (Linear-
 * CFR semantics — ev_sum is discounted at periodic ticks; ev_count is
 * unweighted).
 *
 * Block A.2 deliverable. Lazy allocation on first
 * bp_set_depth_limit_cont_strats(s, 1) call; not allocated when the
 * feature is OFF (so legacy / Phase 3 runs are unaffected).
 *
 * Self-contained TU — does not depend on card_abstraction / hand_eval
 * / OMP, so the unit test can link against just this file.
 */
#ifndef MCCFR_CONT_EV_TABLE_H
#define MCCFR_CONT_EV_TABLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of cont-bias slots per (action_hash, player, bucket) entry.
 * Matches Pluribus §C2: 0=baseline, 1=fold-bias, 2=call/check-bias,
 * 3=raise/bet-bias. Mirrors src/cuda/street_solve.cuh:41
 * `#define SS_NUM_CONT_STRATS 4`. */
#define BP_CONT_EV_NUM_BIASES 4

/* One slot in the cont-ev hash table. Atomic CAS state machine on
 * `occupied` mirrors info_table_find_or_create:
 *   0 = empty
 *   1 = ready (key fields populated, ev_sum/ev_count valid)
 *   2 = initializing (some thread is mid-publish; spin-wait) */
typedef struct {
    /* Key. */
    uint64_t action_hash;        /* compute_action_hash at boundary position */
    int player;                  /* traverser identity (slot owner) */
    int bucket;                  /* traverser's flop bucket */
    /* CAS state machine. */
    int occupied;
    /* Value (indexed by traverser's cont_bias). */
    float ev_sum[BP_CONT_EV_NUM_BIASES];      /* Linear-CFR-discounted */
    int64_t ev_count[BP_CONT_EV_NUM_BIASES];  /* unweighted visit count */
} BPContEVEntry;

typedef struct {
    BPContEVEntry *entries;          /* calloc'd; NULL means "feature OFF" */
    int64_t table_size;              /* prime-promoted count of entries */
    int64_t num_entries;             /* relaxed atomic */
    int64_t insertion_failures;      /* relaxed atomic; probe-cap exhaustions */
    int64_t max_probe_observed;      /* relaxed atomic */
} BPContEVTable;

/**
 * Allocate the cont-ev hash table.
 *
 * `table_size` is promoted to the next prime to eliminate modular
 * collision patterns (same trick as info_table_init in
 * mccfr_blueprint.c:572). Pass 0 to leave the table un-allocated
 * (entries == NULL) — in which case bp_cont_ev_find_or_create returns
 * NULL.
 *
 * Returns 0 on success, -1 on OOM or invalid size.
 */
int bp_cont_ev_init(BPContEVTable *t, int64_t table_size);

/**
 * Free the entries array and zero the table struct.
 * Safe to call on a zero-init or already-freed table.
 */
void bp_cont_ev_free(BPContEVTable *t);

/**
 * Lock-free find-or-create with atomic CAS on the occupied flag.
 * Mirrors info_table_find_or_create (mccfr_blueprint.c:638) — the
 * same 0→2→1 state machine, the same de-dup-on-publish scan, the
 * same probe cap.
 *
 * Returns a pointer to the entry, or NULL if:
 *   - the table is unallocated (entries == NULL)
 *   - the probe cap is exhausted (table is too full)
 *
 * Concurrent calls with the same key all return the same entry.
 */
BPContEVEntry *bp_cont_ev_find_or_create(BPContEVTable *t,
                                          uint64_t action_hash,
                                          int player, int bucket);

/**
 * Read-only probe. Returns the matching entry, or NULL on miss.
 * Used by tests + the BPR5 reader path. No mutation, no allocation.
 */
BPContEVEntry *bp_cont_ev_find(const BPContEVTable *t,
                                uint64_t action_hash,
                                int player, int bucket);

/**
 * Number of slots currently occupied (atomic load; safe to call from
 * any thread). Matches the trainer's BPInfoTable::num_entries
 * convention.
 */
int64_t bp_cont_ev_count(const BPContEVTable *t);

/**
 * Hogwild-safe atomic float add via compare-and-swap loop.
 * Used to accumulate ev_sum across threads without taking a lock.
 * Mirrors the per-action regret update's Hogwild semantics
 * (mccfr_blueprint.c:1671-1682) but for float — int regret update is
 * just a non-atomic += because integers tolerate Hogwild slop;
 * float-precision-loss tolerance is similar but we add a CAS loop
 * because non-atomic float += is undefined-behavior-adjacent under
 * concurrent writes.
 */
void bp_cont_ev_atomic_add_f32(float *dst, float val);

/**
 * Apply a Linear-CFR (or DCFR) discount factor to ALL ev_sum entries
 * in the table. Mirrors apply_discount (mccfr_blueprint.c:2276) for
 * regrets — same `discount` value, same multiply-in-place semantics.
 *
 * ev_count is NOT discounted (per spec §4: "int64 ev_count[4] —
 * visit count per bias", unweighted).
 *
 * Caller wires this from the same `if (do_discount)` site that calls
 * apply_discount in bp_solve.
 */
void bp_cont_ev_apply_discount(BPContEVTable *t, float discount);

/* ── Boundary EV query (Block B.2) ─────────────────────────────────
 *
 * Read-only query API for the kernel/runtime side. Status enum forces
 * callers to check coverage rather than silently using a NaN sentinel.
 *
 * Per spec §5: NO silent fallback. Caller decides what EMPTY_BIAS or
 * MISS means in their context (kernel default per spec: fail loud). */

typedef enum {
    BP_BOUNDARY_EV_OK         = 0,  /* slot found, ev_count[bias] > 0 — *out_ev valid */
    BP_BOUNDARY_EV_EMPTY_BIAS = 1,  /* slot found, ev_count[bias] == 0 — *out_ev = NaN */
    BP_BOUNDARY_EV_MISS       = 2,  /* slot not in table — *out_ev = NaN */
    BP_BOUNDARY_EV_INVALID    = 3,  /* cont_bias out of range — *out_ev = NaN */
} BPBoundaryEVStatus;

/**
 * Query the cont-bias EV at a boundary slot.
 *
 * On OK, *out_ev = ev_sum[cont_bias] / ev_count[cont_bias] — the
 * Linear-CFR-weighted mean boundary EV from the traverser's
 * perspective for the specified cont-bias.
 *
 * On any non-OK status, *out_ev is set to a NaN sentinel. Callers
 * MUST check the returned status; do not consume *out_ev on non-OK.
 *
 * Pure read; safe to call concurrently with other readers (the
 * underlying find uses ACQUIRE loads).
 *
 * `bps_query_boundary_ev` is the kernel/Python-facing alias of this
 * function — same semantics, both symbols ship from
 * mccfr_cont_ev_table.c so no extra TU has to be linked into the
 * kernel-side build.
 */
BPBoundaryEVStatus bp_query_boundary_ev(const BPContEVTable *t,
                                         uint64_t action_hash,
                                         int player, int bucket,
                                         int cont_bias,
                                         float *out_ev);

BPBoundaryEVStatus bps_query_boundary_ev(const BPContEVTable *t,
                                          uint64_t action_hash,
                                          int player, int bucket,
                                          int cont_bias,
                                          float *out_ev);

#ifdef __cplusplus
}
#endif

#endif /* MCCFR_CONT_EV_TABLE_H */
