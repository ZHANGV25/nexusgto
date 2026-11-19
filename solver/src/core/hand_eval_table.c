/*
 * hand_eval_table.c - the single process-wide definition of the eval5 lookup
 * table declared in hand_eval.h.
 *
 * At library load (constructor) all C(52,5) = 2,598,960 five-card sets are
 * enumerated in colex order and ranked by eval5_compute - exact by
 * construction, nothing shipped on disk. The table pointer is published with
 * a release-store only AFTER the fill completes; eval5 acquire-loads it and
 * falls back to eval5_compute while NULL (pre-publish or alloc failure), so
 * every read is exact at every point in time and no reader ever blocks.
 *
 * Kill switch: -DPOKER_EVAL5_TABLE_DISABLE (hand_eval.h then routes eval5
 * straight to eval5_compute and this TU keeps only the no-op init).
 */
#include "hand_eval.h"

#include <stdlib.h>

#ifdef POKER_EVAL5_TABLE

const int32_t *poker_eval5_table = NULL;

static int32_t *poker_eval5_build(void) {
    int32_t *tab =
        (int32_t *)malloc((size_t)POKER_EVAL5_TABLE_SIZE * sizeof(int32_t));
    if (!tab) return NULL;
    /* Colex enumeration: this loop nest visits sets in exactly increasing
     * poker_eval5_index order, so idx++ matches the index formula (pinned
     * exhaustively by tests/test_eval5_table.c). */
    size_t idx = 0;
    for (int c4 = 4; c4 < 52; c4++)
        for (int c3 = 3; c3 < c4; c3++)
            for (int c2 = 2; c2 < c3; c2++)
                for (int c1 = 1; c1 < c2; c1++)
                    for (int c0 = 0; c0 < c1; c0++)
                        tab[idx++] = (int32_t)eval5_compute(c0, c1, c2, c3, c4);
    return tab;
}

/* Idempotent and thread-safe: exactly one caller wins the claim CAS and
 * builds; every other caller returns immediately (eval5 keeps the exact
 * compute fallback until the winner's release-store publishes). Normally the
 * loader constructor below runs this once per process before any solver
 * thread exists. The table is process-global and intentionally never freed. */
void poker_eval5_table_init(void) {
    static int claimed = 0;
    int expected = 0;
    if (!__atomic_compare_exchange_n(&claimed, &expected, 1, 0,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        return;
    int32_t *tab = poker_eval5_build();
    if (tab)
        __atomic_store_n(&poker_eval5_table, (const int32_t *)tab,
                         __ATOMIC_RELEASE);
    /* On alloc failure the pointer stays NULL: exact fallback, no retry. */
}

__attribute__((constructor))
static void poker_eval5_table_ctor(void) { poker_eval5_table_init(); }

#else  /* POKER_EVAL5_TABLE_DISABLE */

void poker_eval5_table_init(void) {}

#endif /* POKER_EVAL5_TABLE */
