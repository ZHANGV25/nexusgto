/*
 * test_eval5_table.c - exhaustive equivalence gate for the table-backed eval5.
 *
 * Walks all C(52,5) = 2,598,960 five-card sets in the same colex order the
 * builder uses and asserts, per set:
 *   1. poker_eval5_index(sorted) == the running colex counter (index formula
 *      AND fill-order agreement - value ties between different sets cannot
 *      mask an index bug because the index itself is checked);
 *   2. poker_eval5_index(permuted) == the same counter, cycling through all
 *      120 permutations of the 5 positions across sets (sorting network);
 *   3. eval5(permuted) == eval5_compute(sorted) (end-to-end table probe).
 * Requires the table to actually be built (no silent fallback pass).
 */
#include <stdio.h>
#include <stdlib.h>
#include "hand_eval.h"

#ifndef POKER_EVAL5_TABLE

int main(void) {
    printf("SKIP: built with POKER_EVAL5_TABLE_DISABLE\n");
    return 0;
}

#else

static int perms[120][5];
static int n_perms = 0;

static void gen_perms(int *a, int k) {
    if (k == 5) {
        for (int i = 0; i < 5; i++) perms[n_perms][i] = a[i];
        n_perms++;
        return;
    }
    for (int i = k; i < 5; i++) {
        int t = a[k]; a[k] = a[i]; a[i] = t;
        gen_perms(a, k + 1);
        t = a[k]; a[k] = a[i]; a[i] = t;
    }
}

int main(void) {
    poker_eval5_table_init();
    if (!poker_eval5_table) {
        fprintf(stderr, "FAIL: eval5 table not built (allocation failure?)\n");
        return 1;
    }
    int base[5] = {0, 1, 2, 3, 4};
    gen_perms(base, 0);
    if (n_perms != 120) {
        fprintf(stderr, "FAIL: generated %d permutations (want 120)\n", n_perms);
        return 1;
    }

    uint32_t idx = 0;
    long long bad = 0;
    for (int c4 = 4; c4 < 52; c4++)
    for (int c3 = 3; c3 < c4; c3++)
    for (int c2 = 2; c2 < c3; c2++)
    for (int c1 = 1; c1 < c2; c1++)
    for (int c0 = 0; c0 < c1; c0++) {
        const int cs[5] = {c0, c1, c2, c3, c4};
        const int *p = perms[idx % 120];
        int p0 = cs[p[0]], p1 = cs[p[1]], p2 = cs[p[2]],
            p3 = cs[p[3]], p4 = cs[p[4]];
        uint32_t want = eval5_compute(c0, c1, c2, c3, c4);
        if (poker_eval5_index(c0, c1, c2, c3, c4) != idx
            || poker_eval5_index(p0, p1, p2, p3, p4) != idx
            || eval5(p0, p1, p2, p3, p4) != want) {
            if (bad < 10)
                fprintf(stderr, "FAIL at colex %u: cards %d %d %d %d %d\n",
                        (unsigned)idx, c0, c1, c2, c3, c4);
            bad++;
        }
        idx++;
    }
    if (idx != POKER_EVAL5_TABLE_SIZE) {
        fprintf(stderr, "FAIL: enumerated %u sets (want %u)\n",
                (unsigned)idx, (unsigned)POKER_EVAL5_TABLE_SIZE);
        return 1;
    }
    if (bad) {
        fprintf(stderr, "FAIL: %lld mismatches over %u sets\n",
                bad, (unsigned)idx);
        return 1;
    }
    printf("PASS: table-backed eval5 == legacy over all %u five-card sets "
           "(index + network + value)\n", (unsigned)idx);
    return 0;
}

#endif /* POKER_EVAL5_TABLE */
