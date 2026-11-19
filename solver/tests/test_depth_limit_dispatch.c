/**
 * test_depth_limit_dispatch — dispatch helper unit tests for the depth-limit
 * code (mccfr_phase_r.c).
 *
 * Verifies the contract of bp_should_dispatch_to_boundary and the
 * bit-identical regression invariant: with the flag OFF the helper
 * returns 0 and consumes no RNG.
 *
 * Runs in <1 s. Build: `make test_depth_limit_dispatch`.
 */

#include "mccfr_blueprint.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static int passed = 0;
static int failed = 0;

#define CHECK(cond, msg) do {                                              \
    if (cond) { passed++; }                                                \
    else {                                                                 \
        failed++;                                                          \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);      \
    }                                                                      \
} while (0)

/* T1 (byte-identity invariant): flag OFF returns 0 AND does NOT consume RNG. */
static void test_flag_off_no_rng_consumption(void) {
    fprintf(stderr, "[T1] flag=0 returns 0 deterministically without "
                    "consuming RNG\n");
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    uint64_t rng_before = rng;
    /* Try every combination of (num_board, n_active) — none should
     * dispatch when flag=0, none should mutate the RNG. */
    for (int nb = 0; nb <= 5; nb++) {
        for (int na = 1; na <= 6; na++) {
            int got = bp_should_dispatch_to_boundary(nb, na, 0, 0.5f, &rng);
            CHECK(got == 0,
                  "flag=0 dispatch returned !=0");
            CHECK(rng == rng_before,
                  "flag=0 dispatch consumed RNG (regression invariant broken)");
        }
    }
}

/* T2: HU flop with flag ON never terminates (matches the search tree's
 * flop rule, which extends when orig_num_players == 2 → HU extends). */
static void test_hu_flop_continues(void) {
    fprintf(stderr, "[T2] HU flop with flag=1 never terminates\n");
    uint64_t rng = 0xDEADBEEFCAFEBABEULL;
    for (int i = 0; i < 1000; i++) {
        int got = bp_should_dispatch_to_boundary(3, 2, 1, 1.0f, &rng);
        CHECK(got == 0, "HU flop with flag=1 unexpectedly terminated");
        if (got != 0) break;
    }
}

/* T3: non-flop num_board never dispatches regardless of n_active. */
static void test_non_flop_never_dispatches(void) {
    fprintf(stderr, "[T3] non-flop num_board never dispatches\n");
    uint64_t rng = 0x12345678ABCDEFULL;
    int test_n_actives[] = {2, 3, 4, 5, 6};
    int test_num_boards[] = {0, 1, 2, 4, 5};  /* preflop, weird, weird, turn, river */
    for (size_t bi = 0; bi < sizeof(test_num_boards)/sizeof(int); bi++) {
        for (size_t ai = 0; ai < sizeof(test_n_actives)/sizeof(int); ai++) {
            int nb = test_num_boards[bi];
            int na = test_n_actives[ai];
            int got = bp_should_dispatch_to_boundary(nb, na, 1, 1.0f, &rng);
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "non-flop dispatched at num_board=%d n_active=%d", nb, na);
            CHECK(got == 0, msg);
        }
    }
}

/* T4: multiway flop, prob=0.0 → never terminates. */
static void test_prob_zero_never_terminates(void) {
    fprintf(stderr, "[T4] prob=0.0 never terminates\n");
    uint64_t rng = 0xBADC0DEFEEDFACE0ULL;
    for (int n = 3; n <= 6; n++) {
        for (int i = 0; i < 1000; i++) {
            int got = bp_should_dispatch_to_boundary(3, n, 1, 0.0f, &rng);
            CHECK(got == 0, "prob=0.0 unexpectedly terminated");
            if (got != 0) break;
        }
    }
}

/* T5: multiway flop, prob=1.0 → always terminates (3p, 4p, 5p, 6p). */
static void test_prob_one_always_terminates(void) {
    fprintf(stderr, "[T5] prob=1.0 always terminates on multiway flop\n");
    uint64_t rng = 0xC0FFEE1234567890ULL;
    for (int n = 3; n <= 6; n++) {
        for (int i = 0; i < 1000; i++) {
            int got = bp_should_dispatch_to_boundary(3, n, 1, 1.0f, &rng);
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "prob=1.0 didn't terminate at n_active=%d", n);
            CHECK(got == 1, msg);
            if (got != 1) break;
        }
    }
}

/* T6: multiway flop, prob=0.5 → terminates ~half the time.
 * 100K trials, expect 50K ± ~1.5K (3σ for binomial p=0.5, n=100K). */
static void test_prob_half_balanced(void) {
    fprintf(stderr, "[T6] prob=0.5 terminates ~half the time (100K trials)\n");
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    int term_count = 0;
    const int N = 100000;
    for (int i = 0; i < N; i++) {
        term_count += bp_should_dispatch_to_boundary(3, 3, 1, 0.5f, &rng);
    }
    int lo = N / 2 - 2000;  /* ±2000 generous bound (~4σ) */
    int hi = N / 2 + 2000;
    char msg[128];
    snprintf(msg, sizeof(msg),
             "prob=0.5 yielded %d / %d terminations (expected ~50000, "
             "tolerated [%d, %d])",
             term_count, N, lo, hi);
    CHECK(term_count >= lo && term_count <= hi, msg);
    fprintf(stderr, "       observed: %d / %d (%.4f)\n",
            term_count, N, (double)term_count / N);
}

/* T7: prob=0.5 produces deterministic sequence given fixed seed.
 * Two runs with same seed produce identical (hit-or-miss) sequences. */
static void test_seed_determinism(void) {
    fprintf(stderr, "[T7] same RNG seed → same dispatch sequence\n");
    uint64_t rng_a = 0x9E3779B97F4A7C15ULL;
    uint64_t rng_b = 0x9E3779B97F4A7C15ULL;
    int diffs = 0;
    for (int i = 0; i < 10000; i++) {
        int a = bp_should_dispatch_to_boundary(3, 4, 1, 0.5f, &rng_a);
        int b = bp_should_dispatch_to_boundary(3, 4, 1, 0.5f, &rng_b);
        if (a != b) diffs++;
    }
    char msg[128];
    snprintf(msg, sizeof(msg),
             "seed determinism violated: %d disagreements in 10K trials",
             diffs);
    CHECK(diffs == 0, msg);
}

/* T8: different seeds → different (with overwhelming probability)
 * dispatch sequences. Sanity check that the RNG is actually being
 * threaded through; not strictly required but catches a "didn't
 * consume RNG when it should have" regression. */
static void test_seed_sensitivity(void) {
    fprintf(stderr, "[T8] different seeds → different dispatch sequences\n");
    uint64_t rng_a = 0x9E3779B97F4A7C15ULL;
    uint64_t rng_b = 0xDEADBEEFCAFEBABEULL;
    int diffs = 0;
    for (int i = 0; i < 10000; i++) {
        int a = bp_should_dispatch_to_boundary(3, 4, 1, 0.5f, &rng_a);
        int b = bp_should_dispatch_to_boundary(3, 4, 1, 0.5f, &rng_b);
        if (a != b) diffs++;
    }
    /* For p=0.5 IID, two independent sequences disagree on each trial
     * with probability 0.5; expected ~5000 disagreements in 10K, ±~150
     * for binomial 3σ. */
    char msg[128];
    snprintf(msg, sizeof(msg),
             "different seeds yielded only %d disagreements in 10K (RNG "
             "may not be threading through helper)", diffs);
    CHECK(diffs >= 4500 && diffs <= 5500, msg);
    fprintf(stderr, "       observed: %d / 10000 disagreements\n", diffs);
}

int main(void) {
    fprintf(stderr,
            "═══════════════════════════════════════════════════════════\n"
            "  Depth-limit boundary: dispatch helper unit tests\n"
            "  bp_should_dispatch_to_boundary contract + flag-OFF invariant\n"
            "═══════════════════════════════════════════════════════════\n");

    test_flag_off_no_rng_consumption();
    test_hu_flop_continues();
    test_non_flop_never_dispatches();
    test_prob_zero_never_terminates();
    test_prob_one_always_terminates();
    test_prob_half_balanced();
    test_seed_determinism();
    test_seed_sensitivity();

    fprintf(stderr,
            "\n[test_depth_limit_dispatch] passed=%d failed=%d\n",
            passed, failed);
    return failed > 0 ? 1 : 0;
}
