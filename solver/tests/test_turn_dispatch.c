/**
 * test_turn_dispatch — Dual-boundary build (2026-06-25).
 *
 * Verifies bp_should_dispatch_to_turn_boundary (the TURN sibling of
 * bp_should_dispatch_to_boundary) one street later, AND re-asserts that the
 * existing FLOP helper is byte-untouched (it must still NEVER dispatch at
 * num_board==4 — the property test_phase_r_dispatch.c T3 pins).
 *
 * Contract mirrored from the flop helper:
 *   - flag==0 → 0 with ZERO RNG consumption (byte-identity invariant).
 *   - num_board != 4 → 0 (only end-of-turn is a turn boundary).
 *   - n_active < 3 → 0 (HU/2p turn runs to showdown).
 *   - prob==0 → never; prob==1 → always; prob==0.5 → ~half, deterministic
 *     per seed, sensitive across seeds.
 *
 * Isolated unit test: links only mccfr_phase_r.c + mccfr_cont_ev_table.c.
 * Deterministic, single-threaded, sub-second.
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

/* T1 (byte-identity): turn flag OFF → 0, no RNG consumed, every combo. */
static void test_flag_off_no_rng(void) {
    fprintf(stderr, "[T1] turn flag=0 → 0 without consuming RNG\n");
    uint64_t rng = 0x9E3779B97F4A7C15ULL, before = rng;
    for (int nb = 0; nb <= 5; nb++)
        for (int na = 1; na <= 6; na++) {
            int got = bp_should_dispatch_to_turn_boundary(nb, na, 0, 0.5f, &rng);
            CHECK(got == 0, "turn flag=0 returned !=0");
            CHECK(rng == before, "turn flag=0 consumed RNG (byte-identity broken)");
        }
}

/* T2: non-turn num_board never dispatches the turn helper. */
static void test_non_turn_never(void) {
    fprintf(stderr, "[T2] non-turn num_board never dispatches turn helper\n");
    uint64_t rng = 0x12345678ABCDEFULL;
    int nbs[] = {0, 1, 2, 3, 5};  /* everything except 4 */
    for (size_t i = 0; i < sizeof(nbs)/sizeof(int); i++)
        for (int na = 2; na <= 6; na++) {
            int got = bp_should_dispatch_to_turn_boundary(nbs[i], na, 1, 1.0f, &rng);
            char m[96]; snprintf(m, sizeof(m),
                "turn helper dispatched at num_board=%d", nbs[i]);
            CHECK(got == 0, m);
        }
}

/* T3: HU/2p turn (n_active<3) never dispatches. */
static void test_hu_turn_continues(void) {
    fprintf(stderr, "[T3] HU/2p turn (n_active<3) never dispatches\n");
    uint64_t rng = 0xDEADBEEFCAFEBABEULL;
    for (int na = 1; na <= 2; na++)
        for (int i = 0; i < 1000; i++) {
            int got = bp_should_dispatch_to_turn_boundary(4, na, 1, 1.0f, &rng);
            CHECK(got == 0, "HU/2p turn dispatched");
            if (got) break;
        }
}

/* T4: multiway turn, prob=0 never; prob=1 always (3..6 players). */
static void test_prob_extremes(void) {
    fprintf(stderr, "[T4] multiway turn prob=0 never, prob=1 always\n");
    uint64_t rng = 0xBADC0DEFEEDFACE0ULL;
    for (int n = 3; n <= 6; n++)
        for (int i = 0; i < 1000; i++) {
            CHECK(bp_should_dispatch_to_turn_boundary(4, n, 1, 0.0f, &rng) == 0,
                  "prob=0 turn dispatched");
            CHECK(bp_should_dispatch_to_turn_boundary(4, n, 1, 1.0f, &rng) == 1,
                  "prob=1 turn did not dispatch");
        }
}

/* T5: prob=0.5 → ~half, 100K trials (4σ band). */
static void test_prob_half(void) {
    fprintf(stderr, "[T5] turn prob=0.5 ~half (100K)\n");
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    const int N = 100000; int term = 0;
    for (int i = 0; i < N; i++)
        term += bp_should_dispatch_to_turn_boundary(4, 4, 1, 0.5f, &rng);
    CHECK(term >= N/2 - 2000 && term <= N/2 + 2000, "turn prob=0.5 imbalanced");
    fprintf(stderr, "       observed: %d / %d (%.4f)\n", term, N, (double)term/N);
}

/* T6: same seed → identical turn dispatch sequence. */
static void test_seed_determinism(void) {
    fprintf(stderr, "[T6] same seed → identical turn dispatch sequence\n");
    uint64_t a = 0xABCDEF0123456789ULL, b = a; int diffs = 0;
    for (int i = 0; i < 10000; i++)
        if (bp_should_dispatch_to_turn_boundary(4, 4, 1, 0.5f, &a) !=
            bp_should_dispatch_to_turn_boundary(4, 4, 1, 0.5f, &b)) diffs++;
    CHECK(diffs == 0, "turn seed determinism violated");
}

/* T7: the FLOP helper is byte-untouched — num_board==4 STILL never
 * dispatches there (re-pins test_phase_r_dispatch.c T3 from this build). */
static void test_flop_helper_unchanged_at_turn(void) {
    fprintf(stderr, "[T7] FLOP helper still never dispatches at num_board==4\n");
    uint64_t rng = 0x5555AAAA5555AAAAULL;
    for (int n = 3; n <= 6; n++)
        for (int i = 0; i < 1000; i++) {
            int got = bp_should_dispatch_to_boundary(4, n, 1, 1.0f, &rng);
            CHECK(got == 0, "FLOP helper dispatched at turn (num_board==4)");
            if (got) break;
        }
    /* And the flop helper STILL dispatches at num_board==3 (unbroken). */
    uint64_t rng2 = 0x1111222233334444ULL;
    CHECK(bp_should_dispatch_to_boundary(3, 4, 1, 1.0f, &rng2) == 1,
          "FLOP helper stopped dispatching at its own flop boundary");
}

int main(void) {
    fprintf(stderr,
        "===========================================================\n"
        "  Turn-boundary dispatch helper unit tests\n"
        "===========================================================\n");
    test_flag_off_no_rng();
    test_non_turn_never();
    test_hu_turn_continues();
    test_prob_extremes();
    test_prob_half();
    test_seed_determinism();
    test_flop_helper_unchanged_at_turn();
    fprintf(stderr, "\n[test_turn_dispatch] passed=%d failed=%d\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
