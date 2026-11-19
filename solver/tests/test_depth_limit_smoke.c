/**
 * test_depth_limit_smoke — byte-identity smoke for the depth-limit
 * code (mccfr_phase_r.c).
 *
 * Verifies that with depth_limit_cont_strats=0 (the default), the
 * trainer is deterministic by seed and produces byte-identical output
 * across two independent runs of the same fixture.
 *
 * **What this proves:** with the depth-limit code paths gated off, they
 * add no source of non-determinism (the TraversalState field, the
 * cont_ev_table fields on BPSolver, the bp_apply_cont_bias helper, the
 * action-sampling gate, the HEADS-branch dispatcher).
 *
 * **What this does NOT prove on its own:** byte-identity to a trainer
 * without the depth-limit code. That follows from this test PLUS:
 *   - T1 of test_depth_limit_dispatch.c (flag=0 dispatch consumes 0 RNG)
 *   - The depth-limit paths fire only when cont_bias_per_player[ap] != -1,
 *     which is set only by the HEADS-branch dispatcher, which fires
 *     only when bp_should_dispatch_to_boundary returns 1, which fires
 *     only with depth_limit_cont_strats=1.
 *   - The cont_ev_table is allocated only on bp_set_depth_limit_cont_strats(s, 1),
 *     so it stays NULL with the flag OFF.
 * Together: the flag-OFF path runs the same instructions on the same
 * RNG state with the same writes to the info table as legacy.
 *
 * Single-thread, no OpenMP needed. Build and run (10K iters):
 *   make test_depth_limit_smoke.
 */

#include "mccfr_blueprint.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static int passed = 0;
static int failed = 0;

#define CHECK(cond, msg) do {                                              \
    if (cond) { passed++; }                                                \
    else {                                                                 \
        failed++;                                                          \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);      \
    }                                                                      \
} while (0)

/* Set up a deterministic 2-player postflop fixture.
 *
 * 2 players (not 6) and a compact betting tree so that the test runs
 * in seconds without OpenMP. The byte-identity invariant holds for any
 * deterministic fixture, so a 6-player setup is not needed to verify
 * it: only a fixture that exercises the trainer's full code path
 * (info-set creation, regret updates, strategy_sum accumulation,
 * periodic discount). 2-player postflop with a single pot-sized bet
 * does that, and the tree is small enough to complete a meaningful
 * sample in <30 s. */
static int setup_solver(BPSolver *s, int64_t train_iters) {
    /* Card encoding: card = rank * 4 + suit. Ranks 0=2, 1=3, ..., 12=A.
     * Suits 0=c, 1=d, 2=h, 3=s. */

    /* Fixed flop. */
    int flop[3] = {
        0 * 4 + 0,   /* 2c */
        5 * 4 + 1,   /* 7d */
        11 * 4 + 2,  /* Kh */
    };

    /* 1 hand per player (identity bucketing). 2 players. */
    int hands[BP_MAX_PLAYERS][BP_MAX_HANDS][2];
    memset(hands, 0, sizeof(hands));
    /* P0: As Qs */
    hands[0][0][0] = 12 * 4 + 3; hands[0][0][1] = 10 * 4 + 3;
    /* P1: 3c 3d */
    hands[1][0][0] =  1 * 4 + 0; hands[1][0][1] =  1 * 4 + 1;

    float weights[BP_MAX_PLAYERS][BP_MAX_HANDS];
    memset(weights, 0, sizeof(weights));
    weights[0][0] = 1.0f;
    weights[1][0] = 1.0f;

    int num_hands[BP_MAX_PLAYERS] = {1, 1, 0, 0, 0, 0};

    float bet_sizes[1] = {1.0f};

    BPConfig config;
    bp_default_config(&config);
    config.include_preflop = 0;
    config.hash_table_size = (int64_t)(1 << 14);   /* 16K slots */
    config.num_threads = 1;                         /* SINGLE-THREAD for byte-identity */
    /* Stretch snapshot past the fixture so snapshot timing doesn't
     * pollute the comparison. Leave the discount schedule alone — it
     * exercises the bp_cont_ev_apply_discount call site (no-op
     * with cont_ev_table un-allocated, but still on the hot path). */
    config.prune_start_iter = train_iters * 2;
    config.snapshot_start_iter = train_iters * 2;

    return bp_init_ex(s, 2, flop,
                      (const int (*)[BP_MAX_HANDS][2])hands,
                      (const float (*)[BP_MAX_HANDS])weights,
                      num_hands,
                      100, 100,
                      bet_sizes, 1,
                      &config);
}

/* Slurp a file's contents to a fresh malloc'd buffer. Caller frees. */
static unsigned char *slurp_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc((size_t)len);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if ((long)got != len) { free(buf); return NULL; }
    *out_len = (size_t)len;
    return buf;
}

int main(int argc, char **argv) {
    /* Default 10K — chosen because the trainer here runs without
     * OpenMP (so it builds with compilers that lack libomp) AND
     * without pruning (the fixture stretches prune_start_iter past
     * the run). Pruning is what makes the trainer scale; without it
     * the per-iter cost is exponential in the betting tree depth.
     * 10K iters is enough to exercise full info-set creation, regret
     * updates, and strategy_sum accumulation — the byte-identity
     * invariant doesn't depend on iter count, only on determinism. */
    int64_t iters = (argc > 1) ? atoll(argv[1]) : (int64_t)10000;

    fprintf(stderr,
            "═══════════════════════════════════════════════════════════\n"
            "  Depth-limit code: byte-identity regression smoke\n"
            "  flag OFF, two independent runs must match byte-for-byte\n"
            "  iters per run: %lld (single-thread)\n"
            "═══════════════════════════════════════════════════════════\n",
            (long long)iters);
    /* Force unbuffered stderr so progress is visible even if the test
     * is killed mid-run. */
    setvbuf(stderr, NULL, _IONBF, 0);

    /* The two regret files go in the directory given as the second
     * argument (the Makefile passes build/), else the working directory,
     * and are removed after the comparison. */
    const char *out_dir = (argc > 2) ? argv[2] : ".";
    char path_a[4096], path_b[4096];
    snprintf(path_a, sizeof(path_a), "%s/depth_limit_smoke_a.bps", out_dir);
    snprintf(path_b, sizeof(path_b), "%s/depth_limit_smoke_b.bps", out_dir);

    /* Run A. */
    {
        BPSolver s;
        memset(&s, 0, sizeof(s));
        int rc = setup_solver(&s, iters);
        CHECK(rc == 0, "run A: bp_init_ex");
        if (rc != 0) return 1;

        /* depth_limit_cont_strats stays at its default 0: this run tests
         * the flag-OFF byte-identity invariant. */
        CHECK(s.depth_limit_cont_strats == 0,
              "run A: depth_limit_cont_strats default is 0");
        CHECK(s.cont_ev_table.entries == NULL,
              "run A: cont_ev_table NOT allocated when flag off");

        clock_t t0 = clock();
        rc = bp_solve(&s, iters);
        clock_t t1 = clock();
        fprintf(stderr, "  run A: bp_solve %lld iters in %.2fs (%.0f iter/s)\n",
                (long long)iters,
                (double)(t1 - t0) / CLOCKS_PER_SEC,
                (double)iters / ((double)(t1 - t0) / CLOCKS_PER_SEC));
        CHECK(rc == 0, "run A: bp_solve");

        rc = bp_save_regrets(&s, path_a);
        CHECK(rc == 0, "run A: bp_save_regrets");
        bp_free(&s);
    }

    /* Run B (independent setup, same seed flow). */
    {
        BPSolver s;
        memset(&s, 0, sizeof(s));
        int rc = setup_solver(&s, iters);
        CHECK(rc == 0, "run B: bp_init_ex");
        if (rc != 0) return 1;

        CHECK(s.depth_limit_cont_strats == 0,
              "run B: depth_limit_cont_strats default is 0");
        CHECK(s.cont_ev_table.entries == NULL,
              "run B: cont_ev_table NOT allocated when flag off");

        clock_t t0 = clock();
        rc = bp_solve(&s, iters);
        clock_t t1 = clock();
        fprintf(stderr, "  run B: bp_solve %lld iters in %.2fs (%.0f iter/s)\n",
                (long long)iters,
                (double)(t1 - t0) / CLOCKS_PER_SEC,
                (double)iters / ((double)(t1 - t0) / CLOCKS_PER_SEC));
        CHECK(rc == 0, "run B: bp_solve");

        rc = bp_save_regrets(&s, path_b);
        CHECK(rc == 0, "run B: bp_save_regrets");
        bp_free(&s);
    }

    /* Compare. */
    size_t len_a = 0, len_b = 0;
    unsigned char *buf_a = slurp_file(path_a, &len_a);
    unsigned char *buf_b = slurp_file(path_b, &len_b);
    CHECK(buf_a != NULL, "slurp run A");
    CHECK(buf_b != NULL, "slurp run B");
    if (buf_a && buf_b) {
        CHECK(len_a == len_b, "file sizes match");
        if (len_a == len_b) {
            int diff_byte = -1;
            for (size_t i = 0; i < len_a; i++) {
                if (buf_a[i] != buf_b[i]) { diff_byte = (int)i; break; }
            }
            char msg[160];
            snprintf(msg, sizeof(msg),
                     "byte-identity: %zu bytes match (first diff at offset %d)",
                     len_a, diff_byte);
            CHECK(diff_byte == -1, msg);
            if (diff_byte != -1) {
                fprintf(stderr,
                        "       run A[%d] = 0x%02x, run B[%d] = 0x%02x\n",
                        diff_byte, buf_a[diff_byte],
                        diff_byte, buf_b[diff_byte]);
            }
        }
        fprintf(stderr, "  total bytes per run: %zu\n", len_a);
    }
    free(buf_a);
    free(buf_b);
    remove(path_a);
    remove(path_b);

    fprintf(stderr, "\n[test_depth_limit_smoke] passed=%d failed=%d\n",
            passed, failed);
    return failed > 0 ? 1 : 0;
}
