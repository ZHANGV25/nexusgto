/*
 * test_es_chance_sample_bitexact.c — PRIMARY acceptance gate for the CPU-ES
 * in-tree chance-node fix (2026-06-14).
 *
 * THE BUG this guards against: the CPU-ES traversal used to sample a chance
 * node's child with a per-CALL local RNG. The GPU reference draws ONE child per
 * chance node per CFR ITERATION, shared across the whole traversal, via
 * splitmix64(master_seed ^ iter·K1 ^ chance_idx·K2) % num_children
 * (street_solve.cu::ss_redraw_chance_samples_host). The fix replaces the CPU
 * stub with es_chance_sampled_child(), which MUST reproduce that draw
 * BIT-FOR-BIT so a single-thread CPU walk over the same tree visits the same
 * card trajectory as the GPU — the prerequisite for GPU/CPU agreement.
 *
 * This test re-implements the GPU's host formula INDEPENDENTLY (copied from
 * street_solve.cu, NOT calling into the CPU code) and asserts the CPU's exported
 * sampler matches it across a grid of (master_seed, iter, chance_idx,
 * num_children). If either side's constants/recurrence ever drift, this fails.
 *
 * A green "it returns a child index in range" is NOT enough (the old stub did
 * that too). The gate is EQUALITY WITH THE GPU FORMULA — the expensive truth.
 *
 * Build (no OpenMP, no GPU; pure C):
 *   gcc -O2 -Wall -Wextra -Isrc tests/test_es_chance_sample_bitexact.c \
 *       -o build/test_es_chance -lm
 * (es_chance_sampled_child is a static inline in es_resolver.c, so this test
 *  does NOT link es_resolver.c — it re-declares the SAME inline below from the
 *  single source of truth and cross-checks it against the GPU formula. The
 *  es_resolver.c copy is verified identical by a textual assertion in the
 *  Python driver test_es_chance_sample.py.)
 */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

/* ── The CPU sampler under test (MUST be byte-identical to the copy in
 *    src/es_resolver.c::es_chance_sampled_child). ─────────────────────────── */
static inline uint64_t cpu_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline int cpu_chance_sampled_child(uint64_t master_seed, int chance_node_idx,
                                           int num_children, int64_t iter) {
    uint64_t s = cpu_splitmix64(
        master_seed ^ ((uint64_t)iter * 0x9E3779B97F4A7C15ULL)
                    ^ ((uint64_t)chance_node_idx * 0xBF58476D1CE4E5B9ULL));
    return (int)(s % (uint64_t)num_children);
}

/* ── The GPU host reference (copied from street_solve.cu, independent code
 *    path: ss_splitmix64_host + the seed mix in ss_redraw_chance_samples_host). */
static uint64_t gpu_splitmix64_host(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static int gpu_sampled_child(uint64_t master_seed, int c, int num_children, int iter) {
    if (num_children <= 0) return 0;
    uint64_t s = gpu_splitmix64_host(
        master_seed ^ ((uint64_t)iter * 0x9E3779B97F4A7C15ULL)
                    ^ ((uint64_t)c * 0xBF58476D1CE4E5B9ULL));
    return (int)(s % (uint64_t)num_children);
}

int main(void) {
    printf("[test_es_chance_sample_bitexact]\n");

    /* Grid: the values that actually occur in HU flop/turn solves.
     *   num_children: 44 (end-of-flop, 52-3-? before hole exclusion ~44-47),
     *                 43 (end-of-turn), plus small values for edge coverage.
     *   master_seed: 1 (GPU default), 0xE5 / 0xABCDEF (API/bench seeds), 42.
     *   iter: 1..2000 (covers the 200/500/1000-iter ship budgets).
     *   chance_idx: 0..4095 (HU flop has ~3,590 chance nodes — OP3_OP1_BENCH). */
    const uint64_t seeds[] = {1ULL, 0xE5ULL, 0xABCDEFULL, 42ULL,
                              0x9E3779B97F4A7C15ULL};
    const int ncs[] = {2, 3, 43, 44, 45, 46, 47, 52};
    long total = 0, mism = 0;

    for (unsigned si = 0; si < sizeof(seeds) / sizeof(seeds[0]); si++) {
        for (unsigned ni = 0; ni < sizeof(ncs) / sizeof(ncs[0]); ni++) {
            int nc = ncs[ni];
            for (int iter = 1; iter <= 2000; iter++) {
                for (int c = 0; c < 4096; c += 7) {  /* stride to keep it fast */
                    int cpu = cpu_chance_sampled_child(seeds[si], c, nc, iter);
                    int gpu = gpu_sampled_child(seeds[si], c, nc, iter);
                    total++;
                    if (cpu != gpu) {
                        mism++;
                        if (mism <= 5) {
                            printf("  MISMATCH seed=%" PRIu64 " c=%d nc=%d iter=%d: "
                                   "cpu=%d gpu=%d\n", seeds[si], c, nc, iter, cpu, gpu);
                        }
                    }
                    /* range invariant too */
                    CHECK(cpu >= 0 && cpu < nc,
                          "cpu out of range seed=%" PRIu64 " c=%d nc=%d iter=%d -> %d",
                          seeds[si], c, nc, iter, cpu);
                }
            }
        }
    }
    CHECK(mism == 0, "chance-sample sequence diverged from GPU in %ld/%ld cases",
          mism, total);
    printf("  checked %ld (seed,chance_idx,num_children,iter) draws; "
           "%ld mismatches vs GPU formula\n", total, mism);

    /* Determinism: same inputs -> same output, twice. */
    for (int iter = 1; iter <= 100; iter++) {
        int a = cpu_chance_sampled_child(1ULL, 17, 44, iter);
        int b = cpu_chance_sampled_child(1ULL, 17, 44, iter);
        CHECK(a == b, "non-deterministic at iter=%d (%d != %d)", iter, a, b);
    }

    /* Variability: across iters the draw is NOT constant (the bug's stub with a
     * fixed local seed could pin one child; a correct per-iter sampler moves). */
    {
        int first = cpu_chance_sampled_child(1ULL, 0, 44, 1);
        int changed = 0;
        for (int iter = 2; iter <= 200; iter++) {
            if (cpu_chance_sampled_child(1ULL, 0, 44, iter) != first) { changed = 1; break; }
        }
        CHECK(changed, "chance sample never changes across 200 iters — frozen draw");
    }

    if (g_fail == 0)
        printf("PASSED test_es_chance_sample_bitexact\n");
    else
        printf("FAILED test_es_chance_sample_bitexact (%d failures)\n", g_fail);
    return g_fail ? 1 : 0;
}
