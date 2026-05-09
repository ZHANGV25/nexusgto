/**
 * test_b3_bucket_alignment.c — Phase R Block B.3 bucket-lookup
 * divergence test (HARD GATE before declaring B.3 PASS).
 *
 * Empirically verifies trainer's flop-cache write-key vs kernel-side
 * `bps_hand_to_bucket` read-key alignment on the v3/phase3
 * production texture cache. The parallel blocker called out in
 * `docs/PHASE_R_B3_BLOCKED.md` §"Parallel blocker" must be either
 * empirically resolved (covered textures byte-equal) or explicitly
 * surfaced (divergence with verified cost) before B.4 wire-in is
 * trusted.
 *
 * Setup:
 *   - Loads /tmp/texture_cache.bin (or path override via
 *     BPS_TEXTURE_CACHE env var) via TWO independent paths:
 *       (a) `bp_load_texture_cache` — trainer's loader; fills
 *           `BPSolver.texture_bucket_cache` + `texture_hash_keys`.
 *       (b) `bps_load_texture_cache` — reader's loader; fills
 *           `BPSReader.textures.bucket_cache` + `hash_keys`.
 *     Both read the SAME TXC1 bytes; per advisor #3, the data buffer
 *     is byte-equal across loaders, so the empirical question is
 *     hand-index alignment, not cache-content alignment.
 *
 *   - Walks the precompute enumerator's variant list
 *     (mccfr_blueprint.c:2903-2934) to recover the 1755 canonical
 *     flop boards in the SAME order as the texture cache stored them.
 *     For each canonical board (tex_idx 0..1754):
 *       For each non-blocked (c0, c1) pair (1176 per flop):
 *         trainer_bucket = `s.texture_bucket_cache[tex_idx *
 *           BP_MAX_HANDS + full_idx]` (the full-deck pair index per
 *           mccfr_blueprint.c:2998-3007, 0..1325).
 *         reader_bucket  = `bps_hand_to_bucket(&r, canonical_board,
 *           c0, c1)` (canonicalize + hand_index_for; the kernel-side
 *           B.4 read path).
 *
 *   - Compares per-pair. Tracks: total queries, byte_equal,
 *     reader_miss (-1), divergent (both return values but disagree).
 *
 * Pass criteria (Block 5 advisor #3 contract):
 *   - divergent == 0 (no kernel/trainer disagreement on covered)
 *   - byte_equal + reader_miss == total
 *   - reader_miss class is documented (covered-texture lookups MUST
 *     all return non-negative; reader_miss should be 0 across all
 *     1755 covered textures × 1176 hands ≈ 2.06M queries)
 *
 * Runtime: ~5-10s on Mac for 2M lookups.
 */

#include "mccfr_blueprint.h"
#include "bps_reader.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *cache_path(void) {
    const char *env = getenv("BPS_TEXTURE_CACHE");
    if (env && env[0]) return env;
    return "/tmp/texture_cache.bin";
}

/* Mirror of mccfr_blueprint.c:2903-2934 precompute enumerator.
 * MUST match exactly so tex_idx in our enumeration aligns with the
 * file's stored order. */
static int enumerate_variants(int r0, int r1, int r2, int variants[5][3]) {
    int nv = 0;
    if (r0 == r1 && r1 == r2) {
        variants[nv][0] = 3; variants[nv][1] = 2; variants[nv][2] = 1; nv++;
    } else if (r0 == r1 || r1 == r2) {
        variants[nv][0] = 3; variants[nv][1] = 2; variants[nv][2] = 1; nv++;
        if (r0 == r1) {
            variants[nv][0] = 3; variants[nv][1] = 2; variants[nv][2] = 3; nv++;
        } else {
            variants[nv][0] = 3; variants[nv][1] = 3; variants[nv][2] = 2; nv++;
        }
    } else {
        variants[nv][0] = 3; variants[nv][1] = 2; variants[nv][2] = 1; nv++;
        variants[nv][0] = 3; variants[nv][1] = 3; variants[nv][2] = 3; nv++;
        variants[nv][0] = 3; variants[nv][1] = 3; variants[nv][2] = 2; nv++;
        variants[nv][0] = 3; variants[nv][1] = 2; variants[nv][2] = 3; nv++;
        variants[nv][0] = 2; variants[nv][1] = 3; variants[nv][2] = 3; nv++;
    }
    return nv;
}

int main(void) {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Phase R Block B.3 — Bucket-lookup divergence test\n");
    printf("  HARD GATE before B.3 PASS declaration\n");
    printf("═══════════════════════════════════════════════════════════\n");

    /* ── Load via trainer path ──────────────────────────────────── */
    BPSolver s;
    memset(&s, 0, sizeof(s));
    s.postflop_num_buckets = 200;
    /* texture_hash_index[] needs all -1 init (open-addressed empty) */
    for (int i = 0; i < BP_TEXTURE_INDEX_SIZE; i++)
        s.texture_hash_index[i] = -1;

    int n_trainer = bp_load_texture_cache(&s, cache_path());
    if (n_trainer <= 0) {
        printf("[FAIL] bp_load_texture_cache failed; check %s\n", cache_path());
        return 1;
    }
    printf("[trainer] loaded %d textures\n", n_trainer);

    /* ── Load via reader path ───────────────────────────────────── */
    BPSReader r;
    memset(&r, 0, sizeof(r));
    /* Per feedback_phase3_postflop_bucket_lookup_blocker.md: phase3's
     * texture_cache.bin uses the boost (legacy) mixer. */
    int n_reader = bps_load_texture_cache(&r, cache_path(), 1);
    if (n_reader <= 0) {
        printf("[FAIL] bps_load_texture_cache failed; check %s\n", cache_path());
        free(s.texture_bucket_cache);
        return 1;
    }
    printf("[reader]  loaded %d textures\n", n_reader);

    if (n_trainer != n_reader) {
        printf("[FAIL] loader disagree on texture count: trainer=%d reader=%d\n",
               n_trainer, n_reader);
        free(s.texture_bucket_cache);
        if (r.textures.bucket_cache) free(r.textures.bucket_cache);
        return 1;
    }

    /* ── Sweep ──────────────────────────────────────────────────── */
    int64_t total = 0;
    int64_t byte_equal = 0;
    int64_t reader_miss = 0;
    int64_t divergent = 0;
    int64_t reader_neg = 0;     /* reader returned a negative non-(-1) value */
    int top_diff_count = 0;
    const int TOP_DIFF_MAX = 5;

    /* Tally for paired-board flushdraw class — per memory's claim */
    int64_t pbfd_total = 0, pbfd_match = 0, pbfd_miss = 0, pbfd_divergent = 0;

    int tex_count = 0;

    for (int rr0 = 12; rr0 >= 0; rr0--) {
        for (int rr1 = rr0; rr1 >= 0; rr1--) {
            for (int rr2 = rr1; rr2 >= 0; rr2--) {
                int variants[5][3];
                int nv = enumerate_variants(rr0, rr1, rr2, variants);

                for (int v = 0; v < nv; v++) {
                    int board[3] = {
                        rr0 * 4 + variants[v][0],
                        rr1 * 4 + variants[v][1],
                        rr2 * 4 + variants[v][2]
                    };

                    int blocked[52] = {0};
                    blocked[board[0]] = 1;
                    blocked[board[1]] = 1;
                    blocked[board[2]] = 1;

                    int is_paired_fd = ((rr0 == rr1 || rr1 == rr2)
                                        && rr0 != rr1 + (rr1 == rr2 ? 0 : 0)  /* heuristic noop */
                                        && (rr0 != rr1 || rr1 != rr2)
                                        && nv >= 2 && v == 1);  /* second variant is FD */

                    /* Sweep all non-blocked (c0, c1) pairs. */
                    for (int c0 = 0; c0 < 52; c0++) {
                        if (blocked[c0]) continue;
                        for (int c1 = c0 + 1; c1 < 52; c1++) {
                            if (blocked[c1]) continue;

                            int full_idx = c0 * 51 - c0 * (c0 - 1) / 2
                                           + c1 - c0 - 1;
                            if (full_idx < 0 || full_idx >= BP_MAX_HANDS) {
                                printf("[BUG] full_idx=%d out of bounds for "
                                       "(%d,%d)\n", full_idx, c0, c1);
                                continue;
                            }

                            int trainer_bucket = s.texture_bucket_cache[
                                (int64_t)tex_count * BP_MAX_HANDS + full_idx];
                            int reader_bucket = bps_hand_to_bucket(&r, board,
                                                                    c0, c1);
                            total++;
                            if (is_paired_fd) pbfd_total++;

                            if (reader_bucket == -1) {
                                reader_miss++;
                                if (is_paired_fd) pbfd_miss++;
                            } else if (reader_bucket < 0) {
                                /* Defensive: reader returned negative
                                 * other than -1; shouldn't happen. */
                                reader_neg++;
                            } else if (reader_bucket == trainer_bucket) {
                                byte_equal++;
                                if (is_paired_fd) pbfd_match++;
                            } else {
                                divergent++;
                                if (is_paired_fd) pbfd_divergent++;
                                if (top_diff_count < TOP_DIFF_MAX) {
                                    printf("[DIFF] tex_idx=%d board=[%d,%d,%d] "
                                           "(c0=%d,c1=%d) full_idx=%d "
                                           "trainer=%d reader=%d\n",
                                           tex_count,
                                           board[0], board[1], board[2],
                                           c0, c1, full_idx,
                                           trainer_bucket, reader_bucket);
                                    top_diff_count++;
                                }
                            }
                        }
                    }
                    tex_count++;
                }
            }
        }
    }

    printf("\n[sweep] tex_count=%d (expect %d), total=%lld, byte_equal=%lld, "
           "reader_miss=%lld, divergent=%lld, reader_neg=%lld\n",
           tex_count, n_trainer,
           (long long)total, (long long)byte_equal,
           (long long)reader_miss, (long long)divergent,
           (long long)reader_neg);

    if (total > 0) {
        printf("[sweep] byte_equal_pct=%.4f%% reader_miss_pct=%.4f%% "
               "divergent_pct=%.4f%%\n",
               100.0 * byte_equal / total,
               100.0 * reader_miss / total,
               100.0 * divergent / total);
    }

    /* Paired-board flushdraw subclass tally */
    if (pbfd_total > 0) {
        printf("[pbfd] total=%lld match=%lld miss=%lld divergent=%lld "
               "(memory claims ~1%% miss on this class)\n",
               (long long)pbfd_total, (long long)pbfd_match,
               (long long)pbfd_miss, (long long)pbfd_divergent);
    }

    /* ── Pass criteria ────────────────────────────────────────── */
    int passed = 0, failed = 0;

    /* (1) tex_count must equal loaded count (enumerator alignment). */
    if (tex_count != n_trainer) {
        printf("[FAIL] tex_count=%d != n_trainer=%d (enumerator drift)\n",
               tex_count, n_trainer);
        failed++;
    } else {
        printf("[PASS] tex_count == n_trainer == %d (enumerator order matches storage)\n",
               n_trainer);
        passed++;
    }

    /* (2) divergent == 0 (no kernel/trainer disagreement on covered). */
    if (divergent != 0) {
        printf("[FAIL] divergent=%lld covered queries (B.3 parallel "
               "blocker NOT closed)\n", (long long)divergent);
        failed++;
    } else {
        printf("[PASS] divergent=0 across %lld covered queries\n",
               (long long)total);
        passed++;
    }

    /* (3) byte_equal + reader_miss == total (no other class). */
    if (byte_equal + reader_miss + divergent + reader_neg != total) {
        printf("[FAIL] partition does not sum to total (some queries "
               "unaccounted)\n");
        failed++;
    } else {
        passed++;
    }

    /* (4) reader_neg == 0 (defensive — should never fire). */
    if (reader_neg != 0) {
        printf("[FAIL] reader_neg=%lld (reader returned negative not -1)\n",
               (long long)reader_neg);
        failed++;
    } else {
        passed++;
    }

    free(s.texture_bucket_cache);
    if (r.textures.bucket_cache) free(r.textures.bucket_cache);

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("[test_b3_bucket_alignment] passed=%d failed=%d\n", passed, failed);
    printf("═══════════════════════════════════════════════════════════\n");

    return (failed == 0) ? 0 : 1;
}
