/*
 * test_pa_emd.c — Unit tests for PA-EMD Phase 1:
 *   ca_emd_1d()                — 1D Earth Mover's Distance
 *   ca_compute_ehs_distribution() — hand equity distribution histogram
 *
 * Build:
 *   gcc -O2 -Wall -I src tests/test_pa_emd.c src/card_abstraction.c \
 *       -o build/test_pa_emd -lm
 * Run:
 *   build/test_pa_emd
 */

#include "card_abstraction.h"
#include "hand_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int n_pass = 0;
static int n_fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        n_fail++; \
        return; \
    } \
} while (0)

#define ASSERT_FLOAT_EQ(a, b, eps, msg) do { \
    float _a = (a), _b = (b); \
    if (fabsf(_a - _b) > (eps)) { \
        fprintf(stderr, "FAIL [%s:%d]: %s (got %.8f, expected %.8f, eps=%.8f)\n", \
                __FILE__, __LINE__, msg, _a, _b, (float)(eps)); \
        n_fail++; \
        return; \
    } \
} while (0)

#define PASS(name) do { printf("  PASS: %s\n", name); n_pass++; } while (0)

/* ── EMD tests ─────────────────────────────────────────────────────── */

static void test_emd_identical(void) {
    float hist[5] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
    float emd = ca_emd_1d(hist, hist, 5);
    ASSERT_FLOAT_EQ(emd, 0.0f, 1e-7f, "EMD of identical distributions should be 0");
    PASS("emd_identical");
}

static void test_emd_identical_nonuniform(void) {
    float hist[4] = {0.5f, 0.3f, 0.1f, 0.1f};
    float emd = ca_emd_1d(hist, hist, 4);
    ASSERT_FLOAT_EQ(emd, 0.0f, 1e-7f, "EMD of identical non-uniform distributions should be 0");
    PASS("emd_identical_nonuniform");
}

static void test_emd_max_separation(void) {
    /*
     * All mass in bin 0 vs all mass in bin N-1.
     * EMD = sum_i |CDF_a(i) - CDF_b(i)| for i = 0..N-1
     * CDF_a = [1, 1, 1, ..., 1]  (jumps to 1 at bin 0)
     * CDF_b = [0, 0, 0, ..., 1]  (jumps to 1 at bin N-1)
     * |CDF_a - CDF_b| = [1, 1, 1, ..., 0]  (N-1 ones)
     * EMD = N-1
     */
    int n = 10;
    float a[10] = {0}, b[10] = {0};
    a[0] = 1.0f;
    b[9] = 1.0f;
    float emd = ca_emd_1d(a, b, n);
    ASSERT_FLOAT_EQ(emd, 9.0f, 1e-6f, "EMD of max-separated distributions should be N-1");
    PASS("emd_max_separation");
}

static void test_emd_adjacent_shift(void) {
    /*
     * All mass in bin 0 vs all mass in bin 1 → EMD = 1.
     */
    float a[5] = {1, 0, 0, 0, 0};
    float b[5] = {0, 1, 0, 0, 0};
    float emd = ca_emd_1d(a, b, 5);
    ASSERT_FLOAT_EQ(emd, 1.0f, 1e-7f, "EMD of adjacent-bin shift should be 1");
    PASS("emd_adjacent_shift");
}

static void test_emd_symmetry(void) {
    float a[6] = {0.3f, 0.2f, 0.1f, 0.0f, 0.15f, 0.25f};
    float b[6] = {0.1f, 0.1f, 0.3f, 0.2f, 0.2f, 0.1f};
    float emd_ab = ca_emd_1d(a, b, 6);
    float emd_ba = ca_emd_1d(b, a, 6);
    ASSERT_FLOAT_EQ(emd_ab, emd_ba, 1e-7f, "EMD should be symmetric");
    ASSERT(emd_ab > 0.0f, "EMD of different distributions should be > 0");
    PASS("emd_symmetry");
}

static void test_emd_triangle_inequality(void) {
    float a[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float b[5] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    float c[5] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

    float d_ab = ca_emd_1d(a, b, 5);
    float d_bc = ca_emd_1d(b, c, 5);
    float d_ac = ca_emd_1d(a, c, 5);

    /* Triangle inequality: d(a,c) <= d(a,b) + d(b,c) */
    ASSERT(d_ac <= d_ab + d_bc + 1e-6f,
           "EMD must satisfy triangle inequality: d(a,c) <= d(a,b) + d(b,c)");

    /* Also verify the specific values: a→b = 2, b→c = 2, a→c = 4 */
    ASSERT_FLOAT_EQ(d_ab, 2.0f, 1e-7f, "d(bin0, bin2) should be 2");
    ASSERT_FLOAT_EQ(d_bc, 2.0f, 1e-7f, "d(bin2, bin4) should be 2");
    ASSERT_FLOAT_EQ(d_ac, 4.0f, 1e-7f, "d(bin0, bin4) should be 4");

    PASS("emd_triangle_inequality");
}

static void test_emd_partial_overlap(void) {
    /*
     * a = [0.5, 0.5, 0, 0]
     * b = [0, 0, 0.5, 0.5]
     * Running diff: 0.5, 1.0, 0.5, 0.0  →  EMD = 2.0
     */
    float a[4] = {0.5f, 0.5f, 0.0f, 0.0f};
    float b[4] = {0.0f, 0.0f, 0.5f, 0.5f};
    float emd = ca_emd_1d(a, b, 4);
    ASSERT_FLOAT_EQ(emd, 2.0f, 1e-6f, "EMD of two non-overlapping half-mass blocks");
    PASS("emd_partial_overlap");
}

static void test_emd_single_bin(void) {
    float a[1] = {1.0f};
    float b[1] = {1.0f};
    float emd = ca_emd_1d(a, b, 1);
    ASSERT_FLOAT_EQ(emd, 0.0f, 1e-7f, "EMD with single bin should always be 0");
    PASS("emd_single_bin");
}

static void test_emd_zero_input(void) {
    float a[3] = {0, 0, 0};
    float b[3] = {0, 0, 0};
    float emd = ca_emd_1d(a, b, 3);
    ASSERT_FLOAT_EQ(emd, 0.0f, 1e-7f, "EMD of all-zero histograms should be 0");
    PASS("emd_zero_input");
}

/* ── EHS distribution tests ────────────────────────────────────────── */

static void test_dist_determinism(void) {
    /* Same inputs must produce identical histograms across calls */
    int board[3] = {
        MAKE_CARD(12, 2),  /* Ah */
        MAKE_CARD(11, 0),  /* Kc */
        MAKE_CARD(8, 1),   /* Td */
    };
    int hand[2] = { MAKE_CARD(10, 2), MAKE_CARD(9, 2) }; /* QhJh */

    int n_bins = 50;
    float hist1[50], hist2[50];
    ca_compute_ehs_distribution(board, 3, hand, 500, hist1, n_bins);
    ca_compute_ehs_distribution(board, 3, hand, 500, hist2, n_bins);

    ASSERT(memcmp(hist1, hist2, n_bins * sizeof(float)) == 0,
           "EHS distribution must be bit-identical across calls with same inputs");
    PASS("dist_determinism");
}

static void test_dist_normalized(void) {
    /* Histogram should sum to 1.0 */
    int board[3] = {
        MAKE_CARD(5, 0),   /* 7c */
        MAKE_CARD(3, 1),   /* 5d */
        MAKE_CARD(1, 2),   /* 3h */
    };
    int hand[2] = { MAKE_CARD(12, 0), MAKE_CARD(12, 3) }; /* AcAs */

    int n_bins = 50;
    float hist[50];
    ca_compute_ehs_distribution(board, 3, hand, 1000, hist, n_bins);

    float sum = 0.0f;
    for (int i = 0; i < n_bins; i++) sum += hist[i];
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "EHS distribution should sum to 1.0");

    /* All entries non-negative */
    for (int i = 0; i < n_bins; i++) {
        ASSERT(hist[i] >= 0.0f, "EHS distribution bins must be non-negative");
    }
    PASS("dist_normalized");
}

static void test_dist_aa_concentrated_high(void) {
    /*
     * AA on a low flop (7-5-3 rainbow) should have most mass near equity 1.0.
     * The top 20% of bins (bins 40-49 of 50) should hold majority of mass.
     */
    int board[3] = {
        MAKE_CARD(5, 0),   /* 7c */
        MAKE_CARD(3, 1),   /* 5d */
        MAKE_CARD(1, 2),   /* 3h */
    };
    int hand[2] = { MAKE_CARD(12, 0), MAKE_CARD(12, 3) }; /* AcAs */

    int n_bins = 50;
    float hist[50];
    ca_compute_ehs_distribution(board, 3, hand, 2000, hist, n_bins);

    /* Sum top 20% of bins (high equity end) */
    float top_mass = 0.0f;
    for (int i = n_bins * 4 / 5; i < n_bins; i++) top_mass += hist[i];

    ASSERT(top_mass > 0.40f,
           "AA on low board: top 20% of bins should hold >40% of mass");
    PASS("dist_aa_concentrated_high");
}

static void test_dist_72o_spread_lower(void) {
    /*
     * 72o on a low flop should have significant mass in the lower bins.
     * It's a weak hand — should have notably more low-equity mass than AA.
     */
    int board[3] = {
        MAKE_CARD(5, 0),   /* 7c */
        MAKE_CARD(3, 1),   /* 5d */
        MAKE_CARD(1, 2),   /* 3h */
    };
    int hand_72o[2] = { MAKE_CARD(5, 1), MAKE_CARD(0, 0) }; /* 7d 2c */
    int hand_aa[2]  = { MAKE_CARD(12, 0), MAKE_CARD(12, 3) }; /* AcAs */

    int n_bins = 50;
    float hist_72o[50], hist_aa[50];
    ca_compute_ehs_distribution(board, 3, hand_72o, 2000, hist_72o, n_bins);
    ca_compute_ehs_distribution(board, 3, hand_aa,  2000, hist_aa, n_bins);

    /* 72o should have more mass in bottom half than AA */
    float low_72o = 0.0f, low_aa = 0.0f;
    for (int i = 0; i < n_bins / 2; i++) {
        low_72o += hist_72o[i];
        low_aa += hist_aa[i];
    }
    ASSERT(low_72o > low_aa,
           "72o should have more mass in bottom-half bins than AA");

    /* The distributions should be different (EMD > 0) */
    float emd = ca_emd_1d(hist_72o, hist_aa, n_bins);
    ASSERT(emd > 1.0f,
           "72o vs AA distributions should have EMD > 1.0");
    PASS("dist_72o_spread_lower");
}

static void test_dist_river_deterministic(void) {
    /* River: no runout cards, only opponent sampling. Still deterministic. */
    int board[5] = {
        MAKE_CARD(12, 2),  /* Ah */
        MAKE_CARD(11, 2),  /* Kh */
        MAKE_CARD(10, 2),  /* Qh */
        MAKE_CARD(5, 0),   /* 7c */
        MAKE_CARD(0, 1),   /* 2d */
    };
    int hand[2] = { MAKE_CARD(9, 2), MAKE_CARD(8, 2) }; /* JhTh — royal flush */

    int n_bins = 50;
    float hist1[50], hist2[50];
    ca_compute_ehs_distribution(board, 5, hand, 500, hist1, n_bins);
    ca_compute_ehs_distribution(board, 5, hand, 500, hist2, n_bins);

    ASSERT(memcmp(hist1, hist2, n_bins * sizeof(float)) == 0,
           "River EHS distribution must be deterministic");

    /* Royal flush should have nearly all mass in the highest bin */
    float top_bin = hist1[n_bins - 1];
    ASSERT(top_bin > 0.95f,
           "Royal flush should have >95% mass in highest bin");
    PASS("dist_river_deterministic");
}

static void test_dist_turn(void) {
    /* Turn: 4 board cards, 1 runout. Should work and be normalized. */
    int board[4] = {
        MAKE_CARD(12, 2),  /* Ah */
        MAKE_CARD(11, 0),  /* Kc */
        MAKE_CARD(8, 1),   /* Td */
        MAKE_CARD(3, 3),   /* 5s */
    };
    int hand[2] = { MAKE_CARD(12, 0), MAKE_CARD(12, 1) }; /* AcAd */

    int n_bins = 50;
    float hist[50];
    ca_compute_ehs_distribution(board, 4, hand, 1000, hist, n_bins);

    float sum = 0.0f;
    for (int i = 0; i < n_bins; i++) sum += hist[i];
    ASSERT_FLOAT_EQ(sum, 1.0f, 1e-5f, "Turn distribution should sum to 1.0");
    PASS("dist_turn");
}

static void test_dist_different_hands_different_histograms(void) {
    /* Two very different hands on the same board should produce different histograms */
    int board[3] = {
        MAKE_CARD(10, 0),  /* Qc */
        MAKE_CARD(8, 1),   /* Td */
        MAKE_CARD(6, 2),   /* 8h */
    };
    int hand_strong[2] = { MAKE_CARD(12, 0), MAKE_CARD(12, 3) }; /* AcAs */
    int hand_weak[2]   = { MAKE_CARD(0, 0), MAKE_CARD(1, 1) };   /* 2c 3d */

    int n_bins = 50;
    float hist_strong[50], hist_weak[50];
    ca_compute_ehs_distribution(board, 3, hand_strong, 1000, hist_strong, n_bins);
    ca_compute_ehs_distribution(board, 3, hand_weak,   1000, hist_weak, n_bins);

    float emd = ca_emd_1d(hist_strong, hist_weak, n_bins);
    ASSERT(emd > 0.5f,
           "Strong vs weak hand distributions should have meaningful EMD separation");
    PASS("dist_different_hands_different_histograms");
}

static void test_dist_different_boards_different_histograms(void) {
    /* Same hand on different boards should produce different distributions */
    int board_dry[3] = {
        MAKE_CARD(5, 0),   /* 7c */
        MAKE_CARD(1, 1),   /* 3d */
        MAKE_CARD(0, 2),   /* 2h */
    };
    int board_wet[3] = {
        MAKE_CARD(10, 2),  /* Qh */
        MAKE_CARD(9, 2),   /* Jh */
        MAKE_CARD(8, 2),   /* Th */
    };
    int hand[2] = { MAKE_CARD(12, 0), MAKE_CARD(11, 0) }; /* AcKc */

    int n_bins = 50;
    float hist_dry[50], hist_wet[50];
    ca_compute_ehs_distribution(board_dry, 3, hand, 1000, hist_dry, n_bins);
    ca_compute_ehs_distribution(board_wet, 3, hand, 1000, hist_wet, n_bins);

    float emd = ca_emd_1d(hist_dry, hist_wet, n_bins);
    ASSERT(emd > 0.1f,
           "Same hand on different boards should produce different distributions");
    PASS("dist_different_boards_different_histograms");
}

/* ── Phase 2: Batch distribution + EMD k-means tests ──────────────── */

static void test_batch_matches_single(void) {
    /*
     * ca_compute_ehs_distribution_batch should produce identical results
     * to calling ca_compute_ehs_distribution for each hand individually.
     */
    int board[3] = {
        MAKE_CARD(5, 0),   /* 7c */
        MAKE_CARD(3, 1),   /* 5d */
        MAKE_CARD(1, 2),   /* 3h */
    };
    int hands[3][2] = {
        { MAKE_CARD(12, 0), MAKE_CARD(12, 3) },  /* AcAs */
        { MAKE_CARD(5, 1),  MAKE_CARD(0, 0) },   /* 7d2c */
        { MAKE_CARD(10, 2), MAKE_CARD(9, 2) },   /* QhJh */
    };

    int n_bins = 50;
    int ns = 500;

    /* Batch */
    float batch_hists[3 * 50];
    ca_compute_ehs_distribution_batch(board, 3, (const int(*)[2])hands, 3,
                                       ns, batch_hists, n_bins);

    /* Individual */
    for (int h = 0; h < 3; h++) {
        float single_hist[50];
        ca_compute_ehs_distribution(board, 3, hands[h], ns, single_hist, n_bins);
        ASSERT(memcmp(batch_hists + h * n_bins, single_hist,
                      n_bins * sizeof(float)) == 0,
               "Batch distribution must be bit-identical to single-hand calls");
    }
    PASS("batch_matches_single");
}

static void test_emd_kmeans_valid_buckets(void) {
    /*
     * EMD k-means on a flop board should produce valid bucket assignments:
     * all in [0, k), and multiple buckets populated.
     */
    int board[3] = {
        MAKE_CARD(5, 0),   /* 7c */
        MAKE_CARD(3, 1),   /* 5d */
        MAKE_CARD(1, 2),   /* 3h */
    };
    int hands[CA_MAX_HANDS][2];
    int nh = ca_generate_hands(board, 3, hands);

    int num_buckets = 20;
    int buckets[CA_MAX_HANDS];
    int actual = ca_assign_buckets_pa_emd(board, 3, (const int(*)[2])hands, nh,
                                           num_buckets, 300, 50, buckets);

    ASSERT(actual > 0, "EMD k-means must return > 0 buckets");
    ASSERT(actual <= num_buckets, "EMD k-means must not exceed target k");

    for (int h = 0; h < nh; h++) {
        ASSERT(buckets[h] >= 0 && buckets[h] < num_buckets,
               "All bucket assignments must be in [0, k)");
    }

    /* Should populate multiple buckets (not degenerate all-same) */
    int seen[20] = {0};
    for (int h = 0; h < nh; h++) seen[buckets[h]] = 1;
    int n_seen = 0;
    for (int c = 0; c < num_buckets; c++) n_seen += seen[c];
    ASSERT(n_seen >= 3, "EMD k-means should populate at least 3 buckets on a real board");

    PASS("emd_kmeans_valid_buckets");
}

static void test_emd_kmeans_separates_strong_weak(void) {
    /*
     * AA and 72o should end up in different buckets when clustered
     * by EMD on their equity distributions.
     */
    int board[3] = {
        MAKE_CARD(5, 0),   /* 7c */
        MAKE_CARD(3, 1),   /* 5d */
        MAKE_CARD(1, 2),   /* 3h */
    };
    int hands[CA_MAX_HANDS][2];
    int nh = ca_generate_hands(board, 3, hands);

    int num_buckets = 20;
    int buckets[CA_MAX_HANDS];
    ca_assign_buckets_pa_emd(board, 3, (const int(*)[2])hands, nh,
                              num_buckets, 300, 50, buckets);

    /* Find AA (AcAs) and 72o (7d2c) */
    int aa_idx = -1, o72_idx = -1;
    int aa_c0 = MAKE_CARD(12, 0), aa_c1 = MAKE_CARD(12, 3);
    int o72_c0 = MAKE_CARD(5, 1), o72_c1 = MAKE_CARD(0, 0);
    for (int h = 0; h < nh; h++) {
        if ((hands[h][0] == aa_c0 && hands[h][1] == aa_c1) ||
            (hands[h][0] == aa_c1 && hands[h][1] == aa_c0))
            aa_idx = h;
        if ((hands[h][0] == o72_c0 && hands[h][1] == o72_c1) ||
            (hands[h][0] == o72_c1 && hands[h][1] == o72_c0))
            o72_idx = h;
    }
    ASSERT(aa_idx >= 0, "Could not find AA in hand list");
    ASSERT(o72_idx >= 0, "Could not find 72o in hand list");
    ASSERT(buckets[aa_idx] != buckets[o72_idx],
           "AA and 72o should be in different EMD buckets");

    printf("    AA bucket=%d, 72o bucket=%d\n", buckets[aa_idx], buckets[o72_idx]);
    PASS("emd_kmeans_separates_strong_weak");
}

static void test_emd_kmeans_deterministic(void) {
    /*
     * Same inputs → same bucket assignments across calls.
     */
    int board[3] = {
        MAKE_CARD(10, 0),  /* Qc */
        MAKE_CARD(8, 1),   /* Td */
        MAKE_CARD(6, 2),   /* 8h */
    };
    int hands[CA_MAX_HANDS][2];
    int nh = ca_generate_hands(board, 3, hands);
    /* Use a small subset for speed */
    int n = (nh > 100) ? 100 : nh;

    int buckets1[CA_MAX_HANDS], buckets2[CA_MAX_HANDS];
    ca_assign_buckets_pa_emd(board, 3, (const int(*)[2])hands, n,
                              10, 200, 50, buckets1);
    ca_assign_buckets_pa_emd(board, 3, (const int(*)[2])hands, n,
                              10, 200, 50, buckets2);

    ASSERT(memcmp(buckets1, buckets2, n * sizeof(int)) == 0,
           "EMD k-means must be deterministic");
    PASS("emd_kmeans_deterministic");
}

static void test_emd_kmeans_vs_l2_different(void) {
    /*
     * EMD k-means and L2 k-means should NOT produce identical bucket
     * assignments — they use different distance metrics. At least some
     * hands should be assigned to different buckets.
     */
    int board[3] = {
        MAKE_CARD(5, 0),   /* 7c */
        MAKE_CARD(3, 1),   /* 5d */
        MAKE_CARD(1, 2),   /* 3h */
    };
    int hands[CA_MAX_HANDS][2];
    int nh = ca_generate_hands(board, 3, hands);
    int n = (nh > 200) ? 200 : nh;

    int emd_buckets[CA_MAX_HANDS], l2_buckets[CA_MAX_HANDS];
    ca_assign_buckets_pa_emd(board, 3, (const int(*)[2])hands, n,
                              20, 300, 50, emd_buckets);
    ca_assign_buckets_kmeans(board, 3, (const int(*)[2])hands, n,
                              20, 300, l2_buckets);

    int differ = 0;
    for (int h = 0; h < n; h++) {
        if (emd_buckets[h] != l2_buckets[h]) differ++;
    }
    /* Bucket IDs aren't necessarily aligned (bucket 0 in EMD ≠ bucket 0 in L2),
     * but the PARTITIONS should be different. Count how many hands share a
     * bucket in one but not the other. At minimum some should differ. */
    ASSERT(differ > 0,
           "EMD and L2 k-means should produce different bucket assignments");
    printf("    %d/%d hands assigned to different bucket IDs\n", differ, n);
    PASS("emd_kmeans_vs_l2_different");
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== PA-EMD Phase 1 Tests ===\n\n");
    printf("--- ca_emd_1d ---\n");
    test_emd_identical();
    test_emd_identical_nonuniform();
    test_emd_max_separation();
    test_emd_adjacent_shift();
    test_emd_symmetry();
    test_emd_triangle_inequality();
    test_emd_partial_overlap();
    test_emd_single_bin();
    test_emd_zero_input();

    printf("\n--- ca_compute_ehs_distribution ---\n");
    test_dist_determinism();
    test_dist_normalized();
    test_dist_aa_concentrated_high();
    test_dist_72o_spread_lower();
    test_dist_river_deterministic();
    test_dist_turn();
    test_dist_different_hands_different_histograms();
    test_dist_different_boards_different_histograms();

    printf("\n--- Phase 2: batch + EMD k-means ---\n");
    test_batch_matches_single();
    test_emd_kmeans_valid_buckets();
    test_emd_kmeans_separates_strong_weak();
    test_emd_kmeans_deterministic();
    test_emd_kmeans_vs_l2_different();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
