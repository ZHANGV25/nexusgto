/*
 * Regression test for bps_init_ochs_river + bps_hand_to_bucket_river.
 *
 * Asserts:
 *   1. bps_init_ochs_river returns 0 and populates the OCHS structures.
 *   2. After init, bps_hand_to_bucket_river routes through the OCHS path
 *      (8-d feature → nearest centroid), producing different buckets than
 *      the legacy 200-sample EHS percentile path on representative fixtures.
 *   3. The OCHS lookup matches an independent OCHS computation byte-for-byte
 *      using the same locally-recomputed centroid table.
 *
 * Why this matters: a blueprint trained with BPS_RIVER_OCHS=1 buckets river
 * hands via OCHS, so any runtime that doesn't bucket via OCHS will hash to the
 * wrong bucket id and miss the blueprint at every river-card lookup.
 *
 * This test does NOT require the .bps file. It only exercises the bucketer.
 *
 * Build: make test_bps_ochs_river
 * Run:   ./build/test_bps_ochs_river
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "card_abstraction.h"
#include "bps_reader.h"

#define C(r, s) ((r) * 4 + (s))

typedef struct {
    const char *desc;
    int board[5];
    int hand[2];
} TestCase;

int main(void) {
    /* Fixtures span made hands and drawing hands at the river — the cases
     * where EHS-percentile saturation collapses everything to bucket 199 but
     * OCHS preserves separation. */
    TestCase cases[] = {
        { "AKs top-two on flushy AhKsTd5c2c",
          {C(8,2), C(11,3), C(12,2), C(3,0), C(0,0)}, {C(12,3), C(11,0)} },
        { "QQ overpair on dry K-high KhTs2c3d9s",
          {C(11,2), C(8,3), C(0,0), C(1,1), C(7,3)}, {C(10,1), C(10,2)} },
        { "Set of 5s w/ straight Ks5h2d3c4c",
          {C(11,3), C(3,2), C(0,1), C(1,0), C(2,0)}, {C(3,1), C(3,3)} },
        { "Ace-high overcards KsQhJd9s2c",
          {C(11,3), C(10,2), C(9,1), C(7,3), C(0,0)}, {C(12,1), C(4,1)} },
        { "Bottom pair Ah9d8s3c2c",
          {C(12,2), C(7,1), C(6,3), C(1,0), C(0,0)}, {C(11,0), C(1,3)} },
        { "T9s straight on 8h7d6cAsKc",
          {C(6,2), C(5,1), C(4,0), C(12,3), C(11,0)}, {C(8,0), C(7,0)} },
        { "Two pair AA+22 As2c2dKhTs",
          {C(12,3), C(0,0), C(0,1), C(11,2), C(8,3)}, {C(12,2), C(0,3)} },
        { "Pocket 7s no help KQJ47",
          {C(11,3), C(10,2), C(9,1), C(2,0), C(5,0)}, {C(5,2), C(5,3)} },
        { "Set of K on rainbow KKAxx",
          {C(11,2), C(11,3), C(12,0), C(4,1), C(2,2)}, {C(11,0), C(11,1)} },
        { "Quad 9s on 9d9hAsKcTd",
          {C(7,1), C(7,2), C(12,3), C(11,0), C(8,1)}, {C(7,0), C(7,3)} },
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    BPSReader r = {0};
    r.num_postflop_buckets = 200;

    int legacy[10];
    for (int i = 0; i < n_cases; i++) {
        legacy[i] = bps_hand_to_bucket_river(&r, cases[i].board,
                                              cases[i].hand[0],
                                              cases[i].hand[1]);
        if (legacy[i] < 0) {
            fprintf(stderr, "FAIL: legacy lookup returned %d on case %d\n",
                    legacy[i], i);
            return 1;
        }
    }

    int rc = bps_init_ochs_river(&r, 8);
    if (rc != 0) {
        fprintf(stderr, "FAIL: bps_init_ochs_river returned %d\n", rc);
        return 2;
    }
    if (!r.ochs_river_enabled || r.ochs_river_centroids_k <= 0 ||
        r.ochs_num_clusters != 8) {
        fprintf(stderr,
                "FAIL: post-init state invalid (enabled=%d, k=%d, kc=%d)\n",
                r.ochs_river_enabled, r.ochs_river_centroids_k,
                r.ochs_num_clusters);
        return 3;
    }

    /* Independent OCHS reference using r.ochs_river_centroids. */
    int ochs_map[169];
    int actual_clusters = ca_compute_ochs_clusters(8, 2000, ochs_map);
    if (actual_clusters <= 0) {
        fprintf(stderr, "FAIL: ca_compute_ochs_clusters returned %d\n", actual_clusters);
        return 4;
    }

    int n_diff_legacy = 0;
    int n_match_ref = 0;
    for (int i = 0; i < n_cases; i++) {
        int ochs_bucket = bps_hand_to_bucket_river(&r, cases[i].board,
                                                    cases[i].hand[0],
                                                    cases[i].hand[1]);
        float feat[8];
        int hands_one[1][2] = {{ cases[i].hand[0], cases[i].hand[1] }};
        ca_river_ochs_features(cases[i].board, 5,
                                (const int(*)[2])hands_one, 1,
                                ochs_map, 8, feat);
        int ref_bucket = ca_nearest_centroid_nd(feat, 8,
                                                 r.ochs_river_centroids,
                                                 r.ochs_river_centroids_k);
        if (ochs_bucket != ref_bucket) {
            fprintf(stderr,
                    "FAIL: case %d %s: lookup=%d, ref=%d\n",
                    i, cases[i].desc, ochs_bucket, ref_bucket);
            return 5;
        }
        n_match_ref++;
        if (legacy[i] != ochs_bucket) n_diff_legacy++;
    }

    if (n_diff_legacy < 5) {
        fprintf(stderr,
                "FAIL: only %d/%d cases differ between EHS-percentile and OCHS — "
                "OCHS path may not be active\n",
                n_diff_legacy, n_cases);
        return 6;
    }

    bps_reader_free(&r);

    printf("PASS: %d/%d OCHS-lookup matches independent OCHS reference; "
           "%d/%d differ from EHS-percentile.\n",
           n_match_ref, n_cases, n_diff_legacy, n_cases);
    return 0;
}
