/**
 * card_abstraction_ochs.c — OCHS (Johanson 2013, AAMAS) clustering + IO.
 *
 * Split out of card_abstraction.c (no behavior change). Holds the OCHS
 * opponent-cluster machinery: 169-class representatives, hand->class mapping,
 * 1-D k-means over the 169 preflop EHS values, cluster save/load, and the
 * river OCHS feature computation (ca_river_ochs_features).
 *
 * Compiled into the SAME shared libraries as card_abstraction.c. Shares the
 * EhsPair struct + ca_cmp_ehs comparator via card_abstraction.h. The three
 * helpers (ochs_build_class_reps, ochs_hand_to_class, ochs_kmeans_1d) stay
 * file-static here.
 */

#include "card_abstraction.h"
#include "hand_eval.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── OCHS (Johanson 2013, AAMAS) ─────────────────────────────────── */

/* Build a class_to_rep[169][2] table for a representative (c0, c1) of each
 * of the 169 classes. Matches ca_preflop_classes ordering. */
static void ochs_build_class_reps(int class_reps[169][2]) {
    int n = 0;
    for (int r0 = 12; r0 >= 0; r0--) {
        for (int r1 = r0; r1 >= 0; r1--) {
            if (r0 == r1) {
                class_reps[n][0] = MAKE_CARD(r0, 0);
                class_reps[n][1] = MAKE_CARD(r1, 1);
                n++;
            } else {
                /* suited */
                class_reps[n][0] = MAKE_CARD(r0, 0);
                class_reps[n][1] = MAKE_CARD(r1, 0);
                n++;
                /* offsuit */
                class_reps[n][0] = MAKE_CARD(r0, 0);
                class_reps[n][1] = MAKE_CARD(r1, 1);
                n++;
            }
        }
    }
}

/* Map a 2-card hand to its class index (0-168). */
static int ochs_hand_to_class(int c0, int c1) {
    int r0 = CARD_RANK(c0), r1 = CARD_RANK(c1);
    int suited = (CARD_SUIT(c0) == CARD_SUIT(c1)) ? 1 : 0;
    if (r0 < r1) { int tmp = r0; r0 = r1; r1 = tmp; }
    /* Walk the same enumeration order as ca_preflop_classes. */
    int idx = 0;
    for (int a = 12; a >= 0; a--) {
        for (int b = a; b >= 0; b--) {
            if (a == b) {
                if (r0 == a && r1 == b) return idx;
                idx++;
            } else {
                /* suited */
                if (r0 == a && r1 == b && suited) return idx;
                idx++;
                /* offsuit */
                if (r0 == a && r1 == b && !suited) return idx;
                idx++;
            }
        }
    }
    return -1;
}

/* 1-D k-means on scalars (used to cluster the 169 preflop EHS values).
 * Percentile-seeded, L2 distance. */
static int ochs_kmeans_1d(const float *values, int n, int k, int n_iters,
                           int *labels_out) {
    if (n <= 0 || k <= 0) return 0;
    if (k > n) k = n;

    /* Percentile seed on sorted values */
    EhsPair *sorted = (EhsPair*)malloc(n * sizeof(EhsPair));
    for (int i = 0; i < n; i++) { sorted[i].idx = i; sorted[i].ehs = values[i]; }
    qsort(sorted, n, sizeof(EhsPair), ca_cmp_ehs);

    float *centroids = (float*)malloc(k * sizeof(float));
    for (int c = 0; c < k; c++) {
        int si = (int)((float)c / k * n);
        if (si >= n) si = n - 1;
        centroids[c] = sorted[si].ehs;
    }
    free(sorted);

    int *counts = (int*)calloc(k, sizeof(int));
    float *sums = (float*)calloc(k, sizeof(float));

    for (int it = 0; it < n_iters; it++) {
        memset(counts, 0, k * sizeof(int));
        memset(sums, 0, k * sizeof(float));
        for (int i = 0; i < n; i++) {
            float v = values[i];
            int best = 0;
            float best_d = (v - centroids[0]) * (v - centroids[0]);
            for (int c = 1; c < k; c++) {
                float d = (v - centroids[c]) * (v - centroids[c]);
                if (d < best_d) { best_d = d; best = c; }
            }
            labels_out[i] = best;
            counts[best]++;
            sums[best] += v;
        }
        for (int c = 0; c < k; c++)
            if (counts[c] > 0) centroids[c] = sums[c] / counts[c];
    }

    /* Renumber clusters so that lower centroid -> lower label. Stabilizes
     * output across runs and makes "bottom cluster = weakest" meaningful. */
    int *order = (int*)malloc(k * sizeof(int));
    for (int c = 0; c < k; c++) order[c] = c;
    for (int i = 1; i < k; i++) {
        int key = order[i];
        float key_v = centroids[key];
        int j = i - 1;
        while (j >= 0 && centroids[order[j]] > key_v) { order[j+1] = order[j]; j--; }
        order[j+1] = key;
    }
    int *remap = (int*)malloc(k * sizeof(int));
    for (int c = 0; c < k; c++) remap[order[c]] = c;
    for (int i = 0; i < n; i++) labels_out[i] = remap[labels_out[i]];

    int actual = 0;
    for (int c = 0; c < k; c++) if (counts[c] > 0) actual++;

    free(order); free(remap); free(centroids); free(counts); free(sums);
    return actual;
}

int ca_compute_ochs_clusters(
    int num_opponent_clusters,
    int ehs_samples,
    int cluster_map[169]
) {
    if (num_opponent_clusters <= 0 || num_opponent_clusters > 169) return -1;
    if (ehs_samples <= 0) ehs_samples = 2000;

    int class_reps[169][2];
    ochs_build_class_reps(class_reps);

    /* Compute all-in EHS for each class: MC over random opponent hand + random
     * 5-card board, 0 board cards given. Fixed seed for determinism.
     * Pass a dummy board pointer (len=0 means it won't be read) to avoid
     * any compilers flagging a NULL memcpy src. */
    int dummy_board[1] = {0};
    float ehs_169[169];
    for (int cls = 0; cls < 169; cls++) {
        int hand[1][2] = {{ class_reps[cls][0], class_reps[cls][1] }};
        float e;
        ca_compute_ehs(dummy_board, 0, (const int(*)[2])hand, 1, ehs_samples, &e);
        ehs_169[cls] = e;
    }

    int labels[169];
    int actual = ochs_kmeans_1d(ehs_169, 169, num_opponent_clusters, 50, labels);
    for (int i = 0; i < 169; i++) cluster_map[i] = labels[i];
    return actual;
}

#define OCHS_MAGIC "OCHS"
#define OCHS_VERSION 1

int ca_save_ochs_clusters(const char *path, int num_clusters,
                          const int cluster_map[169]) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int version = OCHS_VERSION;
    int k = num_clusters;
    int count = 169;
    int ok = 1;
    if (fwrite(OCHS_MAGIC, 1, 4, f) != 4) ok = 0;
    if (ok && fwrite(&version, sizeof(int), 1, f) != 1) ok = 0;
    if (ok && fwrite(&k, sizeof(int), 1, f) != 1) ok = 0;
    if (ok && fwrite(&count, sizeof(int), 1, f) != 1) ok = 0;
    if (ok && fwrite(cluster_map, sizeof(int), 169, f) != 169) ok = 0;
    fclose(f);
    return ok ? 0 : -1;
}

int ca_load_ochs_clusters(const char *path, int cluster_map[169]) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    int version = 0, k = 0, count = 0;
    int ok = 1;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, OCHS_MAGIC, 4) != 0) ok = 0;
    if (ok && fread(&version, sizeof(int), 1, f) != 1) ok = 0;
    if (ok && version != OCHS_VERSION) ok = 0;
    if (ok && fread(&k, sizeof(int), 1, f) != 1) ok = 0;
    if (ok && fread(&count, sizeof(int), 1, f) != 1) ok = 0;
    if (ok && count != 169) ok = 0;
    if (ok && fread(cluster_map, sizeof(int), 169, f) != 169) ok = 0;
    fclose(f);
    return ok ? k : -1;
}

int ca_river_ochs_features(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    const int *ochs_cluster_of_169,
    int num_clusters,
    float *features  /* [num_hands * num_clusters] */
) {
    if (num_board != 5 || num_hands <= 0 || num_clusters <= 0) return -1;
    if (!board || !hands || !ochs_cluster_of_169 || !features) return -1;

    int board_blocked[52] = {0};
    for (int i = 0; i < 5; i++) board_blocked[board[i]] = 1;

    /* Precompute evaluated board strength vs each of the 1326 opponent combos
     * (skipping those that conflict with the board) once per board. For each
     * hero hand we only need to exclude the hero's two cards from the
     * 1326-sized opponent enumeration.
     *
     * Store per-combo: (c0, c1, cluster, opp_strength). */
    typedef struct {
        int c0, c1;
        int cluster;
        uint32_t opp_str;
    } OppCombo;

    OppCombo *combos = (OppCombo*)malloc(1326 * sizeof(OppCombo));
    int n_combos = 0;
    for (int c0 = 0; c0 < 52; c0++) {
        if (board_blocked[c0]) continue;
        for (int c1 = c0 + 1; c1 < 52; c1++) {
            if (board_blocked[c1]) continue;
            int cls = ochs_hand_to_class(c0, c1);
            if (cls < 0) continue;
            int cluster = ochs_cluster_of_169[cls];
            if (cluster < 0 || cluster >= num_clusters) continue;
            int opp7[7] = {board[0], board[1], board[2], board[3], board[4],
                           c0, c1};
            combos[n_combos].c0 = c0;
            combos[n_combos].c1 = c1;
            combos[n_combos].cluster = cluster;
            combos[n_combos].opp_str = eval7(opp7);
            n_combos++;
        }
    }

    /* Per-cluster accumulators. Heap-alloc to allow num_clusters > 8 without
     * requiring VLAs (MSVC-incompatible). */
    double *sum_eq = (double*)malloc(num_clusters * sizeof(double));
    int *cnt = (int*)malloc(num_clusters * sizeof(int));
    if (!sum_eq || !cnt) { free(combos); free(sum_eq); free(cnt); return -1; }

    /* Per-hand k-dim features */
    for (int h = 0; h < num_hands; h++) {
        int hc0 = hands[h][0], hc1 = hands[h][1];
        int hero7[7] = {board[0], board[1], board[2], board[3], board[4],
                        hc0, hc1};
        uint32_t hero_str = eval7(hero7);

        for (int c = 0; c < num_clusters; c++) { sum_eq[c] = 0.0; cnt[c] = 0; }

        for (int i = 0; i < n_combos; i++) {
            int oc0 = combos[i].c0, oc1 = combos[i].c1;
            if (oc0 == hc0 || oc0 == hc1 || oc1 == hc0 || oc1 == hc1) continue;
            int cl = combos[i].cluster;
            uint32_t os = combos[i].opp_str;
            double eq;
            if (hero_str > os) eq = 1.0;
            else if (hero_str == os) eq = 0.5;
            else eq = 0.0;
            sum_eq[cl] += eq;
            cnt[cl]++;
        }

        float *out = features + (size_t)h * num_clusters;
        for (int c = 0; c < num_clusters; c++) {
            out[c] = (cnt[c] > 0) ? (float)(sum_eq[c] / cnt[c]) : 0.5f;
        }
    }

    free(combos);
    free(sum_eq);
    free(cnt);
    return 0;
}
