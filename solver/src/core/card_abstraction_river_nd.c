/**
 * card_abstraction_river_nd.c — n-dim k-means bucketing + river centroids.
 *
 * Companion to card_abstraction.c. Holds the three
 * n-dimensional clustering routines used for river OCHS bucketing:
 *   ca_assign_buckets_kmeans_nd, ca_nearest_centroid_nd, ca_compute_river_centroids.
 *
 * Compiled into the SAME shared libraries as card_abstraction.c. Shares the
 * EhsPair struct + ca_cmp_ehs comparator via card_abstraction.h.
 */

#include "card_abstraction.h"
#include "hand_eval.h"
#include <stdlib.h>
#include <string.h>

int ca_assign_buckets_kmeans_nd(
    const float *features, int num_hands, int num_dims,
    int num_buckets, int n_iters,
    int *bucket_out
) {
    if (num_hands <= 0 || num_dims <= 0 || num_buckets <= 0) return 0;
    int k = (num_buckets < num_hands) ? num_buckets : num_hands;
    if (k <= 1) { for (int i = 0; i < num_hands; i++) bucket_out[i] = 0; return 1; }

    /* Percentile seed on dim 0 (1D proxy for strength) */
    EhsPair *sorted = (EhsPair*)malloc(num_hands * sizeof(EhsPair));
    for (int i = 0; i < num_hands; i++) {
        sorted[i].idx = i;
        sorted[i].ehs = features[(size_t)i * num_dims];
    }
    qsort(sorted, num_hands, sizeof(EhsPair), ca_cmp_ehs);

    float *centroids = (float*)malloc((size_t)k * num_dims * sizeof(float));
    for (int c = 0; c < k; c++) {
        int si = (int)((float)c / k * num_hands);
        if (si >= num_hands) si = num_hands - 1;
        int fi = sorted[si].idx;
        memcpy(centroids + (size_t)c * num_dims,
               features + (size_t)fi * num_dims,
               num_dims * sizeof(float));
    }
    free(sorted);

    int *counts = (int*)calloc(k, sizeof(int));
    float *sums = (float*)calloc((size_t)k * num_dims, sizeof(float));

    for (int it = 0; it < n_iters; it++) {
        memset(counts, 0, k * sizeof(int));
        memset(sums, 0, (size_t)k * num_dims * sizeof(float));

        for (int h = 0; h < num_hands; h++) {
            const float *fh = features + (size_t)h * num_dims;
            float best_d = 1e30f;
            int best = 0;
            for (int c = 0; c < k; c++) {
                const float *cc = centroids + (size_t)c * num_dims;
                float d = 0.0f;
                for (int dim = 0; dim < num_dims; dim++) {
                    float diff = fh[dim] - cc[dim];
                    d += diff * diff;
                }
                if (d < best_d) { best_d = d; best = c; }
            }
            bucket_out[h] = best;
            counts[best]++;
            float *sc = sums + (size_t)best * num_dims;
            for (int dim = 0; dim < num_dims; dim++) sc[dim] += fh[dim];
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                float *cc = centroids + (size_t)c * num_dims;
                float *sc = sums + (size_t)c * num_dims;
                for (int dim = 0; dim < num_dims; dim++) cc[dim] = sc[dim] / counts[c];
            }
        }
    }

    int actual = 0;
    for (int c = 0; c < k; c++) if (counts[c] > 0) actual++;

    free(centroids); free(counts); free(sums);
    return actual;
}

int ca_nearest_centroid_nd(
    const float *feat, int num_dims,
    const float *centroids, int num_centroids
) {
    float best_d = 1e30f;
    int best = 0;
    for (int c = 0; c < num_centroids; c++) {
        const float *cc = centroids + (size_t)c * num_dims;
        float d = 0.0f;
        for (int dim = 0; dim < num_dims; dim++) {
            float diff = feat[dim] - cc[dim];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; best = c; }
    }
    return best;
}

int ca_compute_river_centroids(
    const int *board, int num_board,
    const int *ochs_cluster_of_169, int num_clusters,
    int num_buckets, int n_iters,
    float *centroids_out
) {
    if (num_board != 5 || num_clusters <= 0 || num_buckets <= 0) return -1;

    int hands[CA_MAX_HANDS][2];
    int nh = ca_generate_hands(board, num_board, hands);
    if (nh <= 0) return -1;

    float *features = (float*)malloc((size_t)nh * num_clusters * sizeof(float));
    if (!features) return -1;

    int rc = ca_river_ochs_features(board, num_board,
                                     (const int(*)[2])hands, nh,
                                     ochs_cluster_of_169, num_clusters,
                                     features);
    if (rc != 0) { free(features); return -1; }

    int *labels = (int*)malloc(nh * sizeof(int));
    if (!labels) { free(features); return -1; }

    int actual = ca_assign_buckets_kmeans_nd(features, nh, num_clusters,
                                              num_buckets, n_iters, labels);

    /* Recompute centroid means from labels */
    int *counts = (int*)calloc(num_buckets, sizeof(int));
    float *sums = (float*)calloc((size_t)num_buckets * num_clusters, sizeof(float));
    for (int h = 0; h < nh; h++) {
        int b = labels[h];
        counts[b]++;
        float *sc = sums + (size_t)b * num_clusters;
        const float *fh = features + (size_t)h * num_clusters;
        for (int c = 0; c < num_clusters; c++) sc[c] += fh[c];
    }
    for (int b = 0; b < num_buckets; b++) {
        float *oc = centroids_out + (size_t)b * num_clusters;
        if (counts[b] > 0) {
            const float *sc = sums + (size_t)b * num_clusters;
            for (int c = 0; c < num_clusters; c++) oc[c] = sc[c] / counts[b];
        } else {
            for (int c = 0; c < num_clusters; c++) oc[c] = 0.0f;
        }
    }

    free(features); free(labels); free(counts); free(sums);
    return actual;
}
