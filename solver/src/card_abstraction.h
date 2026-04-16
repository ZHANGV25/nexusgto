/**
 * card_abstraction.h — EHS computation + k-means bucketing for card abstraction
 *
 * Computes Expected Hand Strength (EHS) for poker hands via Monte Carlo
 * sampling, then clusters into buckets using k-means on domain features.
 *
 * Pluribus uses 169 lossless preflop buckets and 200 lossy buckets
 * per postflop street, clustered via k-means on domain-specific features
 * (EHS + hand potential), per Johanson et al. 2013.
 *
 * Two bucketing methods available:
 *   ca_assign_buckets()        — fast percentile bucketing (1D, EHS only)
 *   ca_assign_buckets_kmeans() — k-means on [EHS, positive_potential, negative_potential]
 */
#ifndef CARD_ABSTRACTION_H
#define CARD_ABSTRACTION_H

#include <stdint.h>

#define CA_MAX_HANDS 1326    /* max possible 2-card hands from 52 cards */

#ifdef _WIN32
#define CA_EXPORT __declspec(dllexport)
#else
#define CA_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute Expected Hand Strength for a set of hands on a given board.
 *
 * EHS = (wins + 0.5*ties) / total against random opponent hands,
 * averaged over random board completions (for flop/turn).
 *
 * Args:
 *   board: board cards (3=flop, 4=turn, 5=river)
 *   num_board: number of board cards
 *   hands: array of (c0, c1) pairs
 *   num_hands: number of hands
 *   n_samples: Monte Carlo samples per hand (more = more accurate)
 *   ehs_out: [num_hands] output array of EHS values in [0, 1]
 *
 * Returns 0 on success.
 */
CA_EXPORT int ca_compute_ehs(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    int n_samples,
    float *ehs_out
);

/**
 * Assign hands to buckets via percentile bucketing on EHS.
 *
 * Sorts hands by EHS, divides into num_buckets equal-population groups.
 * Bucket 0 = weakest hands, bucket num_buckets-1 = strongest.
 *
 * Args:
 *   ehs: [num_hands] EHS values (from ca_compute_ehs)
 *   num_hands: number of hands
 *   num_buckets: target number of buckets
 *   bucket_out: [num_hands] output bucket index for each hand
 *
 * Returns actual number of buckets (may be < num_buckets if fewer hands).
 */
CA_EXPORT int ca_assign_buckets(
    const float *ehs, int num_hands,
    int num_buckets,
    int *bucket_out
);

/**
 * Generate all 1326 possible 2-card hands from a 52-card deck,
 * excluding cards in the given board.
 *
 * Args:
 *   board: cards to exclude
 *   num_board: number of board cards
 *   hands_out: [CA_MAX_HANDS][2] output array
 *
 * Returns number of hands generated.
 */
CA_EXPORT int ca_generate_hands(
    const int *board, int num_board,
    int hands_out[][2]
);

/**
 * Generate the 169 strategically unique preflop hand classes.
 * Each class is (rank1, rank2, suited), mapped to a representative hand.
 *
 * Args:
 *   classes_out: [169][2] representative (c0, c1) for each class
 *   hand_to_class: [1326] maps each of the 1326 hands to its class index
 *   hands: [1326][2] all possible hands (from ca_generate_hands with empty board)
 *   num_hands: 1326
 *
 * Returns 169.
 */
/**
 * Assign hands to buckets via k-means clustering on domain features.
 *
 * Features per hand (Johanson et al. 2013 / Pluribus):
 *   [0] EHS — expected hand strength vs random opponent
 *   [1] Positive potential — P(behind now but ahead after more cards)
 *   [2] Negative potential — P(ahead now but behind after more cards)
 *
 * K-means with Lloyd's algorithm, 20 iterations, percentile-seeded centroids.
 *
 * Args:
 *   board, num_board: current board cards
 *   hands: array of (c0, c1) pairs
 *   num_hands: number of hands
 *   num_buckets: target k
 *   n_samples: Monte Carlo samples for EHS + potential computation
 *   bucket_out: [num_hands] output bucket index for each hand
 *
 * Returns actual number of buckets used.
 */
CA_EXPORT int ca_assign_buckets_kmeans(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    int num_buckets, int n_samples,
    int *bucket_out
);

/**
 * Compute [EHS, PPot, NPot] feature vectors for a set of hands.
 * Used for k-means bucketing and centroid precomputation.
 */
CA_EXPORT void ca_compute_features(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    int n_samples,
    float features[][3]
);

/**
 * Find the nearest centroid for a single 3D feature vector.
 * Returns the centroid index (0 to k-1).
 */
static inline int ca_nearest_centroid(
    const float feat[3], const float centroids[][3], int k
) {
    float best_dist = 1e30f;
    int best = 0;
    for (int c = 0; c < k; c++) {
        float d0 = feat[0] - centroids[c][0];
        float d1 = feat[1] - centroids[c][1];
        float d2 = feat[2] - centroids[c][2];
        float dist = d0*d0 + d1*d1 + d2*d2;
        if (dist < best_dist) { best_dist = dist; best = c; }
    }
    return best;
}

CA_EXPORT int ca_preflop_classes(
    int classes_out[][2],
    int *hand_to_class,
    const int hands[][2], int num_hands
);

/* ── OCHS (Opponent Cluster Hand Strength) — Johanson 2013 ──────── */

#define CA_OCHS_DEFAULT_CLUSTERS 8
#define CA_OCHS_DIMS CA_OCHS_DEFAULT_CLUSTERS

/**
 * Compute preflop all-in EHS for each of the 169 classes (vs random hand
 * over 5 random board cards), then k-means cluster the 169 scalars into
 * `num_opponent_clusters` groups. Output is `cluster_map[169]` in [0, k-1].
 *
 * Ordering matches ca_preflop_classes (pairs 0-12, then suited+offsuit
 * pairs in descending rank order). Pluribus-faithful default: k=8.
 * Deterministic: fixed RNG seed; same inputs produce same output.
 *
 * Returns the number of clusters actually populated (<= num_opponent_clusters).
 */
CA_EXPORT int ca_compute_ochs_clusters(
    int num_opponent_clusters,
    int ehs_samples,
    int cluster_map[169]
);

/**
 * Serialize OCHS cluster map to disk (magic "OCHS" + version + k + int32[169]).
 * Returns 0 on success.
 */
CA_EXPORT int ca_save_ochs_clusters(
    const char *path, int num_clusters, const int cluster_map[169]
);

/**
 * Load OCHS cluster map from disk. Returns num_clusters on success, -1 on error.
 */
CA_EXPORT int ca_load_ochs_clusters(
    const char *path, int cluster_map[169]
);

/**
 * Compute 8-dim OCHS feature vector for each hand on a river board.
 *
 * For each hand h and each opponent cluster i, features[h][i] = showdown
 * equity of h against the uniform distribution over 2-card combos whose
 * preflop class is in cluster i, excluding combos that conflict with
 * h or the board. If a cluster has zero valid opponent combos the entry
 * is 0.5 (uninformative).
 *
 * Args:
 *   board:         river board (5 cards)
 *   num_board:     must be 5
 *   hands:         [num_hands][2] hero hand cards
 *   num_hands:     number of hands
 *   ochs_cluster_of_169: precomputed cluster map (from ca_compute_ochs_clusters)
 *   num_clusters:  k (8 for Pluribus-faithful)
 *   features:      [num_hands][num_clusters] output, each value in [0, 1]
 *
 * Returns 0 on success, -1 on invalid input.
 */
CA_EXPORT int ca_river_ochs_features(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    const int *ochs_cluster_of_169,
    int num_clusters,
    float *features
);

/**
 * Generic n-dim k-means bucketing (L2 distance, percentile-seeded on dim 0).
 *
 * Args:
 *   features:     [num_hands * num_dims] flat row-major input
 *   num_hands:    number of hands
 *   num_dims:     feature dimensionality
 *   num_buckets:  target k
 *   n_iters:      Lloyd iterations (20 is standard)
 *   bucket_out:   [num_hands] bucket assignments
 *
 * Returns actual number of non-empty buckets.
 */
CA_EXPORT int ca_assign_buckets_kmeans_nd(
    const float *features, int num_hands, int num_dims,
    int num_buckets, int n_iters,
    int *bucket_out
);

/**
 * Nearest centroid lookup in n-dim L2 space.
 */
CA_EXPORT int ca_nearest_centroid_nd(
    const float *feat, int num_dims,
    const float *centroids, int num_centroids
);

/**
 * Compute num_buckets river centroids for a specific board by k-means on
 * OCHS features. `centroids_out` must be sized [num_buckets * num_clusters].
 *
 * Returns actual number of non-empty centroids.
 */
CA_EXPORT int ca_compute_river_centroids(
    const int *board, int num_board,
    const int *ochs_cluster_of_169, int num_clusters,
    int num_buckets, int n_iters,
    float *centroids_out
);

/* ── PA-EMD (Potential-Aware Earth Mover's Distance) ─────────────── */

/**
 * 1D Earth Mover's Distance between two histograms.
 *
 * For normalized 1D histograms, EMD = sum_i |CDF_a(i) - CDF_b(i)|.
 * Closed-form O(n_bins) — no linear programming needed.
 *
 * Both histograms must sum to 1.0 (or at least to the same total).
 * If either is all-zero, returns 0.0.
 *
 * Args:
 *   hist_a, hist_b: [n_bins] histogram vectors (probability mass per bin)
 *   n_bins: number of bins (must be > 0)
 *
 * Returns EMD in [0, n_bins - 1]. Distance 0 means identical distributions.
 */
CA_EXPORT float ca_emd_1d(
    const float *hist_a, const float *hist_b,
    int n_bins
);

/**
 * Compute hand equity distribution as a histogram (potential-aware
 * representation per Ganzfried & Sandholm 2014 / Johanson 2013).
 *
 * For each MC sample, draws an opponent hand + any remaining board cards,
 * evaluates showdown equity (1.0 win, 0.5 tie, 0.0 loss), and bins the
 * result into the histogram. The histogram captures the DISTRIBUTION of
 * outcomes — not just the scalar mean — preserving information about
 * hand volatility (draw potential, vulnerability).
 *
 * Street behavior:
 *   - Flop (n_board=3): samples opp hand + turn + river
 *   - Turn (n_board=4): samples opp hand + river
 *   - River (n_board=5): samples opp hand only
 *
 * RNG is deterministic: seeded from board + hand cards. Same inputs
 * produce identical output across calls and platforms.
 *
 * Args:
 *   board:       board cards (3/4/5)
 *   n_board:     number of board cards
 *   hand:        hero's 2 hole cards [c0, c1]
 *   num_samples: MC samples for opponent + runout (e.g. 1000)
 *   out_hist:    [n_bins] output histogram, normalized to sum=1.0
 *   n_bins:      number of histogram bins (e.g. 50)
 */
CA_EXPORT void ca_compute_ehs_distribution(
    const int *board, int n_board,
    const int *hand,
    int num_samples,
    float *out_hist,
    int n_bins
);

#ifdef __cplusplus
}
#endif

#endif /* CARD_ABSTRACTION_H */
