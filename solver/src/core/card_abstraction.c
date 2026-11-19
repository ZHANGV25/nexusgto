/**
 * card_abstraction.c — Fast EHS computation + percentile bucketing
 *
 * Uses hand_eval.h for 7-card evaluation via Monte Carlo sampling.
 * Designed for speed: ~1000 hands × 1000 samples = ~1M eval7 calls,
 * each taking ~200ns = ~0.2 seconds total on modern CPU.
 */

#include "card_abstraction.h"
#include "hand_eval.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── RNG ──────────────────────────────────────────────────────────── */

static inline uint64_t ca_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline int ca_rng_int(uint64_t *state, int n) {
    return (int)(ca_rng_next(state) % (uint64_t)n);
}

/* Fisher-Yates partial shuffle: pick k items from array of n */
static void partial_shuffle(int *arr, int n, int k, uint64_t *rng) {
    for (int i = 0; i < k && i < n; i++) {
        int j = i + ca_rng_int(rng, n - i);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* ── EHS computation ─────────────────────────────────────────────── */

int ca_compute_ehs(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    int n_samples,
    float *ehs_out
) {
    if (num_hands <= 0 || n_samples <= 0) return -1;

    /* Build blocked card set from board */
    int board_blocked[52] = {0};
    for (int i = 0; i < num_board; i++)
        board_blocked[board[i]] = 1;

    uint64_t rng_state = 0xCAFEBABE12345678ULL;

    for (int h = 0; h < num_hands; h++) {
        int c0 = hands[h][0], c1 = hands[h][1];

        /* Cards blocked by this hand + board */
        int blocked[52];
        memcpy(blocked, board_blocked, sizeof(blocked));
        blocked[c0] = 1;
        blocked[c1] = 1;

        /* Available cards for opponent + board completion */
        int avail[52];
        int n_avail = 0;
        for (int c = 0; c < 52; c++)
            if (!blocked[c]) avail[n_avail++] = c;

        int wins = 0, ties = 0, total = 0;
        int cards_needed = 2 + (5 - num_board); /* 2 opp + remaining board */

        for (int s = 0; s < n_samples; s++) {
            if (n_avail < cards_needed) break;

            /* Shuffle and pick: first 2 = opponent, rest = board completion */
            partial_shuffle(avail, n_avail, cards_needed, &rng_state);

            int oc0 = avail[0], oc1 = avail[1];

            /* Build full 5-card board */
            int full_board[5];
            int nb_copy = (num_board <= 5) ? num_board : 5;
            memcpy(full_board, board, nb_copy * sizeof(int));
            for (int b = num_board; b < 5; b++)
                full_board[b] = avail[2 + (b - num_board)];

            /* Evaluate both hands */
            int hero_cards[7] = {full_board[0], full_board[1], full_board[2],
                                  full_board[3], full_board[4], c0, c1};
            int opp_cards[7] = {full_board[0], full_board[1], full_board[2],
                                 full_board[3], full_board[4], oc0, oc1};

            uint32_t hero_str = eval7(hero_cards);
            uint32_t opp_str = eval7(opp_cards);

            if (hero_str > opp_str) wins++;
            else if (hero_str == opp_str) ties++;
            total++;
        }

        ehs_out[h] = (total > 0) ? ((float)wins + 0.5f * (float)ties) / (float)total : 0.5f;
    }

    return 0;
}

/* ── Percentile bucketing ────────────────────────────────────────── */

/* Comparison function for sorting by EHS.
 * EhsPair typedef + this comparator are declared in card_abstraction.h and
 * defined here (the single non-static definition) so the river-nd / ochs
 * split translation units can reuse the same qsort callback. */
int ca_cmp_ehs(const void *a, const void *b) {
    float ea = ((const EhsPair*)a)->ehs;
    float eb = ((const EhsPair*)b)->ehs;
    if (ea < eb) return -1;
    if (ea > eb) return 1;
    return 0;
}

int ca_assign_buckets(
    const float *ehs, int num_hands,
    int num_buckets,
    int *bucket_out
) {
    if (num_hands <= 0) return 0;
    int actual = (num_buckets < num_hands) ? num_buckets : num_hands;

    /* Sort by EHS */
    EhsPair *pairs = (EhsPair*)malloc(num_hands * sizeof(EhsPair));
    for (int i = 0; i < num_hands; i++) {
        pairs[i].idx = i;
        pairs[i].ehs = ehs[i];
    }
    qsort(pairs, num_hands, sizeof(EhsPair), ca_cmp_ehs);

    /* Assign buckets: equal-population percentile */
    for (int rank = 0; rank < num_hands; rank++) {
        int bucket = rank * actual / num_hands;
        if (bucket >= actual) bucket = actual - 1;
        bucket_out[pairs[rank].idx] = bucket;
    }

    free(pairs);
    return actual;
}

/* ── K-means bucketing on domain features (Pluribus-style) ──────── */

/* Compute EHS + positive/negative hand potential via Monte Carlo.
 * Features: [EHS, pos_potential, neg_potential] per hand.
 * Positive potential: fraction of samples where hand is behind now but wins
 * Negative potential: fraction of samples where hand is ahead now but loses */
void ca_compute_features(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    int n_samples,
    float features[][3]  /* [num_hands][3] output */
) {
    int board_blocked[52] = {0};
    for (int i = 0; i < num_board; i++)
        board_blocked[board[i]] = 1;

    uint64_t rng = 0xFEA70BE12345678ULL;

    for (int h = 0; h < num_hands; h++) {
        int c0 = hands[h][0], c1 = hands[h][1];

        int blocked[52];
        memcpy(blocked, board_blocked, sizeof(blocked));
        blocked[c0] = 1;
        blocked[c1] = 1;

        int avail[52], n_avail = 0;
        for (int c = 0; c < 52; c++)
            if (!blocked[c]) avail[n_avail++] = c;

        int wins = 0, ties = 0, total = 0;
        int behind_now_win_later = 0, behind_now_total = 0;
        int ahead_now_lose_later = 0, ahead_now_total = 0;

        int cards_needed = 2 + (5 - num_board);
        if (n_avail < cards_needed) {
            features[h][0] = 0.5f;
            features[h][1] = 0.0f;
            features[h][2] = 0.0f;
            continue;
        }

        for (int s = 0; s < n_samples; s++) {
            partial_shuffle(avail, n_avail, cards_needed, &rng);
            int oc0 = avail[0], oc1 = avail[1];

            int full_board[5];
            memcpy(full_board, board, num_board * sizeof(int));
            for (int b = num_board; b < 5; b++)
                full_board[b] = avail[2 + (b - num_board)];

            /* Evaluate at full board (5 cards) for EHS */
            int hero7_full[7] = {full_board[0], full_board[1], full_board[2],
                                  full_board[3], full_board[4], c0, c1};
            int opp7_full[7] = {full_board[0], full_board[1], full_board[2],
                                 full_board[3], full_board[4], oc0, oc1};
            uint32_t hero_full = eval7(hero7_full);
            uint32_t opp_full = eval7(opp7_full);

            if (hero_full > opp_full) wins++;
            else if (hero_full == opp_full) ties++;
            total++;

            /* Hand potential: compare "current" vs "final" strength.
             * Current = strength using only the board cards dealt so far.
             * Final = strength using the complete 5-card board.
             * eval7 needs 7 cards. With < 5 board cards, pad the
             * current board with INDEPENDENT random cards (not the actual
             * future cards used in full_board). This ensures "current" and
             * "final" evaluations can differ, enabling PPot/NPot > 0.
             * Use cards from the END of the avail array (after the ones
             * used for opp hand + board completion). */
            if (num_board < 5) {
                int hero_cur[7], opp_cur[7];
                for (int i = 0; i < num_board; i++) {
                    hero_cur[i] = board[i];
                    opp_cur[i] = board[i];
                }
                /* Pad with cards from the tail of avail (independent of
                 * the future board cards at avail[2..cards_needed-1]) */
                int pad_src = cards_needed; /* start after used cards */
                for (int i = num_board; i < 5; i++) {
                    int pad_card = (pad_src < n_avail) ? avail[pad_src++] : avail[0];
                    hero_cur[i] = pad_card;
                    opp_cur[i] = pad_card;
                }
                hero_cur[5] = c0; hero_cur[6] = c1;
                opp_cur[5] = oc0; opp_cur[6] = oc1;
                uint32_t hc = eval7(hero_cur);
                uint32_t oc = eval7(opp_cur);

                int cur_ahead = (hc > oc);
                int cur_behind = (hc < oc);
                int final_win = (hero_full > opp_full);
                int final_lose = (hero_full < opp_full);

                if (cur_behind) {
                    behind_now_total++;
                    if (final_win) behind_now_win_later++;
                }
                if (cur_ahead) {
                    ahead_now_total++;
                    if (final_lose) ahead_now_lose_later++;
                }
            }
        }

        features[h][0] = (total > 0) ? ((float)wins + 0.5f * (float)ties) / (float)total : 0.5f;
        features[h][1] = (behind_now_total > 0) ? (float)behind_now_win_later / (float)behind_now_total : 0.0f;
        features[h][2] = (ahead_now_total > 0) ? (float)ahead_now_lose_later / (float)ahead_now_total : 0.0f;
    }
}

int ca_assign_buckets_kmeans(
    const int *board, int num_board,
    const int hands[][2], int num_hands,
    int num_buckets, int n_samples,
    int *bucket_out
) {
    if (num_hands <= 0) return 0;
    int k = (num_buckets < num_hands) ? num_buckets : num_hands;
    if (k <= 1) { for (int i = 0; i < num_hands; i++) bucket_out[i] = 0; return 1; }

    /* Step 1: Compute 3D features */
    float (*features)[3] = (float(*)[3])malloc(num_hands * 3 * sizeof(float));
    ca_compute_features(board, num_board, hands, num_hands, n_samples, features);

    /* Step 2: Initialize centroids via percentile seeding on EHS (feature[0]).
     * Sort by EHS and pick k evenly-spaced points. */
    EhsPair *sorted = (EhsPair*)malloc(num_hands * sizeof(EhsPair));
    for (int i = 0; i < num_hands; i++) {
        sorted[i].idx = i;
        sorted[i].ehs = features[i][0];
    }
    qsort(sorted, num_hands, sizeof(EhsPair), ca_cmp_ehs);

    float (*centroids)[3] = (float(*)[3])malloc(k * 3 * sizeof(float));
    for (int c = 0; c < k; c++) {
        int seed_idx = sorted[(int)((float)c / k * num_hands)].idx;
        centroids[c][0] = features[seed_idx][0];
        centroids[c][1] = features[seed_idx][1];
        centroids[c][2] = features[seed_idx][2];
    }
    free(sorted);

    /* Step 3: Lloyd's k-means, 20 iterations */
    int *counts = (int*)calloc(k, sizeof(int));
    float (*sums)[3] = (float(*)[3])calloc(k * 3, sizeof(float));

    for (int iter = 0; iter < 20; iter++) {
        /* Assign each hand to nearest centroid (L2 distance in feature space) */
        memset(counts, 0, k * sizeof(int));
        memset(sums, 0, k * 3 * sizeof(float));

        for (int h = 0; h < num_hands; h++) {
            float best_dist = 1e30f;
            int best_c = 0;
            for (int c = 0; c < k; c++) {
                float d0 = features[h][0] - centroids[c][0];
                float d1 = features[h][1] - centroids[c][1];
                float d2 = features[h][2] - centroids[c][2];
                float dist = d0*d0 + d1*d1 + d2*d2;
                if (dist < best_dist) { best_dist = dist; best_c = c; }
            }
            bucket_out[h] = best_c;
            counts[best_c]++;
            sums[best_c][0] += features[h][0];
            sums[best_c][1] += features[h][1];
            sums[best_c][2] += features[h][2];
        }

        /* Update centroids */
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                centroids[c][0] = sums[c][0] / counts[c];
                centroids[c][1] = sums[c][1] / counts[c];
                centroids[c][2] = sums[c][2] / counts[c];
            }
        }
    }

    /* Count actual non-empty buckets */
    int actual = 0;
    for (int c = 0; c < k; c++)
        if (counts[c] > 0) actual++;

    free(features);
    free(centroids);
    free(counts);
    free(sums);
    return actual;
}

/* ── Hand generation ─────────────────────────────────────────────── */

int ca_generate_hands(
    const int *board, int num_board,
    int hands_out[][2]
) {
    int blocked[52] = {0};
    for (int i = 0; i < num_board; i++)
        blocked[board[i]] = 1;

    int n = 0;
    for (int c0 = 0; c0 < 52; c0++) {
        if (blocked[c0]) continue;
        for (int c1 = c0 + 1; c1 < 52; c1++) {
            if (blocked[c1]) continue;
            if (n >= CA_MAX_HANDS) goto done;
            hands_out[n][0] = c0;
            hands_out[n][1] = c1;
            n++;
        }
    }
done:
    return n;
}

/* ── Preflop hand classes (169 strategically unique) ─────────────── */

int ca_preflop_classes(
    int classes_out[][2],
    int *hand_to_class,
    const int hands[][2], int num_hands
) {
    /* 169 classes: 13 pairs + 78 suited + 78 offsuit
     * Encoding: pairs at 0-12, suited at 13-90, offsuit at 91-168
     * Class = f(rank0, rank1, suited) */
    int n_classes = 0;
    int class_map[13][13][2]; /* [rank_hi][rank_lo][suited] = class_idx */
    memset(class_map, -1, sizeof(class_map));

    /* Build class map */
    for (int r0 = 12; r0 >= 0; r0--) {
        for (int r1 = r0; r1 >= 0; r1--) {
            if (r0 == r1) {
                /* Pair */
                class_map[r0][r1][0] = n_classes;
                class_map[r0][r1][1] = n_classes; /* same class for suited/offsuit pair */
                classes_out[n_classes][0] = MAKE_CARD(r0, 0);
                classes_out[n_classes][1] = MAKE_CARD(r1, 1);
                n_classes++;
            } else {
                /* Suited */
                class_map[r0][r1][1] = n_classes;
                classes_out[n_classes][0] = MAKE_CARD(r0, 0);
                classes_out[n_classes][1] = MAKE_CARD(r1, 0);
                n_classes++;
                /* Offsuit */
                class_map[r0][r1][0] = n_classes;
                classes_out[n_classes][0] = MAKE_CARD(r0, 0);
                classes_out[n_classes][1] = MAKE_CARD(r1, 1);
                n_classes++;
            }
        }
    }

    /* Map each hand to its class */
    for (int h = 0; h < num_hands; h++) {
        int r0 = CARD_RANK(hands[h][0]);
        int r1 = CARD_RANK(hands[h][1]);
        int suited = (CARD_SUIT(hands[h][0]) == CARD_SUIT(hands[h][1])) ? 1 : 0;
        /* Ensure r0 >= r1 */
        if (r0 < r1) { int tmp = r0; r0 = r1; r1 = tmp; }
        hand_to_class[h] = class_map[r0][r1][suited];
    }

    return n_classes;
}

/* ── PA-EMD (Potential-Aware Earth Mover's Distance) ─────────────── */

float ca_emd_1d(
    const float *hist_a, const float *hist_b,
    int n_bins
) {
    if (n_bins <= 0 || !hist_a || !hist_b) return 0.0f;

    /*
     * For 1D histograms on a uniform grid, EMD equals the L1 norm of the
     * difference of CDFs:  EMD = sum_{i=0}^{n-1} |CDF_a(i) - CDF_b(i)|
     *
     * Equivalently, maintain a running sum of (hist_a[i] - hist_b[i]).
     * The absolute value of this running sum at each step is the CDF gap.
     * Sum those absolute values.
     */
    double running = 0.0;
    double emd = 0.0;
    for (int i = 0; i < n_bins; i++) {
        running += (double)hist_a[i] - (double)hist_b[i];
        emd += (running >= 0.0) ? running : -running;
    }
    return (float)emd;
}

/*
 * Deterministic seed for EHS distribution: mix board + hand cards into
 * a 64-bit seed via splitmix64-style finalizer. Same inputs → same seed
 * on any platform (no pointer-dependent state).
 */
static uint64_t ca_ehs_dist_seed(const int *board, int n_board,
                                  const int *hand) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;  /* golden ratio constant */
    for (int i = 0; i < n_board; i++) {
        h ^= (uint64_t)(unsigned)board[i];
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 31;
    }
    h ^= (uint64_t)(unsigned)hand[0];
    h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    h ^= (uint64_t)(unsigned)hand[1];
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 31;
    /* Ensure nonzero (xorshift requires it) */
    if (h == 0) h = 1;
    return h;
}

void ca_compute_ehs_distribution(
    const int *board, int n_board,
    const int *hand,
    int num_samples,
    float *out_hist,
    int n_bins
) {
    if (n_bins <= 0 || num_samples <= 0 || !board || !hand || !out_hist) return;
    if (n_board < 3 || n_board > 5) return;

    /* Zero histogram */
    for (int i = 0; i < n_bins; i++) out_hist[i] = 0.0f;

    int c0 = hand[0], c1 = hand[1];

    /* Build blocked set: board + hero hand */
    int blocked[52] = {0};
    for (int i = 0; i < n_board; i++) blocked[board[i]] = 1;
    blocked[c0] = 1;
    blocked[c1] = 1;

    /* Available cards */
    int avail[52];
    int n_avail = 0;
    for (int c = 0; c < 52; c++)
        if (!blocked[c]) avail[n_avail++] = c;

    /* Each sample needs 2 (opponent) + (5 - n_board) (remaining board) cards */
    int runout_cards = 5 - n_board;
    int cards_needed = 2 + runout_cards;
    if (n_avail < cards_needed) return;

    /* Deterministic RNG seeded from board + hand */
    uint64_t rng = ca_ehs_dist_seed(board, n_board, hand);

    int total = 0;

    for (int s = 0; s < num_samples; s++) {
        /* Fisher-Yates partial shuffle to pick cards_needed cards */
        partial_shuffle(avail, n_avail, cards_needed, &rng);

        int oc0 = avail[0], oc1 = avail[1];

        /* Build complete 5-card board */
        int full_board[5];
        for (int i = 0; i < n_board; i++) full_board[i] = board[i];
        for (int i = 0; i < runout_cards; i++)
            full_board[n_board + i] = avail[2 + i];

        /* Evaluate hero vs opponent on final board */
        int hero7[7] = {full_board[0], full_board[1], full_board[2],
                        full_board[3], full_board[4], c0, c1};
        int opp7[7]  = {full_board[0], full_board[1], full_board[2],
                        full_board[3], full_board[4], oc0, oc1};

        uint32_t hero_str = eval7(hero7);
        uint32_t opp_str  = eval7(opp7);

        /* Equity outcome: 1.0 (win), 0.5 (tie), 0.0 (loss) */
        float eq;
        if (hero_str > opp_str) eq = 1.0f;
        else if (hero_str == opp_str) eq = 0.5f;
        else eq = 0.0f;

        /* Map equity to bin index: [0, 1] → [0, n_bins-1] */
        int bin = (int)(eq * (n_bins - 1) + 0.5f);
        if (bin < 0) bin = 0;
        if (bin >= n_bins) bin = n_bins - 1;

        out_hist[bin] += 1.0f;
        total++;
    }

    /* Normalize to probability distribution (sum = 1.0) */
    if (total > 0) {
        float inv = 1.0f / (float)total;
        for (int i = 0; i < n_bins; i++) out_hist[i] *= inv;
    }
}

void ca_compute_ehs_distribution_batch(
    const int *board, int n_board,
    const int hands[][2], int num_hands,
    int num_samples,
    float *out_hists,
    int n_bins
) {
    if (!board || !hands || !out_hists || num_hands <= 0) return;
    for (int h = 0; h < num_hands; h++) {
        ca_compute_ehs_distribution(
            board, n_board,
            hands[h],
            num_samples,
            out_hists + (size_t)h * n_bins,
            n_bins
        );
    }
}

/* ── EMD k-means ──────────────────────────────────────────────── */

/*
 * Compute mean EHS from a histogram (for percentile seeding).
 * EHS ≈ sum_i (bin_center_i * hist[i]) where bin_center_i = i/(n_bins-1).
 */
static float hist_mean_ehs(const float *hist, int n_bins) {
    if (n_bins <= 1) return 0.5f;
    double sum = 0.0;
    double inv = 1.0 / (double)(n_bins - 1);
    for (int i = 0; i < n_bins; i++) {
        sum += (double)i * inv * (double)hist[i];
    }
    return (float)sum;
}

int ca_assign_buckets_kmeans_emd(
    const float *hists, int num_hands, int n_bins,
    int num_buckets, int n_iters,
    int *bucket_out
) {
    if (num_hands <= 0 || n_bins <= 0 || num_buckets <= 0) return 0;
    int k = (num_buckets < num_hands) ? num_buckets : num_hands;
    if (k <= 1) {
        for (int i = 0; i < num_hands; i++) bucket_out[i] = 0;
        return 1;
    }

    /*
     * Step 1: Percentile seed centroids on mean EHS (same strategy as
     * ca_assign_buckets_kmeans, but seeding from histogram means).
     */
    EhsPair *sorted = (EhsPair*)malloc(num_hands * sizeof(EhsPair));
    for (int i = 0; i < num_hands; i++) {
        sorted[i].idx = i;
        sorted[i].ehs = hist_mean_ehs(hists + (size_t)i * n_bins, n_bins);
    }
    qsort(sorted, num_hands, sizeof(EhsPair), ca_cmp_ehs);

    /* Centroids are histograms: [k × n_bins] */
    float *centroids = (float*)malloc((size_t)k * n_bins * sizeof(float));
    for (int c = 0; c < k; c++) {
        int si = (int)((float)c / k * num_hands);
        if (si >= num_hands) si = num_hands - 1;
        int fi = sorted[si].idx;
        memcpy(centroids + (size_t)c * n_bins,
               hists + (size_t)fi * n_bins,
               n_bins * sizeof(float));
    }
    free(sorted);

    /*
     * Step 2: Lloyd's k-means with EMD distance.
     *
     * Assignment: each hand goes to the centroid with smallest EMD.
     * Update: centroid = element-wise average of member histograms.
     * (For 1D histograms on a fixed grid, the element-wise mean IS
     * the Wasserstein-2 barycenter — Cuturi & Doucet 2014.)
     */
    int *counts = (int*)calloc(k, sizeof(int));
    float *sums = (float*)calloc((size_t)k * n_bins, sizeof(float));

    for (int iter = 0; iter < n_iters; iter++) {
        memset(counts, 0, k * sizeof(int));
        memset(sums, 0, (size_t)k * n_bins * sizeof(float));

        /* Assign each hand to nearest centroid by EMD */
        for (int h = 0; h < num_hands; h++) {
            const float *hh = hists + (size_t)h * n_bins;
            float best_d = 1e30f;
            int best_c = 0;
            for (int c = 0; c < k; c++) {
                float d = ca_emd_1d(hh, centroids + (size_t)c * n_bins, n_bins);
                if (d < best_d) { best_d = d; best_c = c; }
            }
            bucket_out[h] = best_c;
            counts[best_c]++;
            float *sc = sums + (size_t)best_c * n_bins;
            for (int b = 0; b < n_bins; b++) sc[b] += hh[b];
        }

        /* Update centroids: element-wise mean */
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                float *cc = centroids + (size_t)c * n_bins;
                float *sc = sums + (size_t)c * n_bins;
                float inv = 1.0f / (float)counts[c];
                for (int b = 0; b < n_bins; b++) cc[b] = sc[b] * inv;
            }
        }
    }

    int actual = 0;
    for (int c = 0; c < k; c++)
        if (counts[c] > 0) actual++;

    free(centroids);
    free(counts);
    free(sums);
    return actual;
}

int ca_assign_buckets_pa_emd(
    const int *board, int n_board,
    const int hands[][2], int num_hands,
    int num_buckets, int n_samples, int n_bins,
    int *bucket_out
) {
    if (num_hands <= 0 || num_buckets <= 0 || n_bins <= 0) return 0;

    /* Step 1: compute distributions for all hands */
    float *hists = (float*)malloc((size_t)num_hands * n_bins * sizeof(float));
    if (!hists) return 0;

    ca_compute_ehs_distribution_batch(board, n_board, hands, num_hands,
                                       n_samples, hists, n_bins);

    /* Step 2: EMD k-means */
    int actual = ca_assign_buckets_kmeans_emd(hists, num_hands, n_bins,
                                               num_buckets, 20, bucket_out);

    free(hists);
    return actual;
}

int ca_nearest_centroid_emd(
    const float *hist,
    const float *centroids, int num_centroids,
    int n_bins
) {
    if (!hist || !centroids || num_centroids <= 0 || n_bins <= 0) return 0;
    float best_d = 1e30f;
    int best = 0;
    for (int c = 0; c < num_centroids; c++) {
        float d = ca_emd_1d(hist, centroids + (size_t)c * n_bins, n_bins);
        if (d < best_d) { best_d = d; best = c; }
    }
    return best;
}
