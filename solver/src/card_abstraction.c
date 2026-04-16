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

/* Comparison function for sorting by EHS */
typedef struct { int idx; float ehs; } EhsPair;

static int cmp_ehs(const void *a, const void *b) {
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
    qsort(pairs, num_hands, sizeof(EhsPair), cmp_ehs);

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
             * For eval7, we need 7 cards. With < 5 board cards, pad the
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
    qsort(sorted, num_hands, sizeof(EhsPair), cmp_ehs);

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
    qsort(sorted, n, sizeof(EhsPair), cmp_ehs);

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
    qsort(sorted, num_hands, sizeof(EhsPair), cmp_ehs);

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
