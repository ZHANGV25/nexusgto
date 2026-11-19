/*
 * rollout_bucket_memo.c - per-solve turn/river bucket-id memoization.
 * See rollout_bucket_memo.h for the contract and the race-safety argument
 * (unsynchronized same-value int16 writes under Hogwild are idempotent).
 */
#include "rollout_bucket_memo.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef POKER_ROLLOUT_BUCKET_MEMO_DISABLE

/* 1326 unordered hole pairs x 52 turn cards; x 52 river cards. Bucket ids
 * are < 32767 (blueprint bucket counts are far smaller), so int16 with a
 * 0xFF-fill sentinel (-1) is safe. Tables are process-global and
 * intentionally never freed. */
#define RBM_PAIRS 1326
#define RBM_TURN_SLOTS  ((size_t)RBM_PAIRS * 52)
#define RBM_RIVER_SLOTS ((size_t)RBM_PAIRS * 52 * 52)

static int16_t *g_turn = NULL;
static int16_t *g_river = NULL;
static uint64_t g_fp = 0;

static uint64_t rbm_fingerprint(const void *reader, const int *flop3) {
    int a = flop3[0], b = flop3[1], c = flop3[2], t;
    if (a > b) { t = a; a = b; b = t; }
    if (b > c) { t = b; b = c; c = t; }
    if (a > b) { t = a; a = b; b = t; }
    return ((uint64_t)(uintptr_t)reader)
           ^ ((uint64_t)(unsigned)a
              | ((uint64_t)(unsigned)b << 8)
              | ((uint64_t)(unsigned)c << 16)
              | (1ULL << 63));  /* never collides with the g_fp=0 unset state */
}

static inline size_t rbm_pair_index(int c0, int c1) {
    int p = c0 < c1 ? c0 : c1;
    int q = c0 < c1 ? c1 : c0;
    return (size_t)q * (size_t)(q - 1) / 2 + (size_t)p;
}

/* Single-threaded setup only (oracle constructors, pre-Hogwild): memset the
 * tables FIRST, then publish the fingerprint, so a stale fingerprint can
 * never validate entries from a previous flop.
 *
 * WARM PATH: when the fingerprint is unchanged (same reader, same flop) the
 * tables are kept as-is. Cached entries are pure functions of the cards for
 * this reader, so they stay exact across solves; consecutive decisions in
 * the same hand share the flop and start pre-warmed. This makes reader
 * POINTER identity load-bearing across solves: a freed reader reallocated
 * at the same address would falsely validate - acceptable because every
 * process loads its blueprint reader once and keeps it for its lifetime. */
void rollout_bucket_memo_reset(const void *reader, const int flop[3]) {
    if (!reader || !flop) return;
    if (flop[0] < 0 || flop[0] > 51 || flop[1] < 0 || flop[1] > 51
        || flop[2] < 0 || flop[2] > 51)
        return;
    uint64_t fp = rbm_fingerprint(reader, flop);
    if (fp == g_fp && g_turn && g_river) return;  /* warm: keep the cache */
    if (!g_turn) g_turn = (int16_t *)malloc(RBM_TURN_SLOTS * sizeof(int16_t));
    if (!g_river) g_river = (int16_t *)malloc(RBM_RIVER_SLOTS * sizeof(int16_t));
    if (!g_turn || !g_river) {
        /* Allocation failure: leave g_fp unset so every lookup bypasses. */
        free(g_turn); free(g_river);
        g_turn = NULL; g_river = NULL;
        g_fp = 0;
        return;
    }
    g_fp = 0;
    memset(g_turn, 0xFF, RBM_TURN_SLOTS * sizeof(int16_t));
    memset(g_river, 0xFF, RBM_RIVER_SLOTS * sizeof(int16_t));
    g_fp = fp;
}

static inline int rbm_ready(const void *reader, const int *board) {
    if (!g_turn || !g_river || g_fp == 0) return 0;
    return rbm_fingerprint(reader, board) == g_fp;
}

int rollout_bucket_memo_lookup_turn(const void *reader, const int *board,
                                    int c0, int c1) {
    if (!rbm_ready(reader, board)) return ROLLOUT_BUCKET_MEMO_BYPASS;
    int16_t v = g_turn[rbm_pair_index(c0, c1) * 52 + (size_t)board[3]];
    return v < 0 ? ROLLOUT_BUCKET_MEMO_MISS : (int)v;
}

int rollout_bucket_memo_lookup_river(const void *reader, const int *board,
                                     int c0, int c1) {
    if (!rbm_ready(reader, board)) return ROLLOUT_BUCKET_MEMO_BYPASS;
    size_t key = (rbm_pair_index(c0, c1) * 52 + (size_t)board[3]) * 52
                 + (size_t)board[4];
    int16_t v = g_river[key];
    return v < 0 ? ROLLOUT_BUCKET_MEMO_MISS : (int)v;
}

void rollout_bucket_memo_store_turn(const void *reader, const int *board,
                                    int c0, int c1, int bucket) {
    if (bucket < 0 || bucket > 32766) return;
    if (!rbm_ready(reader, board)) return;
    g_turn[rbm_pair_index(c0, c1) * 52 + (size_t)board[3]] = (int16_t)bucket;
}

void rollout_bucket_memo_store_river(const void *reader, const int *board,
                                     int c0, int c1, int bucket) {
    if (bucket < 0 || bucket > 32766) return;
    if (!rbm_ready(reader, board)) return;
    size_t key = (rbm_pair_index(c0, c1) * 52 + (size_t)board[3]) * 52
                 + (size_t)board[4];
    g_river[key] = (int16_t)bucket;
}

#else  /* POKER_ROLLOUT_BUCKET_MEMO_DISABLE */

void rollout_bucket_memo_reset(const void *reader, const int flop[3]) {
    (void)reader; (void)flop;
}
int rollout_bucket_memo_lookup_turn(const void *reader, const int *board,
                                    int c0, int c1) {
    (void)reader; (void)board; (void)c0; (void)c1;
    return ROLLOUT_BUCKET_MEMO_BYPASS;
}
int rollout_bucket_memo_lookup_river(const void *reader, const int *board,
                                     int c0, int c1) {
    (void)reader; (void)board; (void)c0; (void)c1;
    return ROLLOUT_BUCKET_MEMO_BYPASS;
}
void rollout_bucket_memo_store_turn(const void *reader, const int *board,
                                    int c0, int c1, int bucket) {
    (void)reader; (void)board; (void)c0; (void)c1; (void)bucket;
}
void rollout_bucket_memo_store_river(const void *reader, const int *board,
                                     int c0, int c1, int bucket) {
    (void)reader; (void)board; (void)c0; (void)c1; (void)bucket;
}

#endif /* POKER_ROLLOUT_BUCKET_MEMO_DISABLE */
