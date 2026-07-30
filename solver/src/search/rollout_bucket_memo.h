/*
 * rollout_bucket_memo.h - per-solve turn/river bucket-id memoization.
 *
 * bps_hand_to_bucket_turn/_river are pure functions of (hole pair, board).
 * Within one solve the flop is fixed, so distinct keys are tiny versus query
 * volume (profile 2026-07-28: ~99% of solve cycles were these recomputations).
 * The memo caches the bucket id keyed by (unordered hole pair, turn[, river])
 * for the current (reader, flop) fingerprint.
 *
 * Contract: rollout_bucket_memo_reset() must be called ONLY from
 * single-threaded setup code (the oracle constructors) before any Hogwild
 * workers run. Lookups/stores are unsynchronized by design: concurrent
 * writers can only ever write the SAME value for a key (the functions are
 * deterministic), so races are idempotent/benign - the same discipline as
 * the bias_regret table in es_sampled_oracle.c.
 *
 * Warm reuse: reset() with an UNCHANGED (reader, flop) fingerprint keeps
 * the tables instead of wiping - entries are pure functions of the cards,
 * so consecutive same-flop solves in one process start pre-warmed and
 * stay exact.
 *
 * Compile out with -DPOKER_ROLLOUT_BUCKET_MEMO_DISABLE: lookups then return
 * the bypass code and rollout.c falls through to plain computation.
 */
#ifndef ROLLOUT_BUCKET_MEMO_H
#define ROLLOUT_BUCKET_MEMO_H

/* Lookup results: >= 0 cached bucket id; -1 miss (compute, then store);
 * -2 bypass (memo unavailable or fingerprint mismatch - compute, no store). */
#define ROLLOUT_BUCKET_MEMO_MISS   (-1)
#define ROLLOUT_BUCKET_MEMO_BYPASS (-2)

void rollout_bucket_memo_reset(const void *reader, const int flop[3]);

int rollout_bucket_memo_lookup_turn(const void *reader, const int *board,
                                    int c0, int c1);
int rollout_bucket_memo_lookup_river(const void *reader, const int *board,
                                     int c0, int c1);

void rollout_bucket_memo_store_turn(const void *reader, const int *board,
                                    int c0, int c1, int bucket);
void rollout_bucket_memo_store_river(const void *reader, const int *board,
                                     int c0, int c1, int bucket);

#endif /* ROLLOUT_BUCKET_MEMO_H */
