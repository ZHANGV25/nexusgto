/**
 * rollout_sampled.c — sampled-cont-strat subgame-leaf evaluation (OP4 / Option c).
 *
 * Split out of rollout.c (no behavior change). Holds the deterministic
 * per-sample bias derivation, the sampled-core worker (rollout_sampled_core),
 * and the two public entry points (rollout_compute_subgame_leaf_values_sampled_c
 * and ..._sampled_perleaf_c).
 *
 * Compiled into the SAME shared libraries as rollout.c. The parent helpers it
 * calls (rollout_simulate_one, rollout_set_error) are shared via
 * rollout_internal.h (promoted from file-static, defs stay in rollout.c); the
 * two seed mixers are header-inline. Everything else is in rollout.h.
 */

#include "rollout.h"
#include "rollout_internal.h"

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* ── Sampled-cont-strat variant (OP4 / Option c) ─────────────────────── */

/* Derive a sample's bias_profile deterministically from
 * (master_seed, leaf_idx, sample_idx). Uses splitmix64, NOT the host's
 * numpy RNG — so Python and C will disagree at the sample level but
 * agree in expectation as num_samples_k → ∞ (both draw IID uniform).
 * Pure-math layers (apply_bias, classify, action_amount, showdown)
 * remain bit-exact parity with Python. */
static inline void rollout_sampled_draw_bias(
    uint64_t master_seed, int32_t leaf_idx, int32_t sample_idx,
    int32_t num_players, int32_t *bias_out
) {
    /* Seed derivation mirrors rollout_derive_seed, with sample_idx in
     * the hero_hand slot and a distinct sentinel in the hero_p slot to
     * avoid aliasing with rollout seeds. */
    uint64_t s = master_seed ^ 0xc3c3c3c3c3c3c3c3ULL;
    uint64_t t;
    t = s ^ (uint64_t)(uint32_t)leaf_idx;    s ^= rollout_splitmix64_next(&t);
    t = s ^ (uint64_t)(uint32_t)sample_idx;  s ^= rollout_splitmix64_next(&t);
    t = s ^ 0xa5a5u;                          s ^= rollout_splitmix64_next(&t);
    if (s == 0) s = 0xfeedfacedeadbeefULL;

    for (int32_t p = 0; p < num_players; p++) {
        bias_out[p] = (int32_t)(rollout_splitmix64_next(&s)
                                 % (uint64_t)ROLLOUT_NUM_BIAS_PROFILES);
    }
}

/* Shared core for the sampled cont-strat-leaf construct.
 *
 * per_leaf_player_hands: optional. When NULL, EVERY leaf samples opponents from
 * the single shared `player_hands` (the historical behavior — the public
 * rollout_compute_subgame_leaf_values_sampled_c wrapper passes NULL, so it is
 * byte-for-byte unchanged). When non-NULL it is an array of n_leaves*num_players
 * RolloutHandList: leaf li_idx uses &per_leaf_player_hands[li_idx*num_players] as
 * its per-player range, so the VILLAIN's sampled-hand weights can be conditioned
 * on the flop action sequence that reached that leaf (the 2p-flop range
 * conditioning, 2026-06-17 — see python/leaf_villain_conditioning.py). HERO rows
 * keep the full static range, so hero is still enumerated over all hands.
 * leaf_infos[li_idx].leaf_idx still drives the output slot; the per-leaf list is
 * indexed by the LOOP index li_idx (1:1 with leaf_infos). */
static int32_t rollout_sampled_core(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    const RolloutLeafInfo *leaf_infos, int32_t n_leaves,
    const int32_t *active, int32_t num_players,
    const RolloutHandList *player_hands,
    const RolloutHandList *per_leaf_player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    int32_t stack_chips,
    int32_t max_hands,
    int32_t num_rollouts,
    int32_t num_samples_k,
    uint64_t master_seed,
    float *out, int32_t out_size
) {
    if (num_samples_k < 1) {
        rollout_set_error(street, num_players,
            "num_samples_k must be >= 1, got %d", num_samples_k);
        return -1;
    }

    /* total = max_leaf_idx + 1 (no cont_per_leaf factor in SAMPLED mode). */
    int32_t max_leaf_idx = 0;
    for (int32_t i = 0; i < n_leaves; i++) {
        if (leaf_infos[i].leaf_idx > max_leaf_idx) {
            max_leaf_idx = leaf_infos[i].leaf_idx;
        }
    }
    int64_t total = (int64_t)(max_leaf_idx + 1);
    int64_t floats_needed = total * (int64_t)num_players * (int64_t)max_hands;

    if (floats_needed > (int64_t)out_size) {
        rollout_set_error(street, num_players,
            "out buffer too small: need %lld floats, got %d",
            (long long)floats_needed, out_size);
        return -1;
    }

    memset(out, 0, (size_t)floats_needed * sizeof(float));

    /* Per-(leaf × sample) parallelism. Each (leaf, sample) pair runs
     * rollout_compute_leaf_values_all_hands_c into a thread-local
     * scratch buffer; results accumulate into out[leaf_idx * P * H]
     * under an atomic per-slot update. OMP reduction across samples
     * via atomicAdd-style float accumulation, finalized to mean after
     * the parallel region.
     *
     * Accumulator layout: running sum in double precision in an aux
     * buffer allocated up front, plus a per-leaf visit counter to
     * drive the final division. Max hands at 6p × 1176 × 8 bytes × up
     * to ~4K leaves ≈ 220 MB worst case; for realistic flop workloads
     * (1 leaf × 1176 hands × 6 players × 8 bytes) this is ~56 KB.
     * Over-allocate conservatively based on total. */
    double *acc = (double*)calloc((size_t)floats_needed, sizeof(double));
    if (!acc) {
        rollout_set_error(street, num_players,
            "sampled: alloc %lld doubles failed", (long long)floats_needed);
        return -1;
    }

    volatile int g_fail = 0;
    char cap_reason[512] = {0};
    int32_t cap_street = street;
    int32_t cap_num_players = num_players;

    #pragma omp parallel for collapse(2) schedule(dynamic, 1) default(none) \
        shared(acc, leaf_infos, n_leaves, num_samples_k, stack_chips, \
                num_players, max_hands, num_rollouts, master_seed, \
                r, board, n_board_cards, street, active, player_hands, \
                per_leaf_player_hands, \
                acting_order, acting_order_len, player_seats, \
                game_action_history, gah_len, g_fail, \
                cap_reason, cap_street, cap_num_players)
    for (int32_t li_idx = 0; li_idx < n_leaves; li_idx++) {
        for (int32_t sample_idx = 0; sample_idx < num_samples_k; sample_idx++) {
            if (g_fail) continue;

            const RolloutLeafInfo *li = &leaf_infos[li_idx];
            int32_t stacks[ROLLOUT_MAX_PLAYERS];
            for (int32_t p = 0; p < num_players; p++) {
                int32_t s = stack_chips - li->bets[p];
                stacks[p] = s < 0 ? 0 : s;
            }

            int32_t bias_profile[ROLLOUT_MAX_PLAYERS];
            rollout_sampled_draw_bias(master_seed, li->leaf_idx,
                                      sample_idx, num_players, bias_profile);

            /* Scratch buffer for this (leaf, sample) output. */
            float scratch[ROLLOUT_MAX_PLAYERS * ROLLOUT_MAX_HANDS];
            if ((int64_t)num_players * (int64_t)max_hands >
                (int64_t)(ROLLOUT_MAX_PLAYERS * ROLLOUT_MAX_HANDS)) {
                #pragma omp critical(rollout_sampled_fail)
                {
                    if (!g_fail) {
                        snprintf(cap_reason, sizeof(cap_reason) - 1,
                                 "sampled: scratch too small for %d×%d",
                                 num_players, max_hands);
                        cap_street = street;
                        cap_num_players = num_players;
                        g_fail = 1;
                    }
                }
                continue;
            }
            /* Per-leaf game history: [passed_in_preflop, leaf->flop_history].
             * Same rationale as the non-sampled path above — without the
             * leaf-specific flop slice the rollout's blueprint queries
             * during turn/river miss 100%. SAMPLED is the actually-called
             * path from postflop_provider.py (K=1 default), so this
             * branch is the load-bearing one for live solves. */
            int32_t per_leaf_history[ROLLOUT_MAX_ACTION_HISTORY];
            int32_t fh_len = li->flop_history_len;
            if (fh_len < 0) fh_len = 0;
            if ((int64_t)gah_len + (int64_t)fh_len > ROLLOUT_MAX_ACTION_HISTORY) {
                #pragma omp critical(rollout_sampled_fail)
                {
                    if (!g_fail) {
                        snprintf(cap_reason, sizeof(cap_reason) - 1,
                                 "sampled: per-leaf history overflow: preflop=%d + flop=%d > %d",
                                 gah_len, fh_len, ROLLOUT_MAX_ACTION_HISTORY);
                        cap_street = street;
                        cap_num_players = num_players;
                        g_fail = 1;
                    }
                }
                continue;
            }
            for (int32_t k = 0; k < gah_len; k++) per_leaf_history[k] = game_action_history[k];
            for (int32_t k = 0; k < fh_len; k++) per_leaf_history[gah_len + k] = li->flop_history[k];
            int32_t per_leaf_history_len = gah_len + fh_len;

            /* Skip the leaf's first betting round when the leaf already
             * carries a closed flop_history — see rollout_simulate_one. */
            int32_t skip_first = (fh_len > 0) ? 1 : 0;
            /* Per-leaf opponent range (2026-06-17): when conditioning is active,
             * leaf li_idx samples opponents from its OWN conditioned ranges
             * (villain weights filtered by the flop line); otherwise the shared
             * static range (the historical path). */
            const RolloutHandList *leaf_hands = per_leaf_player_hands
                ? &per_leaf_player_hands[(int64_t)li_idx * num_players]
                : player_hands;
            /* Derive a per-sample seed for the rollout's RNG chain.
             * leaf_idx slot reuses li->leaf_idx; combo slot carries
             * sample_idx so each sample's rollout stream is distinct. */
            int rc = rollout_compute_leaf_values_all_hands_c(
                r, board, n_board_cards, street, li->pot_chips,
                active, num_players, stacks, li->bets,
                leaf_hands, acting_order, acting_order_len,
                player_seats, per_leaf_history, per_leaf_history_len,
                bias_profile, max_hands, num_rollouts, master_seed,
                li->leaf_idx, sample_idx, skip_first, scratch);
            if (rc != 0) {
                #pragma omp critical(rollout_sampled_fail)
                {
                    if (!g_fail) {
                        const char *reason = NULL;
                        int32_t s_st = 0, s_np = 0;
                        rollout_get_last_error(&reason, &s_st, &s_np);
                        if (reason) {
                            strncpy(cap_reason, reason,
                                    sizeof(cap_reason) - 1);
                            cap_reason[sizeof(cap_reason) - 1] = '\0';
                            cap_street = s_st;
                            cap_num_players = s_np;
                        }
                        g_fail = 1;
                    }
                }
                continue;
            }

            /* Accumulate into acc[leaf_idx * P * H + p * H + h].
             * Multiple samples may write the same leaf concurrently
             * (different sample_idx on same li->leaf_idx), but only if
             * two different li_idx values resolve to the same
             * leaf_idx — which shouldn't happen for valid input.
             * Still, guard with atomics in case of duplicates. */
            int64_t off = (int64_t)li->leaf_idx
                          * (int64_t)num_players * (int64_t)max_hands;
            int64_t n_floats = (int64_t)num_players * (int64_t)max_hands;
            for (int64_t k = 0; k < n_floats; k++) {
                #pragma omp atomic update
                acc[off + k] += (double)scratch[k];
            }
        }
    }

    if (g_fail) {
        free(acc);
        rollout_set_error(cap_street, cap_num_players, "%s",
                          cap_reason[0] ? cap_reason
                                        : "sampled rollout worker failed");
        return -1;
    }

    /* Finalize: divide by K, cast back to float. */
    const double inv_k = 1.0 / (double)num_samples_k;
    for (int32_t li_idx = 0; li_idx < n_leaves; li_idx++) {
        const RolloutLeafInfo *li = &leaf_infos[li_idx];
        int64_t off = (int64_t)li->leaf_idx
                      * (int64_t)num_players * (int64_t)max_hands;
        int64_t n_floats = (int64_t)num_players * (int64_t)max_hands;
        for (int64_t k = 0; k < n_floats; k++) {
            out[off + k] = (float)(acc[off + k] * inv_k);
        }
    }

    free(acc);
    return 0;
}

/* Public sampled construct (historical ABI, unchanged). Delegates to the shared
 * core with NO per-leaf ranges → every leaf uses the single shared player_hands,
 * byte-for-byte identical to the pre-refactor behavior. This is the symbol the
 * 3p+ flop and 5-6p external-sampling paths bind and call. */
BPS_EXPORT int32_t rollout_compute_subgame_leaf_values_sampled_c(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    const RolloutLeafInfo *leaf_infos, int32_t n_leaves,
    const int32_t *active, int32_t num_players,
    const RolloutHandList *player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    int32_t stack_chips,
    int32_t max_hands,
    int32_t num_rollouts,
    int32_t num_samples_k,
    uint64_t master_seed,
    float *out, int32_t out_size
) {
    return rollout_sampled_core(
        r, board, n_board_cards, street, leaf_infos, n_leaves,
        active, num_players, player_hands, /*per_leaf=*/NULL,
        acting_order, acting_order_len, player_seats,
        game_action_history, gah_len, stack_chips, max_hands,
        num_rollouts, num_samples_k, master_seed, out, out_size);
}

/* Per-leaf sampled construct (2026-06-17). Identical to the above except each
 * leaf samples opponents from its OWN conditioned ranges:
 * per_leaf_player_hands is an array of n_leaves*num_players RolloutHandList,
 * indexed [li_idx*num_players + player]. Used ONLY by the 2p-flop opponent-range
 * conditioning (gated n_players==2 && street==flop in the engine). player_hands
 * is still passed for the shared-fallback contract but is ignored when
 * per_leaf_player_hands is non-NULL. */
BPS_EXPORT int32_t rollout_compute_subgame_leaf_values_sampled_perleaf_c(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    const RolloutLeafInfo *leaf_infos, int32_t n_leaves,
    const int32_t *active, int32_t num_players,
    const RolloutHandList *player_hands,
    const RolloutHandList *per_leaf_player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    int32_t stack_chips,
    int32_t max_hands,
    int32_t num_rollouts,
    int32_t num_samples_k,
    uint64_t master_seed,
    float *out, int32_t out_size
) {
    if (!per_leaf_player_hands) {
        rollout_set_error(street, num_players,
            "perleaf construct called with NULL per_leaf_player_hands");
        return -1;
    }
    return rollout_sampled_core(
        r, board, n_board_cards, street, leaf_infos, n_leaves,
        active, num_players, player_hands, per_leaf_player_hands,
        acting_order, acting_order_len, player_seats,
        game_action_history, gah_len, stack_chips, max_hands,
        num_rollouts, num_samples_k, master_seed, out, out_size);
}

/* ── Stratified four-bias marginal variant ────────────────────────────── */

/* Draw a uniform permutation of {0,1,2,3} from a stream keyed by
 * (master_seed, leaf_idx, round, player). The seed derivation follows
 * rollout_sampled_draw_bias's splitmix64 mixing discipline, with a distinct
 * sentinel so B4 permutations cannot alias the historical sampled draws. */
static inline void rollout_bias4_draw_permutation(
    uint64_t master_seed, int32_t leaf_idx, int32_t round, int32_t player,
    int32_t permutation[ROLLOUT_NUM_BIAS_PROFILES]
) {
    uint64_t s = master_seed ^ 0x4b4b4b4b4b4b4b4bULL;
    uint64_t t;
    t = s ^ (uint64_t)(uint32_t)leaf_idx; s ^= rollout_splitmix64_next(&t);
    t = s ^ (uint64_t)(uint32_t)round;    s ^= rollout_splitmix64_next(&t);
    t = s ^ (uint64_t)(uint32_t)player;   s ^= rollout_splitmix64_next(&t);
    if (s == 0) s = 0xdecafbad12345678ULL;

    for (int32_t b = 0; b < ROLLOUT_NUM_BIAS_PROFILES; b++) {
        permutation[b] = b;
    }
    for (int32_t b = ROLLOUT_NUM_BIAS_PROFILES - 1; b > 0; b--) {
        int32_t swap_idx = (int32_t)(
            rollout_splitmix64_next(&s) % (uint64_t)(b + 1));
        int32_t tmp = permutation[b];
        permutation[b] = permutation[swap_idx];
        permutation[swap_idx] = tmp;
    }
}

static int32_t rollout_bias4_core(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    const RolloutLeafInfo *leaf_infos, int32_t n_leaves,
    const int32_t *active, int32_t num_players,
    const RolloutHandList *player_hands,
    const RolloutHandList *per_leaf_player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    int32_t stack_chips,
    int32_t max_hands,
    int32_t num_rollouts,
    int32_t num_rounds,
    uint64_t master_seed,
    float *out, int32_t out_size
) {
    if (num_rounds < 1) {
        rollout_set_error(street, num_players,
            "num_rounds must be >= 1, got %d", num_rounds);
        return -1;
    }
    if (num_rounds > (INT32_MAX - 3) / ROLLOUT_NUM_BIAS_PROFILES) {
        rollout_set_error(street, num_players,
            "num_rounds too large for rollout seed slot: %d", num_rounds);
        return -1;
    }

    int32_t max_leaf_idx = 0;
    for (int32_t i = 0; i < n_leaves; i++) {
        if (leaf_infos[i].leaf_idx > max_leaf_idx) {
            max_leaf_idx = leaf_infos[i].leaf_idx;
        }
    }
    int64_t total = (int64_t)(max_leaf_idx + 1)
                    * ROLLOUT_NUM_BIAS_PROFILES;
    int64_t floats_needed = total * (int64_t)num_players
                            * (int64_t)max_hands;

    if (floats_needed > (int64_t)out_size) {
        rollout_set_error(street, num_players,
            "out buffer too small: need %lld floats, got %d",
            (long long)floats_needed, out_size);
        return -1;
    }

    memset(out, 0, (size_t)floats_needed * sizeof(float));

    /* Double accumulation matches rollout_sampled_core. Different rounds can
     * target the same [leaf][bias][player][hand] slot concurrently, so updates
     * remain atomic even though the four batches within one round are
     * disjoint for each player. */
    double *acc = (double *)calloc((size_t)floats_needed, sizeof(double));
    if (!acc) {
        rollout_set_error(street, num_players,
            "bias4: alloc %lld doubles failed", (long long)floats_needed);
        return -1;
    }

    volatile int g_fail = 0;
    char cap_reason[512] = {0};
    int32_t cap_street = street;
    int32_t cap_num_players = num_players;

    #pragma omp parallel for collapse(3) schedule(dynamic, 1) default(none) \
        shared(acc, leaf_infos, n_leaves, num_rounds, stack_chips, \
                num_players, max_hands, num_rollouts, master_seed, \
                r, board, n_board_cards, street, active, player_hands, \
                per_leaf_player_hands, \
                acting_order, acting_order_len, player_seats, \
                game_action_history, gah_len, g_fail, \
                cap_reason, cap_street, cap_num_players)
    for (int32_t li_idx = 0; li_idx < n_leaves; li_idx++) {
        for (int32_t round = 0; round < num_rounds; round++) {
            for (int32_t batch = 0;
                 batch < ROLLOUT_NUM_BIAS_PROFILES; batch++) {
                if (g_fail) continue;

                const RolloutLeafInfo *li = &leaf_infos[li_idx];
                int32_t stacks[ROLLOUT_MAX_PLAYERS];
                for (int32_t p = 0; p < num_players; p++) {
                    int32_t s = stack_chips - li->bets[p];
                    stacks[p] = s < 0 ? 0 : s;
                }

                int32_t permutations[ROLLOUT_MAX_PLAYERS]
                                    [ROLLOUT_NUM_BIAS_PROFILES];
                int32_t bias_profile[ROLLOUT_MAX_PLAYERS];
                for (int32_t p = 0; p < num_players; p++) {
                    rollout_bias4_draw_permutation(
                        master_seed, li->leaf_idx, round, p,
                        permutations[p]);
                    bias_profile[p] = permutations[p][batch];
                }

                float scratch[ROLLOUT_MAX_PLAYERS * ROLLOUT_MAX_HANDS];
                if ((int64_t)num_players * (int64_t)max_hands >
                    (int64_t)(ROLLOUT_MAX_PLAYERS * ROLLOUT_MAX_HANDS)) {
                    #pragma omp critical(rollout_bias4_fail)
                    {
                        if (!g_fail) {
                            snprintf(cap_reason, sizeof(cap_reason) - 1,
                                     "bias4: scratch too small for %dx%d",
                                     num_players, max_hands);
                            cap_street = street;
                            cap_num_players = num_players;
                            g_fail = 1;
                        }
                    }
                    continue;
                }

                int32_t per_leaf_history[ROLLOUT_MAX_ACTION_HISTORY];
                int32_t fh_len = li->flop_history_len;
                if (fh_len < 0) fh_len = 0;
                if ((int64_t)gah_len + (int64_t)fh_len >
                    ROLLOUT_MAX_ACTION_HISTORY) {
                    #pragma omp critical(rollout_bias4_fail)
                    {
                        if (!g_fail) {
                            snprintf(cap_reason, sizeof(cap_reason) - 1,
                                     "bias4: per-leaf history overflow: "
                                     "preflop=%d + flop=%d > %d",
                                     gah_len, fh_len,
                                     ROLLOUT_MAX_ACTION_HISTORY);
                            cap_street = street;
                            cap_num_players = num_players;
                            g_fail = 1;
                        }
                    }
                    continue;
                }
                for (int32_t k = 0; k < gah_len; k++) {
                    per_leaf_history[k] = game_action_history[k];
                }
                for (int32_t k = 0; k < fh_len; k++) {
                    per_leaf_history[gah_len + k] = li->flop_history[k];
                }
                int32_t per_leaf_history_len = gah_len + fh_len;
                int32_t skip_first = fh_len > 0 ? 1 : 0;

                const RolloutHandList *leaf_hands = per_leaf_player_hands
                    ? &per_leaf_player_hands[
                        (int64_t)li_idx * num_players]
                    : player_hands;
                int32_t sample_idx =
                    round * ROLLOUT_NUM_BIAS_PROFILES + batch;
                int rc = rollout_compute_leaf_values_all_hands_c(
                    r, board, n_board_cards, street, li->pot_chips,
                    active, num_players, stacks, li->bets,
                    leaf_hands, acting_order, acting_order_len,
                    player_seats, per_leaf_history, per_leaf_history_len,
                    bias_profile, max_hands, num_rollouts, master_seed,
                    li->leaf_idx, sample_idx, skip_first, scratch);
                if (rc != 0) {
                    #pragma omp critical(rollout_bias4_fail)
                    {
                        if (!g_fail) {
                            const char *reason = NULL;
                            int32_t s_st = 0, s_np = 0;
                            rollout_get_last_error(&reason, &s_st, &s_np);
                            if (reason) {
                                strncpy(cap_reason, reason,
                                        sizeof(cap_reason) - 1);
                                cap_reason[sizeof(cap_reason) - 1] = '\0';
                                cap_street = s_st;
                                cap_num_players = s_np;
                            }
                            g_fail = 1;
                        }
                    }
                    continue;
                }

                for (int32_t p = 0; p < num_players; p++) {
                    int32_t bias = bias_profile[p];
                    int64_t out_off =
                        (((int64_t)li->leaf_idx
                           * ROLLOUT_NUM_BIAS_PROFILES + bias)
                          * num_players + p) * max_hands;
                    int64_t scratch_off = (int64_t)p * max_hands;
                    for (int32_t h = 0; h < max_hands; h++) {
                        #pragma omp atomic update
                        acc[out_off + h] +=
                            (double)scratch[scratch_off + h];
                    }
                }
            }
        }
    }

    if (g_fail) {
        free(acc);
        rollout_set_error(cap_street, cap_num_players, "%s",
                          cap_reason[0] ? cap_reason
                                        : "bias4 rollout worker failed");
        return -1;
    }

    const double inv_rounds = 1.0 / (double)num_rounds;
    for (int32_t li_idx = 0; li_idx < n_leaves; li_idx++) {
        const RolloutLeafInfo *li = &leaf_infos[li_idx];
        for (int32_t bias = 0;
             bias < ROLLOUT_NUM_BIAS_PROFILES; bias++) {
            int64_t off =
                (((int64_t)li->leaf_idx * ROLLOUT_NUM_BIAS_PROFILES + bias)
                  * num_players) * max_hands;
            int64_t n_floats = (int64_t)num_players * max_hands;
            for (int64_t k = 0; k < n_floats; k++) {
                out[off + k] = (float)(acc[off + k] * inv_rounds);
            }
        }
    }

    free(acc);
    return 0;
}

BPS_EXPORT int32_t rollout_compute_subgame_leaf_values_bias4_c(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    const RolloutLeafInfo *leaf_infos, int32_t n_leaves,
    const int32_t *active, int32_t num_players,
    const RolloutHandList *player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    int32_t stack_chips,
    int32_t max_hands,
    int32_t num_rollouts,
    int32_t num_rounds,
    uint64_t master_seed,
    float *out, int32_t out_size
) {
    return rollout_bias4_core(
        r, board, n_board_cards, street, leaf_infos, n_leaves,
        active, num_players, player_hands, /*per_leaf=*/NULL,
        acting_order, acting_order_len, player_seats,
        game_action_history, gah_len, stack_chips, max_hands,
        num_rollouts, num_rounds, master_seed, out, out_size);
}

BPS_EXPORT int32_t rollout_compute_subgame_leaf_values_bias4_perleaf_c(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    const RolloutLeafInfo *leaf_infos, int32_t n_leaves,
    const int32_t *active, int32_t num_players,
    const RolloutHandList *player_hands,
    const RolloutHandList *per_leaf_player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    int32_t stack_chips,
    int32_t max_hands,
    int32_t num_rollouts,
    int32_t num_rounds,
    uint64_t master_seed,
    float *out, int32_t out_size
) {
    if (!per_leaf_player_hands) {
        rollout_set_error(street, num_players,
            "bias4 perleaf construct called with NULL "
            "per_leaf_player_hands");
        return -1;
    }
    return rollout_bias4_core(
        r, board, n_board_cards, street, leaf_infos, n_leaves,
        active, num_players, player_hands, per_leaf_player_hands,
        acting_order, acting_order_len, player_seats,
        game_action_history, gah_len, stack_chips, max_hands,
        num_rollouts, num_rounds, master_seed, out, out_size);
}
