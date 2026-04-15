/**
 * rollout.c — Pluribus-faithful depth-limited rollout leaf values (C port).
 *
 * Skeleton. All entry points return -1 with a "not implemented" error
 * until the algorithm is ported function-for-function from
 * python/rollout.py. See docs/C2_CPU_ROLLOUT_PLAN.md.
 */
#include "rollout.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "hand_eval.h"

/* ── splitmix64 (matches src/bps_reader.c:18-23) ─────────────────────── */

static inline uint64_t rollout_splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t rollout_derive_seed(
    uint64_t master,
    int32_t leaf_idx, int32_t combo,
    int32_t hero_p, int32_t hero_hand, int32_t rollout_idx
) {
    uint64_t s = master ^ 0xa5a5a5a5a5a5a5a5ULL;
    uint64_t t;
    t = s ^ (uint64_t)(uint32_t)leaf_idx;    s ^= rollout_splitmix64_next(&t);
    t = s ^ (uint64_t)(uint32_t)combo;       s ^= rollout_splitmix64_next(&t);
    t = s ^ (uint64_t)(uint32_t)hero_p;      s ^= rollout_splitmix64_next(&t);
    t = s ^ (uint64_t)(uint32_t)hero_hand;   s ^= rollout_splitmix64_next(&t);
    t = s ^ (uint64_t)(uint32_t)rollout_idx; s ^= rollout_splitmix64_next(&t);
    return s == 0 ? 0xdeadbeefcafebabeULL : s;
}

/* ── Thread-local error surface ──────────────────────────────────────── */

typedef struct {
    char    reason[256];
    int32_t street;
    int32_t num_players;
} RolloutLastError;

#if defined(_MSC_VER)
static __declspec(thread) RolloutLastError g_rollout_last_error;
#else
static _Thread_local RolloutLastError g_rollout_last_error;
#endif

static void rollout_set_error(int32_t street, int32_t num_players,
                              const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_rollout_last_error.reason,
              sizeof(g_rollout_last_error.reason), fmt, ap);
    va_end(ap);
    g_rollout_last_error.street = street;
    g_rollout_last_error.num_players = num_players;
}

BPS_EXPORT void rollout_get_last_error(
    const char **reason, int32_t *street, int32_t *num_players
) {
    if (reason)      *reason      = g_rollout_last_error.reason;
    if (street)      *street      = g_rollout_last_error.street;
    if (num_players) *num_players = g_rollout_last_error.num_players;
}

/* ── Pure-math exports (stubs) ───────────────────────────────────────── */

BPS_EXPORT void rollout_apply_bias_c(
    const double *strategy, int32_t n,
    int32_t bias_profile,
    const int32_t *categories,
    double *out
) {
    (void)strategy; (void)n; (void)bias_profile; (void)categories; (void)out;
    /* Phase 3b. */
    if (out && n > 0) memset(out, 0, (size_t)n * sizeof(double));
}

BPS_EXPORT int32_t rollout_classify_postflop_actions_c(
    int32_t facing_bet, int32_t raises_so_far,
    int32_t *out_cats, int32_t cap
) {
    (void)facing_bet; (void)raises_so_far; (void)out_cats; (void)cap;
    /* Phase 3b. */
    return 0;
}

BPS_EXPORT void rollout_action_amount_chips_c(
    int32_t action_idx,
    int32_t facing_bet, int32_t raises_so_far,
    int32_t pot, int32_t to_call, int32_t stack,
    int32_t *out_kind, int32_t *out_amount
) {
    (void)action_idx; (void)facing_bet; (void)raises_so_far;
    (void)pot; (void)to_call; (void)stack;
    /* Phase 3b. */
    if (out_kind)   *out_kind   = ROLLOUT_KIND_CHECK;
    if (out_amount) *out_amount = 0;
}

BPS_EXPORT float rollout_compute_showdown_payoff_c(
    const int *full_board,
    const int *hands_c0, const int *hands_c1,
    int32_t num_players,
    const int32_t *active,
    const int32_t *bets,
    int32_t pot_chips,
    int32_t hero_player
) {
    (void)full_board; (void)hands_c0; (void)hands_c1;
    (void)num_players; (void)active; (void)bets;
    (void)pot_chips; (void)hero_player;
    /* Phase 3b. */
    return 0.0f;
}

/* ── Rollout entry points (stubs) ────────────────────────────────────── */

BPS_EXPORT int32_t rollout_compute_leaf_value_c(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    int32_t pot_chips,
    const int32_t *active, int32_t num_players,
    const int32_t *stacks,
    const int32_t *bets,
    const RolloutHandList *player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    const int32_t *bias_profile,
    int32_t hero_player,
    int32_t hero_hand_idx,
    int32_t num_rollouts,
    uint64_t master_seed,
    int32_t leaf_idx_for_seed,
    int32_t combo_for_seed,
    float *out_value
) {
    (void)r; (void)board; (void)n_board_cards; (void)street;
    (void)pot_chips; (void)active; (void)num_players; (void)stacks;
    (void)bets; (void)player_hands; (void)acting_order;
    (void)acting_order_len; (void)player_seats; (void)game_action_history;
    (void)gah_len; (void)bias_profile; (void)hero_player;
    (void)hero_hand_idx; (void)num_rollouts; (void)master_seed;
    (void)leaf_idx_for_seed; (void)combo_for_seed;
    /* Touch derive_seed so the Phase 3a build doesn't warn about an
     * unused static in -Wall -Wextra mode. */
    (void)rollout_derive_seed(master_seed, leaf_idx_for_seed,
                              combo_for_seed, hero_player,
                              hero_hand_idx, 0);
    rollout_set_error(street, num_players,
        "rollout_compute_leaf_value_c: not implemented (Phase 3b)");
    if (out_value) *out_value = 0.0f;
    return -1;
}

BPS_EXPORT int32_t rollout_compute_leaf_values_all_hands_c(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street,
    int32_t pot_chips,
    const int32_t *active, int32_t num_players,
    const int32_t *stacks,
    const int32_t *bets,
    const RolloutHandList *player_hands,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    const int32_t *bias_profile,
    int32_t max_hands,
    int32_t num_rollouts,
    uint64_t master_seed,
    int32_t leaf_idx_for_seed,
    int32_t combo_for_seed,
    float *out
) {
    (void)r; (void)board; (void)n_board_cards; (void)street;
    (void)pot_chips; (void)active; (void)num_players; (void)stacks;
    (void)bets; (void)player_hands; (void)acting_order;
    (void)acting_order_len; (void)player_seats; (void)game_action_history;
    (void)gah_len; (void)bias_profile; (void)max_hands;
    (void)num_rollouts; (void)master_seed; (void)leaf_idx_for_seed;
    (void)combo_for_seed;
    rollout_set_error(street, num_players,
        "rollout_compute_leaf_values_all_hands_c: not implemented (Phase 3b)");
    if (out && num_players > 0 && max_hands > 0) {
        memset(out, 0, (size_t)num_players * (size_t)max_hands * sizeof(float));
    }
    return -1;
}

BPS_EXPORT int32_t rollout_compute_subgame_leaf_values_c(
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
    uint64_t master_seed,
    float *out, int32_t out_size
) {
    (void)r; (void)board; (void)n_board_cards; (void)street;
    (void)leaf_infos; (void)n_leaves; (void)active; (void)num_players;
    (void)player_hands; (void)acting_order; (void)acting_order_len;
    (void)player_seats; (void)game_action_history; (void)gah_len;
    (void)stack_chips; (void)max_hands; (void)num_rollouts;
    (void)master_seed; (void)out; (void)out_size;
    rollout_set_error(street, num_players,
        "rollout_compute_subgame_leaf_values_c: not implemented (Phase 3b)");
    return -1;
}
