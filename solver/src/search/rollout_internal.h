/**
 * rollout_internal.h — cross-TU internal declarations shared between rollout.c
 * and the sampled-core split rollout_sampled.c.
 *
 * NOT a public API. These were file-static in rollout.c; sharing them lets the
 * sampled subgame-leaf block live in a sibling translation unit compiled into
 * the SAME shared library. No behavior change.
 *
 * The two seed helpers are kept `static inline` here (header-shared, each TU
 * inlines its own copy — no exported symbol). rollout_set_error and
 * rollout_simulate_one are promoted to extern (definitions stay in rollout.c,
 * which owns the thread-local error buffer and the simulator's globals).
 */
#ifndef ROLLOUT_INTERNAL_H
#define ROLLOUT_INTERNAL_H

#include <stdint.h>
#include "rollout.h"

/* Compile-time cap shared with rollout.c (max 2-card hands). Guarded so the
 * parent's own definition (identical value) does not warn. */
#ifndef ROLLOUT_MAX_HANDS
#define ROLLOUT_MAX_HANDS 1326
#endif


/* Hidden-visibility for promoted-from-static internals (see mccfr_blueprint_internal.h).
 * Keeps these un-exported so a second in-process .so cannot interpose them. */
#ifndef NEXUS_INTERNAL
#  if defined(__GNUC__) || defined(__clang__)
#    define NEXUS_INTERNAL __attribute__((visibility("hidden")))
#  else
#    define NEXUS_INTERNAL
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* splitmix64 step — shared seed mixer (header-inline; no symbol). */
static inline uint64_t rollout_splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* Deterministic per-rollout seed derivation (header-inline; no symbol). */
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

/* Set the thread-local last-error (printf-style). Definition (and the
 * g_rollout_last_error thread-local) stay in rollout.c. */
NEXUS_INTERNAL void rollout_set_error(int32_t street, int32_t num_players,
                       const char *fmt, ...);

/* Pinned-runout deal helper (header-inline; no symbol). A caller that wants
 * every hero hand of one vector leaf visit to see the SAME turn/river draws
 * the runout once (from the deck minus board minus the opponents' cards) and
 * passes it as `pinned`; the simulator consumes it in order, skipping cards
 * that are already used in the current world (the hero's own hole cards).
 * Returns -1 when `pinned` is NULL or exhausted so the caller falls back to a
 * fresh deal. Never touches the RNG, so a NULL sequence is byte-identical to
 * the historical deal path. The first two surviving cards of a uniformly
 * random sequence are a uniform ordered pair from the hand-conditioned deck,
 * so the shared runout is unbiased for every hero hand. */
static inline int32_t rollout_pinned_next_card(
    const int32_t *pinned, int32_t pinned_len, int32_t *pos, uint64_t used_mask
) {
    if (!pinned || !pos) return -1;
    while (*pos < pinned_len) {
        int32_t c = pinned[(*pos)++];
        if (c < 0 || c >= 52) continue;
        if (!(used_mask & (1ULL << c))) return c;
    }
    return -1;
}

/* Simulate one full rollout from a subgame leaf to terminal; writes payoff.
 * Definition in rollout.c (owns the simulator state/globals). */
NEXUS_INTERNAL int rollout_simulate_one(
    const BPSReader *r,
    const int *board_in, int32_t n_board_in,
    int32_t leaf_street,
    int32_t pot_in,
    const int32_t *active_in,
    const int32_t *stacks_in,
    const int32_t *bets_in,
    int32_t num_players,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_history_in, int32_t gah_len_in,
    const int32_t *bias_profile,
    int32_t hero_player,
    const int32_t *sampled_hands_c0,
    const int32_t *sampled_hands_c1,
    const int32_t *has_sample,
    int32_t skip_first_betting,
    uint64_t *rng,
    float *out_payoff
);

/* Same simulator with an optional caller-pinned turn/river sequence (see
 * rollout_pinned_next_card). rollout_simulate_one is this with NULL/0. */
NEXUS_INTERNAL int rollout_simulate_one_ex(
    const BPSReader *r,
    const int *board_in, int32_t n_board_in,
    int32_t leaf_street,
    int32_t pot_in,
    const int32_t *active_in,
    const int32_t *stacks_in,
    const int32_t *bets_in,
    int32_t num_players,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_history_in, int32_t gah_len_in,
    const int32_t *bias_profile,
    int32_t hero_player,
    const int32_t *sampled_hands_c0,
    const int32_t *sampled_hands_c1,
    const int32_t *has_sample,
    int32_t skip_first_betting,
    const int32_t *pinned_runout, int32_t pinned_len,
    uint64_t *rng,
    float *out_payoff
);

#ifdef __cplusplus
}
#endif

#endif /* ROLLOUT_INTERNAL_H */
