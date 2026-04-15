/**
 * rollout.c — Pluribus-faithful depth-limited rollout leaf values (C port).
 *
 * Port of python/rollout.py, function-for-function. See
 * docs/C2_CPU_ROLLOUT_PLAN.md for module layout, RNG seed derivation,
 * known parity-with-Python limitations, and the test plan.
 *
 * Algorithm: Brown, Sandholm, "Superhuman AI for multiplayer poker"
 * (Science 2019, supplementary §4 / Algorithm 2).
 */
#include "rollout.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "hand_eval.h"

/* ── Compile-time caps ───────────────────────────────────────────────── */

#define ROLLOUT_MAX_HANDS          1326
#define ROLLOUT_MAX_ACTION_HISTORY 256
#define ROLLOUT_DECK               52

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

/* Uniform [0.0, 1.0) from a 64-bit state. */
static inline double rollout_rand_double(uint64_t *state) {
    /* Use top 53 bits for a mantissa-exact double in [0, 1). */
    uint64_t x = rollout_splitmix64_next(state);
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);  /* 2^53 */
}

/* Weighted discrete sample. weights[:n] need not sum to 1; total > 0.
 * Returns index in [0, n). */
static inline int32_t rollout_rand_weighted(
    uint64_t *state, const double *weights, int32_t n, double total
) {
    double u = rollout_rand_double(state) * total;
    double acc = 0.0;
    for (int32_t i = 0; i < n; i++) {
        acc += weights[i];
        if (u < acc) return i;
    }
    return n - 1;  /* numerical slack guard */
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

/* ── apply_bias ──────────────────────────────────────────────────────── */

/* Matches python/rollout.py::apply_bias.
 *
 * Semantics:
 *  - strategy sum <= 0 → uniform(n)
 *  - bias_profile == 0 → strategy / sum(strategy)
 *  - bias_profile ∈ {1,2,3} → multiply entries matching target category
 *    by BIAS_MULTIPLIER, renormalize. If post-bias sum is 0 (no mass on
 *    target, e.g. raise-biased at check-only node) → fall back to
 *    unbiased strategy/sum, NOT uniform.
 */
BPS_EXPORT void rollout_apply_bias_c(
    const double *strategy, int32_t n,
    int32_t bias_profile,
    const int32_t *categories,
    double *out
) {
    if (n <= 0 || !out) return;

    double s_sum = 0.0;
    for (int32_t i = 0; i < n; i++) s_sum += strategy[i];

    if (s_sum <= 0.0) {
        double u = 1.0 / (double)n;
        for (int32_t i = 0; i < n; i++) out[i] = u;
        return;
    }

    if (bias_profile == 0) {
        double inv = 1.0 / s_sum;
        for (int32_t i = 0; i < n; i++) out[i] = strategy[i] * inv;
        return;
    }

    int32_t target;
    switch (bias_profile) {
        case 1:  target = ROLLOUT_CAT_FOLD;  break;
        case 2:  target = ROLLOUT_CAT_CALL;  break;
        case 3:  target = ROLLOUT_CAT_RAISE; break;
        default: {
            /* Unknown profile — unbiased (safer than uniform). */
            double inv = 1.0 / s_sum;
            for (int32_t i = 0; i < n; i++) out[i] = strategy[i] * inv;
            return;
        }
    }

    double biased[ROLLOUT_MAX_ACTIONS_PER_NODE];
    if (n > ROLLOUT_MAX_ACTIONS_PER_NODE) {
        /* Oversized input — shouldn't happen given our tree, but guard. */
        double inv = 1.0 / s_sum;
        for (int32_t i = 0; i < n; i++) out[i] = strategy[i] * inv;
        return;
    }
    double total = 0.0;
    for (int32_t i = 0; i < n; i++) {
        biased[i] = strategy[i];
        if (categories && i < n && categories[i] == target) {
            biased[i] *= ROLLOUT_BIAS_MULTIPLIER;
        }
        total += biased[i];
    }

    if (total <= 0.0) {
        /* No mass on target — unbiased fallback. */
        double inv = 1.0 / s_sum;
        for (int32_t i = 0; i < n; i++) out[i] = strategy[i] * inv;
        return;
    }

    double inv = 1.0 / total;
    for (int32_t i = 0; i < n; i++) out[i] = biased[i] * inv;
}

/* ── classify_postflop_actions ───────────────────────────────────────── */

/* Matches python/rollout.py::classify_postflop_actions.
 *
 * Returns the number of actions. Layout:
 *   tree-closed + facing_bet: [fold, call]
 *   tree-closed + !facing_bet: [check]
 *   open + facing_bet (first raise): [fold, call, raise×0.5, raise×1.0, allin]
 *   open + facing_bet (subsequent): [fold, call, raise×1.0, allin]
 *   open + !facing_bet:              [check, bet×0.5, bet×1.0, allin]
 */
BPS_EXPORT int32_t rollout_classify_postflop_actions_c(
    int32_t facing_bet, int32_t raises_so_far,
    int32_t *out_cats, int32_t cap
) {
    int32_t n = 0;
    int32_t tree_closed = (raises_so_far >= ROLLOUT_POSTFLOP_MAX_RAISES);

    if (tree_closed) {
        if (facing_bet) {
            if (cap >= 2 && out_cats) {
                out_cats[0] = ROLLOUT_CAT_FOLD;
                out_cats[1] = ROLLOUT_CAT_CALL;
            }
            return 2;
        } else {
            if (cap >= 1 && out_cats) out_cats[0] = ROLLOUT_CAT_CALL;  /* check */
            return 1;
        }
    }

    if (facing_bet) {
        int32_t n_raises = (raises_so_far > 0) ? ROLLOUT_SUBSEQUENT_RAISE_N
                                               : ROLLOUT_FIRST_RAISE_N;
        /* [fold, call, raise×…, allin] */
        n = 2 + n_raises + 1;
        if (cap >= n && out_cats) {
            out_cats[0] = ROLLOUT_CAT_FOLD;
            out_cats[1] = ROLLOUT_CAT_CALL;
            for (int32_t i = 0; i < n_raises; i++) {
                out_cats[2 + i] = ROLLOUT_CAT_RAISE;
            }
            out_cats[n - 1] = ROLLOUT_CAT_RAISE;  /* allin */
        }
        return n;
    }

    /* not facing bet: [check, bet×0.5, bet×1.0, allin] */
    n = 1 + ROLLOUT_FIRST_RAISE_N + 1;
    if (cap >= n && out_cats) {
        out_cats[0] = ROLLOUT_CAT_CALL;  /* check */
        for (int32_t i = 0; i < ROLLOUT_FIRST_RAISE_N; i++) {
            out_cats[1 + i] = ROLLOUT_CAT_RAISE;
        }
        out_cats[n - 1] = ROLLOUT_CAT_RAISE;  /* allin */
    }
    return n;
}

/* ── action_amount_chips ─────────────────────────────────────────────── */

/* Matches python/rollout.py::action_amount_chips. See that function's
 * docstring for (pot, to_call, stack) semantics; raise_to formula is
 * Pluribus's: raise_to = to_call + frac * (pot + to_call). */
BPS_EXPORT void rollout_action_amount_chips_c(
    int32_t action_idx,
    int32_t facing_bet, int32_t raises_so_far,
    int32_t pot, int32_t to_call, int32_t stack,
    int32_t *out_kind, int32_t *out_amount
) {
    int32_t cats_n = rollout_classify_postflop_actions_c(
        facing_bet, raises_so_far, NULL, 0);
    int32_t tree_closed = (raises_so_far >= ROLLOUT_POSTFLOP_MAX_RAISES);

    if (action_idx < 0 || action_idx >= cats_n) {
        /* Python raises ValueError; we return a conservative check. */
        if (out_kind)   *out_kind   = ROLLOUT_KIND_CHECK;
        if (out_amount) *out_amount = 0;
        return;
    }

    if (facing_bet) {
        if (action_idx == 0) {
            if (out_kind)   *out_kind   = ROLLOUT_KIND_FOLD;
            if (out_amount) *out_amount = 0;
            return;
        }
        if (action_idx == 1) {
            int32_t a = to_call < stack ? to_call : stack;
            if (out_kind)   *out_kind   = ROLLOUT_KIND_CALL;
            if (out_amount) *out_amount = a;
            return;
        }
        if (tree_closed) {
            int32_t a = to_call < stack ? to_call : stack;
            if (out_kind)   *out_kind   = ROLLOUT_KIND_CALL;
            if (out_amount) *out_amount = a;
            return;
        }
        if (action_idx == cats_n - 1) {
            if (out_kind)   *out_kind   = ROLLOUT_KIND_ALLIN;
            if (out_amount) *out_amount = stack;
            return;
        }
        /* Sized raise */
        int32_t frac_slot = action_idx - 2;
        double frac;
        if (raises_so_far > 0) {
            /* SUBSEQUENT_BET_FRACTIONS = (1.0,) */
            frac = 1.0;
        } else {
            /* POSTFLOP_BET_FRACTIONS = (0.5, 1.0) */
            frac = (frac_slot == 0) ? 0.5 : 1.0;
        }
        /* raise_to = to_call + int(frac * (pot + to_call)) — Python int() truncates toward 0 */
        int32_t raise_to = to_call + (int32_t)(frac * (double)(pot + to_call));
        int32_t amount = raise_to < stack ? raise_to : stack;
        /* Python: max(amount, to_call + 1) if stack > to_call + 1 else stack */
        int32_t kind_amt;
        if (stack > to_call + 1) {
            kind_amt = amount > (to_call + 1) ? amount : (to_call + 1);
        } else {
            kind_amt = stack;
        }
        if (out_kind)   *out_kind   = ROLLOUT_KIND_RAISE;
        if (out_amount) *out_amount = kind_amt;
        return;
    }

    /* Not facing bet */
    if (action_idx == 0) {
        if (out_kind)   *out_kind   = ROLLOUT_KIND_CHECK;
        if (out_amount) *out_amount = 0;
        return;
    }
    if (tree_closed) {
        if (out_kind)   *out_kind   = ROLLOUT_KIND_CHECK;
        if (out_amount) *out_amount = 0;
        return;
    }
    if (action_idx == cats_n - 1) {
        if (out_kind)   *out_kind   = ROLLOUT_KIND_ALLIN;
        if (out_amount) *out_amount = stack;
        return;
    }
    int32_t frac_slot = action_idx - 1;
    double frac = (frac_slot == 0) ? 0.5 : 1.0;
    int32_t bet_amount = (int32_t)(frac * (double)pot);
    /* Python: min(max(bet_amount, 1), stack) */
    int32_t a = bet_amount > 1 ? bet_amount : 1;
    if (a > stack) a = stack;
    if (out_kind)   *out_kind   = ROLLOUT_KIND_BET;
    if (out_amount) *out_amount = a;
}

/* ── Showdown payoff ─────────────────────────────────────────────────── */

/* Matches python/rollout.py::compute_showdown_payoff — winner-takes /
 * chop, no side pots. */
BPS_EXPORT float rollout_compute_showdown_payoff_c(
    const int *full_board,
    const int *hands_c0, const int *hands_c1,
    int32_t num_players,
    const int32_t *active,
    const int32_t *bets,
    int32_t pot_chips,
    int32_t hero_player
) {
    /* Live players at showdown */
    int32_t live[ROLLOUT_MAX_PLAYERS];
    int32_t n_live = 0;
    for (int32_t i = 0; i < num_players; i++) {
        if (active[i]) live[n_live++] = i;
    }
    if (n_live == 0) {
        return -(float)bets[hero_player];
    }

    /* Strength per live player */
    uint32_t strengths[ROLLOUT_MAX_PLAYERS];
    uint32_t best = 0;
    for (int32_t k = 0; k < n_live; k++) {
        int32_t p = live[k];
        int seven[7] = {
            full_board[0], full_board[1], full_board[2],
            full_board[3], full_board[4],
            hands_c0[p], hands_c1[p]
        };
        uint32_t s = eval7(seven);
        strengths[k] = s;
        if (s > best) best = s;
    }

    int32_t n_winners = 0;
    int32_t hero_wins = 0;
    for (int32_t k = 0; k < n_live; k++) {
        if (strengths[k] == best) {
            n_winners++;
            if (live[k] == hero_player) hero_wins = 1;
        }
    }

    double hero_share = 0.0;
    if (hero_wins) {
        hero_share = (double)pot_chips / (double)n_winners;
    }
    return (float)(hero_share - (double)bets[hero_player]);
}

/* ── Blueprint lookup (fail-loud) ────────────────────────────────────── */

/* Returns num_actions written to out_strat (normalized), or 0 on miss.
 * Matches python/rollout.py::_lookup_strategy except Python-level
 * coverage recording (handled at the reference path).
 *
 * Note: bps_query_strategy already normalizes; zero-sum inputs are
 * turned into uniform by C. To match Python's fail-loud-on-zero-sum we
 * would need the raw bucket counts, which aren't exposed. This matches
 * the observable behavior of Python-over-BPSReaderC: never raises on
 * zero-sum because the C side already smoothed to uniform. Flagged in
 * docs/C2_CPU_ROLLOUT_PLAN.md "Known parity-with-Python limitations". */
static int rollout_lookup_strategy(
    const BPSReader *r,
    const int32_t *action_history, int32_t ah_len,
    int32_t player, int32_t bucket, int32_t street,
    float *out_strat, int32_t cap
) {
    if (cap < BPS_MAX_ACTIONS) {
        /* Should never happen — callers pass a big-enough buffer. */
        return 0;
    }
    /* bps_query_strategy signature: const int *action_history, int action_len.
     * Our ah values are int32_t which matches int on ILP32/LP64/LLP64. */
    int na = bps_query_strategy(r, (const int *)action_history,
                                 (int)ah_len,
                                 (int)player, (int)street, (int)bucket,
                                 out_strat);
    return na;  /* 0 means node not found */
}

/* ── Chance / dealing ────────────────────────────────────────────────── */

/* Uniform random card from the 52 not in used_mask. Returns -1 on
 * exhausted deck. */
static int32_t rollout_deal_one_card(uint64_t *rng, uint64_t used_mask) {
    /* Count unused */
    int32_t unused[ROLLOUT_DECK];
    int32_t n_unused = 0;
    for (int32_t c = 0; c < ROLLOUT_DECK; c++) {
        if (!(used_mask & (1ULL << c))) unused[n_unused++] = c;
    }
    if (n_unused == 0) return -1;
    uint64_t r = rollout_splitmix64_next(rng);
    int32_t idx = (int32_t)(r % (uint64_t)n_unused);
    return unused[idx];
}

/* Sample one opponent hand weighted by reach, avoiding conflicts.
 * Returns 0 on success, 1 if no valid hand. */
static int rollout_sample_opp_hand(
    uint64_t *rng,
    const RolloutHand *candidates, int32_t n_cand,
    uint64_t used_mask,
    int32_t *out_c0, int32_t *out_c1
) {
    /* Gather valid candidates + weights. */
    double weights[ROLLOUT_MAX_HANDS];
    int32_t valid_idx[ROLLOUT_MAX_HANDS];
    int32_t n_valid = 0;
    double total = 0.0;

    int32_t cap = n_cand;
    if (cap > ROLLOUT_MAX_HANDS) cap = ROLLOUT_MAX_HANDS;
    for (int32_t i = 0; i < cap; i++) {
        if (candidates[i].weight <= 0.0f) continue;
        int32_t c0 = candidates[i].c0;
        int32_t c1 = candidates[i].c1;
        if (c0 < 0 || c0 >= 52 || c1 < 0 || c1 >= 52) continue;
        if (used_mask & (1ULL << c0)) continue;
        if (used_mask & (1ULL << c1)) continue;
        weights[n_valid] = (double)candidates[i].weight;
        valid_idx[n_valid] = i;
        total += weights[n_valid];
        n_valid++;
    }
    if (n_valid == 0 || total <= 0.0) return 1;

    int32_t pick = rollout_rand_weighted(rng, weights, n_valid, total);
    int32_t idx = valid_idx[pick];
    *out_c0 = candidates[idx].c0;
    *out_c1 = candidates[idx].c1;
    return 0;
}

/* ── _simulate_one ───────────────────────────────────────────────────── */

/* Streets: 1=flop 2=turn 3=river. Returns 0 on success (payoff written
 * to *out_payoff), -1 on LeafValueUnavailable (last-error set). */
static int rollout_simulate_one(
    const BPSReader *r,
    /* immutable inputs */
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
    const int32_t *has_sample,  /* 1 if sampled_hands[p] is set */
    uint64_t *rng,
    /* out */
    float *out_payoff
) {
    /* Local copies (mutable during rollout) */
    int32_t active[ROLLOUT_MAX_PLAYERS];
    int32_t stacks[ROLLOUT_MAX_PLAYERS];
    int32_t bets[ROLLOUT_MAX_PLAYERS];
    int32_t board[5];
    int32_t n_board = n_board_in;
    int32_t game_history[ROLLOUT_MAX_ACTION_HISTORY];
    int32_t gah_len = gah_len_in;

    for (int32_t i = 0; i < num_players; i++) {
        active[i] = active_in[i];
        stacks[i] = stacks_in[i];
        bets[i]   = bets_in[i];
    }
    for (int32_t i = 0; i < n_board_in; i++) board[i] = board_in[i];
    if (gah_len > ROLLOUT_MAX_ACTION_HISTORY) {
        rollout_set_error(leaf_street, num_players,
            "game_action_history too long (%d > %d)",
            gah_len, ROLLOUT_MAX_ACTION_HISTORY);
        return -1;
    }
    for (int32_t i = 0; i < gah_len; i++) game_history[i] = game_history_in[i];

    int32_t pot = pot_in;
    int32_t street = leaf_street;

    while (1) {
        /* Betting round for current street */
        int32_t street_bets[ROLLOUT_MAX_PLAYERS] = {0};
        int32_t current_bet = 0;
        int32_t raises_this_street = 0;
        int32_t acted[ROLLOUT_MAX_PLAYERS] = {0};
        int32_t safety = num_players * (ROLLOUT_POSTFLOP_MAX_RAISES + 2) + 4;

        for (int32_t iter_idx = 0; iter_idx < safety; iter_idx++) {
            /* Find next active player owing action, in acting_order */
            int32_t chosen = -1;
            for (int32_t k = 0; k < acting_order_len; k++) {
                int32_t p = acting_order[k];
                if (!active[p]) continue;
                if (!acted[p] || street_bets[p] < current_bet) {
                    chosen = p;
                    break;
                }
            }
            if (chosen < 0) break;

            int32_t n_active = 0;
            for (int32_t i = 0; i < num_players; i++) if (active[i]) n_active++;
            if (n_active <= 1) break;

            int32_t p = chosen;
            if (!has_sample[p]) {
                /* No hand sampled — treat as silent check/fold step.
                 * Matches Python: "treat as check/fold conservatively". */
                acted[p] = 1;
                continue;
            }

            int32_t c0 = sampled_hands_c0[p];
            int32_t c1 = sampled_hands_c1[p];
            int bucket = bps_hand_to_bucket(r, board, c0, c1);
            if (bucket < 0) {
                rollout_set_error(street, num_players,
                    "hand_to_bucket returned %d for (%d,%d) on %d-card board "
                    "(street=%d) — likely turn/river lookup where only flop "
                    "buckets are cached", bucket, c0, c1, n_board, street);
                return -1;
            }

            int32_t facing_bet = (current_bet > street_bets[p]);
            int32_t to_call = facing_bet ? (current_bet - street_bets[p]) : 0;
            int32_t cats[ROLLOUT_MAX_ACTIONS_PER_NODE];
            int32_t n_actions = rollout_classify_postflop_actions_c(
                facing_bet, raises_this_street, cats,
                ROLLOUT_MAX_ACTIONS_PER_NODE);

            /* Blueprint lookup — full seat for the blueprint's
             * seat-indexed storage. */
            int32_t full_seat = player_seats[p];
            float strat_full[BPS_MAX_ACTIONS];
            int bp_na = rollout_lookup_strategy(
                r, game_history, gah_len,
                full_seat, bucket, street,
                strat_full, BPS_MAX_ACTIONS);
            if (bp_na <= 0) {
                rollout_set_error(street, num_players,
                    "blueprint sigma missing: player=%d(seat=%d) bucket=%d "
                    "street=%d ah_len=%d", p, full_seat, bucket, street, gah_len);
                return -1;
            }

            /* Truncate/pad to rollout tree's action count — matches
             * python/rollout.py:485-503. */
            double strat[ROLLOUT_MAX_ACTIONS_PER_NODE];
            int32_t cats_active_n = n_actions;
            if (bp_na >= n_actions) {
                for (int32_t i = 0; i < n_actions; i++)
                    strat[i] = (double)strat_full[i];
            } else {
                for (int32_t i = 0; i < n_actions; i++) strat[i] = 0.0;
                for (int32_t i = 0; i < bp_na; i++)
                    strat[i] = (double)strat_full[i];
                /* Python: cats = cats[:max(len(strat_full), 1)] */
                cats_active_n = bp_na > 0 ? bp_na : 1;
                if (cats_active_n > n_actions) cats_active_n = n_actions;
                /* Degenerate: all zero — fail loud */
                double sum = 0.0;
                for (int32_t i = 0; i < n_actions; i++) sum += strat[i];
                if (sum <= 0.0) {
                    rollout_set_error(street, num_players,
                        "blueprint empty strategy slice: player=%d bucket=%d",
                        p, bucket);
                    return -1;
                }
            }

            /* apply_bias — note Python passes `cats` (possibly truncated)
             * but strat has n_actions entries. apply_bias iterates to
             * min(len(biased), len(categories)) via the `if i < len(biased)`
             * guard. Entries beyond cats_active_n simply aren't category-
             * matched (category doesn't match target → not multiplied),
             * so effectively contribute raw strategy mass. Match that. */
            double biased[ROLLOUT_MAX_ACTIONS_PER_NODE];
            /* Build an effective category array that matches Python's
             * truncation: indices >= cats_active_n get a sentinel that
             * won't match any target. */
            int32_t eff_cats[ROLLOUT_MAX_ACTIONS_PER_NODE];
            for (int32_t i = 0; i < n_actions; i++) {
                eff_cats[i] = (i < cats_active_n) ? cats[i] : -1;
            }
            rollout_apply_bias_c(strat, n_actions, bias_profile[p],
                                 eff_cats, biased);

            /* Defensive re-normalize (python/rollout.py:506-510) */
            double bsum = 0.0;
            for (int32_t i = 0; i < n_actions; i++) bsum += biased[i];
            if (bsum <= 0.0) {
                double u = 1.0 / (double)n_actions;
                for (int32_t i = 0; i < n_actions; i++) biased[i] = u;
                bsum = 1.0;
            }

            int32_t action_idx = rollout_rand_weighted(rng, biased, n_actions, bsum);

            /* Execute action */
            int32_t kind, amt;
            rollout_action_amount_chips_c(
                action_idx, facing_bet, raises_this_street,
                pot, to_call, stacks[p], &kind, &amt);

            switch (kind) {
                case ROLLOUT_KIND_FOLD:
                    active[p] = 0;
                    acted[p] = 1;
                    break;
                case ROLLOUT_KIND_CALL:
                case ROLLOUT_KIND_CHECK:
                    stacks[p]      -= amt;
                    street_bets[p] += amt;
                    bets[p]        += amt;
                    pot            += amt;
                    acted[p] = 1;
                    break;
                case ROLLOUT_KIND_BET:
                case ROLLOUT_KIND_RAISE:
                case ROLLOUT_KIND_ALLIN:
                    stacks[p]      -= amt;
                    street_bets[p] += amt;
                    bets[p]        += amt;
                    pot            += amt;
                    if (street_bets[p] > current_bet) {
                        current_bet = street_bets[p];
                    }
                    raises_this_street++;
                    acted[p] = 1;
                    /* Reopen action for other live players */
                    for (int32_t q = 0; q < num_players; q++) {
                        if (q != p && active[q]) acted[q] = 0;
                    }
                    break;
                default:
                    rollout_set_error(street, num_players,
                        "unknown action kind %d", kind);
                    return -1;
            }

            /* Append to game history */
            if (gah_len >= ROLLOUT_MAX_ACTION_HISTORY) {
                rollout_set_error(street, num_players,
                    "game_action_history overflow during rollout");
                return -1;
            }
            game_history[gah_len++] = action_idx;

            int32_t n_active2 = 0;
            for (int32_t i = 0; i < num_players; i++) if (active[i]) n_active2++;
            if (n_active2 <= 1) break;
        }

        /* End of street */
        int32_t n_live = 0;
        for (int32_t i = 0; i < num_players; i++) if (active[i]) n_live++;
        if (n_live <= 1) break;

        /* Deal next street */
        int32_t next_street;
        if      (street == ROLLOUT_STREET_FLOP)  next_street = ROLLOUT_STREET_TURN;
        else if (street == ROLLOUT_STREET_TURN)  next_street = ROLLOUT_STREET_RIVER;
        else                                      break;  /* river done → showdown */

        /* Build used mask: board + all sampled player hands */
        uint64_t used = 0ULL;
        for (int32_t i = 0; i < n_board; i++) used |= (1ULL << board[i]);
        for (int32_t i = 0; i < num_players; i++) {
            if (has_sample[i]) {
                used |= (1ULL << sampled_hands_c0[i]);
                used |= (1ULL << sampled_hands_c1[i]);
            }
        }
        int32_t new_card = rollout_deal_one_card(rng, used);
        if (new_card < 0) {
            rollout_set_error(street, num_players, "deck exhausted during deal");
            return -1;
        }
        if (n_board >= 5) {
            rollout_set_error(street, num_players, "board overflow during deal");
            return -1;
        }
        board[n_board++] = new_card;
        street = next_street;
    }

    /* Terminal */
    int32_t n_live = 0;
    int32_t last_live = -1;
    for (int32_t i = 0; i < num_players; i++) {
        if (active[i]) { n_live++; last_live = i; }
    }
    if (n_live <= 1) {
        if (n_live == 1 && last_live == hero_player) {
            *out_payoff = (float)pot - (float)bets[hero_player];
        } else {
            *out_payoff = -(float)bets[hero_player];
        }
        return 0;
    }

    /* Showdown — deal remaining board if needed */
    while (n_board < 5) {
        uint64_t used = 0ULL;
        for (int32_t i = 0; i < n_board; i++) used |= (1ULL << board[i]);
        for (int32_t i = 0; i < num_players; i++) {
            if (has_sample[i]) {
                used |= (1ULL << sampled_hands_c0[i]);
                used |= (1ULL << sampled_hands_c1[i]);
            }
        }
        int32_t new_card = rollout_deal_one_card(rng, used);
        if (new_card < 0) {
            rollout_set_error(street, num_players, "deck exhausted pre-showdown");
            return -1;
        }
        board[n_board++] = new_card;
    }

    /* Fill hands_c0/c1 arrays for showdown; inactive get (0,1) sentinel. */
    int full_board[5];
    for (int i = 0; i < 5; i++) full_board[i] = board[i];

    int hands_c0[ROLLOUT_MAX_PLAYERS];
    int hands_c1[ROLLOUT_MAX_PLAYERS];
    for (int32_t i = 0; i < num_players; i++) {
        if (has_sample[i]) {
            hands_c0[i] = sampled_hands_c0[i];
            hands_c1[i] = sampled_hands_c1[i];
        } else {
            hands_c0[i] = 0;
            hands_c1[i] = 1;
        }
    }

    *out_payoff = rollout_compute_showdown_payoff_c(
        full_board, hands_c0, hands_c1, num_players,
        active, bets, pot, hero_player);
    return 0;
}

/* ── Public entry points ─────────────────────────────────────────────── */

/* Common args → compute mean payoff for one hero hand. */
static int rollout_compute_leaf_value_impl(
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
    if (num_players <= 0 || num_players > ROLLOUT_MAX_PLAYERS) {
        rollout_set_error(street, num_players, "bad num_players=%d", num_players);
        return -1;
    }
    if (hero_player < 0 || hero_player >= num_players) {
        rollout_set_error(street, num_players,
            "bad hero_player=%d", hero_player);
        return -1;
    }
    if (!active[hero_player]) {
        /* Hero folded — pay -bets[hero]. Matches Python. */
        *out_value = -(float)bets[hero_player];
        return 0;
    }
    const RolloutHandList *hh = &player_hands[hero_player];
    if (hero_hand_idx < 0 || hero_hand_idx >= hh->n_hands) {
        rollout_set_error(street, num_players,
            "hero_hand_idx %d out of range (%d)",
            hero_hand_idx, hh->n_hands);
        return -1;
    }
    int32_t hc0 = hh->hands[hero_hand_idx].c0;
    int32_t hc1 = hh->hands[hero_hand_idx].c1;
    float hw   = hh->hands[hero_hand_idx].weight;

    if (hw <= 0.0f) { *out_value = 0.0f; return 0; }

    /* Blocked by board? */
    uint64_t board_mask = 0ULL;
    for (int32_t i = 0; i < n_board_cards; i++) board_mask |= (1ULL << board[i]);
    if ((board_mask & (1ULL << hc0)) || (board_mask & (1ULL << hc1))) {
        *out_value = 0.0f;
        return 0;
    }

    double payoff_sum = 0.0;
    int32_t payoffs_recorded = 0;

    for (int32_t r_idx = 0; r_idx < num_rollouts; r_idx++) {
        uint64_t rng = rollout_derive_seed(
            master_seed, leaf_idx_for_seed, combo_for_seed,
            hero_player, hero_hand_idx, r_idx);

        /* Sample opponent hands */
        int32_t sampled_c0[ROLLOUT_MAX_PLAYERS];
        int32_t sampled_c1[ROLLOUT_MAX_PLAYERS];
        int32_t has_sample[ROLLOUT_MAX_PLAYERS];
        for (int32_t i = 0; i < num_players; i++) {
            sampled_c0[i] = 0; sampled_c1[i] = 0; has_sample[i] = 0;
        }
        sampled_c0[hero_player] = hc0;
        sampled_c1[hero_player] = hc1;
        has_sample[hero_player] = 1;

        uint64_t used = board_mask | (1ULL << hc0) | (1ULL << hc1);
        int ok = 1;
        for (int32_t p = 0; p < num_players; p++) {
            if (p == hero_player) continue;
            if (!active[p]) continue;
            int32_t c0, c1;
            if (rollout_sample_opp_hand(&rng,
                    player_hands[p].hands, player_hands[p].n_hands,
                    used, &c0, &c1) != 0) {
                ok = 0;
                break;
            }
            sampled_c0[p] = c0;
            sampled_c1[p] = c1;
            has_sample[p] = 1;
            used |= (1ULL << c0);
            used |= (1ULL << c1);
        }
        if (!ok) continue;

        float payoff;
        int rc = rollout_simulate_one(
            r, board, n_board_cards, street, pot_chips,
            active, stacks, bets, num_players,
            acting_order, acting_order_len,
            player_seats, game_action_history, gah_len,
            bias_profile, hero_player,
            sampled_c0, sampled_c1, has_sample,
            &rng, &payoff);
        if (rc != 0) {
            /* Propagate LeafValueUnavailable. */
            return -1;
        }
        payoff_sum += (double)payoff;
        payoffs_recorded++;
    }

    if (payoffs_recorded == 0) { *out_value = 0.0f; return 0; }
    *out_value = (float)(payoff_sum / (double)payoffs_recorded);
    return 0;
}

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
    return rollout_compute_leaf_value_impl(
        r, board, n_board_cards, street, pot_chips,
        active, num_players, stacks, bets,
        player_hands, acting_order, acting_order_len,
        player_seats, game_action_history, gah_len,
        bias_profile, hero_player, hero_hand_idx,
        num_rollouts, master_seed,
        leaf_idx_for_seed, combo_for_seed, out_value);
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
    /* Zero-init */
    memset(out, 0, (size_t)num_players * (size_t)max_hands * sizeof(float));

    uint64_t board_mask = 0ULL;
    for (int32_t i = 0; i < n_board_cards; i++) board_mask |= (1ULL << board[i]);

    for (int32_t p = 0; p < num_players; p++) {
        if (!active[p]) continue;
        const RolloutHandList *hl = &player_hands[p];
        int32_t n = hl->n_hands < max_hands ? hl->n_hands : max_hands;
        for (int32_t h = 0; h < n; h++) {
            const RolloutHand *hd = &hl->hands[h];
            if (hd->weight <= 0.0f) continue;
            if (board_mask & (1ULL << hd->c0)) continue;
            if (board_mask & (1ULL << hd->c1)) continue;

            float v;
            int rc = rollout_compute_leaf_value_impl(
                r, board, n_board_cards, street, pot_chips,
                active, num_players, stacks, bets,
                player_hands, acting_order, acting_order_len,
                player_seats, game_action_history, gah_len,
                bias_profile, p, h, num_rollouts, master_seed,
                leaf_idx_for_seed, combo_for_seed, &v);
            if (rc != 0) return -1;
            out[p * max_hands + h] = v;
        }
    }
    return 0;
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
    /* Compute cont_per_leaf = NUM_BIAS_PROFILES^num_players */
    int64_t cont_per_leaf = 1;
    for (int32_t i = 0; i < num_players; i++) {
        cont_per_leaf *= ROLLOUT_NUM_BIAS_PROFILES;
    }

    /* total = (max_leaf_idx + 1) * cont_per_leaf */
    int32_t max_leaf_idx = 0;
    for (int32_t i = 0; i < n_leaves; i++) {
        if (leaf_infos[i].leaf_idx > max_leaf_idx) {
            max_leaf_idx = leaf_infos[i].leaf_idx;
        }
    }
    int64_t total = (int64_t)(max_leaf_idx + 1) * cont_per_leaf;
    int64_t floats_needed = total * (int64_t)num_players * (int64_t)max_hands;

    if (floats_needed > (int64_t)out_size) {
        rollout_set_error(street, num_players,
            "out buffer too small: need %lld floats, got %d",
            (long long)floats_needed, out_size);
        return -1;
    }

    memset(out, 0, (size_t)floats_needed * sizeof(float));

    /* bias combo decoding: bias[p] = (combo / NUM_BIAS_PROFILES^p) % NUM_BIAS_PROFILES */
    for (int32_t li_idx = 0; li_idx < n_leaves; li_idx++) {
        const RolloutLeafInfo *li = &leaf_infos[li_idx];
        int32_t stacks[ROLLOUT_MAX_PLAYERS];
        for (int32_t p = 0; p < num_players; p++) {
            int32_t s = stack_chips - li->bets[p];
            stacks[p] = s < 0 ? 0 : s;
        }

        for (int64_t combo = 0; combo < cont_per_leaf; combo++) {
            int32_t bias_profile[ROLLOUT_MAX_PLAYERS];
            int64_t c = combo;
            for (int32_t p = 0; p < num_players; p++) {
                bias_profile[p] = (int32_t)(c % ROLLOUT_NUM_BIAS_PROFILES);
                c /= ROLLOUT_NUM_BIAS_PROFILES;
            }

            int64_t flat = (int64_t)li->leaf_idx * cont_per_leaf + combo;
            float *slot = out + flat * (int64_t)num_players * (int64_t)max_hands;

            int rc = rollout_compute_leaf_values_all_hands_c(
                r, board, n_board_cards, street, li->pot_chips,
                active, num_players, stacks, li->bets,
                player_hands, acting_order, acting_order_len,
                player_seats, game_action_history, gah_len,
                bias_profile, max_hands, num_rollouts, master_seed,
                li->leaf_idx, (int32_t)combo, slot);
            if (rc != 0) return -1;
        }
    }
    return 0;
}
