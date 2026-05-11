/**
 * rollout.h — Pluribus-faithful depth-limited rollout leaf values (C port).
 *
 * Port of python/rollout.py. The Python module is the canonical
 * correctness reference. This C port is the shipping path for live
 * flop solves; the API imports it via ctypes in python/rollout_c.py.
 *
 * See docs/C2_CPU_ROLLOUT_PLAN.md for:
 *   - Function-for-function mapping to python/rollout.py
 *   - RNG seed derivation spec (splitmix64 chain)
 *   - Known parity-with-Python limitations
 *   - Test plan and bench target
 *
 * Algorithm: Brown, Sandholm, "Superhuman AI for multiplayer poker"
 * (Science 2019, supplementary §4 / Algorithm 2).
 *
 * Returns 0 on success, -1 on error. Errors set a thread-local
 * rollout_last_error readable via rollout_get_last_error().
 */
#ifndef ROLLOUT_H
#define ROLLOUT_H

#include <stdint.h>
#include <stddef.h>

#include "bps_reader.h"            /* BPSReader and BPS_EXPORT */
#include "mccfr_cont_ev_table.h"   /* BPContEVTable (Phase F consumer) */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants — match python/rollout.py ─────────────────────────────── */

#define ROLLOUT_NUM_ROLLOUTS_DEFAULT   3
#define ROLLOUT_BIAS_MULTIPLIER        5.0
#define ROLLOUT_NUM_BIAS_PROFILES      4
#define ROLLOUT_POSTFLOP_MAX_RAISES    3
#define ROLLOUT_MAX_PLAYERS            6
/* [0.5, 1.0] first raise; [1.0] subsequent — matches training's
 * POSTFLOP_BET_SIZES in precompute/blueprint_worker_unified.py:51. */
#define ROLLOUT_FIRST_RAISE_N          2   /* 0.5, 1.0 */
#define ROLLOUT_SUBSEQUENT_RAISE_N     1   /* 1.0 */

/* Action categories for the bias transform. Matches
 * python/rollout.py::ActionCategory. CALL covers check; RAISE covers
 * bet / raise / all-in of any size. */
#define ROLLOUT_CAT_FOLD  0
#define ROLLOUT_CAT_CALL  1
#define ROLLOUT_CAT_RAISE 2

/* Action kinds returned by rollout_action_amount_chips_c. */
#define ROLLOUT_KIND_FOLD  0
#define ROLLOUT_KIND_CALL  1
#define ROLLOUT_KIND_CHECK 2
#define ROLLOUT_KIND_BET   3
#define ROLLOUT_KIND_RAISE 4
#define ROLLOUT_KIND_ALLIN 5

/* Street ints match bps_reader.h. */
#define ROLLOUT_STREET_FLOP  1
#define ROLLOUT_STREET_TURN  2
#define ROLLOUT_STREET_RIVER 3

/* Upper bound on action count at any single node: [fold, call, 0.5, 1.0,
 * allin] (first raise) = 5. We reserve 9 for safety against future
 * tier widening. */
#define ROLLOUT_MAX_ACTIONS_PER_NODE 9

/* Upper bound on the action history a per-leaf rollout receives. Used
 * by the rollout's internal game_history buffer (private to .c) and by
 * the per-leaf flop_history field on RolloutLeafInfo (public — must
 * match the ctypes mirror in python/rollout_c.py). 256 is more than
 * any 6-player postflop tree can produce (postflop max raises is 3 per
 * street, 6 players × 3 streets ≈ 24 actions). */
#define ROLLOUT_MAX_ACTION_HISTORY 256

/* ── Public structs ──────────────────────────────────────────────────── */

typedef struct {
    int32_t c0;
    int32_t c1;
    float   weight;
} RolloutHand;

typedef struct {
    const RolloutHand *hands;
    int32_t n_hands;
} RolloutHandList;

typedef struct {
    int32_t leaf_idx;                               /* canonical leaf index */
    int32_t pot_chips;
    int32_t bets[ROLLOUT_MAX_PLAYERS];              /* per compact slot */
    /* Per-leaf flop-action history (training-encoded indices, see
     * python/leaf_values.py::_train_action_idx). The rollout
     * concatenates [passed_in_preflop, leaf->flop_history] when
     * launching each per-leaf rollout so blueprint queries during the
     * subsequent street (turn/river) hash to the keys training stored.
     * Without this, the prefix is [preflop_only] → 100% miss → uniform
     * leaf values across hands → CFR converges to uniform 1/n. */
    int32_t flop_history[ROLLOUT_MAX_ACTION_HISTORY];
    int32_t flop_history_len;
} RolloutLeafInfo;

/* ── Pure-math exports (no RNG, no blueprint; for bit-exact unit tests) ── */

/* apply_bias: s[:n] * bias profile, renormalized. Matches
 * python/rollout.py::apply_bias. Output written to out[:n]. */
BPS_EXPORT void rollout_apply_bias_c(
    const double *strategy, int32_t n,
    int32_t bias_profile,
    const int32_t *categories,
    double *out
);

/* classify_postflop_actions: writes the action categories at this
 * decision node. Returns num_actions (1..ROLLOUT_MAX_ACTIONS_PER_NODE). */
BPS_EXPORT int32_t rollout_classify_postflop_actions_c(
    int32_t facing_bet, int32_t raises_so_far,
    int32_t *out_cats, int32_t cap
);

/* action_amount_chips: translates an action index to (kind, chip_amount).
 * kind is one of ROLLOUT_KIND_*. */
BPS_EXPORT void rollout_action_amount_chips_c(
    int32_t action_idx,
    int32_t facing_bet, int32_t raises_so_far,
    int32_t pot, int32_t to_call, int32_t stack,
    int32_t *out_kind, int32_t *out_amount
);

/* Winner-takes-all / chop showdown payoff (no side pots — matches
 * Python). Returns hero's net chip payoff. */
BPS_EXPORT float rollout_compute_showdown_payoff_c(
    const int *full_board,                /* 5 card ints */
    const int *hands_c0, const int *hands_c1,
    int32_t num_players,
    const int32_t *active,
    const int32_t *bets,
    int32_t pot_chips,
    int32_t hero_player
);

/* ── Rollout entry points ────────────────────────────────────────────── */

/* Single (leaf, bias_profile, hero, hero_hand) rollout mean.
 * out_value receives the mean payoff.
 * Returns 0 on success, -1 on genuine error (deck exhausted, board
 * overflow, bucket lookup failure, etc.) — see rollout_get_last_error.
 * Blueprint coverage misses are handled inline via uniform-on-legal-
 * actions (Pluribus Algorithm 1); call rollout_get_observability to
 * count substitutions. */
BPS_EXPORT int32_t rollout_compute_leaf_value_c(
    const BPSReader *r,
    /* leaf state */
    const int *board, int32_t n_board_cards,
    int32_t street,
    int32_t pot_chips,
    const int32_t *active, int32_t num_players,
    const int32_t *stacks,
    const int32_t *bets,
    const RolloutHandList *player_hands,        /* length num_players */
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,                /* compact → full seat */
    const int32_t *game_action_history, int32_t gah_len,
    /* query */
    const int32_t *bias_profile,                /* length num_players */
    int32_t hero_player,
    int32_t hero_hand_idx,
    int32_t num_rollouts,
    uint64_t master_seed,
    int32_t leaf_idx_for_seed,
    int32_t combo_for_seed,
    /* out */
    float *out_value
);

/* Multi-hand wrapper. out is float32[num_players * max_hands] row-major.
 * Returns 0 on success; -1 on genuine error (see single-hand entry).
 * Blueprint coverage misses are absorbed by uniform substitution.
 *
 * skip_first_betting: when 1, skip the betting loop on the leaf's
 * street and advance straight to the next street's chance/deal step.
 * Pass 1 from subgame leaf paths when the leaf's flop_history is
 * non-empty (the leaf already represents end-of-street state and the
 * betting we'd fictitiously redo would corrupt blueprint queries on
 * subsequent streets). See rollout.c::rollout_simulate_one. */
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
    int32_t skip_first_betting,
    float *out                                  /* [num_players * max_hands] */
);

/* Full-subgame wrapper — API-facing. Produces
 * float32[total * num_players * max_hands] where
 * total = (max leaf_idx in leaf_infos + 1) * cont_per_leaf
 * and cont_per_leaf = ROLLOUT_NUM_BIAS_PROFILES ^ num_players.
 *
 * Flat index: out[(leaf_infos[i].leaf_idx * cont_per_leaf + combo)
 *                  * num_players * max_hands
 *                + p * max_hands + h].
 *
 * out_size is the caller-provided capacity in floats; returns -1 if
 * smaller than required. */
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
);

/* Sampled-cont-strat variant (OP4 / Option c). For each betting leaf,
 * draws num_samples_k joint bias profiles uniformly from {0,1,2,3}^N
 * and averages leaf values across samples. Output shape is
 * float32[(max leaf_idx + 1) * num_players * max_hands] — NO
 * cont-strat axis. Pairs with the SAMPLED tree builder
 * (SS_CONT_STRAT_MODE == SAMPLED) where each betting leaf is a single
 * terminal.
 *
 * Flat index: out[li->leaf_idx * num_players * max_hands
 *                 + p * max_hands + h].
 *
 * See docs/OP4_CONT_STRAT_MARGINAL_PLAN.md. */
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
);

/* ── Error surface ───────────────────────────────────────────────────── */

BPS_EXPORT void rollout_get_last_error(
    const char **reason, int32_t *street, int32_t *num_players
);

/* Returns sizeof(RolloutLeafInfo). Used by the ctypes mirror in
 * python/rollout_c.py to assert layout parity at module load — if the C
 * struct grows or its field order shifts and the Python mirror isn't
 * updated, the per-leaf flop_history lands at the wrong offset and the
 * rollout silently misbehaves. */
BPS_EXPORT int32_t rollout_leaf_info_sizeof(void);

/* ── Observability counters (per-thread) ─────────────────────────────── */

/* Reset both counters to zero. Call before each subgame solve to scope
 * the counter readout. */
BPS_EXPORT void rollout_reset_observability(void);

/* Read counters. Pass NULL for either pointer to skip. Pluribus Algorithm
 * 1 supplementary substitutes uniform-on-legal-actions for missing or
 * zero-sum σ̄; the counters keep that substitution observable.
 *   *out_miss    — σ̄ entry not in blueprint (coverage gap; expected
 *                  outside the trained tree).
 *   *out_zerosum — σ̄ entry present but the truncated strategy summed
 *                  to zero (data integrity issue; should be rare). */
BPS_EXPORT void rollout_get_observability(
    uint64_t *out_miss, uint64_t *out_zerosum
);

/* Per-thread bucket-lookup miss counter. Bumped when the tolerate-miss
 * gate (see rollout_set_tolerate_bucket_miss) substitutes uniform for an
 * otherwise-fatal hand_to_bucket failure. Distinct from blueprint miss:
 * blueprint miss = bucket existed but no σ̄ stored; bucket miss = no
 * bucket assignment exists for the (board, hand) pair (e.g. paired-
 * flushdraw flop textures missing from texture_cache.bin). */
BPS_EXPORT void rollout_get_bucket_miss_count(uint64_t *out_bucket_miss);

/* Add to the per-thread observability counters. Used by Python
 * orchestrations to fold worker-process counter totals into the parent
 * thread's view after a multiprocessing.Pool fan-out (each worker has
 * its own thread-local copy; without this the parent stays at zero
 * despite real misses in workers). */
BPS_EXPORT void rollout_add_observability(
    uint64_t add_miss, uint64_t add_zerosum, uint64_t add_bucket_miss
);

/* ── Phase F: cont_ev_table consumer at end-of-flop multiway boundary ──
 *
 * Phase R's trainer populates a `cont_ev_table` (BPR5 trailer of the
 * regret-state file) with the traverser's expected showdown EV at the
 * end-of-flop multiway boundary, keyed by
 * (action_hash_at_boundary, traverser, hand_idx, traverser_cont_bias).
 * Opponents' biases are averaged out via uniform MCCFR sampling during
 * training (docs/PHASE_R_TRAINER_SPEC.md §3-4).
 *
 * When this consumer is enabled, the rollout short-circuits at the
 * end-of-flop multiway boundary instead of rolling out turn + river:
 * it computes the boundary action_hash, looks up the slot, and returns
 * the uniform mean over the 4 traverser-bias EVs as the leaf value.
 * The mean integrates over the same uniform bias distribution the
 * producer sampled at training — the matching consumer to producer.
 *
 * Boundary criteria (all must hold; otherwise legacy turn+river rollout):
 *   - flag enabled (rollout_set_use_cont_ev_at_flop_boundary)
 *   - cont_ev_table handle set (rollout_set_cont_ev_table)
 *   - street == ROLLOUT_STREET_FLOP at street_closed
 *   - >= 3 active players (multiway)
 *   - hero is one of the active players and has sampled hole cards
 *   - n_board == 3 (flop dealt, turn not yet)
 *
 * On any cont_ev_table query returning != BP_BOUNDARY_EV_OK
 * (EMPTY_BIAS / MISS / INVALID), the rollout fails loud — sets
 * rollout_last_error and returns -1. No silent NaN-papering, no
 * heuristic fallback (per docs/PHASE_R_TRAINER_SPEC.md §5 + the
 * project-wide no-shortcuts directive).
 *
 * Flag OFF (default) is byte-identical to the legacy code path. */

/* Set the cont_ev_table the rollout will query at the boundary. Pass
 * NULL to clear. The table is owned by the caller (typically a loaded
 * BPSolver's BPR5 trailer) and must outlive rollout calls. Process-
 * global; set once at startup before launching workers. Thread-safe
 * w.r.t. concurrent reads (the underlying find uses ACQUIRE loads). */
BPS_EXPORT void rollout_set_cont_ev_table(const BPContEVTable *t);
BPS_EXPORT const BPContEVTable *rollout_get_cont_ev_table(void);

/* Enable / disable the boundary lookup. When disabled (default), the
 * rollout runs legacy turn + river simulation regardless of whether
 * a cont_ev_table is set. Process-global; set once at startup. */
BPS_EXPORT void rollout_set_use_cont_ev_at_flop_boundary(int enabled);
BPS_EXPORT int  rollout_get_use_cont_ev_at_flop_boundary(void);

/* Per-thread observability counter: number of times the boundary
 * lookup was attempted and returned OK on all 4 biases (the success
 * path that short-circuited the legacy rollout). MISS / EMPTY_BIAS
 * fail loud and don't increment any counter — they raise via
 * rollout_set_error. Used for smoke-test reporting. */
BPS_EXPORT void rollout_get_cont_ev_boundary_hit_count(uint64_t *out_hits);
BPS_EXPORT void rollout_reset_cont_ev_boundary_hit_count(void);

/* ── Bucket-miss tolerance gate ─────────────────────────────────────── */

/* When enabled, a hand_to_bucket failure no longer aborts the rollout.
 * Instead, the call falls through to the same uniform-on-legal-actions
 * fallback used for missing-strategy info sets (Pluribus Algorithm 1
 * supplementary). The bucket-miss counter is bumped so the substitution
 * remains observable; callers can read it via rollout_get_bucket_miss_count.
 *
 * Off by default. Two ways to enable:
 *   1. Set env ROLLOUT_TOLERATE_BUCKET_MISS=1 before the .so loads (read
 *      lazily on first hot-path use; cached per process).
 *   2. Call rollout_set_tolerate_bucket_miss(1) at runtime (overrides env).
 *
 * Background: the texture-cache precompute in mccfr_blueprint.c has a
 * paired-flushdraw enumeration bug that omits ~1% of flop textures.
 * The .bps strategies for those textures don't exist either, so the
 * principled fallback (uniform per §C1) is the same fallback used at
 * any coverage gap. Calling ca_assign_buckets_kmeans inline is NOT a
 * viable alternative — it's a batch precompute (Lloyd's 20-iter on all
 * ~1081 hands per board with Monte Carlo EHS) that costs hundreds of
 * ms per board and produces bucket numbers indexed against textures
 * the .bps wasn't trained on. */
BPS_EXPORT void rollout_set_tolerate_bucket_miss(int enabled);
BPS_EXPORT int  rollout_get_tolerate_bucket_miss(void);

#ifdef __cplusplus
}
#endif

#endif /* ROLLOUT_H */
