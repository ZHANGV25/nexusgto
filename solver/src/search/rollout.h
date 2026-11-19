/**
 * rollout.h — Pluribus-faithful depth-limited rollout leaf values (C port).
 *
 * Port of python/engine/rollout.py, function-for-function. The Python
 * module is the canonical correctness reference. This C port is the path
 * flop solves use; Python loads it via ctypes in
 * python/engine/rollout_c.py. RNG seeds follow a splitmix64 chain.
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
#include "mccfr_cont_ev_table.h"   /* BPContEVTable (boundary consumer) */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants — match python/engine/rollout.py ──────────────────────── */

#define ROLLOUT_NUM_ROLLOUTS_DEFAULT   3
#define ROLLOUT_BIAS_MULTIPLIER        5.0
#define ROLLOUT_NUM_BIAS_PROFILES      4
#define ROLLOUT_POSTFLOP_MAX_RAISES    3
#define ROLLOUT_MAX_PLAYERS            6

/* Postflop bet-vocab is RUNTIME state, not a compile-time constant. The
 * consumer MUST enumerate the SAME first/subsequent postflop sizes the loaded
 * blueprint was TRAINED with, or its action-index sequence hashes to a key
 * training never stored (all-in mis-key). A .bps manifest footer records them
 * (for example postflop_bet_sizes=[0.75], subsequent_bet_sizes=[1.0]);
 * the caller reads that footer and calls rollout_set_train_vocab() before any
 * rollout. Until it does, the runtime state DEFAULTS to the values compiled
 * below (ROLLOUT_FIRST_RAISE_DEFAULT_N / _FRACS), so a never-set process
 * keeps those fixed sizes (byte-identity invariant). See
 * rollout_set_train_vocab. */
/* Hard cap on the length of either size list (array sizing). NOTE: the
 * EFFECTIVE bound rollout_set_train_vocab enforces is TIGHTER:
 * ROLLOUT_MAX_ACTIONS_PER_NODE - 3 (= 6 with the current cap of 9). A facing
 * node emits [fold, call, <n sized raises>, allin] = n + 3 actions, which must
 * fit ROLLOUT_MAX_ACTIONS_PER_NODE; a vocab of 8 sizes would need 11 slots and
 * overflow the fixed per-node buffers. rollout_set_train_vocab rejects any
 * n_first/n_subseq > ROLLOUT_MAX_ACTIONS_PER_NODE - 3 fail-loud. This 8 only
 * bounds the g_rollout_*_fracs arrays; do not raise it above
 * ROLLOUT_MAX_ACTIONS_PER_NODE - 3 as a usable vocab length without also
 * enlarging the per-node action buffers. */
#define ROLLOUT_MAX_RAISE_SIZES        8   /* array cap; effective usable bound is ROLLOUT_MAX_ACTIONS_PER_NODE-3 */
/* Compiled defaults (the byte-identity baseline). NOT necessarily the loaded
 * blueprint's training vocab (for example [0.75]/[1.0]); an un-set process
 * uses these. */
#define ROLLOUT_FIRST_RAISE_DEFAULT_N  2   /* 0.5, 1.0 */
#define ROLLOUT_SUBSEQUENT_DEFAULT_N   1   /* 1.0 */

/* Action categories for the bias transform. Matches
 * python/engine/rollout.py::ActionCategory. CALL covers check; RAISE covers
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

/* Upper bound on action count at any single node: [fold, call] + up to
 * ROLLOUT_MAX_RAISE_SIZES sized raises + allin. 9 is reserved for safety;
 * a [0.75] vocab gives [fold, call, raise×0.75, allin] = 4, and the compiled
 * default [fold, call, 0.5, 1.0, allin] = 5. */
#define ROLLOUT_MAX_ACTIONS_PER_NODE 9

/* Upper bound on the action history a per-leaf rollout receives. Used
 * by the rollout's internal game_history buffer (private to .c) and by
 * the per-leaf flop_history field on RolloutLeafInfo (public — must
 * match the ctypes mirror in python/engine/rollout_c.py). 256 is more than
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
 * python/engine/rollout.py::apply_bias. Output written to out[:n]. */
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

/* ── Training bet-vocab setter ──────────────────────────────────────────
 *
 * Set the postflop bet vocabulary the rollout enumerates so its action-index
 * sequence matches what the loaded blueprint was TRAINED with. The caller
 * reads first_fracs/subseq_fracs from the loaded .bps manifest
 * (postflop_bet_sizes / subsequent_bet_sizes) and calls this ONCE at startup
 * before launching workers. Process-global; set-once.
 *
 * n_first  : number of first-raise (opening bet) pot fractions (1..MAX).
 * first_fracs[n_first]  : e.g. {0.75f}.
 * n_subseq : number of subsequent-raise pot fractions (1..MAX).
 * subseq_fracs[n_subseq]: e.g. {1.0f}.
 *
 * Returns 0 on success; -1 on a bad argument (NULL list, count <=0 or
 * > ROLLOUT_MAX_RAISE_SIZES) — sets rollout_last_error. FAIL-LOUD: never
 * silently truncates or falls back to constants.
 *
 * Until called, the runtime state DEFAULTS to the compiled values
 * ({0.5,1.0} first / {1.0} subsequent) (byte-identity invariant). */
BPS_EXPORT int32_t rollout_set_train_vocab(
    int32_t n_first, const float *first_fracs,
    int32_t n_subseq, const float *subseq_fracs
);

/* Read back the currently-configured training vocab (for the manifest-vs-
 * runtime assert and tests). Pass NULL for any out pointer to skip it.
 * out_first_fracs / out_subseq_fracs must have room for at least
 * ROLLOUT_MAX_RAISE_SIZES floats when non-NULL. */
BPS_EXPORT void rollout_get_train_vocab(
    int32_t *out_n_first, float *out_first_fracs,
    int32_t *out_n_subseq, float *out_subseq_fracs
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

/* Single-hand entry WITH skip_first_betting (subgame leaf path). Identical to
 * rollout_compute_leaf_value_c plus a skip_first_betting flag: pass 1 when the
 * leaf's game_action_history carries a closed flop_history (the leaf already
 * represents end-of-leaf-street betting state — re-running it double-counts
 * chips and corrupts subsequent-street blueprint queries). See the all-hands
 * wrapper's skip_first_betting doc below. Purely additive. */
BPS_EXPORT int32_t rollout_compute_leaf_value_skip_c(
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
    int32_t skip_first_betting,
    float *out_value
);

/* Multi-hand wrapper. out is float32[num_players * max_hands] row-major.
 * Returns 0 on success; -1 on genuine error (see single-hand entry).
 * Blueprint coverage misses are absorbed by uniform substitution.
 *
 * skip_first_betting: when 1, skip the betting loop on the leaf's
 * street and advance straight to the next street's chance/deal step.
 * Pass 1 from subgame leaf paths when the leaf's flop_history is
 * non-empty (the leaf already represents end-of-street state, and
 * fictitiously redoing its betting would corrupt blueprint queries on
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

/* Sampled-cont-strat variant. For each betting leaf,
 * draws num_samples_k joint bias profiles uniformly from {0,1,2,3}^N
 * and averages leaf values across samples. Output shape is
 * float32[(max leaf_idx + 1) * num_players * max_hands] — NO
 * cont-strat axis. Pairs with the SAMPLED tree builder
 * (SS_CONT_STRAT_MODE == SAMPLED) where each betting leaf is a single
 * terminal.
 *
 * Flat index: out[li->leaf_idx * num_players * max_hands
 *                 + p * max_hands + h]. */
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

/* Per-leaf conditioned variant of the SAMPLED construct. Identical
 * to rollout_compute_subgame_leaf_values_sampled_c except for the extra
 * `per_leaf_player_hands` array (n_leaves * num_players RolloutHandLists): leaf
 * li_idx samples opponents from &per_leaf_player_hands[li_idx*num_players], so the
 * villain's sampled-hand weights can be conditioned on the action line that
 * reached that leaf (the 2p flop/turn range conditioning — see
 * python/engine/leaf_villain_conditioning.py). With per_leaf_player_hands==NULL this is
 * byte-identical to the plain sampled_c (both delegate to rollout_sampled_core),
 * so the 3p+/5-6p shared-range path is provably unchanged. */
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
);

/* Stratified four-bias marginal construct. For each (leaf, round), draws one
 * independent permutation of {0,1,2,3} per player and runs four joint-profile
 * batches. Each player's batch value is accumulated into that player's
 * selected bias bin, so every [leaf][bias][player][hand] slot receives exactly
 * one sample per round while opponent biases remain uniform in expectation.
 *
 * Output shape is float32[max_leaf_idx + 1][4][num_players][max_hands].
 * Flat index:
 *   out[(((leaf_idx * 4 + bias) * num_players) + player) * max_hands
 *       + hand].
 */
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
);

/* Per-leaf conditioned-range variant of the stratified bias4 construct. The
 * per_leaf_player_hands array is [n_leaves][num_players], indexed by the
 * leaf_infos loop position exactly like the sampled per-leaf entry. */
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
);

/* ── Error surface ───────────────────────────────────────────────────── */

BPS_EXPORT void rollout_get_last_error(
    const char **reason, int32_t *street, int32_t *num_players
);

/* Returns sizeof(RolloutLeafInfo). Used by the ctypes mirror in
 * python/engine/rollout_c.py to assert layout parity at module load — if the C
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

/* ── cont_ev_table consumer at end-of-flop multiway boundary ──────────
 *
 * The blueprint trainer populates a `cont_ev_table` (BPR5 trailer of the
 * regret-state file) with the traverser's expected showdown EV at the
 * end-of-flop multiway boundary, keyed by
 * (action_hash_at_boundary, traverser, hand_idx, traverser_cont_bias).
 * Opponents' biases are averaged out via uniform MCCFR sampling during
 * training.
 *
 * When this consumer is enabled, the rollout short-circuits at the
 * end-of-flop multiway boundary instead of rolling out turn + river:
 * it computes the boundary action_hash, looks up the slot, and returns
 * the uniform mean over the 4 traverser-bias EVs as the leaf value.
 * The mean integrates over the same uniform bias distribution the
 * producer sampled at training — the matching consumer to producer.
 *
 * Boundary criteria (all must hold; otherwise live turn+river rollout):
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
 * heuristic fallback.
 *
 * Flag OFF (default) is byte-identical to the live-rollout code path. */

/* Set the cont_ev_table the rollout will query at the boundary. Pass
 * NULL to clear. The table is owned by the caller (typically a loaded
 * BPSolver's BPR5 trailer) and must outlive rollout calls. Process-
 * global; set once at startup before launching workers. Thread-safe
 * w.r.t. concurrent reads (the underlying find uses ACQUIRE loads). */
BPS_EXPORT void rollout_set_cont_ev_table(const BPContEVTable *t);
BPS_EXPORT const BPContEVTable *rollout_get_cont_ev_table(void);

/* SYNTHETIC mode (DEBUG / MEASUREMENT ONLY — default OFF). When ON, a
 * boundary MISS/EMPTY_BIAS is filled with a dummy finite EV (keyed identically
 * to the query) instead of failing loud, so the armed-RESIDUAL latency probe
 * can measure the cont_ev-armed wall-clock without a populated walk artifact.
 * Applies to BOTH the flop and turn boundaries. The real (synthetic OFF) path
 * is unchanged and still fail-loud. NEVER enable in a real solve:
 * synthetic EVs are dummies (0.0). */
BPS_EXPORT void rollout_set_cont_ev_synthetic(int enabled);
BPS_EXPORT int  rollout_get_cont_ev_synthetic(void);

/* Enable / disable the boundary lookup. When disabled (default), the
 * rollout runs live turn + river simulation regardless of whether
 * a cont_ev_table is set. Process-global; set once at startup. */
BPS_EXPORT void rollout_set_use_cont_ev_at_flop_boundary(int enabled);
BPS_EXPORT int  rollout_get_use_cont_ev_at_flop_boundary(void);

/* Per-thread observability counter: number of times the boundary
 * lookup was attempted and returned OK on all 4 biases (the success
 * path that short-circuited the live rollout). MISS / EMPTY_BIAS
 * fail loud and don't increment any counter — they raise via
 * rollout_set_error. Used for test reporting. */
BPS_EXPORT void rollout_get_cont_ev_boundary_hit_count(uint64_t *out_hits);
BPS_EXPORT void rollout_reset_cont_ev_boundary_hit_count(void);

/* Hit-rate denominator: number of times the flop-boundary predicate fired
 * and a table query was ATTEMPTED. Incremented before the lookup, so a MISS
 * (which fails loud) still counts here but not in the hit count. The reset
 * above clears both counters. hit_rate = hits / attempts. */
BPS_EXPORT void rollout_get_cont_ev_boundary_attempt_count(uint64_t *out_attempts);

/* ── Bias-as-hero-action consumer mode (mean-field) ─────────────────────
 *
 * Applies to BOTH boundaries (flop and
 * turn), on any cont_ev HIT. Independent of the flop/turn enable flags — this
 * only changes HOW an armed boundary combines its 4 stored per-bias EVs.
 *
 * Default 0 (OFF): return the uniform mean of the 4 stored EVs
 * (byte-identical to a consumer without this mode).
 *
 * 1 (ON) = the querying player's 4 stored per-bias EVs become a CFR-selectable
 * action at the leaf: the consumer regret-matches over the 4 EVs (uniform over
 * positive regret mass; 1/4 when none positive), returns the regret-matched
 * MIXTURE value (Σ p_b·EV_b), and updates the per-infoset regrets toward the
 * realized per-bias EVs (regret_b += EV_b − returned). All 4 EVs are already
 * in the slot — NO extra rollouts.
 *
 * PER-INFOSET, NOT PER-VISIT: the regret state persists across visits within a
 * solve (keyed by leaf slot key + querying player) so the mixture converges to
 * the best bias, and is RESET per solve via rollout_reset_cont_ev_bias_regrets.
 * The regret store is lazily allocated on the first hit in this mode and freed
 * at reset/teardown.
 *
 * SEMANTICS NOTE: whether this computes the true
 * mean-field value or only a best-response-to-uniform-opponents "half measure"
 * depends on the .cev the values were WALKED under (uniform vs mean_field). The
 * C layer only exposes the mode + a select-count stat; the PYTHON caller reads
 * the <cev>.cev.meta.json sidecar and LOGS which semantics ran (absent meta =>
 * uniform). Fail-loud on MISS/EMPTY_BIAS is unchanged.
 *
 * Process-global; set once at startup (like the enable flags). */
BPS_EXPORT void rollout_set_cont_ev_bias_action(int enabled);
BPS_EXPORT int  rollout_get_cont_ev_bias_action(void);

/* Number of bias-action mixtures returned since the last reset (per solve).
 * Zeroed by rollout_reset_cont_ev_bias_regrets. Pass NULL to skip. */
BPS_EXPORT void rollout_get_cont_ev_bias_action_stats(uint64_t *out_selects);

/* Free the persistent per-solve bias-regret store and zero the select
 * counter. Call between solves (never while a rollout is running) so the
 * regret state does not leak across solves — same lifecycle as the hit-count
 * reset. Idempotent; safe when the store was never allocated. */
BPS_EXPORT void rollout_reset_cont_ev_bias_regrets(void);

/* ── TURN-boundary consumer ────────────────────────────────────────────
 *
 * Mirror of the flop consumer above, one street later. When enabled, the
 * rollout short-circuits at the end-of-multiway-TURN boundary
 * (street == ROLLOUT_STREET_TURN, n_board == 4, >= 3 live) instead of
 * dealing the river: it queries the SAME shared cont_ev_table (set via
 * rollout_set_cont_ev_table) with the TURN street salt + the 4-card turn
 * hand index, and returns the uniform mean over the 4 biases. Same
 * fail-loud contract on MISS/EMPTY_BIAS/INVALID.
 *
 * INDEPENDENT of the flop flag — either, both, or neither may be enabled.
 * Default OFF; flag-OFF is byte-identical to the live river rollout.
 * SEPARATE turn hit/attempt counters keep the per-boundary hit-rate
 * unambiguous (the flop counters are NOT touched by the turn path). */
BPS_EXPORT void rollout_set_use_cont_ev_at_turn_boundary(int enabled);
BPS_EXPORT int  rollout_get_use_cont_ev_at_turn_boundary(void);
BPS_EXPORT void rollout_get_cont_ev_turn_boundary_hit_count(uint64_t *out_hits);
BPS_EXPORT void rollout_get_cont_ev_turn_boundary_attempt_count(uint64_t *out_attempts);
BPS_EXPORT void rollout_reset_cont_ev_turn_boundary_hit_count(void);

/* ── Pinned-tail MISS fallback (MEASUREMENT ONLY, default OFF) ──────────
 *
 * When enabled, a cont_ev boundary query whose slot is MISS/EMPTY_BIAS does
 * NOT fail loud: that ONE rollout falls through to the live playout
 * (deal next street + blueprint continuation), exactly as if the boundary
 * were not armed for that query. INVALID still fails loud (structural
 * corruption is never papered over). Applies to both boundaries.
 *
 * Why this is measurement-sound: the construct's RNG is per-(leaf, combo)
 * splitmix64 derived from master_seed (see compute_subgame_leaf_values_
 * sampled_c) — so with the SAME master_seed, a key that misses in BOTH
 * tables of an A/B produces a BIT-IDENTICAL fallback
 * value in both arms (the tail is "pinned" to common ground truth and
 * contributes zero to the measured delta). The uncovered tail can be a large
 * share of the key space (power-law, unreachable by traffic walking), so
 * without this mode such an A/B cannot run at all; with it, the delta cleanly
 * isolates covered-key semantics.
 *
 * Never enable this in a real solve. The solve contract remains fail-loud;
 * a hybrid fallback would pay the live rollout on every uncovered query and
 * destroy the armed latency win.
 * Per-boundary fallback counters make the substitution observable
 * (fallbacks = attempts − hits on this mode's runs). */
BPS_EXPORT void rollout_set_cont_ev_miss_fallback(int enabled);
BPS_EXPORT int  rollout_get_cont_ev_miss_fallback(void);
BPS_EXPORT void rollout_get_cont_ev_fallback_count(uint64_t *out_fallbacks);
BPS_EXPORT void rollout_get_cont_ev_turn_fallback_count(uint64_t *out_fallbacks);
BPS_EXPORT void rollout_reset_cont_ev_fallback_counts(void);

/* ── Per-spot exhaustive FILL mode (default OFF) ──────────────────────────
 *
 * MEASUREMENT/BUILD ONLY. When ON and a cont_ev boundary is armed, a boundary
 * query does NOT read/short-circuit: the slot is find_or_create'd under the
 * EXACT read-path key (full seat; TURN-salted hash at the turn boundary; the
 * INVALID hand-index case still fails loud — no bucket=-1 slot is ever
 * inserted) and THIS rollout runs the same live playout the
 * miss-fallback uses. The rollout's terminal hero-perspective payoff is then
 * atomically accumulated into ALL FOUR bias lanes of the slot (ev_sum[b] +=
 * payoff, ev_count[b] += 1 — same atomics as the trainer's boundary
 * accumulate) and the street's fill counter is bumped once per fill rollout.
 * `hits` is never incremented; `attempts` increments exactly as on the read
 * path, so attempts == fill_count on a pure fill pass.
 *
 * Precedence when multiple modes are set: FILL > synthetic > miss-fallback.
 * The driver arms ONE street per fill pass; a rollout crossing armed FLOP and
 * armed TURN boundaries in fill mode fails loud (driver-contract violation).
 * find_or_create is the trainer's lock-free CAS insert, and the accumulates
 * are the trainer's Hogwild atomics, so fill is safe under the construct's
 * OMP parallelism. Never enable in a real solve: the solve contract stays
 * read-only fail-loud. */
BPS_EXPORT void     rollout_set_cont_ev_fill(int enable);
BPS_EXPORT uint64_t rollout_get_cont_ev_fill_flop_count(void);
BPS_EXPORT uint64_t rollout_get_cont_ev_fill_turn_count(void);
BPS_EXPORT void     rollout_reset_cont_ev_fill_counts(void);

/* Test hook: the EXACT static turn hand-index the live consumer uses.
 * Exposed so the key-alignment test asserts against the real consumer
 * code path. board = 4 turn cards. Returns the canonical pair index in
 * [0,1326) or -1 (blocked/out-of-range). */
BPS_EXPORT int rollout_hand_index_for_turn_test(const int32_t *board,
                                                int32_t c0, int32_t c1);

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
 * principled fallback (uniform) is the same fallback used at
 * any coverage gap. Calling ca_assign_buckets_kmeans inline is NOT a
 * viable alternative — it's a batch precompute (Lloyd's 20-iter on all
 * ~1081 hands per board with Monte Carlo EHS) that costs hundreds of
 * ms per board and produces bucket numbers indexed against textures
 * the .bps wasn't trained on. */
BPS_EXPORT void rollout_set_tolerate_bucket_miss(int enabled);
BPS_EXPORT int  rollout_get_tolerate_bucket_miss(void);

/* Per-hand leaf-EV discriminator hooks.  These exports are diagnostic-only
 * and are never called by the solve's rollout path. */
BPS_EXPORT int32_t rollout_debug_sample_eager_hands_c(
    const int *board, int32_t n_board_cards,
    const int32_t *active, int32_t num_players,
    const RolloutHandList *player_hands,
    int32_t hero_player, int32_t hero_hand_idx,
    int32_t num_samples, uint64_t master_seed,
    int32_t leaf_idx_for_seed, int32_t combo_for_seed,
    int32_t *out_hands, uint64_t *out_rng_states,
    int32_t *out_attempts);

BPS_EXPORT int32_t rollout_debug_simulate_fixed_c(
    const BPSReader *r,
    const int *board, int32_t n_board_cards,
    int32_t street, int32_t pot_chips,
    const int32_t *active, int32_t num_players,
    const int32_t *stacks, const int32_t *bets,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_action_history, int32_t gah_len,
    const int32_t *bias_profile, int32_t hero_player,
    const int32_t *sampled_hands_c0,
    const int32_t *sampled_hands_c1,
    const int32_t *has_sample,
    int32_t skip_first_betting,
    uint64_t rng_seed,
    float *out_payoff, uint64_t *out_rng_state);

/* Current-flop five-card made-hand strength (category in bits 20+), using the
 * same eval5 implementation as rollout showdowns.  Higher is stronger. */
BPS_EXPORT uint32_t rollout_debug_flop_strength_c(
    const int *board3, int32_t c0, int32_t c1);

#ifdef __cplusplus
}
#endif

#endif /* ROLLOUT_H */
