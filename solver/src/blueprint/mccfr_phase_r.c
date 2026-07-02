/**
 * mccfr_phase_r.c — Phase R helpers.
 *
 * Pure dispatch logic + setter functions for the depth-limit + cont-strat
 * sampling extension of the trainer (mccfr_blueprint.c). Kept in a tiny
 * standalone translation unit so unit tests can link against just this
 * file + the BPSolver struct definition, without pulling the trainer's
 * card_abstraction / hand_eval / OMP dependency chain.
 *
 * Spec: docs/PHASE_R_TRAINER_SPEC.md
 *
 * Block A.1 deliverable. Block A.2 will fill in the HEADS-branch rollout
 * (cont-bias-reweighted σ̄ + EV slot accumulation) elsewhere.
 */

#include "mccfr_blueprint.h"
#include <stdio.h>
#include <stdint.h>

/* xorshift64 — kept identical to mccfr_blueprint.c::rng_next so that the
 * dispatch helper's RNG consumption pattern matches the trainer's per-
 * thread RNG state (same xorshift sequence). Inlined here so this TU is
 * self-contained. */
static inline uint64_t phase_r_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* Pure dispatch decision. See header for contract.
 *
 * Critical invariant: with flag == 0 we return 0 BEFORE touching
 * `*rng_state`. This is what makes the bit-identical regression test
 * (spec §7) work — flag-OFF runs are byte-equivalent to legacy because
 * the per-thread RNG sequence is unchanged. */
int bp_should_dispatch_to_boundary(int num_board, int n_active,
                                    int flag, float prob,
                                    uint64_t *rng_state) {
    /* Fast path: feature off → no RNG consumed, no behavior change. */
    if (flag == 0) return 0;

    /* Mirror of street_solve.cu:176-186 §4 dispatch, adapted to the
     * trainer (which has no concept of subgame-root-street; trainer
     * always starts preflop). The only boundary that produces cont-strat
     * EV slots the kernel will read is multiway flop end-of-street. */
    if (num_board != 3) return 0;        /* only end-of-flop is a boundary */
    if (n_active < 3) return 0;          /* HU flop continues to showdown */

    /* Multiway flop boundary with flag ON: coin flip. Probability `prob`
     * of HEADS (terminate; sample cont-bias; A.2 rollout populates EV
     * slot). 1 - prob of TAILS (continue legacy recursion past flop —
     * keeps multiway turn/river σ̄ co-evolving). */
    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;

    /* Same RNG consumption pattern as mccfr_blueprint.c::rng_float:
     * upper 24 bits → uniform float in [0, 1). One call per dispatch. */
    uint64_t x = phase_r_rng_next(rng_state);
    float u = (float)(x & 0xFFFFFF) / (float)0x1000000;
    return (u < prob) ? 1 : 0;
}

/* Dual-boundary build (2026-06-25) — TURN-boundary dispatch decision.
 *
 * The flop helper above hardcodes `num_board != 3 → 0` (and a test pins
 * that num_board==4 NEVER dispatches there). This is its TURN sibling, one
 * street later: it dispatches only at num_board == 4. Kept as a SEPARATE
 * function so the flop dispatch matrix + tests are byte-untouched.
 *
 * Same byte-identity invariant: flag == 0 returns 0 BEFORE touching
 * `*rng_state`, so a turn-flag-OFF run is byte-equivalent to legacy. The
 * RNG consumption pattern (one draw, upper-24-bit uniform float) is
 * identical to the flop helper and to mccfr_blueprint.c::rng_float, so the
 * coin-flip is drawn from the same stream the trainer would otherwise use. */
int bp_should_dispatch_to_turn_boundary(int num_board, int n_active,
                                        int flag, float prob,
                                        uint64_t *rng_state) {
    /* Fast path: feature off → no RNG consumed, no behavior change. */
    if (flag == 0) return 0;

    if (num_board != 4) return 0;        /* only end-of-turn is a turn boundary */
    if (n_active < 3) return 0;          /* HU turn continues to showdown */

    if (prob <= 0.0f) return 0;
    if (prob >= 1.0f) return 1;

    uint64_t x = phase_r_rng_next(rng_state);
    float u = (float)(x & 0xFFFFFF) / (float)0x1000000;
    return (u < prob) ? 1 : 0;
}

int bp_set_depth_limit_cont_strats_turn(BPSolver *s, int enabled) {
    if (s == NULL) return -1;
    s->depth_limit_cont_strats_turn = enabled ? 1 : 0;
    printf("[BP] depth_limit_cont_strats_turn = %d (%s)\n",
           s->depth_limit_cont_strats_turn,
           s->depth_limit_cont_strats_turn
               ? "Phase R: multiway-TURN boundary dispatch active"
               : "legacy: no turn-boundary dispatch");

    /* Lazily allocate the cont-ev table on first enable, exactly like the
     * flop setter. Idempotent: if the flop path already allocated it, leave
     * it. The table is SHARED across both boundaries (street disambiguation
     * is in the key-hash, not in separate tables). */
    if (enabled && s->cont_ev_table.entries == NULL) {
        int64_t target_size = (s->cont_ev_table_size_hint > 0)
            ? s->cont_ev_table_size_hint
            : (int64_t)(1LL << 26);
        if (bp_cont_ev_init(&s->cont_ev_table, target_size) != 0) {
            fprintf(stderr,
                    "[BP] WARNING: bp_cont_ev_init(%lld) failed; "
                    "depth_limit_cont_strats_turn will dispatch but no "
                    "slots will accumulate\n", (long long)target_size);
            return -1;
        }
        printf("[BP] cont_ev_table allocated (turn enable): %lld slots "
               "(~%.1f MB)\n",
               (long long)s->cont_ev_table.table_size,
               (double)s->cont_ev_table.table_size *
                   (double)sizeof(BPContEVEntry) / (1024.0 * 1024.0));
    }
    return 0;
}

int bp_set_depth_limit_cont_strats(BPSolver *s, int enabled) {
    if (s == NULL) return -1;
    s->depth_limit_cont_strats = enabled ? 1 : 0;
    printf("[BP] depth_limit_cont_strats = %d (%s)\n",
           s->depth_limit_cont_strats,
           s->depth_limit_cont_strats
               ? "Phase R: multiway-flop boundary dispatch active"
               : "legacy: full-game-depth MCCFR");

    /* Allocate the cont-ev table on first enable. Idempotent: if the
     * table is already allocated, leave it alone. If enabled=0, leave
     * the existing table allocated (so toggling on/off mid-run doesn't
     * lose accumulated EVs). bp_free handles teardown. */
    if (enabled && s->cont_ev_table.entries == NULL) {
        /* Spec §4 default: 1<<26 ≈ 67M slots. Tests can override via
         * bp_set_cont_ev_table_size before this call. */
        int64_t target_size = (s->cont_ev_table_size_hint > 0)
            ? s->cont_ev_table_size_hint
            : (int64_t)(1LL << 26);
        if (bp_cont_ev_init(&s->cont_ev_table, target_size) != 0) {
            fprintf(stderr,
                    "[BP] WARNING: bp_cont_ev_init(%lld) failed; "
                    "depth_limit_cont_strats will dispatch but no slots "
                    "will accumulate\n", (long long)target_size);
            return -1;
        }
        printf("[BP] cont_ev_table allocated: %lld slots (~%.1f MB)\n",
               (long long)s->cont_ev_table.table_size,
               (double)s->cont_ev_table.table_size *
                   (double)sizeof(BPContEVEntry) / (1024.0 * 1024.0));
    }
    return 0;
}

int bp_set_cont_ev_table_size(BPSolver *s, int64_t size_hint) {
    if (s == NULL) return -1;
    if (size_hint < 0) return -1;
    s->cont_ev_table_size_hint = size_hint;
    if (size_hint > 0) {
        printf("[BP] cont_ev_table_size_hint = %lld\n", (long long)size_hint);
    }
    return 0;
}

/* Stage 1 (2026-07-01) — mean-field opponent-bias-as-action mode setter.
 * See docs/STAGE1_MEANFIELD_PLAN_2026-07-01.md Unit 2 item 2. */
int bp_set_cont_ev_bias_mode(BPSolver *s, int mode) {
    if (s == NULL) return -1;
    if (mode != BP_BIAS_MODE_UNIFORM && mode != BP_BIAS_MODE_MEANFIELD) {
        fprintf(stderr,
                "[BP] bp_set_cont_ev_bias_mode: unknown mode=%d "
                "(expected %d=UNIFORM or %d=MEANFIELD)\n",
                mode, BP_BIAS_MODE_UNIFORM, BP_BIAS_MODE_MEANFIELD);
        return -1;
    }
    s->cont_ev_bias_mode = mode;
    printf("[BP] cont_ev_bias_mode = %d (%s)\n",
           s->cont_ev_bias_mode,
           s->cont_ev_bias_mode == BP_BIAS_MODE_MEANFIELD
               ? "Stage 1: opponents sample bias by regret matching"
               : "legacy: opponents' bias drawn uniformly (rng_int)");

    /* MEANFIELD: lazily allocate the bias-regret sidecar to the same
     * requested capacity as the cont_ev table (idempotent — leave an
     * existing table alone so toggling doesn't lose accumulated regrets).
     * UNIFORM: no allocation; the flag-OFF path never touches the sidecar. */
    if (mode == BP_BIAS_MODE_MEANFIELD &&
        s->bias_regrets_table.entries == NULL) {
        int64_t target_size = (s->cont_ev_table_size_hint > 0)
            ? s->cont_ev_table_size_hint
            : (int64_t)(1LL << 26);
        if (bp_bias_regrets_init(&s->bias_regrets_table, target_size) != 0) {
            fprintf(stderr,
                    "[BP] ERROR: bp_bias_regrets_init(%lld) failed; "
                    "mean-field bias mode requested but the sidecar could "
                    "not be allocated. Refusing (fail-loud).\n",
                    (long long)target_size);
            s->cont_ev_bias_mode = BP_BIAS_MODE_UNIFORM;
            return -1;
        }
        printf("[BP] bias_regrets_table allocated: %lld slots (~%.1f MB)\n",
               (long long)s->bias_regrets_table.table_size,
               (double)s->bias_regrets_table.table_size *
                   (double)sizeof(BPBiasRegretEntry) / (1024.0 * 1024.0));
    }
    return 0;
}

int bp_set_cont_strat_prob(BPSolver *s, float prob) {
    if (s == NULL) return -1;
    if (!(prob >= 0.0f && prob <= 1.0f)) {
        fprintf(stderr,
                "[BP] bp_set_cont_strat_prob: prob=%f out of [0, 1]\n",
                (double)prob);
        return -1;
    }
    s->cont_strat_prob = prob;
    printf("[BP] cont_strat_prob = %.3f\n", (double)prob);
    return 0;
}

/* Block A.2 — Pluribus §C2 cont-bias reweighting.
 *
 * For each action a, the reweighted weight is:
 *   - bias=1 (fold-bias):  5 * avg[a] if action_types[a] == BP_ACT_FOLD,
 *                          else avg[a]
 *   - bias=2 (call-bias):  5 * avg[a] if BP_ACT_CHECK or BP_ACT_CALL,
 *                          else avg[a]
 *   - bias=3 (raise-bias): 5 * avg[a] if BP_ACT_BET, else avg[a]
 *   - bias=0 (or out-of-range): baseline, no multiplier
 *
 * The result is renormalized to sum to 1. If all reweighted weights
 * sum to 0 (degenerate case: zero input + bias on absent class),
 * fall back to uniform.
 *
 * Tested in tests/test_cont_bias_strategy.c. */
void bp_apply_cont_bias(const float *avg_strategy,
                         const int *action_types,
                         int num_actions, int bias,
                         float *out) {
    if (num_actions <= 0) return;

    float sum = 0.0f;
    for (int a = 0; a < num_actions; a++) {
        float w = avg_strategy[a];
        int boost = 0;
        switch (bias) {
            case 1: /* fold-bias */
                boost = (action_types[a] == BP_ACT_FOLD);
                break;
            case 2: /* call/check-bias */
                boost = (action_types[a] == BP_ACT_CHECK ||
                         action_types[a] == BP_ACT_CALL);
                break;
            case 3: /* raise-bias */
                boost = (action_types[a] == BP_ACT_BET);
                break;
            default: /* bias=0 baseline (or out-of-range): no boost */
                boost = 0;
                break;
        }
        if (boost) w *= 5.0f;
        out[a] = w;
        sum += w;
    }

    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int a = 0; a < num_actions; a++) out[a] *= inv;
    } else {
        /* Degenerate fall-back: uniform. Hits when avg_strategy was
         * all zeros AND the bias targets a non-existent class — both
         * combine to leave every weight at 0. Mirrors regret_match's
         * uniform fallback at mccfr_blueprint.c:840. */
        float u = 1.0f / (float)num_actions;
        for (int a = 0; a < num_actions; a++) out[a] = u;
    }
}
