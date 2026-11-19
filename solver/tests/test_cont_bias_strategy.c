/**
 * test_cont_bias_strategy — cont-bias reweighting tests for the
 * depth-limited trainer (mccfr_phase_r.c).
 *
 * Verifies the contract of bp_apply_cont_bias (Pluribus §C2 four biases,
 * ×5 multiplier on matching action class, L1-renormalize).
 *
 * Runs in <1 s. Build: `make test_cont_bias_strategy`.
 */

#include "mccfr_blueprint.h"

#include <stdio.h>
#include <math.h>

static int passed = 0;
static int failed = 0;

#define EPS 1e-5f

#define CHECK(cond, msg) do {                                              \
    if (cond) { passed++; }                                                \
    else {                                                                 \
        failed++;                                                          \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);      \
    }                                                                      \
} while (0)

#define CHECK_FLOAT_EQ(a, b, msg) do {                                     \
    float _aa = (a), _bb = (b);                                            \
    if (fabsf(_aa - _bb) < EPS) { passed++; }                              \
    else {                                                                 \
        failed++;                                                          \
        fprintf(stderr,                                                    \
                "FAIL %s:%d: %s (got %.6f, want %.6f, diff %.2e)\n",       \
                __FILE__, __LINE__, msg, _aa, _bb,                         \
                (double)fabsf(_aa - _bb));                                 \
    }                                                                      \
} while (0)

/* T1: bias=0 (baseline) — output equals input (already normalized). */
static void test_bias_zero_is_passthrough(void) {
    fprintf(stderr, "[T1] bias=0 reproduces input distribution\n");
    float avg[3] = {0.4f, 0.4f, 0.2f};
    int types[3] = {BP_ACT_FOLD, BP_ACT_CALL, BP_ACT_BET};
    float out[3];
    bp_apply_cont_bias(avg, types, 3, 0, out);
    CHECK_FLOAT_EQ(out[0], 0.4f, "out[0]=0.4");
    CHECK_FLOAT_EQ(out[1], 0.4f, "out[1]=0.4");
    CHECK_FLOAT_EQ(out[2], 0.2f, "out[2]=0.2");
}

/* T2: bias=1 (fold-bias) — fold gets ×5, others stay, renorm. */
static void test_fold_bias(void) {
    fprintf(stderr, "[T2] bias=1 (fold-bias) ×5 on fold action\n");
    float avg[3] = {0.4f, 0.4f, 0.2f};
    int types[3] = {BP_ACT_FOLD, BP_ACT_CALL, BP_ACT_BET};
    /* Pre-norm: [0.4*5, 0.4, 0.2] = [2.0, 0.4, 0.2], sum=2.6
     * Post-norm: [0.7692, 0.1538, 0.0769] */
    float out[3];
    bp_apply_cont_bias(avg, types, 3, 1, out);
    CHECK_FLOAT_EQ(out[0], 2.0f / 2.6f, "fold bumps to 2.0/2.6");
    CHECK_FLOAT_EQ(out[1], 0.4f / 2.6f, "call stays 0.4/2.6");
    CHECK_FLOAT_EQ(out[2], 0.2f / 2.6f, "bet stays 0.2/2.6");
    /* L1 norm. */
    CHECK_FLOAT_EQ(out[0] + out[1] + out[2], 1.0f, "renorm sums to 1");
}

/* T3: bias=2 (call/check-bias) — both CHECK and CALL get ×5. */
static void test_call_bias(void) {
    fprintf(stderr, "[T3] bias=2 (call/check-bias) ×5 on CHECK and CALL\n");

    /* Sub-T3a: action set with CALL only. */
    {
        float avg[3] = {0.4f, 0.4f, 0.2f};
        int types[3] = {BP_ACT_FOLD, BP_ACT_CALL, BP_ACT_BET};
        /* Pre-norm: [0.4, 0.4*5, 0.2] = [0.4, 2.0, 0.2], sum=2.6 */
        float out[3];
        bp_apply_cont_bias(avg, types, 3, 2, out);
        CHECK_FLOAT_EQ(out[0], 0.4f / 2.6f, "fold stays");
        CHECK_FLOAT_EQ(out[1], 2.0f / 2.6f, "call bumps");
        CHECK_FLOAT_EQ(out[2], 0.2f / 2.6f, "bet stays");
    }

    /* Sub-T3b: action set with CHECK (start of betting round). */
    {
        float avg[2] = {0.6f, 0.4f};
        int types[2] = {BP_ACT_CHECK, BP_ACT_BET};
        /* Pre-norm: [0.6*5, 0.4] = [3.0, 0.4], sum=3.4 */
        float out[2];
        bp_apply_cont_bias(avg, types, 2, 2, out);
        CHECK_FLOAT_EQ(out[0], 3.0f / 3.4f, "check bumps");
        CHECK_FLOAT_EQ(out[1], 0.4f / 3.4f, "bet stays");
    }
}

/* T4: bias=3 (raise/bet-bias) — BET gets ×5. */
static void test_raise_bias(void) {
    fprintf(stderr, "[T4] bias=3 (raise/bet-bias) ×5 on BET\n");
    float avg[3] = {0.4f, 0.4f, 0.2f};
    int types[3] = {BP_ACT_FOLD, BP_ACT_CALL, BP_ACT_BET};
    /* Pre-norm: [0.4, 0.4, 0.2*5] = [0.4, 0.4, 1.0], sum=1.8 */
    float out[3];
    bp_apply_cont_bias(avg, types, 3, 3, out);
    CHECK_FLOAT_EQ(out[0], 0.4f / 1.8f, "fold stays 0.4/1.8");
    CHECK_FLOAT_EQ(out[1], 0.4f / 1.8f, "call stays 0.4/1.8");
    CHECK_FLOAT_EQ(out[2], 1.0f / 1.8f, "bet bumps 1.0/1.8");
}

/* T5: 1-action info-set (only check) — all biases collapse to [1.0]. */
static void test_single_action(void) {
    fprintf(stderr, "[T5] 1-action info-set collapses to [1.0] under all biases\n");
    float avg[1] = {1.0f};
    int types[1] = {BP_ACT_CHECK};
    for (int bias = 0; bias < 4; bias++) {
        float out[1];
        bp_apply_cont_bias(avg, types, 1, bias, out);
        char msg[64];
        snprintf(msg, sizeof(msg), "bias=%d: out[0]=1.0", bias);
        CHECK_FLOAT_EQ(out[0], 1.0f, msg);
    }
}

/* T6: bias targets non-existent action class — distribution unchanged
 * (no action gets the boost; renorm leaves baseline). */
static void test_bias_on_absent_class(void) {
    fprintf(stderr, "[T6] bias on absent action class → baseline (no change)\n");
    /* No FOLD action (e.g., no bet to call yet). */
    float avg[2] = {0.6f, 0.4f};
    int types[2] = {BP_ACT_CHECK, BP_ACT_BET};
    float out[2];
    bp_apply_cont_bias(avg, types, 2, 1, out);   /* fold-bias on no FOLD */
    CHECK_FLOAT_EQ(out[0], 0.6f, "check unchanged");
    CHECK_FLOAT_EQ(out[1], 0.4f, "bet unchanged");
}

/* T7: zero-input degenerate fallback — all-zeros input with any bias
 * on an absent class → uniform fallback. */
static void test_degenerate_fallback(void) {
    fprintf(stderr, "[T7] all-zero input + bias on absent class → uniform\n");
    float avg[3] = {0.0f, 0.0f, 0.0f};
    int types[3] = {BP_ACT_CHECK, BP_ACT_CALL, BP_ACT_BET};
    float out[3];
    bp_apply_cont_bias(avg, types, 3, 1, out);   /* fold-bias on no FOLD */
    /* Sum is 0 → uniform fallback = 1/3 each. */
    CHECK_FLOAT_EQ(out[0], 1.0f / 3.0f, "uniform fallback [0]");
    CHECK_FLOAT_EQ(out[1], 1.0f / 3.0f, "uniform fallback [1]");
    CHECK_FLOAT_EQ(out[2], 1.0f / 3.0f, "uniform fallback [2]");
}

/* T8: input doesn't sum to 1 (treat as relative weights). */
static void test_unnormalized_input(void) {
    fprintf(stderr, "[T8] unnormalized input is treated as relative weights\n");
    float avg[3] = {2.0f, 2.0f, 1.0f};   /* sum=5, not 1 */
    int types[3] = {BP_ACT_FOLD, BP_ACT_CALL, BP_ACT_BET};
    float out[3];
    bp_apply_cont_bias(avg, types, 3, 0, out);
    /* Baseline: should renormalize to 0.4 / 0.4 / 0.2. */
    CHECK_FLOAT_EQ(out[0], 0.4f, "renorm to 0.4");
    CHECK_FLOAT_EQ(out[1], 0.4f, "renorm to 0.4");
    CHECK_FLOAT_EQ(out[2], 0.2f, "renorm to 0.2");
}

/* T9: out-of-range bias values are treated as baseline. */
static void test_invalid_bias(void) {
    fprintf(stderr, "[T9] out-of-range bias → baseline\n");
    float avg[3] = {0.5f, 0.3f, 0.2f};
    int types[3] = {BP_ACT_FOLD, BP_ACT_CALL, BP_ACT_BET};
    float out[3];
    /* bias=42 is out-of-range; the contract is "out-of-range produces baseline." */
    bp_apply_cont_bias(avg, types, 3, 42, out);
    CHECK_FLOAT_EQ(out[0], 0.5f, "OOR bias acts as baseline [0]");
    CHECK_FLOAT_EQ(out[1], 0.3f, "OOR bias acts as baseline [1]");
    CHECK_FLOAT_EQ(out[2], 0.2f, "OOR bias acts as baseline [2]");
    /* bias=-1 same. */
    bp_apply_cont_bias(avg, types, 3, -1, out);
    CHECK_FLOAT_EQ(out[0], 0.5f, "negative bias acts as baseline [0]");
    CHECK_FLOAT_EQ(out[1], 0.3f, "negative bias acts as baseline [1]");
    CHECK_FLOAT_EQ(out[2], 0.2f, "negative bias acts as baseline [2]");
}

int main(void) {
    fprintf(stderr,
            "═══════════════════════════════════════════════════════════\n"
            "  Depth-limit cont-bias: bp_apply_cont_bias unit tests\n"
            "  Contract: Pluribus §C2 (four biases, x5, L1 renormalize)\n"
            "═══════════════════════════════════════════════════════════\n");

    test_bias_zero_is_passthrough();
    test_fold_bias();
    test_call_bias();
    test_raise_bias();
    test_single_action();
    test_bias_on_absent_class();
    test_degenerate_fallback();
    test_unnormalized_input();
    test_invalid_bias();

    fprintf(stderr, "\n[test_cont_bias_strategy] passed=%d failed=%d\n",
            passed, failed);
    return failed > 0 ? 1 : 0;
}
