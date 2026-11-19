/*
 * test_es_bias_oracle.c - table-backed bias-oracle fixed points.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "es_bias_oracle.h"

static int g_fail = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); g_fail++; \
    } \
} while (0)
#define CHECK_CLOSE(a, b, tol, ...) do { \
    double _delta = fabs((double)(a) - (double)(b)); \
    if (_delta > (tol)) { \
        printf("  FAIL (|%.9g - %.9g|=%.3g): ", \
               (double)(a), (double)(b), _delta); \
        printf(__VA_ARGS__); printf("\n"); g_fail++; \
    } \
} while (0)

static void test_dominant_marginal(void) {
    printf("[test] marginal dominant bias converges to max\n");
    float values[1 * 4 * 2 * 1];
    memset(values, 0, sizeof(values));
    const float hero[4] = {0.0f, 1.0f, 2.0f, 8.0f};
    for (int b = 0; b < 4; b++) values[b * 2] = hero[b];

    void *ctx = es_make_bias_oracle(
        values, ES_BIAS_MARGINAL, 2, 1, 1);
    CHECK(ctx != NULL, "marginal context allocated");
    if (!ctx) return;

    ESLeafOracle fn = es_bias_oracle_fn();
    int sampled[2] = {0, 0};
    SSNode node;
    memset(&node, 0, sizeof(node));
    double first = fn(ctx, 0, 0, 0, sampled, &node, NULL);
    double last = first;
    for (int visit = 0; visit < 512; visit++)
        last = fn(ctx, 0, 0, 0, sampled, &node, NULL);

    CHECK_CLOSE(first, 2.75, 1e-7, "first call uses uniform bias mixture");
    CHECK(last > 7.99 && last <= 8.0001,
          "dominant mixture %.9g did not approach max 8", last);

    float effective[2] = {0.0f, 0.0f};
    int got = es_bias_oracle_get_effective_table(
        ctx, values, effective, 2);
    CHECK(got == 2, "effective table returned %d, expected 2", got);
    CHECK(effective[0] > 7.99f && effective[0] <= 8.0001f,
          "effective dominant value %.9g", (double)effective[0]);

    (void)fn(ctx, 0, 0, 1, sampled, &node, NULL);
    uint64_t visits = 0, errors = 0;
    es_bias_oracle_stats(ctx, &visits, &errors);
    CHECK(visits == 514, "visits=%llu, expected 514",
          (unsigned long long)visits);
    CHECK(errors == 1, "errors=%llu, expected 1",
          (unsigned long long)errors);
    es_free_bias_oracle(ctx);
}

static void test_joint_matching_pennies(void) {
    printf("[test] JOINT2P matching pennies stays at mixed equilibrium\n");
    float values[16 * 2];
    for (int player1_bias = 0; player1_bias < 4; player1_bias++) {
        for (int player0_bias = 0; player0_bias < 4; player0_bias++) {
            int combo = player0_bias + 4 * player1_bias;
            float payoff0 =
                ((player0_bias & 1) == (player1_bias & 1))
                    ? 1.0f : -1.0f;
            values[combo * 2] = payoff0;
            values[combo * 2 + 1] = -payoff0;
        }
    }

    void *ctx = es_make_bias_oracle(
        values, ES_BIAS_JOINT2P, 2, 1, 1);
    CHECK(ctx != NULL, "JOINT2P context allocated");
    if (!ctx) return;

    ESLeafOracle fn = es_bias_oracle_fn();
    int sampled[2] = {0, 0};
    SSNode node;
    memset(&node, 0, sizeof(node));
    double value0 = 0.0, value1 = 0.0;
    for (int visit = 0; visit < 2048; visit++) {
        value0 = fn(ctx, 0, 0, 0, sampled, &node, NULL);
        value1 = fn(ctx, 0, 1, 0, sampled, &node, NULL);
    }
    CHECK_CLOSE(value0, 0.0, 1e-7,
                "player 0 matching-pennies equilibrium value");
    CHECK_CLOSE(value1, 0.0, 1e-7,
                "player 1 matching-pennies equilibrium value");

    /* Negative opponent hand is the documented uniform-mixture fallback. */
    int missing_opponent[2] = {0, -1};
    double fallback = fn(
        ctx, 0, 0, 0, missing_opponent, &node, NULL);
    CHECK_CLOSE(fallback, 0.0, 1e-7,
                "negative opponent hand uniform fallback");

    float effective[2] = {99.0f, 99.0f};
    int got = es_bias_oracle_get_effective_table(
        ctx, values, effective, 2);
    CHECK(got == 2, "JOINT effective table returned %d", got);
    CHECK_CLOSE(effective[0], 0.0, 1e-7,
                "JOINT reporting table player 0");
    CHECK_CLOSE(effective[1], 0.0, 1e-7,
                "JOINT reporting table player 1");
    es_free_bias_oracle(ctx);
}

static void test_joint_combo_decode_order(void) {
    printf("[test] JOINT2P player 0 is the low base-4 digit\n");
    float values[16 * 2];
    memset(values, 0, sizeof(values));
    for (int player1_bias = 0; player1_bias < 4; player1_bias++) {
        for (int player0_bias = 0; player0_bias < 4; player0_bias++) {
            int combo = player0_bias + 4 * player1_bias;
            values[combo * 2] =
                (float)(100 * player0_bias + 10 * player1_bias);
            values[combo * 2 + 1] =
                (float)(1000 * player1_bias + player0_bias);
        }
    }

    void *ctx = es_make_bias_oracle(
        values, ES_BIAS_JOINT2P, 2, 1, 1);
    CHECK(ctx != NULL, "decode-order context allocated");
    if (!ctx) return;
    int sampled[2] = {0, 0};
    SSNode node;
    memset(&node, 0, sizeof(node));
    double player0_first = es_bias_oracle_fn()(
        ctx, 0, 0, 0, sampled, &node, NULL);
    es_free_bias_oracle(ctx);

    ctx = es_make_bias_oracle(
        values, ES_BIAS_JOINT2P, 2, 1, 1);
    CHECK(ctx != NULL, "second decode-order context allocated");
    if (!ctx) return;
    double player1_first = es_bias_oracle_fn()(
        ctx, 0, 1, 0, sampled, &node, NULL);
    CHECK_CLOSE(player0_first, 165.0, 1e-6,
                "player 0 low-digit decode");
    CHECK_CLOSE(player1_first, 1501.5, 1e-5,
                "player 1 high-digit decode");
    es_free_bias_oracle(ctx);
}

static void test_factory_guards(void) {
    printf("[test] factory validation\n");
    float values[32] = {0};
    CHECK(es_make_bias_oracle(
              NULL, ES_BIAS_MARGINAL, 2, 1, 1) == NULL,
          "NULL values accepted");
    CHECK(es_make_bias_oracle(values, 7, 2, 1, 1) == NULL,
          "invalid mode accepted");
    CHECK(es_make_bias_oracle(
              values, ES_BIAS_JOINT2P, 3, 1, 1) == NULL,
          "JOINT2P np=3 accepted");
    CHECK(es_make_bias_oracle(
              values, ES_BIAS_MARGINAL, 2, 0, 1) == NULL,
          "max_h=0 accepted");
}

int main(void) {
    printf("=== bias table oracle tests ===\n");
    test_dominant_marginal();
    test_joint_matching_pennies();
    test_joint_combo_decode_order();
    test_factory_guards();

    if (g_fail == 0) {
        printf("\nALL BIAS ORACLE TESTS PASS\n");
        return 0;
    }
    printf("\n%d BIAS ORACLE TEST(S) FAILED\n", g_fail);
    return 1;
}
