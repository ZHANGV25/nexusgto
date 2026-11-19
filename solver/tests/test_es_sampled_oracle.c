/*
 * test_es_sampled_oracle.c - sampled leaf mode contracts.
 *
 * This test deliberately does NOT link rollout.c.  The exact-signature stub
 * below drives the real sampled-oracle policy logic with controlled
 * payoffs, while also checking that traversal hands and bias profiles reach
 * rollout_simulate_one unchanged.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "es_sampled_oracle.h"
#include "rollout_internal.h"

static int g_fail = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); g_fail++; \
    } \
} while (0)
#define CHECK_CLOSE(a, b, tol, ...) do { \
    double _d = fabs((double)(a) - (double)(b)); \
    if (_d > (tol)) { \
        printf("  FAIL (|%.9g - %.9g|=%.3g): ", \
               (double)(a), (double)(b), _d); \
        printf(__VA_ARGS__); printf("\n"); g_fail++; \
    } \
} while (0)

enum {
    STUB_DETERMINISTIC = 0,
    STUB_DOMINANT_BIAS3 = 1,
    STUB_MAPPING = 2
};

static int g_stub_mode = STUB_DETERMINISTIC;
static int g_stub_calls = 0;
static int g_stub_bad = 0;
static int g_expected_batch = 0;
static int g_expected_opponent_bias[3] = {-1, -1, -1};
static int g_pin_joint_bias = 0;
static int g_pinned_bias[3] = {0, 0, 0};

static uint64_t stub_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* Exact internal rollout primitive signature. */
int rollout_simulate_one(
    const BPSReader *r,
    const int *board_in, int32_t n_board_in,
    int32_t leaf_street, int32_t pot_in,
    const int32_t *active_in, const int32_t *stacks_in,
    const int32_t *bets_in, int32_t num_players,
    const int32_t *acting_order, int32_t acting_order_len,
    const int32_t *player_seats,
    const int32_t *game_history_in, int32_t gah_len_in,
    const int32_t *bias_profile, int32_t hero_player,
    const int32_t *sampled_hands_c0, const int32_t *sampled_hands_c1,
    const int32_t *has_sample, int32_t skip_first_betting,
    uint64_t *rng, float *out_payoff
) {
    (void)r;
    g_stub_calls++;
    if (n_board_in != 3 || board_in[0] != 10 || board_in[1] != 11
        || board_in[2] != 12 || leaf_street != 1 || pot_in != 600
        || num_players != 3 || acting_order_len != 3
        || acting_order[0] != 0 || acting_order[1] != 1
        || acting_order[2] != 2 || player_seats[0] != 1
        || player_seats[1] != 3 || player_seats[2] != 5
        || gah_len_in != 1 || game_history_in[0] != 7
        || skip_first_betting != 1
        || sampled_hands_c0[0] != 0 || sampled_hands_c1[0] != 1)
        g_stub_bad++;

    if (g_stub_mode != STUB_MAPPING) {
        if (!has_sample[0] || !has_sample[1] || !has_sample[2]
            || sampled_hands_c0[1] != 2 || sampled_hands_c1[1] != 3
            || sampled_hands_c0[2] != 4 || sampled_hands_c1[2] != 5)
            g_stub_bad++;
    } else {
        /* p1 is active but unassigned: no hand and blueprint bias 0.  p2 is
         * inactive but assigned: its folded cards remain dead, bias still 0. */
        if (!has_sample[0] || has_sample[1] || !has_sample[2]
            || sampled_hands_c0[2] != 4 || sampled_hands_c1[2] != 5
            || bias_profile[1] != 0 || bias_profile[2] != 0
            || active_in[0] != 1 || active_in[1] != 1 || active_in[2] != 0
            || stacks_in[0] != 900 || stacks_in[1] != 800
            || stacks_in[2] != 700 || bets_in[0] != 100
            || bets_in[1] != 200 || bets_in[2] != 300)
            g_stub_bad++;
    }

    if (g_pin_joint_bias
        && (bias_profile[0] != g_pinned_bias[0]
            || bias_profile[1] != g_pinned_bias[1]
            || bias_profile[2] != g_pinned_bias[2]))
        g_stub_bad++;

    if (g_expected_batch > 0) {
        int batch_pos = (g_stub_calls - 1) % g_expected_batch;
        for (int p = 0; p < num_players; p++) {
            if (p == hero_player) continue;
            if (batch_pos > 0
                && g_expected_opponent_bias[p] != bias_profile[p])
                g_stub_bad++;
            if (batch_pos == 0)
                g_expected_opponent_bias[p] = bias_profile[p];
        }
    }

    if (g_stub_mode == STUB_DOMINANT_BIAS3) {
        int visit_pos = (g_stub_calls - 1) % 8;
        if (bias_profile[hero_player] != visit_pos / 2) g_stub_bad++;
        *out_payoff = bias_profile[hero_player] == 3 ? 1.0f : 0.0f;
    } else {
        uint64_t x = stub_rng_next(rng);
        *out_payoff = (float)(bias_profile[0] * 16
                              + bias_profile[1] * 4
                              + bias_profile[2])
                      + (float)(x & 255U) / 256.0f;
    }
    return 0;
}

static void build_leaf_tree(SSTreeData *t, SSNode *node, int *children) {
    memset(t, 0, sizeof(*t));
    memset(node, 0, sizeof(*node));
    t->nodes = node;
    t->children = children;
    t->num_nodes = 1;
    t->num_players = 3;
    t->num_leaves = 1;
    t->num_board = 3;
    t->board[0] = 10; t->board[1] = 11; t->board[2] = 12;
    for (int p = 0; p < 3; p++) {
        t->num_hands[p] = 1;
        t->weights[p][0] = 1.0f;
        t->hands[p][0][0] = 2 * p;
        t->hands[p][0][1] = 2 * p + 1;
    }
    node->type = 4;  /* ES_T_LEAF */
    node->leaf_idx = 0;
    node->num_players = 3;
    node->pot = 600;
    for (int p = 0; p < 3; p++) node->active_players[p] = 1;
}

static void *make_ctx(const SSTreeData *t, int policy, int num_rollouts,
                      const int active[3]) {
    int pot[1] = {600};
    int bets[3] = {100, 200, 300};
    int hist[ROLLOUT_MAX_ACTION_HISTORY];
    int hist_len[1] = {1};
    int player_seats[3] = {1, 3, 5};
    memset(hist, 0, sizeof(hist));
    hist[0] = 7;
    return es_make_sampled_oracle(
        (const void *)(uintptr_t)1, t, 1, 1000, num_rollouts, policy,
        1, 1, pot, bets, active, hist, hist_len, player_seats);
}

static void reset_stub(int mode, int batch) {
    g_stub_mode = mode;
    g_stub_calls = 0;
    g_stub_bad = 0;
    g_expected_batch = batch;
    for (int p = 0; p < 3; p++) g_expected_opponent_bias[p] = -1;
    g_pin_joint_bias = 0;
}

static void test_determinism_policy(const SSTreeData *t, int policy) {
    printf("[test] policy %d same-seed payoff sequence\n", policy);
    int active[3] = {1, 1, 1};
    void *a = make_ctx(t, policy, 3, active);
    void *b = make_ctx(t, policy, 3, active);
    CHECK(a != NULL && b != NULL, "sampled contexts allocated");
    if (!a || !b) {
        es_free_sampled_oracle(a); es_free_sampled_oracle(b); return;
    }
    ESLeafOracle fn = es_sampled_oracle_fn();
    int sampled[3] = {0, 0, 0};
    uint64_t rng_a = UINT64_C(0x123456789abcdef0);
    uint64_t rng_b = UINT64_C(0x123456789abcdef0);
    for (int i = 0; i < 12; i++) {
        int visit_batch = policy == ES_SAMPLED_BIAS_REGRET_MATCHED ? 12 : 3;
        reset_stub(STUB_DETERMINISTIC, visit_batch);
        uint64_t expected_rng = rng_a;
        if (policy == ES_SAMPLED_BIAS_UNIFORM_FRESH) {
            g_pin_joint_bias = 1;
            for (int p = 0; p < 3; p++)
                g_pinned_bias[p] = (int)(stub_rng_next(&expected_rng) % 4U);
            for (int rollout = 0; rollout < 3; rollout++)
                (void)stub_rng_next(&expected_rng);
        }
        double va = fn(a, 0, 0, 0, sampled, &t->nodes[0], &rng_a);
        int bad_a = g_stub_bad;
        if (policy == ES_SAMPLED_BIAS_UNIFORM_FRESH)
            CHECK(rng_a == expected_rng,
                  "uniform policy visit %d passed-RNG endpoint", i);
        reset_stub(STUB_DETERMINISTIC, visit_batch);
        if (policy == ES_SAMPLED_BIAS_UNIFORM_FRESH) {
            g_pin_joint_bias = 1;
            uint64_t ignored = rng_b;
            for (int p = 0; p < 3; p++)
                g_pinned_bias[p] = (int)(stub_rng_next(&ignored) % 4U);
        }
        double vb = fn(b, 0, 0, 0, sampled, &t->nodes[0], &rng_b);
        CHECK(va == vb, "policy %d visit %d payoff mismatch %.9g vs %.9g",
              policy, i, va, vb);
        CHECK(rng_a == rng_b, "policy %d visit %d RNG mismatch", policy, i);
        CHECK(bad_a == 0 && g_stub_bad == 0,
              "policy %d visit %d rollout inputs/bias batch", policy, i);
    }
    uint64_t visits = 0, rollouts = 0, errors = 99;
    uint64_t uv = 0, ur = 0, rv = 0, rr = 0;
    es_sampled_oracle_stats(a, &visits, &rollouts, &errors);
    es_sampled_oracle_stats2(a, &uv, &ur, &rv, &rr);
    uint64_t expected_rollouts = policy == ES_SAMPLED_BIAS_REGRET_MATCHED
                                     ? 144 : 36;
    CHECK(visits == 12 && rollouts == expected_rollouts && errors == 0,
          "policy %d aggregate stats visits=%llu rollouts=%llu errors=%llu",
          policy, (unsigned long long)visits,
          (unsigned long long)rollouts, (unsigned long long)errors);
    if (policy == ES_SAMPLED_BIAS_UNIFORM_FRESH) {
        CHECK(uv == 12 && ur == 36 && rv == 0 && rr == 0,
              "uniform policy breakdown");
    } else {
        CHECK(uv == 0 && ur == 0 && rv == 12 && rr == 144,
              "regret policy breakdown");
    }
    es_free_sampled_oracle(a);
    es_free_sampled_oracle(b);
}

static void test_regret_matching(const SSTreeData *t) {
    printf("[test] policy 1 dominant-bias regret probability grows\n");
    int active[3] = {1, 1, 1};
    void *ctx = make_ctx(t, ES_SAMPLED_BIAS_REGRET_MATCHED, 2, active);
    CHECK(ctx != NULL, "regret context allocated");
    if (!ctx) return;
    ESLeafOracle fn = es_sampled_oracle_fn();
    int sampled[3] = {0, 0, 0};
    uint64_t rng = UINT64_C(0xfeedbeef12345678);
    reset_stub(STUB_DOMINANT_BIAS3, 8);
    double first = fn(ctx, 0, 0, 0, sampled, &t->nodes[0], &rng);
    double second = fn(ctx, 0, 0, 0, sampled, &t->nodes[0], &rng);
    /* Stub payoff is the indicator of traverser bias 3, so the returned
     * mixture is exactly that bias's current regret-matched probability. */
    CHECK_CLOSE(first, 0.25, 1e-7, "initial uniform bias-3 probability");
    CHECK_CLOSE(second, 1.0, 1e-7, "dominant bias-3 probability after update");
    CHECK(second > first, "dominant probability did not grow");
    CHECK(g_stub_bad == 0, "opponent bias changed within enumeration batch");

    /* Opponent rows are read but never updated. A different player's first
     * traverser visit must still start at the uniform 1/4 mixture. */
    double fresh_player = fn(ctx, 0, 1, 0, sampled, &t->nodes[0], &rng);
    CHECK_CLOSE(fresh_player, 0.25, 1e-7,
                "bias regrets leaked across player rows");
    CHECK(g_stub_bad == 0, "distinct-row enumeration/bias batch invalid");

    uint64_t visits = 0, rollouts = 0, errors = 99;
    uint64_t uv = 99, ur = 99, rv = 0, rr = 0;
    es_sampled_oracle_stats(ctx, &visits, &rollouts, &errors);
    es_sampled_oracle_stats2(ctx, &uv, &ur, &rv, &rr);
    CHECK(visits == 3 && rollouts == 24 && errors == 0,
          "aggregate stats visits=%llu rollouts=%llu errors=%llu",
          (unsigned long long)visits, (unsigned long long)rollouts,
          (unsigned long long)errors);
    CHECK(uv == 0 && ur == 0 && rv == 3 && rr == 24,
          "policy stats uv=%llu ur=%llu rv=%llu rr=%llu",
          (unsigned long long)uv, (unsigned long long)ur,
          (unsigned long long)rv, (unsigned long long)rr);
    es_free_sampled_oracle(ctx);
}

static void test_given_hand_mapping(const SSTreeData *t) {
    printf("[test] assigned folded cards stay dead; unassigned bias stays zero\n");
    int active[3] = {1, 1, 0};
    int sampled[3] = {0, -1, 0};
    void *ctx = make_ctx(t, ES_SAMPLED_BIAS_UNIFORM_FRESH, 1, active);
    CHECK(ctx != NULL, "mapping context allocated");
    if (!ctx) return;
    uint64_t rng = UINT64_C(0x1020304050607080);
    reset_stub(STUB_MAPPING, 1);
    double value = es_sampled_oracle_fn()(
        ctx, 0, 0, 0, sampled, &t->nodes[0], &rng);
    (void)value;
    CHECK(g_stub_calls == 1 && g_stub_bad == 0,
          "given-hand mapping/bias contract bad=%d calls=%d",
          g_stub_bad, g_stub_calls);
    es_free_sampled_oracle(ctx);
}

static void test_vec_guard(const SSTreeData *t) {
    printf("[test] sampled oracle rejects vector traverser mode\n");
    int active[3] = {1, 1, 1};
    void *ctx = make_ctx(t, ES_SAMPLED_BIAS_UNIFORM_FRESH, 1, active);
    ESResolver *r = es_create(t, es_sampled_oracle_fn(), ctx, 42);
    CHECK(r != NULL, "guard resolver allocated");
    if (r) {
        es_set_vec_mode(r, 1);
        int rc = es_run(r, 10, 1);
        const char *err = es_get_last_error();
        CHECK(rc == -1, "vec + sampled returned %d, expected -1", rc);
        CHECK(err && strstr(err, "sampled") != NULL,
              "vec guard error missing sampled-mode detail: %s",
              err ? err : "(null)");
        CHECK(es_iterations_run(r) == 0,
              "guarded run reported completed iterations");
        es_free(r);
    }
    es_free_sampled_oracle(ctx);
}

static void test_deadline(const SSTreeData *t) {
    printf("[test] 1ms deadline stops large scalar and Hogwild runs early\n");
    float table[3] = {0.0f, 0.0f, 0.0f};
    void *ctx = es_make_table_oracle(table, 3, 1);
    CHECK(ctx != NULL, "deadline table oracle allocated");
    if (!ctx) return;
    for (int threads = 1; threads <= 2; threads++) {
        ESResolver *r = es_create(t, es_table_oracle_fn(), ctx,
                                  (uint64_t)(123 + threads));
        CHECK(r != NULL, "deadline resolver allocated (threads=%d)", threads);
        if (!r) continue;
        const int64_t requested = INT64_C(1000000000);
        es_set_deadline_ms(r, 1);
        int rc = es_run(r, requested, threads);
        int64_t completed = es_iterations_run(r);
        CHECK(rc == 0, "deadline run failed rc=%d threads=%d", rc, threads);
        CHECK(completed >= 0 && completed < requested,
              "deadline completed %lld of %lld (threads=%d)",
              (long long)completed, (long long)requested, threads);
        es_free(r);
    }
    es_free_oracle_ctx(ctx);
}

int main(void) {
    SSTreeData tree;
    SSNode node;
    int children[1] = {0};
    build_leaf_tree(&tree, &node, children);

    printf("=== sampled leaf oracle tests ===\n");
    test_determinism_policy(&tree, ES_SAMPLED_BIAS_UNIFORM_FRESH);
    test_determinism_policy(&tree, ES_SAMPLED_BIAS_REGRET_MATCHED);
    test_regret_matching(&tree);
    test_given_hand_mapping(&tree);
    test_vec_guard(&tree);
    test_deadline(&tree);

    if (g_fail == 0) {
        printf("\nALL SAMPLED LEAF TESTS PASS\n");
        return 0;
    }
    printf("\n%d SAMPLED LEAF TEST(S) FAILED\n", g_fail);
    return 1;
}
