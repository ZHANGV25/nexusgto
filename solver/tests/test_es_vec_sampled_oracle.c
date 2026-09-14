/*
 * test_es_vec_sampled_oracle.c - vec-coherent per-visit sampled leaf oracle
 * contracts.
 *
 * rollout.c is NOT linked: the exact-signature stub below records what
 * reaches rollout_simulate_one_ex (joint bias profile, pinned runout, hero
 * cards) and returns controlled payoffs, so the tests can assert
 *   - COHERENCE: every hand valued in one vector visit sees the SAME
 *     opponent-bias realization and the SAME pinned runout, and no hand draws
 *     its own world (the traversal RNG advances by a per-visit amount only);
 *   - DETERMINISM: same RNG seed => bit-identical values and RNG endpoint;
 *   - ENUMERATE regret matching converges on a dominant bias;
 *   - SAMPLE mode is unbiased and also converges on a dominant bias;
 *   - the resolver hook: vec_mode=1 runs, is single-thread deterministic,
 *     and vec_mode=0 with a registered vector oracle fails loud.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "es_vec_sampled_oracle.h"
#include "es_resolver_internal.h"
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

enum { STUB_RECORD = 0, STUB_HERO_BIAS3 = 1, STUB_LINEAR = 2 };

#define STUB_MAX_CALLS 4096
typedef struct {
    int32_t bias[SS_MAX_PLAYERS];
    int32_t pinned[8];
    int32_t pinned_len;
    int32_t hero_c0, hero_c1;
    int32_t hero_player;
    int32_t opp_c0, opp_c1, opp_has;
    uint64_t rng_in;
} StubCall;

static int g_stub_mode = STUB_RECORD;
static int g_stub_calls = 0;
static int g_stub_bad = 0;
static StubCall g_calls[STUB_MAX_CALLS];

static uint64_t stub_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

int rollout_simulate_one_ex(
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
    const int32_t *pinned_runout, int32_t pinned_len,
    uint64_t *rng, float *out_payoff
) {
    (void)r; (void)stacks_in; (void)acting_order_len; (void)player_seats;
    (void)bets_in; (void)active_in;
    if (g_stub_calls < STUB_MAX_CALLS) {
        StubCall *c = &g_calls[g_stub_calls];
        memset(c, 0, sizeof(*c));
        for (int p = 0; p < num_players; p++) c->bias[p] = bias_profile[p];
        c->pinned_len = pinned_len;
        for (int k = 0; k < pinned_len && k < 8; k++)
            c->pinned[k] = pinned_runout[k];
        c->hero_c0 = sampled_hands_c0[hero_player];
        c->hero_c1 = sampled_hands_c1[hero_player];
        c->hero_player = hero_player;
        if (num_players == 2) {
            int opp_slot = 1 - hero_player;
            c->opp_c0 = sampled_hands_c0[opp_slot];
            c->opp_c1 = sampled_hands_c1[opp_slot];
            c->opp_has = has_sample[opp_slot];
        }
        c->rng_in = *rng;
    }
    g_stub_calls++;
    if (n_board_in != 3 || board_in[0] != 10 || board_in[1] != 11
        || board_in[2] != 12 || leaf_street != 1 || pot_in != 600
        || num_players != 2 || acting_order[0] != 0 || acting_order[1] != 1
        || gah_len_in != 1 || game_history_in[0] != 7
        || skip_first_betting != 1 || !has_sample[hero_player])
        g_stub_bad++;
    for (int k = 0; k < pinned_len; k++) {
        int32_t c = pinned_runout[k];
        /* The oracle never pins a board card or an opponent's card. */
        if (c == 10 || c == 11 || c == 12) g_stub_bad++;
        for (int p = 0; p < num_players; p++) {
            if (p == hero_player || !has_sample[p]) continue;
            if (c == sampled_hands_c0[p] || c == sampled_hands_c1[p])
                g_stub_bad++;
        }
    }
    if (g_stub_mode == STUB_HERO_BIAS3) {
        *out_payoff = bias_profile[hero_player] == 3 ? 1.0f : 0.0f;
    } else if (g_stub_mode == STUB_LINEAR) {
        /* Hero bias b is worth b chips; opponent bias subtracts; runout
         * adds a deterministic hash so different worlds differ. */
        int opp = 1 - hero_player;
        float v = (float)bias_profile[hero_player] * 10.0f
                  - (float)bias_profile[opp] * 3.0f;
        for (int k = 0; k < pinned_len; k++) v += 0.01f * (float)pinned_runout[k];
        uint64_t x = stub_rng_next(rng);
        *out_payoff = v + (float)(x & 255U) / 256.0f;
    } else {
        uint64_t x = stub_rng_next(rng);
        *out_payoff = (float)(x & 1023U) / 1024.0f;
    }
    return 0;
}

/* Single-leaf 2p tree: 6 hands per player from a disjoint deck region. */
#define NH 6
static void build_leaf_tree(SSTreeData *t, SSNode *node, int *children) {
    memset(t, 0, sizeof(*t));
    memset(node, 0, sizeof(*node));
    t->nodes = node;
    t->children = children;
    t->num_nodes = 1;
    t->num_players = 2;
    t->num_leaves = 1;
    t->num_board = 3;
    t->board[0] = 10; t->board[1] = 11; t->board[2] = 12;
    for (int p = 0; p < 2; p++) {
        t->num_hands[p] = NH;
        for (int h = 0; h < NH; h++) {
            t->hands[p][h][0] = 20 + p * 12 + 2 * h;
            t->hands[p][h][1] = 21 + p * 12 + 2 * h;
            t->weights[p][h] = 1.0f;
        }
    }
    node->type = 4;  /* ES_T_LEAF */
    node->leaf_idx = 0;
    node->num_players = 2;
    node->pot = 600;
    node->active_players[0] = 1;
    node->active_players[1] = 1;
    node->bets[0] = 100; node->bets[1] = 100;
}

static void *make_ctx(const SSTreeData *t, int hero_mode, int runout_mode,
                      int nr, double eps) {
    int pot[1] = {600};
    int bets[2] = {100, 100};
    int active[2] = {1, 1};
    int hist[ROLLOUT_MAX_ACTION_HISTORY];
    int hist_len[1] = {1};
    int player_seats[2] = {1, 3};
    memset(hist, 0, sizeof(hist));
    hist[0] = 7;
    return es_make_vec_sampled_oracle(
        (const void *)(uintptr_t)1, t, 1, 1000, nr, hero_mode, runout_mode,
        eps, 1, NH, pot, bets, active, hist, hist_len, player_seats);
}

static void reset_stub(int mode) {
    g_stub_mode = mode;
    g_stub_calls = 0;
    g_stub_bad = 0;
}

static void test_coherence(const SSTreeData *t, int hero_mode) {
    printf("[test] hero_mode=%d: one visit = one world for every hand\n",
           hero_mode);
    void *ctx = make_ctx(t, hero_mode, ES_VS_RUNOUT_SHARED, 1, 0.05);
    CHECK(ctx != NULL, "context allocated");
    if (!ctx) return;
    ESLeafOracleVec fn = es_vec_sampled_oracle_vec_fn();
    int opp[2] = {-1, 2};
    int dead[NH] = {0, 0, 1, 0, 0, 0};
    double out[NH];
    uint64_t rng = UINT64_C(0x123456789abcdef0);
    reset_stub(STUB_RECORD);
    fn(ctx, 0, 0, opp, &t->nodes[0], &rng, NH, dead, out);
    int per_hand = hero_mode == ES_VS_HERO_BIAS_ENUMERATE ? 4 : 1;
    CHECK(g_stub_calls == 5 * per_hand,
          "expected %d rollouts for 5 live hands, got %d", 5 * per_hand,
          g_stub_calls);
    CHECK(g_stub_bad == 0, "stub saw a malformed rollout input (%d)",
          g_stub_bad);
    CHECK(out[2] == 0.0, "dead hand must be written as 0");
    for (int i = 1; i < g_stub_calls; i++) {
        CHECK(g_calls[i].bias[1] == g_calls[0].bias[1],
              "opponent bias changed within the visit (call %d)", i);
        CHECK(g_calls[i].pinned_len == 4
              && memcmp(g_calls[i].pinned, g_calls[0].pinned,
                        4 * sizeof(int32_t)) == 0,
              "pinned runout changed within the visit (call %d)", i);
    }
    /* Hero cards follow the hand index; opponent hand 2 is fixed. */
    int seen_hand[NH] = {0};
    for (int i = 0; i < g_stub_calls; i++) {
        for (int h = 0; h < NH; h++) {
            if (g_calls[i].hero_c0 == t->hands[0][h][0]
                && g_calls[i].hero_c1 == t->hands[0][h][1])
                seen_hand[h] = 1;
        }
    }
    CHECK(!seen_hand[2] && seen_hand[0] && seen_hand[1] && seen_hand[3]
          && seen_hand[4] && seen_hand[5], "hero hands valued != live set");
    if (hero_mode == ES_VS_HERO_BIAS_ENUMERATE) {
        /* Four biases per hand, common random numbers across the four. */
        for (int i = 0; i < g_stub_calls; i += 4) {
            for (int b = 0; b < 4; b++) {
                CHECK(g_calls[i + b].bias[0] == b,
                      "enumerate order broken at call %d", i + b);
                CHECK(g_calls[i + b].rng_in == g_calls[i].rng_in,
                      "CRN: bias %d of hand block %d used a new stream", b, i);
            }
        }
    }
    uint64_t lv = 0, hv = 0, ro = 0, er = 9, ns = 0;
    es_vec_sampled_oracle_stats(ctx, &lv, &hv, &ro, &er, &ns);
    CHECK(lv == 1 && hv == 5 && ro == (uint64_t)(5 * per_hand) && er == 0,
          "stats visits=%llu hands=%llu rollouts=%llu errors=%llu",
          (unsigned long long)lv, (unsigned long long)hv,
          (unsigned long long)ro, (unsigned long long)er);
    es_free_vec_sampled_oracle(ctx);
}

static void test_determinism(const SSTreeData *t, int hero_mode,
                             int runout_mode) {
    printf("[test] hero_mode=%d runout=%d: same seed => identical values\n",
           hero_mode, runout_mode);
    void *a = make_ctx(t, hero_mode, runout_mode, 2, 0.05);
    void *b = make_ctx(t, hero_mode, runout_mode, 2, 0.05);
    CHECK(a && b, "contexts allocated");
    if (!a || !b) { es_free_vec_sampled_oracle(a); es_free_vec_sampled_oracle(b); return; }
    ESLeafOracleVec fn = es_vec_sampled_oracle_vec_fn();
    uint64_t ra = UINT64_C(0xfeedbeef12345678), rb = ra;
    int dead[NH] = {0};
    for (int visit = 0; visit < 8; visit++) {
        int opp[2] = {-1, visit % NH};
        double oa[NH], ob[NH];
        reset_stub(STUB_LINEAR);
        fn(a, 0, 0, opp, &t->nodes[0], &ra, NH, dead, oa);
        int calls_a = g_stub_calls;
        uint64_t rng_draws_expected = ra;
        reset_stub(STUB_LINEAR);
        fn(b, 0, 0, opp, &t->nodes[0], &rb, NH, dead, ob);
        CHECK(calls_a == g_stub_calls, "visit %d rollout count differs", visit);
        CHECK(memcmp(oa, ob, sizeof(oa)) == 0, "visit %d values differ", visit);
        CHECK(ra == rb && rb == rng_draws_expected,
              "visit %d traversal RNG endpoint differs", visit);
        /* The traversal stream advances only by the per-visit draws: one
         * opponent-bias draw, the pinned-runout draws, one visit seed. */
    }
    es_free_vec_sampled_oracle(a);
    es_free_vec_sampled_oracle(b);
}

static void test_per_visit_rng_budget(const SSTreeData *t) {
    printf("[test] traversal RNG advances per visit, never per hand\n");
    void *ctx = make_ctx(t, ES_VS_HERO_BIAS_ENUMERATE, ES_VS_RUNOUT_FRESH, 1, 0.05);
    CHECK(ctx != NULL, "context allocated");
    if (!ctx) return;
    ESLeafOracleVec fn = es_vec_sampled_oracle_vec_fn();
    int opp[2] = {-1, 0};
    double out[NH];
    /* Count draws: FRESH runout => exactly 2 xorshift steps per visit
     * (opponent bias u, visit seed) regardless of how many hands are live. */
    for (int live = 1; live <= NH; live++) {
        int dead[NH];
        for (int h = 0; h < NH; h++) dead[h] = h >= live;
        uint64_t rng = UINT64_C(0x0123456789abcdef);
        uint64_t expect = rng;
        stub_rng_next(&expect); stub_rng_next(&expect);
        reset_stub(STUB_RECORD);
        fn(ctx, 0, 0, opp, &t->nodes[0], &rng, NH, dead, out);
        CHECK(rng == expect, "live=%d: traversal RNG drew per hand", live);
        CHECK(g_stub_calls == 4 * live, "live=%d: rollouts %d", live, g_stub_calls);
    }
    es_free_vec_sampled_oracle(ctx);
}

static void test_enumerate_regret(const SSTreeData *t) {
    printf("[test] ENUMERATE: dominant bias 3 probability grows to 1\n");
    void *ctx = make_ctx(t, ES_VS_HERO_BIAS_ENUMERATE, ES_VS_RUNOUT_SHARED, 1, 0.05);
    CHECK(ctx != NULL, "context allocated");
    if (!ctx) return;
    ESLeafOracleVec fn = es_vec_sampled_oracle_vec_fn();
    int opp[2] = {-1, 1};
    int dead[NH] = {0};
    double out[NH];
    uint64_t rng = UINT64_C(0x1111222233334444);
    reset_stub(STUB_HERO_BIAS3);
    fn(ctx, 0, 0, opp, &t->nodes[0], &rng, NH, dead, out);
    for (int h = 0; h < NH; h++)
        CHECK_CLOSE(out[h], 0.25, 1e-7, "hand %d initial uniform mixture", h);
    fn(ctx, 0, 0, opp, &t->nodes[0], &rng, NH, dead, out);
    for (int h = 0; h < NH; h++)
        CHECK_CLOSE(out[h], 1.0, 1e-7, "hand %d dominant bias after update", h);
    /* The other player's rows are untouched: its first visit is uniform. */
    int opp2[2] = {3, -1};
    fn(ctx, 0, 1, opp2, &t->nodes[0], &rng, NH, dead, out);
    for (int h = 0; h < NH; h++)
        CHECK_CLOSE(out[h], 0.25, 1e-7, "player 1 hand %d leaked regrets", h);
    CHECK(g_stub_bad == 0, "stub inputs");
    es_free_vec_sampled_oracle(ctx);
}

static void test_sample_mode(const SSTreeData *t) {
    printf("[test] SAMPLE: unbiased value estimate, converges on bias 3\n");
    double eps = 0.05;
    void *ctx = make_ctx(t, ES_VS_HERO_BIAS_SAMPLE, ES_VS_RUNOUT_SHARED, 1, eps);
    CHECK(ctx != NULL, "context allocated");
    if (!ctx) return;
    ESLeafOracleVec fn = es_vec_sampled_oracle_vec_fn();
    int opp[2] = {-1, 4};
    int dead[NH] = {0};
    double out[NH];
    uint64_t rng = UINT64_C(0x5555666677778888);
    reset_stub(STUB_HERO_BIAS3);
    /* First visits: rows are zero, sigma uniform, so each returned value is
     * either 0 (bias != 3 drawn) or sigma/q = 0.25 / 0.25 = 1. */
    fn(ctx, 0, 0, opp, &t->nodes[0], &rng, NH, dead, out);
    for (int h = 0; h < NH; h++)
        CHECK(out[h] == 0.0 || fabs(out[h] - 1.0) < 1e-6,
              "hand %d first-visit value %g not in {0, 1}", h, out[h]);
    /* Convergence: after many visits sigma(3) -> 1, q(3) = 1 - 0.75 eps.
     * Returned values are then 1/q(3) with prob q(3) and 0 otherwise, whose
     * mean is exactly 1 - the mixture value of the dominant bias. */
    double sum = 0.0; int n = 0; int zeros = 0;
    for (int visit = 0; visit < 600; visit++) {
        fn(ctx, 0, 0, opp, &t->nodes[0], &rng, NH, dead, out);
        if (visit >= 200) {
            for (int h = 0; h < NH; h++) {
                sum += out[h]; n++;
                if (out[h] == 0.0) zeros++;
            }
        }
    }
    double mean = sum / (double)n;
    double zero_frac = (double)zeros / (double)n;
    double q3 = 1.0 - 0.75 * eps;
    CHECK(fabs(mean - 1.0) < 0.05, "SAMPLE mean %.4f != 1.0", mean);
    CHECK(fabs(zero_frac - (1.0 - q3)) < 0.03,
          "SAMPLE exploration zero fraction %.4f vs %.4f", zero_frac, 1.0 - q3);
    for (int h = 0; h < NH; h++)
        CHECK(out[h] == 0.0 || fabs(out[h] - 1.0 / q3) < 1e-4,
              "hand %d converged value %g", h, out[h]);
    CHECK(g_stub_bad == 0, "stub inputs");
    es_free_vec_sampled_oracle(ctx);
}

static void test_bias_modes(const SSTreeData *t) {
    printf("[test] bias modes: opponent unbiased / uniform, fixed hero\n");
    CHECK(es_vec_sampled_oracle_set_bias_modes(NULL, 0, 0) == -1,
          "NULL context accepted");
    void *ctx = make_ctx(t, ES_VS_HERO_BIAS_ENUMERATE, ES_VS_RUNOUT_SHARED, 1, 0.05);
    CHECK(ctx != NULL, "context allocated");
    if (!ctx) return;
    CHECK(es_vec_sampled_oracle_set_bias_modes(ctx, 3, 0) == -1,
          "opponent mode 3 accepted");
    CHECK(es_vec_sampled_oracle_set_bias_modes(ctx, 0, 2) == -1,
          "hero_bias_fixed 2 accepted");
    ESLeafOracleVec fn = es_vec_sampled_oracle_vec_fn();
    int dead[NH] = {0};
    double out[NH];
    uint64_t rng = UINT64_C(0x0f0f0f0f12345678);

    /* UNBIASED: every opponent realization is the unbiased continuation. */
    CHECK(es_vec_sampled_oracle_set_bias_modes(ctx, ES_VS_OPP_BIAS_UNBIASED, 0) == 0,
          "set UNBIASED");
    int biased = 0;
    for (int visit = 0; visit < 32; visit++) {
        int opp[2] = {-1, visit % NH};
        reset_stub(STUB_RECORD);
        fn(ctx, 0, 0, opp, &t->nodes[0], &rng, NH, dead, out);
        for (int i = 0; i < g_stub_calls; i++)
            if (g_calls[i].bias[1] != 0) biased++;
    }
    CHECK(biased == 0, "UNBIASED: %d rollouts saw a biased opponent", biased);

    /* UNIFORM: all four opponent biases appear across visits. */
    CHECK(es_vec_sampled_oracle_set_bias_modes(ctx, ES_VS_OPP_BIAS_UNIFORM, 0) == 0,
          "set UNIFORM");
    int seen[4] = {0, 0, 0, 0};
    for (int visit = 0; visit < 64; visit++) {
        int opp[2] = {-1, visit % NH};
        reset_stub(STUB_RECORD);
        fn(ctx, 0, 0, opp, &t->nodes[0], &rng, NH, dead, out);
        if (g_stub_calls > 0 && g_calls[0].bias[1] >= 0 && g_calls[0].bias[1] < 4)
            seen[g_calls[0].bias[1]] = 1;
    }
    CHECK(seen[0] && seen[1] && seen[2] && seen[3],
          "UNIFORM: opponent biases seen %d%d%d%d", seen[0], seen[1], seen[2], seen[3]);

    /* HERO FIXED: one rollout per live hand at bias 0; the stub pays only
     * bias 3, so a learning hero would drift to 1.0 and a fixed hero stays 0. */
    CHECK(es_vec_sampled_oracle_set_bias_modes(ctx, ES_VS_OPP_BIAS_REGRET, 1) == 0,
          "set hero fixed");
    int opp[2] = {-1, 1};
    reset_stub(STUB_HERO_BIAS3);
    for (int visit = 0; visit < 3; visit++) {
        fn(ctx, 0, 0, opp, &t->nodes[0], &rng, NH, dead, out);
        for (int h = 0; h < NH; h++)
            CHECK_CLOSE(out[h], 0.0, 1e-9, "hero fixed visit %d hand %d", visit, h);
    }
    CHECK(g_stub_calls == 3 * NH, "hero fixed: %d rollouts for %d hand visits",
          g_stub_calls, 3 * NH);
    for (int i = 0; i < g_stub_calls; i++)
        CHECK(g_calls[i].bias[0] == 0, "hero fixed call %d used bias %d", i,
              g_calls[i].bias[0]);
    CHECK(g_stub_bad == 0, "stub inputs");
    es_free_vec_sampled_oracle(ctx);
}

static int opp_hand_index(const SSTreeData *t, const StubCall *call) {
    for (int h = 0; h < NH; h++)
        if (call->opp_c0 == t->hands[1][h][0] && call->opp_c1 == t->hands[1][h][1])
            return h;
    return -1;
}

static void test_root_range_opponents(const SSTreeData *t) {
    printf("[test] ROOT_RANGE: opponent leaf hands drawn per hand from the range\n");
    CHECK(es_vec_sampled_oracle_set_opp_hand_mode(NULL, 1) == -1,
          "NULL context accepted");
    /* FRESH runout: a pinned card may legitimately equal a drawn hand. */
    void *ctx = make_ctx(t, ES_VS_HERO_BIAS_SAMPLE, ES_VS_RUNOUT_FRESH, 1, 0.05);
    CHECK(ctx != NULL, "context allocated");
    if (!ctx) return;
    CHECK(es_vec_sampled_oracle_set_opp_hand_mode(ctx, 2) == -1, "mode 2 accepted");
    ESLeafOracleVec fn = es_vec_sampled_oracle_vec_fn();
    int dead[NH] = {0};
    double out[NH];
    uint64_t rng = UINT64_C(0x2468ace013579bdf);
    int opp[2] = {-1, 2};

    /* TRAVERSAL (default): every hand is valued against opponent hand 2. */
    reset_stub(STUB_RECORD);
    fn(ctx, 0, 0, opp, &t->nodes[0], &rng, NH, dead, out);
    for (int i = 0; i < g_stub_calls; i++)
        CHECK(opp_hand_index(t, &g_calls[i]) == 2,
              "TRAVERSAL call %d used opponent hand %d", i,
              opp_hand_index(t, &g_calls[i]));

    CHECK(es_vec_sampled_oracle_set_opp_hand_mode(ctx, ES_VS_OPP_HAND_ROOT_RANGE) == 0,
          "set ROOT_RANGE");
    int seen[NH] = {0};
    int distinct = 0;
    for (int visit = 0; visit < 8; visit++) {
        reset_stub(STUB_RECORD);
        fn(ctx, 0, 0, opp, &t->nodes[0], &rng, NH, dead, out);
        CHECK(g_stub_calls == NH, "ROOT_RANGE visit %d: %d rollouts", visit, g_stub_calls);
        for (int i = 0; i < g_stub_calls; i++) {
            const StubCall *call = &g_calls[i];
            int idx = opp_hand_index(t, call);
            CHECK(idx >= 0 && call->opp_has, "visit %d call %d opponent hand not in range",
                  visit, i);
            CHECK(call->opp_c0 != call->hero_c0 && call->opp_c0 != call->hero_c1
                  && call->opp_c1 != call->hero_c0 && call->opp_c1 != call->hero_c1,
                  "visit %d call %d opponent collides with hero", visit, i);
            if (idx >= 0 && !seen[idx]) { seen[idx] = 1; distinct++; }
        }
    }
    CHECK(distinct >= 4, "ROOT_RANGE drew only %d distinct opponent hands", distinct);
    CHECK(g_stub_bad == 0, "stub inputs");
    es_free_vec_sampled_oracle(ctx);
}

/* Two-node tree: root decision (p0: 2 actions) -> leaf 0 / leaf 1. */
typedef struct {
    SSTreeData tree;
    SSNode nodes[4];
    int children[4];
    int decision_nodes[1];
    int leaf_nodes[2];
} Fixture;

static void build_resolver_tree(Fixture *f) {
    memset(f, 0, sizeof(*f));
    SSTreeData *t = &f->tree;
    t->nodes = f->nodes;
    t->children = f->children;
    t->num_players = 2;
    t->num_board = 3;
    t->board[0] = 10; t->board[1] = 11; t->board[2] = 12;
    for (int p = 0; p < 2; p++) {
        t->num_hands[p] = NH;
        for (int h = 0; h < NH; h++) {
            t->hands[p][h][0] = 20 + p * 12 + 2 * h;
            t->hands[p][h][1] = 21 + p * 12 + 2 * h;
            t->weights[p][h] = 1.0f;
        }
    }
    t->starting_pot = 600;
    t->effective_stack = 1000;
    for (int i = 0; i < 3; i++) {
        SSNode *n = &f->nodes[i];
        n->player = -1; n->leaf_idx = -1; n->fold_player = -1;
        n->chance_node_idx = -1; n->first_child = -1;
        n->pot = 600; n->num_board = 3;
        n->board_cards[0] = 10; n->board_cards[1] = 11; n->board_cards[2] = 12;
        n->bets[0] = 100; n->bets[1] = 100;
        n->active_players[0] = 1; n->active_players[1] = 1;
        n->num_players = 2;
    }
    f->nodes[0].type = 0; f->nodes[0].player = 0;
    f->nodes[0].first_child = 0; f->nodes[0].num_children = 2;
    f->children[0] = 1; f->children[1] = 2;
    f->nodes[1].type = 4; f->nodes[1].leaf_idx = 0;
    f->nodes[2].type = 4; f->nodes[2].leaf_idx = 1;
    t->num_nodes = 3;
    t->num_children_total = 2;
    t->num_leaves = 2;
    f->decision_nodes[0] = 0;
    t->decision_node_indices = f->decision_nodes;
    t->num_decision_nodes = 1;
    f->leaf_nodes[0] = 1; f->leaf_nodes[1] = 2;
    t->leaf_node_indices = f->leaf_nodes;
    t->num_leaf_nodes = 2;
    t->subgame_root_street = 3;
}

static void *make_resolver_ctx(const SSTreeData *t, int hero_mode) {
    int pot[2] = {600, 600};
    int bets[4] = {100, 100, 100, 100};
    int active[4] = {1, 1, 1, 1};
    int hist[2 * ROLLOUT_MAX_ACTION_HISTORY];
    int hist_len[2] = {1, 1};
    int player_seats[2] = {1, 3};
    memset(hist, 0, sizeof(hist));
    hist[0] = 7; hist[ROLLOUT_MAX_ACTION_HISTORY] = 7;
    return es_make_vec_sampled_oracle(
        (const void *)(uintptr_t)1, t, 1, 1000, 1, hero_mode,
        ES_VS_RUNOUT_SHARED, 0.05, 2, NH, pot, bets, active, hist, hist_len,
        player_seats);
}

static void test_resolver_hook(void) {
    printf("[test] resolver hook: vec run deterministic, scalar run rejected\n");
    Fixture fa, fb;
    build_resolver_tree(&fa);
    build_resolver_tree(&fb);
    double sa[NH * 2], sb[NH * 2];
    for (int mode = 0; mode < 2; mode++) {
        void *ca = make_resolver_ctx(&fa.tree, mode);
        void *cb = make_resolver_ctx(&fb.tree, mode);
        CHECK(ca && cb, "resolver contexts allocated");
        if (!ca || !cb) return;
        ESResolver *ra = es_create(&fa.tree, es_vec_sampled_oracle_scalar_fn(), ca, 77);
        ESResolver *rb = es_create(&fb.tree, es_vec_sampled_oracle_scalar_fn(), cb, 77);
        CHECK(ra && rb, "resolvers allocated");
        if (!ra || !rb) return;
        es_set_vec_leaf_oracle(ra, es_vec_sampled_oracle_vec_fn(), ca);
        es_set_vec_leaf_oracle(rb, es_vec_sampled_oracle_vec_fn(), cb);
        /* Scalar traversal with a registered vector oracle must fail loud. */
        reset_stub(STUB_LINEAR);
        int rc = es_run(ra, 10, 1);
        const char *err = es_get_last_error();
        CHECK(rc == -1 && err && strstr(err, "vector leaf oracle") != NULL,
              "mode %d: scalar run with vector oracle rc=%d err=%s", mode, rc,
              err ? err : "(null)");
        es_set_vec_mode(ra, 1);
        es_set_vec_mode(rb, 1);
        es_set_cfr_mode(ra, 1);
        es_set_cfr_mode(rb, 1);
        reset_stub(STUB_LINEAR);
        rc = es_run(ra, 200, 1);
        int calls_a = g_stub_calls;
        CHECK(rc == 0, "mode %d: vec run rc=%d (%s)", mode, rc,
              es_get_last_error() ? es_get_last_error() : "");
        reset_stub(STUB_LINEAR);
        rc = es_run(rb, 200, 1);
        CHECK(rc == 0 && calls_a == g_stub_calls,
              "mode %d: second resolver rc=%d calls %d vs %d", mode, rc,
              calls_a, g_stub_calls);
        CHECK(calls_a > 0, "mode %d: no rollouts reached the stub", mode);
        int na = es_final_strategy(ra, 0, sa, NH * 2);
        int nb = es_final_strategy(rb, 0, sb, NH * 2);
        CHECK(na == nb && na > 0 && memcmp(sa, sb, sizeof(double) * (size_t)na) == 0,
              "mode %d: same-seed strategies differ (na=%d nb=%d)", mode, na, nb);
        ESStats st; es_get_stats(ra, &st);
        uint64_t lv = 0, hv = 0, ro = 0, er = 9, ns = 0;
        es_vec_sampled_oracle_stats(ca, &lv, &hv, &ro, &er, &ns);
        CHECK(er == 0 && lv > 0 && hv == (uint64_t)st.leaf_rollouts,
              "mode %d: oracle stats visits=%llu hands=%llu vs resolver leaf "
              "rollouts=%lld errors=%llu", mode, (unsigned long long)lv,
              (unsigned long long)hv, (long long)st.leaf_rollouts,
              (unsigned long long)er);
        /* Hogwild: runs without error, values finite. */
        reset_stub(STUB_LINEAR);
        rc = es_run(ra, 200, 4);
        CHECK(rc == 0, "mode %d: Hogwild run rc=%d", mode, rc);
        na = es_final_strategy(ra, 0, sa, NH * 2);
        for (int i = 0; i < na; i++)
            CHECK(isfinite(sa[i]), "mode %d: non-finite strategy entry %d", mode, i);
        es_free(ra); es_free(rb);
        es_free_vec_sampled_oracle(ca); es_free_vec_sampled_oracle(cb);
    }
}

static void test_pinned_next_card(void) {
    printf("[test] rollout_pinned_next_card skips used cards, then exhausts\n");
    int32_t pinned[4] = {5, 6, 7, 8};
    int32_t pos = 0;
    uint64_t used = (1ULL << 6) | (1ULL << 8);
    CHECK(rollout_pinned_next_card(pinned, 4, &pos, used) == 5, "first card");
    CHECK(rollout_pinned_next_card(pinned, 4, &pos, used) == 7, "skips 6");
    CHECK(rollout_pinned_next_card(pinned, 4, &pos, used) == -1, "exhausted");
    pos = 0;
    CHECK(rollout_pinned_next_card(NULL, 4, &pos, 0) == -1, "NULL sequence");
}

int main(void) {
    SSTreeData tree;
    SSNode node;
    int children[1] = {0};
    build_leaf_tree(&tree, &node, children);

    printf("=== vec-coherent sampled leaf oracle tests ===\n");
    test_pinned_next_card();
    test_coherence(&tree, ES_VS_HERO_BIAS_ENUMERATE);
    test_coherence(&tree, ES_VS_HERO_BIAS_SAMPLE);
    test_determinism(&tree, ES_VS_HERO_BIAS_ENUMERATE, ES_VS_RUNOUT_SHARED);
    test_determinism(&tree, ES_VS_HERO_BIAS_SAMPLE, ES_VS_RUNOUT_FRESH);
    test_per_visit_rng_budget(&tree);
    test_enumerate_regret(&tree);
    test_sample_mode(&tree);
    test_bias_modes(&tree);
    test_root_range_opponents(&tree);
    test_resolver_hook();

    if (g_fail == 0) {
        printf("\nALL VEC SAMPLED LEAF TESTS PASS\n");
        return 0;
    }
    printf("\n%d VEC SAMPLED LEAF TEST(S) FAILED\n", g_fail);
    return 1;
}
