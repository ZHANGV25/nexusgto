/*
 * test_rollout_bias4.c - stratified B4 fill contracts.
 *
 * The all-hands rollout boundary is stubbed with a controlled two-player
 * continuation fixture. This isolates permutation/bin routing, averaging,
 * sample-seed uniqueness, and per-leaf range threading from blueprint I/O.
 */
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rollout.h"
#include "rollout_internal.h"

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

enum {
    STUB_INTERACTION = 0,
    STUB_CONDITIONING = 1,
    STUB_PERLEAF = 2
};

static int g_stub_mode = STUB_INTERACTION;
static int g_track_seed_slots = 0;
static int g_seed_slots[32768];
static int g_bad_seed_slot = 0;
static char g_error[512];
static int32_t g_error_street = 0;
static int32_t g_error_np = 0;

void rollout_set_error(int32_t street, int32_t num_players,
                       const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
    g_error_street = street;
    g_error_np = num_players;
}

void rollout_get_last_error(
    const char **reason, int32_t *street, int32_t *num_players
) {
    if (reason) *reason = g_error;
    if (street) *street = g_error_street;
    if (num_players) *num_players = g_error_np;
}

/* Exact public all-hands boundary used by rollout_sampled.c. */
int32_t rollout_compute_leaf_values_all_hands_c(
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
    float *out
) {
    (void)r; (void)board; (void)n_board_cards; (void)street;
    (void)pot_chips; (void)active; (void)stacks; (void)bets;
    (void)acting_order; (void)acting_order_len; (void)player_seats;
    (void)game_action_history; (void)gah_len; (void)num_rollouts;
    (void)master_seed; (void)leaf_idx_for_seed; (void)skip_first_betting;

    if (g_track_seed_slots) {
        #pragma omp critical(test_bias4_seed_slots)
        {
            if (combo_for_seed < 0
                || combo_for_seed >= g_track_seed_slots) {
                g_bad_seed_slot++;
            } else {
                g_seed_slots[combo_for_seed]++;
            }
        }
    }

    for (int p = 0; p < num_players; p++) {
        for (int h = 0; h < max_hands; h++) {
            float value;
            if (g_stub_mode == STUB_INTERACTION) {
                int b0 = bias_profile[0];
                int b1 = bias_profile[1];
                value = p == 0
                            ? (float)(7 * b0 + 3 * b1 + 2 * b0 * b1 + h)
                            : (float)(-(5 * b1 + 4 * b0 + b0 * b1) + h);
            } else if (g_stub_mode == STUB_CONDITIONING) {
                value = (float)(10 * p + h);
                if (bias_profile[p] == 2) value += 100.0f;
            } else {
                value = (float)player_hands[p].n_hands;
            }
            out[p * max_hands + h] = value;
        }
    }
    return 0;
}

static int run_bias4(const RolloutLeafInfo *leaf,
                     const RolloutHandList *shared_hands,
                     const RolloutHandList *per_leaf_hands,
                     int rounds, float *out) {
    int board[3] = {10, 11, 12};
    int32_t active[2] = {1, 1};
    int32_t order[2] = {0, 1};
    int32_t seats[2] = {0, 1};
    int32_t history[1] = {0};
    if (per_leaf_hands) {
        return rollout_compute_subgame_leaf_values_bias4_perleaf_c(
            NULL, board, 3, ROLLOUT_STREET_FLOP, leaf, 1,
            active, 2, shared_hands, per_leaf_hands,
            order, 2, seats, history, 0, 1000, 2, 1, rounds,
            UINT64_C(0x123456789abcdef0), out, 16);
    }
    return rollout_compute_subgame_leaf_values_bias4_c(
        NULL, board, 3, ROLLOUT_STREET_FLOP, leaf, 1,
        active, 2, shared_hands, order, 2, seats, history, 0,
        1000, 2, 1, rounds, UINT64_C(0x123456789abcdef0), out, 16);
}

static int run_sampled(const RolloutLeafInfo *leaf,
                       const RolloutHandList *hands,
                       int samples, float *out) {
    int board[3] = {10, 11, 12};
    int32_t active[2] = {1, 1};
    int32_t order[2] = {0, 1};
    int32_t seats[2] = {0, 1};
    int32_t history[1] = {0};
    return rollout_compute_subgame_leaf_values_sampled_c(
        NULL, board, 3, ROLLOUT_STREET_FLOP, leaf, 1,
        active, 2, hands, order, 2, seats, history, 0,
        1000, 2, 1, samples, UINT64_C(0x123456789abcdef0), out, 4);
}

static void test_bin_average_matches_sampled(
    const RolloutLeafInfo *leaf, const RolloutHandList *hands
) {
    printf("[test] B4 bin-average matches high-K sampled estimand\n");
    enum { ROUNDS = 2048, SAMPLES = ROUNDS * 4 };
    float bias4[16];
    float sampled[4];
    memset(bias4, 0, sizeof(bias4));
    memset(sampled, 0, sizeof(sampled));
    memset(g_seed_slots, 0, sizeof(g_seed_slots));
    g_bad_seed_slot = 0;
    g_stub_mode = STUB_INTERACTION;
    g_track_seed_slots = SAMPLES;

    CHECK(run_bias4(leaf, hands, NULL, ROUNDS, bias4) == 0,
          "bias4 interaction fill returned failure");
    g_track_seed_slots = 0;
    CHECK(g_bad_seed_slot == 0, "bias4 produced out-of-range sample seed slot");
    for (int sample = 0; sample < SAMPLES; sample++) {
        CHECK(g_seed_slots[sample] == 1,
              "sample seed slot %d visited %d times",
              sample, g_seed_slots[sample]);
    }
    CHECK(run_sampled(leaf, hands, SAMPLES, sampled) == 0,
          "sampled interaction fill returned failure");

    for (int p = 0; p < 2; p++) {
        for (int h = 0; h < 2; h++) {
            double mean = 0.0;
            for (int b = 0; b < 4; b++)
                mean += bias4[((b * 2 + p) * 2) + h];
            mean *= 0.25;
            CHECK_CLOSE(mean, sampled[p * 2 + h], 0.35,
                        "player=%d hand=%d bin mean vs sampled", p, h);
        }
    }
}

static void test_conditioning_routes_only_target_bin(
    const RolloutLeafInfo *leaf, const RolloutHandList *hands
) {
    printf("[test] conditioned strategy shift stays in its own bias bin\n");
    float out[16];
    memset(out, 0, sizeof(out));
    g_stub_mode = STUB_CONDITIONING;
    CHECK(run_bias4(leaf, hands, NULL, 16, out) == 0,
          "bias4 conditioning fill returned failure");

    for (int b = 0; b < 4; b++) {
        for (int p = 0; p < 2; p++) {
            for (int h = 0; h < 2; h++) {
                float expected = (float)(10 * p + h)
                                 + (b == 2 ? 100.0f : 0.0f);
                CHECK_CLOSE(out[((b * 2 + p) * 2) + h], expected, 1e-6,
                            "bias=%d player=%d hand=%d", b, p, h);
            }
        }
    }
}

static void test_perleaf_ranges(
    const RolloutLeafInfo *leaf, const RolloutHandList *hands
) {
    printf("[test] per-leaf entry threads its range row and fails on NULL\n");
    RolloutHandList per_leaf[2] = {{NULL, 7}, {NULL, 9}};
    float out[16];
    memset(out, 0, sizeof(out));
    g_stub_mode = STUB_PERLEAF;
    CHECK(run_bias4(leaf, hands, per_leaf, 2, out) == 0,
          "bias4 per-leaf fill returned failure");
    for (int b = 0; b < 4; b++) {
        for (int h = 0; h < 2; h++) {
            CHECK_CLOSE(out[((b * 2 + 0) * 2) + h], 7.0, 1e-6,
                        "player 0 per-leaf range");
            CHECK_CLOSE(out[((b * 2 + 1) * 2) + h], 9.0, 1e-6,
                        "player 1 per-leaf range");
        }
    }

    int board[3] = {10, 11, 12};
    int32_t active[2] = {1, 1};
    int32_t order[2] = {0, 1};
    int32_t seats[2] = {0, 1};
    int32_t history[1] = {0};
    int rc = rollout_compute_subgame_leaf_values_bias4_perleaf_c(
        NULL, board, 3, ROLLOUT_STREET_FLOP, leaf, 1,
        active, 2, hands, NULL, order, 2, seats, history, 0,
        1000, 2, 1, 1, 7, out, 16);
    CHECK(rc == -1, "NULL per-leaf range returned %d, expected -1", rc);
}

int main(void) {
    RolloutLeafInfo leaf;
    memset(&leaf, 0, sizeof(leaf));
    leaf.leaf_idx = 0;
    leaf.pot_chips = 200;
    leaf.bets[0] = 50;
    leaf.bets[1] = 50;

    RolloutHand hand_buf[2] = {{0, 1, 1.0f}, {2, 3, 1.0f}};
    RolloutHandList hands[2] = {
        {hand_buf, 2},
        {hand_buf, 2}
    };

    printf("=== stratified bias4 fill tests ===\n");
    test_bin_average_matches_sampled(&leaf, hands);
    test_conditioning_routes_only_target_bin(&leaf, hands);
    test_perleaf_ranges(&leaf, hands);

    if (g_fail == 0) {
        printf("\nALL BIAS4 FILL TESTS PASS\n");
        return 0;
    }
    printf("\n%d BIAS4 FILL TEST(S) FAILED\n", g_fail);
    return 1;
}
