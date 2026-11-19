/**
 * Regression coverage for betting-closed rollout leaves.
 *
 * The test is artifact-free.  All exercised paths reach chance/showdown
 * without querying a blueprint, so a NULL BPSReader is intentional.  Link-only
 * stubs at the bottom satisfy rollout.c's unused blueprint/continuation-EV
 * references and count any accidental call as a test failure.
 */
#define _POSIX_C_SOURCE 200809L

#include "rollout.h"
#include "rollout_bucket_memo.h"
#include "hand_eval.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define TEST_NUM_PLAYERS 2
#define TEST_POT 200
#define TEST_BET 100
#define TEST_MASTER_SEED 12ULL
#define TEST_CONT_PER_LEAF 16

static int passed = 0;
static int failed = 0;
static volatile int unexpected_dependency_calls = 0;

#define CHECK(cond, msg) do {                                           \
    if (cond) {                                                         \
        passed++;                                                       \
    } else {                                                            \
        failed++;                                                       \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
    }                                                                   \
} while (0)

#define CHECK_CLOSE(actual, expected, tolerance, msg) do {                 \
    float _actual = (float)(actual);                                        \
    float _expected = (float)(expected);                                    \
    float _tolerance = (float)(tolerance);                                  \
    if (fabsf(_actual - _expected) <= _tolerance) {                         \
        passed++;                                                           \
    } else {                                                                \
        failed++;                                                           \
        fprintf(stderr,                                                     \
                "FAIL %s:%d: %s (got %.9g, want %.9g, tol %.9g)\n",       \
                __FILE__, __LINE__, msg, _actual, _expected, _tolerance);   \
    }                                                                       \
} while (0)

/* Ah Kd 7c 2s; card = rank * 4 + suit. */
static const int turn_board[4] = {50, 45, 20, 3};
static const int river_board[5] = {50, 45, 20, 3, 21};

/* Hero QcQd versus equal-weight AsJc, KhJh, 8c8d, and Tc9c. */
static const RolloutHand hero_range[1] = {
    {40, 41, 1.0f}
};
static const RolloutHand opponent_range[4] = {
    {51, 36, 1.0f},
    {46, 38, 1.0f},
    {24, 25, 1.0f},
    {32, 28, 1.0f}
};

typedef struct {
    int32_t allin_rc;
    float allin_value;
    int32_t non_allin_rc;
    float non_allin_values[TEST_NUM_PLAYERS];
    int32_t unexpected_calls;
} ProbeResult;

typedef struct {
    float value;
    int outcomes;
    int wins;
    int losses;
    int ties;
} ExactResult;

static void init_hand_lists(RolloutHandList lists[TEST_NUM_PLAYERS]) {
    lists[0].hands = hero_range;
    lists[0].n_hands = 1;
    lists[1].hands = opponent_range;
    lists[1].n_hands = 4;
}

static int card_is_used(int card, const RolloutHand *opponent) {
    for (int i = 0; i < 4; i++) {
        if (turn_board[i] == card) return 1;
    }
    if (hero_range[0].c0 == card || hero_range[0].c1 == card) return 1;
    return opponent->c0 == card || opponent->c1 == card;
}

/* Enumerate every opponent combo and every legal river.  The four opponent
 * combos have equal weights, so the arithmetic mean is the exact leaf value. */
static ExactResult enumerate_exact_allin_value(void) {
    ExactResult result;
    double payoff_sum = 0.0;
    memset(&result, 0, sizeof(result));

    for (int h = 0; h < 4; h++) {
        const RolloutHand *opponent = &opponent_range[h];
        for (int river = 0; river < 52; river++) {
            if (card_is_used(river, opponent)) continue;

            int hero_cards[7] = {
                turn_board[0], turn_board[1], turn_board[2], turn_board[3],
                river, hero_range[0].c0, hero_range[0].c1
            };
            int opponent_cards[7] = {
                turn_board[0], turn_board[1], turn_board[2], turn_board[3],
                river, opponent->c0, opponent->c1
            };
            uint32_t hero_strength = eval7(hero_cards);
            uint32_t opponent_strength = eval7(opponent_cards);
            float payoff;

            if (hero_strength > opponent_strength) {
                result.wins++;
                payoff = (float)(TEST_POT - TEST_BET);
            } else if (hero_strength < opponent_strength) {
                result.losses++;
                payoff = -(float)TEST_BET;
            } else {
                result.ties++;
                payoff = (float)(TEST_POT / 2 - TEST_BET);
            }
            payoff_sum += (double)payoff;
            result.outcomes++;
        }
    }

    if (result.outcomes > 0) {
        result.value = (float)(payoff_sum / (double)result.outcomes);
    }
    return result;
}

static int compute_allin_value(float *out_value) {
    const int32_t active[TEST_NUM_PLAYERS] = {1, 1};
    const int32_t stacks[TEST_NUM_PLAYERS] = {0, 0};
    const int32_t bets[TEST_NUM_PLAYERS] = {TEST_BET, TEST_BET};
    const int32_t acting_order[TEST_NUM_PLAYERS] = {0, 1};
    const int32_t player_seats[TEST_NUM_PLAYERS] = {0, 1};
    const int32_t game_history[1] = {0};
    const int32_t bias_profile[TEST_NUM_PLAYERS] = {0, 0};
    RolloutHandList player_hands[TEST_NUM_PLAYERS];
    init_hand_lists(player_hands);

    return rollout_compute_leaf_value_c(
        NULL,
        turn_board, 4, ROLLOUT_STREET_TURN, TEST_POT,
        active, TEST_NUM_PLAYERS, stacks, bets, player_hands,
        acting_order, TEST_NUM_PLAYERS, player_seats,
        game_history, 0, bias_profile,
        0, 0, ROLLOUT_NUM_ROLLOUTS_DEFAULT,
        TEST_MASTER_SEED, 0, 0, out_value);
}

/* Both players retain chips.  skip_first_betting closes the already-complete
 * river without a blueprint query, while seven opponent samples make this
 * value sensitive to any accidental use of the all-in rollout count. */
static int compute_non_allin_values(float out[TEST_NUM_PLAYERS]) {
    const int32_t active[TEST_NUM_PLAYERS] = {1, 1};
    const int32_t stacks[TEST_NUM_PLAYERS] = {1, 1};
    const int32_t bets[TEST_NUM_PLAYERS] = {TEST_BET, TEST_BET};
    const int32_t acting_order[TEST_NUM_PLAYERS] = {0, 1};
    const int32_t player_seats[TEST_NUM_PLAYERS] = {0, 1};
    const int32_t game_history[1] = {0};
    const int32_t bias_profile[TEST_NUM_PLAYERS] = {0, 0};
    RolloutHandList player_hands[TEST_NUM_PLAYERS];
    init_hand_lists(player_hands);

    return rollout_compute_leaf_values_all_hands_c(
        NULL,
        river_board, 5, ROLLOUT_STREET_RIVER, TEST_POT,
        active, TEST_NUM_PLAYERS, stacks, bets, player_hands,
        acting_order, TEST_NUM_PLAYERS, player_seats,
        game_history, 0, bias_profile,
        1, 7, TEST_MASTER_SEED, 0, 0,
        1, out);
}

static int write_full(int fd, const void *data, size_t size) {
    const unsigned char *p = (const unsigned char *)data;
    size_t written = 0;
    while (written < size) {
        ssize_t n = write(fd, p + written, size - written);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        written += (size_t)n;
    }
    return 0;
}

static int read_full(int fd, void *data, size_t size) {
    unsigned char *p = (unsigned char *)data;
    size_t received = 0;
    while (received < size) {
        ssize_t n = read(fd, p + received, size - received);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        received += (size_t)n;
    }
    return 0;
}

/* Each child gets a fresh copy of rollout.c's lazy cache.  This permits a
 * default-64 probe and an env=3 probe without adding a production reset API. */
static int run_isolated_probe(const char *allin_rollouts, ProbeResult *out) {
    int fds[2];
    if (pipe(fds) != 0) return -1;

    pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return -1;
    }
    if (pid == 0) {
        ProbeResult result;
        memset(&result, 0, sizeof(result));
        close(fds[0]);

        int env_rc = allin_rollouts
            ? setenv("BPS_ROLLOUT_ALLIN_ROLLOUTS", allin_rollouts, 1)
            : unsetenv("BPS_ROLLOUT_ALLIN_ROLLOUTS");
        if (env_rc != 0) _exit(10);

        result.allin_rc = compute_allin_value(&result.allin_value);
        result.non_allin_rc = compute_non_allin_values(
            result.non_allin_values);
        result.unexpected_calls = __atomic_load_n(
            &unexpected_dependency_calls, __ATOMIC_RELAXED);

        if (write_full(fds[1], &result, sizeof(result)) != 0) _exit(11);
        close(fds[1]);
        _exit(0);
    }

    close(fds[1]);
    int read_rc = read_full(fds[0], out, sizeof(*out));
    close(fds[0]);

    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);

    if (read_rc != 0 || waited != pid || !WIFEXITED(status)
        || WEXITSTATUS(status) != 0) {
        return -1;
    }
    return 0;
}

static void test_allin_accuracy_and_non_allin_identity(void) {
    ExactResult exact = enumerate_exact_allin_value();
    ProbeResult legacy;
    ProbeResult enhanced;
    ProbeResult configured_64;
    memset(&legacy, 0, sizeof(legacy));
    memset(&enhanced, 0, sizeof(enhanced));
    memset(&configured_64, 0, sizeof(configured_64));

    int legacy_probe_rc = run_isolated_probe("3", &legacy);
    int enhanced_probe_rc = run_isolated_probe(NULL, &enhanced);
    int configured_64_probe_rc = run_isolated_probe("64", &configured_64);
    CHECK(legacy_probe_rc == 0, "env=3 isolated probe completed");
    CHECK(enhanced_probe_rc == 0, "default-64 isolated probe completed");
    CHECK(configured_64_probe_rc == 0,
          "explicit env=64 isolated probe completed");

    CHECK(exact.outcomes == 176, "exact enumeration visited 4 * 44 outcomes");
    CHECK(exact.wins == 90 && exact.losses == 86 && exact.ties == 0,
          "exact enumeration has expected win/loss/tie counts");
    CHECK_CLOSE(exact.value, 2.2727273f, 1e-5f,
                "exact enumerated net value");

    if (legacy_probe_rc == 0 && enhanced_probe_rc == 0
        && configured_64_probe_rc == 0) {
        CHECK(legacy.allin_rc == 0, "three-rollout all-in call succeeded");
        CHECK(enhanced.allin_rc == 0, "default-64 all-in call succeeded");
        CHECK(configured_64.allin_rc == 0,
              "explicit-64 all-in call succeeded");
        CHECK(legacy.unexpected_calls == 0,
              "env=3 probe made no blueprint/dependency calls");
        CHECK(enhanced.unexpected_calls == 0,
              "default probe made no blueprint/dependency calls");
        CHECK(configured_64.unexpected_calls == 0,
              "env=64 probe made no blueprint/dependency calls");

        CHECK_CLOSE(legacy.allin_value, 100.0f, 1e-6f,
                    "fixed seed pins the three-rollout value");
        CHECK_CLOSE(enhanced.allin_value, 3.125f, 1e-6f,
                    "fixed seed pins the default-64 value");
        CHECK(memcmp(&enhanced.allin_value, &configured_64.allin_value,
                     sizeof(float)) == 0,
              "default all-in count matches explicit env=64");

        float legacy_error = fabsf(legacy.allin_value - exact.value);
        float enhanced_error = fabsf(enhanced.allin_value - exact.value);
        CHECK(enhanced_error <= 0.03f * (float)TEST_POT,
              "default-64 value is within 3 percent of pot of exact value");
        CHECK(legacy_error - enhanced_error >= 0.10f * (float)TEST_POT,
              "default-64 value is materially closer than three rollouts");

        CHECK(legacy.non_allin_rc == 0,
              "env=3 non-all-in call succeeded");
        CHECK(configured_64.non_allin_rc == 0,
              "env=64 non-all-in call succeeded");
        CHECK(memcmp(legacy.non_allin_values, configured_64.non_allin_values,
                     sizeof(legacy.non_allin_values)) == 0,
              "non-all-in values are byte-identical under env=3 and env=64");
        CHECK_CLOSE(configured_64.non_allin_values[0], -100.0f / 7.0f, 1e-5f,
                    "non-all-in probe retained exactly seven samples");
    }

    printf("all-in exact=%.6f three=%.6f default64=%.6f\n",
           exact.value, legacy.allin_value, enhanced.allin_value);
}

static void test_eager_allin_slots_are_identical(void) {
    const int32_t active[TEST_NUM_PLAYERS] = {1, 1};
    const int32_t acting_order[TEST_NUM_PLAYERS] = {0, 1};
    const int32_t player_seats[TEST_NUM_PLAYERS] = {0, 1};
    const int32_t game_history[1] = {0};
    RolloutHandList player_hands[TEST_NUM_PLAYERS];
    RolloutLeafInfo leaf;
    float out[TEST_CONT_PER_LEAF * TEST_NUM_PLAYERS];

    init_hand_lists(player_hands);
    memset(&leaf, 0, sizeof(leaf));
    memset(out, 0, sizeof(out));
    leaf.leaf_idx = 0;
    leaf.pot_chips = TEST_POT;
    leaf.bets[0] = TEST_BET;
    leaf.bets[1] = TEST_BET;

    CHECK(unsetenv("BPS_ROLLOUT_ALLIN_ROLLOUTS") == 0,
          "parent cleared all-in rollout override before first cached read");
    int rc = rollout_compute_subgame_leaf_values_c(
        NULL,
        turn_board, 4, ROLLOUT_STREET_TURN,
        &leaf, 1, active, TEST_NUM_PLAYERS, player_hands,
        acting_order, TEST_NUM_PLAYERS, player_seats,
        game_history, 0,
        TEST_BET, 1, ROLLOUT_NUM_ROLLOUTS_DEFAULT, TEST_MASTER_SEED,
        out, (int32_t)(sizeof(out) / sizeof(out[0])));
    CHECK(rc == 0, "eager all-in leaf fill succeeded");

    if (rc == 0) {
        CHECK_CLOSE(out[0], 3.125f, 1e-6f,
                    "eager combo 0 computed the default-64 hero value");
        for (int combo = 1; combo < TEST_CONT_PER_LEAF; combo++) {
            CHECK(memcmp(out, out + combo * TEST_NUM_PLAYERS,
                         TEST_NUM_PLAYERS * sizeof(float)) == 0,
                  "eager all-in combo slot equals combo 0 byte-for-byte");
        }
    }
    CHECK(__atomic_load_n(&unexpected_dependency_calls, __ATOMIC_RELAXED) == 0,
          "eager all-in fill made no blueprint/dependency calls");
}

int main(void) {
    test_allin_accuracy_and_non_allin_identity();
    test_eager_allin_slots_are_identical();

    printf("[test_rollout_allin_leaf] passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}

/* Link-only stubs.  None is reachable in the fixtures above; every call is
 * counted so the NULL-BPSReader assumption is itself under test. */
static void note_unexpected_dependency_call(void) {
    __atomic_add_fetch(&unexpected_dependency_calls, 1, __ATOMIC_RELAXED);
}

int bps_query_strategy(const BPSReader *r, const int *action_history,
                       int action_len, int player, int street, int bucket,
                       float *strategy_out) {
    (void)r; (void)action_history; (void)action_len; (void)player;
    (void)street; (void)bucket; (void)strategy_out;
    note_unexpected_dependency_call();
    return 0;
}

int bps_hand_to_bucket(const BPSReader *r, const int *board, int c0, int c1) {
    (void)r; (void)board; (void)c0; (void)c1;
    note_unexpected_dependency_call();
    return -1;
}

int bps_hand_to_bucket_turn(const BPSReader *r, const int *board,
                            int c0, int c1) {
    (void)r; (void)board; (void)c0; (void)c1;
    note_unexpected_dependency_call();
    return -1;
}

int bps_hand_to_bucket_river(const BPSReader *r, const int *board,
                             int c0, int c1) {
    (void)r; (void)board; (void)c0; (void)c1;
    note_unexpected_dependency_call();
    return -1;
}

int rollout_bucket_memo_lookup_turn(const void *reader, const int *board,
                                    int c0, int c1) {
    (void)reader; (void)board; (void)c0; (void)c1;
    note_unexpected_dependency_call();
    return ROLLOUT_BUCKET_MEMO_BYPASS;
}

int rollout_bucket_memo_lookup_river(const void *reader, const int *board,
                                     int c0, int c1) {
    (void)reader; (void)board; (void)c0; (void)c1;
    note_unexpected_dependency_call();
    return ROLLOUT_BUCKET_MEMO_BYPASS;
}

void rollout_bucket_memo_store_turn(const void *reader, const int *board,
                                    int c0, int c1, int bucket) {
    (void)reader; (void)board; (void)c0; (void)c1; (void)bucket;
    note_unexpected_dependency_call();
}

void rollout_bucket_memo_store_river(const void *reader, const int *board,
                                     int c0, int c1, int bucket) {
    (void)reader; (void)board; (void)c0; (void)c1; (void)bucket;
    note_unexpected_dependency_call();
}

/* The fill cores arm the memo before their parallel region. Unlike the
 * lookup/store stubs above (which flag an unexpected blueprint dependency if a
 * stubbed-out rollout somehow reached them), reset is an EXPECTED single-
 * threaded setup call, so it is a plain no-op. */
void rollout_bucket_memo_reset(const void *reader, const int flop[3]) {
    (void)reader; (void)flop;
}

uint64_t bp_compute_action_hash(const int *actions, int num_actions) {
    (void)actions; (void)num_actions;
    note_unexpected_dependency_call();
    return 0;
}

uint64_t bp_cont_ev_boundary_key_hash(uint64_t action_hash, int street) {
    (void)street;
    note_unexpected_dependency_call();
    return action_hash;
}

int bp_cont_ev_init(BPContEVTable *t, int64_t table_size) {
    (void)t; (void)table_size;
    note_unexpected_dependency_call();
    return -1;
}

void bp_cont_ev_free(BPContEVTable *t) {
    (void)t;
    note_unexpected_dependency_call();
}

BPContEVEntry *bp_cont_ev_find_or_create(BPContEVTable *t,
                                         uint64_t action_hash,
                                         int player, int bucket) {
    (void)t; (void)action_hash; (void)player; (void)bucket;
    note_unexpected_dependency_call();
    return NULL;
}

void bp_cont_ev_atomic_add_f32(float *dst, float val) {
    (void)dst; (void)val;
    note_unexpected_dependency_call();
}

BPBoundaryEVStatus bp_query_boundary_ev(const BPContEVTable *t,
                                        uint64_t action_hash,
                                        int player, int bucket,
                                        int cont_bias, float *out_ev) {
    (void)t; (void)action_hash; (void)player; (void)bucket; (void)cont_bias;
    if (out_ev) *out_ev = 0.0f;
    note_unexpected_dependency_call();
    return BP_BOUNDARY_EV_MISS;
}
