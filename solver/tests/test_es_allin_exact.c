/*
 * Exact sampled-pair evaluation at betting-closed ES depth-limit leaves.
 * Artifact-free: hand-built SSTreeData fixtures, table oracle, and eval7 only.
 */

#include "es_resolver.h"
#include "hand_eval.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int passed = 0;
static int failed = 0;

#define CHECK(cond, msg) do {                                           \
    if (cond) {                                                         \
        passed++;                                                       \
    } else {                                                            \
        failed++;                                                       \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
    }                                                                   \
} while (0)

#define CHECK_CLOSE(actual, expected, tolerance, msg) do {                 \
    double _actual = (double)(actual);                                      \
    double _expected = (double)(expected);                                  \
    double _tolerance = (double)(tolerance);                                \
    if (fabs(_actual - _expected) <= _tolerance) {                          \
        passed++;                                                           \
    } else {                                                                \
        failed++;                                                           \
        fprintf(stderr,                                                     \
                "FAIL %s:%d: %s (got %.17g, want %.17g, tol %.3g)\n",     \
                __FILE__, __LINE__, msg, _actual, _expected, _tolerance);   \
    }                                                                       \
} while (0)

typedef struct {
    SSTreeData tree;
    SSNode node;
    int leaf_node_index;
} LeafFixture;

typedef struct {
    int outcomes;
    int p0_wins;
    int p0_losses;
    int ties;
    double value[2];
} HUExact;

static LeafFixture *make_leaf_fixture(
    int np, const int *board, int nb, int pot, const int *bets,
    const int *active, int effective_stack,
    const int hands[SS_MAX_PLAYERS][2])
{
    LeafFixture *f = (LeafFixture *)calloc(1, sizeof(*f));
    if (!f) return NULL;

    f->tree.nodes = &f->node;
    f->tree.num_nodes = 1;
    f->tree.num_players = np;
    f->tree.num_board = nb;
    f->tree.num_leaves = 1;
    f->tree.num_leaf_nodes = 1;
    f->leaf_node_index = 0;
    f->tree.leaf_node_indices = &f->leaf_node_index;
    f->tree.effective_stack = effective_stack;
    f->tree.is_river = nb == 5;
    f->tree.subgame_root_street = nb;
    int bet_total = 0;
    for (int b = 0; b < nb; b++) f->tree.board[b] = board[b];
    for (int p = 0; p < np; p++) {
        f->tree.num_hands[p] = 1;
        f->tree.hands[p][0][0] = hands[p][0];
        f->tree.hands[p][0][1] = hands[p][1];
        f->tree.weights[p][0] = 1.0f;
        bet_total += bets[p];
    }
    f->tree.starting_pot = pot - bet_total;

    f->node.type = SS_NODE_LEAF;
    f->node.player = -1;
    f->node.first_child = -1;
    f->node.pot = pot;
    f->node.num_board = nb;
    f->node.leaf_idx = 0;
    f->node.fold_player = -1;
    f->node.chance_node_idx = -1;
    for (int b = 0; b < nb; b++) f->node.board_cards[b] = board[b];
    for (int p = 0; p < np; p++) {
        f->node.bets[p] = bets[p];
        f->node.active_players[p] = active[p] ? 1 : 0;
        if (active[p]) f->node.num_players++;
    }
    return f;
}

static ESResolver *make_table_resolver(LeafFixture *f, float *table,
                                       void **out_ctx, uint64_t seed) {
    void *ctx = es_make_table_oracle(table, f->tree.num_players, 1);
    if (!ctx) return NULL;
    ESResolver *r = es_create(&f->tree, es_table_oracle_fn(), ctx, seed);
    if (!r) {
        es_free_oracle_ctx(ctx);
        return NULL;
    }
    *out_ctx = ctx;
    return r;
}

static int card_in_prefix(const int *board, int nb, int card) {
    for (int b = 0; b < nb; b++) if (board[b] == card) return 1;
    return 0;
}

static int card_in_hu_holes(const int hands[SS_MAX_PLAYERS][2], int card) {
    for (int p = 0; p < 2; p++)
        if (hands[p][0] == card || hands[p][1] == card) return 1;
    return 0;
}

static void hu_record(HUExact *out, const int board5[5], int pot,
                      const int *bets,
                      const int hands[SS_MAX_PLAYERS][2]) {
    int cards0[7] = {
        board5[0], board5[1], board5[2], board5[3], board5[4],
        hands[0][0], hands[0][1]
    };
    int cards1[7] = {
        board5[0], board5[1], board5[2], board5[3], board5[4],
        hands[1][0], hands[1][1]
    };
    uint32_t s0 = eval7(cards0);
    uint32_t s1 = eval7(cards1);
    if (s0 > s1) {
        out->p0_wins++;
        out->value[0] += (double)pot - bets[0];
        out->value[1] -= (double)bets[1];
    } else if (s0 < s1) {
        out->p0_losses++;
        out->value[0] -= (double)bets[0];
        out->value[1] += (double)pot - bets[1];
    } else {
        out->ties++;
        out->value[0] += (double)pot / 2.0 - bets[0];
        out->value[1] += (double)pot / 2.0 - bets[1];
    }
    out->outcomes++;
}

static HUExact enumerate_hu(const int *board, int nb, int pot,
                            const int *bets,
                            const int hands[SS_MAX_PLAYERS][2]) {
    HUExact out;
    memset(&out, 0, sizeof(out));
    int board5[5];
    for (int b = 0; b < nb; b++) board5[b] = board[b];

    if (nb == 4) {
        for (int river = 0; river < 52; river++) {
            if (card_in_prefix(board, nb, river)
                || card_in_hu_holes(hands, river)) continue;
            board5[4] = river;
            hu_record(&out, board5, pot, bets, hands);
        }
    } else if (nb == 3) {
        for (int turn = 0; turn < 52; turn++) {
            if (card_in_prefix(board, nb, turn)
                || card_in_hu_holes(hands, turn)) continue;
            for (int river = turn + 1; river < 52; river++) {
                if (card_in_prefix(board, nb, river)
                    || card_in_hu_holes(hands, river)) continue;
                board5[3] = turn;
                board5[4] = river;
                hu_record(&out, board5, pot, bets, hands);
            }
        }
    }
    if (out.outcomes > 0) {
        out.value[0] /= (double)out.outcomes;
        out.value[1] /= (double)out.outcomes;
    }
    return out;
}

static void test_turn_exact_both_traversers_and_paths(void) {
    const int board[4] = {0, 1, 2, 3};
    const int bets[2] = {100, 100};
    const int active[2] = {1, 1};
    const int hands[SS_MAX_PLAYERS][2] = {
        {44, 4},  /* Kc3c */
        {40, 8}   /* Qc4c */
    };
    const int sampled[2] = {0, 0};
    float table[2] = {-999.0f, 999.0f};
    HUExact exact = enumerate_hu(board, 4, 300, bets, hands);

    CHECK(exact.outcomes == 44, "turn brute force visited 44 rivers");
    CHECK(exact.p0_wins == 37 && exact.p0_losses == 0 && exact.ties == 7,
          "turn brute force has 37 wins and 7 ties");
    CHECK_CLOSE(exact.value[0], 7750.0 / 44.0, 1e-12,
                "turn brute-force p0 value");
    CHECK_CLOSE(exact.value[1], -3350.0 / 44.0, 1e-12,
                "turn brute-force p1 value");

    LeafFixture *f = make_leaf_fixture(
        2, board, 4, 300, bets, active, 100, hands);
    CHECK(f != NULL, "allocated turn leaf fixture");
    if (!f) return;
    void *ctx = NULL;
    ESResolver *scalar = make_table_resolver(f, table, &ctx, 123);
    CHECK(scalar != NULL, "created scalar exact resolver");
    if (!scalar) { free(f); return; }
    ESResolver *vec = es_create(&f->tree, es_table_oracle_fn(), ctx, 123);
    CHECK(vec != NULL, "created vector exact resolver");
    if (!vec) {
        es_free(scalar); es_free_oracle_ctx(ctx); free(f); return;
    }
    es_set_exact_allin_leaves(scalar, 1);
    es_set_exact_allin_leaves(vec, 1);
    es_set_vec_mode(vec, 1);
    CHECK(es_run(scalar, 1, 1) == 0, "short scalar exact run succeeded");
    CHECK(es_run(vec, 1, 1) == 0, "short vector exact run succeeded");

    ESStats scalar_stats, vec_stats;
    es_get_stats(scalar, &scalar_stats);
    es_get_stats(vec, &vec_stats);
    CHECK(scalar_stats.leaf_rollouts == 2,
          "scalar run observed one exact leaf value per traverser");
    CHECK(vec_stats.leaf_rollouts == 2,
          "vector run observed one exact leaf value per traverser hand");

    for (int p = 0; p < 2; p++) {
        double sv = es_test_leaf_value(scalar, 0, p, sampled);
        double vv = es_test_leaf_value_vec(vec, 0, p, sampled);
        CHECK_CLOSE(sv, exact.value[p], 1e-9,
                    "scalar leaf path matches turn brute force");
        CHECK_CLOSE(vv, exact.value[p], 1e-9,
                    "vector leaf path matches turn brute force");
        CHECK_CLOSE(vv, sv, 1e-12,
                    "scalar and vector leaf paths agree");
    }

    es_free(vec);
    es_free(scalar);
    es_free_oracle_ctx(ctx);
    free(f);
}

static void test_always_win_loss_and_tie(void) {
    const int royal_board[4] = {50, 46, 42, 38}; /* Ah Kh Qh Jh */
    const int quad_board[4] = {0, 1, 2, 3};
    const int bets[2] = {100, 100};
    const int active[2] = {1, 1};
    const int sampled[2] = {0, 0};
    float table[2] = {17.0f, 19.0f};

    const int win_hands[SS_MAX_PLAYERS][2] = {
        {34, 0},  /* Th2c: royal flush */
        {49, 45}  /* AdKd */
    };
    HUExact win_exact = enumerate_hu(
        royal_board, 4, 300, bets, win_hands);
    CHECK(win_exact.outcomes == 44 && win_exact.p0_wins == 44,
          "royal-flush fixture wins all 44 rivers");
    LeafFixture *win_f = make_leaf_fixture(
        2, royal_board, 4, 300, bets, active, 100, win_hands);
    CHECK(win_f != NULL, "allocated always-win fixture");
    if (!win_f) return;
    void *win_ctx = NULL;
    ESResolver *win_r = make_table_resolver(win_f, table, &win_ctx, 11);
    CHECK(win_r != NULL, "created always-win resolver");
    if (!win_r) { free(win_f); return; }
    es_set_exact_allin_leaves(win_r, 1);
    CHECK_CLOSE(es_test_leaf_value(win_r, 0, 0, sampled), 200.0, 1e-9,
                "traverser wins every runout: pot minus own bet");
    CHECK_CLOSE(es_test_leaf_value(win_r, 0, 1, sampled), -100.0, 1e-9,
                "traverser loses every runout: minus own bet");
    es_free(win_r);
    es_free_oracle_ctx(win_ctx);
    free(win_f);

    const int tie_hands[SS_MAX_PLAYERS][2] = {
        {48, 4},  /* Ac3c */
        {49, 8}   /* Ad4c */
    };
    HUExact tie_exact = enumerate_hu(quad_board, 4, 300, bets, tie_hands);
    CHECK(tie_exact.outcomes == 44 && tie_exact.ties == 44,
          "quad-board fixture ties all 44 rivers");
    LeafFixture *tie_f = make_leaf_fixture(
        2, quad_board, 4, 300, bets, active, 100, tie_hands);
    CHECK(tie_f != NULL, "allocated always-tie fixture");
    if (!tie_f) return;
    void *tie_ctx = NULL;
    ESResolver *tie_r = make_table_resolver(tie_f, table, &tie_ctx, 12);
    CHECK(tie_r != NULL, "created always-tie resolver");
    if (!tie_r) { free(tie_f); return; }
    es_set_exact_allin_leaves(tie_r, 1);
    CHECK_CLOSE(es_test_leaf_value(tie_r, 0, 0, sampled), 50.0, 1e-9,
                "p0 ties every runout: half pot minus own bet");
    CHECK_CLOSE(es_test_leaf_value(tie_r, 0, 1, sampled), 50.0, 1e-9,
                "p1 ties every runout: half pot minus own bet");
    es_free(tie_r);
    es_free_oracle_ctx(tie_ctx);
    free(tie_f);
}

static void test_flop_990_runouts(void) {
    const int board[3] = {50, 46, 42}; /* Ah Kh Qh */
    const int bets[2] = {100, 100};
    const int active[2] = {1, 1};
    const int hands[SS_MAX_PLAYERS][2] = {
        {38, 34}, /* JhTh: royal flush */
        {49, 45}  /* AdKd */
    };
    const int sampled[2] = {0, 0};
    float table[2] = {-123.0f, 456.0f};
    HUExact exact = enumerate_hu(board, 3, 300, bets, hands);
    CHECK(exact.outcomes == 990, "flop brute force visited 990 runouts");
    CHECK(exact.p0_wins == 990,
          "flopped royal flush wins every remaining runout");
    CHECK_CLOSE(exact.value[0], 200.0, 1e-12,
                "flop brute-force value is pot minus own bet");

    LeafFixture *f = make_leaf_fixture(
        2, board, 3, 300, bets, active, 100, hands);
    CHECK(f != NULL, "allocated flop leaf fixture");
    if (!f) return;
    void *ctx = NULL;
    ESResolver *r = make_table_resolver(f, table, &ctx, 22);
    CHECK(r != NULL, "created flop exact resolver");
    if (!r) { free(f); return; }
    es_set_exact_allin_leaves(r, 1);
    CHECK_CLOSE(es_test_leaf_value(r, 0, 0, sampled), exact.value[0], 1e-9,
                "exact flop leaf matches 990-runout brute force");
    es_free(r);
    es_free_oracle_ctx(ctx);
    free(f);
}

static void test_three_player_layered_allin(void) {
    const int board[4] = {50, 46, 42, 38}; /* Ah Kh Qh Jh */
    const int bets[3] = {100, 60, 100};
    const int active[3] = {1, 1, 1};
    const int hands[SS_MAX_PLAYERS][2] = {
        {30, 4},  /* 9h3c: wins the p0/p2 side pot */
        {34, 0},  /* Th2c: wins the main pot */
        {20, 17}  /* 7c6d */
    };
    const int sampled[3] = {0, 0, 0};
    const int hand_c0[3] = {30, 34, 20};
    const int hand_c1[3] = {4, 0, 17};
    double brute_sum[3] = {0.0, 0.0, 0.0};
    int outcomes = 0;
    int every_runout_matches = 1;
    for (int river = 0; river < 52; river++) {
        if (card_in_prefix(board, 4, river)) continue;
        int used = 0;
        for (int p = 0; p < 3; p++)
            if (hands[p][0] == river || hands[p][1] == river) used = 1;
        if (used) continue;
        int board5[5] = {board[0], board[1], board[2], board[3], river};
        const double expected[3] = {-20.0, 180.0, -100.0};
        for (int p = 0; p < 3; p++) {
            double v = es_test_showdown_value(
                3, board5, hand_c0, hand_c1, active, bets, 320, p);
            brute_sum[p] += v;
            if (fabs(v - expected[p]) > 1e-12) every_runout_matches = 0;
        }
        outcomes++;
    }
    CHECK(outcomes == 42, "three-player brute force visited 42 rivers");
    CHECK(every_runout_matches,
          "es_layered_award gives 240 main / 80 side on every river");
    for (int p = 0; p < 3; p++) brute_sum[p] /= (double)outcomes;

    float table[3] = {1.0f, 2.0f, 3.0f};
    LeafFixture *f = make_leaf_fixture(
        3, board, 4, 320, bets, active, 100, hands);
    CHECK(f != NULL, "allocated three-player layered fixture");
    if (!f) return;
    void *ctx = NULL;
    ESResolver *r = make_table_resolver(f, table, &ctx, 33);
    CHECK(r != NULL, "created three-player exact resolver");
    if (!r) { free(f); return; }
    es_set_exact_allin_leaves(r, 1);
    const double expected[3] = {-20.0, 180.0, -100.0};
    for (int p = 0; p < 3; p++) {
        double v = es_test_leaf_value(r, 0, p, sampled);
        CHECK_CLOSE(v, brute_sum[p], 1e-9,
                    "three-player exact leaf matches layered brute force");
        CHECK_CLOSE(v, expected[p], 1e-9,
                    "three-player exact net award is correct");
    }
    es_free(r);
    es_free_oracle_ctx(ctx);
    free(f);
}

static void test_table_fallbacks(void) {
    const int board[4] = {0, 1, 2, 3};
    const int active[2] = {1, 1};
    const int hands[SS_MAX_PLAYERS][2] = {{44, 4}, {40, 8}};
    const int sampled[2] = {0, 0};

    const int open_bets[2] = {50, 50};
    float open_table[2] = {321.25f, -654.5f};
    LeafFixture *open_f = make_leaf_fixture(
        2, board, 4, 200, open_bets, active, 100, hands);
    CHECK(open_f != NULL, "allocated non-closed fixture");
    if (!open_f) return;
    void *open_ctx = NULL;
    ESResolver *open_r = make_table_resolver(
        open_f, open_table, &open_ctx, 44);
    CHECK(open_r != NULL, "created non-closed resolver");
    if (!open_r) { free(open_f); return; }
    es_set_exact_allin_leaves(open_r, 1);
    CHECK(es_run(open_r, 0, 1) == 0,
          "non-closed exact-mode cache preparation succeeded");
    CHECK_CLOSE(es_test_leaf_value(open_r, 0, 0, sampled), 321.25, 0.0,
                "non-closed leaf keeps p0 table cell");
    CHECK_CLOSE(es_test_leaf_value_vec(open_r, 0, 1, sampled), -654.5, 0.0,
                "non-closed vector leaf keeps p1 table cell");
    es_free(open_r);
    es_free_oracle_ctx(open_ctx);
    free(open_f);

    const int closed_bets[2] = {100, 100};
    float closed_table[2] = {777.25f, -888.5f};
    LeafFixture *closed_f = make_leaf_fixture(
        2, board, 4, 300, closed_bets, active, 100, hands);
    CHECK(closed_f != NULL, "allocated default-off fixture");
    if (!closed_f) return;
    void *closed_ctx = NULL;
    ESResolver *closed_r = make_table_resolver(
        closed_f, closed_table, &closed_ctx, 55);
    CHECK(closed_r != NULL, "created default-off resolver");
    if (!closed_r) { free(closed_f); return; }
    /* Deliberately do not call es_set_exact_allin_leaves. */
    CHECK_CLOSE(es_test_leaf_value(closed_r, 0, 0, sampled), 777.25, 0.0,
                "default-off closed leaf keeps p0 table cell");
    CHECK_CLOSE(es_test_leaf_value_vec(closed_r, 0, 1, sampled), -888.5, 0.0,
                "default-off closed vector leaf keeps p1 table cell");
    es_free(closed_r);
    es_free_oracle_ctx(closed_ctx);
    free(closed_f);
}

static void test_impossible_sampled_worlds_are_neutral(void) {
    const int board[4] = {0, 1, 2, 3};
    const int bets[2] = {100, 100};
    const int active[2] = {1, 1};
    const int sampled[2] = {0, 0};
    float table[2] = {11.0f, 22.0f};

    const int shared_hands[SS_MAX_PLAYERS][2] = {
        {44, 4}, {44, 8}
    };
    LeafFixture *shared_f = make_leaf_fixture(
        2, board, 4, 300, bets, active, 100, shared_hands);
    CHECK(shared_f != NULL, "allocated private-card collision fixture");
    if (!shared_f) return;
    void *shared_ctx = NULL;
    ESResolver *shared_r = make_table_resolver(
        shared_f, table, &shared_ctx, 66);
    CHECK(shared_r != NULL, "created private-card collision resolver");
    if (!shared_r) { free(shared_f); return; }
    es_set_exact_allin_leaves(shared_r, 1);
    CHECK_CLOSE(es_test_leaf_value(shared_r, 0, 0, sampled), 0.0, 0.0,
                "scalar private-card collision is neutral");
    CHECK_CLOSE(es_test_leaf_value_vec(shared_r, 0, 0, sampled), 0.0, 0.0,
                "vector private-card collision is neutral");
    es_free(shared_r);
    es_free_oracle_ctx(shared_ctx);
    free(shared_f);

    const int board_collision_hands[SS_MAX_PLAYERS][2] = {
        {44, 4}, {0, 8}
    };
    LeafFixture *board_f = make_leaf_fixture(
        2, board, 4, 300, bets, active, 100, board_collision_hands);
    CHECK(board_f != NULL, "allocated board collision fixture");
    if (!board_f) return;
    void *board_ctx = NULL;
    ESResolver *board_r = make_table_resolver(
        board_f, table, &board_ctx, 77);
    CHECK(board_r != NULL, "created board collision resolver");
    if (!board_r) { free(board_f); return; }
    es_set_exact_allin_leaves(board_r, 1);
    CHECK_CLOSE(es_test_leaf_value(board_r, 0, 0, sampled), 0.0, 0.0,
                "scalar opponent-board collision is neutral");
    CHECK_CLOSE(es_test_leaf_value_vec(board_r, 0, 0, sampled), 0.0, 0.0,
                "vector opponent-board collision is neutral");
    es_free(board_r);
    es_free_oracle_ctx(board_ctx);
    free(board_f);
}

int main(void) {
    poker_eval5_table_init();
    test_turn_exact_both_traversers_and_paths();
    test_always_win_loss_and_tie();
    test_flop_990_runouts();
    test_three_player_layered_allin();
    test_table_fallbacks();
    test_impossible_sampled_worlds_are_neutral();

    printf("[test_es_allin_exact] passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
