/**
 * Cross-street contribution carry + hero-line freezing in the ES resolver.
 *
 * Artifact-free: hand-built SSTreeData fixtures and eval7 only.
 *
 * 1. CARRY. The builder resets SSNode.bets to 0 below a chance node and keeps
 *    the closing street's bets on the chance node. A terminal that charged
 *    only this street's bets would, on a river-extended turn tree, make a
 *    fold pay the turn bet while a call that reached the river paid nothing
 *    for the turn: calling any turn bet or jam would be over-valued by the
 *    call amount (measured: ace-high called a jam at 1.00). The resolver
 *    charges bets + earlier-street carry at every terminal.
 * 2. FREEZING. SSTreeData.frozen_action[node] >= 0 walks that child with
 *    probability one (traverser or opponent) and accumulates no regret or
 *    strategy sum there, in the scalar, vector and reach paths.
 */
#include "es_resolver.h"
#include "hand_eval.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int passed = 0;
static int failed = 0;

#define CHECK(cond, msg) do {                                             \
    if (cond) { passed++; }                                               \
    else { failed++; fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__,      \
                             __LINE__, msg); }                            \
} while (0)

#define CHECK_NEAR(a, b, tol, msg) do {                                   \
    double _a = (a), _b = (b);                                            \
    if (fabs(_a - _b) <= (tol)) { passed++; }                             \
    else { failed++; fprintf(stderr, "FAIL %s:%d: %s: got %.6f expected " \
                             "%.6f\n", __FILE__, __LINE__, msg, _a, _b); } \
} while (0)

static int card(const char *s) {
    const char *R = "23456789TJQKA";
    const char *S = "cdhs";
    return (int)(strchr(R, s[0]) - R) * 4 + (int)(strchr(S, s[1]) - S);
}

typedef struct {
    SSTreeData tree;
    SSNode nodes[16];
    int children[16];
    int chance_cards[8];
    int chance_nodes[2];
    int frozen[16];
} Fixture;

static void node_init(SSNode *n, int type, int player, int pot, int nb,
                      const int *board, int b0, int b1, int a0, int a1) {
    memset(n, 0, sizeof(*n));
    n->type = type;
    n->player = player;
    n->pot = pot;
    n->num_board = nb;
    for (int i = 0; i < nb; i++) n->board_cards[i] = board[i];
    n->bets[0] = b0;
    n->bets[1] = b1;
    n->active_players[0] = a0;
    n->active_players[1] = a1;
    n->num_players = a0 + a1;
    n->leaf_idx = -1;
    n->fold_player = -1;
    n->chance_node_idx = -1;
    n->first_child = -1;
}

static void tree_common(Fixture *f, const int hero[2], const int vil[2],
                        const int *board, int nb, int starting_pot) {
    memset(f, 0, sizeof(*f));
    SSTreeData *t = &f->tree;
    t->nodes = f->nodes;
    t->children = f->children;
    t->num_players = 2;
    t->num_board = nb;
    for (int i = 0; i < nb; i++) t->board[i] = board[i];
    t->num_hands[0] = 1;
    t->num_hands[1] = 1;
    t->hands[0][0][0] = hero[0]; t->hands[0][0][1] = hero[1];
    t->hands[1][0][0] = vil[0];  t->hands[1][0][1] = vil[1];
    t->weights[0][0] = 1.0f;
    t->weights[1][0] = 1.0f;
    t->starting_pot = starting_pot;
    t->effective_stack = 5000;
    t->is_river = (nb == 5);
    t->subgame_root_street = nb;
}

/* Two-street tree. Node 0: hero (p0) faces 300 more (bets 100/400, pot 1000):
 * child 0 = fold (node 1), child 1 = call -> chance node 2 (closing bets
 * 400/400, pot 1300) -> river showdown nodes 3 and 4 (rivers r0, r1). */
static void build_two_street(Fixture *f, const int hero[2], const int vil[2],
                             const int *turn, int r0, int r1) {
    tree_common(f, hero, vil, turn, 4, 500);
    SSTreeData *t = &f->tree;
    node_init(&f->nodes[0], SS_NODE_DECISION, 0, 1000, 4, turn, 100, 400, 1, 1);
    f->nodes[0].first_child = 0; f->nodes[0].num_children = 2;
    f->children[0] = 1; f->children[1] = 2;
    node_init(&f->nodes[1], SS_NODE_FOLD, 1, 1000, 4, turn, 100, 400, 0, 1);
    f->nodes[1].fold_player = 0;
    node_init(&f->nodes[2], SS_NODE_CHANCE, -1, 1300, 4, turn, 400, 400, 1, 1);
    f->nodes[2].first_child = 2; f->nodes[2].num_children = 2;
    f->children[2] = 3; f->children[3] = 4;
    f->nodes[2].chance_node_idx = 0;
    f->nodes[2].chance_card_per_child_offset = 0;
    f->chance_cards[0] = r0; f->chance_cards[1] = r1;
    t->chance_card_per_child = f->chance_cards;
    t->chance_card_per_child_total = 2;
    f->chance_nodes[0] = 2;
    t->chance_node_indices = f->chance_nodes;
    t->num_chance_nodes = 1;
    int riv[5];
    memcpy(riv, turn, 4 * sizeof(int));
    riv[4] = r0;
    node_init(&f->nodes[3], SS_NODE_SHOWDOWN, -1, 1300, 5, riv, 0, 0, 1, 1);
    riv[4] = r1;
    node_init(&f->nodes[4], SS_NODE_SHOWDOWN, -1, 1300, 5, riv, 0, 0, 1, 1);
    t->num_nodes = 5;
    t->num_children_total = 4;
}

/* River tree for freezing. Node 0: hero (p0) check (node 1) / bet 500 (node
 * 2). Node 1: villain (p1) check -> showdown node 3 (pot 1000). Node 2:
 * villain fold (node 4, hero wins) / call -> showdown node 5 (pot 2000). */
static void build_freeze_tree(Fixture *f, const int hero[2], const int vil[2],
                              const int *river) {
    tree_common(f, hero, vil, river, 5, 1000);
    SSTreeData *t = &f->tree;
    node_init(&f->nodes[0], SS_NODE_DECISION, 0, 1000, 5, river, 0, 0, 1, 1);
    f->nodes[0].first_child = 0; f->nodes[0].num_children = 2;
    f->children[0] = 1; f->children[1] = 2;
    node_init(&f->nodes[1], SS_NODE_DECISION, 1, 1000, 5, river, 0, 0, 1, 1);
    f->nodes[1].first_child = 2; f->nodes[1].num_children = 1;
    f->children[2] = 3;
    node_init(&f->nodes[2], SS_NODE_DECISION, 1, 1500, 5, river, 500, 0, 1, 1);
    f->nodes[2].first_child = 3; f->nodes[2].num_children = 2;
    f->children[3] = 4; f->children[4] = 5;
    node_init(&f->nodes[3], SS_NODE_SHOWDOWN, -1, 1000, 5, river, 0, 0, 1, 1);
    node_init(&f->nodes[4], SS_NODE_FOLD, 0, 1500, 5, river, 500, 0, 1, 0);
    f->nodes[4].fold_player = 1;
    node_init(&f->nodes[5], SS_NODE_SHOWDOWN, -1, 2000, 5, river, 500, 500, 1, 1);
    t->num_nodes = 6;
    t->num_children_total = 5;
    for (int i = 0; i < 16; i++) f->frozen[i] = -1;
}

static ESResolver *make(Fixture *f, int mode, uint64_t seed) {
    ESResolver *r = es_create(&f->tree, es_table_oracle_fn(), NULL, seed);
    if (!r) return NULL;
    es_set_chance_mode(r, 1);         /* enumerate: exact expectations */
    if (mode == 1) es_set_vec_mode(r, 1);
    if (mode == 2) es_set_reach_mode(r, 1);
    return r;
}

static const char *mode_name(int mode) {
    return mode == 0 ? "scalar" : (mode == 1 ? "vec" : "reach");
}

/* ── 1. terminal payoffs charge the earlier-street contribution ─────────── */
static void test_carry_terminal_payoffs(void) {
    int hero[2] = { card("7c"), card("2d") };
    int vil[2]  = { card("As"), card("Ad") };
    int turn[4] = { card("Kh"), card("Qs"), card("9c"), card("4d") };
    Fixture f;
    build_two_street(&f, hero, vil, turn, card("5h"), card("Tc"));
    ESResolver *r = make(&f, 0, 7);
    CHECK(r != NULL, "es_create on the two-street fixture");
    if (!r) return;
    int sampled[SS_MAX_PLAYERS] = { 0, 0 };
    CHECK_NEAR(es_test_terminal_payoff(r, 1, 0, sampled), -100.0, 1e-9,
               "hero's fold pays his 100 turn bet");
    CHECK_NEAR(es_test_terminal_payoff(r, 1, 1, sampled), 600.0, 1e-9,
               "villain's fold win is pot 1000 minus his 400");
    CHECK_NEAR(es_test_terminal_payoff(r, 3, 0, sampled), -400.0, 1e-9,
               "river showdown charges hero's 400 turn contribution (was 0)");
    CHECK_NEAR(es_test_terminal_payoff(r, 3, 1, sampled), 900.0, 1e-9,
               "villain wins 1300 minus his 400 turn contribution");
    CHECK_NEAR(es_test_terminal_payoff(r, 4, 0, sampled), -400.0, 1e-9,
               "second river, same charge");
    es_free(r);
}

/* ── 2. the solve prefers the correct action in all three modes ─────────── */
static double action_prob_after_run(Fixture *f, int mode, int iters,
                                    int action) {
    ESResolver *r = make(f, mode, 11);
    if (!r) return -1.0;
    if (es_run(r, iters, 1) != 0) { es_free(r); return -2.0; }
    double out[SS_MAX_ACTIONS];
    memset(out, 0, sizeof(out));
    int n = es_final_strategy(r, 0, out, SS_MAX_ACTIONS);
    double p = (n >= 2) ? out[action] : -3.0;
    es_free(r);
    return p;
}

static void test_call_is_not_free_across_the_chance_node(void) {
    int turn[4] = { card("Kh"), card("Qs"), card("9c"), card("4d") };
    int loser[2] = { card("7c"), card("2d") };
    int winner[2] = { card("As"), card("Ad") };
    for (int mode = 0; mode < 3; mode++) {
        char msg[160];
        Fixture f;
        /* hero always loses: call = -400 < fold = -100 -> fold */
        build_two_street(&f, loser, winner, turn, card("5h"), card("Tc"));
        double p_fold = action_prob_after_run(&f, mode, 300, 0);
        snprintf(msg, sizeof msg, "%s: hero who always loses folds the turn "
                 "bet (call is -400, fold -100); fold prob=%.3f",
                 mode_name(mode), p_fold);
        CHECK(p_fold > 0.95, msg);
        /* hero always wins: call = 1300 - 400 = +900 > fold -100 -> call */
        build_two_street(&f, winner, loser, turn, card("5h"), card("Tc"));
        double p_call = action_prob_after_run(&f, mode, 300, 1);
        snprintf(msg, sizeof msg, "%s: hero who always wins calls; call "
                 "prob=%.3f", mode_name(mode), p_call);
        CHECK(p_call > 0.95, msg);
    }
}

/* ── 3. single-street trees are unaffected (no carry array) ─────────────── */
static void test_single_street_has_no_carry(void) {
    int hero[2] = { card("As"), card("Ad") };
    int vil[2]  = { card("Kc"), card("Kd") };
    int river[5] = { card("2c"), card("5d"), card("8h"), card("Jh"), card("Qs") };
    Fixture f;
    build_freeze_tree(&f, hero, vil, river);
    ESResolver *r = make(&f, 0, 3);
    CHECK(r != NULL, "es_create on the river fixture");
    if (!r) return;
    int sampled[SS_MAX_PLAYERS] = { 0, 0 };
    CHECK_NEAR(es_test_terminal_payoff(r, 4, 0, sampled), 1000.0, 1e-9,
               "fold win: pot 1500 minus hero's 500 river bet");
    CHECK_NEAR(es_test_terminal_payoff(r, 5, 0, sampled), 1500.0, 1e-9,
               "showdown win: pot 2000 minus 500");
    CHECK_NEAR(es_test_terminal_payoff(r, 5, 1, sampled), -500.0, 1e-9,
               "showdown loss: minus 500");
    es_free(r);
}

/* ── 4. hero-line freezing ──────────────────────────────────────────────── */
static void test_frozen_action_is_honoured(void) {
    /* Hero holds KK against AA: checking is hero's preferred line, so the
     * frozen bet forces a line self-play would abandon after one iteration. */
    int hero[2] = { card("Kc"), card("Kd") };
    int vil[2]  = { card("As"), card("Ad") };
    int river[5] = { card("2c"), card("5d"), card("8h"), card("Jh"), card("Qs") };
    for (int mode = 0; mode < 3; mode++) {
        char msg[200];
        Fixture f;
        build_freeze_tree(&f, hero, vil, river);
        /* control: nothing frozen -> both hero children are visited */
        {
            ESResolver *r = make(&f, mode, 5);
            CHECK(r != NULL, "create control");
            if (!r) continue;
            if (mode != 2) es_enable_visit_counter(r);
            CHECK(es_run(r, 200, 1) == 0, "run control");
            if (mode != 2) {
                snprintf(msg, sizeof msg, "%s control: villain's check node "
                         "is visited", mode_name(mode));
                CHECK(es_get_visit_count(r, 1) > 0, msg);
            }
            CHECK(es_strategy_sum_mass(r, 0, 0) > 0.0,
                  "control: hero root accumulates strategy mass");
            es_free(r);
        }
        /* frozen: hero must bet -> only villain's bet node is walked */
        build_freeze_tree(&f, hero, vil, river);
        f.frozen[0] = 1;
        f.tree.frozen_action = f.frozen;
        ESResolver *r = make(&f, mode, 5);
        CHECK(r != NULL, "create frozen");
        if (!r) continue;
        if (mode != 2) es_enable_visit_counter(r);
        CHECK(es_run(r, 200, 1) == 0, "run frozen");
        if (mode != 2) {
            snprintf(msg, sizeof msg, "%s frozen: villain's check node is "
                     "never visited", mode_name(mode));
            CHECK(es_get_visit_count(r, 1) == 0, msg);
            snprintf(msg, sizeof msg, "%s frozen: villain's bet node is "
                     "visited", mode_name(mode));
            CHECK(es_get_visit_count(r, 2) > 0, msg);
        }
        snprintf(msg, sizeof msg, "%s frozen: hero's frozen node accumulates "
                 "no strategy mass", mode_name(mode));
        CHECK(es_strategy_sum_mass(r, 0, 0) == 0.0, msg);
        snprintf(msg, sizeof msg, "%s frozen: villain trains at the bet node",
                 mode_name(mode));
        CHECK(es_strategy_sum_mass(r, 2, 0) > 0.0, msg);
        double out[SS_MAX_ACTIONS];
        memset(out, 0, sizeof(out));
        int n = es_final_strategy(r, 0, out, SS_MAX_ACTIONS);
        snprintf(msg, sizeof msg, "%s frozen: hero's frozen node has no "
                 "regrets (uniform final strategy)", mode_name(mode));
        CHECK(n >= 2 && fabs(out[0] - 0.5) < 1e-9 && fabs(out[1] - 0.5) < 1e-9,
              msg);
        /* villain facing the forced bet with AA vs KK on a blank board calls */
        memset(out, 0, sizeof(out));
        n = es_final_strategy(r, 2, out, SS_MAX_ACTIONS);
        snprintf(msg, sizeof msg, "%s frozen: villain (AA vs KK) calls the "
                 "forced bet; call prob=%.3f", mode_name(mode),
                 n >= 2 ? out[1] : -1.0);
        CHECK(n >= 2 && out[1] > 0.95, msg);
        es_free(r);
    }
}

int main(void) {
    poker_eval5_table_init();
    test_carry_terminal_payoffs();
    test_call_is_not_free_across_the_chance_node();
    test_single_street_has_no_carry();
    test_frozen_action_is_honoured();
    printf("[test_es_carry_freeze] passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
