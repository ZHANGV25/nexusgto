/**
 * test_unequal_stack_abi.c -- deterministic unequal-stack live-resolve gates.
 *
 * This test exercises only the additive ABI.  The legacy scalar builder is
 * compared against the new vector builder with equal stacks, then the new
 * geometry is driven through cases that the scalar ABI cannot represent:
 *
 *   1. Equal vector input is byte-for-byte identical to the scalar tree.
 *   2. HU: bet 100, short player calls all-in for 40.
 *   3. HU: one player has zero chips at the street root but remains active.
 *   4. 3p: a root all-in seat is skipped for action but remains contesting.
 *   5. Root/current side pots, ties, folds, and dead money conserve chips.
 * Invalid action widths and malformed root-pot ledgers are also rejected.
 *
 * There is no random input; the test is deterministic by construction.
 */

#include "cuda/street_solve.cuh"
#include "es_resolver.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);            \
            fprintf(stderr, __VA_ARGS__);                                    \
            fputc('\n', stderr);                                             \
            return -1;                                                       \
        }                                                                    \
    } while (0)

static int card(int rank, int suit) {
    return rank * 4 + suit;
}

static int close_double(double actual, double expected) {
    return fabs(actual - expected) <= 1e-9;
}

static double root_pot_value_both(
    int np, const int *board, const int *hand_c0, const int *hand_c1,
    const int *active, const int *bets, int pot, int traverser,
    const int *root_amounts, const uint32_t *root_masks, int n_root)
{
    double scalar = es_test_showdown_value_root_pots(
        np, board, hand_c0, hand_c1, active, bets, pot, traverser,
        root_amounts, root_masks, n_root);
    double vector = es_test_showdown_value_root_pots_vec(
        np, board, hand_c0, hand_c1, active, bets, pot, traverser,
        root_amounts, root_masks, n_root);
    if (isnan(scalar) || isnan(vector) || !close_double(scalar, vector)) {
        fprintf(stderr,
                "FAIL: root-pot scalar/vector mismatch for player %d "
                "(%g vs %g)\n",
                traverser, scalar, vector);
        return NAN;
    }
    return scalar;
}

static int compare_int_payload(const int *a, const int *b, int count,
                               const char *name) {
    if (count <= 0) return 0;
    if (!a || !b) {
        fprintf(stderr, "FAIL: %s pointer differs (%p vs %p)\n",
                name, (const void *)a, (const void *)b);
        return -1;
    }
    if (memcmp(a, b, (size_t)count * sizeof(int)) != 0) {
        fprintf(stderr, "FAIL: %s payload differs (%d ints)\n", name, count);
        return -1;
    }
    return 0;
}

/* Compare every populated tree payload that can differ without comparing the
 * allocation addresses themselves.  The river fixture has no chance nodes,
 * so this is also a strict regression check on the legacy hot path. */
static int compare_tree_payload(const SSTreeData *a, const SSTreeData *b) {
    CHECK(a->num_nodes == b->num_nodes,
          "num_nodes differs: %d vs %d", a->num_nodes, b->num_nodes);
    CHECK(a->num_children_total == b->num_children_total,
          "num_children_total differs: %d vs %d",
          a->num_children_total, b->num_children_total);
    CHECK(a->num_players == b->num_players,
          "num_players differs: %d vs %d", a->num_players, b->num_players);
    CHECK(a->num_leaves == b->num_leaves,
          "num_leaves differs: %d vs %d", a->num_leaves, b->num_leaves);
    CHECK(a->max_depth == b->max_depth,
          "max_depth differs: %d vs %d", a->max_depth, b->max_depth);
    CHECK(a->num_decision_nodes == b->num_decision_nodes,
          "decision count differs: %d vs %d",
          a->num_decision_nodes, b->num_decision_nodes);
    CHECK(a->num_showdown_nodes == b->num_showdown_nodes,
          "showdown count differs: %d vs %d",
          a->num_showdown_nodes, b->num_showdown_nodes);
    CHECK(a->num_leaf_nodes == b->num_leaf_nodes,
          "leaf-node count differs: %d vs %d",
          a->num_leaf_nodes, b->num_leaf_nodes);
    CHECK(a->num_fold_nodes == b->num_fold_nodes,
          "fold count differs: %d vs %d",
          a->num_fold_nodes, b->num_fold_nodes);
    CHECK(a->num_chance_nodes == b->num_chance_nodes,
          "chance count differs: %d vs %d",
          a->num_chance_nodes, b->num_chance_nodes);
    CHECK(a->chance_card_per_child_total == b->chance_card_per_child_total,
          "chance-card count differs: %d vs %d",
          a->chance_card_per_child_total, b->chance_card_per_child_total);
    CHECK(a->num_abstract_future_infosets == b->num_abstract_future_infosets,
          "future-infoset count differs: %d vs %d",
          a->num_abstract_future_infosets,
          b->num_abstract_future_infosets);
    CHECK(a->starting_pot == b->starting_pot,
          "starting_pot differs: %d vs %d", a->starting_pot, b->starting_pot);
    CHECK(a->effective_stack == b->effective_stack,
          "effective_stack differs: %d vs %d",
          a->effective_stack, b->effective_stack);
    CHECK(a->is_river == b->is_river,
          "is_river differs: %d vs %d", a->is_river, b->is_river);
    CHECK(a->subgame_root_street == b->subgame_root_street,
          "root street differs: %d vs %d",
          a->subgame_root_street, b->subgame_root_street);
    CHECK(a->num_board == b->num_board,
          "num_board differs: %d vs %d", a->num_board, b->num_board);
    CHECK(memcmp(a->board, b->board, sizeof(a->board)) == 0,
          "root board payload differs");
    CHECK(memcmp(a->subgame_root_flop, b->subgame_root_flop,
                 sizeof(a->subgame_root_flop)) == 0,
          "root flop payload differs");
    CHECK(memcmp(a->num_hands, b->num_hands, sizeof(a->num_hands)) == 0,
          "num_hands payload differs");
    CHECK(memcmp(a->hands, b->hands, sizeof(a->hands)) == 0,
          "hands payload differs");
    CHECK(memcmp(a->weights, b->weights, sizeof(a->weights)) == 0,
          "weights payload differs");
    CHECK(a->leaf_values == NULL && b->leaf_values == NULL,
          "fresh builder unexpectedly populated leaf_values");
    CHECK(a->frozen_action == NULL && b->frozen_action == NULL,
          "fresh builder unexpectedly populated frozen_action");
    CHECK(a->sampled_child_per_chance == NULL
          && b->sampled_child_per_chance == NULL,
          "fresh builder unexpectedly populated chance samples");

    CHECK(a->nodes && b->nodes, "tree nodes pointer is NULL");
    CHECK(memcmp(a->nodes, b->nodes,
                 (size_t)a->num_nodes * sizeof(SSNode)) == 0,
          "SSNode payload differs");
    CHECK(compare_int_payload(a->children, b->children,
                              a->num_children_total, "children") == 0,
          "children comparison failed");
    CHECK(compare_int_payload(a->level_order, b->level_order,
                              a->num_nodes, "level_order") == 0,
          "level_order comparison failed");
    CHECK(compare_int_payload(a->node_depth, b->node_depth,
                              a->num_nodes, "node_depth") == 0,
          "node_depth comparison failed");
    CHECK(compare_int_payload(a->decision_node_indices,
                              b->decision_node_indices,
                              a->num_decision_nodes,
                              "decision_node_indices") == 0,
          "decision-node index comparison failed");
    CHECK(compare_int_payload(a->showdown_node_indices,
                              b->showdown_node_indices,
                              a->num_showdown_nodes,
                              "showdown_node_indices") == 0,
          "showdown-node index comparison failed");
    CHECK(compare_int_payload(a->leaf_node_indices, b->leaf_node_indices,
                              a->num_leaf_nodes,
                              "leaf_node_indices") == 0,
          "leaf-node index comparison failed");
    CHECK(compare_int_payload(a->fold_node_indices, b->fold_node_indices,
                              a->num_fold_nodes,
                              "fold_node_indices") == 0,
          "fold-node index comparison failed");
    CHECK(compare_int_payload(a->chance_node_indices,
                              b->chance_node_indices,
                              a->num_chance_nodes,
                              "chance_node_indices") == 0,
          "chance-node index comparison failed");
    CHECK(compare_int_payload(a->chance_card_per_child,
                              b->chance_card_per_child,
                              a->chance_card_per_child_total,
                              "chance_card_per_child") == 0,
          "chance-card comparison failed");
    CHECK(compare_int_payload(a->future_round_regret_remap,
                              b->future_round_regret_remap,
                              a->num_chance_nodes > 0 ? a->num_nodes : 0,
                              "future_round_regret_remap") == 0,
          "future-round remap comparison failed");
    return 0;
}

static int test_equal_vector_matches_scalar(void) {
    const int board[5] = {
        card(0, 0), card(5, 1), card(7, 2), card(9, 0), card(10, 3)
    };
    const float first_sizes[2] = { 0.5f, 1.0f };
    const float subsequent_sizes[1] = { 1.0f };
    const int acting_order[2] = { 0, 1 };
    const int stacks[2] = { 350, 350 };
    SSTreeData scalar_tree;
    SSTreeData vector_tree;
    memset(&scalar_tree, 0, sizeof(scalar_tree));
    memset(&vector_tree, 0, sizeof(vector_tree));

    int scalar_rc = ss_build_tree2(
        board, 5, 100, 350,
        first_sizes, 2, subsequent_sizes, 1,
        2, acting_order, 1, 0, 0, &scalar_tree);
    CHECK(scalar_rc == 0, "legacy ss_build_tree2 returned %d", scalar_rc);

    int vector_rc = ss_build_tree_stacks(
        board, 5, 100, stacks,
        first_sizes, 2, subsequent_sizes, 1,
        2, acting_order, 1, 0, 0, &vector_tree);
    CHECK(vector_rc == 0, "ss_build_tree_stacks returned %d", vector_rc);
    CHECK(compare_tree_payload(&scalar_tree, &vector_tree) == 0,
          "equal-stack vector tree differs from scalar tree");

    ss_free_tree(&vector_tree);
    ss_free_tree(&scalar_tree);
    return 0;
}

static int test_two_player_short_allin_call(void) {
    enum { RANK_THREE = 1, RANK_KING = 11, RANK_ACE = 12 };
    const int board[5] = {
        card(0, 0), card(5, 1), card(7, 2), card(9, 0), card(10, 3)
    };
    const float first_sizes[1] = { 1.0f };
    const float too_many_sizes[7] = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f
    };
    const int acting_order[2] = { 0, 1 };
    const int stacks[2] = { 200, 40 };
    SSTreeData tree;
    memset(&tree, 0, sizeof(tree));

    int rc = ss_build_tree_stacks(
        board, 5, 100, stacks,
        too_many_sizes, 7, NULL, 0,
        2, acting_order, 1, 0, 0, &tree);
    CHECK(rc == -1 && tree.nodes == NULL && tree.children == NULL,
          "vector builder accepted more actions than SSNode can store");

    rc = ss_build_tree_stacks(
        board, 5, 100, stacks,
        first_sizes, 1, NULL, 0,
        2, acting_order, 1, 0, 0, &tree);
    CHECK(rc == 0, "short-all-in tree build returned %d", rc);

    CHECK(tree.num_nodes > 0, "short-all-in tree is empty");
    const SSNode *root = &tree.nodes[0];
    CHECK(root->type == SS_NODE_DECISION && root->player == 0,
          "unexpected root: type=%d player=%d", root->type, root->player);
    CHECK(root->num_children == 2,
          "root expected [check, pot-bet], got %d children",
          root->num_children);

    int bet_idx = tree.children[root->first_child + 1];
    const SSNode *facing = &tree.nodes[bet_idx];
    CHECK(facing->type == SS_NODE_DECISION && facing->player == 1,
          "pot-bet child is not short player's decision: type=%d player=%d",
          facing->type, facing->player);
    CHECK(facing->pot == 200 && facing->bets[0] == 100
          && facing->bets[1] == 0,
          "unexpected facing state: pot=%d bets=[%d,%d]",
          facing->pot, facing->bets[0], facing->bets[1]);
    CHECK(facing->num_children == 2,
          "short player expected [fold, all-in call], got %d children",
          facing->num_children);

    int call_idx = tree.children[facing->first_child + 1];
    const SSNode *terminal = &tree.nodes[call_idx];
    CHECK(terminal->type == SS_NODE_SHOWDOWN,
          "short all-in call did not close at showdown (type=%d)",
          terminal->type);
    CHECK(terminal->pot == 240,
          "short all-in call produced pot=%d, expected 240", terminal->pot);
    CHECK(terminal->bets[0] == 100 && terminal->bets[1] == 40,
          "short all-in terminal bets=[%d,%d], expected [100,40]",
          terminal->bets[0], terminal->bets[1]);
    CHECK(terminal->active_players[0] == 1
          && terminal->active_players[1] == 1
          && terminal->num_players == 2,
          "all-in caller stopped contesting the pot");

    /* Common root pot 100 plus matched current contributions 40+40 form a
     * 180-chip contestable pot.  The deep player's unmatched 60 is returned
     * through the current-contribution side-pot layer. */
    const int root_amounts[1] = { 100 };
    const uint32_t root_masks[1] = { UINT32_C(0x3) };
    int hand_c0[2] = { card(RANK_KING, 0), card(RANK_ACE, 0) };
    int hand_c1[2] = { card(RANK_KING, 1), card(RANK_ACE, 1) };
    for (int p = 0; p < 2; p++) {
        tree.num_hands[p] = 1;
        tree.hands[p][0][0] = hand_c0[p];
        tree.hands[p][0][1] = hand_c1[p];
        tree.weights[p][0] = 1.0f;
    }

    ESResolver *resolver = es_create_with_root_pots(
        &tree, NULL, NULL, UINT64_C(0x51DEBEEF),
        root_amounts, root_masks, 1);
    CHECK(resolver != NULL,
          "es_create_with_root_pots rejected the valid HU river tree: %s",
          es_get_last_error());
    CHECK(es_run(resolver, 2, 1) == 0,
          "scalar production traversal rejected root-pot geometry: %s",
          es_get_last_error());
    CHECK(es_iterations_run(resolver) == 2,
          "scalar production traversal did not complete both iterations");
    CHECK(isnan(es_exploitability_2p(resolver, NULL, NULL)),
          "root-pot resolver unexpectedly entered unsupported exploitability");
    es_set_reach_mode(resolver, 1);
    CHECK(es_run(resolver, 1, 1) == -1,
          "root-pot resolver unexpectedly entered unsupported reach mode");
    es_free(resolver);

    resolver = es_create_with_root_pots(
        &tree, NULL, NULL, UINT64_C(0x51DEBEEF),
        root_amounts, root_masks, 1);
    CHECK(resolver != NULL,
          "root-pot resolver recreation failed: %s", es_get_last_error());
    es_set_vec_mode(resolver, 1);
    CHECK(es_run(resolver, 2, 1) == 0,
          "vector production traversal rejected root-pot geometry: %s",
          es_get_last_error());
    CHECK(es_iterations_run(resolver) == 2,
          "vector production traversal did not complete both iterations");
    es_free(resolver);

    const int bad_sum[1] = { 99 };
    resolver = es_create_with_root_pots(
        &tree, NULL, NULL, UINT64_C(1), bad_sum, root_masks, 1);
    CHECK(resolver == NULL,
          "root-pot constructor accepted amounts that do not sum to root pot");
    const uint32_t bad_main_mask[1] = { UINT32_C(0x1) };
    resolver = es_create_with_root_pots(
        &tree, NULL, NULL, UINT64_C(1), root_amounts, bad_main_mask, 1);
    CHECK(resolver == NULL,
          "root-pot constructor accepted a non-common main layer");
    SSTreeData nonriver = tree;
    nonriver.is_river = 0;
    resolver = es_create_with_root_pots(
        &nonriver, NULL, NULL, UINT64_C(1), root_amounts, root_masks, 1);
    CHECK(resolver == NULL,
          "root-pot constructor accepted a non-river tree");

    double deep_loses = root_pot_value_both(
        2, board, hand_c0, hand_c1, terminal->active_players,
        terminal->bets, terminal->pot, 0,
        root_amounts, root_masks, 1);
    double short_wins = root_pot_value_both(
        2, board, hand_c0, hand_c1, terminal->active_players,
        terminal->bets, terminal->pot, 1,
        root_amounts, root_masks, 1);
    CHECK(close_double(deep_loses, -40.0),
          "deep losing net=%g, expected -40", deep_loses);
    CHECK(close_double(short_wins, 140.0),
          "short winning net=%g, expected +140", short_wins);
    CHECK(close_double(deep_loses + short_wins, 100.0),
          "short-win nets do not preserve root pot: %g + %g",
          deep_loses, short_wins);

    hand_c0[0] = card(RANK_ACE, 0);
    hand_c1[0] = card(RANK_ACE, 1);
    hand_c0[1] = card(RANK_THREE, 0);
    hand_c1[1] = card(RANK_THREE, 1);
    double deep_wins = root_pot_value_both(
        2, board, hand_c0, hand_c1, terminal->active_players,
        terminal->bets, terminal->pot, 0,
        root_amounts, root_masks, 1);
    double short_loses = root_pot_value_both(
        2, board, hand_c0, hand_c1, terminal->active_players,
        terminal->bets, terminal->pot, 1,
        root_amounts, root_masks, 1);
    CHECK(close_double(deep_wins, 140.0),
          "deep winning net=%g, expected +140", deep_wins);
    CHECK(close_double(short_loses, -40.0),
          "short losing net=%g, expected -40", short_loses);
    CHECK(close_double(deep_wins + short_loses, 100.0),
          "deep-win nets do not preserve root pot: %g + %g",
          deep_wins, short_loses);

    double short_break_even = -short_loses / (short_wins - short_loses);
    CHECK(close_double(short_break_even, 2.0 / 9.0),
          "short call price=%0.12f, expected 40/180=2/9",
          short_break_even);

    ss_free_tree(&tree);
    return 0;
}

static int test_root_zero_stack_is_terminal(void) {
    enum { RANK_KING = 11, RANK_ACE = 12 };
    const int board[5] = {
        card(0, 0), card(5, 1), card(7, 2), card(9, 0), card(10, 3)
    };
    const float first_sizes[1] = { 1.0f };
    const int acting_order[2] = { 0, 1 };
    const int stacks[2] = { 0, 200 };
    SSTreeData tree;
    memset(&tree, 0, sizeof(tree));

    int rc = ss_build_tree_stacks(
        board, 5, 100, stacks,
        first_sizes, 1, NULL, 0,
        2, acting_order, 1, 0, 0, &tree);
    CHECK(rc == 0, "zero-root-stack tree build returned %d", rc);
    CHECK(tree.num_nodes == 1,
          "zero-root-stack tree has %d nodes, expected one terminal",
          tree.num_nodes);
    const SSNode *root = &tree.nodes[0];
    CHECK(root->type == SS_NODE_SHOWDOWN,
          "zero-root-stack root type=%d, expected showdown", root->type);
    CHECK(root->num_children == 0,
          "zero-root-stack root has %d fake actions", root->num_children);
    CHECK(root->pot == 100 && root->bets[0] == 0 && root->bets[1] == 0,
          "zero-root-stack terminal geometry is pot=%d bets=[%d,%d]",
          root->pot, root->bets[0], root->bets[1]);
    CHECK(root->active_players[0] == 1 && root->active_players[1] == 1
          && root->num_players == 2,
          "zero-stack player is not retained as a showdown contestant");

    const int active[2] = { 1, 1 };
    const int bets[2] = { 0, 0 };
    const int root_amounts[1] = { 100 };
    const uint32_t root_masks[1] = { UINT32_C(0x3) };
    int hand_c0[2] = { card(RANK_ACE, 0), card(RANK_KING, 0) };
    int hand_c1[2] = { card(RANK_ACE, 1), card(RANK_KING, 1) };
    double allin_wins = root_pot_value_both(
        2, board, hand_c0, hand_c1, active, bets, 100, 0,
        root_amounts, root_masks, 1);
    double deep_loses = root_pot_value_both(
        2, board, hand_c0, hand_c1, active, bets, 100, 1,
        root_amounts, root_masks, 1);
    CHECK(close_double(allin_wins, 100.0)
          && close_double(deep_loses, 0.0),
          "root all-in winner values=[%g,%g], expected [100,0]",
          allin_wins, deep_loses);

    hand_c0[0] = card(RANK_KING, 0);
    hand_c1[0] = card(RANK_KING, 1);
    hand_c0[1] = card(RANK_ACE, 0);
    hand_c1[1] = card(RANK_ACE, 1);
    double allin_loses = root_pot_value_both(
        2, board, hand_c0, hand_c1, active, bets, 100, 0,
        root_amounts, root_masks, 1);
    CHECK(close_double(allin_loses, 0.0),
          "root all-in losing value=%g, expected 0", allin_loses);

    hand_c0[0] = card(RANK_ACE, 0);
    hand_c1[0] = card(RANK_ACE, 1);
    hand_c0[1] = card(RANK_ACE, 2);
    hand_c1[1] = card(RANK_ACE, 3);
    double allin_ties = root_pot_value_both(
        2, board, hand_c0, hand_c1, active, bets, 100, 0,
        root_amounts, root_masks, 1);
    CHECK(close_double(allin_ties, 50.0),
          "root all-in tie value=%g, expected 50", allin_ties);

    ss_free_tree(&tree);
    return 0;
}

static int test_sole_responder_cannot_raise(void) {
    const int board[5] = {
        card(0, 0), card(5, 1), card(7, 2), card(9, 0), card(10, 3)
    };
    const float first_sizes[1] = { 1.0f };
    const int acting_order[2] = { 0, 1 };
    const int stacks[2] = { 40, 200 };
    SSTreeData tree;
    memset(&tree, 0, sizeof(tree));

    int rc = ss_build_tree_stacks(
        board, 5, 100, stacks,
        first_sizes, 1, NULL, 0,
        2, acting_order, 1, 0, 0, &tree);
    CHECK(rc == 0, "sole-responder tree build returned %d", rc);

    const SSNode *root = &tree.nodes[0];
    CHECK(root->type == SS_NODE_DECISION && root->player == 0,
          "sole-responder fixture has unexpected root");
    CHECK(root->num_children == 2,
          "short opener expected [check, all-in], got %d children",
          root->num_children);
    int jam_idx = tree.children[root->first_child + 1];
    const SSNode *responder = &tree.nodes[jam_idx];
    CHECK(responder->type == SS_NODE_DECISION && responder->player == 1,
          "all-in did not reach the deep responder");
    CHECK(responder->num_children == 2,
          "sole responder has %d actions, expected fold/call only",
          responder->num_children);

    int call_idx = tree.children[responder->first_child + 1];
    const SSNode *terminal = &tree.nodes[call_idx];
    CHECK(terminal->type == SS_NODE_SHOWDOWN
          && terminal->pot == 180
          && terminal->bets[0] == 40
          && terminal->bets[1] == 40,
          "sole responder's call produced type=%d pot=%d bets=[%d,%d]",
          terminal->type, terminal->pot,
          terminal->bets[0], terminal->bets[1]);

    ss_free_tree(&tree);
    return 0;
}

static int test_three_player_root_allin_is_skipped(void) {
    const int board[5] = {
        card(0, 0), card(5, 1), card(7, 2), card(9, 0), card(10, 3)
    };
    const float first_sizes[1] = { 1.0f };
    const int acting_order[3] = { 0, 1, 2 };
    const int stacks[3] = { 0, 100, 100 };
    SSTreeData tree;
    memset(&tree, 0, sizeof(tree));

    int rc = ss_build_tree_stacks(
        board, 5, 700, stacks,
        first_sizes, 1, NULL, 0,
        3, acting_order, 1, 0, 0, &tree);
    CHECK(rc == 0, "3p root-all-in tree build returned %d", rc);

    const SSNode *root = &tree.nodes[0];
    CHECK(root->type == SS_NODE_DECISION && root->player == 1,
          "root all-in seat was not skipped: type=%d player=%d",
          root->type, root->player);
    CHECK(root->active_players[0] == 1 && root->num_players == 3,
          "root all-in seat stopped contesting before decisions");

    int p1_check_idx = tree.children[root->first_child];
    const SSNode *p2 = &tree.nodes[p1_check_idx];
    CHECK(p2->type == SS_NODE_DECISION && p2->player == 2,
          "check did not rotate to the other live stack");
    CHECK(p2->active_players[0] == 1 && p2->num_players == 3,
          "root all-in seat stopped contesting mid-round");

    int p2_check_idx = tree.children[p2->first_child];
    const SSNode *terminal = &tree.nodes[p2_check_idx];
    CHECK(terminal->type == SS_NODE_SHOWDOWN
          && terminal->active_players[0] == 1
          && terminal->num_players == 3,
          "3p check-through did not retain the all-in contestant");

    const int root_amounts[2] = { 300, 400 };
    const uint32_t root_masks[2] = {
        UINT32_C(0x7), UINT32_C(0x6)
    };
    ESResolver *resolver = es_create_with_root_pots(
        &tree, NULL, NULL, UINT64_C(0x3A11),
        root_amounts, root_masks, 2);
    CHECK(resolver != NULL,
          "constructor rejected valid 3p nested root layers: %s",
          es_get_last_error());
    es_free(resolver);

    const int three_layers[3] = { 300, 200, 200 };
    const uint32_t non_nested_masks[3] = {
        UINT32_C(0x7), UINT32_C(0x6), UINT32_C(0x5)
    };
    resolver = es_create_with_root_pots(
        &tree, NULL, NULL, UINT64_C(1),
        three_layers, non_nested_masks, 3);
    CHECK(resolver == NULL,
          "root-pot constructor accepted non-nested eligibility masks");

    const uint32_t repeated_masks[2] = {
        UINT32_C(0x7), UINT32_C(0x7)
    };
    resolver = es_create_with_root_pots(
        &tree, NULL, NULL, UINT64_C(1),
        root_amounts, repeated_masks, 2);
    CHECK(resolver == NULL,
          "root-pot constructor accepted unconsolidated repeated masks");

    SSNode *mutable_terminal = &tree.nodes[p2_check_idx];
    int saved_active[3];
    memcpy(saved_active, mutable_terminal->active_players,
           sizeof(saved_active));
    mutable_terminal->active_players[0] = 1;
    mutable_terminal->active_players[1] = 0;
    mutable_terminal->active_players[2] = 0;
    resolver = es_create_with_root_pots(
        &tree, NULL, NULL, UINT64_C(1),
        root_amounts, root_masks, 2);
    CHECK(resolver == NULL,
          "root-pot constructor accepted a terminal with no side-pot survivor");
    memcpy(mutable_terminal->active_players, saved_active,
           sizeof(saved_active));

    ss_free_tree(&tree);
    return 0;
}

static int test_three_player_root_side_pot(void) {
    enum { RANK_THREE = 1, RANK_KING = 11, RANK_ACE = 12 };
    const int board[5] = {
        card(0, 0), card(5, 1), card(7, 2), card(9, 0), card(10, 3)
    };
    /* Player 0 has AA and wins only the 300-chip main pot.  Player 1
     * has KK and wins the 400-chip side pot over player 2's 33. */
    const int hand_c0[3] = {
        card(RANK_ACE, 0), card(RANK_KING, 0), card(RANK_THREE, 0)
    };
    const int hand_c1[3] = {
        card(RANK_ACE, 1), card(RANK_KING, 1), card(RANK_THREE, 1)
    };
    const int active[3] = { 1, 1, 1 };
    const int bets[3] = { 0, 0, 0 };
    const int root_amounts[2] = { 300, 400 };
    const uint32_t root_masks[2] = {
        UINT32_C(0x7), /* main: players 0, 1, 2 */
        UINT32_C(0x6)  /* side: players 1, 2 */
    };

    double p0 = root_pot_value_both(
        3, board, hand_c0, hand_c1, active, bets, 700, 0,
        root_amounts, root_masks, 2);
    double p1 = root_pot_value_both(
        3, board, hand_c0, hand_c1, active, bets, 700, 1,
        root_amounts, root_masks, 2);
    double p2 = root_pot_value_both(
        3, board, hand_c0, hand_c1, active, bets, 700, 2,
        root_amounts, root_masks, 2);
    CHECK(close_double(p0, 300.0),
          "main-pot winner net=%g, expected 300", p0);
    CHECK(close_double(p1, 400.0),
          "side-pot winner net=%g, expected 400", p1);
    CHECK(close_double(p2, 0.0),
          "side-pot loser net=%g, expected 0", p2);
    CHECK(close_double(p0 + p1 + p2, 700.0),
          "3p root-layer awards sum to %g, expected 700", p0 + p1 + p2);
    return 0;
}

static int test_root_pots_with_current_bets_and_dead_money(void) {
    enum { RANK_THREE = 1, RANK_KING = 11, RANK_ACE = 12 };
    const int board[5] = {
        card(0, 0), card(5, 1), card(7, 2), card(9, 0), card(10, 3)
    };
    int hand_c0[3] = {
        card(RANK_ACE, 0), card(RANK_KING, 0), card(RANK_THREE, 0)
    };
    int hand_c1[3] = {
        card(RANK_ACE, 1), card(RANK_KING, 1), card(RANK_THREE, 1)
    };
    const int all_active[3] = { 1, 1, 1 };
    const int current_bets[3] = { 0, 50, 100 };
    const int root_amounts[2] = { 300, 400 };
    const uint32_t root_masks[2] = {
        UINT32_C(0x7), UINT32_C(0x6)
    };

    double p0 = root_pot_value_both(
        3, board, hand_c0, hand_c1, all_active, current_bets, 850, 0,
        root_amounts, root_masks, 2);
    double p1 = root_pot_value_both(
        3, board, hand_c0, hand_c1, all_active, current_bets, 850, 1,
        root_amounts, root_masks, 2);
    double p2 = root_pot_value_both(
        3, board, hand_c0, hand_c1, all_active, current_bets, 850, 2,
        root_amounts, root_masks, 2);
    CHECK(close_double(p0, 300.0) && close_double(p1, 450.0)
          && close_double(p2, -50.0),
          "root/current layer values=[%g,%g,%g], expected [300,450,-50]",
          p0, p1, p2);
    CHECK(close_double(p0 + p1 + p2, 700.0),
          "root/current layer nets sum to %g, expected root pot 700",
          p0 + p1 + p2);

    /* Players 1 and 2 tie within the side-pot mask. Player 2's unmatched
     * current 50 remains a one-player return. */
    hand_c0[1] = card(RANK_KING, 0);
    hand_c1[1] = card(RANK_KING, 1);
    hand_c0[2] = card(RANK_KING, 2);
    hand_c1[2] = card(RANK_KING, 3);
    p1 = root_pot_value_both(
        3, board, hand_c0, hand_c1, all_active, current_bets, 850, 1,
        root_amounts, root_masks, 2);
    p2 = root_pot_value_both(
        3, board, hand_c0, hand_c1, all_active, current_bets, 850, 2,
        root_amounts, root_masks, 2);
    CHECK(close_double(p1, 200.0) && close_double(p2, 200.0),
          "side-mask tie nets=[%g,%g], expected [200,200]", p1, p2);

    /* A root-side-pot eligible player folds; the other eligible active player
     * receives that layer even though player 0 has the best hand overall. */
    hand_c0[2] = card(RANK_THREE, 0);
    hand_c1[2] = card(RANK_THREE, 1);
    const int one_folded[3] = { 1, 0, 1 };
    const int no_bets[3] = { 0, 0, 0 };
    p0 = root_pot_value_both(
        3, board, hand_c0, hand_c1, one_folded, no_bets, 700, 0,
        root_amounts, root_masks, 2);
    p1 = root_pot_value_both(
        3, board, hand_c0, hand_c1, one_folded, no_bets, 700, 1,
        root_amounts, root_masks, 2);
    p2 = root_pot_value_both(
        3, board, hand_c0, hand_c1, one_folded, no_bets, 700, 2,
        root_amounts, root_masks, 2);
    CHECK(close_double(p0, 300.0) && close_double(p1, 0.0)
          && close_double(p2, 400.0),
          "folded side eligibility values=[%g,%g,%g]",
          p0, p1, p2);

    /* Folded excess above the highest active contribution is common dead
     * money, not a vanished layer: root 100 + capped current 120 + excess 60. */
    const int excess_active[3] = { 1, 1, 0 };
    const int excess_bets[3] = { 40, 40, 100 };
    const int common_amount[1] = { 100 };
    const uint32_t common_mask[1] = { UINT32_C(0x7) };
    p0 = root_pot_value_both(
        3, board, hand_c0, hand_c1, excess_active, excess_bets, 280, 0,
        common_amount, common_mask, 1);
    p1 = root_pot_value_both(
        3, board, hand_c0, hand_c1, excess_active, excess_bets, 280, 1,
        common_amount, common_mask, 1);
    p2 = root_pot_value_both(
        3, board, hand_c0, hand_c1, excess_active, excess_bets, 280, 2,
        common_amount, common_mask, 1);
    CHECK(close_double(p0, 240.0) && close_double(p1, -40.0)
          && close_double(p2, -100.0),
          "folded-excess nets=[%g,%g,%g], expected [240,-40,-100]",
          p0, p1, p2);
    CHECK(close_double(p0 + p1 + p2, 100.0),
          "folded-excess nets sum to %g, expected root pot 100",
          p0 + p1 + p2);
    return 0;
}

int main(void) {
    if (test_equal_vector_matches_scalar() != 0) return 1;
    if (test_two_player_short_allin_call() != 0) return 1;
    if (test_root_zero_stack_is_terminal() != 0) return 1;
    if (test_sole_responder_cannot_raise() != 0) return 1;
    if (test_three_player_root_allin_is_skipped() != 0) return 1;
    if (test_three_player_root_side_pot() != 0) return 1;
    if (test_root_pots_with_current_bets_and_dead_money() != 0) return 1;

    printf("PASS: unequal-stack tree ABI and 2-3p root side-pot payoffs\n");
    return 0;
}
