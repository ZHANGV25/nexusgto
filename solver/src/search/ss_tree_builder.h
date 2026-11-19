/**
 * ss_tree_builder.h — tree-builder API for the N-player search tree
 * (used by ss_tree_api.c).
 *
 * Defines the SSTreeBuilder struct so callers can stack-allocate it.
 * Keep the layout stable — touching this struct means re-running the
 * byte-identity smoke test (`tests/test_depth_limit_smoke.c`).
 */
#ifndef SS_TREE_BUILDER_H
#define SS_TREE_BUILDER_H

#include "tree.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SSTreeBuilder {
    SSNode *nodes;
    int num_nodes;
    int cap_nodes;
    int *children;
    int num_children;
    int cap_children;
    int num_leaves;

    /* Multi-street state. Under single-street mode
     * (depth_rule_enabled=0) only `nodes`, `children`, and `num_leaves`
     * are populated. Under
     * depth_rule_enabled=1, `root_street` and `orig_num_players` drive
     * the Pluribus §4 depth-rule dispatch, and `chance_card_per_child` accumulates
     * a flat array of valid next-street cards alongside tree building. */
    /* Mode bits. Existing callers continue to use literal 0/1 for the depth
     * rule; the additive public wrapper ORs the private stack-aware bit without
     * changing this structure's ABI. */
    int depth_rule_enabled;
    int root_street;
    int orig_num_players;
    /* PLURIBUS-FAITHFUL flop depth limit. The Pluribus paper
     * (Science 2019 supp): "If search is being done on the first betting round,
     * the subgame extends to the end of the round, with LEAF NODES at the chance
     * nodes at the start of the second round" — i.e. the FLOP subgame terminates
     * at a continuation-strategy LEAF at the turn boundary, for ALL player counts,
     * NOT an in-tree turn+river expansion. Extending the 2p flop through turn
     * AND river costs 873k nodes, ~5 orders of magnitude more (the paper's own
     * estimate of depth-limiting's savings).
     *   1 (default under depth_rule_enabled): flop root -> cont-strat leaf at the
     *     turn boundary (Pluribus-faithful, ~21 nodes, sub-ms). 2p flop matches
     *     the 3p+ flop path.
     *   0: 2p flop extends through the chance node (the validated but
     *     expensive reach-CFR path; kept for research and A/B runs).
     * Turn-root extension is governed separately by `turn_depth_limit`. */
    int flop_depth_limit;
    /* PLURIBUS-FAITHFUL turn depth limit. Same rationale as
     * flop_depth_limit, applied to a TURN subgame root: the paper depth-limits
     * EVERY post-first-round street to a cont-strat leaf at the next chance
     * boundary, NOT just the flop. Extending a turn root through the river
     * (in-tree river runout -> ~5k-node 2p turn tree solved via reach-CFR)
     * needs many iterations and is under-converged at interactive speed.
     * Depth-limiting the turn makes it a cont-strat-leaf tree at the RIVER
     * boundary (sub-second), identical in shape to the flop, valued by the
     * same conditioned blueprint rollout.
     *   1 (default under depth_rule_enabled): turn root -> cont-strat leaf at the
     *     river boundary (Pluribus-faithful, sub-second).
     *   0: turn extends through the river chance node (the reach-CFR
     *     research path; POKER_TURN_DEPTH_LIMIT=0). */
    int turn_depth_limit;
    int *chance_card_per_child;
    int  chance_card_per_child_len;
    int  chance_card_per_child_cap;
    /* Sampled-card chance subsets (in-tree flop solve). When a
     * FLOP root extends through chance nodes, every turn chance node deals
     * only `chance_turn_cards` cards (0 = all 49) drawn once per build, and
     * every river chance node under turn card t deals the same
     * `chance_river_cards` cards (0 = all 48) drawn once per t. Betting lines
     * are therefore compared on identical futures, and the resolver's
     * per-iteration child sampling stays uniform over the subset. Turn-root
     * trees are never subset (their river extension is unchanged). */
    int chance_turn_cards;
    int chance_river_cards;
    unsigned long long chance_subset_seed;
    int turn_subset[52];
    int turn_subset_len;        /* -1 until drawn */
    int river_subset[52][52];   /* indexed by turn card */
    int river_subset_len[52];   /* -1 until drawn */
} SSTreeBuilder;

#define STB_MODE_DEPTH_RULE      0x1
#define STB_MODE_UNEQUAL_STACK   0x2

void stb_init(SSTreeBuilder *tb);

/* Build N-player single-street betting tree recursively (see
 * ss_build_tree in tree.h for the arguments). Per-level `bet_sizes_first`
 * (used when num_raises == 0) vs `bet_sizes_subsequent` (used when
 * num_raises > 0). NULL/0 subsequent → falls back to first (matches
 * trainer behavior at mccfr_blueprint.c:1598 when
 * s->num_subsequent_bet_sizes == 0).
 */
/* `stacks` is a per-player array of street-ENTRY stacks (length num_players);
 * the actor's remaining this street is stacks[p] - bets[p]. */
int build_street_tree_n(
    SSTreeBuilder *tb, int is_river, int num_players,
    const int *acting_order, int num_in_order,
    int current_order_idx, int pot, const int *stacks,
    int *bets, int *has_acted, int *active,
    int num_raises,
    const float *bet_sizes_first, int num_bet_sizes_first,
    const float *bet_sizes_subsequent, int num_bet_sizes_subsequent,
    int add_explicit_allin,
    const int *board, int num_board);

int build_chance_node(
    SSTreeBuilder *tb, int num_players,
    const int *acting_order, int num_in_order,
    int pot, const int *stacks, const int *entry_bets,
    const int *active,
    const float *bet_sizes_first, int num_bet_sizes_first,
    const float *bet_sizes_subsequent, int num_bet_sizes_subsequent,
    int add_explicit_allin,
    const int *board, int num_board);

void expand_continuation_leaves_n(SSTreeBuilder *tb, int num_players,
                                   const int *active_at_root);

#ifdef __cplusplus
}
#endif

#endif /* SS_TREE_BUILDER_H */
