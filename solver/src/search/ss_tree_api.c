/* ss_tree_api.c — CPU-portable extern-C tree-build API for street_solve.
 *
 * MOVED (not copied) from src/cuda/street_solve.cu so a single definition is
 * shared by BOTH the nvcc GPU build (street_solve.so) and a gcc CPU-only build
 * (street_solve_cpu.so). The four functions here are pure C (malloc/free only,
 * no CUDA) — they build + free the betting/cont-strat tree that the CPU-ES
 * re-solver consumes via snapshot_tree. Keeping them here (vs duplicated in the
 * .cu) makes drift impossible: edit once, both builds compile it.
 *
 * The GPU solve entry point ss_solve_gpu STAYS in street_solve.cu (it needs the
 * CUDA kernels + the static host helpers ss_splitmix64_host /
 * ss_redraw_chance_samples_host that live there). The CPU .so links a separate
 * ss_cpu_stub.c that aborts on ss_solve_gpu — never a fabricated solve.
 *
 * Linkage note: in a C translation unit, file-scope functions already have
 * external C linkage, so the `extern "C"` qualifier from the .cu is dropped
 * (it is a C++-only construct). The header ss_tree_builder.h (included by both
 * this .c and the .cu) resolves the structs; SS_EXPORT comes from street_solve.cuh.
 */
#include "ss_tree_builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

SS_EXPORT int ss_build_tree(
    const int *board, int num_board,
    int starting_pot, int effective_stack,
    const float *bet_sizes, int num_bet_sizes,
    int num_players, const int *acting_order,
    int is_river, int use_cont_strats,
    int add_explicit_allin,
    SSTreeData *tree_out
) {
    return ss_build_tree2(board, num_board, starting_pot, effective_stack,
                          bet_sizes, num_bet_sizes,
                          NULL, 0,
                          num_players, acting_order,
                          is_river, use_cont_strats,
                          add_explicit_allin, tree_out);
}

static int ss_build_tree_impl(
    const int *board, int num_board,
    int starting_pot, int effective_stack,
    const int *root_stacks, int unequal_stack_mode,
    const float *bet_sizes_first, int num_bet_sizes_first,
    const float *bet_sizes_subsequent, int num_bet_sizes_subsequent,
    int num_players, const int *acting_order,
    int is_river, int use_cont_strats,
    int add_explicit_allin,
    SSTreeData *tree_out
) {
    memset(tree_out, 0, sizeof(SSTreeData));

    /* [SS] tree-build progress logging is OFF by default: it fires on every
     * tree build = every /v1/solve and would flood production stdout. Opt in
     * with POKER_SS_DEBUG=1 to restore the diagnostic prints. */
    const int ss_debug = (getenv("POKER_SS_DEBUG") != NULL);

    SSTreeBuilder tb;
    stb_init(&tb);

    /* OP3 — POKER_SEARCH_DEPTH env var gates the §4 depth-rule
     * extension. Default "single" preserves legacy single-street
     * behavior (what every existing test expects). "full" activates
     * the Pluribus §4 rule: HU flop/turn → rest-of-game, multiway
     * turn → rest-of-game, multiway flop → single-street (2-raise
     * cap unchanged). River is always natural single-street.
     *
     * Note: under "full" the kernel path must also support
     * SS_NODE_CHANCE (commit 4); until then, ss_solve_gpu refuses
     * multi-street trees with a clear error. */
    {
        const char *d = getenv("POKER_SEARCH_DEPTH");
        tb.depth_rule_enabled = (d && strcmp(d, "full") == 0)
                                ? STB_MODE_DEPTH_RULE : 0;
        tb.root_street = num_board;
        tb.orig_num_players = num_players;
        /* PLURIBUS-FAITHFUL flop depth limit (2026-06-15). Default 1: a FLOP
         * subgame terminates at a cont-strat LEAF at the turn boundary (the
         * Science 2019 rule, verified verbatim) for ALL player counts — ~21
         * nodes, sub-ms, vs the legacy 2p-flop full turn+river expansion (873k
         * nodes, ~5 orders of magnitude more per the paper). Set
         * POKER_FLOP_DEPTH_LIMIT=0 to restore the full 2p-flop extension (the
         * validated reach-CFR research path). */
        const char *fdl = getenv("POKER_FLOP_DEPTH_LIMIT");
        tb.flop_depth_limit = (fdl && (fdl[0] == '0' || fdl[0] == 'n' || fdl[0] == 'N'))
                              ? 0 : 1;
        /* PLURIBUS-FAITHFUL turn depth limit (2026-06-18). Default 1: a TURN
         * subgame terminates at a cont-strat LEAF at the river boundary, exactly
         * mirroring the flop — sub-second, valued by the conditioned blueprint
         * rollout. The paper depth-limits EVERY post-first-round street, not just
         * the flop. POKER_TURN_DEPTH_LIMIT=0 restores the legacy river extension
         * (the reach-CFR research path, iter-hungry / under-converged at
         * interactive iters — EC2 probe 2026-06-17). */
        const char *tdl = getenv("POKER_TURN_DEPTH_LIMIT");
        tb.turn_depth_limit = (tdl && (tdl[0] == '0' || tdl[0] == 'n' || tdl[0] == 'N'))
                              ? 0 : 1;
        /* Sampled-card chance subsets (2026-09-17): a flop-root extension
         * deals POKER_CHANCE_TURN_CARDS turn cards and, per turn card,
         * POKER_CHANCE_RIVER_CARDS river cards (0/unset = every card), drawn
         * from POKER_CHANCE_SUBSET_SEED. Keeps the in-tree flop solve's
         * regret storage near a turn tree's instead of 873k nodes. */
        const char *ctc = getenv("POKER_CHANCE_TURN_CARDS");
        const char *crc = getenv("POKER_CHANCE_RIVER_CARDS");
        const char *css = getenv("POKER_CHANCE_SUBSET_SEED");
        tb.chance_turn_cards = ctc ? atoi(ctc) : 0;
        tb.chance_river_cards = crc ? atoi(crc) : 0;
        if (css) tb.chance_subset_seed = strtoull(css, NULL, 10);
    }
    if (unequal_stack_mode)
        tb.depth_rule_enabled |= STB_MODE_UNEQUAL_STACK;

    /* Initialize betting state */
    int bets[SS_MAX_PLAYERS] = {0};
    int has_acted[SS_MAX_PLAYERS] = {0};
    int active[SS_MAX_PLAYERS] = {0};
    for (int i = 0; i < num_players; i++) active[i] = 1;

    /* Per-player street-entry stacks (chip-convention §2 fix). Legacy callers
     * still populate every seat from effective_stack. The additive vector ABI
     * copies root_stacks and separates contesting from can-act state. */
    int stacks[SS_MAX_PLAYERS] = {0};
    for (int i = 0; i < num_players; i++)
        stacks[i] = unequal_stack_mode ? root_stacks[i] : effective_stack;

    build_street_tree_n(&tb, is_river, num_players,
                        acting_order, num_players,
                        0, starting_pot, stacks,
                        bets, has_acted, active,
                        0,
                        bet_sizes_first, num_bet_sizes_first,
                        bet_sizes_subsequent, num_bet_sizes_subsequent,
                        add_explicit_allin,
                        board, num_board);

    if (ss_debug)
        printf("[SS] After betting tree: %d nodes, %d leaves\n", tb.num_nodes, tb.num_leaves);

    if (use_cont_strats && !is_river && tb.num_leaves > 0) {
        expand_continuation_leaves_n(&tb, num_players, active);
        if (ss_debug)
            printf("[SS] After cont strat expansion: %d nodes, %d terminal leaves\n",
                   tb.num_nodes, tb.num_leaves);
    }

    /* Transfer to output */
    tree_out->nodes = tb.nodes;
    tree_out->children = tb.children;
    tree_out->num_nodes = tb.num_nodes;
    tree_out->num_children_total = tb.num_children;
    tree_out->num_leaves = tb.num_leaves;
    tree_out->is_river = is_river;
    tree_out->starting_pot = starting_pot;
    tree_out->effective_stack = effective_stack;
    tree_out->num_players = num_players;
    for (int i = 0; i < num_board; i++) tree_out->board[i] = board[i];
    tree_out->num_board = num_board;

    /* BFS level order */
    int N = tb.num_nodes;
    tree_out->level_order = (int*)malloc(N * sizeof(int));
    tree_out->node_depth = (int*)calloc(N, sizeof(int));
    {
        int *queue = (int*)malloc(N * sizeof(int));
        int qh = 0, qt = 0, li = 0;
        queue[qt++] = 0;
        while (qh < qt) {
            int n = queue[qh++];
            tree_out->level_order[li++] = n;
            int d = tree_out->node_depth[n];
            if (d > tree_out->max_depth) tree_out->max_depth = d;
            SSNode *nd = &tb.nodes[n];
            for (int a = 0; a < nd->num_children; a++) {
                int child = tb.children[nd->first_child + a];
                tree_out->node_depth[child] = d + 1;
                queue[qt++] = child;
            }
        }
        free(queue);
    }

    /* Classify nodes */
    tree_out->num_decision_nodes = 0;
    tree_out->num_showdown_nodes = 0;
    tree_out->num_leaf_nodes = 0;
    tree_out->num_fold_nodes = 0;
    tree_out->num_chance_nodes = 0;
    for (int i = 0; i < N; i++) {
        switch (tb.nodes[i].type) {
            case SS_NODE_DECISION: tree_out->num_decision_nodes++; break;
            case SS_NODE_SHOWDOWN: tree_out->num_showdown_nodes++; break;
            case SS_NODE_LEAF:     tree_out->num_leaf_nodes++; break;
            case SS_NODE_FOLD:     tree_out->num_fold_nodes++; break;
            case SS_NODE_CHANCE:   tree_out->num_chance_nodes++; break;
        }
    }
    tree_out->decision_node_indices = (int*)malloc((tree_out->num_decision_nodes + 1) * sizeof(int));
    tree_out->showdown_node_indices = (int*)malloc((tree_out->num_showdown_nodes + 1) * sizeof(int));
    tree_out->leaf_node_indices = (int*)malloc((tree_out->num_leaf_nodes + 1) * sizeof(int));
    tree_out->fold_node_indices = (int*)malloc((tree_out->num_fold_nodes + 1) * sizeof(int));
    tree_out->chance_node_indices = tree_out->num_chance_nodes > 0
        ? (int*)malloc(tree_out->num_chance_nodes * sizeof(int))
        : NULL;

    int di = 0, si = 0, li2 = 0, fi = 0, ci = 0;
    for (int i = 0; i < N; i++) {
        switch (tb.nodes[i].type) {
            case SS_NODE_DECISION: tree_out->decision_node_indices[di++] = i; break;
            case SS_NODE_SHOWDOWN: tree_out->showdown_node_indices[si++] = i; break;
            case SS_NODE_LEAF:     tree_out->leaf_node_indices[li2++] = i; break;
            case SS_NODE_FOLD:     tree_out->fold_node_indices[fi++] = i; break;
            case SS_NODE_CHANCE:
                tree_out->chance_node_indices[ci] = i;
                tb.nodes[i].chance_node_idx = ci;
                ci++;
                break;
        }
    }

    /* OP3 — transfer chance_card_per_child flat array and subgame-root
     * metadata to tree_out. Under single-street mode tb.chance_card_*
     * are both zero/NULL (matches tree_out's memset). */
    if (tb.chance_card_per_child_len > 0) {
        tree_out->chance_card_per_child = tb.chance_card_per_child;
        tree_out->chance_card_per_child_total = tb.chance_card_per_child_len;
        tb.chance_card_per_child = NULL;  /* ownership transferred */
    }
    tree_out->subgame_root_street = num_board;
    for (int i = 0; i < 3 && i < num_board; i++) {
        tree_out->subgame_root_flop[i] = board[i];
    }

    /* OP3+OP1 commit 5: build the future-round regret remap.
     *
     * For each decision node whose street (num_board) differs from the
     * subgame-root street, assign an abstract-infoset id keyed by
     * (street, player, action_hash_from_root). Concrete decision nodes
     * sharing the key collide to the same abstract id; at CFR traversal
     * they read/write the same shared regret slot via atomicAdd, yielding
     * bucket-pooled regret across board-sampling trajectories — the §2
     * "future-round lossy bucketing" spec.
     *
     * Current-round decision nodes (num_board == subgame_root_street)
     * keep remap = -1 and use the per-hand per-node layout for losslessness
     * per Pluribus §2.
     *
     * The action-path hash propagates top-down via BFS: at each decision
     * node child, hash' = hash * 1315423911 + edge_action; at each chance
     * node child, hash' = hash (pool across sampled boards — that pooling
     * is the whole point of bucket-keyed storage).
     *
     * Bucket dimension (500) is pre-allocated in d_regrets_fut but indexed
     * at bucket=0 in commit 5 — the per-flop PA-EMD centroids land in
     * commit 7; until then all hands collapse to slot 0. */
    if (tree_out->num_chance_nodes > 0) {
        tree_out->future_round_regret_remap = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) tree_out->future_round_regret_remap[i] = -1;

        /* Action-path hash per node, filled during BFS. */
        uint64_t *node_hash = (uint64_t*)calloc((size_t)N, sizeof(uint64_t));

        /* Open-addressing hash map: (key64 -> abstract_id), linear probe.
         * Capacity: 2× num_decision_nodes, rounded to a power of 2. An
         * entry is occupied iff `occ[slot] != 0`. Key stored separately
         * for collision resolution; abstract id stored alongside. */
        int cap = 1;
        while (cap < tree_out->num_decision_nodes * 2) cap <<= 1;
        if (cap < 16) cap = 16;
        uint64_t *map_key = (uint64_t*)calloc((size_t)cap, sizeof(uint64_t));
        int *map_val = (int*)calloc((size_t)cap, sizeof(int));
        unsigned char *map_occ = (unsigned char*)calloc((size_t)cap, 1);

        int next_abstract = 0;

        /* BFS over level_order (already populated above). */
        for (int li = 0; li < N; li++) {
            int nidx = tree_out->level_order[li];
            SSNode *n = &tb.nodes[nidx];

            if (n->type == SS_NODE_DECISION
                && n->num_board != tree_out->subgame_root_street
                && n->num_board > 0) {
                /* Future-round decision: assign / lookup abstract id. */
                uint64_t key = ((uint64_t)n->num_board << 56)
                             ^ ((uint64_t)(n->player & 0xFF) << 48)
                             ^ node_hash[nidx];
                /* Mix further so the key distributes across hash slots. */
                uint64_t kmix = key ^ (key >> 33);
                kmix *= 0xff51afd7ed558ccdULL;
                kmix ^= kmix >> 33;

                unsigned mask = (unsigned)(cap - 1);
                unsigned slot = (unsigned)(kmix & mask);
                int aid = -1;
                while (map_occ[slot]) {
                    if (map_key[slot] == key) { aid = map_val[slot]; break; }
                    slot = (slot + 1) & mask;
                }
                if (aid < 0) {
                    aid = next_abstract++;
                    map_occ[slot] = 1;
                    map_key[slot] = key;
                    map_val[slot] = aid;
                }
                tree_out->future_round_regret_remap[nidx] = aid;
            }

            /* Propagate hash to children. */
            if (n->type == SS_NODE_DECISION) {
                for (int a = 0; a < n->num_children; a++) {
                    int ch = tb.children[n->first_child + a];
                    node_hash[ch] = node_hash[nidx] * 1315423911ULL + (uint64_t)(a + 1);
                }
            } else if (n->type == SS_NODE_CHANCE) {
                /* Chance transitions do NOT participate in the action hash —
                 * that's how future-round decision nodes downstream of
                 * different sampled boards collide to the same abstract id. */
                for (int a = 0; a < n->num_children; a++) {
                    int ch = tb.children[n->first_child + a];
                    node_hash[ch] = node_hash[nidx];
                }
            }
            /* FOLD/SHOWDOWN/LEAF: terminals, no children. */
        }

        tree_out->num_abstract_future_infosets = next_abstract;

        free(node_hash);
        free(map_key);
        free(map_val);
        free(map_occ);

        if (ss_debug)
            printf("[SS] OP3+OP1 commit 5: %d abstract future-round infosets "
                   "(collapsed from future-round decision nodes across sampled boards)\n",
                   next_abstract);
    }

    if (ss_debug)
        printf("[SS] Tree: %d nodes (%d decision, %d fold, %d showdown, %d leaf, %d chance), "
               "%d players, depth=%d\n",
               N, tree_out->num_decision_nodes, tree_out->num_fold_nodes,
               tree_out->num_showdown_nodes, tree_out->num_leaf_nodes,
               tree_out->num_chance_nodes,
               num_players, tree_out->max_depth);

    return 0;
}

SS_EXPORT int ss_build_tree2(
    const int *board, int num_board,
    int starting_pot, int effective_stack,
    const float *bet_sizes_first, int num_bet_sizes_first,
    const float *bet_sizes_subsequent, int num_bet_sizes_subsequent,
    int num_players, const int *acting_order,
    int is_river, int use_cont_strats,
    int add_explicit_allin,
    SSTreeData *tree_out
) {
    return ss_build_tree_impl(
        board, num_board, starting_pot, effective_stack,
        NULL, 0,
        bet_sizes_first, num_bet_sizes_first,
        bet_sizes_subsequent, num_bet_sizes_subsequent,
        num_players, acting_order,
        is_river, use_cont_strats, add_explicit_allin, tree_out);
}

SS_EXPORT int ss_build_tree_stacks(
    const int *board, int num_board,
    int starting_pot, const int *root_stacks,
    const float *bet_sizes_first, int num_bet_sizes_first,
    const float *bet_sizes_subsequent, int num_bet_sizes_subsequent,
    int num_players, const int *acting_order,
    int is_river, int use_cont_strats,
    int add_explicit_allin,
    SSTreeData *tree_out
) {
    if (!tree_out) return -1;
    memset(tree_out, 0, sizeof(*tree_out));

    /* This first coherent slice is deliberately river-only.  Non-river trees
     * terminate in leaf/chance paths whose rollout ABI still has scalar root
     * geometry and cannot settle pre-existing side pots correctly. */
    if (!board || num_board != 5 || !is_river || use_cont_strats
        || starting_pot < 0 || !root_stacks || !acting_order
        || num_players < 2 || num_players > 3
        || num_bet_sizes_first < 0 || num_bet_sizes_subsequent < 0
        || (num_bet_sizes_first > 0 && !bet_sizes_first)
        || (num_bet_sizes_subsequent > 0 && !bet_sizes_subsequent))
        return -1;
    int max_size_count = SS_MAX_ACTIONS - 2
                         - (add_explicit_allin ? 1 : 0);
    if (num_bet_sizes_first > max_size_count
        || num_bet_sizes_subsequent > max_size_count)
        return -1;

    unsigned seen = 0;
    int all_equal = 1;
    for (int i = 0; i < num_players; i++) {
        int p = acting_order[i];
        if (root_stacks[i] < 0 || p < 0 || p >= num_players
            || (seen & (1u << p)))
            return -1;
        seen |= 1u << p;
        if (root_stacks[i] != root_stacks[0]) all_equal = 0;
    }

    /* Positive equal vectors route through the literal legacy builder.  This
     * makes both tree bytes and solve hot loops identical for the equal-stack
     * case; zero-vector roots still need the can-act-aware termination rule. */
    int unequal_stack_mode = !all_equal || root_stacks[0] == 0;
    int effective_stack = all_equal ? root_stacks[0] : -1;
    return ss_build_tree_impl(
        board, num_board, starting_pot, effective_stack,
        root_stacks, unequal_stack_mode,
        bet_sizes_first, num_bet_sizes_first,
        bet_sizes_subsequent, num_bet_sizes_subsequent,
        num_players, acting_order,
        is_river, use_cont_strats, add_explicit_allin, tree_out);
}

SS_EXPORT void ss_free_tree(SSTreeData *td) {
    if (td->nodes) free(td->nodes);
    if (td->children) free(td->children);
    if (td->level_order) free(td->level_order);
    if (td->node_depth) free(td->node_depth);
    if (td->decision_node_indices) free(td->decision_node_indices);
    if (td->showdown_node_indices) free(td->showdown_node_indices);
    if (td->leaf_node_indices) free(td->leaf_node_indices);
    if (td->fold_node_indices) free(td->fold_node_indices);
    if (td->leaf_values) free(td->leaf_values);
    /* OP3 allocations — NULL-safe free, also covers single-street mode
     * where these stay NULL. sampled_child_per_chance is populated in
     * commit 4 (per-iter scratch); listed here so the free path is
     * complete when commit 4 lands. */
    if (td->chance_node_indices) free(td->chance_node_indices);
    if (td->chance_card_per_child) free(td->chance_card_per_child);
    if (td->sampled_child_per_chance) free(td->sampled_child_per_chance);
    if (td->future_round_regret_remap) free(td->future_round_regret_remap);
    memset(td, 0, sizeof(SSTreeData));
}

SS_EXPORT void ss_free_output(SSOutput *out) {
    if (out->root_strategy) free(out->root_strategy);
    if (out->root_ev) free(out->root_ev);
    if (out->avg_strategies) free(out->avg_strategies);
    if (out->avg_strategy_node_ids) free(out->avg_strategy_node_ids);
    if (out->final_strategies) free(out->final_strategies);
    if (out->final_strategy_node_ids) free(out->final_strategy_node_ids);
    memset(out, 0, sizeof(SSOutput));
}
