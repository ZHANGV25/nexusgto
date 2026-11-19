/**
 * tree.h — search tree layout (SSTreeData, SSNode, SS_* limits).
 *
 * Plain-C tree types for the CPU search: ss_tree_builder.c and ss_tree_api.c
 * build an SSTreeData, and es_resolver.c solves it. The include guard keeps
 * the header's earlier name, STREET_SOLVE_CUH.
 *
 * Pluribus-style depth-limited search for 2-6 player subgames: one betting
 * round (optionally extended through chance nodes) with leaf values supplied
 * from outside the tree.
 *
 * N-player design (matching Pluribus):
 *   - Tree has up to SS_MAX_PLAYERS decision-making players
 *   - CFR cycles through all N players as traversers
 *   - At depth-limit leaves, ALL N remaining players simultaneously
 *     choose among 4 continuation strategies
 *   - Showdown compares all remaining hands (pot split for ties)
 *   - Fold awards pot to remaining players (or last player standing)
 *
 * Betting rotation:
 *   Players act in position order. In a single-street tree:
 *   - Preflop: SB(0), BB(1), UTG(2), MP(3), CO(4), BTN(5)
 *   - Postflop: SB(0), BB(1), UTG(2), ..., BTN(last)
 *   Players who have folded are skipped. The tree builder takes
 *   an array of active player indices and rotates through them.
 *
 * Continuation strategies at leaves (from Pluribus paper):
 *   At each non-river leaf, ALL remaining players simultaneously
 *   choose among 4 continuation strategies. JOINT mode models this as N
 *   sequential (but unobserved) decision nodes; the default SAMPLED mode
 *   keeps one terminal per leaf and samples the joint choice in the leaf
 *   value (see SS_CONT_STRAT_MODE below).
 */
#ifndef STREET_SOLVE_CUH
#define STREET_SOLVE_CUH

#include <stdint.h>

/* ── Limits ──────────────────────────────────────────────────────────── */

#define SS_MAX_PLAYERS    6       /* max players in a subgame (6-max) */
#define SS_MAX_HANDS      1400    /* max hands per player */
#define SS_MAX_ACTIONS    8       /* max actions per decision node */
#define SS_MAX_TREE_NODES 4096    /* max nodes (larger for N-player trees) */
#define SS_MAX_BOARD      5
#define SS_MAX_LEAVES     64      /* max depth-limit leaves before cont expansion */
#define SS_NUM_CONT_STRATS 4      /* unmodified, fold-biased, call-biased, raise-biased */
#define SS_MAX_RAISES     3       /* max raises per street (2p or river) */
/* Per-flop PA-EMD bucket count for future-round decision nodes, following
 * the Pluribus search abstraction (500 buckets per round, per flop). The CPU
 * search does not read it. */
#define SS_OP1_BUCKETS    500
/* Pluribus Science 2019 supplement §4: multi-way non-river subgames cap at
 * 2 raises ahead to keep the tree tractable. Override via compile-time
 * -DSS_MULTIWAY_MAX_RAISES=<n> if needed for experimentation. */
#ifndef SS_MULTIWAY_MAX_RAISES
#define SS_MULTIWAY_MAX_RAISES 2
#endif

/* Cont-strat tree mode.
 *   JOINT  — chain of N decision nodes + 4^N terminals per betting leaf.
 *            Kept for regression comparison. Has a sequential-observation
 *            deviation from Pluribus §4 AND blows up at 5-6p.
 *   SAMPLED — one terminal per betting leaf; leaf-value integrator samples
 *             one joint cont-strat assignment per CFR iteration (MCCFR
 *             external-sampling on the cont-strat chance step). Avoids
 *             both problems. The default. */
#define SS_CONT_STRAT_MODE_JOINT   0
#define SS_CONT_STRAT_MODE_SAMPLED 1
/* Default SAMPLED. Build with -DSS_CONT_STRAT_MODE=0 for JOINT (regression
 * comparison only). */
#ifndef SS_CONT_STRAT_MODE
#define SS_CONT_STRAT_MODE SS_CONT_STRAT_MODE_SAMPLED
#endif

/* Node types */
#define SS_NODE_DECISION  0
#define SS_NODE_FOLD      1       /* a player folded; node stores who folded */
#define SS_NODE_SHOWDOWN  2       /* river terminal — N-way showdown */
#define SS_NODE_CHANCE    3       /* card-deal outcome (turn or river). children
                                   * are next-street betting subtree roots, one
                                   * per valid card. The resolver enumerates
                                   * the children by default; in sampling
                                   * mode it traverses one child per CFR
                                   * iteration (es_chance_sampled_child). */
#define SS_NODE_LEAF      4       /* depth-limit leaf — external value */

/* ── Flat tree node ──────────────────────────────────────────────────── */

typedef struct {
    int type;                     /* SS_NODE_DECISION/FOLD/SHOWDOWN/CHANCE/LEAF */
    int player;                   /* which player acts (0..N-1), or -1 for terminal/chance */
    int num_children;             /* for decision: num actions; for chance: num valid cards */
    int first_child;              /* index into children array */
    int pot;                      /* total pot in chips */
    int bets[SS_MAX_PLAYERS];     /* each player's total bet this street */
    int board_cards[5];
    int num_board;
    int leaf_idx;                 /* for SS_NODE_LEAF: index into leaf_values */
    int fold_player;              /* for SS_NODE_FOLD: which player folded (-1 if N/A) */
    int num_players;              /* number of active (non-folded) players at this node */
    int active_players[SS_MAX_PLAYERS]; /* which players are still active (1=active, 0=folded) */
    /* Chance-node fields. SS_NODE_CHANCE only; unused (=0) for other types.
     * children[first_child .. first_child + num_children) are the num_children
     * next-street betting subtree roots; each corresponds to dealing the card
     * indexed by chance_card_per_child[i] (0..51). The child's SSNode.board_cards
     * and num_board already reflect the dealt card. */
    int chance_card_per_child_offset; /* offset into SSTreeData.chance_card_per_child */
    int chance_node_idx;              /* idx into SSTreeData.chance_node_indices; -1 if not SS_NODE_CHANCE */
} SSNode;

/* ── Search tree (input to the resolver) ────────────────────────────── */

typedef struct {
    /* Tree topology (built by ss_tree_builder.c) */
    SSNode *nodes;
    int *children;
    int num_nodes;
    int num_children_total;

    /* Players */
    int num_players;              /* N: number of players in this subgame (2-6) */

    /* Hands and weights — per player */
    int hands[SS_MAX_PLAYERS][SS_MAX_HANDS][2];
    float weights[SS_MAX_PLAYERS][SS_MAX_HANDS];
    int num_hands[SS_MAX_PLAYERS];

    /* Board */
    int board[SS_MAX_BOARD];
    int num_board;

    /* Leaf values: leaf_values[leaf_idx * num_players * max_hands + p * max_hands + h]
     * EV for player p, hand h, at leaf leaf_idx. */
    float *leaf_values;
    int num_leaves;

    /* BFS level ordering */
    int *level_order;
    int *node_depth;
    int max_depth;

    /* Node type indices */
    int *decision_node_indices;
    int num_decision_nodes;
    int *showdown_node_indices;
    int num_showdown_nodes;
    int *leaf_node_indices;
    int num_leaf_nodes;
    int *fold_node_indices;
    int num_fold_nodes;

    /* Pot and stacks */
    int starting_pot;
    int effective_stack;

    /* Whether this is a river solve */
    int is_river;

    /* Hero-line strategy freezing (Pluribus safe subgame re-solving).
     * Per-node array: frozen_action[node_idx] = action index that hero took
     * at this node (-1 if not frozen). At frozen nodes, hero's strategy is
     * locked to 100% on the taken action during all CFR iterations.
     * This ensures the re-solve is "safe" — it won't exploit hero's past actions.
     * Set to NULL if no freezing needed. */
    int *frozen_action;

    /* ── Multi-street fields ─────────────────────────────────────────────
     * All fields below are zero / NULL under single-street mode
     * (POKER_SEARCH_DEPTH=single). They become populated when ss_build_tree
     * runs under the Pluribus §4 depth rule. */

    /* Subgame root street (3=flop, 4=turn, 5=river) — matches num_board.
     * Drives the depth-rule dispatch inside the tree builder. */
    int subgame_root_street;

    /* Indices (into `nodes`) of all SS_NODE_CHANCE nodes, in BFS order.
     * NULL / num_chance_nodes=0 under single-street mode. */
    int *chance_node_indices;
    int num_chance_nodes;

    /* Per-chance-node child cards: for SSNode at chance_node_indices[i],
     * the valid child-card list lives at
     *   chance_card_per_child[nodes[chance_node_indices[i]].chance_card_per_child_offset ..]
     * running for nodes[...].num_children entries. Values are 0..51 card
     * indices. Populated once at tree build; read by the resolver at
     * chance nodes. */
    int *chance_card_per_child;
    int  chance_card_per_child_total;

    /* Per-iteration sampled child index for each chance node, entry i in
     * [0, nodes[chance_node_indices[i]].num_children). The CPU tree builder
     * leaves it NULL (the resolver derives each sample on demand in
     * es_chance_sampled_child); the field keeps the struct layout. */
    int *sampled_child_per_chance;

    /* Dense abstract-infoset-id remap for future-round decision nodes.
     * Entry i = -1 means "node i is current-round, use the lossless
     * per-hand regret slot". Entry i >= 0 means "node i is future-round
     * and shares abstract infoset `entry` with every node of the same
     * (street, player, action path)". Built by ss_tree_api.c; the CPU
     * resolver does not read it. NULL under single-street mode (all
     * decision nodes are current-round). Length: num_nodes. */
    int *future_round_regret_remap;
    int  num_abstract_future_infosets;

    /* Subgame-root flop used to key the per-flop 500-bucket PA-EMD
     * centroid slice. Copied from the caller-supplied board (first 3
     * cards) at build time. Passed through bps_hand_to_bucket_*_search
     * at CFR traversal for any future-round decision-node bucket lookup. */
    int subgame_root_flop[3];
} SSTreeData;

/* ── Output (ss_solve_gpu ABI; the CPU build never fills it) ─────────── */

typedef struct {
    /* Strategy at the root node — final iteration */
    float *root_strategy;         /* [num_actions][num_hands[root_player]] */
    int root_num_actions;
    int root_player;

    /* Per-hand EV at root for ALL players */
    float *root_ev;               /* [num_players][max_hands] */

    /* Weighted average strategies (sigma-bar) at ALL decision nodes.
     * Used for range narrowing / belief updates. */
    float *avg_strategies;
    int *avg_strategy_node_ids;
    int num_avg_nodes;

    /* Final-iteration strategies at ALL decision nodes (Pluribus: used for
     * play decisions at postflop subgame roots). Parallel to avg_strategies.
     * The last CFR iteration's regret-matched strategy, not running-averaged. */
    float *final_strategies;
    int *final_strategy_node_ids;
    int num_final_nodes;

    /* Number of players (for interpreting arrays) */
    int num_players;
    int max_hands;
} SSOutput;

/* ── Host API ────────────────────────────────────────────────────────── */

#ifdef _WIN32
#define SS_EXPORT __declspec(dllexport)
#else
#define SS_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Build an N-player single-street betting tree on CPU.
 *
 * Args:
 *   board, num_board: current board cards
 *   starting_pot: pot at start of this street (chips, scale=100)
 *   effective_stack: remaining stack per player (assumes equal stacks)
 *   bet_sizes, num_bet_sizes: bet size fractions
 *   num_players: number of active players (2-6)
 *   acting_order: array of player indices in acting order (length num_players)
 *                 e.g., [0,1] for heads-up OOP/IP, [0,1,2] for 3-way
 *   is_river: 1 if river (showdown terminals), 0 otherwise (leaf terminals)
 *   use_cont_strats: 1 to expand leaves with continuation strategy nodes
 *   add_explicit_allin: 1 (default) appends an unconditional all-in
 *     raise child whenever no bet_sizes-derived raise reached the stack;
 *     0 omits it (use when the blueprint was trained with a fixed
 *     bet_sizes set that doesn't include all-in — appending one would
 *     create an off-tree branch the σ̄ rollout falls back to uniform on)
 *   tree_out: output tree data
 */
SS_EXPORT int ss_build_tree(
    const int *board, int num_board,
    int starting_pot, int effective_stack,
    const float *bet_sizes, int num_bet_sizes,
    int num_players, const int *acting_order,
    int is_river, int use_cont_strats,
    int add_explicit_allin,
    SSTreeData *tree_out
);

/**
 * Per-level bet_sizes API.
 *
 * Mirrors the trainer's per-call-site cur_bet_sizes selection
 * (mccfr_blueprint.c:1573-1606): postflop first raise uses
 * `bet_sizes_first`, subsequent raises (num_raises > 0) use
 * `bet_sizes_subsequent`, so the search tree's action encoding matches
 * the trainer's at every raise level.
 *
 * Backward-compat: pass `bet_sizes_subsequent = NULL,
 * num_bet_sizes_subsequent = 0` to replicate `bet_sizes_first` for
 * both raise levels (matches trainer fallthrough at
 * mccfr_blueprint.c:1598 when `s->num_subsequent_bet_sizes == 0`).
 *
 * Preflop tiers do not enter this API: the tree root is at the flop or
 * later.
 *
 * The legacy ss_build_tree() is retained as a thin shim that
 * forwards `bet_sizes_first = bet_sizes, subsequent = NULL/0`.
 */
SS_EXPORT int ss_build_tree2(
    const int *board, int num_board,
    int starting_pot, int effective_stack,
    const float *bet_sizes_first, int num_bet_sizes_first,
    const float *bet_sizes_subsequent, int num_bet_sizes_subsequent,
    int num_players, const int *acting_order,
    int is_river, int use_cont_strats,
    int add_explicit_allin,
    SSTreeData *tree_out
);

/**
 * Additive 2-3 player unequal-stack river-tree ABI.
 *
 * `root_stacks[p]` is player p's remaining stack at the subgame root.  Zero is
 * valid: the player remains active/eligible for showdown but receives no
 * decision node.  Positive equal vectors take the literal legacy tree-builder
 * path.  Unequal-stack flop/turn leaf-oracle settlement is not yet supported,
 * so this entry point currently rejects non-river/use_cont_strats requests.
 * The resulting vector-geometry tree is for the CPU ES resolver together with
 * es_create_with_root_pots; ss_solve_gpu has no root-pot-mask ABI.
 * Existing ss_build_tree/ss_build_tree2 signatures and behavior are unchanged.
 */
SS_EXPORT int ss_build_tree_stacks(
    const int *board, int num_board,
    int starting_pot, const int *root_stacks,
    const float *bet_sizes_first, int num_bet_sizes_first,
    const float *bet_sizes_subsequent, int num_bet_sizes_subsequent,
    int num_players, const int *acting_order,
    int is_river, int use_cont_strats,
    int add_explicit_allin,
    SSTreeData *tree_out
);

/**
 * Solve entry point of the earlier CUDA solver, which is not part of this
 * tree. The symbol stays because the ctypes binding
 * (solver/python/engine/street_solver_gpu.py) looks it up at load; the CPU
 * build links ss_cpu_stub.c, whose ss_solve_gpu aborts. Trees are solved
 * by es_resolver.c.
 */
SS_EXPORT int ss_solve_gpu(
    SSTreeData *tree_data,
    int max_iterations,
    SSOutput *output
);

SS_EXPORT void ss_free_tree(SSTreeData *tree_data);
SS_EXPORT void ss_free_output(SSOutput *output);

#ifdef __cplusplus
}
#endif

#endif /* STREET_SOLVE_CUH */
