/**
 * street_solve.cuh — N-player single-street GPU solver
 *
 * Pluribus-style depth-limited search for 2-6 player subgames.
 * Solves ONE betting round on GPU with externally-provided leaf values.
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
 *   choose among 4 continuation strategies. Modeled as N sequential
 *   (but unobserved) decision nodes. CFR converges over all players'
 *   choices jointly.
 */
#ifndef STREET_SOLVE_CUH
#define STREET_SOLVE_CUH

#include <stdint.h>

/* ── Limits ──────────────────────────────────────────────────────────── */

#define SS_MAX_PLAYERS    6       /* max players in a subgame (6-max) */
#define SS_MAX_HANDS      200     /* max hands per player */
#define SS_MAX_ACTIONS    8       /* max actions per decision node */
#define SS_MAX_TREE_NODES 4096    /* max nodes (larger for N-player trees) */
#define SS_MAX_BOARD      5
#define SS_MAX_LEAVES     64      /* max depth-limit leaves before cont expansion */
#define SS_NUM_CONT_STRATS 4      /* unmodified, fold-biased, call-biased, raise-biased */
#define SS_MAX_RAISES     3       /* max raises per street (2p or river) */
/* OP1: per-flop PA-EMD bucket count at future rounds. Pluribus §2 spec:
 * "500 buckets per round, per flop, for the search-time future-round
 * abstraction." The dimension is pre-allocated in d_regrets_fut even in
 * commit 5 (where bucket lookup is stubbed at 0) so commit 7 can wire
 * the per-flop centroid tables without reallocating device memory.
 * See docs/OP3_OP1_MULTI_STREET_PLAN.md §"Bucket-keyed regret storage". */
#define SS_OP1_BUCKETS    500
/* Pluribus Science 2019 supplement §4: multi-way non-river subgames cap at
 * 2 raises ahead to keep the tree tractable. Override via compile-time
 * -DSS_MULTIWAY_MAX_RAISES=<n> if needed for experimentation. */
#ifndef SS_MULTIWAY_MAX_RAISES
#define SS_MULTIWAY_MAX_RAISES 2
#endif

/* Cont-strat tree mode. See docs/OP4_CONT_STRAT_MARGINAL_PLAN.md.
 *   JOINT  — chain of N decision nodes + 4^N terminals per betting leaf.
 *            Legacy path, retained for regression diff. Has a sequential-
 *            observation deviation from Pluribus §4 AND blows up at 5-6p.
 *   SAMPLED — one terminal per betting leaf; leaf-value integrator samples
 *             one joint cont-strat assignment per CFR iteration (MCCFR
 *             external-sampling on the cont-strat chance step). Closes
 *             both deviations. Shipping mode. */
#define SS_CONT_STRAT_MODE_JOINT   0
#define SS_CONT_STRAT_MODE_SAMPLED 1
/* Default SAMPLED per OP4 commit 10 (2026-04-17). JOINT retained
 * for regression diff via -DSS_CONT_STRAT_MODE=0; see
 * docs/OP4_CONT_STRAT_MARGINAL_PLAN.md and docs/OP4_BENCH.md. */
#ifndef SS_CONT_STRAT_MODE
#define SS_CONT_STRAT_MODE SS_CONT_STRAT_MODE_SAMPLED
#endif

/* Node types */
#define SS_NODE_DECISION  0
#define SS_NODE_FOLD      1       /* a player folded; node stores who folded */
#define SS_NODE_SHOWDOWN  2       /* river terminal — N-way showdown */
#define SS_NODE_CHANCE    3       /* card-deal outcome (turn or river). children
                                   * are next-street betting subtree roots, one
                                   * per valid card. Under MCCFR external
                                   * sampling only one child is traversed per
                                   * CFR iteration; selection lives in
                                   * d_sampled_card_per_chance at runtime.
                                   * See docs/OP3_OP1_MULTI_STREET_PLAN.md. */
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
    /* OP3 chance-node fields. SS_NODE_CHANCE only; unused (=0) for other types.
     * children[first_child .. first_child + num_children) are the num_children
     * next-street betting subtree roots; each corresponds to dealing the card
     * indexed by chance_card_per_child[i] (0..51). The child's SSNode.board_cards
     * and num_board already reflect the dealt card. */
    int chance_card_per_child_offset; /* offset into SSTreeData.chance_card_per_child */
    int chance_node_idx;              /* idx into SSTreeData.chance_node_indices; -1 if not SS_NODE_CHANCE */
} SSNode;

/* ── Input to the GPU solver ────────────────────────────────────────── */

typedef struct {
    /* Tree (built by CPU, uploaded to GPU) */
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

    /* A3 strategy freezing (Pluribus safe subgame re-solving).
     * Per-node array: frozen_action[node_idx] = action index that hero took
     * at this node (-1 if not frozen). At frozen nodes, hero's strategy is
     * locked to 100% on the taken action during all CFR iterations.
     * This ensures the re-solve is "safe" — it won't exploit hero's past actions.
     * Set to NULL if no freezing needed. */
    int *frozen_action;

    /* ── OP3 multi-street fields ─────────────────────────────────────────
     * All fields below are zero / NULL under single-street mode (the legacy
     * POKER_SEARCH_DEPTH=single code path). They become populated when
     * ss_build_tree runs under the §4 depth rule (commit 3 onward).
     * See docs/OP3_OP1_MULTI_STREET_PLAN.md. */

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
     * indices. Populated once at tree build; read by the kernel and by
     * the per-iter sampled-card redraw. */
    int *chance_card_per_child;
    int  chance_card_per_child_total;

    /* OP1: per-iter sampled card index for each chance node. Redrawn on
     * host (or via a tiny device kernel) at the start of each CFR
     * iteration — entry i ∈ [0, nodes[chance_node_indices[i]].num_children).
     * Consumed by reach propagation and CFV traversal to select the
     * single sampled child per iter (MCCFR external sampling).
     * NULL under single-street mode. */
    int *sampled_child_per_chance;

    /* OP1: dense abstract-infoset-id remap for future-round decision
     * nodes. Entry i = -1 means "node i is current-round, use the
     * lossless per-hand regret slot as today". Entry i >= 0 means
     * "node i is future-round, use the bucket-keyed shared slot at
     * d_regrets_fut[entry * SS_MAX_ACTIONS]". NULL under
     * single-street mode (all decision nodes are current-round).
     * Length: num_nodes. */
    int *future_round_regret_remap;
    int  num_abstract_future_infosets;

    /* OP1: subgame-root flop used to key the per-flop 500-bucket PA-EMD
     * centroid slice. Copied from the caller-supplied board (first 3
     * cards) at build time. Passed through bps_hand_to_bucket_*_search
     * at CFR traversal for any future-round decision-node bucket lookup. */
    int subgame_root_flop[3];
} SSTreeData;

/* ── Output ──────────────────────────────────────────────────────────── */

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
     * Extracted from d_strategy after the last CFR iteration; regret-matched,
     * not running-averaged. */
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
 * Phase R Block B.3 (Step 6): per-level bet_sizes API.
 *
 * Mirrors the trainer's per-call-site cur_bet_sizes selection
 * (mccfr_blueprint.c:1573-1606): postflop first raise uses
 * `bet_sizes_first`, subsequent raises (num_raises > 0) use
 * `bet_sizes_subsequent`. Closes encoding-alignment Divergence #1
 * documented in docs/PHASE_R_B3_BLOCKED.md.
 *
 * Backward-compat: pass `bet_sizes_subsequent = NULL,
 * num_bet_sizes_subsequent = 0` to replicate `bet_sizes_first` for
 * both raise levels (matches trainer fallthrough at
 * mccfr_blueprint.c:1598 when `s->num_subsequent_bet_sizes == 0`).
 *
 * Preflop tiers do NOT enter this API — kernel root is at flop;
 * preflop bytes come via `root_action_prefix` per the
 * ss_action_hash contract (src/ss_action_hash.h:53-60).
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
 * Solve the N-player single-street tree on GPU using Linear CFR.
 *
 * CFR cycles through all num_players traversers per iteration.
 * Returns 0 on success.
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
