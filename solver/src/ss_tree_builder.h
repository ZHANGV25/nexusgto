/**
 * ss_tree_builder.h — CPU-only tree-builder API for the N-player
 * single-street solver.
 *
 * Extracted from src/cuda/street_solve.cu in Phase R Block B.3
 * (Step 6) so the encoding-alignment test
 * (`tests/test_b3_encoding_alignment.c`) can drive the kernel's tree
 * construction on a Mac dev machine without nvcc. The CUDA file
 * still includes this header and uses the same symbols at link time
 * for GPU paths.
 *
 * Pure structural extraction — NO semantic change beyond the per-
 * level bet_sizes API extension that landed in the same session
 * (Block 1). Byte-identity smoke (`test_phase_r_regression_smoke`,
 * 1036 B) is the regression net for the move.
 *
 * Forward references the SSTreeBuilder struct so callers can
 * stack-allocate it. Keep the layout stable — touching this struct
 * means re-running the byte-identity smoke.
 */
#ifndef SS_TREE_BUILDER_H
#define SS_TREE_BUILDER_H

#include "cuda/street_solve.cuh"

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

    /* OP3 multi-street state. Under single-street mode
     * (depth_rule_enabled=0) only `nodes`, `children`, and `num_leaves`
     * are populated — matches legacy behavior. Under
     * depth_rule_enabled=1, `root_street` and `orig_num_players` drive
     * the §4 depth-rule dispatch, and `chance_card_per_child` accumulates
     * a flat array of valid next-street cards alongside tree building. */
    int depth_rule_enabled;
    int root_street;
    int orig_num_players;
    int *chance_card_per_child;
    int  chance_card_per_child_len;
    int  chance_card_per_child_cap;
} SSTreeBuilder;

void stb_init(SSTreeBuilder *tb);

/* Build N-player single-street betting tree recursively. See
 * src/cuda/street_solve.cu for the original docstring; behavior is
 * unchanged. Per-level `bet_sizes_first` (used when num_raises == 0)
 * vs `bet_sizes_subsequent` (used when num_raises > 0). NULL/0
 * subsequent → falls back to first (matches trainer behavior at
 * mccfr_blueprint.c:1598 when s->num_subsequent_bet_sizes == 0).
 */
/* `stacks` is a per-player array of street-ENTRY stacks (length num_players);
 * the actor's remaining this street is stacks[p] - bets[p]. Replaces the old
 * scalar `stack` (chip-convention §2 fix). */
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
