/**
 * ss_tree_builder.c — CPU portion of the N-player single-street
 * solver's tree construction. Extracted from src/cuda/street_solve.cu
 * in Phase R Block B.3 (Step 6) so the encoding-alignment test can
 * drive `build_street_tree_n` on Mac without nvcc.
 *
 * Pure structural move — NO semantic change beyond the Block 1
 * per-level bet_sizes API extension. The CUDA file still owns the
 * GPU kernels and the `ss_build_tree`/`ss_build_tree2` extern-C
 * orchestration; it now includes this header for the CPU helpers.
 */

#include "ss_tree_builder.h"

#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * SSTreeBuilder helpers
 * ═══════════════════════════════════════════════════════════════════════ */

void stb_init(SSTreeBuilder *tb) {
    tb->cap_nodes = 512;
    tb->nodes = (SSNode*)malloc(tb->cap_nodes * sizeof(SSNode));
    tb->num_nodes = 0;
    tb->cap_children = 2048;
    tb->children = (int*)malloc(tb->cap_children * sizeof(int));
    tb->num_children = 0;
    tb->num_leaves = 0;
    tb->depth_rule_enabled = 0;
    tb->root_street = 0;
    tb->orig_num_players = 0;
    tb->flop_depth_limit = 1;  /* Pluribus-faithful default (flop -> cont-strat leaf) */
    tb->turn_depth_limit = 1;  /* Pluribus-faithful default (turn -> cont-strat leaf) */
    tb->chance_card_per_child = NULL;
    tb->chance_card_per_child_len = 0;
    tb->chance_card_per_child_cap = 0;
}

static int stb_depth_rule_enabled(const SSTreeBuilder *tb) {
    return (tb->depth_rule_enabled & STB_MODE_DEPTH_RULE) != 0;
}

static int stb_unequal_stack_mode(const SSTreeBuilder *tb) {
    return (tb->depth_rule_enabled & STB_MODE_UNEQUAL_STACK) != 0;
}

static int stb_alloc_chance_cards(SSTreeBuilder *tb, int count) {
    if (tb->chance_card_per_child_len + count > tb->chance_card_per_child_cap) {
        int new_cap = tb->chance_card_per_child_cap == 0
                      ? 256 : tb->chance_card_per_child_cap * 2;
        while (tb->chance_card_per_child_len + count > new_cap) new_cap *= 2;
        tb->chance_card_per_child = (int*)realloc(tb->chance_card_per_child,
                                                  new_cap * sizeof(int));
        tb->chance_card_per_child_cap = new_cap;
    }
    int start = tb->chance_card_per_child_len;
    tb->chance_card_per_child_len += count;
    return start;
}

static int stb_alloc_node(SSTreeBuilder *tb) {
    if (tb->num_nodes >= tb->cap_nodes) {
        tb->cap_nodes *= 2;
        tb->nodes = (SSNode*)realloc(tb->nodes, tb->cap_nodes * sizeof(SSNode));
    }
    int idx = tb->num_nodes++;
    memset(&tb->nodes[idx], 0, sizeof(SSNode));
    tb->nodes[idx].player = -1;
    tb->nodes[idx].leaf_idx = -1;
    tb->nodes[idx].fold_player = -1;
    tb->nodes[idx].chance_node_idx = -1;
    return idx;
}

static int stb_alloc_children(SSTreeBuilder *tb, int count) {
    while (tb->num_children + count > tb->cap_children) {
        tb->cap_children *= 2;
        tb->children = (int*)realloc(tb->children, tb->cap_children * sizeof(int));
    }
    int start = tb->num_children;
    tb->num_children += count;
    return start;
}

static void set_node_board(SSNode *n, const int *board, int num_board) {
    for (int i = 0; i < num_board && i < 5; i++) n->board_cards[i] = board[i];
    n->num_board = num_board;
}

static void set_node_active(SSNode *n, const int *active, int num_players) {
    n->num_players = 0;
    for (int i = 0; i < num_players && i < SS_MAX_PLAYERS; i++) {
        n->active_players[i] = active[i];
        if (active[i]) n->num_players++;
    }
}

/* Count active players */
static int count_active(const int *active, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (active[i]) c++;
    return c;
}

/* Find next active player after `current` in acting order */
static int next_active_player(const int *acting_order, int num_in_order,
                               const int *active, int num_players,
                               int current_order_idx) {
    for (int i = 1; i <= num_in_order; i++) {
        int idx = (current_order_idx + i) % num_in_order;
        int p = acting_order[idx];
        if (p < num_players && active[p]) return idx;
    }
    return -1; /* shouldn't happen if >1 active */
}

/* Unequal-stack action rotation.  `active` means still contesting the pot;
 * stack exhaustion is a separate state and removes a player only from future
 * decisions, never from showdown eligibility. */
static int player_can_act(int player, const int *stacks, const int *bets,
                          const int *active) {
    return active[player] && stacks[player] > bets[player];
}

static int count_can_act(const int *stacks, const int *bets,
                         const int *active, int num_players) {
    int count = 0;
    for (int p = 0; p < num_players; p++)
        if (player_can_act(p, stacks, bets, active)) count++;
    return count;
}

static int next_action_player(const int *acting_order, int num_in_order,
                              const int *stacks, const int *bets,
                              const int *active, int num_players,
                              int current_order_idx) {
    for (int i = 1; i <= num_in_order; i++) {
        int idx = (current_order_idx + i) % num_in_order;
        int p = acting_order[idx];
        if (p < num_players && player_can_act(p, stacks, bets, active))
            return idx;
    }
    return -1;
}

/* Find the max bet among active players */
static int max_bet(const int *bets, const int *active, int n) {
    int mx = 0;
    for (int i = 0; i < n; i++)
        if (active[i] && bets[i] > mx) mx = bets[i];
    return mx;
}

/* OP3 depth-rule dispatch — Pluribus §4 Algorithm 2.
 *
 * Returns 1 iff we should extend the tree past end-of-street via a
 * chance node (SS_NODE_CHANCE). Returns 0 iff we should terminate at
 * the current node — caller then uses the legacy logic (showdown on
 * river, betting-leaf-for-cont-strat elsewhere).
 *
 * Rule summary (subgame-root-street at `tb->root_street`, encoded as
 * num_board: 3=flop, 4=turn, 5=river):
 *   - River: never extend (showdown is natural).
 *   - Flop root, HU: extend to turn.
 *   - Flop root, multi-way (≥3p): terminate at end-of-street betting
 *     leaf; cont-strat rollout takes over. 2-raise cap may also fire
 *     mid-street (existing SS_MULTIWAY_MAX_RAISES logic).
 *   - Turn root: Pluribus-faithful = terminate at a cont-strat LEAF at the
 *     river boundary (turn_depth_limit=1, default); =0 extends to the river
 *     (the reach-CFR research path).
 *   - Past the root street (in an extended subtree): always extend
 *     through to river ("all other cases — end of game"). */
static int should_extend_via_chance(const SSTreeBuilder *tb,
                                    int current_num_board,
                                    int is_river) {
    if (!stb_depth_rule_enabled(tb)) return 0;
    if (is_river || current_num_board >= 5) return 0;
    if (current_num_board == tb->root_street) {
        /* FLOP root: Pluribus-faithful = terminate at a cont-strat LEAF at the
         * turn boundary (flop_depth_limit=1, default) — for ALL player counts,
         * matching the 3p+ flop path. flop_depth_limit=0 restores the legacy 2p
         * full-extension (reach-CFR research path). 3p+ flop already terminated
         * at a leaf (the old `orig_num_players==2` only extended the 2p case),
         * so flop_depth_limit=1 simply drops the 2p exception. */
        if (tb->root_street == 3)
            return tb->flop_depth_limit ? 0 : (tb->orig_num_players == 2);
        /* TURN root: Pluribus-faithful = terminate at a cont-strat LEAF at the
         * river boundary (turn_depth_limit=1, default), exactly mirroring the
         * flop. turn_depth_limit=0 restores the legacy full river extension
         * (reach-CFR research path). */
        if (tb->root_street == 4)
            return tb->turn_depth_limit ? 0 : 1;
        return 1;  /* river root never reaches here (is_river guard above) */
    }
    return current_num_board < 5;
}

/* Check if all active players have acted and bets are equal */
static int round_complete(const int *bets, const int *active, const int *has_acted,
                          int num_players) {
    int mx = max_bet(bets, active, num_players);
    for (int i = 0; i < num_players; i++) {
        if (!active[i]) continue;
        if (!has_acted[i]) return 0;
        if (bets[i] != mx) return 0;
    }
    return 1;
}

/* Unequal-stack completion ignores all-in contestants.  With one player still
 * able to act, the round closes automatically once that player is not facing a
 * bet: poker does not offer a bet into opponents who cannot respond. */
static int round_complete_stacks(const int *stacks, const int *bets,
                                 const int *active, const int *has_acted,
                                 int num_players) {
    int mx = max_bet(bets, active, num_players);
    int n_can_act = 0;
    int sole = -1;
    for (int p = 0; p < num_players; p++) {
        if (!player_can_act(p, stacks, bets, active)) continue;
        n_can_act++;
        sole = p;
    }
    if (n_can_act == 0) return 1;
    if (n_can_act == 1) return bets[sole] >= mx;
    for (int p = 0; p < num_players; p++) {
        if (!player_can_act(p, stacks, bets, active)) continue;
        if (!has_acted[p] || bets[p] != mx) return 0;
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * build_street_tree_n + build_chance_node — public, mutually recursive
 * ═══════════════════════════════════════════════════════════════════════ */

int build_street_tree_n(
    SSTreeBuilder *tb, int is_river, int num_players,
    const int *acting_order, int num_in_order,
    int current_order_idx, int pot, const int *stacks,
    int *bets, int *has_acted, int *active,
    int num_raises,
    const float *bet_sizes_first, int num_bet_sizes_first,
    const float *bet_sizes_subsequent, int num_bet_sizes_subsequent,
    int add_explicit_allin,
    const int *board, int num_board
) {
    /* Per-level bet_sizes selection — Block B.3 closure of
     * encoding-alignment Divergence #1. Matches trainer dispatch at
     * mccfr_blueprint.c:1594-1602 for postflop. Preflop tiers do not
     * enter (kernel root is at flop). */
    const float *bet_sizes;
    int num_bet_sizes;
    if (num_raises == 0 ||
        bet_sizes_subsequent == NULL || num_bet_sizes_subsequent == 0) {
        bet_sizes = bet_sizes_first;
        num_bet_sizes = num_bet_sizes_first;
    } else {
        bet_sizes = bet_sizes_subsequent;
        num_bet_sizes = num_bet_sizes_subsequent;
    }

    int n_active = count_active(active, num_players);

    /* Only 1 player left — they win */
    if (n_active <= 1) {
        int idx = stb_alloc_node(tb);
        tb->nodes[idx].type = SS_NODE_FOLD;
        for (int i = 0; i < num_players; i++)
            if (active[i]) { tb->nodes[idx].player = i; break; }
        tb->nodes[idx].pot = pot;
        for (int i = 0; i < num_players; i++) tb->nodes[idx].bets[i] = bets[i];
        set_node_board(&tb->nodes[idx], board, num_board);
        set_node_active(&tb->nodes[idx], active, num_players);
        return idx;
    }

    /* Check if round is complete */
    int is_complete = stb_unequal_stack_mode(tb)
                      ? round_complete_stacks(
                            stacks, bets, active, has_acted, num_players)
                      : round_complete(
                            bets, active, has_acted, num_players);
    if (is_complete) {
        if (should_extend_via_chance(tb, num_board, is_river)) {
            return build_chance_node(tb, num_players,
                                     acting_order, num_in_order,
                                     pot, stacks, bets, active,
                                     bet_sizes_first, num_bet_sizes_first,
                                     bet_sizes_subsequent, num_bet_sizes_subsequent,
                                     add_explicit_allin,
                                     board, num_board);
        }
        int idx = stb_alloc_node(tb);
        if (is_river) {
            tb->nodes[idx].type = SS_NODE_SHOWDOWN;
        } else {
            tb->nodes[idx].type = SS_NODE_LEAF;
            tb->nodes[idx].leaf_idx = tb->num_leaves++;
        }
        tb->nodes[idx].pot = pot;
        for (int i = 0; i < num_players; i++) tb->nodes[idx].bets[i] = bets[i];
        set_node_board(&tb->nodes[idx], board, num_board);
        set_node_active(&tb->nodes[idx], active, num_players);
        return idx;
    }

    int acting_player = acting_order[current_order_idx];

    if (!active[acting_player]
        || (stb_unequal_stack_mode(tb)
            && !player_can_act(acting_player, stacks, bets, active))) {
        int next_idx = stb_unequal_stack_mode(tb)
                       ? next_action_player(
                             acting_order, num_in_order, stacks, bets, active,
                             num_players, current_order_idx)
                       : next_active_player(
                             acting_order, num_in_order, active, num_players,
                             current_order_idx);
        if (next_idx < 0) {
            int idx = stb_alloc_node(tb);
            tb->nodes[idx].type = SS_NODE_SHOWDOWN;
            tb->nodes[idx].pot = pot;
            for (int i = 0; i < num_players; i++) tb->nodes[idx].bets[i] = bets[i];
            set_node_board(&tb->nodes[idx], board, num_board);
            set_node_active(&tb->nodes[idx], active, num_players);
            return idx;
        }
        return build_street_tree_n(tb, is_river, num_players,
                                    acting_order, num_in_order,
                                    next_idx, pot, stacks,
                                    bets, has_acted, active,
                                    num_raises,
                                    bet_sizes_first, num_bet_sizes_first,
                                    bet_sizes_subsequent, num_bet_sizes_subsequent,
                                    add_explicit_allin,
                                    board, num_board);
    }

    int node = stb_alloc_node(tb);
    tb->nodes[node].type = SS_NODE_DECISION;
    tb->nodes[node].player = acting_player;
    tb->nodes[node].pot = pot;
    for (int i = 0; i < num_players; i++) tb->nodes[node].bets[i] = bets[i];
    set_node_board(&tb->nodes[node], board, num_board);
    set_node_active(&tb->nodes[node], active, num_players);

    int mx = max_bet(bets, active, num_players);
    int to_call = mx - bets[acting_player];
    if (to_call < 0) to_call = 0;

    /* rem = the acting player's REMAINING chips this street (street-entry stack
     * minus what they've already committed). Used to cap calls AND raises at
     * the actor's real all-in, instead of the constant street-entry `stack`
     * the builder used before (§2 stack-tracking fix, see the memo). */
    int rem = stacks[acting_player] - bets[acting_player];
    if (rem < 0) rem = 0;

    int temp_children[16];
    int nc = 0;

    int next_order = stb_unequal_stack_mode(tb)
                     ? next_action_player(
                           acting_order, num_in_order, stacks, bets, active,
                           num_players, current_order_idx)
                     : next_active_player(
                           acting_order, num_in_order, active, num_players,
                           current_order_idx);
    if (next_order < 0) next_order = current_order_idx;

    /* ── Fold (only if facing a bet) ────────────────── */
    if (to_call > 0) {
        int new_active[SS_MAX_PLAYERS];
        memcpy(new_active, active, num_players * sizeof(int));
        new_active[acting_player] = 0;

        int fold_child = build_street_tree_n(tb, is_river, num_players,
                                              acting_order, num_in_order,
                                              next_order, pot, stacks,
                                              bets, has_acted, new_active,
                                              num_raises,
                                              bet_sizes_first, num_bet_sizes_first,
                                              bet_sizes_subsequent, num_bet_sizes_subsequent,
                                              add_explicit_allin,
                                              board, num_board);
        temp_children[nc++] = fold_child;
    }

    /* ── Check or Call ──────────────────────────────── */
    {
        int new_bets[SS_MAX_PLAYERS];
        int new_has_acted[SS_MAX_PLAYERS];
        memcpy(new_bets, bets, num_players * sizeof(int));
        memcpy(new_has_acted, has_acted, num_players * sizeof(int));
        /* Stack-cap the call: a short stack calls all-in for `rem`, not the
         * full to_call (the trainer caps calls, rollout.c:473/492). When
         * rem >= to_call this is identical to the old behaviour
         * (bets[actor]+to_call == mx, pot+to_call). */
        int call_amt = to_call < rem ? to_call : rem;
        new_bets[acting_player] = bets[acting_player] + call_amt;
        new_has_acted[acting_player] = 1;
        int new_pot = pot + call_amt;

        int call_child = build_street_tree_n(tb, is_river, num_players,
                                              acting_order, num_in_order,
                                              next_order, new_pot, stacks,
                                              new_bets, new_has_acted, active,
                                              num_raises,
                                              bet_sizes_first, num_bet_sizes_first,
                                              bet_sizes_subsequent, num_bet_sizes_subsequent,
                                              add_explicit_allin,
                                              board, num_board);
        temp_children[nc++] = call_child;
    }

    /* ── Bet / Raise ────────────────────────────────── */
    int raise_cap = SS_MAX_RAISES;
    if (num_players > 2 && !is_river && num_board < 5) {
        int at_root_street = !stb_depth_rule_enabled(tb)
                             || (num_board == tb->root_street);
        if (at_root_street) raise_cap = SS_MULTIWAY_MAX_RAISES;
    }

    int responders_available = !stb_unequal_stack_mode(tb)
                               || count_can_act(
                                      stacks, bets, active, num_players) > 1;
    if (responders_available && num_raises < raise_cap && nc < 14) {
        int added_allin = 0;
        for (int i = 0; i < num_bet_sizes && nc < 14; i++) {
            int bet_amount;
            if (to_call == 0)
                bet_amount = (int)(bet_sizes[i] * pot);
            else
                bet_amount = to_call + (int)(bet_sizes[i] * (pot + to_call));

            if (bet_amount >= rem) bet_amount = rem;
            if (bet_amount <= to_call) continue;

            if (bet_amount >= rem) {
                if (added_allin) continue;
                added_allin = 1;
            }

            int new_bets[SS_MAX_PLAYERS];
            int new_has_acted[SS_MAX_PLAYERS];
            memcpy(new_bets, bets, num_players * sizeof(int));
            memcpy(new_has_acted, has_acted, num_players * sizeof(int));
            /* CHIP-CONVENTION FIX (docs/SS_TREE_BUILDER_CHIP_CONVENTION_FINDING.md):
             * bet_amount is the trainer's chips-ADDED (already includes to_call,
             * line 326). The trainer ADDS it to the actor's running total
             * (rollout.c:968 bets[p]+=amt), so the level is bets[actor]+amt, NOT
             * mx+amt — the old form overcounted by to_call (=mx-bets[actor]) and
             * inflated the next player's to_call. */
            new_bets[acting_player] = bets[acting_player] + bet_amount;
            new_has_acted[acting_player] = 1;
            for (int p = 0; p < num_players; p++)
                if (p != acting_player && active[p]) new_has_acted[p] = 0;
            int new_pot = pot + bet_amount;

            int raise_child = build_street_tree_n(tb, is_river, num_players,
                                                   acting_order, num_in_order,
                                                   next_order, new_pot, stacks,
                                                   new_bets, new_has_acted, active,
                                                   num_raises + 1,
                                                   bet_sizes_first, num_bet_sizes_first,
                                                   bet_sizes_subsequent, num_bet_sizes_subsequent,
                                                   add_explicit_allin,
                                                   board, num_board);
            temp_children[nc++] = raise_child;
        }

        if (add_explicit_allin && !added_allin && rem > to_call && nc < 14) {
            /* Explicit all-in = the actor's REMAINING stack this street, added
             * to their own running total (chip-convention fix, as above). */
            int ba = rem;
            int new_bets[SS_MAX_PLAYERS];
            int new_has_acted[SS_MAX_PLAYERS];
            memcpy(new_bets, bets, num_players * sizeof(int));
            memcpy(new_has_acted, has_acted, num_players * sizeof(int));
            new_bets[acting_player] = bets[acting_player] + ba;
            new_has_acted[acting_player] = 1;
            for (int p = 0; p < num_players; p++)
                if (p != acting_player && active[p]) new_has_acted[p] = 0;
            int new_pot = pot + ba;

            int ai_child = build_street_tree_n(tb, is_river, num_players,
                                                acting_order, num_in_order,
                                                next_order, new_pot, stacks,
                                                new_bets, new_has_acted, active,
                                                num_raises + 1,
                                                bet_sizes_first, num_bet_sizes_first,
                                                bet_sizes_subsequent, num_bet_sizes_subsequent,
                                                add_explicit_allin,
                                                board, num_board);
            temp_children[nc++] = ai_child;
        }
    }

    int start = stb_alloc_children(tb, nc);
    for (int i = 0; i < nc; i++) tb->children[start + i] = temp_children[i];
    tb->nodes[node].first_child = start;
    tb->nodes[node].num_children = nc;

    return node;
}

int build_chance_node(
    SSTreeBuilder *tb, int num_players,
    const int *acting_order, int num_in_order,
    int pot, const int *stacks, const int *entry_bets,
    const int *active,
    const float *bet_sizes_first, int num_bet_sizes_first,
    const float *bet_sizes_subsequent, int num_bet_sizes_subsequent,
    int add_explicit_allin,
    const int *board, int num_board
) {
    int cidx = stb_alloc_node(tb);
    tb->nodes[cidx].type = SS_NODE_CHANCE;
    tb->nodes[cidx].pot = pot;
    set_node_board(&tb->nodes[cidx], board, num_board);
    set_node_active(&tb->nodes[cidx], active, num_players);
    /* Closing-street contributions (2026-09-09): the next street's bets reset
     * to 0 below, so the resolver reads the closed street's per-player bets
     * from the chance node to charge each player's subgame contribution at
     * river terminals (a turn call must not be free at a river showdown). A
     * chance node has no betting semantics of its own, so the field is free. */
    for (int i = 0; i < num_players; i++)
        tb->nodes[cidx].bets[i] = entry_bets ? entry_bets[i] : 0;

    /* Cross-street stack carryover (2026-06-14, HU over-bet fix). On the
     * NEXT street `bets` reset to 0, so the all-in clamp downstream reads
     * `rem = next_stacks[p] - 0`. If we passed the unchanged street-ENTRY
     * `stacks` through, each player would regain their full entry stack on
     * the next street as if the just-closed street's bets cost nothing —
     * inflating effective stacks, which rewards aggression and makes the HU
     * flop solve OVER-BET (measured: QJs/AhKsTd converged to bet 1.0 vs the
     * blueprint's check 0.96; 3p was unaffected because multiway flop roots
     * terminate at the leaf and never extend through this chance node). The
     * cross-street separately"). Decrement each player's stack by what they
     * committed this street so the next street starts from the true
     * remaining stack. Pot already includes those chips. */
    int next_stacks[SS_MAX_PLAYERS];
    for (int i = 0; i < num_players; i++) {
        next_stacks[i] = stacks[i] - (entry_bets ? entry_bets[i] : 0);
        if (next_stacks[i] < 0) next_stacks[i] = 0;  /* defensive: all-in */
    }

    int on_board[52] = {0};
    for (int i = 0; i < num_board; i++) {
        if (board[i] >= 0 && board[i] < 52) on_board[board[i]] = 1;
    }

    int new_first_order_idx = 0;
    for (int i = 0; i < num_in_order; i++) {
        int p = acting_order[i];
        if (p < num_players && active[p]
            && (!stb_unequal_stack_mode(tb) || next_stacks[p] > 0)) {
            new_first_order_idx = i;
            break;
        }
    }

    int temp_children[64];
    int temp_cards[64];
    int nc = 0;

    for (int c = 0; c < 52 && nc < 52; c++) {
        if (on_board[c]) continue;

        int new_board[5];
        for (int i = 0; i < num_board; i++) new_board[i] = board[i];
        new_board[num_board] = c;
        int new_nb = num_board + 1;
        int new_is_river = (new_nb == 5);

        int new_bets[SS_MAX_PLAYERS] = {0};
        int new_has_acted[SS_MAX_PLAYERS] = {0};
        int child_active[SS_MAX_PLAYERS];
        memcpy(child_active, active, num_players * sizeof(int));

        int child = build_street_tree_n(
            tb, new_is_river, num_players,
            acting_order, num_in_order,
            new_first_order_idx, pot, next_stacks,
            new_bets, new_has_acted, child_active,
            0,
            bet_sizes_first, num_bet_sizes_first,
            bet_sizes_subsequent, num_bet_sizes_subsequent,
            add_explicit_allin,
            new_board, new_nb);

        temp_children[nc] = child;
        temp_cards[nc] = c;
        nc++;
    }

    int start = stb_alloc_children(tb, nc);
    for (int i = 0; i < nc; i++) tb->children[start + i] = temp_children[i];
    tb->nodes[cidx].first_child = start;
    tb->nodes[cidx].num_children = nc;

    int card_offset = stb_alloc_chance_cards(tb, nc);
    for (int i = 0; i < nc; i++) {
        tb->chance_card_per_child[card_offset + i] = temp_cards[i];
    }
    tb->nodes[cidx].chance_card_per_child_offset = card_offset;

    return cidx;
}

void expand_continuation_leaves_n(SSTreeBuilder *tb, int num_players,
                                   const int *active_at_root) {
#if SS_CONT_STRAT_MODE == SS_CONT_STRAT_MODE_SAMPLED
    (void)tb; (void)num_players; (void)active_at_root;
    return;
#else
    int orig_count = tb->num_nodes;
    typedef struct {
        int idx; int pot; int bets[SS_MAX_PLAYERS]; int board[5]; int nb;
        int active[SS_MAX_PLAYERS]; int n_active;
    } LeafInfo;

    LeafInfo *leaves = (LeafInfo*)malloc(orig_count * sizeof(LeafInfo));
    int nl = 0;
    for (int i = 0; i < orig_count; i++) {
        if (tb->nodes[i].type != SS_NODE_LEAF) continue;
        LeafInfo *li = &leaves[nl++];
        li->idx = i;
        li->pot = tb->nodes[i].pot;
        memcpy(li->bets, tb->nodes[i].bets, sizeof(li->bets));
        memcpy(li->board, tb->nodes[i].board_cards, sizeof(li->board));
        li->nb = tb->nodes[i].num_board;
        memcpy(li->active, tb->nodes[i].active_players, sizeof(li->active));
        li->n_active = tb->nodes[i].num_players;
    }

    tb->num_leaves = 0;

    for (int li_idx = 0; li_idx < nl; li_idx++) {
        LeafInfo *li = &leaves[li_idx];

        int active_list[SS_MAX_PLAYERS];
        int n_act = 0;
        for (int p = 0; p < num_players; p++)
            if (li->active[p]) active_list[n_act++] = p;

        if (n_act == 0) continue;

        int orig = li->idx;
        tb->nodes[orig].type = SS_NODE_DECISION;
        tb->nodes[orig].player = active_list[0];
        tb->nodes[orig].leaf_idx = -1;

        int *cur_parents = (int*)malloc(sizeof(int));
        cur_parents[0] = orig;
        int n_parents = 1;

        for (int level = 0; level < n_act; level++) {
            int player = active_list[level];
            int is_last = (level == n_act - 1);
            int n_next = n_parents * SS_NUM_CONT_STRATS;
            int *next_parents = (int*)malloc(n_next * sizeof(int));
            int ni = 0;

            for (int pi = 0; pi < n_parents; pi++) {
                int parent = cur_parents[pi];

                if (level > 0) {
                    tb->nodes[parent].type = SS_NODE_DECISION;
                    tb->nodes[parent].player = player;
                }

                int p_children[SS_NUM_CONT_STRATS];
                for (int s = 0; s < SS_NUM_CONT_STRATS; s++) {
                    if (is_last) {
                        int term = stb_alloc_node(tb);
                        tb->nodes[term].type = SS_NODE_LEAF;
                        tb->nodes[term].leaf_idx = tb->num_leaves++;
                        tb->nodes[term].pot = li->pot;
                        memcpy(tb->nodes[term].bets, li->bets, sizeof(li->bets));
                        set_node_board(&tb->nodes[term], li->board, li->nb);
                        set_node_active(&tb->nodes[term], li->active, num_players);
                        p_children[s] = term;
                    } else {
                        int inter = stb_alloc_node(tb);
                        tb->nodes[inter].pot = li->pot;
                        memcpy(tb->nodes[inter].bets, li->bets, sizeof(li->bets));
                        set_node_board(&tb->nodes[inter], li->board, li->nb);
                        set_node_active(&tb->nodes[inter], li->active, num_players);
                        p_children[s] = inter;
                        next_parents[ni++] = inter;
                    }
                }

                int cs = stb_alloc_children(tb, SS_NUM_CONT_STRATS);
                for (int s = 0; s < SS_NUM_CONT_STRATS; s++)
                    tb->children[cs + s] = p_children[s];
                tb->nodes[parent].first_child = cs;
                tb->nodes[parent].num_children = SS_NUM_CONT_STRATS;
            }

            free(cur_parents);
            if (is_last) {
                free(next_parents);
                cur_parents = NULL;
            } else {
                cur_parents = next_parents;
                n_parents = ni;
            }
        }
        if (cur_parents) free(cur_parents);
    }

    free(leaves);
#endif /* SS_CONT_STRAT_MODE */
}
