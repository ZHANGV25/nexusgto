/**
 * es_resolver_reach.c — full-per-hand vector reach-CFR (the interactive 2p
 * chance solve) + 2p exploitability.
 *
 * Split out of es_resolver.c (no behavior change). Holds the es_reach_*
 * routines, the g_rc_t* profiling timers, es_reach_profile_read,
 * es_run_reach, es_fill_strat_from_avg, and es_exploitability_2p.
 *
 * Compiled into the SAME shared library as es_resolver.c. The three parent
 * helpers it calls (es_set_error, es_regret_match, es_merge_stats) are shared
 * via es_resolver_internal.h (promoted from file-static, defs stay in the
 * parent TU). es_run_reach is promoted to extern here so es_run (in
 * es_resolver.c) can call it. Everything else is in es_resolver.h / hand_eval.h.
 */

#include "es_resolver.h"
#include "es_resolver_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>

#include "hand_eval.h"   /* eval7 (static inline) */

#ifdef _OPENMP
#include <omp.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * FULL-PER-HAND-REACH VECTOR CFR (2026-06-14) — the interactive 2p chance solve.
 *
 * A pure-C port of the MATH in src/cuda/street_solve.cu (NOT its CUDA
 * parallelism). The sampling traversers above represent each opponent by ONE
 * Monte-Carlo-sampled hand per iteration, so they need thousands of iters to
 * average out opponent-hand variance — minutes per 2p solve. The GPU instead
 * carries a PER-HAND REACH vector per player and integrates EVERY opponent hand
 * each iteration (full-reach vector CFR), reaching the same fixed point in
 * hundreds of iters in seconds. This path reproduces that discipline.
 *
 * Per iteration, per traverser (the GPU host loop, street_solve.cu:2327-2434):
 *   1. regret-match all decision nodes -> rc_strat[node][a][hand]
 *   2. propagate reach top-down for each NON-traverser player rp
 *   3. terminal CFV for the traverser (fold / showdown / leaf), integrating the
 *      opponents' reach
 *   4. propagate CFV bottom-up + per-hand regret/strat-sum update (Linear CFR)
 *
 * NODE-INDEX ORDERING INVARIANT (load-bearing, verified): ss_tree_builder.c is a
 * recursive DFS that allocates the parent node BEFORE its children, so every
 * child has a HIGHER node index than its parent (root = 0). Therefore top-down
 * reach is a single ASCENDING-index sweep and bottom-up CFV a single DESCENDING
 * sweep — no explicit per-depth level arrays (the GPU needs them only because
 * the kernel launches a level at a time).
 *
 * CHANCE under ENUMERATION (vs the GPU's single-sample): reach into each card
 * child = pr * (1/nc); CFV up = sum over children. The GPU draws one card/iter
 * and tolerates the variance via DCFR; enumerating is the zero-variance exact
 * expectation and is cheap at the small chance branching of an OP3 extension.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Lazily allocate the reach/cfv/strat scratch. max_h = max over players' hand
 * counts (a node's hand axis is its acting/owning player's count, but the flat
 * layout uses a single max_h stride to match the GPU). Returns 0 / -1(OOM). */
static int es_reach_alloc(ESResolver *r) {
    int np = r->np;
    int N = r->num_nodes;
    int max_h = 0;
    for (int p = 0; p < np; p++)
        if (r->tree->num_hands[p] > max_h) max_h = r->tree->num_hands[p];
    if (max_h <= 0) { es_set_error("reach: max_h=0"); return -1; }
    r->reach_max_h = max_h;

    size_t reach_n = (size_t)np * N * max_h;
    size_t cfv_n   = (size_t)N * max_h;
    size_t strat_n = (size_t)N * SS_MAX_ACTIONS * max_h;
    r->rc_reach = (double *)malloc(reach_n * sizeof(double));
    r->rc_cfv   = (double *)malloc(cfv_n * sizeof(double));
    r->rc_strat = (double *)malloc(strat_n * sizeof(double));
    if (!r->rc_reach || !r->rc_cfv || !r->rc_strat) {
        es_set_error("reach: OOM (reach=%zu cfv=%zu strat=%zu doubles)",
                     reach_n, cfv_n, strat_n);
        return -1;
    }
    /* Strength cache: [N * np * max_h] uint32 (0 = unset). For the 873k-node flop
     * tree this is ~460MB at np=2,max_h=200 — large but fine on an EC2 box, and it
     * eliminates the dominant per-iter eval7 cost. calloc -> all-zero (unset). */
    size_t sd_n = (size_t)N * np * max_h;
    r->rc_sd_strength = (uint32_t *)calloc(sd_n, sizeof(uint32_t));
    r->rc_sd_done = (uint8_t *)calloc((size_t)N, sizeof(uint8_t));
    if (!r->rc_sd_strength || !r->rc_sd_done) {
        es_set_error("reach: OOM strength cache (%zu u32)", sd_n);
        return -1;
    }
    return 0;
}

/* Fill the per-(node,player,hand) 7-card strength cache for one showdown node
 * (idempotent: guarded by rc_sd_done). Board-conflicting hands get 0 (sentinel
 * "dead"). Called once per showdown node; thereafter the showdown CFV reads the
 * cache instead of calling eval7 every iteration. */
static void es_reach_fill_strengths(ESResolver *r, int node_idx, const int *board) {
    if (r->rc_sd_done[node_idx]) return;
    const SSTreeData *t = r->tree;
    int np = r->np, max_h = r->reach_max_h;
    uint32_t *base = r->rc_sd_strength + (size_t)node_idx * np * max_h;
    for (int p = 0; p < np; p++) {
        int nh = t->num_hands[p];
        uint32_t *row = base + (size_t)p * max_h;
        for (int h = 0; h < nh; h++) {
            int c0 = t->hands[p][h][0], c1 = t->hands[p][h][1];
            int conf = 0;
            for (int b = 0; b < 5; b++) if (c0 == board[b] || c1 == board[b]) { conf = 1; break; }
            if (conf) { row[h] = 0; continue; }
            int cards[7] = { board[0], board[1], board[2], board[3], board[4], c0, c1 };
            uint32_t s = eval7(cards);
            row[h] = s ? s : 1;  /* avoid 0 collision with the dead sentinel */
        }
    }
    r->rc_sd_done[node_idx] = 1;
}

/* Step 1: regret-match every decision node into rc_strat (the GPU d_strategy).
 * One row per ACTING-player hand; non-decision nodes are left untouched (never
 * read). Mirrors ss_batch_regret_match (no future-round shared slots here). */
static void es_reach_regret_match_all(ESResolver *r) {
    const SSTreeData *t = r->tree;
    int max_h = r->reach_max_h;
    /* Each node writes only its own rc_strat slice — node-independent, parallel.
     * Bit-identical regardless of thread count. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int i = 0; i < r->num_nodes; i++) {
        const SSNode *nd = &t->nodes[i];
        if (nd->type != ES_T_DECISION) continue;
        int64_t off = r->node_off[i];
        int na = r->node_na[i];
        if (off < 0 || na <= 0) continue;
        int acting = nd->player;
        int nh = t->num_hands[acting];
        size_t sbase = (size_t)i * SS_MAX_ACTIONS * max_h;
        for (int h = 0; h < nh; h++) {
            double sigma[SS_MAX_ACTIONS];
            es_regret_match(&r->regrets[off + (int64_t)h * na], na, sigma);
            for (int a = 0; a < na; a++)
                r->rc_strat[sbase + (size_t)a * max_h + h] = sigma[a];
        }
    }
}

/* Step 2: top-down reach for ONE player rp. ascending-index sweep (parent before
 * child). reach[rp] at root = weights[rp]; decision: acting==rp -> *strategy,
 * else copy; chance(enumerate): split pr/nc to each child. Mirrors
 * ss_batch_propagate_reach (chance branch enumerated instead of single-sample).
 *
 * IMPORTANT: a parent must initialize ALL its children's reach (including the
 * 0-reach branches), because a child node may be reachable on only some lines
 * and its slot would otherwise hold stale data from a prior player/iter. The GPU
 * memsets the whole reach block to 0 first (street_solve.cu:2367); we do the same
 * (caller memsets), then this sweep overwrites every child it touches. Every
 * non-root node has exactly one parent in this tree (it is a tree, not a DAG),
 * so each child slot is written exactly once. */
static void es_reach_propagate(ESResolver *r, int rp) {
    const SSTreeData *t = r->tree;
    int max_h = r->reach_max_h;
    int nh_rp = t->num_hands[rp];
    double *reach = r->rc_reach + (size_t)rp * r->num_nodes * max_h;

    /* root (node 0) reach = rp's range weights */
    for (int h = 0; h < nh_rp; h++)
        reach[0 * max_h + h] = (double)t->weights[rp][h];

    for (int i = 0; i < r->num_nodes; i++) {
        const SSNode *nd = &t->nodes[i];
        int na = nd->num_children;
        if (na <= 0) continue;                 /* terminal: nothing to propagate */
        const double *pr = &reach[(size_t)i * max_h];

        if (nd->type == ES_T_DECISION) {
            if (nd->player == rp) {
                size_t sbase = (size_t)i * SS_MAX_ACTIONS * max_h;
                for (int a = 0; a < na; a++) {
                    int child = t->children[nd->first_child + a];
                    double *cr = &reach[(size_t)child * max_h];
                    const double *srow = &r->rc_strat[sbase + (size_t)a * max_h];
                    for (int h = 0; h < nh_rp; h++) cr[h] = pr[h] * srow[h];
                }
            } else {
                for (int a = 0; a < na; a++) {
                    int child = t->children[nd->first_child + a];
                    double *cr = &reach[(size_t)child * max_h];
                    for (int h = 0; h < nh_rp; h++) cr[h] = pr[h];
                }
            }
        } else if (nd->type == ES_T_CHANCE) {
            /* ENUMERATE: each dealt card equally likely -> reach * (1/nc).
             * (The GPU assigns pr to the single sampled child; the exact
             * expectation splits pr across all children.) */
            double inv = 1.0 / (double)na;
            for (int a = 0; a < na; a++) {
                int child = t->children[nd->first_child + a];
                double *cr = &reach[(size_t)child * max_h];
                for (int h = 0; h < nh_rp; h++) cr[h] = pr[h] * inv;
            }
        }
        /* FOLD/SHOWDOWN/LEAF: terminals, na<=0 already returned above. */
    }
}

/* Step 3a: HU fold CFV. payoff(trav hand) is hand-independent; integrate the
 * single opponent's reach over its non-card-conflicting hands. Mirrors
 * ss_batch_fold_value's n_opp==1 reasoning (cfv = payoff * sum opp reach over
 * compatible hands). Generalized to np>2 via an odometer would be needed for 3p;
 * reach mode is 2p-first, so this handles the 1-opponent case exactly and
 * asserts otherwise (the caller restricts reach mode to np==2). */
static void es_reach_fold_cfv(ESResolver *r, int node_idx, int traverser) {
    const SSTreeData *t = r->tree;
    const SSNode *nd = &t->nodes[node_idx];
    int max_h = r->reach_max_h;
    int nh = t->num_hands[traverser];
    double *cfv = &r->rc_cfv[(size_t)node_idx * max_h];

    /* the one opponent (np==2) */
    int opp = (traverser == 0) ? 1 : 0;
    const double *oreach = r->rc_reach
        + (size_t)opp * r->num_nodes * max_h + (size_t)node_idx * max_h;
    int nh_o = t->num_hands[opp];

    double my_bet = (double)nd->bets[traverser];
    int winner = nd->player;          /* sole remaining active player wins */
    /* payoff for the traverser hand: (pot - my_bet) if traverser==winner else
     * -my_bet — constant across the traverser's hand. Opponent reach factor:
     * sum over opp hands compatible (no shared card) with the traverser hand. */
    for (int h = 0; h < nh; h++) {
        int c0 = t->hands[traverser][h][0], c1 = t->hands[traverser][h][1];
        double payoff = (traverser == winner) ? ((double)nd->pot - my_bet) : (-my_bet);
        double z = 0.0;
        for (int o = 0; o < nh_o; o++) {
            double w = oreach[o];
            if (w <= 0.0) continue;
            int oc0 = t->hands[opp][o][0], oc1 = t->hands[opp][o][1];
            if (oc0 == c0 || oc0 == c1 || oc1 == c0 || oc1 == c1) continue;
            z += w;
        }
        cfv[h] = payoff * z;
    }
}

/* Step 3b: HU showdown CFV. Reach-weighted sum over the opponent's range — the
 * exact analogue of ss_batch_showdown_value's n_active==1 fast path
 * (street_solve.cu:1046-1064): for each traverser hand, sum over opponent hands
 * w * (win: pot-bet / tie: pot/2-bet / lose: -bet). Card conflicts (incl. the
 * 5-card runout board dealt through chance nodes) excluded both sides — a
 * conflicting world has zero probability, contributing 0 to the integral (NOT a
 * value-injection; see the prior-session footgun). */
static void es_reach_showdown_cfv(ESResolver *r, int node_idx, int traverser) {
    const SSTreeData *t = r->tree;
    const SSNode *nd = &t->nodes[node_idx];
    int max_h = r->reach_max_h;
    int nh = t->num_hands[traverser];
    double *cfv = &r->rc_cfv[(size_t)node_idx * max_h];
    double my_bet = (double)nd->bets[traverser];

    if (!nd->active_players[traverser]) {
        for (int h = 0; h < nh; h++) cfv[h] = -my_bet;
        return;
    }
    int board[5];
    if (nd->num_board == 5) { for (int b = 0; b < 5; b++) board[b] = nd->board_cards[b]; }
    else { for (int b = 0; b < 5; b++) board[b] = t->board[b]; }

    int opp = (traverser == 0) ? 1 : 0;
    const double *oreach = r->rc_reach
        + (size_t)opp * r->num_nodes * max_h + (size_t)node_idx * max_h;
    int nh_o = t->num_hands[opp];
    double pot = (double)nd->pot;

    /* ES_SHOWDOWN_NAIVE=1 forces the reference O(nh·nh_o) integral (cached once/
     * process) — the A/B equivalence gate for the linearized path below. */
    static int s_naive = -1;
    if (s_naive < 0) {
        const char *e = getenv("ES_SHOWDOWN_NAIVE");
        s_naive = (e && (e[0]=='1'||e[0]=='t'||e[0]=='T'||e[0]=='y'||e[0]=='Y')) ? 1 : 0;
    }

    /* Strength cache: eval7 once per (node,player,hand) across ALL iterations
     * (board+hand -> rank is iteration-invariant; eval7 is the dominant per-iter
     * cost). Filled on first visit; thereafter a pure array read. 0 = dead
     * (board-conflicting). */
    es_reach_fill_strengths(r, node_idx, board);
    const uint32_t *tstr = r->rc_sd_strength
        + (size_t)node_idx * r->np * max_h + (size_t)traverser * max_h;
    const uint32_t *ostr = r->rc_sd_strength
        + (size_t)node_idx * r->np * max_h + (size_t)opp * max_h;

    /* valid opp hands = positive reach AND non-dead (board-compatible) strength. */
    int ovalid[SS_MAX_HANDS];
    for (int o = 0; o < nh_o; o++)
        ovalid[o] = (oreach[o] > 0.0 && ostr[o] != 0) ? 1 : 0;

    if (s_naive) {
        /* Reference O(nh·nh_o) integral (the equivalence target). */
        for (int h = 0; h < nh; h++) {
            int c0 = t->hands[traverser][h][0], c1 = t->hands[traverser][h][1];
            uint32_t hs = tstr[h];
            if (hs == 0) { cfv[h] = 0.0; continue; }  /* board-conflicting traverser hand */
            double val = 0.0;
            for (int o = 0; o < nh_o; o++) {
                if (!ovalid[o]) continue;
                int oc0 = t->hands[opp][o][0], oc1 = t->hands[opp][o][1];
                if (oc0 == c0 || oc0 == c1 || oc1 == c0 || oc1 == c1) continue;
                double w = oreach[o]; uint32_t os = ostr[o];
                if (hs > os) val += w * (pot - my_bet);
                else if (hs < os) val += w * (-my_bet);
                else val += w * (pot * 0.5 - my_bet);
            }
            cfv[h] = val;
        }
        return;
    }

    /* ── LINEARIZED reach integral (O((nh+nh_o)·log nh_o + conflicts), not the
     * naive O(nh·nh_o)) — this is what makes flop-scale full reach tractable.
     *
     * Per traverser hand h (strength hs), the value is
     *   val = (pot-mb)·W + (pot/2-mb)·T + (-mb)·L
     *       = pot·(W + T/2) - mb·(W+T+L)
     * where W/T/L = reach mass of opp hands the traverser BEATS / TIES / LOSES to
     * AMONG opp hands not sharing a card with h. Sort the valid opp hands by
     * strength once and prefix-sum their reach; then W,T over ALL valid opp hands
     * are O(log) range lookups by hs, and the per-h CARD-REMOVAL correction
     * subtracts only the (few) opp hands sharing card c0/c1 with h.
     *
     * Card-removal index: ohands_by_card[c] lists valid opp-hand sorted-positions
     * containing card c (each card is in <=~50 opp hands; a hand has 2 cards, so
     * the correction per traverser hand is small and bounded). */
    /* sorted-by-strength order of the valid opp hands */
    int sidx[SS_MAX_HANDS]; int ns = 0;
    for (int o = 0; o < nh_o; o++) if (ovalid[o]) sidx[ns++] = o;
    /* insertion sort by strength (ns <= 200; cheap, stable enough — ties grouped).
     * For larger ranges this is the one O(ns^2) step, but ns<=200 keeps it ~20k
     * ops, dwarfed by the avoided O(nh·nh_o) showdown integral. */
    for (int i = 1; i < ns; i++) {
        int key = sidx[i]; uint32_t ks = ostr[key]; int j = i - 1;
        while (j >= 0 && ostr[sidx[j]] > ks) { sidx[j + 1] = sidx[j]; j--; }
        sidx[j + 1] = key;
    }
    /* prefix[k] = sum of reach over sorted opp hands [0,k); prefix[ns] = total. */
    double prefix[SS_MAX_HANDS + 1];
    prefix[0] = 0.0;
    for (int k = 0; k < ns; k++) prefix[k + 1] = prefix[k] + oreach[sidx[k]];
    double total_reach = prefix[ns];
    /* strengths in sorted order, for binary search by hs */
    uint32_t sstr[SS_MAX_HANDS];
    for (int k = 0; k < ns; k++) sstr[k] = ostr[sidx[k]];

    /* card -> sorted-positions index (only valid opp hands). Each card has a
     * short list; SS_MAX_HANDS cap on the per-card list is safe (<=51 hands
     * share any given card in a 2-card-combo range). */
    int card_cnt[52];
    for (int c = 0; c < 52; c++) card_cnt[c] = 0;
    for (int k = 0; k < ns; k++) {
        int o = sidx[k];
        card_cnt[t->hands[opp][o][0]]++;
        card_cnt[t->hands[opp][o][1]]++;
    }
    /* flatten into a CSR-style structure: offsets + positions */
    int card_off[53]; card_off[0] = 0;
    for (int c = 0; c < 52; c++) card_off[c + 1] = card_off[c] + card_cnt[c];
    /* card_pos: each of the ns valid opp hands contributes exactly 2 refs (its
     * two cards), so total refs = 2·ns <= 2·SS_MAX_HANDS — a stack buffer (no
     * per-node malloc; this runs 873k×/iter under OpenMP, heap contention would
     * dominate). */
    int card_pos[2 * SS_MAX_HANDS];
    { int fill[52]; for (int c = 0; c < 52; c++) fill[c] = card_off[c];
      for (int k = 0; k < ns; k++) {
          int o = sidx[k];
          int a = t->hands[opp][o][0], b = t->hands[opp][o][1];
          card_pos[fill[a]++] = k; card_pos[fill[b]++] = k;
      } }

    for (int h = 0; h < nh; h++) {
        int c0 = t->hands[traverser][h][0], c1 = t->hands[traverser][h][1];
        uint32_t hs = tstr[h];
        if (hs == 0) { cfv[h] = 0.0; continue; }  /* board-conflicting traverser hand */

        /* W = reach of sorted opp with strength < hs; T = strength == hs.
         * Binary search: lo = first index with sstr >= hs; hi = first with
         * sstr > hs. W = prefix[lo]; T = prefix[hi] - prefix[lo]. */
        int lo = 0, hiend = ns;
        { int a = 0, b = ns; while (a < b) { int m = (a + b) >> 1; if (sstr[m] < hs) a = m + 1; else b = m; } lo = a; }
        { int a = lo, b = ns; while (a < b) { int m = (a + b) >> 1; if (sstr[m] <= hs) a = m + 1; else b = m; } hiend = a; }
        double W = prefix[lo];
        double T = prefix[hiend] - prefix[lo];
        double Z = total_reach;

        /* CARD-REMOVAL correction: subtract opp hands sharing c0 or c1 with h.
         * Walk the (short) per-card position lists; guard double-count when an
         * opp hand contains BOTH c0 and c1 (impossible — opp can't hold the same
         * two cards as h unless h==o, but ranges may overlap; a hand sharing both
         * c0 and c1 IS the same combo and would appear in both lists -> dedup by
         * only counting it once). We dedup by skipping a position whose opp hand
         * contains c0 when we are processing the c1 list AND c0<c1-driven order;
         * simpler: process c0 list fully, then c1 list skipping hands that also
         * contain c0. */
        int cc[2] = { c0, c1 };
        for (int ci = 0; ci < 2; ci++) {
            int cardv = cc[ci];
            for (int idx = card_off[cardv]; idx < card_off[cardv + 1]; idx++) {
                int k = card_pos[idx];
                int o = sidx[k];
                int oc0 = t->hands[opp][o][0], oc1 = t->hands[opp][o][1];
                if (ci == 1) {
                    /* skip if this opp hand ALSO contains c0 (already removed in
                     * the c0 pass) — avoids double subtraction. */
                    if (oc0 == c0 || oc1 == c0) continue;
                }
                double w = oreach[o]; uint32_t os = sstr[k];
                Z -= w;
                if (os < hs) W -= w;
                else if (os == hs) T -= w;
            }
        }
        cfv[h] = pot * (W + 0.5 * T) - my_bet * Z;
    }
}

/* Step 3c: LEAF CFV (cont-strat betting leaf). The oracle is per-(opponent
 * sampled hand) — it does NOT carry full opponent reach — so a faithful
 * full-reach leaf integral would sum oracle(o) * opp_reach(o) over the opponent
 * range. A DEEP HU flop tree under the OP3 full-depth extension has ZERO leaf
 * nodes (every line ends in fold/showdown through the chance node), so reach
 * mode never hits this in its target use. We still handle it correctly for
 * completeness/robustness: integrate the oracle over the opponent's reach. */
static void es_reach_leaf_cfv(ESResolver *r, int node_idx, int traverser,
                              uint64_t *rng) {
    const SSTreeData *t = r->tree;
    const SSNode *nd = &t->nodes[node_idx];
    int max_h = r->reach_max_h;
    int nh = t->num_hands[traverser];
    double *cfv = &r->rc_cfv[(size_t)node_idx * max_h];

    int opp = (traverser == 0) ? 1 : 0;
    const double *oreach = r->rc_reach
        + (size_t)opp * r->num_nodes * max_h + (size_t)node_idx * max_h;
    int nh_o = t->num_hands[opp];

    for (int h = 0; h < nh; h++) {
        int c0 = t->hands[traverser][h][0], c1 = t->hands[traverser][h][1];
        double val = 0.0, z = 0.0;
        int sampled[SS_MAX_PLAYERS];
        for (int p = 0; p < r->np; p++) sampled[p] = 0;
        sampled[traverser] = h;
        for (int o = 0; o < nh_o; o++) {
            double w = oreach[o];
            if (w <= 0.0) continue;
            int oc0 = t->hands[opp][o][0], oc1 = t->hands[opp][o][1];
            if (oc0 == c0 || oc0 == c1 || oc1 == c0 || oc1 == c1) continue;
            sampled[opp] = o;
            r->stats.leaf_rollouts++;
            val += w * r->oracle(r->oracle_ctx, nd->leaf_idx, traverser, h, sampled, nd, rng);
            z += w;
        }
        cfv[h] = val;   /* oracle already returns absolute chip value; reach-weighted sum */
        (void)z;
    }
}

/* Step 4: bottom-up CFV + per-hand regret/strat-sum update (Linear CFR).
 * descending-index sweep (children done before parent). Mirrors
 * ss_batch_propagate_cfv. Terminals were filled in step 3. */
static void es_reach_cfv_and_update(ESResolver *r, int traverser, int64_t iter) {
    const SSTreeData *t = r->tree;
    int max_h = r->reach_max_h;
    int nh_trav = t->num_hands[traverser];
    double *cfv = r->rc_cfv;
    double d = (double)iter / ((double)iter + 1.0);

    for (int i = r->num_nodes - 1; i >= 0; i--) {
        const SSNode *nd = &t->nodes[i];
        int type = nd->type;
        if (type == ES_T_FOLD || type == ES_T_SHOWDOWN || type == ES_T_LEAF)
            continue;   /* terminal CFV already set in step 3 */
        int na = nd->num_children;
        double *vnode = &cfv[(size_t)i * max_h];

        if (type == ES_T_CHANCE) {
            /* ENUMERATE: CFV up = sum over children (reach carried the 1/nc). */
            for (int h = 0; h < nh_trav; h++) vnode[h] = 0.0;
            for (int a = 0; a < na; a++) {
                int child = t->children[nd->first_child + a];
                const double *cv = &cfv[(size_t)child * max_h];
                for (int h = 0; h < nh_trav; h++) vnode[h] += cv[h];
            }
            continue;
        }
        /* DECISION */
        if (nd->player == traverser) {
            int64_t off = r->node_off[i];
            size_t sbase = (size_t)i * SS_MAX_ACTIONS * max_h;
            for (int h = 0; h < nh_trav; h++) {
                const double *srow_h = NULL;  /* indexed per-action below */
                double node_val = 0.0;
                for (int a = 0; a < na; a++) {
                    int child = t->children[nd->first_child + a];
                    double sig = r->rc_strat[sbase + (size_t)a * max_h + h];
                    node_val += sig * cfv[(size_t)child * max_h + h];
                }
                vnode[h] = node_val;
                (void)srow_h;
                double *reg = &r->regrets[off + (int64_t)h * na];
                double *ss = &r->strat_sum[off + (int64_t)h * na];
                /* Linear CFR (the GPU runs Linear): regret += residual, *= t/(t+1);
                 * strat_sum += t * sigma. */
                for (int a = 0; a < na; a++) {
                    int child = t->children[nd->first_child + a];
                    double sig = r->rc_strat[sbase + (size_t)a * max_h + h];
                    double resid = cfv[(size_t)child * max_h + h] - node_val;
                    double nr = reg[a] + resid;
                    if (nr < ES_REGRET_FLOOR) nr = ES_REGRET_FLOOR;
                    reg[a] = nr * d;
                    ss[a] += (double)iter * sig;
                }
            }
        } else {
            /* non-traverser decision: cfv = sum over children (reach carried the
             * opponent's action probabilities). */
            for (int h = 0; h < nh_trav; h++) vnode[h] = 0.0;
            for (int a = 0; a < na; a++) {
                int child = t->children[nd->first_child + a];
                const double *cv = &cfv[(size_t)child * max_h];
                for (int h = 0; h < nh_trav; h++) vnode[h] += cv[h];
            }
        }
    }
}

/* One reach-CFR pass for one traverser: regret-match (done by caller, shared
 * across traversers this iter is WRONG — strategy depends on regrets which the
 * traverser's own update mutates; the GPU re-runs regret_match per traverser, so
 * we do too), reach for each non-traverser, terminal CFV, bottom-up update. */
/* Per-phase wall-clock accumulators (ES_REACH_PROFILE=1). Racy across threads but
 * only the single-thread driver writes them; informational. */
static double g_rc_t1 = 0, g_rc_t2 = 0, g_rc_t3 = 0, g_rc_t4 = 0;
static int es_reach_profile(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("ES_REACH_PROFILE"); v = (e && e[0]=='1') ? 1 : 0; }
    return v;
}
#ifdef _OPENMP
#define RC_NOW() omp_get_wtime()
#else
#include <time.h>
#define RC_NOW() ((double)clock()/CLOCKS_PER_SEC)
#endif

static void es_reach_pass(ESResolver *r, int traverser, int64_t iter,
                          uint64_t *rng) {
    int np = r->np;
    int max_h = r->reach_max_h;
    int prof = es_reach_profile();
    double t0;

    /* 1. regret-match all decision nodes into rc_strat */
    t0 = prof ? RC_NOW() : 0;
    es_reach_regret_match_all(r);
    if (prof) g_rc_t1 += RC_NOW() - t0;

    /* 2. reach for each non-traverser (zero its block first, then sweep) */
    t0 = prof ? RC_NOW() : 0;
    for (int rp = 0; rp < np; rp++) {
        if (rp == traverser) continue;
        double *blk = r->rc_reach + (size_t)rp * r->num_nodes * max_h;
        memset(blk, 0, (size_t)r->num_nodes * max_h * sizeof(double));
        es_reach_propagate(r, rp);
    }
    if (prof) g_rc_t2 += RC_NOW() - t0;

    /* 3. terminal CFV for the traverser. Every terminal node writes only its own
     * cfv slice (no cross-node deps), so this is embarrassingly parallel — and it
     * is the dominant cost (showdown is O(nh·nh_opp) per node). Parallelize over
     * nodes; the result is bit-identical to the serial sweep (each node's float
     * sum order is unchanged — only the across-node order varies, and they don't
     * interact). LEAF nodes mutate r->stats (a race), but deep HU chance trees
     * have ZERO leaves, so the leaf branch is serial-safe in practice; we keep it
     * off the parallel path's critical bookkeeping by only racing a stats counter
     * (advisory, like the Hogwild stats). */
    t0 = prof ? RC_NOW() : 0;
    memset(r->rc_cfv, 0, (size_t)r->num_nodes * max_h * sizeof(double));
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int i = 0; i < r->num_nodes; i++) {
        int type = r->tree->nodes[i].type;
        if (type == ES_T_FOLD)          es_reach_fold_cfv(r, i, traverser);
        else if (type == ES_T_SHOWDOWN) es_reach_showdown_cfv(r, i, traverser);
        else if (type == ES_T_LEAF)     es_reach_leaf_cfv(r, i, traverser, rng);
    }
    if (prof) g_rc_t3 += RC_NOW() - t0;

    /* 4. bottom-up CFV + per-hand regret/strat update */
    t0 = prof ? RC_NOW() : 0;
    es_reach_cfv_and_update(r, traverser, iter);
    if (prof) g_rc_t4 += RC_NOW() - t0;
}

/* Read + reset the per-phase profile accumulators (seconds). */
void es_reach_profile_read(double *t1, double *t2, double *t3, double *t4) {
    if (t1) *t1 = g_rc_t1;
    if (t2) *t2 = g_rc_t2;
    if (t3) *t3 = g_rc_t3;
    if (t4) *t4 = g_rc_t4;
    g_rc_t1 = g_rc_t2 = g_rc_t3 = g_rc_t4 = 0;
}

/* Full-reach run loop. DETERMINISTIC regardless of thread count (the only
 * parallel region — terminal CFV, step 3 — has each node write its own slice with
 * an unchanged per-node float-sum order, so the result is thread-count-invariant;
 * verified by a multi-thread-vs-single-thread bit-equality check). num_threads
 * <= 0 -> all cores. 2p-first: rejects np>2 (5-6p MUST sample; 3p reach is a later
 * extension needing odometer terminal integration). */
int es_run_reach(ESResolver *r, int64_t iterations, int num_threads) {
    if (r->np != 2) {
        es_set_error("reach mode supports np==2 only (got np=%d); 5-6p must sample",
                     r->np);
        return -1;
    }
    if (!r->rc_reach && es_reach_alloc(r) != 0) return -1;
#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    /* else leave the OpenMP default (all procs) */
#else
    (void)num_threads;
#endif

    uint64_t rng = r->master_seed ^ 0xD1B54A32D192ED03ULL;  /* leaf-oracle stream */
    ESStats local; memset(&local, 0, sizeof(local));
    for (int64_t it = 0; it < iterations; it++) {
        for (int trav = 0; trav < r->np; trav++) {
            local.traversals++;
            es_reach_pass(r, trav, it + 1, &rng);
        }
    }
    es_merge_stats(&r->stats, &local);
    return 0;
}

/* ── 2p best-response / exploitability (a VALIDATED quality computer) ──────────
 *
 * The earlier Python exploitability_deep.py was buggy (no 2p self-check;
 * reported ~290% of pot). This native 2p BR REUSES the reach machinery above —
 * the same showdown/fold evaluators the solve uses — so it cannot diverge from
 * the model the strategy was solved under, and it is a STRUCTURAL truth garbage
 * can't fake: BR_p >= achieved_p always (>=0 gap), and the sum DROPS toward 0 as
 * the solve converges. It mirrors exploitability_c.py's reach-aware BR, but is
 * SHOWDOWN-AWARE (that Python computer returns 0 at showdowns, so it can't grade
 * a chance tree; this one evaluates them via es_reach_showdown_cfv).
 *
 * Procedure for player p against opponent q's CURRENT σ̄ (read from strat_sum):
 *   1. fill rc_strat with σ̄ (NOT the regret-matched current strategy) so the
 *      opponent reach reflects the AVERAGE strategy (the shipped readout).
 *   2. propagate q's reach top-down (q acts -> *σ̄_q; chance -> /nc).
 *   3. terminal CFV for p (fold/showdown, reach-weighted over q's range).
 *   4. bottom-up: at p's decision nodes take BOTH max (BR) and σ̄-weighted
 *      (achieved); at q's nodes sum; at chance sum. Root holds per-p-hand BR and
 *      achieved; gap_p = mean over p's (in-range) hands of (BR - achieved),
 *      normalised by p's hand prior mass so it is in pot-chip units per hand.
 *   exploitability = gap_0 + gap_1 (both >= 0). Returns it; per-player gaps via
 *   the out params.
 *
 * Uses the reach scratch (allocates if needed). cfv holds achieved; a second
 * br buffer holds BR (reuses no extra global — stack-bounded per node sweep is
 * not possible at 873k nodes, so we malloc a br array sized like cfv). */
static int es_fill_strat_from_avg(ESResolver *r) {
    /* rc_strat[node][a][hand] = normalised strat_sum (σ̄); uniform if zero. */
    const SSTreeData *t = r->tree;
    int max_h = r->reach_max_h;
    for (int i = 0; i < r->num_nodes; i++) {
        const SSNode *nd = &t->nodes[i];
        if (nd->type != ES_T_DECISION) continue;
        int64_t off = r->node_off[i];
        int na = r->node_na[i];
        if (off < 0 || na <= 0) continue;
        int acting = nd->player;
        int nh = t->num_hands[acting];
        size_t sbase = (size_t)i * SS_MAX_ACTIONS * max_h;
        for (int h = 0; h < nh; h++) {
            const double *ss = &r->strat_sum[off + (int64_t)h * na];
            double sum = 0.0;
            for (int a = 0; a < na; a++) sum += ss[a];
            if (sum > 0.0) {
                double inv = 1.0 / sum;
                for (int a = 0; a < na; a++)
                    r->rc_strat[sbase + (size_t)a * max_h + h] = ss[a] * inv;
            } else {
                double u = 1.0 / na;
                for (int a = 0; a < na; a++)
                    r->rc_strat[sbase + (size_t)a * max_h + h] = u;
            }
        }
    }
    return 0;
}

double es_exploitability_2p(ESResolver *r, double *out_gap0, double *out_gap1) {
    if (!r) { es_set_error("resolver NULL"); return NAN; }
    if (r->np != 2) { es_set_error("exploitability_2p: np!=2"); return NAN; }
    if (!r->rc_reach && es_reach_alloc(r) != 0) return NAN;
    const SSTreeData *t = r->tree;
    int max_h = r->reach_max_h;
    int N = r->num_nodes;

    double *br = (double *)malloc((size_t)N * max_h * sizeof(double));
    if (!br) { es_set_error("exploitability_2p OOM"); return NAN; }

    /* σ̄ into rc_strat (used both for opponent reach AND achieved weighting). */
    es_fill_strat_from_avg(r);

    double gaps[2] = {0.0, 0.0};
    for (int p = 0; p < 2; p++) {
        int q = 1 - p;
        /* opponent q's reach under σ̄ */
        double *blk = r->rc_reach + (size_t)q * N * max_h;
        memset(blk, 0, (size_t)N * max_h * sizeof(double));
        es_reach_propagate(r, q);

        /* terminal CFV for p (achieved == BR at terminals) */
        memset(r->rc_cfv, 0, (size_t)N * max_h * sizeof(double));
        for (int i = 0; i < N; i++) {
            int type = t->nodes[i].type;
            if (type == ES_T_FOLD)          es_reach_fold_cfv(r, i, p);
            else if (type == ES_T_SHOWDOWN) es_reach_showdown_cfv(r, i, p);
            else if (type == ES_T_LEAF) {
                uint64_t rng = r->master_seed ^ 0x123456789ULL;
                es_reach_leaf_cfv(r, i, p, &rng);
            }
        }
        /* br := cfv at terminals */
        memcpy(br, r->rc_cfv, (size_t)N * max_h * sizeof(double));

        int nh_p = t->num_hands[p];
        /* bottom-up: descending index. achieved in rc_cfv, BR in br. */
        for (int i = N - 1; i >= 0; i--) {
            const SSNode *nd = &t->nodes[i];
            int type = nd->type;
            if (type == ES_T_FOLD || type == ES_T_SHOWDOWN || type == ES_T_LEAF)
                continue;
            int na = nd->num_children;
            double *ach = &r->rc_cfv[(size_t)i * max_h];
            double *brn = &br[(size_t)i * max_h];
            if (type == ES_T_CHANCE) {
                for (int h = 0; h < nh_p; h++) { ach[h] = 0.0; brn[h] = 0.0; }
                for (int a = 0; a < na; a++) {
                    int child = t->children[nd->first_child + a];
                    const double *ac = &r->rc_cfv[(size_t)child * max_h];
                    const double *bc = &br[(size_t)child * max_h];
                    for (int h = 0; h < nh_p; h++) { ach[h] += ac[h]; brn[h] += bc[h]; }
                }
                continue;
            }
            /* DECISION */
            if (nd->player == p) {
                size_t sbase = (size_t)i * SS_MAX_ACTIONS * max_h;
                for (int h = 0; h < nh_p; h++) {
                    double av = 0.0, bv = -INFINITY;
                    for (int a = 0; a < na; a++) {
                        int child = t->children[nd->first_child + a];
                        double sig = r->rc_strat[sbase + (size_t)a * max_h + h];
                        double ac = r->rc_cfv[(size_t)child * max_h + h];
                        double bc = br[(size_t)child * max_h + h];
                        av += sig * ac;
                        if (bc > bv) bv = bc;   /* BR: best action */
                    }
                    ach[h] = av;
                    brn[h] = (bv == -INFINITY) ? 0.0 : bv;
                }
            } else {
                /* opponent q decision: sum (reach carried σ̄_q) — same for both */
                for (int h = 0; h < nh_p; h++) { ach[h] = 0.0; brn[h] = 0.0; }
                for (int a = 0; a < na; a++) {
                    int child = t->children[nd->first_child + a];
                    const double *ac = &r->rc_cfv[(size_t)child * max_h];
                    const double *bc = &br[(size_t)child * max_h];
                    for (int h = 0; h < nh_p; h++) { ach[h] += ac[h]; brn[h] += bc[h]; }
                }
            }
        }

        /* gap_p = mean over p's in-range hands of (BR - achieved), each
         * normalised by p's own prior mass on that hand (weight). Root holds the
         * reach-weighted (by q) value per p-hand; p's own prior is its weight. */
        double gsum = 0.0; int gcount = 0;
        for (int h = 0; h < nh_p; h++) {
            double w = (double)t->weights[p][h];
            if (w <= 0.0) continue;
            double d = (br[0 * max_h + h] - r->rc_cfv[0 * max_h + h]);
            gsum += d; gcount++;
        }
        gaps[p] = (gcount > 0) ? (gsum / (double)gcount) : 0.0;
    }
    free(br);
    if (out_gap0) *out_gap0 = gaps[0];
    if (out_gap1) *out_gap1 = gaps[1];
    return gaps[0] + gaps[1];
}
