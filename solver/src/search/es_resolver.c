/*
 * es_resolver.c — CPU External-Sampling MCCFR subgame re-solver.
 *
 * C port of python/es_resolver/es_solver.py. VANILLA external sampling (the
 * Python `linear_cfr` flag is dead code; see es_resolver.h header + memory
 * es-resolver-vanilla-not-linear-cfr). Same algorithm → same fixed point.
 *
 * Reuses primitives from mccfr_blueprint.c (xorshift RNG, regret matching by
 * convention) and the SSTreeData node layout from cuda/street_solve.cuh. Uses
 * the storage-layout idiom of subgame_preflop.c (per-node offset tables into a
 * flat dense array) but NOT its math (that is vector CFR with reach weighting;
 * this is single-path external sampling).
 *
 * 1a SS_MAX_HANDS audit (2026-07-15): no additional live src/ hand/range/
 * weight array was hard-coded to 200, so A4 required no replacement beyond
 * the canonical SS_MAX_HANDS definition. Other live literal 200s are card-
 * abstraction/sample constants or stale complexity comments; the archived
 * FS_MAX_HANDS cap is outside this CPU-ES probe.
 */

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "es_resolver.h"
#include "es_resolver_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <time.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "hand_eval.h"   /* eval7 (static inline) */

#ifdef _OPENMP
#include <omp.h>
#endif

/* ES_T_* node type codes and ES_REGRET_FLOOR are defined in
 * es_resolver_internal.h (shared with es_resolver_reach.c). */

/* ── Thread-local last-error ─────────────────────────────────────────────── */
static __thread char g_es_error[512];

void es_set_error(const char *fmt, ...) {
    va_list ap;
    __builtin_va_start(ap, fmt);
    vsnprintf(g_es_error, sizeof(g_es_error), fmt, ap);
    __builtin_va_end(ap);
}
const char *es_get_last_error(void) {
    return g_es_error[0] ? g_es_error : NULL;
}

/* The sampled oracle implementation lives only in es_resolver_real, while this
 * resolver core is also compiled into the lightweight validation DLL. Keep the
 * public callback identity here and dispatch through the sampled ctx's first
 * member (`ESLeafOracle invoke`). A pointer to a struct, suitably converted,
 * points to its initial member, so this typed load is alias-safe. The sampled
 * TU pins the layout with a compile-time offsetof check. */
static double es_sampled_oracle_adapter(
    void *ctx, int leaf_idx, int traverser, int traverser_hand,
    const int *sampled_hands, const SSNode *leaf_node, uint64_t *rng)
{
    if (!ctx) {
        es_set_error("sampled oracle ctx is NULL");
        return 0.0;
    }
    ESLeafOracle invoke = *(ESLeafOracle *)ctx;
    if (!invoke) {
        es_set_error("sampled oracle dispatch is NULL");
        return 0.0;
    }
    return invoke(ctx, leaf_idx, traverser, traverser_hand, sampled_hands,
                  leaf_node, rng);
}

ES_EXPORT ESLeafOracle es_sampled_oracle_fn(void) {
    return es_sampled_oracle_adapter;
}

/* Monotonic milliseconds for deadline checks. QueryPerformanceCounter and
 * CLOCK_MONOTONIC are unaffected by wall-clock adjustments. */
uint64_t es_monotonic_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, now;
    if (!QueryPerformanceFrequency(&freq) || !QueryPerformanceCounter(&now)
        || freq.QuadPart <= 0)
        return UINT64_MAX;
    return (uint64_t)((double)now.QuadPart * 1000.0
                      / (double)freq.QuadPart);
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return UINT64_MAX;
    return (uint64_t)ts.tv_sec * 1000ULL
           + (uint64_t)ts.tv_nsec / 1000000ULL;
#endif
}

int es_deadline_expired(const ESResolver *r, uint64_t start_ms) {
    if (!r || r->deadline_ms <= 0) return 0;
    uint64_t now_ms = es_monotonic_ms();
    if (start_ms == UINT64_MAX || now_ms == UINT64_MAX) return 0;
    if (now_ms < start_ms) return 0;  /* backward jump: do not fire */
    return now_ms - start_ms >= (uint64_t)r->deadline_ms;
}

/* ── RNG: xorshift64, identical to mccfr_blueprint.c:62 ──────────────────── */
static inline uint64_t es_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}
static inline int es_rng_int(uint64_t *state, int n) {
    return (int)(es_rng_next(state) % (uint64_t)n);
}
static inline double es_rng_double(uint64_t *state) {
    /* 53-bit mantissa fraction in [0,1). */
    return (double)(es_rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

/* DIAGNOSTIC ONLY (2026-06-14): ES_NO_CONFLICT_SKIP=1 disables the showdown
 * runout-conflict `return 0.0` guard, to A/B whether that value-injection is the
 * deep-HU over-bet cause. Cached on first read; remove once the bug is settled. */
static int es_no_conflict_skip(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("ES_NO_CONFLICT_SKIP");
        v = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y' || e[0] == 'Y')) ? 1 : 0;
    }
    return v;
}

/* ── Chance-node MCCFR sampler (GPU-BIT-EXACT) ────────────────────────────────
 *
 * In-tree SS_NODE_CHANCE nodes (the OP3 §4 depth-rule HU flop/turn extension —
 * deal the next street explicitly) must be solved as MCCFR external sampling
 * over the card outcomes: at each CFR iteration, sample EXACTLY ONE child per
 * chance node, SHARED across the whole iteration's traversal, and propagate
 * regret/strat-sum updates only down that one sampled trajectory. No regret is
 * accumulated AT the chance node (it is not a decision); its value is just the
 * value of the sampled child (a single-outcome estimator of E_card[V(child)]).
 *
 * The GPU reference (street_solve.cu) draws this per-iter sample host-side via
 * ss_redraw_chance_samples_host: splitmix64(master_seed ^ iter·K1 ^ c·K2)
 * % num_children, where `c` is the chance node's chance_node_idx and `iter`
 * runs 1..max_iterations. We reproduce that recurrence and seed mix EXACTLY so
 * a single-thread CPU-ES walk over the SAME tree draws the IDENTICAL card per
 * (chance node, iter) — the prerequisite for the GPU bit-exact validation gate.
 *
 * This is computed ON DEMAND at the chance node (a pure function of
 * (master_seed, iter, chance_node_idx)) rather than from a shared per-iter
 * buffer. That is deliberate and load-bearing for two reasons:
 *   (1) Correctness within one CFR walk. The scalar traverser EXPLORES ALL of
 *       the traverser's actions, so a chance node deeper in the tree can be
 *       reached via several action paths inside a SINGLE es_traverse call. A
 *       per-call RNG (the old stub) would draw a DIFFERENT card on each of
 *       those visits, making child_vals[a] incomparable across actions and
 *       corrupting the regret — exactly the degenerate-pure-strategy bug.
 *       Keying the draw by (iter, chance_node_idx) makes every visit within an
 *       iteration return the same card. ✔
 *   (2) Hogwild-safety. Under multi-thread, different threads run different
 *       `iter` values concurrently; a single shared buffer could not hold a
 *       per-iter sample for each in-flight iteration. A stateless pure function
 *       of the thread's own `iter` has no shared mutable state and no race.
 *
 * The constants and recurrence are duplicated from street_solve.cu (the host
 * splitmix64 + the seed mix in ss_redraw_chance_samples_host). They MUST stay
 * byte-identical to the GPU — the bit-exact gate (tests) is the tripwire if
 * either side drifts.
 */
static inline uint64_t es_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/* The sampled child INDEX (0..num_children) for chance node `chance_node_idx`
 * at CFR iteration `iter` (1-based, matching the GPU loop). num_children must
 * be > 0. Bit-identical to ss_redraw_chance_samples_host(). */
static inline int es_chance_sampled_child(uint64_t master_seed, int chance_node_idx,
                                          int num_children, int64_t iter) {
    uint64_t s = es_splitmix64(
        master_seed ^ ((uint64_t)iter * 0x9E3779B97F4A7C15ULL)
                    ^ ((uint64_t)chance_node_idx * 0xBF58476D1CE4E5B9ULL));
    return (int)(s % (uint64_t)num_children);
}

/* DIAGNOSTIC (2026-06-14): ES_CHANCE_KEY_BY_STREET=1 keys the chance sample by
 * the STREET (num_board) instead of the tree-node-position chance_node_idx, so
 * ALL chance nodes dealing the same street draw the SAME physical card per CFR
 * iteration. This makes the runout COHERENT across betting lines — the turn p0
 * sees after CHECK is the same turn it sees after BET — which is required for
 * external-sampling regret to compare commensurate continuations. (The default
 * per-node key matches the GPU's draw, but the GPU is full-reach and integrates
 * over all outcomes, so its per-node independence is harmless there; external
 * sampling compares SINGLE samples, so it needs cross-line coherence.) The
 * candidate cards are identical across same-street nodes here (the builder
 * excludes only the board, and all lines share the subgame board), so keying by
 * street yields the same child index = the same card. */
static int es_chance_key_by_street(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("ES_CHANCE_KEY_BY_STREET");
        v = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y' || e[0] == 'Y')) ? 1 : 0;
    }
    return v;
}

/* Process default for chance handling: ENUMERATE unless ES_CHANCE_SAMPLE=1. */
static int es_chance_enumerate_default(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("ES_CHANCE_SAMPLE");
        int sample = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y' || e[0] == 'Y')) ? 1 : 0;
        v = sample ? 0 : 1;   /* enumerate unless explicitly asked to sample */
    }
    return v;
}

/* Opponent-action handling. DEFAULT = SAMPLE (the validated external-sampling
 * path — preserves ALL existing 2-6p behavior). Set ES_OPP_ENUMERATE=1 to
 * recurse ALL opponent actions sigma-weighted (full CFR). With chance ALSO
 * enumerated the traversal is then DETERMINISTIC, so a depth-limited 2p subgame
 * converges monotonically (kills the QJs single-sample oscillation). This is
 * OPT-IN, not the global default, because full opponent enumeration is only
 * tractable at SMALL player counts — at 5-6p the action-product across opponents
 * blows up, where external sampling is mandatory and validated. cpu_es_solver
 * turns it ON for the 2p chance-node path ONLY. */
static int es_enumerate_opponent_default(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("ES_OPP_ENUMERATE");
        v = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y' || e[0] == 'Y')) ? 1 : 0;
    }
    return v;
}
/* es_chance_enumerate(r) / es_enumerate_opponent(r) — per-resolver resolution
 * (explicit r->*_mode wins; -1 inherits the env default) — are defined just
 * after the ESResolver struct (they read its fields). */
static inline int es_chance_sampled_child_keyed(uint64_t master_seed,
                                                int chance_node_idx, int street_key,
                                                int num_children, int64_t iter) {
    int key = es_chance_key_by_street() ? street_key : chance_node_idx;
    return es_chance_sampled_child(master_seed, key, num_children, iter);
}

/* struct ESResolver is defined in es_resolver_internal.h (shared with
 * es_resolver_reach.c). */

/* Per-resolver mode resolution: explicit r->*_mode wins; -1 inherits the env
 * default. (Defined here — after the struct — because they read its fields.) */
static inline int es_chance_enumerate(const ESResolver *r) {
    return (r->chance_mode >= 0) ? r->chance_mode : es_chance_enumerate_default();
}
static inline int es_enumerate_opponent(const ESResolver *r) {
    return (r->opp_mode >= 0) ? r->opp_mode : es_enumerate_opponent_default();
}

/* Exact betting-closed leaf caches are keyed by NODE rather than leaf_idx:
 * board, bets, and the active set are node state, while leaf_idx only selects
 * an eager oracle table cell. Runouts initially exclude the node board only;
 * the fixed sampled hole cards are filtered on each exact-value call. */
static void es_clear_exact_allin_cache(ESResolver *r) {
    if (!r) return;
    if (r->exact_allin_leaf_cache) {
        for (int i = 0; i < r->num_nodes; i++) {
            free(r->exact_allin_leaf_cache[i].runout_masks);
            free(r->exact_allin_leaf_cache[i].strengths);
        }
    }
    free(r->exact_allin_leaf_cache);
    free(r->exact_allin_leaf_flags);
    r->exact_allin_leaf_cache = NULL;
    r->exact_allin_leaf_flags = NULL;
    r->exact_allin_ready = 0;
}

static int es_build_exact_allin_leaf_cache(ESResolver *r, int node_idx) {
    const SSTreeData *t = r->tree;
    const SSNode *nd = &t->nodes[node_idx];
    ESExactAllinLeafCache *cache = &r->exact_allin_leaf_cache[node_idx];
    int nb = nd->num_board;
    if (nb < 3 || nb > 5) {
        es_set_error("exact all-in leaf node %d has num_board=%d (expected 3..5)",
                     node_idx, nb);
        return -1;
    }

    uint64_t board_mask = 0;
    for (int b = 0; b < nb; b++) {
        int card = nd->board_cards[b];
        if (card < 0 || card >= 52 || (board_mask & (1ULL << card))) {
            es_set_error("exact all-in leaf node %d has invalid board card %d "
                         "at slot %d", node_idx, card, b);
            return -1;
        }
        board_mask |= 1ULL << card;
    }
    cache->board_mask = board_mask;

    int missing = 5 - nb;
    int available = 52 - nb;
    int nr = missing == 0 ? 1
             : (missing == 1 ? available
                              : available * (available - 1) / 2);
    cache->num_runouts = nr;
    cache->runout_masks = (uint64_t *)malloc((size_t)nr * sizeof(uint64_t));
    size_t runout_card_count = (size_t)nr * (size_t)missing;
    uint8_t *runout_cards = runout_card_count > 0
        ? (uint8_t *)malloc(runout_card_count * sizeof(uint8_t)) : NULL;
    if (!cache->runout_masks || (runout_card_count > 0 && !runout_cards)) {
        free(runout_cards);
        es_set_error("OOM exact all-in runouts at node %d (%d runouts)",
                     node_idx, nr);
        return -1;
    }

    int ri = 0;
    if (missing == 0) {
        cache->runout_masks[ri++] = 0;
    } else if (missing == 1) {
        for (int c = 0; c < 52; c++) {
            if (board_mask & (1ULL << c)) continue;
            cache->runout_masks[ri] = 1ULL << c;
            runout_cards[ri] = (uint8_t)c;
            ri++;
        }
    } else {
        for (int c0 = 0; c0 < 52; c0++) {
            if (board_mask & (1ULL << c0)) continue;
            for (int c1 = c0 + 1; c1 < 52; c1++) {
                if (board_mask & (1ULL << c1)) continue;
                cache->runout_masks[ri] = (1ULL << c0) | (1ULL << c1);
                runout_cards[(size_t)ri * 2] = (uint8_t)c0;
                runout_cards[(size_t)ri * 2 + 1] = (uint8_t)c1;
                ri++;
            }
        }
    }
    if (ri != nr) {
        free(runout_cards);
        es_set_error("exact all-in runout count mismatch at node %d (%d != %d)",
                     node_idx, ri, nr);
        return -1;
    }

    size_t total_hands = 0;
    for (int p = 0; p < r->np; p++) {
        int nh = t->num_hands[p];
        if (nh < 0 || nh > SS_MAX_HANDS) {
            free(runout_cards);
            es_set_error("exact all-in player %d has num_hands=%d", p, nh);
            return -1;
        }
        cache->hand_base[p] = total_hands;
        total_hands += (size_t)nh;
    }
    if (total_hands > SIZE_MAX / (size_t)nr
        || total_hands * (size_t)nr > SIZE_MAX / sizeof(uint32_t)) {
        free(runout_cards);
        es_set_error("exact all-in strength cache size overflow at node %d",
                     node_idx);
        return -1;
    }
    size_t strength_count = total_hands * (size_t)nr;
    cache->strengths = (uint32_t *)calloc(
        strength_count > 0 ? strength_count : 1, sizeof(uint32_t));
    if (!cache->strengths) {
        free(runout_cards);
        es_set_error("OOM exact all-in strengths at node %d (%zu entries)",
                     node_idx, strength_count);
        return -1;
    }

    for (int p = 0; p < r->np; p++) {
        for (int h = 0; h < t->num_hands[p]; h++) {
            int c0 = t->hands[p][h][0];
            int c1 = t->hands[p][h][1];
            if (c0 < 0 || c0 >= 52 || c1 < 0 || c1 >= 52 || c0 == c1) {
                free(runout_cards);
                es_set_error("exact all-in player %d hand %d is invalid (%d,%d)",
                             p, h, c0, c1);
                return -1;
            }
            uint64_t hand_mask = (1ULL << c0) | (1ULL << c1);
            if (hand_mask & board_mask) continue;
            uint32_t *row = cache->strengths
                + (cache->hand_base[p] + (size_t)h) * (size_t)nr;
            for (int rix = 0; rix < nr; rix++) {
                if (cache->runout_masks[rix] & hand_mask) continue;
                int board[5];
                for (int b = 0; b < nb; b++) board[b] = nd->board_cards[b];
                for (int b = 0; b < missing; b++)
                    board[nb + b] = runout_cards[(size_t)rix * missing + b];
                int cards[7] = {
                    board[0], board[1], board[2], board[3], board[4], c0, c1
                };
                uint32_t strength = eval7(cards);
                row[rix] = strength ? strength : 1;
            }
        }
    }
    free(runout_cards);
    return 0;
}

int es_prepare_exact_allin_leaves(ESResolver *r) {
    if (!r || !r->exact_allin_leaves || r->exact_allin_ready) return 0;
    es_clear_exact_allin_cache(r);

    r->exact_allin_leaf_flags = (uint8_t *)calloc(
        (size_t)r->num_nodes, sizeof(uint8_t));
    r->exact_allin_leaf_cache = (ESExactAllinLeafCache *)calloc(
        (size_t)r->num_nodes, sizeof(ESExactAllinLeafCache));
    if (!r->exact_allin_leaf_flags || !r->exact_allin_leaf_cache) {
        es_set_error("OOM exact all-in leaf metadata (%d nodes)", r->num_nodes);
        es_clear_exact_allin_cache(r);
        return -1;
    }

    int flagged = 0;
    for (int i = 0; i < r->num_nodes; i++) {
        const SSNode *nd = &r->tree->nodes[i];
        if (nd->type != ES_T_LEAF) continue;
        int active_with_chips = 0;
        for (int p = 0; p < r->np; p++) {
            int64_t remaining = (int64_t)r->tree->effective_stack
                                - (int64_t)nd->bets[p];
            if (nd->active_players[p] && remaining > 0)
                active_with_chips++;
        }
        if (active_with_chips < 2) {
            r->exact_allin_leaf_flags[i] = 1;
            flagged++;
        }
    }

    if (flagged > 0) {
        /* Normally initialized by hand_eval_table.c's loader constructor;
         * explicit initialization pins the cache-build contract as well. */
        poker_eval5_table_init();
        for (int i = 0; i < r->num_nodes; i++) {
            if (!r->exact_allin_leaf_flags[i]) continue;
            if (es_build_exact_allin_leaf_cache(r, i) != 0) {
                es_clear_exact_allin_cache(r);
                return -1;
            }
        }
    }
    r->exact_allin_ready = 1;
    return 0;
}

/* ── Setup ───────────────────────────────────────────────────────────────── */

static int es_build_offsets(ESResolver *r) {
    const SSTreeData *t = r->tree;
    r->num_nodes = t->num_nodes;
    r->np = t->num_players;

    r->node_off = (int64_t *)malloc((size_t)r->num_nodes * sizeof(int64_t));
    r->node_na = (int *)malloc((size_t)r->num_nodes * sizeof(int));
    if (!r->node_off || !r->node_na) {
        es_set_error("OOM offset tables (num_nodes=%d)", r->num_nodes);
        return -1;
    }
    int64_t off = 0;
    for (int i = 0; i < r->num_nodes; i++) {
        const SSNode *nd = &t->nodes[i];
        if (nd->type == ES_T_DECISION && nd->player >= 0 && nd->num_children > 0) {
            int acting = nd->player;
            int nh = t->num_hands[acting];
            int na = nd->num_children;
            r->node_off[i] = off;
            r->node_na[i] = na;
            off += (int64_t)nh * na;
        } else {
            r->node_off[i] = -1;
            r->node_na[i] = 0;
        }
    }
    r->total_elems = off;
    r->regrets = (double *)calloc((size_t)(off > 0 ? off : 1), sizeof(double));
    r->strat_sum = (double *)calloc((size_t)(off > 0 ? off : 1), sizeof(double));
    if (!r->regrets || !r->strat_sum) {
        es_set_error("OOM regret/strat tables (%lld elems)", (long long)off);
        return -1;
    }
    return 0;
}

/* Per-node carry of earlier-street contributions (see es_resolver_internal.h).
 * Single ascending pass: ss_tree_builder allocates every child after its parent
 * (the same invariant the reach path relies on), so carry(child) = carry(parent)
 * + parent's closing bets when the parent is a chance node. Fail loud if the
 * invariant does not hold. NULL (byte-identical path) without chance nodes. */
static int es_build_node_carry(ESResolver *r) {
    const SSTreeData *t = r->tree;
    int n = t->num_nodes;
    int np = t->num_players;
    r->node_carry = NULL;
    if (t->num_chance_nodes <= 0) return 0;
    int *carry = (int *)calloc((size_t)n * (size_t)np, sizeof(int));
    if (!carry) { es_set_error("OOM node carry (%d nodes)", n); return -1; }
    for (int i = 0; i < n; i++) {
        const SSNode *nd = &t->nodes[i];
        int is_chance = (nd->type == ES_T_CHANCE);
        for (int a = 0; a < nd->num_children; a++) {
            int child = t->children[nd->first_child + a];
            if (child <= i || child >= n) {
                free(carry);
                es_set_error("node carry: child %d is not after parent %d",
                             child, i);
                return -1;
            }
            for (int p = 0; p < np; p++)
                carry[(size_t)child * np + p] =
                    carry[(size_t)i * np + p] + (is_chance ? nd->bets[p] : 0);
        }
    }
    r->node_carry = carry;
    return 0;
}

ESResolver *es_create(const SSTreeData *tree, ESLeafOracle leaf_oracle,
                      void *oracle_ctx, uint64_t seed) {
    g_es_error[0] = '\0';
    if (!tree) { es_set_error("tree is NULL"); return NULL; }
    if (tree->num_players < 2 || tree->num_players > SS_MAX_PLAYERS) {
        es_set_error("num_players=%d out of [2,%d]", tree->num_players, SS_MAX_PLAYERS);
        return NULL;
    }
    if (tree->num_nodes <= 0) { es_set_error("num_nodes=%d", tree->num_nodes); return NULL; }

    ESResolver *r = (ESResolver *)calloc(1, sizeof(ESResolver));
    if (!r) { es_set_error("OOM resolver"); return NULL; }
    r->tree = tree;
    r->oracle = leaf_oracle;
    r->oracle_ctx = oracle_ctx;
    r->oracle_is_sampled = (leaf_oracle == es_sampled_oracle_adapter);
    r->master_seed = seed ? seed : 0x9E3779B97F4A7C15ULL;
    r->actual_traverser_player = -1;
    r->actual_traverser_hand = -1;
    r->actual_traverser_fraction = 0.0;
    r->chance_mode = -1;   /* inherit process/env default until set */
    r->opp_mode = -1;
    r->reach_mode = 0;     /* OFF until set (validated sampling path is default) */
    if (es_build_offsets(r) != 0) { es_free(r); return NULL; }
    if (es_build_node_carry(r) != 0) { es_free(r); return NULL; }
    return r;
}

ESResolver *es_create_with_root_pots(
    const SSTreeData *tree, ESLeafOracle leaf_oracle, void *oracle_ctx,
    uint64_t seed, const int *root_pot_amounts,
    const uint32_t *root_pot_eligible_masks, int num_root_pots)
{
    ESResolver *r = es_create(tree, leaf_oracle, oracle_ctx, seed);
    if (!r) return NULL;

    if (r->np < 2 || r->np > 3) {
        es_set_error("root-pot geometry supports 2-3 players, got %d", r->np);
        es_free(r);
        return NULL;
    }
    if (!tree->is_river || tree->num_leaf_nodes > 0
        || tree->num_chance_nodes > 0) {
        es_set_error("root-pot geometry currently supports native river trees "
                     "only (is_river=%d leaves=%d chance=%d)",
                     tree->is_river, tree->num_leaf_nodes,
                     tree->num_chance_nodes);
        es_free(r);
        return NULL;
    }
    if (!root_pot_amounts || !root_pot_eligible_masks
        || num_root_pots < 1 || num_root_pots > ES_MAX_ROOT_POT_LAYERS) {
        es_set_error("invalid root-pot arrays/count (%d)", num_root_pots);
        es_free(r);
        return NULL;
    }

    const uint32_t allowed = (1u << r->np) - 1u;
    int64_t total = 0;
    for (int i = 0; i < num_root_pots; i++) {
        int amount = root_pot_amounts[i];
        uint32_t mask = root_pot_eligible_masks[i];
        if (amount < 0 || mask == 0 || (mask & ~allowed) != 0) {
            es_set_error(
                "invalid root-pot layer %d (amount=%d mask=0x%x allowed=0x%x)",
                i, amount, (unsigned)mask, (unsigned)allowed);
            es_free(r);
            return NULL;
        }
        total += (int64_t)amount;
        if (total > INT_MAX) {
            es_set_error("root-pot layer sum exceeds INT_MAX");
            es_free(r);
            return NULL;
        }
        r->root_pot_amounts[i] = amount;
        r->root_pot_eligible_masks[i] = mask;
    }
    if (root_pot_eligible_masks[0] != allowed) {
        es_set_error("root-pot layer 0 must be the all-player main pot "
                     "(mask=0x%x, expected=0x%x)",
                     (unsigned)root_pot_eligible_masks[0],
                     (unsigned)allowed);
        es_free(r);
        return NULL;
    }
    for (int i = 1; i < num_root_pots; i++) {
        if (root_pot_eligible_masks[i] == root_pot_eligible_masks[i - 1]
            || (root_pot_eligible_masks[i]
                & ~root_pot_eligible_masks[i - 1]) != 0) {
            es_set_error("root-pot masks must form a proper nested main-to-side "
                         "chain (layer %d mask=0x%x previous=0x%x)",
                         i, (unsigned)root_pot_eligible_masks[i],
                         (unsigned)root_pot_eligible_masks[i - 1]);
            es_free(r);
            return NULL;
        }
    }
    if (total != (int64_t)tree->starting_pot) {
        es_set_error("root-pot layers sum to %lld, starting_pot=%d",
                     (long long)total, tree->starting_pot);
        es_free(r);
        return NULL;
    }

    /* Validate the masks against every terminal in the static tree.  A valid
     * poker side-pot chain always retains at least one eligible contestant per
     * showdown layer; at a fold terminal the sole winner must be eligible for
     * every layer.  Reject inconsistent metadata instead of silently dropping
     * or mis-awarding chips during a 500k-iteration solve. */
    for (int n = 0; n < tree->num_nodes; n++) {
        const SSNode *nd = &tree->nodes[n];
        if (nd->type == ES_T_LEAF || nd->type == ES_T_CHANCE) {
            es_set_error("root-pot geometry found unsupported node type %d "
                         "at node %d", nd->type, n);
            es_free(r);
            return NULL;
        } else if (nd->type == ES_T_SHOWDOWN) {
            uint32_t live = 0;
            for (int p = 0; p < r->np; p++)
                if (nd->active_players[p]) live |= 1u << p;
            for (int i = 0; i < num_root_pots; i++) {
                if ((live & root_pot_eligible_masks[i]) == 0) {
                    es_set_error("root-pot layer %d has no eligible player at "
                                 "showdown node %d", i, n);
                    es_free(r);
                    return NULL;
                }
            }
        } else if (nd->type == ES_T_FOLD) {
            int winner = nd->player;
            if (winner < 0 || winner >= r->np) {
                es_set_error("invalid fold winner %d at node %d", winner, n);
                es_free(r);
                return NULL;
            }
            for (int i = 0; i < num_root_pots; i++) {
                if (!(root_pot_eligible_masks[i] & (1u << winner))) {
                    es_set_error("fold winner %d is ineligible for root-pot "
                                 "layer %d at node %d", winner, i, n);
                    es_free(r);
                    return NULL;
                }
            }
        }
    }
    r->root_pot_count = num_root_pots;
    r->root_pot_total = (int)total;
    return r;
}

void es_set_cfr_mode(ESResolver *r, int mode) {
    if (r) r->cfr_mode = (mode == 1) ? 1 : 0;
}

int es_set_traverser_actual_hand(ESResolver *r, int player, int hand,
                                  double fraction) {
    if (!r) {
        es_set_error("resolver is NULL");
        return -1;
    }
    if (player < 0 || player >= r->np
        || hand < 0 || hand >= r->tree->num_hands[player]
        || !isfinite(fraction) || fraction < 0.0 || fraction > 1.0) {
        es_set_error(
            "invalid actual traverser hand (player=%d hand=%d fraction=%g)",
            player, hand, fraction);
        return -1;
    }
    r->actual_traverser_player = player;
    r->actual_traverser_hand = hand;
    r->actual_traverser_fraction = fraction;
    return 0;
}

void es_set_vec_mode(ESResolver *r, int mode) {
    if (r) r->vec_traverser = (mode == 1) ? 1 : 0;
}

void es_set_exact_allin_leaves(ESResolver *r, int enabled) {
    if (r) r->exact_allin_leaves = enabled ? 1 : 0;
}

/* Chance-node handling: 1 = ENUMERATE (correct, zero-variance — average over all
 * dealt cards), 0 = SAMPLE (single card/iter — high variance), -1 = inherit the
 * env default. Set BEFORE es_run. */
void es_set_chance_mode(ESResolver *r, int mode) {
    if (r) r->chance_mode = (mode == 0 || mode == 1) ? mode : -1;
}

/* Opponent-action handling: 1 = ENUMERATE (full CFR — recurse all opp actions
 * sigma-weighted; deterministic-modulo-hand-sampling, for SMALL player counts
 * only — 2p/3p), 0 = SAMPLE (external sampling, mandatory at 5-6p), -1 = inherit
 * the env default. Set BEFORE es_run. */
void es_set_opp_mode(ESResolver *r, int mode) {
    if (r) r->opp_mode = (mode == 0 || mode == 1) ? mode : -1;
}

/* Full-per-hand-reach vector CFR: 1 = ON (the interactive 2p chance solve — see
 * the reach_mode field comment), 0 = OFF (default; sampling traversers). When ON,
 * es_run takes the es_run_reach path (Linear CFR + chance-enumerate + full
 * opponent reach, deterministic — no RNG). Set BEFORE es_run. */
void es_set_reach_mode(ESResolver *r, int mode) {
    if (r) r->reach_mode = (mode == 1) ? 1 : 0;
}

void es_set_deadline_ms(ESResolver *r, int64_t ms) {
    if (r) r->deadline_ms = (ms > 0) ? ms : 0;
}

int64_t es_iterations_run(const ESResolver *r) {
    return r ? r->iterations_run : 0;
}

/* DEBUG visit counter (single-thread localization only). */
void es_enable_visit_counter(ESResolver *r) {
    if (!r) return;
    free(r->visit_count);
    r->visit_count = (int64_t *)calloc((size_t)r->num_nodes, sizeof(int64_t));
}
int64_t es_get_visit_count(const ESResolver *r, int node_idx) {
    if (!r || !r->visit_count || node_idx < 0 || node_idx >= r->num_nodes) return -1;
    return r->visit_count[node_idx];
}

void es_free(ESResolver *r) {
    if (!r) return;
    if (r) { free(r->node_carry); r->node_carry = NULL; }
    free(r->node_off);
    free(r->node_na);
    free(r->regrets);
    free(r->strat_sum);
    free(r->visit_count);
    free(r->rc_reach);
    free(r->rc_cfv);
    free(r->rc_strat);
    free(r->rc_sd_strength);
    free(r->rc_sd_done);
    es_clear_exact_allin_cache(r);
    free(r);
}

/* ── Regret matching for one (node, hand) row ────────────────────────────────
 * sigma(a) ∝ max(0, R(a)); uniform if all <= 0. Mirrors
 * es_solver.py InfoSets.strategy. Operates on float64 to match the reference.
 */
void es_regret_match(const double *r, int na, double *out) {
    double s = 0.0;
    for (int a = 0; a < na; a++) {
        double v = r[a];
        if (v > 0.0) { out[a] = v; s += v; } else { out[a] = 0.0; }
    }
    if (s > 0.0) {
        double inv = 1.0 / s;
        for (int a = 0; a < na; a++) out[a] *= inv;
    } else {
        double u = 1.0 / na;
        for (int a = 0; a < na; a++) out[a] = u;
    }
}

/* ── Hand sampling ───────────────────────────────────────────────────────────
 * The internal form optionally reserves one fixed hand before sampling every
 * other player. fixed_player=-1 is the existing modeled joint sampler exactly.
 * chosen[p] = -1 if a conflict-free hand cannot be assigned.
 */
static void es_sample_hands_with_fixed(ESResolver *r, uint64_t *rng,
                                       int *chosen, int fixed_player,
                                       int fixed_hand) {
    const SSTreeData *t = r->tree;
    int np = r->np;

    /* used-cards bitmask over 52 cards. */
    uint64_t used = 0;
    for (int b = 0; b < t->num_board; b++) used |= (1ULL << t->board[b]);

    for (int p = 0; p < np; p++) chosen[p] = -1;

    if (fixed_player >= 0) {
        int c0 = t->hands[fixed_player][fixed_hand][0];
        int c1 = t->hands[fixed_player][fixed_hand][1];
        if (((used >> c0) & 1ULL) || ((used >> c1) & 1ULL)) return;
        chosen[fixed_player] = fixed_hand;
        used |= (1ULL << c0);
        used |= (1ULL << c1);
    }

    /* Randomized assignment order (Fisher-Yates). The fixed player's cards
     * are already reserved, so opponents are drawn from the correctly
     * card-conditioned modeled ranges without any hot-loop allocation. */
    int order[SS_MAX_PLAYERS];
    int n_order = 0;
    for (int p = 0; p < np; p++) {
        if (p != fixed_player) order[n_order++] = p;
    }
    for (int i = 0; i < n_order; i++) {
        int j = i + es_rng_int(rng, n_order - i);
        int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    for (int oi = 0; oi < n_order; oi++) {
        int p = order[oi];
        int nh = t->num_hands[p];
        /* reservoir over conflict-free, weighted hands: one weighted pick in a
         * single pass (avoids allocating candidate lists). Equivalent to
         * numpy choice over the conflict-free weighted set. */
        double total_w = 0.0;
        int pick = -1;
        for (int h = 0; h < nh; h++) {
            float w = t->weights[p][h];
            if (w <= 0.0f) continue;
            int c0 = t->hands[p][h][0];
            int c1 = t->hands[p][h][1];
            if ((used >> c0) & 1ULL) continue;
            if ((used >> c1) & 1ULL) continue;
            total_w += (double)w;
            /* weighted reservoir: accept h with prob w/total_w */
            if (es_rng_double(rng) * total_w < (double)w) pick = h;
        }
        if (pick >= 0) {
            chosen[p] = pick;
            used |= (1ULL << t->hands[p][pick][0]);
            used |= (1ULL << t->hands[p][pick][1]);
        }
    }
}

/* One modeled hand per player, matching the historical sampler and RNG use. */
static void es_sample_hands(ESResolver *r, uint64_t *rng, int *chosen) {
    es_sample_hands_with_fixed(r, rng, chosen, -1, -1);
}

static inline void es_sample_hands_for_traverser(
    ESResolver *r, uint64_t *rng, int traverser, int *chosen
) {
    if (traverser != r->actual_traverser_player
        || r->actual_traverser_fraction <= 0.0) {
        /* In particular, every villain-as-traverser pass stays on the modeled
         * joint range exactly as before. */
        es_sample_hands(r, rng, chosen);
        return;
    }

    int use_actual = (r->actual_traverser_fraction >= 1.0)
        || (es_rng_double(rng) < r->actual_traverser_fraction);
    if (use_actual) {
        /* External-sampling MCCFR's traverser-hand proposal changes only the
         * convergence RATE of each separate infoset row, never its regret
         * fixed point. Thus oversampling the request's actual row is unbiased;
         * no range epsilon floor or importance correction belongs here. */
        es_sample_hands_with_fixed(r, rng, chosen,
                                   r->actual_traverser_player,
                                   r->actual_traverser_hand);
    } else {
        es_sample_hands(r, rng, chosen);
    }
}

/* ── Opponent-only hand sampling (VEC mode) ───────────────────────────────────
 * Identical reservoir logic to es_sample_hands but SKIPS the `skip` (traverser)
 * slot — so the traverser's cards are NEVER reserved out of the opponents' pool.
 *
 * WHY this matters (the 3p vec-fixed-point bug, 2026-06-10): the scalar path
 * samples ALL players including the traverser, removing the traverser's 2 cards
 * from the deck before later opponents draw. In VEC mode the traverser hand is
 * NOT fixed (we enumerate all of them), so reserving a spurious traverser hand
 * biases the opponent distribution each enumerated hand sees — opponents can no
 * longer hold the (spuriously-removed) traverser cards, missing those worlds.
 * The dead-mask is the correct per-hand card-removal conditioner ONLY IF the
 * base opponent draw is board+inter-opponent (NOT traverser-conditioned). This
 * compounds with opponent count (≈0 at 2p one-opponent, large at 3p+) — the
 * exact 2p-pass / 3p-fail signature. chosen[skip] is left = -1. */
/* Diagnostic-only replay of scalar ES joint sampling.  Draw normally first,
 * then retain the requested hero hand so randomized assignment order remains
 * exactly production-identical. */
ES_EXPORT int32_t es_debug_sample_hands_conditioned_c(
    const SSTreeData *tree,
    uint64_t stream_seed,
    int32_t conditioned_player,
    int32_t conditioned_hand,
    int32_t num_samples,
    int32_t max_attempts,
    int32_t *out_hands,
    uint64_t *out_rng_states,
    int32_t *out_attempts,
    int32_t *out_incomplete
) {
    if (out_attempts) *out_attempts = 0;
    if (out_incomplete) *out_incomplete = 0;
    if (!tree || !out_hands || num_samples <= 0
        || max_attempts < num_samples
        || tree->num_players < 2 || tree->num_players > SS_MAX_PLAYERS
        || conditioned_player < 0
        || conditioned_player >= tree->num_players
        || conditioned_hand < 0
        || conditioned_hand >= tree->num_hands[conditioned_player])
        return -1;
    for (int p = 0; p < tree->num_players; p++) {
        if (tree->num_hands[p] < 0 || tree->num_hands[p] > SS_MAX_HANDS)
            return -1;
    }
    if (tree->num_board < 0 || tree->num_board > SS_MAX_BOARD) return -1;
    for (int b = 0; b < tree->num_board; b++) {
        if (tree->board[b] < 0 || tree->board[b] >= 52) return -1;
    }

    ESResolver view;
    memset(&view, 0, sizeof(view));
    view.tree = tree;
    view.np = tree->num_players;

    uint64_t rng = stream_seed ? stream_seed : 0xD1B54A32D192ED03ULL;
    int32_t kept = 0;
    int32_t attempts = 0;
    int32_t incomplete = 0;
    int chosen[SS_MAX_PLAYERS];
    while (kept < num_samples && attempts < max_attempts) {
        es_sample_hands(&view, &rng, chosen);
        attempts++;
        if (chosen[conditioned_player] != conditioned_hand) continue;

        int missing = 0;
        for (int p = 0; p < view.np; p++) {
            out_hands[(size_t)kept * (size_t)view.np + (size_t)p] = chosen[p];
            if (chosen[p] < 0) missing = 1;
        }
        if (out_rng_states) out_rng_states[kept] = rng;
        incomplete += missing;
        kept++;
    }
    if (out_attempts) *out_attempts = attempts;
    if (out_incomplete) *out_incomplete = incomplete;
    return kept;
}

/* The opponent-only VEC implementation documented above begins here. */
static void es_sample_opponents(ESResolver *r, uint64_t *rng, int skip,
                                int *chosen) {
    const SSTreeData *t = r->tree;
    int np = r->np;

    uint64_t used = 0;
    for (int b = 0; b < t->num_board; b++) used |= (1ULL << t->board[b]);
    for (int p = 0; p < np; p++) chosen[p] = -1;

    /* randomized order over OPPONENTS only (skip excluded) */
    int order[SS_MAX_PLAYERS]; int no = 0;
    for (int p = 0; p < np; p++) if (p != skip) order[no++] = p;
    for (int i = 0; i < no; i++) {
        int j = i + es_rng_int(rng, no - i);
        int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    for (int oi = 0; oi < no; oi++) {
        int p = order[oi];
        int nh = t->num_hands[p];
        double total_w = 0.0;
        int pick = -1;
        for (int h = 0; h < nh; h++) {
            float w = t->weights[p][h];
            if (w <= 0.0f) continue;
            int c0 = t->hands[p][h][0];
            int c1 = t->hands[p][h][1];
            if ((used >> c0) & 1ULL) continue;
            if ((used >> c1) & 1ULL) continue;
            total_w += (double)w;
            if (es_rng_double(rng) * total_w < (double)w) pick = h;
        }
        if (pick >= 0) {
            chosen[p] = pick;
            used |= (1ULL << t->hands[p][pick][0]);
            used |= (1ULL << t->hands[p][pick][1]);
        }
    }
}

/* ── Terminal values (match es_solver.py) ────────────────────────────────── */

/* FOLD terminal: the sole remaining active player (node.player) wins the pot.
 * net = (pot - my_bet) if I'm the winner else -my_bet. */
static inline double es_fold_value(const ESResolver *r, int node_idx,
                                   const SSNode *nd, int traverser) {
    int winner = nd->player;          /* street_solve.cu sets winner = node->player */
    double my_bet = es_my_bet(r, node_idx, nd, traverser);
    if (traverser == winner) return (double)nd->pot - my_bet;
    return -my_bet;
}

/* ── SHOWDOWN terminal (river): multiway side-pot layering ────────────────────
 *
 * LEGACY-SCALAR SCOPE NOTE (2026-06-10, Lane 3; qualified 2026-09-01).
 * SSNode carries only
 * per-THIS-STREET bets (SSNode.bets[]) plus a lump `pot` that folds in all
 * prior-street money. Consequences:
 *
 *   1. EXACTLY representable by legacy es_create: a common lump entry pot plus
 *      side pots layered only by THIS-street contribution caps. The legacy
 *      model zeroes bets[] at each street entry and has no entry-pot
 *      eligibility metadata.
 *   2. NOT representable through legacy es_create: side pots from prior-street
 *      unequal all-ins. The additive es_create_with_root_pots constructor now
 *      supplies those eligibility layers in an opaque resolver sidecar.
 *   3. The legacy scalar wrapper gives every player the same cap, its legacy
 *      completion rule requires equal active bets, and its calls equalize to
 *      the active maximum. On those reachable nodes es_showdown_is_single_pot
 *      retains the historical award BIT-EXACTLY. The additive
 *      ss_build_tree_stacks path can emit short all-in calls; its root-pot
 *      constructor routes them through the stack-aware award below.
 *   4. Defensive accounting: the layered award distributes EXACTLY nd->pot.
 *      Any accounting slack (or folded contribution above the top active
 *      level) lands in the base layer, preserving the system invariant
 *      Σ_p award_p == nd->pot.
 *
 * CONVENTIONS (shared with the Python exact reference validate_sidepot.py):
 *   - Levels = distinct ACTIVE street-bet values, ascending.
 *   - Contribution per player (active or folded) = min(bets[p], top level);
 *     folded money below a level boundary is dead money in that layer.
 *   - Base layer = nd->pot − Σ contributions (entry pot + folder excess above
 *     the top active level + builder accounting slack), awarded with the
 *     bottom layer (same eligibility: every showdown-active player).
 *   - Layer (prev, L] eligible = active players with bets[p] >= L; best
 *     eligible hand(s) split the layer by double division (ties chop, the
 *     existing convention).
 */

/* TRUE iff the single-pot chop is exact at this node: all ACTIVE street bets
 * equal and no FOLDED street bet above that level. Every showdown the current
 * builder emits satisfies this (see scope note) — those nodes stay BIT-EXACT
 * to the historical single-pot behavior. */
static int es_showdown_is_single_pot(const SSNode *nd, int np) {
    int amin = -1, amax = -1, fmax = 0;
    for (int p = 0; p < np; p++) {
        int b = nd->bets[p];
        if (nd->active_players[p]) {
            if (amin < 0 || b < amin) amin = b;
            if (b > amax) amax = b;
        } else {
            if (b > fmax) fmax = b;
        }
    }
    if (amax < 0) return 1;           /* no actives — caller handled upstream */
    return (amin == amax) && (fmax <= amax);
}

/* GROSS layered award to `target` (before subtracting target's own street
 * bet). strength[p] must be valid for every ACTIVE p; `target` must be
 * active. Total award across players == nd->pot exactly (in exact arith). */
static double es_layered_award(const SSNode *nd, int np,
                               const uint32_t *strength, int target) {
    /* Distinct ACTIVE bet levels, ascending (insertion sort; np <= 6). */
    int levels[SS_MAX_PLAYERS]; int nlev = 0;
    for (int p = 0; p < np; p++) {
        if (!nd->active_players[p]) continue;
        int b = nd->bets[p];
        int j = 0;
        while (j < nlev && levels[j] < b) j++;
        if (j < nlev && levels[j] == b) continue;
        for (int k = nlev; k > j; k--) levels[k] = levels[k - 1];
        levels[j] = b; nlev++;
    }
    if (nlev == 0) return 0.0;        /* no active players */
    int top = levels[nlev - 1];

    /* Contributions capped at the top active level; the remainder of the pot
     * (entry pot + folder excess + accounting slack) is the base layer. */
    int contrib[SS_MAX_PLAYERS];
    double base = (double)nd->pot;
    for (int p = 0; p < np; p++) {
        int c = nd->bets[p];
        if (c > top) c = top;
        if (c < 0) c = 0;
        contrib[p] = c;
        base -= (double)c;
    }

    double award = 0.0;
    int prev = 0;
    for (int j = 0; j < nlev; j++) {
        int L = levels[j];
        double layer = (j == 0) ? base : 0.0;
        for (int p = 0; p < np; p++) {
            int c = contrib[p];
            int lo = (c < prev) ? c : prev;
            int hi = (c < L) ? c : L;
            if (hi > lo) layer += (double)(hi - lo);
        }
        /* Best eligible hand(s) split this layer. */
        uint32_t best = 0; int nwin = 0; int target_in = 0;
        for (int p = 0; p < np; p++) {
            if (!nd->active_players[p] || nd->bets[p] < L) continue;
            uint32_t s = strength[p];
            if (nwin == 0 || s > best) {
                best = s; nwin = 1; target_in = (p == target);
            } else if (s == best) {
                nwin++;
                if (p == target) target_in = 1;
            }
        }
        if (nwin > 0 && target_in) award += layer / (double)nwin;
        prev = L;
    }
    return award;
}

/* Award one independently eligible layer.  `eligible_mask` is rooted at the
 * subgame's compact player indices; players who folded after the root are
 * removed by intersecting with nd->active_players. */
static double es_masked_layer_award(
    const SSNode *nd, int np, const uint32_t *strength, int target,
    double layer, uint32_t eligible_mask)
{
    uint32_t best = 0;
    int nwin = 0;
    int target_in = 0;
    for (int p = 0; p < np; p++) {
        if (!(eligible_mask & (1u << p)) || !nd->active_players[p]) continue;
        uint32_t s = strength[p];
        if (nwin == 0 || s > best) {
            best = s;
            nwin = 1;
            target_in = (p == target);
        } else if (s == best) {
            nwin++;
            if (p == target) target_in = 1;
        }
    }
    return (nwin > 0 && target_in) ? layer / (double)nwin : 0.0;
}

/* GROSS award for the additive unequal-stack ABI.  Root-pot layers settle
 * first using their explicit eligibility masks.  Contributions made after the
 * subgame root (SSNode.bets[]) are then stripped into ordinary side-pot levels.
 * The target's current-subgame contribution is subtracted by the caller.
 *
 * Contributions are capped at the highest ACTIVE level, exactly like the
 * legacy layered evaluator.  Folded excess and any defensive accounting slack
 * therefore become common dead money attached to root layer 0, whose
 * constructor contract requires the all-player main-pot mask. */
static double es_root_pot_award(const ESResolver *r, const SSNode *nd,
                                const uint32_t *strength, int target) {
    const int np = r->np;
    /* Distinct positive ACTIVE contribution levels, ascending.  Folded chips
     * are included in each layer's amount but never in its eligible mask. */
    int levels[3];
    int nlev = 0;
    for (int p = 0; p < np; p++) {
        if (!nd->active_players[p] || nd->bets[p] <= 0) continue;
        int b = nd->bets[p];
        int j = 0;
        while (j < nlev && levels[j] < b) j++;
        if (j < nlev && levels[j] == b) continue;
        for (int k = nlev; k > j; k--) levels[k] = levels[k - 1];
        levels[j] = b;
        nlev++;
    }

    int top_active = nlev > 0 ? levels[nlev - 1] : 0;
    int64_t capped_future_total = 0;
    for (int p = 0; p < np; p++) {
        int contribution = nd->bets[p];
        if (contribution < 0) contribution = 0;
        if (contribution > top_active) contribution = top_active;
        capped_future_total += (int64_t)contribution;
    }
    double slack = (double)((int64_t)nd->pot
                            - (int64_t)r->root_pot_total
                            - capped_future_total);

    double award = 0.0;
    for (int i = 0; i < r->root_pot_count; i++) {
        double layer = (double)r->root_pot_amounts[i];
        if (i == 0) layer += slack;
        award += es_masked_layer_award(
            nd, np, strength, target, layer,
            r->root_pot_eligible_masks[i]);
    }

    int prev = 0;
    for (int j = 0; j < nlev; j++) {
        int level = levels[j];
        double layer = 0.0;
        uint32_t eligible = 0;
        for (int p = 0; p < np; p++) {
            int contribution = nd->bets[p] > 0 ? nd->bets[p] : 0;
            int hi = contribution < level ? contribution : level;
            if (hi > prev) layer += (double)(hi - prev);
            if (nd->active_players[p] && contribution >= level)
                eligible |= 1u << p;
        }
        award += es_masked_layer_award(
            nd, np, strength, target, layer, eligible);
        prev = level;
    }
    return award;
}

/* Exact expected showdown for one sampled active-hand profile at a flagged
 * depth-limit leaf. The strength cache removes eval7 from this hot path; each
 * call only filters runouts, compares cached ranks, and awards the pot. */
static double es_exact_allin_leaf_value(ESResolver *r, int node_idx,
                                        int traverser, const int *sampled) {
    const SSTreeData *t = r->tree;
    const SSNode *nd = &t->nodes[node_idx];
    const ESExactAllinLeafCache *cache =
        &r->exact_allin_leaf_cache[node_idx];
    double my_bet = es_my_bet(r, node_idx, nd, traverser);
    if (!nd->active_players[traverser]) return -my_bet;
    if (cache->num_runouts <= 0 || !cache->runout_masks || !cache->strengths)
        return 0.0;

    int selected[SS_MAX_PLAYERS];
    uint64_t used = cache->board_mask;
    for (int p = 0; p < r->np; p++) {
        selected[p] = -1;
        if (!nd->active_players[p]) continue;
        int h = sampled[p];
        if (h < 0 || h >= t->num_hands[p]) return 0.0;
        int c0 = t->hands[p][h][0];
        int c1 = t->hands[p][h][1];
        if (c0 < 0 || c0 >= 52 || c1 < 0 || c1 >= 52 || c0 == c1)
            return 0.0;
        uint64_t hand_mask = (1ULL << c0) | (1ULL << c1);
        /* This catches board collisions, traverser/opponent collisions, and
         * inter-opponent collisions. The whole sampled world is then neutral. */
        if (hand_mask & used) return 0.0;
        used |= hand_mask;
        selected[p] = h;
    }

    int single_pot = r->root_pot_count == 0
                     && es_showdown_is_single_pot(nd, r->np);
    double payoff_sum = 0.0;
    int valid_runouts = 0;
    for (int ri = 0; ri < cache->num_runouts; ri++) {
        if (cache->runout_masks[ri] & used) continue;
        uint32_t strength[SS_MAX_PLAYERS];
        uint32_t best = 0;
        int valid = 1;
        for (int p = 0; p < r->np; p++) {
            if (!nd->active_players[p]) continue;
            const uint32_t *row = cache->strengths
                + (cache->hand_base[p] + (size_t)selected[p])
                  * (size_t)cache->num_runouts;
            uint32_t s = row[ri];
            if (s == 0) { valid = 0; break; }
            strength[p] = s;
            if (s > best) best = s;
        }
        if (!valid) continue;

        double award;
        if (single_pot) {
            int nwin = 0;
            int traverser_wins = 0;
            for (int p = 0; p < r->np; p++) {
                if (!nd->active_players[p] || strength[p] != best) continue;
                nwin++;
                if (p == traverser) traverser_wins = 1;
            }
            award = traverser_wins ? (double)nd->pot / (double)nwin : 0.0;
        } else if (r->root_pot_count > 0) {
            award = es_root_pot_award(r, nd, strength, traverser);
        } else {
            award = es_layered_award(nd, r->np, strength, traverser);
        }
        payoff_sum += award - my_bet;
        valid_runouts++;
    }
    return valid_runouts > 0 ? payoff_sum / (double)valid_runouts : 0.0;
}

double es_leaf_payoff(ESResolver *r, int node_idx, int traverser,
                      const int *sampled, uint64_t *rng) {
    const SSNode *nd = &r->tree->nodes[node_idx];
    if (r->exact_allin_leaves && r->exact_allin_ready
        && r->exact_allin_leaf_flags
        && r->exact_allin_leaf_flags[node_idx]) {
        return es_exact_allin_leaf_value(r, node_idx, traverser, sampled);
    }
    /* The oracle's convention is payoff minus THIS street's bets; charge the
     * earlier streets' contribution here (zero on single-street trees). */
    return r->oracle(r->oracle_ctx, nd->leaf_idx, traverser,
                     sampled[traverser], sampled, nd, rng)
           - es_carry(r, node_idx, traverser);
}

/* SHOWDOWN terminal (river). Single-pot fast path (bit-exact historical
 * behavior, exact for every node the current builder can emit) + layered
 * side-pot path for uneven this-street all-in contributions (scope above). */
static double es_showdown_value(ESResolver *r, int node_idx, const SSNode *nd,
                                int traverser, const int *sampled) {
    const SSTreeData *t = r->tree;
    int np = r->np;
    double my_bet = es_my_bet(r, node_idx, nd, traverser);
    if (!nd->active_players[traverser]) return -my_bet;

    /* board must be 5 cards at a river showdown. Use the NODE's board, not the
     * subgame-root tree board: under the OP3 HU flop/turn extension the showdown
     * is reached THROUGH in-tree chance nodes that dealt the turn+river, so the
     * runout lives in nd->board_cards (the builder fills it as cards are dealt —
     * street_solve.cuh: "the child's SSNode.board_cards already reflect the
     * dealt card"). t->board has only the subgame-root cards (3 at a flop root),
     * so t->board[3]/[4] would be garbage. For a RIVER subgame root nd->board ==
     * t->board (5 cards) so this is identical to the old behavior there. */
    int board[5];
    if (nd->num_board == 5) {
        for (int b = 0; b < 5; b++) board[b] = nd->board_cards[b];
    } else {
        /* Defensive: a showdown with <5 board cards should not occur; fall back
         * to the tree board (historical behavior) rather than read garbage. */
        for (int b = 0; b < 5; b++) board[b] = t->board[b];
    }

    uint32_t strength[SS_MAX_PLAYERS];
    int live[SS_MAX_PLAYERS]; int nlive = 0;
    uint32_t best = 0;
    for (int p = 0; p < np; p++) {
        if (!nd->active_players[p]) continue;
        int hi = sampled[p];
        if (hi < 0) return 0.0;       /* unassignable — skip (caller upstream) */
        int c0 = t->hands[p][hi][0];
        int c1 = t->hands[p][hi][1];
        /* A sampled hand that collides with the (5-card runout) board is an
         * impossible world — under the OP3 chance extension the turn/river were
         * dealt after this hand was sampled. Mirror the GPU `blocked` exclusion;
         * with single-hand sampling the whole showdown is undefined, so return
         * the neutral 0 (this world contributes nothing this iter). */
        if (!es_no_conflict_skip())
            for (int b = 0; b < 5; b++)
                if (c0 == board[b] || c1 == board[b]) return 0.0;
        int cards[7] = { board[0], board[1], board[2], board[3], board[4], c0, c1 };
        uint32_t s = eval7(cards);
        strength[p] = s;
        live[nlive++] = p;
        if (s > best) best = s;
    }
    if (r->root_pot_count == 0 && es_showdown_is_single_pot(nd, np)) {
        int nwin = 0;
        int trav_wins = 0;
        for (int k = 0; k < nlive; k++) {
            int p = live[k];
            if (strength[p] == best) {
                nwin++;
                if (p == traverser) trav_wins = 1;
            }
        }
        if (trav_wins) return (double)nd->pot / (double)nwin - my_bet;
        return -my_bet;
    }
    if (r->root_pot_count > 0)
        return es_root_pot_award(r, nd, strength, traverser) - my_bet;
    return es_layered_award(nd, np, strength, traverser) - my_bet;
}

/* LEAF: lazy betting-leaf value via the caller oracle (one sampled-opponent
 * rollout in the real path; a table lookup in validation). */
static inline double es_leaf_value(ESResolver *r, int node_idx,
                                   int traverser, const int *sampled,
                                   ESStats *local, uint64_t *rng) {
    local->leaf_rollouts++;
    return es_leaf_payoff(r, node_idx, traverser, sampled, rng);
}

/* ── Traversal (mirrors es_solver.py _traverse exactly) ──────────────────────
 *
 * Per-thread `rng` and `local` stats; the regret/strat-sum tables are shared.
 * Under single-thread these are exactly the reference's updates. Under Hogwild
 * the table reads/writes race (Pluribus-style), validated to still converge.
 *
 * Scratch: child_vals is stack-allocated per call frame; SS_MAX_ACTIONS bounds it.
 */
static double es_traverse(ESResolver *r, int node_idx, int traverser,
                          const int *sampled, uint64_t *rng, ESStats *local,
                          int64_t iter) {
    const SSTreeData *t = r->tree;
    const SSNode *nd = &t->nodes[node_idx];
    int type = nd->type;

    if (type == ES_T_FOLD) {
        local->fold_terminals++;
        return es_fold_value(r, node_idx, nd, traverser);
    }
    if (type == ES_T_SHOWDOWN) {
        local->showdown_terminals++;
        return es_showdown_value(r, node_idx, nd, traverser, sampled);
    }
    if (type == ES_T_LEAF) {
        return es_leaf_value(r, node_idx, traverser, sampled, local, rng);
    }
    /* CHANCE node (OP3 §4 HU flop/turn extension — deal the next street).
     * MCCFR external sampling: descend the ONE child sampled for this CFR
     * iteration (shared across the whole walk, GPU-bit-exact via
     * es_chance_sampled_child). The chance node's value is the value of that
     * sampled child — a single-outcome estimator of E_card[V(child)]. NO regret
     * is accumulated here (it is not a decision node), matching the GPU CFV
     * kernel (street_solve.cu:1880-1894: cfv[chance] = cfv[sampled_child]).
     * Keying the draw by (iter, chance_node_idx) — NOT a per-call RNG — is what
     * makes the deep HU solve correct: every visit to this chance node within
     * one iteration returns the same card, so the traverser's child_vals across
     * its actions are comparable and the regret is coherent. */
    if (type == ES_T_CHANCE) {
        int nc = nd->num_children;
        if (nc <= 0) return 0.0;  /* defensive: empty chance node */
        if (es_chance_enumerate(r)) {
            /* CORRECT (default): ENUMERATE the card outcomes and average — the
             * chance node's value is E_card[V(child)] over all dealt cards
             * (uniform). This is zero-variance at the chance node (the GPU's
             * full-reach equivalent) and is what makes the deep multi-street HU
             * solve converge correctly: every action's continuation is evaluated
             * under the SAME full distribution of runouts, so the traverser's
             * regret across actions is commensurate. Tractable for a depth-
             * limited HU subgame (turn ~47, river ~46 outcomes). Card-conflict
             * worlds are handled at the downstream terminals (impossible runouts
             * contribute their neutral terminal value, same as the GPU). */
            double acc = 0.0;
            for (int a = 0; a < nc; a++) {
                int child = t->children[nd->first_child + a];
                acc += es_traverse(r, child, traverser, sampled, rng, local, iter);
            }
            return acc / (double)nc;
        }
        /* SAMPLED (diagnostic): one card per iter — high variance, incoherent
         * across betting lines unless keyed by street. Kept for A/B only. */
        int a = es_chance_sampled_child_keyed(r->master_seed, nd->chance_node_idx,
                                              nd->num_board, nc, iter);
        int child = t->children[nd->first_child + a];
        return es_traverse(r, child, traverser, sampled, rng, local, iter);
    }

    /* DECISION node */
    local->decision_visits++;
    int acting = nd->player;
    int na = nd->num_children;
    /* HERO-LINE FREEZING (2026-09-09): a node whose action is fixed by the
     * api (hero's earlier decision on this street) is walked through its
     * frozen child with probability one, for the traverser and for an
     * opponent alike, and accumulates no regret or strategy sum. */
    {
        int fa = t->frozen_action ? t->frozen_action[node_idx] : -1;
        if (fa >= 0 && fa < na) {
            int child = t->children[nd->first_child + fa];
            return es_traverse(r, child, traverser, sampled, rng, local, iter);
        }
    }
    int hand = sampled[acting];
    if (hand < 0) {
        /* acting player has no assignable hand — pass through child 0 (rare;
         * matches es_solver.py). */
        int child = t->children[nd->first_child];
        return es_traverse(r, child, traverser, sampled, rng, local, iter);
    }

    int64_t off = r->node_off[node_idx];
    /* regret-matched sigma for (node, acting-hand) */
    double sigma[SS_MAX_ACTIONS];
    {
        const double *row = &r->regrets[off + (int64_t)hand * na];
        es_regret_match(row, na, sigma);
    }

    if (acting == traverser) {
        if (r->visit_count) r->visit_count[node_idx]++;
        /* Explore ALL actions. */
        double child_vals[SS_MAX_ACTIONS];
        double node_val = 0.0;
        for (int a = 0; a < na; a++) {
            int child = t->children[nd->first_child + a];
            double cv = es_traverse(r, child, traverser, sampled, rng, local, iter);
            child_vals[a] = cv;
            node_val += sigma[a] * cv;
        }
        double *reg = &r->regrets[off + (int64_t)hand * na];
        double *ss = &r->strat_sum[off + (int64_t)hand * na];
        if (r->cfr_mode == 1) {
            /* LINEAR CFR — matches the GPU (street_solve.cu:1933-1943):
             * regret += residual, then regrets ·= t/(t+1) per visit;
             * strat-sum += t·σ. Converges to the same Nash, faster. */
            double d = (double)iter / ((double)iter + 1.0);
            for (int a = 0; a < na; a++) {
                double nr = reg[a] + (child_vals[a] - node_val);
                if (nr < ES_REGRET_FLOOR) nr = ES_REGRET_FLOOR;
                reg[a] = nr * d;
                ss[a] += (double)iter * sigma[a];
            }
        } else {
            /* VANILLA — the validated reference fixed point (no discount, no
             * iter weight; matches mccfr_blueprint.c:2037 + es_solver.py:341). */
            for (int a = 0; a < na; a++) {
                double nr = reg[a] + (child_vals[a] - node_val);
                if (nr < ES_REGRET_FLOOR) nr = ES_REGRET_FLOOR;
                reg[a] = nr;
                ss[a] += sigma[a];
            }
        }
        return node_val;
    } else {
        if (es_enumerate_opponent(r)) {
            /* FULL CFR (default): recurse into ALL of the opponent's actions and
             * return the sigma-weighted expected value. No regret/strat-sum is
             * updated here (the opponent's own regrets update when THEY are the
             * traverser). Combined with chance enumeration, the whole traversal
             * is DETERMINISTIC — no opponent-action sampling variance — so the
             * depth-limited 2p subgame converges monotonically (the oscillation
             * the single-action-sample path showed on large ranges is gone). */
            double ev = 0.0;
            for (int a = 0; a < na; a++) {
                int child = t->children[nd->first_child + a];
                ev += sigma[a] * es_traverse(r, child, traverser, sampled, rng,
                                             local, iter);
            }
            return ev;
        }
        /* SAMPLED (diagnostic): one opponent action from sigma — external
         * sampling, lower per-iter cost but high variance on large ranges. */
        double rd = es_rng_double(rng);
        double cum = 0.0;
        int a = na - 1;
        for (int k = 0; k < na; k++) {
            cum += sigma[k];
            if (rd <= cum) { a = k; break; }
        }
        int child = t->children[nd->first_child + a];
        return es_traverse(r, child, traverser, sampled, rng, local, iter);
    }
}

typedef double (*ESVecScratch)[SS_MAX_ACTIONS][SS_MAX_HANDS];

/* Exact number of simultaneously-live traverser-decision frames on the
 * deepest root-to-terminal path. The tree is acyclic by construction (the
 * recursive traversals below rely on the same property), so this structural
 * prepass gives a tight scratch bound without using SS_MAX_TREE_NODES slots. */
static int es_vec_scratch_slots_node(const SSTreeData *t, int node_idx,
                                     int traverser) {
    const SSNode *nd = &t->nodes[node_idx];
    int max_below = 0;
    for (int a = 0; a < nd->num_children; a++) {
        int child = t->children[nd->first_child + a];
        int below = es_vec_scratch_slots_node(t, child, traverser);
        if (below > max_below) max_below = below;
    }
    return max_below
           + (nd->type == ES_T_DECISION && nd->player == traverser ? 1 : 0);
}

static int es_vec_scratch_slots(const ESResolver *r) {
    int max_slots = 0;
    for (int trav = 0; trav < r->np; trav++) {
        int slots = es_vec_scratch_slots_node(r->tree, 0, trav);
        if (slots > max_slots) max_slots = slots;
    }
    return max_slots;
}

static ESVecScratch es_alloc_vec_scratch(int slots) {
    if (slots <= 0) return NULL;
    ESVecScratch scratch = (ESVecScratch)malloc(
        (size_t)slots * sizeof(*scratch));
    if (!scratch) {
        fprintf(stderr,
                "FATAL es_run: OOM allocating vec scratch "
                "(%d slots x %d actions x %d hands)\n",
                slots, SS_MAX_ACTIONS, SS_MAX_HANDS);
        abort();
    }
    return scratch;
}

/* ── Vectorized traverser ────────────────────────────────────────────────────
 *
 * One walk down a SINGLE opponent-sampled path updates ALL traverser hands at
 * once. The opponents' path (hand + sampled action) does NOT depend on the
 * traverser's hand, so the tree traversal is SHARED across hands; only the
 * traverser's per-hand regret-match / value / update differ. This amortizes the
 * tree-walk overhead across all nh traverser hands and is the textbook
 * external-sampling variance reduction (fewer iters for the same exploitability).
 *
 * Returns per-hand traverser values into out_vals[0..nh) (nh = the traverser's
 * hand count). `dead[h]` (precomputed once per walk) marks traverser hands that
 * share a card with the board or an opponent's sampled hand — those worlds are
 * impossible, so the hand gets NO update this walk and a neutral 0 value (it
 * never enters a regret/strat-sum or showdown/leaf computation). This dead-hand
 * skip is BOTH crash-avoidance (no duplicate cards into eval7) AND the unbiased
 * estimator (a hand is only updated in card-removal-compatible worlds).
 *
 * `iter` weight: advances by 1 per vectorized walk (NOT by nh) — one vec-walk is
 * one CFR round over all traverser infosets; each hand still receives exactly
 * ONE regret update per walk (see VECTOR_TRAVERSER_DESIGN.md Gate 2, settled).
 */
static void es_traverse_vec(ESResolver *r, int node_idx, int traverser,
                            const int *opp_sampled, const int *dead,
                            uint64_t *rng, ESStats *local, int64_t iter,
                            ESVecScratch scratch, int scratch_depth,
                            double *out_vals) {
    const SSTreeData *t = r->tree;
    const SSNode *nd = &t->nodes[node_idx];
    int type = nd->type;
    int nh = t->num_hands[traverser];

    if (type == ES_T_FOLD) {
        /* Hand-independent for the traverser: net = (pot - my_bet) if traverser
         * is the winner (== nd->player), else -my_bet. Broadcast (dead hands get
         * 0 — they don't exist in this world). */
        local->fold_terminals++;
        double v = es_fold_value(r, node_idx, nd, traverser);
        for (int h = 0; h < nh; h++) out_vals[h] = dead[h] ? 0.0 : v;
        return;
    }
    if (type == ES_T_SHOWDOWN) {
        /* Per-hand showdown: traverser's hand h vs opponents' FIXED sampled
         * hands. eval7 the traverser side per live hand; opponents are constant
         * for this walk so precompute their strengths once.
         * Single-pot fast path (bit-exact historical behavior; exact on every
         * node the current builder emits) + layered side-pot path for uneven
         * this-street all-ins — see the scope note above es_layered_award. */
        local->showdown_terminals++;
        int np = r->np;
        double my_bet = es_my_bet(r, node_idx, nd, traverser);
        if (!nd->active_players[traverser]) {
            for (int h = 0; h < nh; h++) out_vals[h] = dead[h] ? 0.0 : -my_bet;
            return;
        }
        /* NODE board (the runout dealt through in-tree chance nodes), not the
         * subgame-root t->board — see es_showdown_value for the rationale. */
        int board[5];
        if (nd->num_board == 5) {
            for (int b = 0; b < 5; b++) board[b] = nd->board_cards[b];
        } else {
            for (int b = 0; b < 5; b++) board[b] = t->board[b];
        }
        /* opponents' strengths + count of live opponents (traverser excluded).
         * A sampled opponent hand that COLLIDES with the (now 5-card runout)
         * board is an impossible world — under the OP3 chance extension the turn/
         * river were dealt AFTER the hand was sampled, so a collision can happen
         * even though the root-board sampler didn't reserve those cards. Mirror
         * the GPU showdown kernel (street_solve.cu:123 `blocked`): exclude the
         * conflicting hand. With single-opponent sampling a board-conflicting
         * opponent hand makes the whole walk's showdown undefined, so return the
         * neutral 0 (this world contributes nothing this iter). */
        uint32_t opp_strength[SS_MAX_PLAYERS];
        int opp_live[SS_MAX_PLAYERS]; int n_opp_live = 0;
        for (int p = 0; p < np; p++) {
            if (p == traverser || !nd->active_players[p]) continue;
            int hi = opp_sampled[p];
            if (hi < 0) {
                for (int h = 0; h < nh; h++) out_vals[h] = 0.0;
                return;
            }
            int c0 = t->hands[p][hi][0], c1 = t->hands[p][hi][1];
            int conflict = 0;
            if (!es_no_conflict_skip())
                for (int b = 0; b < 5; b++)
                    if (c0 == board[b] || c1 == board[b]) { conflict = 1; break; }
            if (conflict) {
                for (int h = 0; h < nh; h++) out_vals[h] = 0.0;
                return;
            }
            int cards[7] = { board[0], board[1], board[2], board[3], board[4], c0, c1 };
            opp_strength[p] = eval7(cards);
            opp_live[n_opp_live++] = p;
        }
        int single_pot = r->root_pot_count == 0
                         && es_showdown_is_single_pot(nd, np);
        for (int h = 0; h < nh; h++) {
            if (dead[h]) { out_vals[h] = 0.0; continue; }
            int c0 = t->hands[traverser][h][0], c1 = t->hands[traverser][h][1];
            /* Traverser hand colliding with the runout board: impossible world,
             * contributes 0 (GPU `blocked`). */
            int tconf = 0;
            if (!es_no_conflict_skip())
                for (int b = 0; b < 5; b++)
                    if (c0 == board[b] || c1 == board[b]) { tconf = 1; break; }
            if (tconf) { out_vals[h] = 0.0; continue; }
            int cards[7] = { board[0], board[1], board[2], board[3], board[4], c0, c1 };
            uint32_t hs = eval7(cards);
            if (single_pot) {
                int nwin = 1; int trav_best = 1;   /* traverser counts itself */
                for (int k = 0; k < n_opp_live; k++) {
                    uint32_t os = opp_strength[opp_live[k]];
                    if (os > hs) { trav_best = 0; break; }
                    if (os == hs) nwin++;
                }
                out_vals[h] = trav_best ? ((double)nd->pot / (double)nwin - my_bet)
                                        : -my_bet;
            } else {
                uint32_t strength[SS_MAX_PLAYERS];
                for (int k = 0; k < n_opp_live; k++)
                    strength[opp_live[k]] = opp_strength[opp_live[k]];
                strength[traverser] = hs;
                double award = r->root_pot_count > 0
                               ? es_root_pot_award(
                                     r, nd, strength, traverser)
                               : es_layered_award(
                                     nd, np, strength, traverser);
                out_vals[h] = award - my_bet;
            }
        }
        return;
    }
    if (type == ES_T_LEAF) {
        /* Per-hand oracle: set sampled[traverser] = hand h's slot, read the
         * opponents' fixed sampled hands. Dead hands skipped. */
        int sampled[SS_MAX_PLAYERS];
        for (int p = 0; p < r->np; p++) sampled[p] = opp_sampled[p];
        for (int h = 0; h < nh; h++) {
            if (dead[h]) { out_vals[h] = 0.0; continue; }
            sampled[traverser] = h;
            out_vals[h] = es_leaf_value(r, node_idx, traverser, sampled,
                                        local, rng);
        }
        return;
    }
    if (type == ES_T_CHANCE) {
        /* MCCFR external sampling — descend the ONE child sampled for this CFR
         * iteration (GPU-bit-exact, shared across the walk). Same discipline as
         * the scalar traverser: per-(iter, chance_node_idx) draw, NOT a per-call
         * RNG, so the per-hand child_vals the traverser combines upstream are
         * computed over a single coherent runout. No regret accumulated here. */
        int nc = nd->num_children;
        if (nc <= 0) {
            for (int h = 0; h < nh; h++) out_vals[h] = 0.0;
            return;
        }
        if (es_chance_enumerate(r)) {
            /* ENUMERATE (default): per-hand average over all dealt cards. */
            double *tmp = (double *)malloc(
                (size_t)SS_MAX_HANDS * sizeof(*tmp));
            if (!tmp) {
                fprintf(stderr,
                        "FATAL es_traverse_vec: OOM allocating chance tmp "
                        "(%d hands)\n",
                        SS_MAX_HANDS);
                abort();
            }
            for (int h = 0; h < nh; h++) out_vals[h] = 0.0;
            for (int a = 0; a < nc; a++) {
                int child = t->children[nd->first_child + a];
                es_traverse_vec(r, child, traverser, opp_sampled, dead, rng,
                                local, iter, scratch, scratch_depth, tmp);
                for (int h = 0; h < nh; h++) out_vals[h] += tmp[h];
            }
            double inv = 1.0 / (double)nc;
            for (int h = 0; h < nh; h++) out_vals[h] *= inv;
            free(tmp);
            return;
        }
        int a = es_chance_sampled_child_keyed(r->master_seed, nd->chance_node_idx,
                                              nd->num_board, nc, iter);
        int child = t->children[nd->first_child + a];
        es_traverse_vec(r, child, traverser, opp_sampled, dead, rng, local,
                        iter, scratch, scratch_depth, out_vals);
        return;
    }

    /* DECISION node */
    local->decision_visits++;
    int acting = nd->player;
    int na = nd->num_children;

    /* HERO-LINE FREEZING: see es_traverse. The frozen child's per-hand vector
     * is the node's vector; no per-hand regret or strategy-sum update. */
    {
        int fa = t->frozen_action ? t->frozen_action[node_idx] : -1;
        if (fa >= 0 && fa < na) {
            int child = t->children[nd->first_child + fa];
            es_traverse_vec(r, child, traverser, opp_sampled, dead, rng, local,
                            iter, scratch, scratch_depth, out_vals);
            return;
        }
    }

    if (acting != traverser) {
        /* Opponent: sample ONE action from THEIR sampled hand's strategy, then
         * recurse ONCE — the returned per-hand vector passes through unchanged
         * (opponent's choice is independent of the traverser's hand). */
        int ohand = opp_sampled[acting];
        if (ohand < 0) {
            int child = t->children[nd->first_child];
            es_traverse_vec(r, child, traverser, opp_sampled, dead, rng, local,
                            iter, scratch, scratch_depth, out_vals);
            return;
        }
        int64_t off = r->node_off[node_idx];
        double sigma[SS_MAX_ACTIONS];
        es_regret_match(&r->regrets[off + (int64_t)ohand * na], na, sigma);
        double rd = es_rng_double(rng);
        double cum = 0.0; int a = na - 1;
        for (int k = 0; k < na; k++) { cum += sigma[k]; if (rd <= cum) { a = k; break; } }
        int child = t->children[nd->first_child + a];
        es_traverse_vec(r, child, traverser, opp_sampled, dead, rng, local,
                        iter, scratch, scratch_depth, out_vals);
        return;
    }

    /* TRAVERSER decision node: explore ALL actions for ALL hands. One shared
     * recursion per action (the opp path doesn't depend on the traverser hand);
     * then per-hand regret-match + combine + update. */
    if (r->visit_count) r->visit_count[node_idx]++;
    int64_t off = r->node_off[node_idx];
    double (*child_vals)[SS_MAX_HANDS] = scratch[scratch_depth];
    for (int a = 0; a < na; a++) {
        int child = t->children[nd->first_child + a];
        es_traverse_vec(r, child, traverser, opp_sampled, dead, rng, local,
                        iter, scratch, scratch_depth + 1, child_vals[a]);
    }
    double d = (double)iter / ((double)iter + 1.0);
    for (int h = 0; h < nh; h++) {
        if (dead[h]) { out_vals[h] = 0.0; continue; }
        double sigma[SS_MAX_ACTIONS];
        double *reg = &r->regrets[off + (int64_t)h * na];
        double *ss = &r->strat_sum[off + (int64_t)h * na];
        es_regret_match(reg, na, sigma);
        double node_val = 0.0;
        for (int a = 0; a < na; a++) node_val += sigma[a] * child_vals[a][h];
        if (r->cfr_mode == 1) {
            for (int a = 0; a < na; a++) {
                double nr = reg[a] + (child_vals[a][h] - node_val);
                if (nr < ES_REGRET_FLOOR) nr = ES_REGRET_FLOOR;
                reg[a] = nr * d;
                ss[a] += (double)iter * sigma[a];
            }
        } else {
            for (int a = 0; a < na; a++) {
                double nr = reg[a] + (child_vals[a][h] - node_val);
                if (nr < ES_REGRET_FLOOR) nr = ES_REGRET_FLOOR;
                reg[a] = nr;
                ss[a] += sigma[a];
            }
        }
        out_vals[h] = node_val;
    }
}

/* Build the dead-hand mask for a vec walk: traverser hand h is dead if either
 * card collides with the board or any opponent's sampled hand. */
static void es_build_dead_mask(ESResolver *r, int traverser,
                               const int *opp_sampled, int *dead) {
    const SSTreeData *t = r->tree;
    int nh = t->num_hands[traverser];
    uint64_t used = 0;
    for (int b = 0; b < t->num_board; b++) used |= (1ULL << t->board[b]);
    for (int p = 0; p < r->np; p++) {
        if (p == traverser) continue;
        int hi = opp_sampled[p];
        if (hi < 0) continue;
        used |= (1ULL << t->hands[p][hi][0]);
        used |= (1ULL << t->hands[p][hi][1]);
    }
    for (int h = 0; h < nh; h++) {
        int c0 = t->hands[traverser][h][0];
        int c1 = t->hands[traverser][h][1];
        dead[h] = (((used >> c0) & 1ULL) || ((used >> c1) & 1ULL)
                   || t->weights[traverser][h] <= 0.0f) ? 1 : 0;
    }
}

/* ── Stats merge (atomic-free; called under critical or single-thread) ─────── */
void es_merge_stats(ESStats *dst, const ESStats *src) {
    dst->traversals += src->traversals;
    dst->leaf_rollouts += src->leaf_rollouts;
    dst->fold_terminals += src->fold_terminals;
    dst->showdown_terminals += src->showdown_terminals;
    dst->decision_visits += src->decision_visits;
}

/* One vec-mode traverser pass: sample OPPONENTS ONLY (the traverser's cards must
 * NOT be reserved — see es_sample_opponents), build the per-hand dead mask, then
 * walk once covering ALL of `trav`'s hands. */
static inline void es_vec_pass(ESResolver *r, int trav, uint64_t *rng,
                               ESStats *local, int64_t iter,
                               ESVecScratch scratch) {
    int sampled[SS_MAX_PLAYERS];
    int dead[SS_MAX_HANDS];
    double out_vals[SS_MAX_HANDS];
#ifdef ES_VEC_OPP_SAMPLER
    es_sample_opponents(r, rng, trav, sampled);   /* EXPERIMENTAL: don't-reserve-trav (UNVALIDATED) */
#else
    es_sample_hands(r, rng, sampled);             /* VALIDATED sampler (default) */
#endif
    es_build_dead_mask(r, trav, sampled, dead);
    es_traverse_vec(r, 0, trav, sampled, dead, rng, local, iter,
                    scratch, 0, out_vals);
}

/* ── Run loop ────────────────────────────────────────────────────────────── */

int es_run(ESResolver *r, int64_t iterations, int num_threads) {
    if (!r) { es_set_error("resolver is NULL"); return -1; }
    r->iterations_run = 0;
    if (r->root_pot_count > 0 && r->reach_mode) {
        es_set_error("root-pot geometry is not supported by reach mode");
        return -1;
    }
    if (r->oracle_is_sampled && (r->vec_traverser || r->reach_mode)) {
        es_set_error("sampled leaf oracle is scalar-only "
                     "(vec_mode=%d reach_mode=%d)",
                     r->vec_traverser, r->reach_mode);
        return -1;
    }
    if (r->exact_allin_leaves && es_prepare_exact_allin_leaves(r) != 0)
        return -1;
    if (iterations <= 0) return 0;

    /* Full-per-hand-reach vector CFR: dedicated deterministic path (no sampling,
     * single-thread, Linear CFR + chance-enumerate by construction). 2p-first.
     * num_threads is ignored (the reach pass is already O(nodes·hands) per iter,
     * not RNG-bound; parallelize later if needed). */
    if (r->reach_mode) return es_run_reach(r, iterations, num_threads);

#ifdef _OPENMP
    int nthreads = num_threads;
    if (nthreads <= 0) nthreads = omp_get_num_procs();
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
    (void)num_threads;
#endif

    int np = r->np;
    int vec_scratch_count = r->vec_traverser ? es_vec_scratch_slots(r) : 0;
    uint64_t deadline_start_ms = (r->deadline_ms > 0)
                                 ? es_monotonic_ms() : 0;

    if (nthreads == 1) {
        /* Deterministic single-thread path (the validated reference). */
        uint64_t rng = r->master_seed ^ 0xD1B54A32D192ED03ULL;
        ESStats local; memset(&local, 0, sizeof(local));
        int sampled[SS_MAX_PLAYERS];
        int64_t completed = 0;
        ESVecScratch vec_scratch = es_alloc_vec_scratch(vec_scratch_count);
        for (int64_t it = 0; it < iterations; it++) {
            if ((it & 4095LL) == 0
                && es_deadline_expired(r, deadline_start_ms))
                break;
            for (int trav = 0; trav < np; trav++) {
                if (r->vec_traverser) {
                    /* vec mode walks ALL traverser hands; es_vec_pass samples
                     * OPPONENTS ONLY (traverser cards not reserved). */
                    local.traversals++;
                    es_vec_pass(r, trav, &rng, &local, it + 1, vec_scratch);
                } else {
                    es_sample_hands_for_traverser(r, &rng, trav, sampled);
                    if (sampled[trav] < 0) continue;   /* traverser unassignable */
                    local.traversals++;
                    es_traverse(r, 0, trav, sampled, &rng, &local, it + 1);
                }
            }
            completed++;
        }
        free(vec_scratch);
        es_merge_stats(&r->stats, &local);
        r->iterations_run = completed;
        return 0;
    }

#ifdef _OPENMP
    /* Hogwild: partition iterations across threads. Shared regret/strat-sum
     * tables, per-thread RNG seeded distinctly. Racy accumulation (Pluribus
     * Hogwild) — validated to still converge. */
    /* Explicit dynamic chunk dispatch is equivalent to schedule(dynamic, 2048)
     * for global Linear-CFR iteration weights, but can stop claiming work when
     * a deadline expires. An OpenMP worksharing loop cannot portably break and
     * would otherwise churn through all remaining requested indices. */
    _Atomic(int64_t) next_it;
    _Atomic(int) deadline_stop;
    atomic_init(&next_it, 0);
    atomic_init(&deadline_stop, 0);
    int64_t completed = 0;
    int use_deadline = (r->deadline_ms > 0);
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        uint64_t rng = r->master_seed
                       ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(tid + 1))
                       ^ 0xD1B54A32D192ED03ULL;
        /* mix once so low-tid states diverge immediately */
        (void)es_rng_next(&rng); (void)es_rng_next(&rng);
        ESStats local; memset(&local, 0, sizeof(local));
        int sampled[SS_MAX_PLAYERS];
        int64_t local_completed = 0;
        ESVecScratch vec_scratch = es_alloc_vec_scratch(vec_scratch_count);

        /* The shared cursor keeps concurrently processed `it` values near
         * current global progress. This is load-bearing for LINEAR CFR: the
         * t-weight must be the global iteration, not a per-thread chunk index.
         * A coarse chunk amortizes the shared-cursor atomic. */
        for (;;) {
            if (use_deadline && atomic_load(&deadline_stop)) break;
            int64_t begin = atomic_fetch_add(&next_it, 2048);
            if (begin >= iterations) break;
            int64_t remain = iterations - begin;
            int64_t end = begin + (remain < 2048 ? remain : 2048);

            for (int64_t it = begin; it < end; it++) {
                if (use_deadline && atomic_load(&deadline_stop)) break;
                if ((it & 4095LL) == 0
                    && es_deadline_expired(r, deadline_start_ms)) {
                    atomic_store(&deadline_stop, 1);
                    break;
                }
                for (int trav = 0; trav < np; trav++) {
                    if (r->vec_traverser) {
                        local.traversals++;
                        es_vec_pass(r, trav, &rng, &local, it + 1,
                                    vec_scratch);
                    } else {
                        es_sample_hands_for_traverser(r, &rng, trav, sampled);
                        if (sampled[trav] < 0) continue;
                        local.traversals++;
                        es_traverse(r, 0, trav, sampled, &rng, &local, it + 1);
                    }
                }
                local_completed++;
            }
        }
        free(vec_scratch);
        #pragma omp critical
        {
            es_merge_stats(&r->stats, &local);
            completed += local_completed;
        }
    }
    r->iterations_run = completed;
#endif
    return 0;
}

/* ── Forward-only uniform EV (variant-independent multiway gate) ───────────── */

static double es_forward_walk_uniform(ESResolver *r, int node_idx, int player,
                                      const int *sampled, uint64_t *rng) {
    const SSTreeData *t = r->tree;
    const SSNode *nd = &t->nodes[node_idx];
    int type = nd->type;
    if (type == ES_T_LEAF) {
        return es_leaf_payoff(r, node_idx, player, sampled, rng);
    }
    if (type == ES_T_FOLD) return es_fold_value(r, node_idx, nd, player);
    if (type == ES_T_SHOWDOWN)
        return es_showdown_value(r, node_idx, nd, player, sampled);
    /* DECISION or CHANCE: uniform over children. */
    int na = nd->num_children;
    int a = es_rng_int(rng, na);
    int child = t->children[nd->first_child + a];
    return es_forward_walk_uniform(r, child, player, sampled, rng);
}

int es_forward_ev_uniform(ESResolver *r, int player, int64_t iterations,
                          double *out_mean, double *out_se) {
    if (!r) { es_set_error("resolver is NULL"); return -1; }
    if (player < 0 || player >= r->np) { es_set_error("player %d", player); return -1; }
    if (r->exact_allin_leaves && es_prepare_exact_allin_leaves(r) != 0)
        return -1;
    uint64_t rng = r->master_seed ^ (0x2545F4914F6CDD1DULL * (uint64_t)(player + 1));
    int sampled[SS_MAX_PLAYERS];
    double tot = 0.0, tot_sq = 0.0;
    int64_t n = 0;
    for (int64_t it = 0; it < iterations; it++) {
        es_sample_hands(r, &rng, sampled);
        if (sampled[player] < 0) continue;
        double v = es_forward_walk_uniform(r, 0, player, sampled, &rng);
        tot += v; tot_sq += v * v; n++;
    }
    double mean = (n > 0) ? tot / (double)n : 0.0;
    double se;
    if (n > 1) {
        double var = tot_sq / (double)n - mean * mean;
        if (var < 0.0) var = 0.0;
        se = sqrt(var / (double)n);
    } else {
        se = INFINITY;
    }
    if (out_mean) *out_mean = mean;
    if (out_se) *out_se = se;
    return 0;
}

/* ── Strategy readout ────────────────────────────────────────────────────── */

int es_avg_strategy(const ESResolver *r, int node_idx, double *out, int out_cap) {
    if (!r || node_idx < 0 || node_idx >= r->num_nodes) {
        es_set_error("bad node_idx"); return -1;
    }
    int64_t off = r->node_off[node_idx];
    int na = r->node_na[node_idx];
    /* CHANCE nodes (and all non-decision nodes) have node_off = -1 — they hold
     * NO strategy. This is correct, not a missing case: a chance node is a
     * card-deal, not a player decision. The strategy of interest lives at the
     * DECISION nodes the traversers reach DOWNSTREAM of the sampled chance
     * outcome; their strat_sum is accumulated normally by the fixed traversers.
     * Reading a chance node's "strategy" is a caller bug → fail loud. */
    if (off < 0 || na <= 0) { es_set_error("node %d not a decision node", node_idx); return -1; }
    int acting = r->tree->nodes[node_idx].player;
    int nh = r->tree->num_hands[acting];
    if (out_cap < nh * na) { es_set_error("buffer too small (%d < %d)", out_cap, nh * na); return -1; }

    for (int h = 0; h < nh; h++) {
        const double *ssrow = &r->strat_sum[off + (int64_t)h * na];
        double *orow = &out[h * na];
        double sum = 0.0;
        for (int a = 0; a < na; a++) sum += ssrow[a];
        if (sum > 0.0) {
            double inv = 1.0 / sum;
            for (int a = 0; a < na; a++) orow[a] = ssrow[a] * inv;
        } else {
            double u = 1.0 / na;
            for (int a = 0; a < na; a++) orow[a] = u;
        }
    }
    return nh * na;
}

double es_strategy_sum_mass(const ESResolver *r, int node_idx, int hand) {
    if (!r || node_idx < 0 || node_idx >= r->num_nodes) {
        es_set_error("bad node_idx");
        return NAN;
    }
    int64_t off = r->node_off[node_idx];
    int na = r->node_na[node_idx];
    if (off < 0 || na <= 0) {
        es_set_error("node %d not a decision node", node_idx);
        return NAN;
    }
    int acting = r->tree->nodes[node_idx].player;
    int nh = r->tree->num_hands[acting];
    if (hand < 0 || hand >= nh) {
        es_set_error("hand %d outside [0,%d) at node %d", hand, nh, node_idx);
        return NAN;
    }
    const double *row = &r->strat_sum[off + (int64_t)hand * na];
    double mass = 0.0;
    for (int a = 0; a < na; a++) mass += row[a];
    return mass;
}

int es_final_strategy(const ESResolver *r, int node_idx, double *out, int out_cap) {
    if (!r || node_idx < 0 || node_idx >= r->num_nodes) {
        es_set_error("bad node_idx"); return -1;
    }
    int64_t off = r->node_off[node_idx];
    int na = r->node_na[node_idx];
    if (off < 0 || na <= 0) { es_set_error("node %d not a decision node", node_idx); return -1; }
    int acting = r->tree->nodes[node_idx].player;
    int nh = r->tree->num_hands[acting];
    if (out_cap < nh * na) { es_set_error("buffer too small"); return -1; }

    for (int h = 0; h < nh; h++) {
        const double *rrow = &r->regrets[off + (int64_t)h * na];
        es_regret_match(rrow, na, &out[h * na]);
    }
    return nh * na;
}

void es_get_stats(const ESResolver *r, ESStats *out) {
    if (r && out) *out = r->stats;
}
void es_reset_stats(ESResolver *r) {
    if (r) memset(&r->stats, 0, sizeof(r->stats));
}

/* ── TEST-ONLY exports — side-pot showdown validation (validate_sidepot.py) ──
 *
 * Drive the REAL showdown code on a caller-described single showdown node with
 * one fixed hand per player (so `sampled` / `opp_sampled` are all index 0).
 * Returns the traverser's NET value (award − own street bet). NAN on invalid
 * arguments. These exist so the ≥100k-case Python fuzz gate can hit both the
 * scalar path (es_showdown_value) and the vectorized path (the inline showdown
 * block in es_traverse_vec) byte-for-byte as shipped — no reimplementation. */

static int es_test_leaf_args_valid(const ESResolver *r, int node_idx,
                                   int traverser, const int *sampled_hands) {
    if (!r || !sampled_hands || node_idx < 0 || node_idx >= r->num_nodes
        || traverser < 0 || traverser >= r->np
        || r->tree->nodes[node_idx].type != ES_T_LEAF)
        return 0;
    for (int p = 0; p < r->np; p++) {
        int h = sampled_hands[p];
        if (h < -1 || h >= r->tree->num_hands[p]) return 0;
    }
    return sampled_hands[traverser] >= 0;
}

ES_EXPORT double es_test_leaf_value(
    ESResolver *r, int node_idx, int traverser, const int *sampled_hands)
{
    if (!es_test_leaf_args_valid(r, node_idx, traverser, sampled_hands))
        return NAN;
    if (r->exact_allin_leaves && es_prepare_exact_allin_leaves(r) != 0)
        return NAN;
    uint64_t rng = r->master_seed ^ 0x9E3779B97F4A7C15ULL;
    ESStats local; memset(&local, 0, sizeof(local));
    return es_leaf_value(r, node_idx, traverser, sampled_hands, &local, &rng);
}

ES_EXPORT double es_test_leaf_value_vec(
    ESResolver *r, int node_idx, int traverser, const int *sampled_hands)
{
    if (!es_test_leaf_args_valid(r, node_idx, traverser, sampled_hands))
        return NAN;
    if (r->exact_allin_leaves && es_prepare_exact_allin_leaves(r) != 0)
        return NAN;
    int dead[SS_MAX_HANDS];
    double out_vals[SS_MAX_HANDS];
    es_build_dead_mask(r, traverser, sampled_hands, dead);
    uint64_t rng = r->master_seed ^ 0x9E3779B97F4A7C15ULL;
    ESStats local; memset(&local, 0, sizeof(local));
    es_traverse_vec(r, node_idx, traverser, sampled_hands, dead, &rng,
                    &local, 1, NULL, 0, out_vals);
    return out_vals[sampled_hands[traverser]];
}

static void es_test_build_ctx(SSTreeData *td, SSNode *node, ESResolver *tmp,
                              int np, const int *board5,
                              const int *hand_c0, const int *hand_c1,
                              const int *active, const int *bets, int pot) {
    memset(td, 0, sizeof(*td));
    td->num_players = np;
    for (int b = 0; b < 5; b++) td->board[b] = board5[b];
    td->num_board = 5;
    for (int p = 0; p < np; p++) {
        td->num_hands[p] = 1;
        td->hands[p][0][0] = hand_c0[p];
        td->hands[p][0][1] = hand_c1[p];
        td->weights[p][0] = 1.0f;
    }
    memset(node, 0, sizeof(*node));
    node->type = ES_T_SHOWDOWN;
    node->player = -1;
    node->leaf_idx = -1;
    node->fold_player = -1;
    node->pot = pot;
    node->num_board = 5;
    for (int b = 0; b < 5; b++) node->board_cards[b] = board5[b];
    for (int p = 0; p < np; p++) {
        node->bets[p] = bets[p];
        node->active_players[p] = active[p] ? 1 : 0;
        if (active[p]) node->num_players++;
    }
    td->nodes = node;
    td->num_nodes = 1;
    memset(tmp, 0, sizeof(*tmp));
    tmp->tree = td;
    tmp->np = np;
    tmp->num_nodes = 1;
}

ES_EXPORT double es_test_showdown_value(
    int np, const int *board5, const int *hand_c0, const int *hand_c1,
    const int *active, const int *bets, int pot, int traverser)
{
    if (np < 2 || np > SS_MAX_PLAYERS || traverser < 0 || traverser >= np)
        return NAN;
    SSTreeData td; SSNode node; ESResolver tmp;
    es_test_build_ctx(&td, &node, &tmp, np, board5, hand_c0, hand_c1,
                      active, bets, pot);
    int sampled[SS_MAX_PLAYERS];
    for (int p = 0; p < np; p++) sampled[p] = 0;
    return es_showdown_value(&tmp, -1, &node, traverser, sampled);
}

ES_EXPORT double es_test_showdown_value_vec(
    int np, const int *board5, const int *hand_c0, const int *hand_c1,
    const int *active, const int *bets, int pot, int traverser)
{
    if (np < 2 || np > SS_MAX_PLAYERS || traverser < 0 || traverser >= np)
        return NAN;
    SSTreeData td; SSNode node; ESResolver tmp;
    es_test_build_ctx(&td, &node, &tmp, np, board5, hand_c0, hand_c1,
                      active, bets, pot);
    int opp_sampled[SS_MAX_PLAYERS];
    for (int p = 0; p < np; p++) opp_sampled[p] = 0;
    int dead[1] = { 0 };
    double out_vals[1] = { 0.0 };
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    ESStats local; memset(&local, 0, sizeof(local));
    es_traverse_vec(&tmp, 0, traverser, opp_sampled, dead, &rng, &local,
                    1, NULL, 0, out_vals);
    return out_vals[0];
}

static int es_test_build_root_pot_ctx(
    SSTreeData *td, SSNode *node, ESResolver *tmp,
    int np, const int *board5, const int *hand_c0, const int *hand_c1,
    const int *active, const int *bets, int pot,
    const int *root_pot_amounts,
    const uint32_t *root_pot_eligible_masks,
    int num_root_pots)
{
    if (np < 2 || np > 3
        || !td || !node || !tmp || !board5 || !hand_c0 || !hand_c1
        || !active || !bets
        || !root_pot_amounts || !root_pot_eligible_masks
        || num_root_pots < 1 || num_root_pots > ES_MAX_ROOT_POT_LAYERS)
        return -1;

    const uint32_t allowed = (1u << np) - 1u;
    int64_t root_total = 0;
    int64_t bet_total = 0;
    for (int i = 0; i < num_root_pots; i++) {
        int amount = root_pot_amounts[i];
        uint32_t mask = root_pot_eligible_masks[i];
        if (amount < 0 || mask == 0 || (mask & ~allowed) != 0)
            return -1;
        root_total += (int64_t)amount;
        if (root_total > INT_MAX) return -1;
    }
    if (root_pot_eligible_masks[0] != allowed) return -1;
    for (int i = 1; i < num_root_pots; i++)
        if (root_pot_eligible_masks[i] == root_pot_eligible_masks[i - 1]
            || (root_pot_eligible_masks[i]
                & ~root_pot_eligible_masks[i - 1]) != 0)
            return -1;
    for (int p = 0; p < np; p++) {
        if (bets[p] < 0) return -1;
        bet_total += (int64_t)bets[p];
    }
    if (root_total + bet_total != (int64_t)pot) return -1;

    es_test_build_ctx(td, node, tmp, np, board5, hand_c0, hand_c1,
                      active, bets, pot);
    td->starting_pot = (int)root_total;
    tmp->root_pot_count = num_root_pots;
    tmp->root_pot_total = (int)root_total;
    for (int i = 0; i < num_root_pots; i++) {
        tmp->root_pot_amounts[i] = root_pot_amounts[i];
        tmp->root_pot_eligible_masks[i] = root_pot_eligible_masks[i];
    }
    return 0;
}

ES_EXPORT double es_test_showdown_value_root_pots(
    int np, const int *board5, const int *hand_c0, const int *hand_c1,
    const int *active, const int *bets, int pot, int traverser,
    const int *root_pot_amounts,
    const uint32_t *root_pot_eligible_masks,
    int num_root_pots)
{
    if (traverser < 0 || traverser >= np) return NAN;
    SSTreeData td; SSNode node; ESResolver tmp;
    if (es_test_build_root_pot_ctx(
            &td, &node, &tmp, np, board5, hand_c0, hand_c1,
            active, bets, pot, root_pot_amounts,
            root_pot_eligible_masks, num_root_pots) != 0)
        return NAN;

    int sampled[SS_MAX_PLAYERS];
    for (int p = 0; p < np; p++) sampled[p] = 0;
    return es_showdown_value(&tmp, -1, &node, traverser, sampled);
}

ES_EXPORT double es_test_showdown_value_root_pots_vec(
    int np, const int *board5, const int *hand_c0, const int *hand_c1,
    const int *active, const int *bets, int pot, int traverser,
    const int *root_pot_amounts,
    const uint32_t *root_pot_eligible_masks,
    int num_root_pots)
{
    if (traverser < 0 || traverser >= np) return NAN;
    SSTreeData td; SSNode node; ESResolver tmp;
    if (es_test_build_root_pot_ctx(
            &td, &node, &tmp, np, board5, hand_c0, hand_c1,
            active, bets, pot, root_pot_amounts,
            root_pot_eligible_masks, num_root_pots) != 0)
        return NAN;

    int opp_sampled[SS_MAX_PLAYERS];
    for (int p = 0; p < np; p++) opp_sampled[p] = 0;
    int dead[1] = { 0 };
    double out_vals[1] = { 0.0 };
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    ESStats local; memset(&local, 0, sizeof(local));
    es_traverse_vec(&tmp, 0, traverser, opp_sampled, dead, &rng, &local,
                    1, NULL, 0, out_vals);
    return out_vals[0];
}

/* TEST-ONLY: the production terminal payoff at node_idx for `traverser` with
 * the given sampled hands (fold / showdown / depth-limit leaf), scalar path. */
ES_EXPORT double es_test_terminal_payoff(ESResolver *r, int node_idx,
                                         int traverser, const int *sampled) {
    if (!r || node_idx < 0 || node_idx >= r->tree->num_nodes) return 0.0;
    const SSNode *nd = &r->tree->nodes[node_idx];
    if (nd->type == ES_T_FOLD) return es_fold_value(r, node_idx, nd, traverser);
    if (nd->type == ES_T_SHOWDOWN)
        return es_showdown_value(r, node_idx, nd, traverser, sampled);
    if (nd->type == ES_T_LEAF) {
        uint64_t rng = r->master_seed;
        return es_leaf_payoff(r, node_idx, traverser, sampled, &rng);
    }
    return 0.0;
}

