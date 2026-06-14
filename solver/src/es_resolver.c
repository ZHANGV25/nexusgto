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
 */

#include "es_resolver.h"

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

/* Node type codes (mirror street_solve.cuh SS_NODE_*). */
#define ES_T_DECISION 0
#define ES_T_FOLD     1
#define ES_T_SHOWDOWN 2
#define ES_T_CHANCE   3
#define ES_T_LEAF     4

/* Pluribus uses -310M on int32; we use float64 regrets with a generous floor
 * (matches es_solver.py REGRET_FLOOR = -1e12). */
#define ES_REGRET_FLOOR (-1e12)

/* ── Thread-local last-error ─────────────────────────────────────────────── */
static __thread char g_es_error[512];

static void es_set_error(const char *fmt, ...) {
    va_list ap;
    __builtin_va_start(ap, fmt);
    vsnprintf(g_es_error, sizeof(g_es_error), fmt, ap);
    __builtin_va_end(ap);
}
const char *es_get_last_error(void) {
    return g_es_error[0] ? g_es_error : NULL;
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

/* Chance-node handling mode. DEFAULT = ENUMERATE (average over all dealt cards —
 * correct, zero chance-variance, the GPU's full-reach equivalent). Set
 * ES_CHANCE_SAMPLE=1 to fall back to the single-card-per-iter SAMPLED path (the
 * original OP3 design — high variance + cross-line incoherence; kept for A/B and
 * for any future cost-bounded mode). Cached on first read. */
static int es_chance_enumerate(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("ES_CHANCE_SAMPLE");
        int sample = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y' || e[0] == 'Y')) ? 1 : 0;
        v = sample ? 0 : 1;   /* enumerate unless explicitly asked to sample */
    }
    return v;
}
static inline int es_chance_sampled_child_keyed(uint64_t master_seed,
                                                int chance_node_idx, int street_key,
                                                int num_children, int64_t iter) {
    int key = es_chance_key_by_street() ? street_key : chance_node_idx;
    return es_chance_sampled_child(master_seed, key, num_children, iter);
}

/* ── Resolver state ──────────────────────────────────────────────────────── */
struct ESResolver {
    const SSTreeData *tree;
    int np;
    int num_nodes;

    ESLeafOracle oracle;
    void *oracle_ctx;
    uint64_t master_seed;

    /* CFR variant. 0 = VANILLA (the validated reference fixed point — plain
     * per-visit strat-sum, no regret discount). 1 = LINEAR CFR matching the GPU
     * (street_solve.cu:1933-1943): strat-sum += t·σ, regrets ·= t/(t+1) per
     * traverser-node visit. Linear converges to the SAME Nash but faster — the
     * GPU runs Linear, so an iso-quality iteration comparison must give the C ES
     * the same acceleration. Vanilla kept for the validation gates (matches
     * es_solver.py, whose linear_cfr flag is dead). */
    int cfr_mode;

    /* Traverser vectorization. 0 = SCALAR (the validated reference: one walk
     * updates exactly one traverser hand = sampled[traverser]). 1 = VECTORIZED:
     * one walk down a single opponent-sampled path updates ALL traverser hands
     * at once (shared tree traversal, per-hand regret-match + update + leaf).
     * Lower variance ⇒ fewer iters for the same exploitability. The fixed point
     * is IDENTICAL — Gate 1 (es_resolver.c design doc) asserts same exploitability
     * as scalar on a validated 2p/3p spot WITH OVERLAPPING ranges (so the
     * dead-hand card-conflict skip is exercised). Set BEFORE es_run. */
    int vec_traverser;

    /* Per-decision-node dense storage, keyed (node, hand, action).
     * For node i:  regrets[off[i] + hand*na[i] + a]. Terminal/non-decision
     * nodes have off[i] = -1 and na[i] = 0. The hand axis uses the ACTING
     * player's hand count at node i (an info set is (node, acting-player hand)).
     */
    int64_t *node_off;        /* [num_nodes]; -1 if not a decision node */
    int *node_na;             /* [num_nodes]; 0 if not a decision node */
    int64_t total_elems;
    double *regrets;          /* [total_elems], shared (Hogwild) */
    double *strat_sum;        /* [total_elems], shared (Hogwild) */

    /* Stats. Under Hogwild these are racy counters (informational, like the
     * Python reference's stats); the validated metric is convergence, not
     * exact counts. */
    ESStats stats;

    /* DEBUG: per-node traverser-visit counter (incremented when acting==traverser
     * at a decision node). Allocated lazily by es_enable_visit_counter; NULL =
     * off (zero overhead). Used to localize the vec reachability bug — find the
     * first node on a path where vec's count diverges from scalar. */
    int64_t *visit_count;   /* [num_nodes] or NULL */
};

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
    r->master_seed = seed ? seed : 0x9E3779B97F4A7C15ULL;
    if (es_build_offsets(r) != 0) { es_free(r); return NULL; }
    return r;
}

void es_set_cfr_mode(ESResolver *r, int mode) {
    if (r) r->cfr_mode = (mode == 1) ? 1 : 0;
}

void es_set_vec_mode(ESResolver *r, int mode) {
    if (r) r->vec_traverser = (mode == 1) ? 1 : 0;
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
    free(r->node_off);
    free(r->node_na);
    free(r->regrets);
    free(r->strat_sum);
    free(r->visit_count);
    free(r);
}

/* ── Regret matching for one (node, hand) row ────────────────────────────────
 * sigma(a) ∝ max(0, R(a)); uniform if all <= 0. Mirrors
 * es_solver.py InfoSets.strategy. Operates on float64 to match the reference.
 */
static inline void es_regret_match(const double *r, int na, double *out) {
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
 * One hand index per player from their reach-weighted range, avoiding card
 * conflicts (board + other players' sampled cards). chosen[p] = -1 if a player
 * can't be assigned a conflict-free hand. Mirrors es_solver.py _sample_hands,
 * including the randomized assignment order so no player is starved.
 */
static void es_sample_hands(ESResolver *r, uint64_t *rng, int *chosen) {
    const SSTreeData *t = r->tree;
    int np = r->np;

    /* used-cards bitmask over 52 cards. */
    uint64_t used = 0;
    for (int b = 0; b < t->num_board; b++) used |= (1ULL << t->board[b]);

    for (int p = 0; p < np; p++) chosen[p] = -1;

    /* randomized order (Fisher-Yates) */
    int order[SS_MAX_PLAYERS];
    for (int p = 0; p < np; p++) order[p] = p;
    for (int i = 0; i < np; i++) {
        int j = i + es_rng_int(rng, np - i);
        int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    for (int oi = 0; oi < np; oi++) {
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
static inline double es_fold_value(const SSNode *nd, int traverser) {
    int winner = nd->player;          /* street_solve.cu sets winner = node->player */
    double my_bet = (double)nd->bets[traverser];
    if (traverser == winner) return (double)nd->pot - my_bet;
    return -my_bet;
}

/* ── SHOWDOWN terminal (river): multiway side-pot layering ────────────────────
 *
 * REPRESENTABLE-SCOPE NOTE (2026-06-10, Lane 3). SSNode carries only
 * per-THIS-STREET bets (SSNode.bets[]) plus a lump `pot` that folds in all
 * prior-street money. Consequences:
 *
 *   1. EXACTLY representable: side pots layered by THIS-street contribution
 *      caps. The lump entry pot is common money every street-entry contestant
 *      has equal claim to — true because the builder zeroes bets[] at every
 *      street entry (subgame root: ss_tree_api.c:76; chance-node street
 *      transition: ss_tree_builder.c:432) with all entrants active and an
 *      equal shared `effective_stack`.
 *   2. NOT representable: side pots from PRIOR-street unequal all-ins (the
 *      per-player prior contributions are unrecoverable from the lump pot).
 *      Out of scope — the river LOOKUP fallback remains for those spots.
 *   3. The CURRENT builder cannot even emit a showdown with unequal ACTIVE
 *      street bets: round_complete (ss_tree_builder.c:146-155) requires
 *      bets[p] == max for every active player, there is no per-player stack
 *      state (`stack` is constant down the street, ss_tree_builder.c:345), a
 *      call always equalizes to max (ss_tree_builder.c:295), and a folder's
 *      street bet is strictly < the final max (you fold facing a bet; max
 *      never decreases). So on every REACHABLE node today the single-pot
 *      award is exact, and es_showdown_is_single_pot routes those nodes
 *      through the historical code BIT-EXACTLY (the regression gate). The
 *      layered path below makes this function correct for any in-scope SSNode
 *      (per-player stacks / unequal mid-street seeding / off-tree all-in
 *      translation — the playing-agent work that lands next).
 *   4. ACCOUNTING CAVEAT (pre-existing, reported not fixed): on a
 *      raise-over-bet the builder sets new_bets = mx + bet_amount where
 *      bet_amount already contains to_call (ss_tree_builder.c:340), while the
 *      trainer does bets[ap] += amount (mccfr_blueprint.c:1981) and BOTH add
 *      only bet_amount to the pot. The builder's bets[] therefore exceed its
 *      own pot increments by to_call per raise-over-bet, so pot − Σ bets is
 *      NOT exactly the entry pot. The layered award is constructed to
 *      distribute EXACTLY nd->pot regardless (slack lands in the base layer),
 *      preserving the system invariant Σ_p award_p == nd->pot.
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

/* SHOWDOWN terminal (river). Single-pot fast path (bit-exact historical
 * behavior, exact for every node the current builder can emit) + layered
 * side-pot path for uneven this-street all-in contributions (scope above). */
static double es_showdown_value(ESResolver *r, const SSNode *nd,
                                int traverser, const int *sampled) {
    const SSTreeData *t = r->tree;
    int np = r->np;
    double my_bet = (double)nd->bets[traverser];
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
    if (es_showdown_is_single_pot(nd, np)) {
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
    return es_layered_award(nd, np, strength, traverser) - my_bet;
}

/* LEAF: lazy betting-leaf value via the caller oracle (one sampled-opponent
 * rollout in the real path; a table lookup in validation). */
static inline double es_leaf_value(ESResolver *r, const SSNode *nd,
                                   int traverser, const int *sampled,
                                   ESStats *local, uint64_t *rng) {
    local->leaf_rollouts++;
    return r->oracle(r->oracle_ctx, nd->leaf_idx, traverser,
                     sampled[traverser], sampled, nd, rng);
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
        return es_fold_value(nd, traverser);
    }
    if (type == ES_T_SHOWDOWN) {
        local->showdown_terminals++;
        return es_showdown_value(r, nd, traverser, sampled);
    }
    if (type == ES_T_LEAF) {
        return es_leaf_value(r, nd, traverser, sampled, local, rng);
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
        if (es_chance_enumerate()) {
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
        /* Opponent: sample ONE action from sigma. */
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
        double v = es_fold_value(nd, traverser);
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
        double my_bet = (double)nd->bets[traverser];
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
            if (hi < 0) { for (int h = 0; h < nh; h++) out_vals[h] = 0.0; return; }
            int c0 = t->hands[p][hi][0], c1 = t->hands[p][hi][1];
            int conflict = 0;
            if (!es_no_conflict_skip())
                for (int b = 0; b < 5; b++)
                    if (c0 == board[b] || c1 == board[b]) { conflict = 1; break; }
            if (conflict) { for (int h = 0; h < nh; h++) out_vals[h] = 0.0; return; }
            int cards[7] = { board[0], board[1], board[2], board[3], board[4], c0, c1 };
            opp_strength[p] = eval7(cards);
            opp_live[n_opp_live++] = p;
        }
        int single_pot = es_showdown_is_single_pot(nd, np);
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
                out_vals[h] = es_layered_award(nd, np, strength, traverser)
                              - my_bet;
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
            local->leaf_rollouts++;
            out_vals[h] = r->oracle(r->oracle_ctx, nd->leaf_idx, traverser,
                                    h, sampled, nd, rng);
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
        if (es_chance_enumerate()) {
            /* ENUMERATE (default): per-hand average over all dealt cards. */
            double tmp[SS_MAX_HANDS];
            for (int h = 0; h < nh; h++) out_vals[h] = 0.0;
            for (int a = 0; a < nc; a++) {
                int child = t->children[nd->first_child + a];
                es_traverse_vec(r, child, traverser, opp_sampled, dead, rng,
                                local, iter, tmp);
                for (int h = 0; h < nh; h++) out_vals[h] += tmp[h];
            }
            double inv = 1.0 / (double)nc;
            for (int h = 0; h < nh; h++) out_vals[h] *= inv;
            return;
        }
        int a = es_chance_sampled_child_keyed(r->master_seed, nd->chance_node_idx,
                                              nd->num_board, nc, iter);
        int child = t->children[nd->first_child + a];
        es_traverse_vec(r, child, traverser, opp_sampled, dead, rng, local,
                        iter, out_vals);
        return;
    }

    /* DECISION node */
    local->decision_visits++;
    int acting = nd->player;
    int na = nd->num_children;

    if (acting != traverser) {
        /* Opponent: sample ONE action from THEIR sampled hand's strategy, then
         * recurse ONCE — the returned per-hand vector passes through unchanged
         * (opponent's choice is independent of the traverser's hand). */
        int ohand = opp_sampled[acting];
        if (ohand < 0) {
            int child = t->children[nd->first_child];
            es_traverse_vec(r, child, traverser, opp_sampled, dead, rng, local,
                            iter, out_vals);
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
                        iter, out_vals);
        return;
    }

    /* TRAVERSER decision node: explore ALL actions for ALL hands. One shared
     * recursion per action (the opp path doesn't depend on the traverser hand);
     * then per-hand regret-match + combine + update. */
    if (r->visit_count) r->visit_count[node_idx]++;
    int64_t off = r->node_off[node_idx];
    /* child_vals[a][h] — na*nh doubles, depth-bounded stack (na<=8, nh<=200). */
    double child_vals[SS_MAX_ACTIONS][SS_MAX_HANDS];
    for (int a = 0; a < na; a++) {
        int child = t->children[nd->first_child + a];
        es_traverse_vec(r, child, traverser, opp_sampled, dead, rng, local,
                        iter, child_vals[a]);
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
static inline void es_merge_stats(ESStats *dst, const ESStats *src) {
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
                               ESStats *local, int64_t iter) {
    int sampled[SS_MAX_PLAYERS];
    int dead[SS_MAX_HANDS];
    double out_vals[SS_MAX_HANDS];
#ifdef ES_VEC_OPP_SAMPLER
    es_sample_opponents(r, rng, trav, sampled);   /* EXPERIMENTAL: don't-reserve-trav (UNVALIDATED) */
#else
    es_sample_hands(r, rng, sampled);             /* VALIDATED sampler (default) */
#endif
    es_build_dead_mask(r, trav, sampled, dead);
    es_traverse_vec(r, 0, trav, sampled, dead, rng, local, iter, out_vals);
}

/* ── Run loop ────────────────────────────────────────────────────────────── */

int es_run(ESResolver *r, int64_t iterations, int num_threads) {
    if (!r) { es_set_error("resolver is NULL"); return -1; }
    if (iterations <= 0) return 0;

#ifdef _OPENMP
    int nthreads = num_threads;
    if (nthreads <= 0) nthreads = omp_get_num_procs();
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
    (void)num_threads;
#endif

    int np = r->np;

    if (nthreads == 1) {
        /* Deterministic single-thread path (the validated reference). */
        uint64_t rng = r->master_seed ^ 0xD1B54A32D192ED03ULL;
        ESStats local; memset(&local, 0, sizeof(local));
        int sampled[SS_MAX_PLAYERS];
        for (int64_t it = 0; it < iterations; it++) {
            for (int trav = 0; trav < np; trav++) {
                if (r->vec_traverser) {
                    /* vec mode walks ALL traverser hands; es_vec_pass samples
                     * OPPONENTS ONLY (traverser cards not reserved). */
                    local.traversals++;
                    es_vec_pass(r, trav, &rng, &local, it + 1);
                } else {
                    es_sample_hands(r, &rng, sampled);
                    if (sampled[trav] < 0) continue;   /* traverser unassignable */
                    local.traversals++;
                    es_traverse(r, 0, trav, sampled, &rng, &local, it + 1);
                }
            }
        }
        es_merge_stats(&r->stats, &local);
        return 0;
    }

#ifdef _OPENMP
    /* Hogwild: partition iterations across threads. Shared regret/strat-sum
     * tables, per-thread RNG seeded distinctly. Racy accumulation (Pluribus
     * Hogwild) — validated to still converge. */
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

        /* schedule(dynamic) so OpenMP hands out `it` from a SHARED counter —
         * concurrently-processed `it` values cluster near current global
         * progress. This is load-bearing for LINEAR CFR: the Linear t-weight
         * (it+1) must track global wall-progress, not a per-thread chunk index.
         * Under schedule(static) thread 23 would stamp weight ≈ 23·T/24 onto the
         * shared strat-sum from wall-time zero (while regrets are immature),
         * incoherent across threads and silently killing Linear's acceleration.
         * Coarse chunk amortizes the shared-counter atomic. (Vanilla mode 0
         * ignores `it`, so it is unaffected either way.) */
        #pragma omp for schedule(dynamic, 2048)
        for (int64_t it = 0; it < iterations; it++) {
            for (int trav = 0; trav < np; trav++) {
                if (r->vec_traverser) {
                    local.traversals++;
                    es_vec_pass(r, trav, &rng, &local, it + 1);
                } else {
                    es_sample_hands(r, &rng, sampled);
                    if (sampled[trav] < 0) continue;
                    local.traversals++;
                    es_traverse(r, 0, trav, sampled, &rng, &local, it + 1);
                }
            }
        }
        #pragma omp critical
        es_merge_stats(&r->stats, &local);
    }
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
        return r->oracle(r->oracle_ctx, nd->leaf_idx, player,
                         sampled[player], sampled, nd, rng);
    }
    if (type == ES_T_FOLD) return es_fold_value(nd, player);
    if (type == ES_T_SHOWDOWN) return es_showdown_value(r, nd, player, sampled);
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
    return es_showdown_value(&tmp, &node, traverser, sampled);
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
                    1, out_vals);
    return out_vals[0];
}
