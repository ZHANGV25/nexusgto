/**
 * es_resolver_internal.h — cross-TU internal declarations shared between
 * es_resolver.c and the reach-CFR translation unit es_resolver_reach.c.
 *
 * NOT a public API. These symbols are extern (each is defined in one of the
 * two files) only so the reach-CFR code can live in a sibling translation
 * unit compiled into the SAME shared library.
 */
#ifndef ES_RESOLVER_INTERNAL_H
#define ES_RESOLVER_INTERNAL_H

#include "es_resolver.h"

/* Node type codes (mirror tree.h SS_NODE_*). Shared by es_resolver.c and
 * es_resolver_reach.c (macros emit no symbol). */
#define ES_T_DECISION 0
#define ES_T_FOLD     1
#define ES_T_SHOWDOWN 2
#define ES_T_CHANCE   3
#define ES_T_LEAF     4

/* Pluribus uses -310M on int32; the resolver uses float64 regrets with a
 * generous floor (matches es_solver.py REGRET_FLOOR = -1e12). */
#define ES_REGRET_FLOOR (-1e12)

/* Immutable exact-showdown cache for one betting-closed depth-limit leaf.
 * Runouts are board-only candidates; the hot path filters the sampled hole
 * cards with runout_masks before reading the hand-major strength rows. */
typedef struct {
    int num_runouts;
    uint64_t board_mask;
    uint64_t *runout_masks;       /* [num_runouts] */
    size_t hand_base[SS_MAX_PLAYERS];
    uint32_t *strengths;          /* [(sum num_hands) * num_runouts] */
} ESExactAllinLeafCache;

/* Full ESResolver state — defined here so es_resolver_reach.c sees the
 * complete type. A struct definition emits no symbol and is not part of the
 * public ABI (ESResolver is opaque in es_resolver.h). */
/* ── Resolver state ──────────────────────────────────────────────────────── */
struct ESResolver {
    const SSTreeData *tree;
    int np;
    int num_nodes;

    ESLeafOracle oracle;
    void *oracle_ctx;
    uint64_t master_seed;

    /* Scalar ES only: one player's traverser pass may mix its modeled range
     * with a fixed actual-hand row. Opponent-as-traverser passes never consult
     * these fields, so their belief about this player's range is unchanged. */
    int actual_traverser_player;  /* -1 = disabled */
    int actual_traverser_hand;
    double actual_traverser_fraction;

    /* Tagged automatically by es_create from the resolver-local sampled
     * trampoline callback. Sampled leaves require scalar traversal: vec mode
     * evaluates hands with incoherent fresh bias draws, while reach mode
     * enumerates hands and shares an RNG through its parallel terminal pass. */
    int oracle_is_sampled;

    /* Per-run deadline configuration + last-run completion count. These stay
     * private rather than extending public ESStats (whose layout is ABI).
     * deadline_ms <= 0 means disabled; iterations_run resets at es_run entry. */
    int64_t deadline_ms;
    int64_t iterations_run;

    /* CFR variant. 0 = VANILLA (the validated reference fixed point — plain
     * per-visit strat-sum, no regret discount). 1 = LINEAR CFR: strat-sum +=
     * t·σ, regrets ·= t/(t+1) per traverser-node visit. Linear converges to the
     * SAME Nash but faster, so an iso-quality iteration comparison against a
     * Linear-CFR solver must give the C ES the same acceleration. Vanilla is
     * the default for the validation gates (matches es_solver.py, whose
     * linear_cfr flag is dead). */
    int cfr_mode;

    /* Traverser vectorization. 0 = SCALAR (the validated reference: one walk
     * updates exactly one traverser hand = sampled[traverser]). 1 = VECTORIZED:
     * one walk down a single opponent-sampled path updates ALL traverser hands
     * at once (shared tree traversal, per-hand regret-match + update + leaf).
     * Lower variance ⇒ fewer iters for the same exploitability. The fixed point
     * is IDENTICAL — python/es_resolver/vec_traverser_check.py asserts the same
     * exploitability as scalar on a validated 2p/3p spot WITH OVERLAPPING ranges
     * (so the dead-hand card-conflict skip is exercised). Set BEFORE es_run. */
    int vec_traverser;

    /* Chance / opponent-action handling modes (per-resolver, set before es_run).
     * -1 = inherit the process default (env var); 0/1 = explicit. Env defaults:
     *   chance:   ENUMERATE unless ES_CHANCE_SAMPLE=1.
     *   opponent: SAMPLE unless ES_OPP_ENUMERATE=1.
     * The caller (cpu_es_solver) sets these per-solve so a 2p chance tree can use
     * (chance-enum + opp-enum + vec) for a deterministic-modulo-opp-hand solve
     * while 5-6p keeps sampling — both in the SAME process (env caching can't do
     * that; these fields can). See es_set_chance_mode / es_set_opp_mode. */
    int chance_mode;   /* -1 default, 0 sample, 1 enumerate */
    int opp_mode;      /* -1 default, 0 sample, 1 enumerate */

    /* Full-per-hand-REACH vector CFR (for interactive 2p chance trees).
     * 0 = OFF (default; the validated sampling traversers above). 1 = ON: carry a
     * per-player per-node per-hand reach vector and integrate ALL opponent hands
     * each iteration with NO opponent-hand sampling (full-reach vector CFR:
     * reach propagation, CFV, showdown value). Zero
     * variance over opponent hands AND chance (chance is ENUMERATED), so the
     * solve converges in ~hundreds of iters in seconds instead of the
     * minutes-per-solve the sampling vec path needs to average out opponent-hand
     * variance. Tractable only for SMALL player counts (the reach pass is O(nodes
     * · max_hands · np) per iter, no action/hand product blow-up — but 5-6p keep
     * SAMPLING, never reach, so the high-N intractability is structural elsewhere,
     * not here). Set via es_set_reach_mode BEFORE es_run; takes the dedicated
     * es_run_reach path and ignores cfr/vec/chance/opp modes (it is always
     * Linear-CFR + chance-enumerate + full reach by construction). */
    int reach_mode;

    /* Reach-CFR scratch (allocated lazily by es_run_reach; freed in es_free).
     * reach[p * num_nodes * max_h + node * max_h + hand] — one block per player.
     * cfv[node * max_h + hand] — the current traverser's counterfactual values.
     * strat[node * SS_MAX_ACTIONS * max_h + a * max_h + hand] — regret-matched
     * strategy per acting player's hand (filled once per traverser-iter).
     * max_h is the per-resolver max over players' hand counts. */
    int reach_max_h;
    double *rc_reach;   /* [np * num_nodes * max_h] or NULL */
    double *rc_cfv;     /* [num_nodes * max_h] or NULL */
    double *rc_strat;   /* [num_nodes * SS_MAX_ACTIONS * max_h] or NULL */

    /* Showdown strength cache (iteration-invariant: board+hand -> 7-card rank).
     * rc_sd_strength[node*np*max_h + player*max_h + hand], 0 = "not yet computed
     * / board-conflicting" (a real eval7 rank is never 0 for a 5-board+2-hole
     * hand). Filled lazily on the first showdown visit per node and reused every
     * iteration — eval7 is the dominant per-iter cost (one per (showdown node,
     * hand)), and it does NOT depend on the strategy. rc_sd_done[node] gates the
     * fill (so it happens once). NULL until the first reach run. */
    uint32_t *rc_sd_strength;   /* [num_nodes * np * max_h] or NULL */
    uint8_t  *rc_sd_done;       /* [num_nodes] 0/1, or NULL */

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
     * off (zero overhead). Used to localize vec reachability problems — find
     * the first node on a path where vec's count diverges from scalar. */
    int64_t *visit_count;   /* [num_nodes] or NULL */

    /* Cold additive sidecar at the end so the hot fields above keep their
     * layout/cache locality.  es_create's calloc leaves root_pot_count=0. */
    int root_pot_count;
    int root_pot_total;
    int root_pot_amounts[ES_MAX_ROOT_POT_LAYERS];
    uint32_t root_pot_eligible_masks[ES_MAX_ROOT_POT_LAYERS];

    /* Opt-in exact evaluation for betting-closed depth-limit leaves. calloc
     * leaves it disabled, so leaves use the caller oracle unless the
     * caller explicitly enables it. Cache entries are indexed by node, because
     * board/bets/active state belong to the node rather than the table leaf id. */
    int exact_allin_leaves;
    int exact_allin_ready;
    uint8_t *exact_allin_leaf_flags;       /* [num_nodes] */
    /* CROSS-STREET CONTRIBUTION CARRY. SSNode.bets holds only
     * THIS street's bets: the builder resets them to 0 below a chance node and
     * keeps the closing street's per-player bets on the chance node itself.
     * node_carry[node*np+p] = chips player p committed on EARLIER streets of
     * this subgame (sum of chance-node bets on the root path). Every terminal
     * charges bets + carry, so a turn call is not free at a river showdown
     * (without the carry, ace-high called a 190-unit jam at 1.00). NULL on a
     * single-street tree, where there is nothing to carry. */
    int *node_carry;                        /* [num_nodes*np] or NULL */
    ESExactAllinLeafCache *exact_allin_leaf_cache; /* [num_nodes] */

    /* Optional vector leaf oracle (es_set_vec_leaf_oracle).
     * NULL = the per-hand scalar-oracle loop in the vector
     * traverser. Cold additive sidecar; calloc leaves it unset. */
    ESLeafOracleVec vec_oracle;
    void *vec_oracle_ctx;
};


/* Hidden-visibility for cross-TU internals (see mccfr_blueprint_internal.h).
 * Keeps these un-exported so a second in-process .so cannot interpose them. */
#ifndef NEXUS_INTERNAL
#  if defined(__GNUC__) || defined(__clang__)
#    define NEXUS_INTERNAL __attribute__((visibility("hidden")))
#  else
#    define NEXUS_INTERNAL
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Set the thread-local last-error string (printf-style). Definition (and the
 * g_es_error thread-local buffer it writes) stay in es_resolver.c. */
NEXUS_INTERNAL void es_set_error(const char *fmt, ...);

/* Cross-platform monotonic clock + deadline predicate shared by the sampling
 * and reach run loops. The start value is captured once at path entry. */
NEXUS_INTERNAL uint64_t es_monotonic_ms(void);
NEXUS_INTERNAL int es_deadline_expired(const ESResolver *r,
                                       uint64_t start_ms);

/* Build the immutable betting-closed flags and strength caches. This is run
 * before Hogwild workers start; the forward/test paths may also call it while
 * single-threaded. */
NEXUS_INTERNAL int es_prepare_exact_allin_leaves(ESResolver *r);

/* Shared no-stats leaf dispatch used by scalar, vector, forward, and reach
 * paths. Exact flagged leaves bypass the caller oracle; all other leaves call
 * the caller oracle with the traversal RNG. */
NEXUS_INTERNAL double es_leaf_payoff(
    ESResolver *r, int node_idx, int traverser,
    const int *sampled, uint64_t *rng);

/* Regret-matching: positive-regret normalize into a strategy (uniform if all
 * non-positive). Definition in es_resolver.c. */
void es_regret_match(const double *r, int na, double *out);

/* Accumulate one ESStats into another (thread-merge). Definition in es_resolver.c. */
void es_merge_stats(ESStats *dst, const ESStats *src);

/* Full per-hand vector reach-CFR driver (np==2 only). Definition in
 * es_resolver_reach.c; called from es_run in es_resolver.c. */
NEXUS_INTERNAL int es_run_reach(ESResolver *r, int64_t iterations, int num_threads);

#ifdef __cplusplus
}
#endif


/* Chips committed on earlier streets of this subgame (0 on single-street
 * trees, on a NULL resolver, or for a synthetic node with node_idx < 0). */
static inline double es_carry(const ESResolver *r, int node_idx, int p) {
    if (!r || !r->node_carry || node_idx < 0) return 0.0;
    return (double)r->node_carry[(size_t)node_idx * (size_t)r->np + (size_t)p];
}

/* The player's total contribution charged at a terminal: this street's bets
 * plus the carry from earlier streets. */
static inline double es_my_bet(const ESResolver *r, int node_idx,
                               const SSNode *nd, int p) {
    return (double)nd->bets[p] + es_carry(r, node_idx, p);
}

#endif /* ES_RESOLVER_INTERNAL_H */
