/**
 * bps_reader.c — Read-only query API for unified .bps blueprint files.
 *
 * Loads decompressed BPS3/BPR3 data into compact node-level hash tables.
 * No OpenMP, no card_abstraction, no training infrastructure — pure reader.
 *
 * Hash functions are copied from mccfr_blueprint.c and MUST stay in sync.
 */

#include "bps_reader.h"
#include "bps_reader_internal.h"
#include "card_abstraction.h"   /* ca_compute_features, ca_nearest_centroid */
#include "hand_eval.h"          /* eval7 (header-only) for river EHS */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Hash functions (must match mccfr_blueprint.c exactly) ──────────── */

/** splitmix64 avalanche mixer — same as mccfr_blueprint.c:252-256. */
static inline uint64_t bp_mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/** splitmix64 hash_combine — same as mccfr_blueprint.c:297-298. */
static inline uint64_t hash_combine(uint64_t a, uint64_t b) {
    return bp_mix64(a ^ bp_mix64(b));
}

/** Legacy boost-style combiner — same as mccfr_blueprint.c:305-308. */
static inline uint64_t hash_combine_boost_legacy(uint64_t a, uint64_t b) {
    a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
    return a;
}

/**
 * Compute action_hash from an action sequence.
 * Same as mccfr_blueprint.c:472-485 with legacy mixer dispatch.
 */
static uint64_t compute_action_hash(const int *actions, int num_actions,
                                     int use_legacy_mixer) {
    uint64_t h = 0xFEDCBA9876543210ULL;
    if (use_legacy_mixer) {
        for (int i = 0; i < num_actions; i++)
            h = hash_combine_boost_legacy(h, (uint64_t)actions[i] * 17 + 3);
    } else {
        for (int i = 0; i < num_actions; i++)
            h = hash_combine(h, (uint64_t)actions[i] * 17 + 3);
    }
    return h;
}

/* ── Internal hash table operations ─────────────────────────────────── */

#define BPS_PROBE_LIMIT 256

/**
 * Compute slot for a node key in the reader's table.
 * This is the reader's OWN slot derivation — does not need to match
 * mccfr_blueprint.c's 5-field composite hash.
 */
static inline int64_t node_slot(const BPSTable *t, uint64_t action_hash,
                                 uint8_t player, uint8_t street) {
    uint64_t h = bp_mix64(action_hash ^ bp_mix64((uint64_t)player * 31 + street));
    return (int64_t)(h % (uint64_t)t->table_size);
}

/** Check if two node keys are equal. */
static inline int node_key_eq(const BPSNodeKey *a, const BPSNodeKey *b) {
    return a->action_hash == b->action_hash &&
           a->player == b->player &&
           a->street == b->street;
}

/**
 * Find a node in the table. Returns slot index, or -1 if not found.
 */
static int64_t table_find(const BPSTable *t, uint64_t action_hash,
                           uint8_t player, uint8_t street) {
    int64_t slot = node_slot(t, action_hash, player, street);
    BPSNodeKey needle;
    needle.action_hash = action_hash;
    needle.player = player;
    needle.street = street;
    needle._pad = 0;

    for (int probe = 0; probe < BPS_PROBE_LIMIT; probe++) {
        int64_t idx = (slot + probe) % t->table_size;
        if (!t->occupied[idx]) return -1;
        if (node_key_eq(&t->keys[idx], &needle)) return idx;
    }
    return -1;
}

/**
 * Find or create a node. Returns slot index, or -1 if table is full.
 */
static int64_t table_find_or_create(BPSTable *t, uint64_t action_hash,
                                     uint8_t player, uint8_t street) {
    int64_t slot = node_slot(t, action_hash, player, street);
    BPSNodeKey needle;
    needle.action_hash = action_hash;
    needle.player = player;
    needle.street = street;
    needle._pad = 0;

    for (int probe = 0; probe < BPS_PROBE_LIMIT; probe++) {
        int64_t idx = (slot + probe) % t->table_size;
        if (!t->occupied[idx]) {
            /* Empty slot — claim it */
            t->occupied[idx] = 1;
            t->keys[idx] = needle;
            memset(&t->meta[idx], 0, sizeof(BPSNodeMeta));
            t->meta[idx].strat_offset = SIZE_MAX;
            t->meta[idx].ev_offset = SIZE_MAX;
            t->num_nodes++;
            return idx;
        }
        if (node_key_eq(&t->keys[idx], &needle)) {
            return idx;
        }
    }
    return -1;  /* table full */
}

/** Next power of 2 >= n. */
static int64_t next_pow2(int64_t n) {
    int64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* ── Helpers for reading packed binary data ──────────────────────────── */

static inline uint8_t read_u8(const unsigned char *p) { return *p; }

static inline uint16_t read_u16(const unsigned char *p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

static inline uint32_t read_u32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline uint64_t read_u64(const unsigned char *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline float read_f32(const unsigned char *p) {
    float v;
    memcpy(&v, p, 4);
    return v;
}

/* ── Count-pass helper for two-pass build ────────────────────────────── */

/* Compact temp slot used only during the count pass: composite hash +
 * na (so we can size strat_data exactly without re-parsing entries). */
typedef struct {
    uint64_t key;       /* hash_combine(action_hash, hash(player, street)) */
    uint8_t  na;        /* num_actions for this node (constant across buckets) */
    uint8_t  occupied;
    uint16_t _pad;
} BPSCountSlot;

/* Walk the BPS3 stream once to compute exact unique-node count and
 * exact total strat_data byte requirement. The temp set is freed before
 * the real allocations, so this only adds transient memory equal to the
 * temp set itself (~16B per slot at the same load factor as the real
 * table). The win is twofold: (a) exact sizing of the real hash table
 * eliminates ~50% over-allocation slack, (b) exact strat_data sizing
 * avoids the 2× realloc spike that previously peaked transient memory
 * at ~old+new during memcpy.
 *
 * Returns 0 on success, -1 on parse error. *out_nodes and
 * *out_strat_bytes are populated on success. */
static int bps_count_strategy_nodes(
    const unsigned char *data, size_t len,
    int postflop_buckets, int preflop_buckets,
    int64_t est_nodes,
    int64_t *out_nodes, size_t *out_strat_bytes
) {
    const unsigned char *p = data;
    if (memcmp(p, "BPS3", 4) != 0) return -1;
    p += 4;
    uint32_t num_entries = read_u32(p); p += 4;
    /* uint32_t num_players = */ read_u32(p); p += 4;

    int64_t temp_size = next_pow2((int64_t)(est_nodes / 0.60) + 1);
    if (temp_size < 1024) temp_size = 1024;
    BPSCountSlot *temp = (BPSCountSlot *)calloc(
        (size_t)temp_size, sizeof(BPSCountSlot));
    if (!temp) {
        fprintf(stderr,
            "[bps_reader] OOM allocating count-pass temp (%lld slots)\n",
            (long long)temp_size);
        return -1;
    }

    int64_t unique_nodes = 0;
    size_t total_strat_bytes = 0;
    const unsigned char *end = data + len;

    for (uint32_t e = 0; e < num_entries; e++) {
        if (p + 21 > end) break;
        uint8_t player = read_u8(p); p += 1;
        uint8_t street = read_u8(p); p += 1;
        uint16_t bucket = read_u16(p); p += 2;
        /* uint64_t board_hash = */ read_u64(p); p += 8;
        uint64_t action_hash = read_u64(p); p += 8;
        uint8_t na = read_u8(p); p += 1;

        if (na > BPS_MAX_ACTIONS) na = BPS_MAX_ACTIONS;
        if (p + na > end) break;

        int nb = (street == 0) ? preflop_buckets : postflop_buckets;
        if (bucket >= nb) {
            p += na;
            continue;
        }

        /* Find or create in temp set. Composite key matches node_slot's
         * derivation so collisions in the temp pass mirror the final
         * table's collisions, keeping the count parity-clean. */
        uint64_t composite = bp_mix64(
            action_hash ^ bp_mix64((uint64_t)player * 31 + street));
        int64_t slot = (int64_t)(composite % (uint64_t)temp_size);
        int found = 0;
        for (int probe = 0; probe < BPS_PROBE_LIMIT; probe++) {
            int64_t idx = (slot + probe) % temp_size;
            if (!temp[idx].occupied) {
                temp[idx].occupied = 1;
                temp[idx].key = composite;
                temp[idx].na = na;
                total_strat_bytes += (size_t)nb * na;
                unique_nodes++;
                found = 1;
                break;
            }
            if (temp[idx].key == composite) {
                /* Already counted */
                found = 1;
                break;
            }
        }
        if (!found) {
            /* Probe limit hit — temp set was undersized. The real
             * table will be no better; bail and let single-pass
             * fallback handle it (caller frees temp on error path). */
            free(temp);
            fprintf(stderr,
                "[bps_reader] count pass: probe limit at entry %u "
                "(temp_size=%lld); falling back\n",
                e, (long long)temp_size);
            return -1;
        }

        p += na;
    }

    free(temp);
    *out_nodes = unique_nodes;
    *out_strat_bytes = total_strat_bytes;
    return 0;
}

/* ── Strategy loader ─────────────────────────────────────────────────── */

int bps_load_strategies(BPSReader *r,
                         const unsigned char *data, size_t len,
                         int use_legacy_mixer,
                         int postflop_buckets, int preflop_buckets) {
    if (!r || !data || len < 12) return -1;

    /* Store config */
    r->use_legacy_mixer = use_legacy_mixer;
    r->num_postflop_buckets = postflop_buckets;
    r->num_preflop_buckets = preflop_buckets;
    r->strategies_loaded = 0;
    r->evs_loaded = 0;

    /* Parse BPS3 inner header */
    const unsigned char *p = data;
    if (memcmp(p, "BPS3", 4) != 0) {
        fprintf(stderr, "[bps_reader] Expected BPS3 magic, got %.4s\n", (const char *)p);
        return -1;
    }
    p += 4;
    uint32_t num_entries = read_u32(p); p += 4;
    /* uint32_t num_players = */ read_u32(p); p += 4;

    printf("[bps_reader] BPS3: %u entries\n", num_entries);
    fflush(stdout);

    /* Conservative upper-bound estimate for the count pass; matches the
     * old single-pass divisor so we never undersize the count temp set. */
    int64_t est_nodes = (int64_t)num_entries / 8 + 10000;

    /* Two-pass build: count exact unique nodes and exact total strat
     * bytes, then size the real allocations exactly. Falls back to the
     * single-pass behavior on count-pass failure (mostly probe limit;
     * shouldn't fire with conservative est_nodes). */
    int64_t actual_nodes = 0;
    size_t exact_strat_bytes = 0;
    int two_pass_ok = (bps_count_strategy_nodes(
        data, len, postflop_buckets, preflop_buckets,
        est_nodes, &actual_nodes, &exact_strat_bytes) == 0);

    int64_t table_size;
    size_t strat_cap;
    if (two_pass_ok) {
        /* Size the real table for the EXACT number of unique nodes,
         * still keeping the 0.60 load factor for hash-collision
         * headroom. Saves ~50–60% vs the upper-bound estimate. */
        table_size = next_pow2((int64_t)(actual_nodes / 0.60) + 1);
        strat_cap = exact_strat_bytes;  /* exact — no realloc */
        printf("[bps_reader] Count pass: %lld unique nodes, "
               "%zu strat bytes (table_size=%lld, was %lld)\n",
               (long long)actual_nodes, exact_strat_bytes,
               (long long)table_size,
               (long long)next_pow2((int64_t)(est_nodes / 0.60) + 1));
        fflush(stdout);
    } else {
        /* Fallback: original conservative sizing + 4 bytes/entry strat
         * estimate that grows via realloc. */
        table_size = next_pow2((int64_t)(est_nodes / 0.60) + 1);
        strat_cap = (size_t)num_entries * 4;
    }
    if (table_size < 1024) table_size = 1024;
    if (strat_cap < 4096) strat_cap = 4096;

    BPSTable *t = &r->table;
    t->table_size = table_size;
    t->num_nodes = 0;
    t->num_ev_nodes = 0;

    /* Allocate table arrays */
    t->keys = (BPSNodeKey *)calloc((size_t)table_size, sizeof(BPSNodeKey));
    t->meta = (BPSNodeMeta *)calloc((size_t)table_size, sizeof(BPSNodeMeta));
    t->occupied = (uint8_t *)calloc((size_t)table_size, 1);
    if (!t->keys || !t->meta || !t->occupied) {
        fprintf(stderr, "[bps_reader] OOM allocating table (%lld slots)\n",
                (long long)table_size);
        bps_reader_free(r);
        return -1;
    }

    t->strat_data = (uint8_t *)malloc(strat_cap);
    if (!t->strat_data) {
        fprintf(stderr, "[bps_reader] OOM allocating strat_data (%zu bytes)\n", strat_cap);
        bps_reader_free(r);
        return -1;
    }
    t->strat_data_cap = strat_cap;
    t->strat_data_used = 0;

    /* No EV data yet */
    t->ev_data = NULL;
    t->ev_data_cap = 0;
    t->ev_data_used = 0;

    /* Single-pass: read each entry, find-or-create node, copy strategy bytes */
    int64_t loaded = 0;
    int64_t nodes_created = 0;
    const unsigned char *end = data + len;

    for (uint32_t e = 0; e < num_entries; e++) {
        /* Bounds check: minimum entry is 21 bytes (1+1+2+8+8+1+0_actions) */
        if (p + 21 > end) {
            fprintf(stderr, "[bps_reader] Truncated data at entry %u\n", e);
            break;
        }

        uint8_t player = read_u8(p); p += 1;
        uint8_t street = read_u8(p); p += 1;
        uint16_t bucket = read_u16(p); p += 2;
        /* uint64_t board_hash = */ read_u64(p); p += 8;
        uint64_t action_hash = read_u64(p); p += 8;
        uint8_t na = read_u8(p); p += 1;

        if (na > BPS_MAX_ACTIONS) na = BPS_MAX_ACTIONS;
        if (p + na > end) {
            fprintf(stderr, "[bps_reader] Truncated strategy bytes at entry %u\n", e);
            break;
        }

        /* Determine bucket count for this street */
        int nb = (street == 0) ? preflop_buckets : postflop_buckets;
        if (bucket >= nb) {
            /* Out-of-range bucket — skip entry */
            p += na;
            continue;
        }

        /* Find or create the node */
        int64_t idx = table_find_or_create(t, action_hash, player, street);
        if (idx < 0) {
            /* Table full — need a larger estimate. This shouldn't happen with
             * our conservative sizing, but handle gracefully. */
            fprintf(stderr, "[bps_reader] Table full at entry %u "
                    "(table_size=%lld, nodes=%lld)\n",
                    e, (long long)t->table_size, (long long)t->num_nodes);
            p += na;
            continue;
        }

        BPSNodeMeta *m = &t->meta[idx];

        /* First time seeing this node: allocate strategy block */
        if (m->strat_offset == SIZE_MAX) {
            m->num_actions = na;
            m->num_buckets = (uint16_t)nb;

            /* Ensure capacity */
            size_t block_size = (size_t)nb * na;
            int oom = 0;
            while (t->strat_data_used + block_size > t->strat_data_cap) {
                size_t new_cap = t->strat_data_cap * 2;
                uint8_t *new_buf = (uint8_t *)realloc(t->strat_data, new_cap);
                if (!new_buf) {
                    fprintf(stderr, "[bps_reader] OOM growing strat_data to %zu\n", new_cap);
                    oom = 1;
                    break;
                }
                t->strat_data = new_buf;
                t->strat_data_cap = new_cap;
            }
            if (oom) {
                p += na;
                continue;
            }

            m->strat_offset = t->strat_data_used;
            t->strat_data_used += block_size;

            /* Initialize with uniform strategy: 255/na per action, truncated.
             * This ensures unvisited buckets have a valid uniform fallback. */
            uint8_t *block = t->strat_data + m->strat_offset;
            uint8_t uniform_val = (uint8_t)(255 / na);
            memset(block, uniform_val, block_size);

            nodes_created++;
        }

        /* Copy this entry's strategy bytes into the node's block at [bucket][0..na-1] */
        if (na == m->num_actions && bucket < m->num_buckets) {
            uint8_t *dst = t->strat_data + m->strat_offset + (size_t)bucket * na;
            memcpy(dst, p, na);
        }

        p += na;
        loaded++;
    }

    printf("[bps_reader] Loaded %lld entries into %lld nodes "
           "(table %lld slots, %.1f%% load)\n",
           (long long)loaded, (long long)nodes_created,
           (long long)t->table_size,
           100.0 * nodes_created / t->table_size);
    fflush(stdout);

    /* Parity guard: the count pass must agree with the populate pass
     * on both the unique-node count and the strat_data byte total.
     * If it disagrees, the count-pass logic has drifted from the
     * populate logic — fail loud, don't degrade silently. */
    if (two_pass_ok) {
        if ((int64_t)nodes_created != actual_nodes ||
            t->strat_data_used != exact_strat_bytes) {
            fprintf(stderr,
                "[bps_reader] count/populate parity drift: "
                "nodes=%lld vs %lld, strat=%zu vs %zu\n",
                (long long)nodes_created, (long long)actual_nodes,
                t->strat_data_used, exact_strat_bytes);
            bps_reader_free(r);
            return -1;
        }
    }

    r->strategies_loaded = 1;
    return 0;
}

/* ── EV loader ───────────────────────────────────────────────────────── */

/* Count-pass for EV blocks: walk BPR3 entries, dedupe unique
 * (action_hash, player, street) triples, accumulate exact ev_data
 * float count using each node's (nb, na). Mirrors
 * bps_count_strategy_nodes; same trade-off (transient temp set,
 * exact final allocation). */
static int bps_count_ev_floats(
    const unsigned char *data, size_t len,
    int postflop_buckets, int preflop_buckets,
    int64_t est_nodes,
    size_t *out_ev_floats
) {
    const unsigned char *p = data;
    if (memcmp(p, "BPR3", 4) != 0) return -1;
    p += 4;
    uint32_t num_entries = read_u32(p); p += 4;

    int64_t temp_size = next_pow2((int64_t)(est_nodes / 0.60) + 1);
    if (temp_size < 1024) temp_size = 1024;
    BPSCountSlot *temp = (BPSCountSlot *)calloc(
        (size_t)temp_size, sizeof(BPSCountSlot));
    if (!temp) return -1;

    size_t total_floats = 0;
    const unsigned char *end = data + len;

    for (uint32_t e = 0; e < num_entries; e++) {
        if (p + 13 > end) break;
        uint8_t player = read_u8(p); p += 1;
        uint8_t street = read_u8(p); p += 1;
        uint16_t bucket = read_u16(p); p += 2;
        uint64_t action_hash = read_u64(p); p += 8;
        uint8_t na = read_u8(p); p += 1;

        if (na > BPS_MAX_ACTIONS) na = BPS_MAX_ACTIONS;
        if (p + (size_t)na * 4 > end) break;

        int nb = (street == 0) ? preflop_buckets : postflop_buckets;
        if (bucket >= nb) {
            p += (size_t)na * 4;
            continue;
        }

        uint64_t composite = bp_mix64(
            action_hash ^ bp_mix64((uint64_t)player * 31 + street));
        int64_t slot = (int64_t)(composite % (uint64_t)temp_size);
        int found = 0;
        for (int probe = 0; probe < BPS_PROBE_LIMIT; probe++) {
            int64_t idx = (slot + probe) % temp_size;
            if (!temp[idx].occupied) {
                temp[idx].occupied = 1;
                temp[idx].key = composite;
                temp[idx].na = na;
                total_floats += (size_t)nb * na;
                found = 1;
                break;
            }
            if (temp[idx].key == composite) {
                found = 1;
                break;
            }
        }
        if (!found) {
            free(temp);
            return -1;
        }

        p += (size_t)na * 4;
    }

    free(temp);
    *out_ev_floats = total_floats;
    return 0;
}

int bps_load_evs(BPSReader *r,
                   const unsigned char *data, size_t len) {
    if (!r || !data || len < 8) return -1;
    if (!r->strategies_loaded) {
        fprintf(stderr, "[bps_reader] Must call bps_load_strategies before bps_load_evs\n");
        return -1;
    }

    BPSTable *t = &r->table;

    /* Parse BPR3 inner header */
    const unsigned char *p = data;
    if (memcmp(p, "BPR3", 4) != 0) {
        fprintf(stderr, "[bps_reader] Expected BPR3 magic, got %.4s\n", (const char *)p);
        return -1;
    }
    p += 4;
    uint32_t num_entries = read_u32(p); p += 4;

    printf("[bps_reader] BPR3: %u entries\n", num_entries);
    fflush(stdout);

    /* Try to size ev_data exactly via a count pass (avoids realloc
     * spikes during populate). Falls back to the conservative
     * num_entries * 4 estimate that grows via 2× realloc. */
    int64_t est_nodes = (int64_t)num_entries / 8 + 10000;
    size_t exact_floats = 0;
    int two_pass_ok = (bps_count_ev_floats(
        data, len, r->num_postflop_buckets, r->num_preflop_buckets,
        est_nodes, &exact_floats) == 0);

    size_t ev_cap;
    if (two_pass_ok) {
        ev_cap = exact_floats;
        printf("[bps_reader] EV count pass: %zu floats (exact)\n",
               exact_floats);
        fflush(stdout);
    } else {
        ev_cap = (size_t)num_entries * 4;
    }
    if (ev_cap < 4096) ev_cap = 4096;
    t->ev_data = (float *)calloc(ev_cap, sizeof(float));
    if (!t->ev_data) {
        fprintf(stderr, "[bps_reader] OOM allocating ev_data (%zu floats)\n", ev_cap);
        return -1;
    }
    t->ev_data_cap = ev_cap;
    t->ev_data_used = 0;

    /* Single-pass: read entries, group by node, store float blocks */
    int64_t loaded = 0;
    int64_t ev_nodes_created = 0;
    const unsigned char *end = data + len;

    for (uint32_t e = 0; e < num_entries; e++) {
        /* Min entry: player(1) + street(1) + bucket(2) + action_hash(8) + na(1) = 13 */
        if (p + 13 > end) {
            fprintf(stderr, "[bps_reader] Truncated EV data at entry %u\n", e);
            break;
        }

        uint8_t player = read_u8(p); p += 1;
        uint8_t street = read_u8(p); p += 1;
        uint16_t bucket = read_u16(p); p += 2;
        uint64_t action_hash = read_u64(p); p += 8;
        uint8_t na = read_u8(p); p += 1;

        if (na > BPS_MAX_ACTIONS) na = BPS_MAX_ACTIONS;
        if (p + (size_t)na * 4 > end) {
            fprintf(stderr, "[bps_reader] Truncated EV floats at entry %u\n", e);
            break;
        }

        int nb = (street == 0) ? r->num_preflop_buckets : r->num_postflop_buckets;
        if (bucket >= nb) {
            p += (size_t)na * 4;
            continue;
        }

        /* Find the node (should exist from strategy loading).
         * If not, create it — EV-only nodes are valid. */
        int64_t idx = table_find_or_create(t, action_hash, player, street);
        if (idx < 0) {
            p += (size_t)na * 4;
            continue;
        }

        BPSNodeMeta *m = &t->meta[idx];

        /* First time seeing EVs for this node: allocate EV block */
        if (m->ev_offset == SIZE_MAX) {
            if (m->num_actions == 0) m->num_actions = na;
            if (m->num_buckets == 0) m->num_buckets = (uint16_t)nb;

            size_t block_floats = (size_t)m->num_buckets * m->num_actions;

            /* Ensure capacity */
            int oom = 0;
            while (t->ev_data_used + block_floats > t->ev_data_cap) {
                size_t new_cap = t->ev_data_cap * 2;
                float *new_buf = (float *)realloc(t->ev_data, new_cap * sizeof(float));
                if (!new_buf) {
                    fprintf(stderr, "[bps_reader] OOM growing ev_data to %zu\n", new_cap);
                    oom = 1;
                    break;
                }
                /* Zero the newly allocated region */
                memset(new_buf + t->ev_data_cap, 0,
                       (new_cap - t->ev_data_cap) * sizeof(float));
                t->ev_data = new_buf;
                t->ev_data_cap = new_cap;
            }
            if (oom) {
                p += (size_t)na * 4;
                continue;
            }

            m->ev_offset = t->ev_data_used;
            /* Zero the block (default EV = 0 for unvisited buckets) */
            memset(t->ev_data + m->ev_offset, 0, block_floats * sizeof(float));
            t->ev_data_used += block_floats;
            ev_nodes_created++;
        }

        /* Copy this entry's EV floats into the node's block at [bucket][0..na-1] */
        if (na == m->num_actions && bucket < m->num_buckets) {
            float *dst = t->ev_data + m->ev_offset + (size_t)bucket * na;
            for (int a = 0; a < na; a++) {
                dst[a] = read_f32(p + (size_t)a * 4);
            }
        }

        p += (size_t)na * 4;
        loaded++;
    }

    t->num_ev_nodes = ev_nodes_created;

    printf("[bps_reader] Loaded %lld EV entries into %lld nodes\n",
           (long long)loaded, (long long)ev_nodes_created);
    fflush(stdout);

    /* Parity guard: count-pass must agree with populate-pass on the
     * total ev_data float count. Drift indicates the count and
     * populate logic have diverged — fail loud. */
    if (two_pass_ok && t->ev_data_used != exact_floats) {
        fprintf(stderr,
            "[bps_reader] EV count/populate parity drift: "
            "floats=%zu vs %zu\n",
            t->ev_data_used, exact_floats);
        return -1;
    }

    r->evs_loaded = 1;
    return 0;
}

/* ── Query functions ─────────────────────────────────────────────────── */

int bps_query_all_strategies(const BPSReader *r,
                              const int *action_history, int action_len,
                              int player, int street,
                              float *out, int out_size,
                              int *num_buckets_out) {
    if (!r || !r->strategies_loaded || !out) return 0;

    uint64_t ah = compute_action_hash(action_history, action_len,
                                       r->use_legacy_mixer);
    int64_t idx = table_find(&r->table, ah, (uint8_t)player, (uint8_t)street);
    if (idx < 0) return 0;

    const BPSNodeMeta *m = &r->table.meta[idx];
    if (m->strat_offset == SIZE_MAX) return 0;

    int na = m->num_actions;
    int nb = m->num_buckets;
    if (num_buckets_out) *num_buckets_out = nb;

    /* Check output buffer capacity */
    if (out_size < nb * na) return 0;

    /* Convert uint8 strategies to normalized float32 */
    const uint8_t *src = r->table.strat_data + m->strat_offset;

    for (int b = 0; b < nb; b++) {
        const uint8_t *row = src + (size_t)b * na;
        float *dst = out + (size_t)b * na;

        /* Sum for normalization */
        float sum = 0.0f;
        for (int a = 0; a < na; a++) {
            float v = (float)row[a];
            dst[a] = v;
            sum += v;
        }
        /* Normalize to sum=1 */
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int a = 0; a < na; a++) dst[a] *= inv;
        } else {
            float uniform = 1.0f / na;
            for (int a = 0; a < na; a++) dst[a] = uniform;
        }
    }

    return na;
}

int bps_query_all_evs(const BPSReader *r,
                       const int *action_history, int action_len,
                       int player, int street,
                       float *out, int out_size,
                       int *num_buckets_out) {
    if (!r || !r->evs_loaded || !out) return 0;

    uint64_t ah = compute_action_hash(action_history, action_len,
                                       r->use_legacy_mixer);
    int64_t idx = table_find(&r->table, ah, (uint8_t)player, (uint8_t)street);
    if (idx < 0) return 0;

    const BPSNodeMeta *m = &r->table.meta[idx];
    if (m->ev_offset == SIZE_MAX) return 0;

    int na = m->num_actions;
    int nb = m->num_buckets;
    if (num_buckets_out) *num_buckets_out = nb;

    if (out_size < nb * na) return 0;

    /* Direct copy — EVs are already float32 */
    const float *src = r->table.ev_data + m->ev_offset;
    memcpy(out, src, (size_t)nb * na * sizeof(float));

    return na;
}

int bps_query_all_visit_counts(const BPSReader *r,
                                 const int *action_history, int action_len,
                                 int player, int street,
                                 uint32_t *out, int out_size,
                                 int *num_buckets_out) {
    if (!r || !r->evs_loaded || !out) return 0;
    if (!r->table.vc_data) return 0;

    uint64_t ah = compute_action_hash(action_history, action_len,
                                       r->use_legacy_mixer);
    int64_t idx = table_find(&r->table, ah, (uint8_t)player, (uint8_t)street);
    if (idx < 0) return 0;

    const BPSNodeMeta *m = &r->table.meta[idx];
    if (m->ev_offset == SIZE_MAX) return 0;

    int nb = m->num_buckets;
    if (num_buckets_out) *num_buckets_out = nb;
    if (out_size < nb) return 0;

    const uint32_t *src = r->table.vc_data + m->vc_offset;
    memcpy(out, src, (size_t)nb * sizeof(uint32_t));

    return nb;
}

int bps_query_strategy(const BPSReader *r,
                        const int *action_history, int action_len,
                        int player, int street, int bucket,
                        float *strategy_out) {
    if (!r || !r->strategies_loaded || !strategy_out) return 0;

    uint64_t ah = compute_action_hash(action_history, action_len,
                                       r->use_legacy_mixer);
    int64_t idx = table_find(&r->table, ah, (uint8_t)player, (uint8_t)street);
    if (idx < 0) return 0;

    const BPSNodeMeta *m = &r->table.meta[idx];
    if (m->strat_offset == SIZE_MAX) return 0;
    if (bucket < 0 || bucket >= m->num_buckets) return 0;

    int na = m->num_actions;
    const uint8_t *row = r->table.strat_data + m->strat_offset + (size_t)bucket * na;

    /* Convert and normalize */
    float sum = 0.0f;
    for (int a = 0; a < na; a++) {
        float v = (float)row[a];
        strategy_out[a] = v;
        sum += v;
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int a = 0; a < na; a++) strategy_out[a] *= inv;
    } else {
        float uniform = 1.0f / na;
        for (int a = 0; a < na; a++) strategy_out[a] = uniform;
    }

    return na;
}

/* ── Board canonicalization (copied from mccfr_blueprint.c:389-470) ── */

/**
 * Canonicalize a board for info set hashing.
 * Maps actual board cards to suit-isomorphic canonical cards.
 * Must match mccfr_blueprint.c:canonicalize_board exactly.
 */
static void canonicalize_board(const int *board, int num_board, int *canon_out) {
    if (num_board == 0) return;

    int sorted[3], canon_flop[3];
    int suit_map[4] = {-1, -1, -1, -1};

    for (int i = 0; i < 3 && i < num_board; i++) sorted[i] = board[i];
    for (int a = 0; a < 2; a++)
        for (int b = a+1; b < 3; b++)
            if ((sorted[a]>>2) < (sorted[b]>>2)) {
                int tmp = sorted[a]; sorted[a] = sorted[b]; sorted[b] = tmp;
            }

    int r0 = sorted[0]>>2, r1 = sorted[1]>>2, r2 = sorted[2]>>2;
    int s0 = sorted[0]&3, s1 = sorted[1]&3, s2 = sorted[2]&3;

    if (r0 == r1 && r1 == r2) {
        canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+1;
    } else if (r0 == r1 || r1 == r2) {
        if (r0 == r1) {
            if (s2 == s0 || s2 == s1) {
                canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+3;
            } else {
                canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+1;
            }
        } else {
            if (s0 == s1 || s0 == s2) {
                canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+3; canon_flop[2] = r2*4+2;
            } else {
                canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+1;
            }
        }
    } else {
        if (s0 == s1 && s1 == s2) {
            canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+3; canon_flop[2] = r2*4+3;
        } else if (s0 == s1) {
            canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+3; canon_flop[2] = r2*4+2;
        } else if (s0 == s2) {
            canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+3;
        } else if (s1 == s2) {
            canon_flop[0] = r0*4+2; canon_flop[1] = r1*4+3; canon_flop[2] = r2*4+3;
        } else {
            canon_flop[0] = r0*4+3; canon_flop[1] = r1*4+2; canon_flop[2] = r2*4+1;
        }
    }

    for (int i = 0; i < 3; i++) {
        int as = sorted[i] & 3;
        int cs = canon_flop[i] & 3;
        if (suit_map[as] == -1) suit_map[as] = cs;
    }
    int nc = 0;
    for (int i = 0; i < 4; i++) {
        if (suit_map[i] == -1) {
            while (nc < 4) {
                int used = 0;
                for (int j = 0; j < 4; j++)
                    if (suit_map[j] == nc) { used = 1; break; }
                if (!used) break;
                nc++;
            }
            if (nc < 4) suit_map[i] = nc++;
            else suit_map[i] = 0;
        }
    }

    for (int i = 0; i < 3 && i < num_board; i++)
        canon_out[i] = canon_flop[i];
    for (int i = 3; i < num_board; i++) {
        int rank = board[i] >> 2;
        int suit = board[i] & 3;
        canon_out[i] = rank * 4 + suit_map[suit];
    }
}

/**
 * Compute board hash from canonical board cards.
 * Same as mccfr_blueprint.c:310-323, with mixer parameter.
 */
uint64_t compute_board_hash(const int *board, int num_board,
                                    int use_legacy_mixer) {
    int canon[5];
    if (num_board >= 3)
        canonicalize_board(board, num_board, canon);
    else
        for (int i = 0; i < num_board; i++) canon[i] = board[i];

    uint64_t h = 0x123456789ABCDEFULL;
    if (use_legacy_mixer) {
        for (int i = 0; i < num_board; i++)
            h = hash_combine_boost_legacy(h, (uint64_t)canon[i] * 31 + 7);
    } else {
        for (int i = 0; i < num_board; i++)
            h = hash_combine(h, (uint64_t)canon[i] * 31 + 7);
    }
    return h;
}

/* ── Texture cache ───────────────────────────────────────────────────── */

int tex_index_lookup(const BPSTextureCache *tc, uint64_t flop_hash) {
    uint64_t mixed = bp_mix64(flop_hash);
    int idx = (int)(mixed & (uint64_t)(BPS_TEXTURE_INDEX_SIZE - 1));
    for (int p = 0; p < 32; p++) {
        int slot = (idx + p) & (BPS_TEXTURE_INDEX_SIZE - 1);
        int t = tc->hash_index[slot];
        if (t < 0) return -1;
        if (tc->hash_keys[t] == flop_hash) return t;
    }
    return -1;
}

static inline void tex_index_insert(BPSTextureCache *tc, uint64_t flop_hash, int tex_idx) {
    uint64_t mixed = bp_mix64(flop_hash);
    int idx = (int)(mixed & (uint64_t)(BPS_TEXTURE_INDEX_SIZE - 1));
    for (int p = 0; p < 32; p++) {
        int slot = (idx + p) & (BPS_TEXTURE_INDEX_SIZE - 1);
        if (tc->hash_index[slot] < 0) {
            tc->hash_index[slot] = tex_idx;
            return;
        }
    }
}

void tex_index_rebuild(BPSTextureCache *tc) {
    for (int i = 0; i < BPS_TEXTURE_INDEX_SIZE; i++)
        tc->hash_index[i] = -1;
    for (int t = 0; t < tc->num_textures; t++)
        tex_index_insert(tc, tc->hash_keys[t], t);
}

/**
 * Compute the hand index for (c0, c1) in the ca_generate_hands ordering,
 * excluding 3 board cards. This reproduces the exact iteration order from
 * card_abstraction.c:ca_generate_hands (nested c0<c1 loops, skip blocked).
 */
/* Phase R Block B.3 (Step 6) FIX: align with trainer's storage formula.
 *
 * Trainer storage (mccfr_blueprint.c:3002) writes to
 *   cache_row[full_idx]
 * where `full_idx = c0*51 - c0*(c0-1)/2 + c1-c0-1` is the full-deck
 * pair index (ignoring board-blocking).
 *
 * Trainer runtime read (mccfr_blueprint.c:1619) indexes
 *   flop_cache->buckets[ts->sampled_hands[ap]]
 * where `sampled_hands[ap]` is an index into `s->hands[ap]`, which
 * was populated by the c0<c1 all-1326-pair enumeration in
 * bp_init_unified at lines 2690-2696. So sampled_hands[ap] IS the
 * full-deck pair index — same index space as storage.
 *
 * Pre-fix: this function used a board-FILTERED counting loop, yielding
 * a different index space than the trainer's storage. Empirically
 * verified divergent on 98.28% of 2.06M covered (board, hand) queries
 * via `tests/test_b3_bucket_alignment.c`. RESOLVED Step 6 with
 * the formula below — same as trainer's storage formula — restoring
 * byte-equal lookups across all covered textures.
 */
static int hand_index_for(const int *board, int c0, int c1) {
    int blocked[52] = {0};
    for (int i = 0; i < 3; i++) blocked[board[i]] = 1;

    /* Hero's cards must not be blocked */
    if (blocked[c0] || blocked[c1]) return -1;

    /* Ensure c0 < c1 */
    if (c0 > c1) { int tmp = c0; c0 = c1; c1 = tmp; }

    return c0 * 51 - c0 * (c0 - 1) / 2 + c1 - c0 - 1;
}

int bps_load_texture_cache(BPSReader *r, const char *path,
                            int cache_uses_legacy_mixer) {
    if (!r || !path) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[bps_reader] Cannot open texture cache: %s\n", path);
        return -1;
    }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TXC1", 4) != 0) {
        fprintf(stderr, "[bps_reader] Invalid texture cache magic\n");
        fclose(f);
        return -1;
    }

    int num_tex, num_buckets;
    if (fread(&num_tex, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
    if (fread(&num_buckets, sizeof(int), 1, f) != 1) { fclose(f); return -1; }

    if (num_tex > BPS_MAX_TEXTURES || num_tex <= 0) {
        fprintf(stderr, "[bps_reader] Invalid texture count: %d\n", num_tex);
        fclose(f);
        return -1;
    }

    BPSTextureCache *tc = &r->textures;
    tc->num_textures = num_tex;
    tc->num_buckets = num_buckets;

    /* Read hash keys */
    if (fread(tc->hash_keys, sizeof(uint64_t), (size_t)num_tex, f) != (size_t)num_tex) {
        fclose(f);
        return -1;
    }

    /* Allocate and read bucket cache */
    size_t cache_size = (size_t)num_tex * BPS_MAX_HANDS;
    tc->bucket_cache = (int *)malloc(cache_size * sizeof(int));
    if (!tc->bucket_cache) {
        fprintf(stderr, "[bps_reader] OOM allocating texture bucket cache\n");
        fclose(f);
        return -1;
    }
    if (fread(tc->bucket_cache, sizeof(int), cache_size, f) != cache_size) {
        free(tc->bucket_cache);
        tc->bucket_cache = NULL;
        fclose(f);
        return -1;
    }

    fclose(f);

    /* Build the hashmap index */
    tex_index_rebuild(tc);
    tc->loaded = 1;

    /* Store the mixer mode used to build the cache keys. When computing
     * a board hash at query time (bps_hand_to_bucket), we must use the
     * SAME mixer that produced the stored hash keys, which may differ
     * from the .bps file's mixer. */
    tc->use_legacy_mixer = cache_uses_legacy_mixer;

    printf("[bps_reader] Loaded texture cache: %d textures, %d buckets from %s\n",
           num_tex, num_buckets, path);
    fflush(stdout);

    return num_tex;
}

/**
 * Derive the suit mapping from actual board → canonical board.
 *
 * canonicalize_board produces canonical cards. We compare the sorted
 * actual cards against the canonical output to extract suit_map[4]:
 *   suit_map[actual_suit] = canonical_suit
 *
 * Must be called with the SAME sorted order that canonicalize_board uses.
 */
static void derive_suit_map(const int *board, int num_board, int suit_map[4]) {
    int canon[5];
    canonicalize_board(board, num_board, canon);

    /* Sort actual board by rank descending (same order canonicalize_board uses) */
    int sorted[5];
    for (int i = 0; i < num_board; i++) sorted[i] = board[i];
    for (int a = 0; a < num_board - 1; a++)
        for (int b = a + 1; b < num_board; b++)
            if ((sorted[a] >> 2) < (sorted[b] >> 2)) {
                int tmp = sorted[a]; sorted[a] = sorted[b]; sorted[b] = tmp;
            }

    /* Build mapping from sorted actual → canonical suits */
    suit_map[0] = suit_map[1] = suit_map[2] = suit_map[3] = -1;
    for (int i = 0; i < num_board; i++) {
        int as = sorted[i] & 3;
        int cs = canon[i] & 3;
        if (suit_map[as] == -1) suit_map[as] = cs;
    }

    /* Fill unmapped suits with remaining canonical suits */
    int nc = 0;
    for (int i = 0; i < 4; i++) {
        if (suit_map[i] == -1) {
            while (nc < 4) {
                int used = 0;
                for (int j = 0; j < 4; j++)
                    if (suit_map[j] == nc) { used = 1; break; }
                if (!used) break;
                nc++;
            }
            if (nc < 4) suit_map[i] = nc++;
            else suit_map[i] = 0;
        }
    }
}

int bps_hand_to_bucket(const BPSReader *r,
                        const int *board, int c0, int c1) {
    if (!r || !r->textures.loaded || !r->textures.bucket_cache) return -1;
    if (!board) return -1;

    /* Compute canonical board hash using the mixer that produced the
     * texture cache hash keys (may differ from the .bps file's mixer). */
    uint64_t flop_hash = compute_board_hash(board, 3, r->textures.use_legacy_mixer);

    int tex_idx = tex_index_lookup(&r->textures, flop_hash);
    if (tex_idx < 0) return -1;

    /* Get canonical board and derive suit mapping.
     * The texture cache stores bucket assignments indexed by canonical
     * hand ordering (ca_generate_hands on the CANONICAL board). We need
     * to map the hero's actual hand to canonical form, then find its
     * index in the canonical hand list. */
    int canon_board[3];
    canonicalize_board(board, 3, canon_board);

    int suit_map[4];
    derive_suit_map(board, 3, suit_map);

    /* Check hero's cards aren't blocked by the board */
    for (int i = 0; i < 3; i++) {
        if (board[i] == c0 || board[i] == c1) return -1;
    }

    /* Map hero's hand to canonical form */
    int cc0 = (c0 >> 2) * 4 + suit_map[c0 & 3];
    int cc1 = (c1 >> 2) * 4 + suit_map[c1 & 3];

    /* Find the canonical hand index (using canonical board for exclusion) */
    int hidx = hand_index_for(canon_board, cc0, cc1);
    if (hidx < 0 || hidx >= BPS_MAX_HANDS) return -1;

    return r->textures.bucket_cache[tex_idx * BPS_MAX_HANDS + hidx];
}

/* ── Cache serialization ─────────────────────────────────────────────── */

/*
 * Cache file format (BPSC):
 *
 * Header (64 bytes):
 *   magic:            "BPSC" (4B)
 *   version:          uint32 (4B) — 1 = strat+ev only, 2 = adds vc_data
 *   use_legacy_mixer: int32 (4B)
 *   postflop_buckets: int32 (4B)
 *   preflop_buckets:  int32 (4B)
 *   table_size:       int64 (8B)
 *   num_nodes:        int64 (8B)
 *   num_ev_nodes:     int64 (8B)
 *   strat_data_used:  uint64 (8B)
 *   ev_data_used:     uint64 (8B) — in floats
 *   _reserved:        4B
 *
 * Body (sequential arrays, full table_size for hash table arrays):
 *   keys[table_size]     — BPSNodeKey (12B each)
 *   meta[table_size]     — BPSNodeMeta (24B each)
 *   occupied[table_size] — uint8 (1B each)
 *   strat_data[strat_data_used] — uint8
 *   ev_data[ev_data_used]       — float32
 *   vc_data[vc_data_used]       — uint32_t (v2 only; size derived from
 *                                  meta walk: sum of m->num_buckets over
 *                                  visited nodes where ev_offset != SIZE_MAX)
 *
 * Writing the full table (including empty slots) makes reload a trivial
 * fread → the hash table is ready to use immediately with no re-hashing.
 * This wastes some disk (empty slots) but makes reload O(1) instead of
 * O(num_nodes) for re-insertion.
 *
 * v2 backwards-compat: meta[] is identical on disk between v1 and v2 — the
 * new vc_offset field reuses 4 bytes that were previously implicit padding.
 * v1 caches load fine under v2 code (vc_data section absent → vc_data
 * stays NULL, queries return 0). v2 caches loaded by v1 code would silently
 * ignore the trailing vc_data bytes (no crash; just no visit counts).
 */

#define BPSC_VERSION 2

int bps_save_cache(const BPSReader *r, const char *path) {
    if (!r || !r->strategies_loaded || !path) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[bps_reader] Cannot create cache file: %s\n", path);
        return -1;
    }

    const BPSTable *t = &r->table;

    /* Header */
    fwrite("BPSC", 1, 4, f);
    uint32_t ver = BPSC_VERSION;
    fwrite(&ver, 4, 1, f);
    int32_t mixer = (int32_t)r->use_legacy_mixer;
    fwrite(&mixer, 4, 1, f);
    int32_t pb = (int32_t)r->num_postflop_buckets;
    fwrite(&pb, 4, 1, f);
    int32_t ppb = (int32_t)r->num_preflop_buckets;
    fwrite(&ppb, 4, 1, f);
    fwrite(&t->table_size, 8, 1, f);
    fwrite(&t->num_nodes, 8, 1, f);
    fwrite(&t->num_ev_nodes, 8, 1, f);
    uint64_t sdu = (uint64_t)t->strat_data_used;
    fwrite(&sdu, 8, 1, f);
    uint64_t edu = (uint64_t)t->ev_data_used;
    fwrite(&edu, 8, 1, f);
    uint32_t reserved = 0;
    fwrite(&reserved, 4, 1, f);
    /* Header = 4+4+4+4+4+8+8+8+8+8+4 = 64 bytes */

    /* Body */
    fwrite(t->keys, sizeof(BPSNodeKey), (size_t)t->table_size, f);
    fwrite(t->meta, sizeof(BPSNodeMeta), (size_t)t->table_size, f);
    fwrite(t->occupied, 1, (size_t)t->table_size, f);
    fwrite(t->strat_data, 1, t->strat_data_used, f);
    if (t->ev_data && t->ev_data_used > 0) {
        fwrite(t->ev_data, sizeof(float), t->ev_data_used, f);
    }
    /* v2 vc_data section (per-bucket visit counts, parallel to ev_data
     * but indexed per bucket not per action; size derivable from meta). */
    if (t->vc_data && t->vc_data_used > 0) {
        fwrite(t->vc_data, sizeof(uint32_t), t->vc_data_used, f);
    }

    fclose(f);

    /* Report size */
    size_t total = 64
        + (size_t)t->table_size * (sizeof(BPSNodeKey) + sizeof(BPSNodeMeta) + 1)
        + t->strat_data_used
        + t->ev_data_used * sizeof(float)
        + t->vc_data_used * sizeof(uint32_t);
    printf("[bps_reader] Saved cache: %zu MB (%lld nodes, %zu strat bytes, "
           "%zu ev floats, %zu vc counts) to %s\n",
           total / (1024*1024), (long long)t->num_nodes,
           t->strat_data_used, t->ev_data_used, t->vc_data_used, path);
    fflush(stdout);

    return 0;
}

int bps_load_cache(BPSReader *r, const char *path) {
    if (!r || !path) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Header */
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BPSC", 4) != 0) {
        fclose(f);
        return -1;
    }

    uint32_t ver;
    if (fread(&ver, 4, 1, f) != 1 || (ver != 1 && ver != 2)) {
        fprintf(stderr, "[bps_reader] Cache version mismatch: got %u, expected 1 or 2\n",
                ver);
        fclose(f);
        return -1;
    }

    int32_t mixer, pb, ppb;
    fread(&mixer, 4, 1, f);
    fread(&pb, 4, 1, f);
    fread(&ppb, 4, 1, f);

    int64_t table_size, num_nodes, num_ev_nodes;
    fread(&table_size, 8, 1, f);
    fread(&num_nodes, 8, 1, f);
    fread(&num_ev_nodes, 8, 1, f);

    uint64_t strat_data_used, ev_data_used;
    fread(&strat_data_used, 8, 1, f);
    fread(&ev_data_used, 8, 1, f);

    uint32_t reserved;
    fread(&reserved, 4, 1, f);

    /* Sanity checks */
    if (table_size <= 0 || table_size > (int64_t)1 << 32) {
        fprintf(stderr, "[bps_reader] Cache has invalid table_size: %lld\n",
                (long long)table_size);
        fclose(f);
        return -1;
    }

    printf("[bps_reader] Loading cache: %lld slots, %lld nodes, "
           "%llu strat bytes, %llu ev floats...\n",
           (long long)table_size, (long long)num_nodes,
           (unsigned long long)strat_data_used,
           (unsigned long long)ev_data_used);
    fflush(stdout);

    /* Store config */
    r->use_legacy_mixer = (int)mixer;
    r->num_postflop_buckets = (int)pb;
    r->num_preflop_buckets = (int)ppb;

    BPSTable *t = &r->table;
    t->table_size = table_size;
    t->num_nodes = num_nodes;
    t->num_ev_nodes = num_ev_nodes;
    t->strat_data_used = (size_t)strat_data_used;
    t->strat_data_cap = (size_t)strat_data_used;
    t->ev_data_used = (size_t)ev_data_used;
    t->ev_data_cap = (size_t)ev_data_used;

    /* Allocate and read arrays */
    t->keys = (BPSNodeKey *)malloc((size_t)table_size * sizeof(BPSNodeKey));
    t->meta = (BPSNodeMeta *)malloc((size_t)table_size * sizeof(BPSNodeMeta));
    t->occupied = (uint8_t *)malloc((size_t)table_size);
    t->strat_data = (uint8_t *)malloc((size_t)strat_data_used);

    if (!t->keys || !t->meta || !t->occupied || !t->strat_data) {
        fprintf(stderr, "[bps_reader] OOM allocating cache arrays\n");
        bps_reader_free(r);
        fclose(f);
        return -1;
    }

    if (fread(t->keys, sizeof(BPSNodeKey), (size_t)table_size, f) != (size_t)table_size ||
        fread(t->meta, sizeof(BPSNodeMeta), (size_t)table_size, f) != (size_t)table_size ||
        fread(t->occupied, 1, (size_t)table_size, f) != (size_t)table_size ||
        fread(t->strat_data, 1, (size_t)strat_data_used, f) != (size_t)strat_data_used) {
        fprintf(stderr, "[bps_reader] Truncated cache file\n");
        bps_reader_free(r);
        fclose(f);
        return -1;
    }

    r->strategies_loaded = 1;

    /* EV data (optional) */
    if (ev_data_used > 0) {
        t->ev_data = (float *)malloc((size_t)ev_data_used * sizeof(float));
        if (!t->ev_data) {
            fprintf(stderr, "[bps_reader] OOM allocating EV cache\n");
            /* strategies still loaded, just no EVs */
        } else if (fread(t->ev_data, sizeof(float), (size_t)ev_data_used, f) != (size_t)ev_data_used) {
            fprintf(stderr, "[bps_reader] Truncated EV cache\n");
            free(t->ev_data);
            t->ev_data = NULL;
            t->ev_data_used = 0;
            t->ev_data_cap = 0;
        } else {
            r->evs_loaded = 1;
        }
    }

    /* v2 vc_data section. Size derived from meta walk: sum num_buckets over
     * visited nodes (those with ev_offset != SIZE_MAX). For v1 caches we
     * skip this and leave vc_data NULL — queries will return 0. */
    if (ver >= 2 && r->evs_loaded) {
        size_t vc_data_used = 0;
        for (int64_t i = 0; i < t->table_size; i++) {
            if (!t->occupied[i]) continue;
            if (t->meta[i].ev_offset == SIZE_MAX) continue;
            vc_data_used += t->meta[i].num_buckets;
        }
        if (vc_data_used > 0) {
            t->vc_data = (uint32_t *)malloc(vc_data_used * sizeof(uint32_t));
            if (!t->vc_data) {
                fprintf(stderr, "[bps_reader] OOM allocating VC cache (%zu uint32s)\n",
                        vc_data_used);
                /* not fatal — EVs still loaded, vc queries return 0 */
            } else if (fread(t->vc_data, sizeof(uint32_t), vc_data_used, f)
                       != vc_data_used) {
                fprintf(stderr, "[bps_reader] Truncated VC cache (expected %zu uint32s)\n",
                        vc_data_used);
                free(t->vc_data);
                t->vc_data = NULL;
            } else {
                t->vc_data_used = vc_data_used;
                t->vc_data_cap = vc_data_used;
            }
        }
    }

    fclose(f);

    printf("[bps_reader] Cache loaded: %lld nodes, %lld ev nodes, "
           "vc_data %s (%.1f MB total, format v%u)\n",
           (long long)num_nodes, (long long)num_ev_nodes,
           t->vc_data ? "present" : "absent",
           ((size_t)table_size * (sizeof(BPSNodeKey) + sizeof(BPSNodeMeta) + 1)
            + strat_data_used + ev_data_used * sizeof(float)
            + t->vc_data_used * sizeof(uint32_t))
           / (1024.0 * 1024.0), ver);
    fflush(stdout);

    return 0;
}

/* ── Utility functions ───────────────────────────────────────────────── */

int bps_has_action_evs(const BPSReader *r) {
    return r ? r->evs_loaded : 0;
}

void bps_reader_stats(const BPSReader *r,
                       int64_t *num_nodes,
                       int64_t *num_ev_nodes,
                       int64_t *strat_bytes_used,
                       int64_t *ev_floats_used) {
    if (!r) return;
    if (num_nodes) *num_nodes = r->table.num_nodes;
    if (num_ev_nodes) *num_ev_nodes = r->table.num_ev_nodes;
    if (strat_bytes_used) *strat_bytes_used = (int64_t)r->table.strat_data_used;
    if (ev_floats_used) *ev_floats_used = (int64_t)r->table.ev_data_used;
}

void bps_reader_free(BPSReader *r) {
    if (!r) return;
    BPSTable *t = &r->table;
    free(t->keys);       t->keys = NULL;
    free(t->meta);       t->meta = NULL;
    free(t->occupied);   t->occupied = NULL;
    free(t->strat_data); t->strat_data = NULL;
    free(t->ev_data);    t->ev_data = NULL;
    free(t->vc_data);    t->vc_data = NULL;
    t->table_size = 0;
    t->num_nodes = 0;
    t->num_ev_nodes = 0;
    t->strat_data_cap = 0;
    t->strat_data_used = 0;
    t->ev_data_cap = 0;
    t->ev_data_used = 0;
    t->vc_data_cap = 0;
    t->vc_data_used = 0;
    free(r->textures.bucket_cache);
    r->textures.bucket_cache = NULL;
    r->textures.loaded = 0;
    r->textures.num_textures = 0;
    if (r->pa_emd_turn_centroids) {
        free(r->pa_emd_turn_centroids);
        r->pa_emd_turn_centroids = NULL;
        r->pa_emd_turn_k = 0;
        r->pa_emd_turn_bins = 0;
    }
    if (r->op1_turn_centroids) {
        free(r->op1_turn_centroids);
        r->op1_turn_centroids = NULL;
    }
    if (r->op1_river_centroids) {
        free(r->op1_river_centroids);
        r->op1_river_centroids = NULL;
    }
    r->op1_turn_loaded = 0;
    r->op1_river_loaded = 0;
    r->op1_num_textures = 0;
    r->op1_k = 0;
    r->op1_n_bins = 0;
    if (r->ochs_river_centroids) {
        free(r->ochs_river_centroids);
        r->ochs_river_centroids = NULL;
    }
    r->ochs_river_centroids_k = 0;
    r->ochs_river_enabled = 0;
    r->ochs_num_clusters = 0;
    r->strategies_loaded = 0;
    r->evs_loaded = 0;
}
