/**
 * bps_reader.h — Read-only query API for unified .bps blueprint files.
 *
 * Loads decompressed BPS3 strategy data and BPR3 action-EV data into
 * compact read-only hash tables. Designed for ctypes consumption from
 * Python (nexusgto-api) without the ~240 GB memory overhead of the
 * Python-based BlueprintV2 loader.
 *
 * Architecture:
 *   - Python decompresses LZMA, passes raw inner-format bytes to C.
 *   - C builds node-level hash tables keyed by (action_hash, player, street).
 *   - Each node stores contiguous [num_buckets][num_actions] arrays.
 *   - Strategies stored as uint8 (quantized 0-255), converted to float on query.
 *   - Action EVs stored as float32 directly.
 *
 * Memory footprint: ~4 GB for the full 1.5B blueprint (vs ~240 GB in Python).
 */
#ifndef BPS_READER_H
#define BPS_READER_H

#include <stdint.h>
#include <stddef.h>

#define BPS_MAX_ACTIONS 16
#define BPS_MAX_HANDS   1326
#define BPS_MAX_TEXTURES 1760
#define BPS_TEXTURE_INDEX_SIZE 4096

#ifdef _WIN32
#define BPS_EXPORT __declspec(dllexport)
#else
#define BPS_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Data structures ─────────────────────────────────────────────────── */

/**
 * Node key: identifies a decision point across all buckets.
 * Packed to 12 bytes for cache efficiency at scale (~6M nodes).
 */
#pragma pack(push, 1)
typedef struct {
    uint64_t action_hash;   /* 8B — from compute_action_hash() */
    uint8_t  player;        /* 1B — 0-5 */
    uint8_t  street;        /* 1B — 0=preflop, 1=flop, 2=turn, 3=river */
    uint16_t _pad;          /* 2B — alignment / future use */
} BPSNodeKey;               /* 12 bytes total */
#pragma pack(pop)

/**
 * Per-node metadata stored alongside the key in the hash table.
 *
 * Offsets are size_t to support >4 GB strategy/EV data buffers. For the
 * full 1.5B blueprint with ~6M nodes × 200 buckets × avg 4 actions,
 * strat_data can reach ~4.8 GB, exceeding uint32_t range.
 */
typedef struct {
    size_t   strat_offset;  /* byte offset into strat_data (SIZE_MAX = none) */
    size_t   ev_offset;     /* float offset into ev_data (SIZE_MAX = none) */
    uint16_t num_buckets;   /* actual bucket count for this node */
    uint8_t  num_actions;   /* 1-16 */
    uint8_t  _pad;
} BPSNodeMeta;

/**
 * Read-only open-addressing hash table for node-level lookups.
 */
typedef struct {
    BPSNodeKey  *keys;          /* [table_size] */
    BPSNodeMeta *meta;          /* [table_size] */
    uint8_t     *occupied;      /* [table_size] — 0=empty, 1=occupied */
    uint8_t     *strat_data;    /* flat: uint8[nb][na] per node, concatenated */
    float       *ev_data;       /* flat: float32[nb][na] per node, concatenated */
    int64_t      table_size;
    int64_t      num_nodes;     /* strategy nodes inserted */
    int64_t      num_ev_nodes;  /* EV nodes inserted (may differ from num_nodes) */
    size_t       strat_data_cap;
    size_t       strat_data_used;
    size_t       ev_data_cap;   /* in floats */
    size_t       ev_data_used;  /* in floats */
} BPSTable;

/**
 * Precomputed flop texture bucket cache.
 *
 * Stores bucket assignments for all 1,326 hands on each of the ~1,755
 * canonical flop textures. Loaded from a TXC1 file produced during
 * blueprint training. ~9.3 MB total.
 *
 * Lookup: canonicalize flop → compute board_hash → hashmap lookup →
 * bucket_cache[texture_idx * 1326 + hand_idx].
 */
typedef struct {
    uint64_t hash_keys[BPS_MAX_TEXTURES];         /* canonical flop hashes */
    int      hash_index[BPS_TEXTURE_INDEX_SIZE];   /* open-addressed hashmap: -1=empty */
    int     *bucket_cache;                         /* [num_textures * 1326] */
    int      num_textures;
    int      num_buckets;                          /* from TXC1 header */
    int      use_legacy_mixer;                     /* mixer used to build cache keys */
    int      loaded;
} BPSTextureCache;

/**
 * Top-level reader handle. Caller allocates (stack or heap), passes to all APIs.
 */
typedef struct {
    BPSTable table;
    BPSTextureCache textures;
    int      use_legacy_mixer;      /* 0=splitmix64, 1=boost */
    int      num_postflop_buckets;  /* typically 200 */
    int      num_preflop_buckets;   /* typically 169 */
    int      strategies_loaded;     /* 1 after bps_load_strategies succeeds */
    int      evs_loaded;            /* 1 after bps_load_evs succeeds */
} BPSReader;

/* ── Public API ──────────────────────────────────────────────────────── */

/**
 * Load decompressed BPS3 strategy bytes into the hash table.
 *
 * Python should: open .bps, read BPS3 outer header, LZMA decompress the
 * strategy section, then pass the raw decompressed bytes here. C parses
 * the inner BPS3 format and builds the node-level hash table.
 *
 * Args:
 *   r: reader handle (caller allocates, zero-initialized)
 *   data: decompressed BPS3 inner bytes (magic + entries)
 *   len: byte length of data
 *   use_legacy_mixer: 0=splitmix64 (v3+), 1=boost (v2 checkpoint)
 *   postflop_buckets: number of postflop buckets (typically 200)
 *   preflop_buckets: number of preflop buckets (typically 169)
 *
 * Returns 0 on success, -1 on error.
 */
BPS_EXPORT int bps_load_strategies(BPSReader *r,
                                    const unsigned char *data, size_t len,
                                    int use_legacy_mixer,
                                    int postflop_buckets, int preflop_buckets);

/**
 * Load decompressed BPR3 action-EV bytes into the hash table.
 *
 * Must be called after bps_load_strategies (needs bucket counts and
 * the table structure). Entries are grouped by node key and stored
 * as float32[num_buckets][num_actions] blocks.
 *
 * Args:
 *   r: reader handle (after bps_load_strategies)
 *   data: decompressed BPR3 inner bytes (magic + entries)
 *   len: byte length of data
 *
 * Returns 0 on success, -1 on error.
 */
BPS_EXPORT int bps_load_evs(BPSReader *r,
                              const unsigned char *data, size_t len);

/**
 * Query strategies for ALL buckets at a decision node.
 *
 * Computes action_hash from action_history, looks up the node, and
 * copies the strategy data (converted from uint8 to normalized float32)
 * into the output buffer.
 *
 * Args:
 *   r: loaded reader
 *   action_history: array of action indices (e.g., [3, 1] = raise, call)
 *   action_len: length of action_history
 *   player: player index (0-5)
 *   street: 0=preflop, 1=flop, 2=turn, 3=river
 *   out: output buffer, row-major float[num_buckets][num_actions]
 *   out_size: capacity of out in floats
 *   num_buckets_out: receives the bucket count for this node
 *
 * Returns num_actions (1-16), or 0 if node not found.
 */
BPS_EXPORT int bps_query_all_strategies(const BPSReader *r,
                                         const int *action_history, int action_len,
                                         int player, int street,
                                         float *out, int out_size,
                                         int *num_buckets_out);

/**
 * Query action EVs for ALL buckets at a decision node.
 *
 * Same lookup as bps_query_all_strategies but returns float32 EVs
 * (already in chips, not quantized).
 *
 * Returns num_actions (1-16), or 0 if node not found or no EVs loaded.
 */
BPS_EXPORT int bps_query_all_evs(const BPSReader *r,
                                   const int *action_history, int action_len,
                                   int player, int street,
                                   float *out, int out_size,
                                   int *num_buckets_out);

/**
 * Query strategy for a single bucket at a decision node.
 *
 * Args:
 *   strategy_out: output buffer, must hold at least BPS_MAX_ACTIONS floats
 *
 * Returns num_actions (1-16), or 0 if node not found.
 */
BPS_EXPORT int bps_query_strategy(const BPSReader *r,
                                    const int *action_history, int action_len,
                                    int player, int street, int bucket,
                                    float *strategy_out);

/**
 * Check if action EVs have been loaded.
 * Returns 1 if bps_load_evs succeeded, 0 otherwise.
 */
BPS_EXPORT int bps_has_action_evs(const BPSReader *r);

/**
 * Get reader statistics.
 * Any out-pointer may be NULL to skip.
 */
BPS_EXPORT void bps_reader_stats(const BPSReader *r,
                                   int64_t *num_nodes,
                                   int64_t *num_ev_nodes,
                                   int64_t *strat_bytes_used,
                                   int64_t *ev_floats_used);

/**
 * Load a precomputed flop texture bucket cache (TXC1 format).
 *
 * This enables correct hand → bucket mapping for flop queries.
 * Without it, bucket assignments are unavailable and bps_hand_to_bucket
 * returns -1.
 *
 * The TXC1 file is produced during blueprint training (~9.3 MB).
 * The hash keys inside use whichever mixer was active at training time.
 *
 * Args:
 *   r: reader handle
 *   path: path to texture_cache.bin
 *   cache_uses_legacy_mixer: 1 if the cache was built with boost mixer
 *
 * Returns number of textures loaded, or -1 on error.
 */
BPS_EXPORT int bps_load_texture_cache(BPSReader *r, const char *path,
                                       int cache_uses_legacy_mixer);

/**
 * Map a specific hand on a specific flop board to its bucket index.
 *
 * Canonicalizes the flop, looks up the texture in the cache, finds the
 * hand's position in the canonical hand ordering, and returns the
 * precomputed bucket assignment.
 *
 * Args:
 *   r: reader with texture cache loaded
 *   board: 3 flop cards (int encoding: rank*4 + suit)
 *   c0, c1: hero's hole cards (c0 < c1)
 *
 * Returns bucket index (0 to num_buckets-1), or -1 if:
 *   - texture cache not loaded
 *   - flop texture not found in cache
 *   - hand conflicts with board cards
 */
BPS_EXPORT int bps_hand_to_bucket(const BPSReader *r,
                                    const int *board, int c0, int c1);

/**
 * Save the loaded hash table state to a binary cache file.
 *
 * Writes keys, meta, occupied, strat_data, and ev_data to a compact
 * binary format that can be reloaded in ~1 second via bps_load_cache().
 * Only occupied slots are written (no empty hash table padding).
 *
 * Call after bps_load_strategies() + bps_load_evs() complete.
 *
 * Returns 0 on success, -1 on error.
 */
BPS_EXPORT int bps_save_cache(const BPSReader *r, const char *path);

/**
 * Load a previously saved cache file, bypassing LZMA decompression
 * and hash table construction entirely.
 *
 * On success, the reader is fully functional (strategies_loaded=1,
 * evs_loaded=1 if EVs were in the cache). Returns 0 on success.
 *
 * Returns -1 if the file doesn't exist, is corrupt, or has a version
 * mismatch. Caller should fall back to the slow bps_load_strategies path.
 */
BPS_EXPORT int bps_load_cache(BPSReader *r, const char *path);

/**
 * Compute per-action expected values via Monte Carlo traversal.
 *
 * Runs num_iterations of σ̄-sampled tree walks (same algorithm as
 * mccfr_blueprint.c:bp_compute_action_evs) but reads strategies from
 * the reader's compact hash table instead of the 95 GB training table.
 *
 * After this returns, bps_query_all_evs() works. Call bps_save_cache()
 * to persist both strategies and EVs for instant reload.
 *
 * Game config parameters must match the training configuration exactly
 * (Pluribus defaults: 6 players, SB=50, BB=100, stack=10000).
 *
 * Args:
 *   r: loaded reader (strategies_loaded must be 1)
 *   num_iterations: number of MC iterations (50M = ~1hr, 500M = overnight)
 *   num_threads: OpenMP threads (0 = auto)
 *   num_players: 2-6
 *   small_blind, big_blind, initial_stack: game parameters
 *   postflop_bet_sizes, num_postflop_bet_sizes: first-raise fractions
 *   subsequent_bet_sizes, num_subsequent_bet_sizes: re-raise fractions
 *   preflop_tier0..3, n_tier0..3: tiered preflop bet sizes per raise level
 *   preflop_max_raises: cap on preflop raises (typically 4)
 *   turn_centroids: [k][3] float array of turn k-means centroids (NULL = skip)
 *   turn_k: number of turn centroids (0 = use EHS percentile for turn)
 *
 * Returns 0 on success, -1 on error.
 */
BPS_EXPORT int bps_compute_evs(BPSReader *r, int64_t num_iterations,
                                int num_threads,
                                int num_players, int small_blind, int big_blind,
                                int initial_stack,
                                const float *postflop_bet_sizes,
                                int num_postflop_bet_sizes,
                                const float *subsequent_bet_sizes,
                                int num_subsequent_bet_sizes,
                                const float *preflop_tier0, int n_tier0,
                                const float *preflop_tier1, int n_tier1,
                                const float *preflop_tier2, int n_tier2,
                                const float *preflop_tier3, int n_tier3,
                                int preflop_max_raises,
                                const float *turn_centroids, int turn_k);

/**
 * Checkpointed variant of bps_compute_evs. Identical semantics except
 * that every `checkpoint_interval` iterations the accumulated EVs are
 * written into the reader's ev_data and persisted to
 * `checkpoint_path`. Writes are atomic (tmp + rename) so a crash during
 * the checkpoint leaves the previous checkpoint intact.
 *
 * The accumulator is preserved across checkpoints — snapshots are
 * observer-only. The final state is written after the last chunk
 * completes (same as the non-checkpointed path).
 *
 * Pass `checkpoint_interval = 0` or `checkpoint_path = NULL` to disable
 * checkpointing (equivalent to bps_compute_evs).
 */
BPS_EXPORT int bps_compute_evs_checkpointed(BPSReader *r, int64_t num_iterations,
                                int num_threads,
                                int num_players, int small_blind, int big_blind,
                                int initial_stack,
                                const float *postflop_bet_sizes,
                                int num_postflop_bet_sizes,
                                const float *subsequent_bet_sizes,
                                int num_subsequent_bet_sizes,
                                const float *preflop_tier0, int n_tier0,
                                const float *preflop_tier1, int n_tier1,
                                const float *preflop_tier2, int n_tier2,
                                const float *preflop_tier3, int n_tier3,
                                int preflop_max_raises,
                                const float *turn_centroids, int turn_k,
                                int64_t checkpoint_interval,
                                const char *checkpoint_path);

/**
 * Free all memory owned by the reader. Safe to call on a zero-initialized
 * or partially-loaded reader.
 */
BPS_EXPORT void bps_reader_free(BPSReader *r);

#ifdef __cplusplus
}
#endif

#endif /* BPS_READER_H */
