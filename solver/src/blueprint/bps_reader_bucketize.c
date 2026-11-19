/**
 * bps_reader_bucketize.c — turn/river/OCHS/PA-EMD hand-to-bucket computation.
 *
 * Holds the runtime bucketizer:
 * turn/river L2 + OCHS + PA-EMD centroid loaders and the hand->bucket lookups
 * (bps_hand_to_bucket_turn/river, _search variants, bps_init_ochs_river,
 * the per-flop centroid loaders, and bps_compute_flop_hash).
 *
 * Compiled into the SAME shared libraries as bps_reader.c. The three parent
 * helpers it calls (compute_board_hash, tex_index_lookup, tex_index_rebuild)
 * are shared via bps_reader_internal.h (definitions stay in the parent TU).
 * Everything else is declared in bps_reader.h / card_abstraction.h.
 */

#include "bps_reader.h"
#include "bps_reader_internal.h"
#include "bps_joint_canon.h"
#include "card_abstraction.h"
#include "hand_eval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -- RBK1 identity and loader -------------------------------------------- */

#define BPS_RBK1_VERSION 1U
#define BPS_RBK1_HEADER_BYTES 52U
#define BPS_RIVER_BUCKET_ALGORITHM_VERSION 1U
#define BPS_RIVER_BUCKET_MODE_OCHS_EXHAUSTIVE 1U
#define BPS_RIVER_BUCKET_SENTINEL UINT16_C(0xffff)

/* Small self-contained SHA-256 implementation. Keeping it local avoids an
 * OpenSSL dependency in every reader/rollout shared library. */
typedef struct {
    uint32_t state[8];
    uint64_t transformed_bits;
    unsigned char block[64];
    size_t used;
} BpsSha256;

static const uint32_t bps_sha256_k[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

static inline uint32_t bps_sha256_rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32U - n));
}

static void bps_sha256_transform(BpsSha256 *ctx,
                                  const unsigned char data[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        size_t j = (size_t)i * 4U;
        w[i] = ((uint32_t)data[j] << 24)
             | ((uint32_t)data[j + 1] << 16)
             | ((uint32_t)data[j + 2] << 8)
             | (uint32_t)data[j + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = bps_sha256_rotr(w[i - 15], 7)
                    ^ bps_sha256_rotr(w[i - 15], 18)
                    ^ (w[i - 15] >> 3);
        uint32_t s1 = bps_sha256_rotr(w[i - 2], 17)
                    ^ bps_sha256_rotr(w[i - 2], 19)
                    ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = ctx->state[0], b = ctx->state[1];
    uint32_t c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5];
    uint32_t g = ctx->state[6], h = ctx->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t s1 = bps_sha256_rotr(e, 6)
                    ^ bps_sha256_rotr(e, 11)
                    ^ bps_sha256_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + s1 + ch + bps_sha256_k[i] + w[i];
        uint32_t s0 = bps_sha256_rotr(a, 2)
                    ^ bps_sha256_rotr(a, 13)
                    ^ bps_sha256_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b;
    ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f;
    ctx->state[6] += g; ctx->state[7] += h;
}

static void bps_sha256_init(BpsSha256 *ctx) {
    ctx->state[0] = 0x6a09e667U; ctx->state[1] = 0xbb67ae85U;
    ctx->state[2] = 0x3c6ef372U; ctx->state[3] = 0xa54ff53aU;
    ctx->state[4] = 0x510e527fU; ctx->state[5] = 0x9b05688cU;
    ctx->state[6] = 0x1f83d9abU; ctx->state[7] = 0x5be0cd19U;
    ctx->transformed_bits = 0;
    ctx->used = 0;
}

static void bps_sha256_update(BpsSha256 *ctx, const void *src, size_t len) {
    const unsigned char *data = (const unsigned char *)src;
    while (len > 0) {
        size_t room = sizeof(ctx->block) - ctx->used;
        size_t take = len < room ? len : room;
        memcpy(ctx->block + ctx->used, data, take);
        ctx->used += take;
        data += take;
        len -= take;
        if (ctx->used == sizeof(ctx->block)) {
            bps_sha256_transform(ctx, ctx->block);
            ctx->transformed_bits += UINT64_C(512);
            ctx->used = 0;
        }
    }
}

static void bps_sha256_final(BpsSha256 *ctx, unsigned char out[32]) {
    uint64_t total_bits = ctx->transformed_bits + (uint64_t)ctx->used * 8U;
    ctx->block[ctx->used++] = 0x80U;
    if (ctx->used > 56U) {
        memset(ctx->block + ctx->used, 0, 64U - ctx->used);
        bps_sha256_transform(ctx, ctx->block);
        ctx->used = 0;
    }
    memset(ctx->block + ctx->used, 0, 56U - ctx->used);
    for (int i = 0; i < 8; i++)
        ctx->block[56 + i] = (unsigned char)(total_bits >> (56 - 8 * i));
    bps_sha256_transform(ctx, ctx->block);

    for (int i = 0; i < 8; i++) {
        out[i * 4] = (unsigned char)(ctx->state[i] >> 24);
        out[i * 4 + 1] = (unsigned char)(ctx->state[i] >> 16);
        out[i * 4 + 2] = (unsigned char)(ctx->state[i] >> 8);
        out[i * 4 + 3] = (unsigned char)ctx->state[i];
    }
}

static void bps_sha256_le32(BpsSha256 *ctx, uint32_t value) {
    unsigned char bytes[4];
    bytes[0] = (unsigned char)value;
    bytes[1] = (unsigned char)(value >> 8);
    bytes[2] = (unsigned char)(value >> 16);
    bytes[3] = (unsigned char)(value >> 24);
    bps_sha256_update(ctx, bytes, sizeof(bytes));
}

static uint32_t bps_read_le32(const unsigned char *p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static uint64_t bps_read_le64(const unsigned char *p) {
    return (uint64_t)bps_read_le32(p)
         | ((uint64_t)bps_read_le32(p + 4) << 32);
}

static int bps_ochs_river_is_active(const BPSReader *r) {
    return r != NULL
        && r->ochs_river_enabled
        && r->ochs_river_centroids_k > 0
        && r->ochs_num_clusters > 0
        && r->ochs_river_centroids != NULL;
}

static int bps_effective_postflop_buckets(const BPSReader *r) {
    return r->num_postflop_buckets > 0 ? r->num_postflop_buckets : 200;
}

static void bps_clear_river_bucket_table(BPSReader *r) {
    if (!r) return;
    free(r->river_bucket_table);
    r->river_bucket_table = NULL;
    r->river_bucket_table_count = 0;
}

int bps_compute_river_bucket_identity(const BPSReader *r,
                                       unsigned char out[32]) {
    static const unsigned char domain[] = {'R','B','K','1','-','I','D'};
    if (!out || !bps_ochs_river_is_active(r)) return -1;
    if (sizeof(float) != 4U) return -1;

    BpsSha256 sha;
    bps_sha256_init(&sha);
    bps_sha256_update(&sha, domain, sizeof(domain));
    bps_sha256_le32(&sha, 3U);  /* street: river */
    bps_sha256_le32(&sha, BPS_RIVER_BUCKET_ALGORITHM_VERSION);
    bps_sha256_le32(&sha, BPS_RIVER_BUCKET_MODE_OCHS_EXHAUSTIVE);
    bps_sha256_le32(&sha, (uint32_t)bps_effective_postflop_buckets(r));
    bps_sha256_le32(&sha, (uint32_t)r->ochs_num_clusters);
    bps_sha256_le32(&sha, (uint32_t)r->ochs_river_centroids_k);

    /* Raw centroid object bytes are identity-bearing by design. The Python
     * generator obtains this digest through this same exported function. */
    size_t centroid_count = (size_t)r->ochs_river_centroids_k
                          * (size_t)r->ochs_num_clusters;
    bps_sha256_update(&sha, r->ochs_river_centroids,
                      centroid_count * sizeof(float));
    for (int i = 0; i < 169; i++)
        bps_sha256_le32(&sha, (uint32_t)(int32_t)r->ochs_cluster_of_169[i]);
    bps_sha256_final(&sha, out);
    return 0;
}

int bps_load_river_bucket_table(BPSReader *r, const char *path) {
    if (!r) return -1;
    /* Failed reloads must never leave an older table armed. */
    bps_clear_river_bucket_table(r);
    if (!path || !path[0]) return -1;
    if (!bps_ochs_river_is_active(r)) {
        fprintf(stderr, "[bps_reader] RBK1 refused: production OCHS river "
                "path is not initialized\n");
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[bps_reader] Cannot open river bucket table: %s\n", path);
        return -1;
    }

    unsigned char header[BPS_RBK1_HEADER_BYTES];
    if (fread(header, 1, sizeof(header), f) != sizeof(header)) {
        fprintf(stderr, "[bps_reader] RBK1 truncated header: %s\n", path);
        fclose(f);
        return -1;
    }
    if (memcmp(header, "RBK1", 4) != 0) {
        fprintf(stderr, "[bps_reader] Invalid river bucket table magic: %s\n", path);
        fclose(f);
        return -1;
    }

    uint32_t version = bps_read_le32(header + 4);
    uint32_t num_buckets = bps_read_le32(header + 8);
    uint64_t num_classes = bps_read_le64(header + 12);
    int expected_buckets = bps_effective_postflop_buckets(r);
    if (version != BPS_RBK1_VERSION
        || num_buckets != (uint32_t)expected_buckets
        || num_classes != (uint64_t)BPS_JOINT_RIVER_NUM_CLASSES) {
        fprintf(stderr, "[bps_reader] RBK1 shape mismatch at %s "
                "(version=%u buckets=%u classes=%llu; expected %u/%d/%llu)\n",
                path, version, num_buckets, (unsigned long long)num_classes,
                BPS_RBK1_VERSION, expected_buckets,
                (unsigned long long)BPS_JOINT_RIVER_NUM_CLASSES);
        fclose(f);
        return -1;
    }

    unsigned char live_identity[32];
    if (bps_compute_river_bucket_identity(r, live_identity) != 0) {
        fprintf(stderr, "[bps_reader] RBK1 identity unavailable: %s\n", path);
        fclose(f);
        return -1;
    }
    if (memcmp(header + 20, live_identity, sizeof(live_identity)) != 0) {
        fprintf(stderr, "[bps_reader] RBK1 identity mismatch at %s\n"
                "[bps_reader]   file: ", path);
        for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", header[20 + i]);
        fprintf(stderr, "\n[bps_reader]   live: ");
        for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", live_identity[i]);
        fprintf(stderr, "\n[bps_reader] Falling through to live OCHS river bucketing\n");
        fclose(f);
        return -1;
    }

    if (num_classes > (uint64_t)(SIZE_MAX / sizeof(uint16_t))) {
        fclose(f);
        return -1;
    }
    size_t payload_bytes = (size_t)num_classes * sizeof(uint16_t);
    uint16_t *table = (uint16_t *)malloc(payload_bytes);
    if (!table) {
        fprintf(stderr, "[bps_reader] OOM loading RBK1 payload (%zu bytes)\n",
                payload_bytes);
        fclose(f);
        return -1;
    }
    if (fread(table, 1, payload_bytes, f) != payload_bytes) {
        fprintf(stderr, "[bps_reader] RBK1 truncated payload: %s\n", path);
        free(table);
        fclose(f);
        return -1;
    }
    int trailing = fgetc(f);
    if (trailing != EOF || ferror(f)) {
        fprintf(stderr, "[bps_reader] RBK1 has trailing data or read error: %s\n",
                path);
        free(table);
        fclose(f);
        return -1;
    }
    fclose(f);

    /* RBK1 payload is little-endian even on a big-endian reader host. */
    const uint16_t endian_probe = UINT16_C(0x0102);
    if (*(const unsigned char *)&endian_probe != 0x02U) {
        unsigned char *raw = (unsigned char *)table;
        for (uint64_t i = 0; i < num_classes; i++) {
            uint16_t value = (uint16_t)raw[i * 2]
                           | (uint16_t)((uint16_t)raw[i * 2 + 1] << 8);
            table[i] = value;
        }
    }

    /* Preserve the bucketizer's return contract even for a corrupt payload. */
    for (uint64_t i = 0; i < num_classes; i++) {
        uint16_t value = table[i];
        if (value != BPS_RIVER_BUCKET_SENTINEL && value >= num_buckets) {
            fprintf(stderr, "[bps_reader] RBK1 invalid bucket %u at class %llu: %s\n",
                    (unsigned)value, (unsigned long long)i, path);
            free(table);
            return -1;
        }
    }

    r->river_bucket_table = table;
    r->river_bucket_table_count = (int64_t)num_classes;
    fprintf(stderr, "[bps_reader] Loaded RBK1 river bucket table: %llu classes "
            "(%.1f MB) from %s\n",
            (unsigned long long)num_classes,
            (double)payload_bytes / (1024.0 * 1024.0), path);
    return (int)num_classes;
}

int bps_has_river_bucket_table(const BPSReader *r) {
    return r != NULL
        && r->river_bucket_table != NULL
        && r->river_bucket_table_count == (int64_t)BPS_JOINT_RIVER_NUM_CLASSES;
}

/* ── Turn / river bucketizer ─────────────────────────────────────────── */

/* xorshift64 — matches bps_ev_walk.c:46-52 exactly. Must stay in sync
 * so runtime river buckets match training-time buckets bit-for-bit. */
static inline uint64_t bpr_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline int bpr_rng_int(uint64_t *state, int n) {
    return (int)(bpr_rng_next(state) % (uint64_t)n);
}

/* Partial Fisher-Yates — matches bps_ev_walk.c:63-68 exactly. */
static void bpr_partial_shuffle(int *arr, int n, int k, uint64_t *rng) {
    for (int i = 0; i < k && i < n; i++) {
        int j = i + bpr_rng_int(rng, n - i);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

int bps_load_turn_centroids(BPSReader *r, const char *path) {
    if (!r) return -1;

    /* Candidate paths in order of preference. */
    const char *candidates[8];
    int n_cand = 0;
    char bps_dir_candidate[1024];
    char cache_dir_candidate[1024];
    if (path && path[0]) {
        candidates[n_cand++] = path;
    }
    candidates[n_cand++] = "turn_centroids.bin";  /* cwd */
    /* The trainer's cache directory (see bp_cache_path in mccfr_blueprint.c). */
    const char *cache_dir = getenv("BP_CACHE_DIR");
    if (cache_dir && cache_dir[0]) {
        snprintf(cache_dir_candidate, sizeof(cache_dir_candidate),
                 "%s/turn_centroids.bin", cache_dir);
        candidates[n_cand++] = cache_dir_candidate;
    }

    for (int ci = 0; ci < n_cand; ci++) {
        FILE *f = fopen(candidates[ci], "rb");
        if (!f) continue;

        char magic[4];
        if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TCN1", 4) != 0) {
            fclose(f);
            continue;
        }
        int k_saved = 0;
        if (fread(&k_saved, sizeof(int), 1, f) != 1 ||
            k_saved <= 0 || k_saved > 200) {
            fclose(f);
            continue;
        }
        size_t n_floats = (size_t)k_saved * 3;
        if (fread(r->turn_centroids, sizeof(float), n_floats, f) != n_floats) {
            fclose(f);
            continue;
        }
        fclose(f);
        r->turn_centroids_k = k_saved;
        fprintf(stderr, "[bps_reader] Loaded %d turn centroids from %s\n",
                k_saved, candidates[ci]);
        (void)bps_dir_candidate;  /* reserved for future bps-dir derivation */
        return k_saved;
    }
    return -1;
}

int bps_precompute_l2_turn_centroids(BPSReader *r, const char *output_path) {
    /* Reproduce the training-time turn-centroid k-means
     * (mccfr_blueprint.c:2710-2858) so inference can bucket turn
     * boards consistently with how training labelled them.
     *
     * Training uses L2 3-feature centroids (EHS, PPot, NPot) over
     * 2000 sampled turn boards × ~300 hands each, k=200 clusters,
     * 20 k-means iterations, percentile-seeded init. Deterministic
     * from xorshift64 seed 0xABCDEF0123456789. The same algorithm
     * runs here so the produced centroids match training bit-for-bit
     * (modulo OS sort order — the centroid-init step uses qsort on
     * EHS percentile, here a portable shellsort, same comparison).
     *
     * Saves TCN1 (4-byte magic + int32 k + float32[k][3] centroids)
     * to output_path and populates r->turn_centroids in-memory so
     * inference can use them immediately. Returns k on success, -1
     * on failure. ~5 minutes single-threaded on a g4dn.2xlarge. */
    if (!r || !output_path) return -1;

    const int k_target = 200;
    const int n_turn_boards = 2000;
    const int feat_samples = 200;
    const int max_hands = BPS_MAX_HANDS;
    const int hand_batch = 300;

    int max_feat = n_turn_boards * (hand_batch + 1);
    float (*all_feat)[3] = (float(*)[3])malloc((size_t)max_feat * 3 * sizeof(float));
    if (!all_feat) return -1;
    int n_feat = 0;

    int (*hands)[2] = (int(*)[2])malloc((size_t)max_hands * 2 * sizeof(int));
    float (*feats)[3] = (float(*)[3])malloc((size_t)max_hands * 3 * sizeof(float));
    if (!hands || !feats) {
        free(all_feat); if (hands) free(hands); if (feats) free(feats);
        return -1;
    }

    uint64_t trng = 0xABCDEF0123456789ULL;
    fprintf(stderr, "[bps_reader] Precomputing L2 turn centroids "
            "(%d boards × %d hands, ~5 min)...\n",
            n_turn_boards, hand_batch);

    for (int ti = 0; ti < n_turn_boards && n_feat < max_feat - hand_batch; ti++) {
        int deck[52];
        for (int c = 0; c < 52; c++) deck[c] = c;
        for (int i = 0; i < 4; i++) {
            int j = i + (int)(bpr_rng_next(&trng) % (uint64_t)(52 - i));
            int tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp;
        }
        int board[4] = {deck[0], deck[1], deck[2], deck[3]};

        int bblk[52] = {0};
        for (int b = 0; b < 4; b++) bblk[board[b]] = 1;
        int nh = 0;
        for (int c0 = 0; c0 < 52 && nh < max_hands; c0++) {
            if (bblk[c0]) continue;
            for (int c1 = c0 + 1; c1 < 52 && nh < max_hands; c1++) {
                if (bblk[c1]) continue;
                hands[nh][0] = c0;
                hands[nh][1] = c1;
                nh++;
            }
        }

        int batch = (nh > hand_batch) ? hand_batch : nh;
        ca_compute_features(board, 4, (const int(*)[2])hands, batch,
                            feat_samples, feats);
        for (int h = 0; h < batch && n_feat < max_feat; h++) {
            all_feat[n_feat][0] = feats[h][0];
            all_feat[n_feat][1] = feats[h][1];
            all_feat[n_feat][2] = feats[h][2];
            n_feat++;
        }
        if ((ti + 1) % 200 == 0) {
            fprintf(stderr, "[bps_reader]   board %d/%d, %d features collected\n",
                    ti + 1, n_turn_boards, n_feat);
        }
    }
    free(hands); free(feats);

    /* Percentile-seeded k-means init: shell-sort by EHS, pick k
     * percentile points. Same as mccfr_blueprint.c:2783-2802. */
    typedef struct { float ehs; int idx; } EhsIdx;
    EhsIdx *sort_buf = (EhsIdx*)malloc((size_t)n_feat * sizeof(EhsIdx));
    if (!sort_buf) { free(all_feat); return -1; }
    for (int f = 0; f < n_feat; f++) {
        sort_buf[f].ehs = all_feat[f][0];
        sort_buf[f].idx = f;
    }
    for (int gap = n_feat / 2; gap > 0; gap /= 2)
        for (int i = gap; i < n_feat; i++) {
            EhsIdx tmp = sort_buf[i];
            int j = i;
            while (j >= gap && sort_buf[j - gap].ehs > tmp.ehs) {
                sort_buf[j] = sort_buf[j - gap];
                j -= gap;
            }
            sort_buf[j] = tmp;
        }

    float (*centroids)[3] = (float(*)[3])malloc((size_t)k_target * 3 * sizeof(float));
    if (!centroids) { free(sort_buf); free(all_feat); return -1; }
    for (int c = 0; c < k_target; c++) {
        int si = (int)((float)c / k_target * n_feat);
        if (si >= n_feat) si = n_feat - 1;
        int fi = sort_buf[si].idx;
        centroids[c][0] = all_feat[fi][0];
        centroids[c][1] = all_feat[fi][1];
        centroids[c][2] = all_feat[fi][2];
    }
    free(sort_buf);

    int *counts = (int*)calloc(k_target, sizeof(int));
    float (*sums)[3] = (float(*)[3])calloc((size_t)k_target * 3, sizeof(float));
    if (!counts || !sums) {
        free(centroids); free(all_feat);
        if (counts) free(counts);
        if (sums) free(sums);
        return -1;
    }
    for (int iter = 0; iter < 20; iter++) {
        memset(counts, 0, (size_t)k_target * sizeof(int));
        memset(sums, 0, (size_t)k_target * 3 * sizeof(float));
        for (int f = 0; f < n_feat; f++) {
            int best_c = 0;
            float best_d = 1e30f;
            for (int c = 0; c < k_target; c++) {
                float d0 = all_feat[f][0] - centroids[c][0];
                float d1 = all_feat[f][1] - centroids[c][1];
                float d2 = all_feat[f][2] - centroids[c][2];
                float d = d0*d0 + d1*d1 + d2*d2;
                if (d < best_d) { best_d = d; best_c = c; }
            }
            counts[best_c]++;
            sums[best_c][0] += all_feat[f][0];
            sums[best_c][1] += all_feat[f][1];
            sums[best_c][2] += all_feat[f][2];
        }
        for (int c = 0; c < k_target; c++) {
            if (counts[c] > 0) {
                centroids[c][0] = sums[c][0] / counts[c];
                centroids[c][1] = sums[c][1] / counts[c];
                centroids[c][2] = sums[c][2] / counts[c];
            }
        }
    }

    int actual_k = 0;
    for (int c = 0; c < k_target && actual_k < 200; c++) {
        if (counts[c] > 0) {
            r->turn_centroids[actual_k][0] = centroids[c][0];
            r->turn_centroids[actual_k][1] = centroids[c][1];
            r->turn_centroids[actual_k][2] = centroids[c][2];
            actual_k++;
        }
    }
    r->turn_centroids_k = actual_k;
    free(counts); free(sums); free(centroids); free(all_feat);

    FILE *fc = fopen(output_path, "wb");
    if (fc) {
        fwrite("TCN1", 1, 4, fc);
        fwrite(&actual_k, sizeof(int), 1, fc);
        fwrite(r->turn_centroids, sizeof(float), (size_t)actual_k * 3, fc);
        fclose(fc);
        fprintf(stderr, "[bps_reader] Saved %d L2 turn centroids to %s\n",
                actual_k, output_path);
    }

    return actual_k;
}

int bps_hand_to_bucket_turn(const BPSReader *r,
                             const int *board, int c0, int c1) {
    if (!r || !board) return -1;

    /* Hand must not collide with the 4-card turn board. */
    for (int i = 0; i < 4; i++) {
        if (board[i] == c0 || board[i] == c1) return -1;
    }

    /* PA-EMD path: compute equity distribution, find nearest centroid by EMD */
    if (r->pa_emd_turn_k > 0 && r->pa_emd_turn_centroids) {
        int hand[2] = {c0, c1};
        int n_bins = r->pa_emd_turn_bins;
        float *hist = (float*)malloc(n_bins * sizeof(float));
        if (!hist) return -1;
        ca_compute_ehs_distribution(board, 4, hand, 1000, hist, n_bins);
        int bucket = ca_nearest_centroid_emd(hist, r->pa_emd_turn_centroids,
                                              r->pa_emd_turn_k, n_bins);
        free(hist);
        return bucket;
    }

    /* L2 path (default): 3D features + nearest centroid */
    if (r->turn_centroids_k <= 0) return -1;
    int hand[1][2] = {{c0, c1}};
    float feat[1][3];
    ca_compute_features(board, 4, (const int(*)[2])hand, 1, 200, feat);
    return ca_nearest_centroid(feat[0],
                                (const float(*)[3])r->turn_centroids,
                                r->turn_centroids_k);
}

int bps_hand_to_bucket_river(const BPSReader *r,
                              const int *board, int c0, int c1) {
    if (!r || !board) return -1;

    /* Offline-table hook: deliberately the first branch after pointer
     * validation. It is armed only for the production exhaustive OCHS mode;
     * legacy sampled-EHS behavior is never canonical-table accelerated.
     * A sentinel or any canonicalization/rank failure is a pure miss and
     * falls through to the existing live code below without changing it. */
    if (bps_has_river_bucket_table(r) && bps_ochs_river_is_active(r)) {
        int hand[2] = {c0, c1};
        int canon_board[5], canon_hand[2];
        if (bps_joint_canon_river(board, hand, canon_board, canon_hand) == 0) {
            int64_t rank = bps_joint_rank_river(canon_board, canon_hand);
            if (rank >= 0 && rank < r->river_bucket_table_count) {
                uint16_t value = r->river_bucket_table[rank];
                if (value != BPS_RIVER_BUCKET_SENTINEL) return (int)value;
            }
        }
    }

    int k = r->num_postflop_buckets > 0 ? r->num_postflop_buckets : 200;

    /* Hand must not collide with the 5-card river board. */
    for (int i = 0; i < 5; i++) {
        if (board[i] == c0 || board[i] == c1) return -1;
    }

    /* OCHS path (matches mccfr_blueprint.c:97-114 training-time bucketer).
     * Required for any blueprint trained with BPS_RIVER_OCHS=1.
     * Pre-init: ochs_river_enabled=0, falls through to legacy EHS percentile. */
    if (r->ochs_river_enabled && r->ochs_river_centroids_k > 0 &&
        r->ochs_num_clusters > 0 && r->ochs_river_centroids != NULL) {
        int hand[1][2] = {{ c0, c1 }};
        /* Cap on opponent clusters; 16 is comfortably above Pluribus-default 8
         * and keeps the feature buffer stack-resident. */
        float feat[16];
        int kc = r->ochs_num_clusters;
        if (kc > 16) kc = 16;
        int rc = ca_river_ochs_features(board, 5, (const int(*)[2])hand, 1,
                                         r->ochs_cluster_of_169,
                                         kc, feat);
        if (rc == 0) {
            return ca_nearest_centroid_nd(feat, kc,
                                           r->ochs_river_centroids,
                                           r->ochs_river_centroids_k);
        }
        /* fall through to legacy on feature failure (very rare) */
    }

    /* Legacy 200-sample EHS percentile path, for blueprints trained
     * without OCHS (or as fallback if OCHS init failed). MUST match
     * mccfr_blueprint.c:117-138 byte-for-byte. */

    /* Build available deck (52 minus board minus hero hand). */
    int blk[52] = {0};
    for (int b = 0; b < 5; b++) blk[board[b]] = 1;
    blk[c0] = 1;
    blk[c1] = 1;
    int av[52];
    int nav = 0;
    for (int c = 0; c < 52; c++) if (!blk[c]) av[nav++] = c;
    if (nav < 2) return -1;

    /* Deterministic RNG seed — MUST match bps_ev_walk.c:880-882 exactly. */
    uint64_t erng = (uint64_t)c0 * 1000003ULL + (uint64_t)c1 * 999983ULL;
    for (int brd = 0; brd < 5; brd++) {
        erng = erng * 6364136223846793005ULL + (uint64_t)board[brd];
    }

    int wins = 0, ties = 0, total = 0;
    int h7[7] = { board[0], board[1], board[2], board[3], board[4], c0, c1 };
    uint32_t hs = eval7(h7);
    for (int si = 0; si < 200 && nav >= 2; si++) {
        bpr_partial_shuffle(av, nav, 2, &erng);
        int o7[7] = { board[0], board[1], board[2], board[3], board[4],
                      av[0], av[1] };
        uint32_t os_ = eval7(o7);
        if (hs > os_) wins++;
        else if (hs == os_) ties++;
        total++;
    }
    float ehs = (total > 0) ? ((float)wins + 0.5f * (float)ties) / (float)total
                            : 0.5f;
    int bucket = (int)(ehs * (float)k);
    if (bucket < 0) bucket = 0;
    if (bucket >= k) bucket = k - 1;
    return bucket;
}

int bps_init_ochs_river(BPSReader *r, int num_clusters) {
    /* Mirrors mccfr_blueprint.c:2964-3120 training-time OCHS init. Single-
     * threaded, deterministic xorshift seed 0x12345678DEADBEEF, 200 sampled
     * river boards × 300 hands subsample, k-means to postflop_num_buckets
     * (typically 200) centroids in 8-d (or num_clusters-d) space. ~1-2s. */
    if (!r) return -1;
    /* Centroids and cluster map are identity-bearing. Reinitializing them
     * invalidates any previously validated offline table before mutation. */
    bps_clear_river_bucket_table(r);
    if (r->ochs_river_centroids) {
        free(r->ochs_river_centroids);
        r->ochs_river_centroids = NULL;
    }
    r->ochs_river_centroids_k = 0;
    r->ochs_river_enabled = 0;

    int kc = num_clusters > 0 ? num_clusters : CA_OCHS_DEFAULT_CLUSTERS;
    if (kc > 16) kc = 16;  /* matches feat[16] cap in lookup path */

    int target_buckets = r->num_postflop_buckets > 0 ? r->num_postflop_buckets : 200;

    /* Step 1: 169 → kc OCHS cluster map. Deterministic. */
    int actual_clusters = ca_compute_ochs_clusters(kc, 2000, r->ochs_cluster_of_169);
    if (actual_clusters <= 0) {
        fprintf(stderr, "[bps_reader] OCHS init: ca_compute_ochs_clusters failed\n");
        return -1;
    }
    r->ochs_num_clusters = kc;

    /* Step 2: k-cluster river centroids by sampling 200 boards × 300 hands.
     * MUST match mccfr_blueprint.c:3041-3120 exactly (seed, sample sizes, etc.) */
    int n_boards = 200;
    int max_feat = n_boards * 300;
    float *all_feat = (float*)malloc((size_t)max_feat * kc * sizeof(float));
    if (!all_feat) {
        fprintf(stderr, "[bps_reader] OCHS init: alloc failed\n");
        return -1;
    }
    int n_feat = 0;

    uint64_t brng = 0x12345678DEADBEEFULL;
    int (*hands_buf)[2] = (int(*)[2])malloc((size_t)BPS_MAX_HANDS * 2 * sizeof(int));
    if (!hands_buf) { free(all_feat); return -1; }

    for (int bi = 0; bi < n_boards && n_feat < max_feat - 300; bi++) {
        int deck[52];
        for (int c = 0; c < 52; c++) deck[c] = c;
        for (int i = 0; i < 5; i++) {
            int j = i + (int)(bpr_rng_next(&brng) % (uint64_t)(52 - i));
            int tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp;
        }
        int board[5] = {deck[0], deck[1], deck[2], deck[3], deck[4]};

        int nh = ca_generate_hands(board, 5, hands_buf);
        int batch = (nh > 300) ? 300 : nh;
        float *feats = (float*)malloc((size_t)batch * kc * sizeof(float));
        if (!feats) continue;
        int rc = ca_river_ochs_features(board, 5,
                                         (const int(*)[2])hands_buf, batch,
                                         r->ochs_cluster_of_169,
                                         kc, feats);
        if (rc == 0) {
            int copy_n = batch;
            if (n_feat + copy_n > max_feat) copy_n = max_feat - n_feat;
            memcpy(all_feat + (size_t)n_feat * kc, feats,
                   (size_t)copy_n * kc * sizeof(float));
            n_feat += copy_n;
        }
        free(feats);
    }
    free(hands_buf);

    if (n_feat <= 0) {
        free(all_feat);
        fprintf(stderr, "[bps_reader] OCHS init: no features collected\n");
        return -1;
    }

    int *labels = (int*)malloc((size_t)n_feat * sizeof(int));
    if (!labels) { free(all_feat); return -1; }
    int actual = ca_assign_buckets_kmeans_nd(all_feat, n_feat, kc,
                                              target_buckets, 20, labels);
    (void)actual;

    int *counts = (int*)calloc(target_buckets, sizeof(int));
    float *sums = (float*)calloc((size_t)target_buckets * kc, sizeof(float));
    if (!counts || !sums) {
        free(labels); free(all_feat);
        if (counts) free(counts);
        if (sums) free(sums);
        return -1;
    }
    for (int f = 0; f < n_feat; f++) {
        int b = labels[f];
        counts[b]++;
        for (int d = 0; d < kc; d++)
            sums[(size_t)b * kc + d] += all_feat[(size_t)f * kc + d];
    }

    float *centroids = (float*)malloc((size_t)target_buckets * kc * sizeof(float));
    if (!centroids) {
        free(labels); free(counts); free(sums); free(all_feat);
        return -1;
    }
    int kept = 0;
    for (int b = 0; b < target_buckets; b++) {
        if (counts[b] == 0) continue;
        for (int d = 0; d < kc; d++)
            centroids[(size_t)kept * kc + d] = sums[(size_t)b * kc + d] / counts[b];
        kept++;
    }

    free(labels); free(counts); free(sums); free(all_feat);

    r->ochs_river_centroids = centroids;
    r->ochs_river_centroids_k = kept;
    r->ochs_river_enabled = 1;

    fprintf(stderr, "[bps_reader] OCHS river bucketing initialized: "
            "%d clusters from %d samples (%d-d)\n",
            kept, n_feat, kc);
    return 0;
}

/* ── PA-EMD integration ─────────────────────────────────────────────── */

int bps_load_pa_emd_flop_cache(BPSReader *r, const char *path) {
    if (!r || !path || !path[0]) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PAEB", 4) != 0) {
        fclose(f); return -1;
    }

    int version, n_tex, n_buckets, n_bins, mc_samples;
    if (fread(&version, sizeof(int), 1, f) != 1 || version != 1) {
        fclose(f); return -1;
    }
    if (fread(&n_tex, sizeof(int), 1, f) != 1 ||
        n_tex <= 0 || n_tex > BPS_MAX_TEXTURES) {
        fclose(f); return -1;
    }
    if (fread(&n_buckets, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
    if (fread(&n_bins, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
    if (fread(&mc_samples, sizeof(int), 1, f) != 1) { fclose(f); return -1; }

    /* Read hashes */
    uint64_t hashes[BPS_MAX_TEXTURES];
    if (fread(hashes, sizeof(uint64_t), n_tex, f) != (size_t)n_tex) {
        fclose(f); return -1;
    }

    /* Allocate bucket cache if needed */
    BPSTextureCache *tc = &r->textures;
    if (!tc->bucket_cache) {
        tc->bucket_cache = (int*)calloc((size_t)BPS_MAX_TEXTURES * 1326, sizeof(int));
        if (!tc->bucket_cache) { fclose(f); return -1; }
    }

    /* Read bucket data directly into the texture cache */
    if (fread(tc->bucket_cache, sizeof(int), (size_t)n_tex * 1326, f)
        != (size_t)n_tex * 1326) {
        fclose(f); return -1;
    }
    fclose(f);

    /* Populate hash keys and rebuild the hash index */
    for (int i = 0; i < n_tex; i++) {
        tc->hash_keys[i] = hashes[i];
    }
    tc->num_textures = n_tex;
    tc->num_buckets = n_buckets;
    tc->loaded = 1;

    /* Rebuild open-addressed hash index */
    tex_index_rebuild(tc);

    r->pa_emd_flop_loaded = 1;
    /* num_postflop_buckets participates in the RBK1 identity. */
    bps_clear_river_bucket_table(r);
    r->num_postflop_buckets = n_buckets;

    fprintf(stderr, "[bps_reader] Loaded PA-EMD flop cache: %d textures, "
            "%d buckets, %d bins, %d MC samples from %s\n",
            n_tex, n_buckets, n_bins, mc_samples, path);
    return n_tex;
}

int bps_load_pa_emd_turn_centroids(BPSReader *r, const char *path) {
    if (!r || !path || !path[0]) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TCNE", 4) != 0) {
        fclose(f); return -1;
    }

    int version, k, n_bins;
    if (fread(&version, sizeof(int), 1, f) != 1 || version != 1) {
        fclose(f); return -1;
    }
    if (fread(&k, sizeof(int), 1, f) != 1 || k <= 0 || k > 500) {
        fclose(f); return -1;
    }
    if (fread(&n_bins, sizeof(int), 1, f) != 1 || n_bins <= 0 || n_bins > 200) {
        fclose(f); return -1;
    }

    float *centroids = (float*)malloc((size_t)k * n_bins * sizeof(float));
    if (!centroids) { fclose(f); return -1; }
    if (fread(centroids, sizeof(float), (size_t)k * n_bins, f) != (size_t)k * n_bins) {
        free(centroids); fclose(f); return -1;
    }
    fclose(f);

    /* Free any previous PA-EMD turn centroids */
    if (r->pa_emd_turn_centroids) free(r->pa_emd_turn_centroids);

    r->pa_emd_turn_centroids = centroids;
    r->pa_emd_turn_k = k;
    r->pa_emd_turn_bins = n_bins;

    fprintf(stderr, "[bps_reader] Loaded PA-EMD turn centroids: %d centroids × "
            "%d bins from %s\n", k, n_bins, path);
    return k;
}

/* ── Per-flop search-time centroids ──────────────────────────────────── */

uint64_t bps_compute_flop_hash(const BPSReader *r, const int *flop) {
    if (!r || !flop) return 0ULL;
    for (int i = 0; i < 3; i++) {
        if (flop[i] < 0 || flop[i] >= 52) return 0ULL;
    }
    /* Use the texture cache's mixer mode if a cache is loaded; otherwise
     * fall back to the reader's mixer. The two should agree after a proper
     * load, but we prefer the texture cache's flag since that's what drives
     * the lookup path in tex_index_lookup. */
    int mixer = r->textures.loaded
                ? r->textures.use_legacy_mixer
                : r->use_legacy_mixer;
    return compute_board_hash(flop, 3, mixer);
}

/* TCN3 loader shared between turn and river (format is identical; street
 * only determines which BPSReader pointer receives the payload). */
static int bps_load_op1_tcn3(BPSReader *r, const char *path,
                              float **out_centroids, int *out_loaded,
                              const char *label) {
    if (!r || !path || !path[0] || !out_centroids || !out_loaded) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[bps_reader] %s: cannot open %s\n", label, path);
        return -1;
    }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TCN3", 4) != 0) {
        fprintf(stderr, "[bps_reader] %s: bad magic at %s\n", label, path);
        fclose(f); return -1;
    }
    int version;
    if (fread(&version, sizeof(int), 1, f) != 1 || version != 1) {
        fprintf(stderr, "[bps_reader] %s: bad version at %s\n", label, path);
        fclose(f); return -1;
    }
    int num_textures, k, n_bins;
    if (fread(&num_textures, sizeof(int), 1, f) != 1 ||
        num_textures <= 0 || num_textures > BPS_MAX_TEXTURES) {
        fprintf(stderr, "[bps_reader] %s: bad num_textures at %s\n", label, path);
        fclose(f); return -1;
    }
    if (fread(&k, sizeof(int), 1, f) != 1 || k <= 0 || k > 4096) {
        fprintf(stderr, "[bps_reader] %s: bad k at %s\n", label, path);
        fclose(f); return -1;
    }
    if (fread(&n_bins, sizeof(int), 1, f) != 1 || n_bins <= 0 || n_bins > 200) {
        fprintf(stderr, "[bps_reader] %s: bad n_bins at %s\n", label, path);
        fclose(f); return -1;
    }

    /* Texture cache must be loaded first — we need num_textures to match
     * so flop_hash lookups index the right centroid slice. */
    if (!r->textures.loaded) {
        fprintf(stderr, "[bps_reader] %s: texture cache not loaded; "
                "call bps_load_texture_cache / bps_load_pa_emd_flop_cache first\n",
                label);
        fclose(f); return -1;
    }
    if (num_textures != r->textures.num_textures) {
        fprintf(stderr, "[bps_reader] %s: num_textures mismatch (TCN3=%d, "
                "texture cache=%d); flop enumeration out of sync\n",
                label, num_textures, r->textures.num_textures);
        fclose(f); return -1;
    }

    /* If we already have a prior table loaded, verify shape compat;
     * either free-and-reload (if shape changed) or reuse the buffer. */
    if (r->op1_num_textures > 0
        && (r->op1_num_textures != num_textures
            || r->op1_k != k
            || r->op1_n_bins != n_bins)) {
        fprintf(stderr, "[bps_reader] %s: TCN3 shape mismatch with prior load "
                "(was %d×%d×%d, now %d×%d×%d)\n",
                label, r->op1_num_textures, r->op1_k, r->op1_n_bins,
                num_textures, k, n_bins);
        fclose(f); return -1;
    }

    size_t n_floats = (size_t)num_textures * (size_t)k * (size_t)n_bins;
    float *centroids = (float*)malloc(n_floats * sizeof(float));
    if (!centroids) { fclose(f); return -1; }
    if (fread(centroids, sizeof(float), n_floats, f) != n_floats) {
        fprintf(stderr, "[bps_reader] %s: short read at %s\n", label, path);
        free(centroids); fclose(f); return -1;
    }
    fclose(f);

    if (*out_centroids) free(*out_centroids);
    *out_centroids = centroids;
    *out_loaded = 1;
    r->op1_num_textures = num_textures;
    r->op1_k = k;
    r->op1_n_bins = n_bins;

    fprintf(stderr, "[bps_reader] %s: %d textures × %d buckets × %d bins "
            "(%.1f MB) from %s\n",
            label, num_textures, k, n_bins,
            (double)(n_floats * sizeof(float)) / (1024.0 * 1024.0), path);
    return k;
}

int bps_load_op1_per_flop_turn_centroids(BPSReader *r, const char *path) {
    return bps_load_op1_tcn3(r, path, &r->op1_turn_centroids,
                             &r->op1_turn_loaded,
                             "per-flop turn centroids");
}

int bps_load_op1_per_flop_river_centroids(BPSReader *r, const char *path) {
    return bps_load_op1_tcn3(r, path, &r->op1_river_centroids,
                             &r->op1_river_loaded,
                             "per-flop river centroids");
}

/* Shared search-time bucket resolver: nearest-centroid-by-EMD against
 * the per-flop centroid slice for this flop_hash. num_board is 4 (turn)
 * or 5 (river). */
static int bps_hand_to_bucket_search_impl(const BPSReader *r,
                                            const float *centroids,
                                            int centroids_loaded,
                                            uint64_t flop_hash,
                                            const int *board, int num_board,
                                            int c0, int c1) {
    if (!r || !board) return -1;
    if (!centroids || !centroids_loaded) return -1;
    if (r->op1_num_textures <= 0 || r->op1_k <= 0 || r->op1_n_bins <= 0)
        return -1;
    if (!r->textures.loaded) return -1;

    /* Strict: no flop-global fallback. flop_hash=0 means
     * caller didn't pass one; we refuse rather than silently use a stale
     * centroid slice. */
    if (flop_hash == 0ULL) return -1;

    /* Hand must not collide with the board. */
    for (int i = 0; i < num_board; i++) {
        if (board[i] == c0 || board[i] == c1) return -1;
    }

    int tex_idx = tex_index_lookup(&r->textures, flop_hash);
    if (tex_idx < 0) return -1;
    if (tex_idx >= r->op1_num_textures) return -1;

    const float *slice = centroids
        + (size_t)tex_idx * (size_t)r->op1_k * (size_t)r->op1_n_bins;

    /* If this flop's slice was never populated (sharded precompute didn't
     * reach it), reject loudly rather than pick the 0-th centroid by
     * default (which is a hist of zeros). Detect via total mass: a valid
     * slice has ≥ one histogram summing to 1.0. */
    int any_nonzero = 0;
    for (int i = 0; i < r->op1_k && !any_nonzero; i++) {
        float s = 0.0f;
        for (int b = 0; b < r->op1_n_bins; b++) {
            s += slice[i * r->op1_n_bins + b];
            if (s > 1e-6f) { any_nonzero = 1; break; }
        }
    }
    if (!any_nonzero) return -1;

    int hand[2] = {c0, c1};
    int n_bins = r->op1_n_bins;
    float *hist = (float*)malloc((size_t)n_bins * sizeof(float));
    if (!hist) return -1;
    ca_compute_ehs_distribution(board, num_board, hand, 1000, hist, n_bins);
    int bucket = ca_nearest_centroid_emd(hist, slice, r->op1_k, n_bins);
    free(hist);
    return bucket;
}

int bps_hand_to_bucket_turn_search(const BPSReader *r,
                                     uint64_t flop_hash,
                                     const int *board4,
                                     int c0, int c1) {
    return bps_hand_to_bucket_search_impl(
        r, r->op1_turn_centroids, r->op1_turn_loaded,
        flop_hash, board4, 4, c0, c1);
}

int bps_hand_to_bucket_river_search(const BPSReader *r,
                                      uint64_t flop_hash,
                                      const int *board5,
                                      int c0, int c1) {
    return bps_hand_to_bucket_search_impl(
        r, r->op1_river_centroids, r->op1_river_loaded,
        flop_hash, board5, 5, c0, c1);
}
