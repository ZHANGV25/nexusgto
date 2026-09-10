/*
 * test_bps_joint_canon.c - gates for the dense joint river canonicalizer.
 *
 * G1 is artifact-free.  It checks global-suit invariance, dense-rank range
 * and sample injectivity on 4,096 deterministic legal river situations.  It
 * also traverses the complete dense domain and checks
 * rank(enumerate(i)) == i for i < 100,000 plus deterministic later samples.
 * The callback deliberately counts every class but reranks only those
 * selected ranks, keeping the required full-count gate reasonably cheap.
 *
 * G2 is opt-in because it needs a production blueprint and its generated
 * table.  Set BPS_RIVER_BUCKET_G2_BLUEPRINT to the outer .bps path.  The test
 * reads only that file's uncompressed JSON metadata tail: strategy bytes are
 * irrelevant to hand bucketing and need not be decompressed.  Set the
 * optional BPS_RIVER_BUCKET_G2_TABLE to the RBK1 path; otherwise the table is
 * expected at <blueprint>.river_buckets.rbk1.  BPS_RIVER_OCHS_CLUSTERS has
 * the same optional override as the serving path (default 8).
 *
 * Build: make build/test_bps_joint_canon
 * Run G1 sample:      ./build/test_bps_joint_canon --g1
 * Run full enumerator: ./build/test_bps_joint_canon --enumerator
 * Run G2: BPS_RIVER_BUCKET_G2_BLUEPRINT=/path/to/blueprint.bps \
 *           ./build/test_bps_joint_canon --g2
 */
#ifndef _WIN32
#define _FILE_OFFSET_BITS 64
#endif

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "bps_joint_canon.h"
#include "bps_reader.h"

#define G1_RANDOM_CASES 4096
#define ENUM_PREFIX_CHECKS 100000
#define ENUM_RANDOM_CHECKS 2048
#define G2_RANDOM_BOARDS 12

static const int SUIT_PERMS[24][4] = {
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1},
    {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2},
    {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
    {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0},
    {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1},
    {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0},
};

typedef struct {
    int64_t rank;
    unsigned char cards[7];
} RankSample;

typedef struct {
    int64_t visited;
    int64_t reranked;
    int64_t failure_rank;
    int failure_kind;
    const int64_t *targets;
    size_t num_targets;
    size_t next_target;
} EnumContext;

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static void deal_cards(uint64_t *rng, int count, int *out) {
    int deck[52];
    for (int i = 0; i < 52; ++i) deck[i] = i;
    for (int i = 0; i < count; ++i) {
        int j = i + (int)(rng_next(rng) % (uint64_t)(52 - i));
        int tmp = deck[i];
        deck[i] = deck[j];
        deck[j] = tmp;
        out[i] = deck[i];
    }
}

static int permute_card(int card, const int perm[4]) {
    return (card / 4) * 4 + perm[card % 4];
}

static int rank_sample_cmp(const void *lhs_, const void *rhs_) {
    const RankSample *lhs = (const RankSample *)lhs_;
    const RankSample *rhs = (const RankSample *)rhs_;
    if (lhs->rank < rhs->rank) return -1;
    if (lhs->rank > rhs->rank) return 1;
    return memcmp(lhs->cards, rhs->cards, sizeof(lhs->cards));
}

static int int64_cmp(const void *lhs_, const void *rhs_) {
    int64_t lhs = *(const int64_t *)lhs_;
    int64_t rhs = *(const int64_t *)rhs_;
    return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

static int run_g1_random(void) {
    RankSample *samples =
        (RankSample *)malloc((size_t)G1_RANDOM_CASES * sizeof(*samples));
    if (!samples) {
        fprintf(stderr, "G1 FAIL: could not allocate rank samples\n");
        return 1;
    }

    uint64_t rng = UINT64_C(0x6a09e667f3bcc909);
    for (int sample = 0; sample < G1_RANDOM_CASES; ++sample) {
        int dealt[7];
        int board[5];
        int hand[2];
        int canon_board[5];
        int canon_hand[2];
        deal_cards(&rng, 7, dealt);
        memcpy(board, dealt, sizeof(board));
        memcpy(hand, dealt + 5, sizeof(hand));

        if (bps_joint_canon_river(board, hand, canon_board, canon_hand) != 0) {
            fprintf(stderr, "G1 FAIL: canonicalizer rejected sample %d\n", sample);
            free(samples);
            return 1;
        }
        int64_t rank = bps_joint_rank_river(board, hand);
        if (rank < 0 || rank >= (int64_t)BPS_JOINT_RIVER_NUM_CLASSES) {
            fprintf(stderr,
                    "G1 FAIL: sample %d rank=%" PRId64 " outside [0, %" PRIu64 ")\n",
                    sample, rank, (uint64_t)BPS_JOINT_RIVER_NUM_CLASSES);
            free(samples);
            return 1;
        }

        int unrank_board[5];
        int unrank_hand[2];
        if (bps_joint_unrank_river(rank, unrank_board, unrank_hand) != 0 ||
            memcmp(unrank_board, canon_board, sizeof(canon_board)) != 0 ||
            memcmp(unrank_hand, canon_hand, sizeof(canon_hand)) != 0) {
            fprintf(stderr,
                    "G1 FAIL: unrank(rank(x)) disagrees with canonical x at sample %d\n",
                    sample);
            free(samples);
            return 1;
        }

        samples[sample].rank = rank;
        for (int i = 0; i < 5; ++i)
            samples[sample].cards[i] = (unsigned char)canon_board[i];
        for (int i = 0; i < 2; ++i)
            samples[sample].cards[5 + i] = (unsigned char)canon_hand[i];

        for (int pi = 0; pi < 24; ++pi) {
            int image_board[5];
            int image_hand[2];
            int image_canon_board[5];
            int image_canon_hand[2];
            for (int i = 0; i < 5; ++i)
                image_board[i] = permute_card(board[i], SUIT_PERMS[pi]);
            for (int i = 0; i < 2; ++i)
                image_hand[i] = permute_card(hand[i], SUIT_PERMS[pi]);

            if (bps_joint_canon_river(image_board, image_hand,
                                      image_canon_board, image_canon_hand) != 0 ||
                memcmp(image_canon_board, canon_board, sizeof(canon_board)) != 0 ||
                memcmp(image_canon_hand, canon_hand, sizeof(canon_hand)) != 0) {
                fprintf(stderr,
                        "G1 FAIL: canonical form changed at sample %d, suit permutation %d\n",
                        sample, pi);
                free(samples);
                return 1;
            }

            int64_t image_rank = bps_joint_rank_river(image_board, image_hand);
            if (image_rank != rank) {
                fprintf(stderr,
                        "G1 FAIL: rank changed at sample %d, permutation %d "
                        "(%" PRId64 " != %" PRId64 ")\n",
                        sample, pi, image_rank, rank);
                free(samples);
                return 1;
            }
        }
    }

    qsort(samples, G1_RANDOM_CASES, sizeof(*samples), rank_sample_cmp);
    int distinct = 1;
    for (int i = 1; i < G1_RANDOM_CASES; ++i) {
        int same_cards =
            memcmp(samples[i - 1].cards, samples[i].cards,
                   sizeof(samples[i].cards)) == 0;
        if (samples[i - 1].rank == samples[i].rank && !same_cards) {
            fprintf(stderr,
                    "G1 FAIL: distinct canonical samples collide at rank=%" PRId64 "\n",
                    samples[i].rank);
            free(samples);
            return 1;
        }
        if (!same_cards) ++distinct;
    }
    free(samples);

    if (distinct < G1_RANDOM_CASES - 16) {
        fprintf(stderr,
                "G1 FAIL: unexpectedly few distinct classes (%d of %d samples)\n",
                distinct, G1_RANDOM_CASES);
        return 1;
    }

    printf("G1 PASS: %d deterministic legal states, all 24 suit permutations; "
           "%d distinct sampled classes are injective.\n",
           G1_RANDOM_CASES, distinct);
    return 0;
}

static int enum_visitor(int64_t rank, const int board[5], const int hand[2],
                        void *ctx_) {
    EnumContext *ctx = (EnumContext *)ctx_;
    if (rank != ctx->visited) {
        ctx->failure_kind = 1;
        ctx->failure_rank = rank;
        return 1;
    }

    int selected = rank < ENUM_PREFIX_CHECKS;
    if (ctx->next_target < ctx->num_targets &&
        ctx->targets[ctx->next_target] == rank) {
        selected = 1;
        ++ctx->next_target;
    }
    if (selected) {
        int64_t reranked = bps_joint_rank_river(board, hand);
        if (reranked != rank) {
            ctx->failure_kind = 2;
            ctx->failure_rank = rank;
            return 1;
        }
        ++ctx->reranked;
    }

    ++ctx->visited;
    return 0;
}

static int run_g1_enumerator(void) {
    int64_t targets[ENUM_RANDOM_CHECKS];
    const int64_t n = (int64_t)BPS_JOINT_RIVER_NUM_CLASSES;
    uint64_t rng = UINT64_C(0xbb67ae8584caa73b);
    for (int i = 0; i < ENUM_RANDOM_CHECKS; ++i) {
        targets[i] = ENUM_PREFIX_CHECKS +
            (int64_t)(rng_next(&rng) % (uint64_t)(n - ENUM_PREFIX_CHECKS));
    }
    qsort(targets, ENUM_RANDOM_CHECKS, sizeof(targets[0]), int64_cmp);

    size_t unique_targets = 0;
    for (size_t i = 0; i < ENUM_RANDOM_CHECKS; ++i) {
        if (unique_targets == 0 || targets[i] != targets[unique_targets - 1])
            targets[unique_targets++] = targets[i];
    }

    EnumContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.failure_rank = -1;
    ctx.targets = targets;
    ctx.num_targets = unique_targets;

    printf("G1 enumerator: visiting all %" PRId64
           " classes (reranking %d prefix + %zu later samples)...\n",
           n, ENUM_PREFIX_CHECKS, unique_targets);
    fflush(stdout);
    int64_t returned = bps_joint_enumerate_river(0, n, enum_visitor, &ctx);
    if (returned != n || ctx.visited != n || ctx.failure_kind != 0) {
        fprintf(stderr,
                "G1 FAIL: enumerator returned=%" PRId64 ", visited=%" PRId64
                ", failure_kind=%d, failure_rank=%" PRId64 " (expected %" PRId64 ")\n",
                returned, ctx.visited, ctx.failure_kind, ctx.failure_rank, n);
        return 1;
    }
    if (ctx.next_target != unique_targets ||
        ctx.reranked != ENUM_PREFIX_CHECKS + (int64_t)unique_targets) {
        fprintf(stderr,
                "G1 FAIL: selected-rank coverage mismatch "
                "(later=%zu/%zu, reranked=%" PRId64 ")\n",
                ctx.next_target, unique_targets, ctx.reranked);
        return 1;
    }

    printf("G1 ENUMERATOR PASS: exactly %" PRId64
           " classes in dense-rank order; reranked %" PRId64 " representatives.\n",
           ctx.visited, ctx.reranked);
    return 0;
}

static int read_u32_le(FILE *f, uint32_t *out) {
    unsigned char b[4];
    if (fread(b, 1, sizeof(b), f) != sizeof(b)) return -1;
    *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}

static int read_u64_le(FILE *f, uint64_t *out) {
    unsigned char b[8];
    if (fread(b, 1, sizeof(b), f) != sizeof(b)) return -1;
    *out = (uint64_t)b[0] | ((uint64_t)b[1] << 8) |
           ((uint64_t)b[2] << 16) | ((uint64_t)b[3] << 24) |
           ((uint64_t)b[4] << 32) | ((uint64_t)b[5] << 40) |
           ((uint64_t)b[6] << 48) | ((uint64_t)b[7] << 56);
    return 0;
}

static int seek_file_u64(FILE *f, uint64_t offset) {
#ifdef _WIN32
    if (offset > (uint64_t)INT64_MAX) return -1;
    return _fseeki64(f, (__int64)offset, SEEK_SET);
#else
    if (offset > (uint64_t)INT64_MAX) return -1;
    return fseeko(f, (off_t)offset, SEEK_SET);
#endif
}

static int json_positive_int(const char *json, const char *key, int fallback,
                             int maximum, int *out) {
    const char *at = strstr(json, key);
    if (!at) {
        *out = fallback;
        return 0;
    }
    at = strchr(at + strlen(key), ':');
    if (!at) return -1;
    ++at;
    while (isspace((unsigned char)*at)) ++at;
    errno = 0;
    char *end = NULL;
    long value = strtol(at, &end, 10);
    if (errno != 0 || end == at || value <= 0 || value > maximum) return -1;
    *out = (int)value;
    return 0;
}

static int blueprint_postflop_buckets(const char *path, int *out_buckets) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "G2 FAIL: cannot open blueprint %s\n", path);
        return -1;
    }

    char magic[4];
    uint64_t compressed_size = 0;
    uint32_t metadata_size = 0;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        memcmp(magic, "BPS3", 4) != 0 ||
        read_u64_le(f, &compressed_size) != 0 ||
        read_u32_le(f, &metadata_size) != 0) {
        fprintf(stderr, "G2 FAIL: %s is not a valid outer BPS3 file\n", path);
        fclose(f);
        return -1;
    }
    if (metadata_size == 0 || metadata_size > 16U * 1024U * 1024U ||
        compressed_size > UINT64_MAX - UINT64_C(16) ||
        seek_file_u64(f, UINT64_C(16) + compressed_size) != 0) {
        fprintf(stderr, "G2 FAIL: invalid or unreachable BPS3 metadata in %s\n", path);
        fclose(f);
        return -1;
    }

    char *metadata = (char *)malloc((size_t)metadata_size + 1U);
    if (!metadata) {
        fprintf(stderr, "G2 FAIL: could not allocate blueprint metadata\n");
        fclose(f);
        return -1;
    }
    if (fread(metadata, 1, metadata_size, f) != metadata_size) {
        fprintf(stderr, "G2 FAIL: truncated BPS3 metadata in %s\n", path);
        free(metadata);
        fclose(f);
        return -1;
    }
    metadata[metadata_size] = '\0';
    fclose(f);

    /* Missing-key fallback deliberately matches BPSReaderC._load_full(). */
    int rc = json_positive_int(metadata, "\"postflop_buckets\"", 200, 32767,
                               out_buckets);
    free(metadata);
    if (rc != 0)
        fprintf(stderr, "G2 FAIL: invalid postflop_buckets metadata in %s\n", path);
    return rc;
}

static int g2_cluster_count(void) {
    const char *value = getenv("BPS_RIVER_OCHS_CLUSTERS");
    if (!value || !value[0]) return 8;
    errno = 0;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed <= 0 || parsed > 16)
        return -1;
    return (int)parsed;
}

static char *g2_table_path(const char *blueprint) {
    const char *override = getenv("BPS_RIVER_BUCKET_G2_TABLE");
    if (override && override[0]) {
        size_t n = strlen(override) + 1U;
        char *copy = (char *)malloc(n);
        if (copy) memcpy(copy, override, n);
        return copy;
    }

    static const char suffix[] = ".river_buckets.rbk1";
    size_t n = strlen(blueprint);
    if (n > SIZE_MAX - sizeof(suffix)) return NULL;
    char *path = (char *)malloc(n + sizeof(suffix));
    if (!path) return NULL;
    memcpy(path, blueprint, n);
    memcpy(path + n, suffix, sizeof(suffix));
    return path;
}

static int clone_ochs_reader(const BPSReader *source, BPSReader *dest) {
    memset(dest, 0, sizeof(*dest));
    dest->num_postflop_buckets = source->num_postflop_buckets;
    dest->num_preflop_buckets = source->num_preflop_buckets;
    dest->ochs_river_enabled = source->ochs_river_enabled;
    dest->ochs_num_clusters = source->ochs_num_clusters;
    dest->ochs_river_centroids_k = source->ochs_river_centroids_k;
    memcpy(dest->ochs_cluster_of_169, source->ochs_cluster_of_169,
           sizeof(dest->ochs_cluster_of_169));

    size_t floats = (size_t)source->ochs_num_clusters *
                    (size_t)source->ochs_river_centroids_k;
    if (floats == 0 || !source->ochs_river_centroids) return -1;
    dest->ochs_river_centroids = (float *)malloc(floats * sizeof(float));
    if (!dest->ochs_river_centroids) return -1;
    memcpy(dest->ochs_river_centroids, source->ochs_river_centroids,
           floats * sizeof(float));
    return 0;
}

static int run_g2(void) {
    const char *blueprint = getenv("BPS_RIVER_BUCKET_G2_BLUEPRINT");
    if (!blueprint || !blueprint[0]) {
        printf("G2 SKIP: set BPS_RIVER_BUCKET_G2_BLUEPRINT to an outer .bps path.\n");
        return 0;
    }

    int buckets = 0;
    int clusters = g2_cluster_count();
    if (clusters < 0 || blueprint_postflop_buckets(blueprint, &buckets) != 0) {
        if (clusters < 0)
            fprintf(stderr, "G2 FAIL: invalid BPS_RIVER_OCHS_CLUSTERS\n");
        return 1;
    }
    char *table_path = g2_table_path(blueprint);
    if (!table_path) {
        fprintf(stderr, "G2 FAIL: could not construct RBK1 table path\n");
        return 1;
    }

    BPSReader live_reader;
    BPSReader table_reader;
    memset(&live_reader, 0, sizeof(live_reader));
    memset(&table_reader, 0, sizeof(table_reader));
    live_reader.num_postflop_buckets = buckets;
    live_reader.num_preflop_buckets = 169;

    if (bps_init_ochs_river(&live_reader, clusters) != 0 ||
        !live_reader.ochs_river_enabled) {
        fprintf(stderr, "G2 FAIL: could not initialize live OCHS reader\n");
        free(table_path);
        bps_reader_free(&live_reader);
        return 1;
    }
    if (clone_ochs_reader(&live_reader, &table_reader) != 0) {
        fprintf(stderr, "G2 FAIL: could not clone live OCHS identity\n");
        free(table_path);
        bps_reader_free(&live_reader);
        bps_reader_free(&table_reader);
        return 1;
    }

    int loaded = bps_load_river_bucket_table(&table_reader, table_path);
    if (loaded != (int)BPS_JOINT_RIVER_NUM_CLASSES ||
        !bps_has_river_bucket_table(&table_reader)) {
        fprintf(stderr,
                "G2 FAIL: RBK1 did not validate/load from %s (rc=%d)\n",
                table_path, loaded);
        free(table_path);
        bps_reader_free(&live_reader);
        bps_reader_free(&table_reader);
        return 1;
    }

    uint64_t rng = UINT64_C(0x3c6ef372fe94f82b);
    int64_t comparisons = 0;
    for (int bi = 0; bi < G2_RANDOM_BOARDS; ++bi) {
        int board[5];
        int blocked[52] = {0};
        deal_cards(&rng, 5, board);
        for (int i = 0; i < 5; ++i) blocked[board[i]] = 1;

        for (int c0 = 0; c0 < 52; ++c0) {
            if (blocked[c0]) continue;
            for (int c1 = c0 + 1; c1 < 52; ++c1) {
                if (blocked[c1]) continue;
                int hand[2] = {c0, c1};
                int64_t rank = bps_joint_rank_river(board, hand);
                if (rank < 0 || rank >= table_reader.river_bucket_table_count ||
                    table_reader.river_bucket_table[rank] == UINT16_MAX) {
                    fprintf(stderr,
                            "G2 FAIL: board %d hand %d/%d has absent table rank=%" PRId64 "\n",
                            bi, c0, c1, rank);
                    free(table_path);
                    bps_reader_free(&live_reader);
                    bps_reader_free(&table_reader);
                    return 1;
                }

                int live = bps_hand_to_bucket_river(&live_reader, board, c0, c1);
                int table = bps_hand_to_bucket_river(&table_reader, board, c0, c1);
                if (live < 0 || table != live ||
                    table != (int)table_reader.river_bucket_table[rank]) {
                    fprintf(stderr,
                            "G2 FAIL: board %d hand %d/%d rank=%" PRId64
                            " table=%d live=%d stored=%u\n",
                            bi, c0, c1, rank, table, live,
                            (unsigned)table_reader.river_bucket_table[rank]);
                    free(table_path);
                    bps_reader_free(&live_reader);
                    bps_reader_free(&table_reader);
                    return 1;
                }
                ++comparisons;
            }
        }
    }

    printf("G2 PASS: validated %" PRId64
           " table/live lookups (%d boards x every legal hand) using %s.\n",
           comparisons, G2_RANDOM_BOARDS, table_path);
    free(table_path);
    bps_reader_free(&live_reader);
    bps_reader_free(&table_reader);
    return 0;
}

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s [--g1] [--enumerator] [--g2]\n"
            "With no flags, runs all gates (G2 skips when its env var is unset).\n",
            program);
}

int main(int argc, char **argv) {
    enum { RUN_G1 = 1, RUN_ENUMERATOR = 2, RUN_G2 = 4 };
    int selected = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--g1") == 0) selected |= RUN_G1;
        else if (strcmp(argv[i], "--enumerator") == 0) selected |= RUN_ENUMERATOR;
        else if (strcmp(argv[i], "--g2") == 0) selected |= RUN_G2;
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (selected == 0) selected = RUN_G1 | RUN_ENUMERATOR | RUN_G2;

    if ((selected & RUN_G1) && run_g1_random() != 0) return 1;
    if ((selected & RUN_ENUMERATOR) && run_g1_enumerator() != 0) return 1;
    if ((selected & RUN_G2) && run_g2() != 0) return 1;
    return 0;
}
