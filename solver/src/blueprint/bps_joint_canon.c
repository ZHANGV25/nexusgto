/**
 * bps_joint_canon.c - exact joint river canonicalizer and dense orbit rank.
 *
 * See bps_joint_canon.h for the pinned trit-column order.  Lookup paths use
 * only fixed-size stack storage plus precomputed process-global DP tables; no
 * function here allocates memory.  Initialization is published atomically so
 * concurrent first callers never observe a partially filled table.
 */
#include "bps_joint_canon.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define BJC_RANKS 13
#define BJC_SUITS 4
#define BJC_BOARD_CARDS 5
#define BJC_HAND_CARDS 2
#define BJC_ROW_CODES 81       /* 3^4 row tuples. */
#define BJC_TIE_STATES 8       /* One bit per adjacent column pair. */
#define BJC_ALL_TIED 7
#define BJC_BAD_TIE 255

typedef struct {
    uint8_t digit[BJC_RANKS];
} BJCColumn;

/* dp[row][board_left][hand_left][tie_mask] is the number of legal suffixes.
 * rank_skip has the sum of subtree sizes for legal row codes strictly less
 * than its final index, turning a hot-path rank into thirteen prefix reads. */
static uint64_t g_dp[BJC_RANKS + 1][BJC_BOARD_CARDS + 1]
                    [BJC_HAND_CARDS + 1][BJC_TIE_STATES];
static uint64_t g_rank_skip[BJC_RANKS][BJC_BOARD_CARDS + 1]
                           [BJC_HAND_CARDS + 1][BJC_TIE_STATES]
                           [BJC_ROW_CODES];
static uint8_t g_row_digit[BJC_ROW_CODES][BJC_SUITS];
static uint8_t g_row_board_count[BJC_ROW_CODES];
static uint8_t g_row_hand_count[BJC_ROW_CODES];
static uint8_t g_next_tie[BJC_TIE_STATES][BJC_ROW_CODES];

/* 0 = uninitialized, 1 = one thread building, 2 = ready, 3 = invariant
 * failure.  GCC/Clang atomics are already the repository's portability
 * convention (including MinGW builds). */
static int g_init_state = 0;

static int bjc_validate_cards(const int board[5], const int hand[2]) {
    uint64_t used = UINT64_C(0);
    int i;

    if (!board || !hand) return -1;
    for (i = 0; i < BJC_BOARD_CARDS; ++i) {
        int card = board[i];
        uint64_t bit;
        if (card < 0 || card >= 52) return -1;
        bit = UINT64_C(1) << (unsigned)card;
        if (used & bit) return -1;
        used |= bit;
    }
    for (i = 0; i < BJC_HAND_CARDS; ++i) {
        int card = hand[i];
        uint64_t bit;
        if (card < 0 || card >= 52) return -1;
        bit = UINT64_C(1) << (unsigned)card;
        if (used & bit) return -1;
        used |= bit;
    }
    return 0;
}

static void bjc_sort_ints(int *values, int count) {
    int i;
    for (i = 1; i < count; ++i) {
        int value = values[i];
        int j = i;
        while (j > 0 && values[j - 1] > value) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = value;
    }
}

static int bjc_encoding_less(const int board_a[5], const int hand_a[2],
                             const int board_b[5], const int hand_b[2]) {
    int i;
    for (i = 0; i < BJC_BOARD_CARDS; ++i) {
        if (board_a[i] != board_b[i]) return board_a[i] < board_b[i];
    }
    for (i = 0; i < BJC_HAND_CARDS; ++i) {
        if (hand_a[i] != hand_b[i]) return hand_a[i] < hand_b[i];
    }
    return 0;
}

int bps_joint_canon_river(const int board[5], const int hand[2],
                          int canon_board[5], int canon_hand[2]) {
    /* Lexicographic list of all S4 elements.  Keeping the explicit 24-entry
     * set makes the canonicalizer's exhaustive contract visible in code. */
    static const uint8_t permutations[24][4] = {
        {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1},
        {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2},
        {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
        {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0},
        {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1},
        {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}
    };
    int best_board[BJC_BOARD_CARDS];
    int best_hand[BJC_HAND_CARDS];
    int image_board[BJC_BOARD_CARDS];
    int image_hand[BJC_HAND_CARDS];
    int have_best = 0;
    int p, i;

    if (!canon_board || !canon_hand || bjc_validate_cards(board, hand) != 0)
        return -1;

    for (p = 0; p < 24; ++p) {
        for (i = 0; i < BJC_BOARD_CARDS; ++i) {
            int card = board[i];
            image_board[i] = (card / 4) * 4 + permutations[p][card % 4];
        }
        for (i = 0; i < BJC_HAND_CARDS; ++i) {
            int card = hand[i];
            image_hand[i] = (card / 4) * 4 + permutations[p][card % 4];
        }
        bjc_sort_ints(image_board, BJC_BOARD_CARDS);
        bjc_sort_ints(image_hand, BJC_HAND_CARDS);
        if (!have_best || bjc_encoding_less(image_board, image_hand,
                                            best_board, best_hand)) {
            memcpy(best_board, image_board, sizeof(best_board));
            memcpy(best_hand, image_hand, sizeof(best_hand));
            have_best = 1;
        }
    }

    memcpy(canon_board, best_board, sizeof(best_board));
    memcpy(canon_hand, best_hand, sizeof(best_hand));
    return 0;
}

static int bjc_column_compare(const BJCColumn *a, const BJCColumn *b) {
    int rank;
    for (rank = 0; rank < BJC_RANKS; ++rank) {
        if (a->digit[rank] != b->digit[rank])
            return (int)a->digit[rank] - (int)b->digit[rank];
    }
    return 0;
}

static int bjc_cards_to_sorted_columns(const int board[5], const int hand[2],
                                       BJCColumn columns[4]) {
    int i;

    if (bjc_validate_cards(board, hand) != 0) return -1;
    memset(columns, 0, (size_t)BJC_SUITS * sizeof(*columns));
    for (i = 0; i < BJC_BOARD_CARDS; ++i)
        columns[board[i] % 4].digit[board[i] / 4] = 1;
    for (i = 0; i < BJC_HAND_CARDS; ++i)
        columns[hand[i] % 4].digit[hand[i] / 4] = 2;

    /* Four-element insertion sort; equal suit signatures are intentionally
     * retained as duplicate multiset elements. */
    for (i = 1; i < BJC_SUITS; ++i) {
        BJCColumn value = columns[i];
        int j = i;
        while (j > 0 && bjc_column_compare(&columns[j - 1], &value) > 0) {
            columns[j] = columns[j - 1];
            --j;
        }
        columns[j] = value;
    }
    return 0;
}

static int bjc_build_tables(void) {
    int code, tie, row, board_left, hand_left;

    memset(g_dp, 0, sizeof(g_dp));
    memset(g_rank_skip, 0, sizeof(g_rank_skip));

    /* Numeric base-3 code order is tuple lexicographic order because column
     * zero is the most-significant digit. */
    for (code = 0; code < BJC_ROW_CODES; ++code) {
        int rest = code;
        int col;
        uint8_t bc = 0, hc = 0;
        for (col = BJC_SUITS - 1; col >= 0; --col) {
            uint8_t digit = (uint8_t)(rest % 3);
            rest /= 3;
            g_row_digit[code][col] = digit;
            if (digit == 1) ++bc;
            if (digit == 2) ++hc;
        }
        g_row_board_count[code] = bc;
        g_row_hand_count[code] = hc;
    }

    for (tie = 0; tie < BJC_TIE_STATES; ++tie) {
        for (code = 0; code < BJC_ROW_CODES; ++code) {
            int col;
            int legal = 1;
            int next = 0;
            for (col = 0; col < BJC_SUITS - 1; ++col) {
                int bit = 1 << col;
                if ((tie & bit) &&
                    g_row_digit[code][col] > g_row_digit[code][col + 1]) {
                    legal = 0;
                    break;
                }
                if ((tie & bit) &&
                    g_row_digit[code][col] == g_row_digit[code][col + 1])
                    next |= bit;
            }
            g_next_tie[tie][code] = legal ? (uint8_t)next : BJC_BAD_TIE;
        }
    }

    /* Once all thirteen rows are consumed, every valid tie pattern has one
     * empty completion precisely when both required-card counts are zero. */
    for (tie = 0; tie < BJC_TIE_STATES; ++tie)
        g_dp[BJC_RANKS][0][0][tie] = UINT64_C(1);

    for (row = BJC_RANKS - 1; row >= 0; --row) {
        for (board_left = 0; board_left <= BJC_BOARD_CARDS; ++board_left) {
            for (hand_left = 0; hand_left <= BJC_HAND_CARDS; ++hand_left) {
                for (tie = 0; tie < BJC_TIE_STATES; ++tie) {
                    uint64_t total = UINT64_C(0);
                    for (code = 0; code < BJC_ROW_CODES; ++code) {
                        uint8_t next = g_next_tie[tie][code];
                        int bc = g_row_board_count[code];
                        int hc = g_row_hand_count[code];
                        if (next == BJC_BAD_TIE || bc > board_left ||
                            hc > hand_left)
                            continue;
                        total += g_dp[row + 1][board_left - bc]
                                     [hand_left - hc][next];
                    }
                    g_dp[row][board_left][hand_left][tie] = total;
                }
            }
        }
    }

    /* This is a load-bearing explicit raise-equivalent, not an assert: builds
     * with NDEBUG or -DNDEBUG must still refuse a changed ranking language. */
    if (g_dp[0][BJC_BOARD_CARDS][BJC_HAND_CARDS][BJC_ALL_TIED] !=
        BPS_JOINT_RIVER_NUM_CLASSES)
        return -1;

    for (row = 0; row < BJC_RANKS; ++row) {
        for (board_left = 0; board_left <= BJC_BOARD_CARDS; ++board_left) {
            for (hand_left = 0; hand_left <= BJC_HAND_CARDS; ++hand_left) {
                for (tie = 0; tie < BJC_TIE_STATES; ++tie) {
                    uint64_t prefix = UINT64_C(0);
                    for (code = 0; code < BJC_ROW_CODES; ++code) {
                        uint8_t next = g_next_tie[tie][code];
                        int bc = g_row_board_count[code];
                        int hc = g_row_hand_count[code];
                        g_rank_skip[row][board_left][hand_left][tie][code] =
                            prefix;
                        if (next == BJC_BAD_TIE || bc > board_left ||
                            hc > hand_left)
                            continue;
                        prefix += g_dp[row + 1][board_left - bc]
                                      [hand_left - hc][next];
                    }
                    if (prefix != g_dp[row][board_left][hand_left][tie])
                        return -1;
                }
            }
        }
    }
    return 0;
}

static int bjc_ensure_tables(void) {
    for (;;) {
        int state = __atomic_load_n(&g_init_state, __ATOMIC_ACQUIRE);
        if (state == 2) return 0;
        if (state == 3) return -1;
        if (state == 0) {
            int expected = 0;
            if (__atomic_compare_exchange_n(&g_init_state, &expected, 1, 0,
                                            __ATOMIC_ACQ_REL,
                                            __ATOMIC_ACQUIRE)) {
                int ok = bjc_build_tables() == 0;
                __atomic_store_n(&g_init_state, ok ? 2 : 3,
                                 __ATOMIC_RELEASE);
                return ok ? 0 : -1;
            }
        }
        /* A competing first caller is filling a bounded ~1.3 MiB table.  It
         * publishes state 2/3 with release semantics; do not read early. */
    }
}

int64_t bps_joint_rank_river(const int board[5], const int hand[2]) {
    BJCColumn columns[BJC_SUITS];
    uint64_t ordinal = UINT64_C(0);
    int board_left = BJC_BOARD_CARDS;
    int hand_left = BJC_HAND_CARDS;
    int tie = BJC_ALL_TIED;
    int row;

    if (bjc_cards_to_sorted_columns(board, hand, columns) != 0 ||
        bjc_ensure_tables() != 0)
        return -1;

    for (row = 0; row < BJC_RANKS; ++row) {
        int code = 0;
        int col;
        int bc, hc;
        uint8_t next;
        for (col = 0; col < BJC_SUITS; ++col)
            code = code * 3 + columns[col].digit[row];
        bc = g_row_board_count[code];
        hc = g_row_hand_count[code];
        next = g_next_tie[tie][code];
        if (next == BJC_BAD_TIE || bc > board_left || hc > hand_left)
            return -1;
        if (g_dp[row + 1][board_left - bc][hand_left - hc][next] == 0)
            return -1;
        ordinal += g_rank_skip[row][board_left][hand_left][tie][code];
        board_left -= bc;
        hand_left -= hc;
        tie = next;
    }

    if (board_left != 0 || hand_left != 0 ||
        ordinal >= BPS_JOINT_RIVER_NUM_CLASSES)
        return -1;
    return (int64_t)ordinal;
}

/* Compare columns for assignment to increasing physical suit numbers in the
 * explicit lex-canonical representative.  At the first board rank where two
 * columns differ, assigning the board-present column the lower suit produces
 * the smaller sorted board card.  If all board membership bits tie, the same
 * argument applies to hand bits.  Therefore descending board-bit vectors,
 * then descending hand-bit vectors, is exactly the 24-permutation minimum. */
static int bjc_canonical_suit_before(const BJCColumn *a, const BJCColumn *b) {
    int rank;
    for (rank = 0; rank < BJC_RANKS; ++rank) {
        int a_has = a->digit[rank] == 1;
        int b_has = b->digit[rank] == 1;
        if (a_has != b_has) return a_has > b_has;
    }
    for (rank = 0; rank < BJC_RANKS; ++rank) {
        int a_has = a->digit[rank] == 2;
        int b_has = b->digit[rank] == 2;
        if (a_has != b_has) return a_has > b_has;
    }
    return 0;
}

static int bjc_columns_to_canonical(const BJCColumn columns[4],
                                    int board[5], int hand[2]) {
    int order[BJC_SUITS] = {0, 1, 2, 3};
    int board_count = 0, hand_count = 0;
    int i, rank, suit;

    for (i = 1; i < BJC_SUITS; ++i) {
        int value = order[i];
        int j = i;
        while (j > 0 &&
               bjc_canonical_suit_before(&columns[value],
                                         &columns[order[j - 1]])) {
            order[j] = order[j - 1];
            --j;
        }
        order[j] = value;
    }

    /* rank-major, then increasing assigned suit, is already ascending card-id
     * order, so no post-construction sort is needed. */
    for (rank = 0; rank < BJC_RANKS; ++rank) {
        for (suit = 0; suit < BJC_SUITS; ++suit) {
            uint8_t digit = columns[order[suit]].digit[rank];
            if (digit == 1) {
                if (board_count >= BJC_BOARD_CARDS) return -1;
                board[board_count++] = rank * 4 + suit;
            } else if (digit == 2) {
                if (hand_count >= BJC_HAND_CARDS) return -1;
                hand[hand_count++] = rank * 4 + suit;
            }
        }
    }
    return board_count == BJC_BOARD_CARDS && hand_count == BJC_HAND_CARDS
               ? 0
               : -1;
}

int bps_joint_unrank_river(int64_t rank, int board[5], int hand[2]) {
    BJCColumn columns[BJC_SUITS];
    uint64_t offset;
    int board_left = BJC_BOARD_CARDS;
    int hand_left = BJC_HAND_CARDS;
    int tie = BJC_ALL_TIED;
    int row;

    if (!board || !hand || rank < 0 ||
        (uint64_t)rank >= BPS_JOINT_RIVER_NUM_CLASSES ||
        bjc_ensure_tables() != 0)
        return -1;

    memset(columns, 0, sizeof(columns));
    offset = (uint64_t)rank;
    for (row = 0; row < BJC_RANKS; ++row) {
        int code;
        int selected = -1;
        for (code = 0; code < BJC_ROW_CODES; ++code) {
            uint8_t next = g_next_tie[tie][code];
            int bc = g_row_board_count[code];
            int hc = g_row_hand_count[code];
            uint64_t count;
            int col;
            if (next == BJC_BAD_TIE || bc > board_left || hc > hand_left)
                continue;
            count = g_dp[row + 1][board_left - bc][hand_left - hc][next];
            if (count == 0) continue;
            if (offset >= count) {
                offset -= count;
                continue;
            }
            for (col = 0; col < BJC_SUITS; ++col)
                columns[col].digit[row] = g_row_digit[code][col];
            board_left -= bc;
            hand_left -= hc;
            tie = next;
            selected = code;
            break;
        }
        if (selected < 0) return -1;
    }

    if (offset != 0 || board_left != 0 || hand_left != 0)
        return -1;
    return bjc_columns_to_canonical(columns, board, hand);
}

typedef struct {
    uint64_t begin;
    uint64_t end;
    int64_t visited;
    int failed;
    BPSJointRiverVisitor visitor;
    void *visitor_ctx;
    BJCColumn columns[BJC_SUITS];
} BJCEnumerateContext;

static void bjc_enumerate_node(BJCEnumerateContext *ctx,
                               int row, int board_left, int hand_left,
                               int tie, uint64_t base) {
    uint64_t node_count;
    uint64_t cursor;
    int code;

    if (ctx->failed) return;
    node_count = g_dp[row][board_left][hand_left][tie];
    if (node_count == 0 || base >= ctx->end ||
        base + node_count <= ctx->begin)
        return;

    if (row == BJC_RANKS) {
        int board[BJC_BOARD_CARDS];
        int hand[BJC_HAND_CARDS];
        if (board_left != 0 || hand_left != 0 ||
            bjc_columns_to_canonical(ctx->columns, board, hand) != 0 ||
            ctx->visitor((int64_t)base, board, hand,
                         ctx->visitor_ctx) != 0) {
            ctx->failed = 1;
            return;
        }
        ++ctx->visited;
        return;
    }

    cursor = base;
    for (code = 0; code < BJC_ROW_CODES && !ctx->failed; ++code) {
        uint8_t next = g_next_tie[tie][code];
        int bc = g_row_board_count[code];
        int hc = g_row_hand_count[code];
        uint64_t child_count;
        int col;
        if (next == BJC_BAD_TIE || bc > board_left || hc > hand_left)
            continue;
        child_count =
            g_dp[row + 1][board_left - bc][hand_left - hc][next];
        if (child_count == 0) continue;
        if (cursor < ctx->end && cursor + child_count > ctx->begin) {
            for (col = 0; col < BJC_SUITS; ++col)
                ctx->columns[col].digit[row] = g_row_digit[code][col];
            bjc_enumerate_node(ctx, row + 1, board_left - bc,
                               hand_left - hc, next, cursor);
        }
        cursor += child_count;
    }
}

int64_t bps_joint_enumerate_river(int64_t begin, int64_t end,
                                  BPSJointRiverVisitor visitor, void *ctx) {
    BJCEnumerateContext enumerate_ctx;

    if (!visitor || begin < 0 || end < begin ||
        (uint64_t)end > BPS_JOINT_RIVER_NUM_CLASSES ||
        bjc_ensure_tables() != 0)
        return -1;

    memset(&enumerate_ctx, 0, sizeof(enumerate_ctx));
    enumerate_ctx.begin = (uint64_t)begin;
    enumerate_ctx.end = (uint64_t)end;
    enumerate_ctx.visitor = visitor;
    enumerate_ctx.visitor_ctx = ctx;
    bjc_enumerate_node(&enumerate_ctx, 0, BJC_BOARD_CARDS,
                       BJC_HAND_CARDS, BJC_ALL_TIED, UINT64_C(0));

    if (enumerate_ctx.failed || enumerate_ctx.visited != end - begin)
        return -1;
    return enumerate_ctx.visited;
}
