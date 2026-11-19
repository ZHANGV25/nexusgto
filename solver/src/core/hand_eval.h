/**
 * hand_eval.h — Fast 7-card poker hand evaluation
 *
 * Uses a lookup-table approach for 5-card evaluation with
 * exhaustive 7-choose-5 = 21 combinations to find the best hand.
 * Hand strengths are represented as 32-bit integers where higher = better.
 *
 * Encoding: category (4 bits) << 20 | kicker info (20 bits)
 * Categories: 1=high card, 2=pair, 3=two pair, 4=trips, 5=straight,
 *             6=flush, 7=full house, 8=quads, 9=straight flush
 */
#ifndef HAND_EVAL_H
#define HAND_EVAL_H

#include <stdint.h>

/* Card representation: 0-51
 * card = rank * 4 + suit
 * rank: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
 * suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
 */
#define CARD_RANK(c) ((c) >> 2)
#define CARD_SUIT(c) ((c) & 3)
#define MAKE_CARD(rank, suit) (((rank) << 2) | (suit))

/* Hand categories */
#define HC_HIGH_CARD     1
#define HC_PAIR          2
#define HC_TWO_PAIR      3
#define HC_TRIPS         4
#define HC_STRAIGHT      5
#define HC_FLUSH         6
#define HC_FULL_HOUSE    7
#define HC_QUADS         8
#define HC_STRAIGHT_FLUSH 9

/**
 * Evaluate 5 cards. Returns a 32-bit strength value (higher = better).
 *
 * eval5_compute is the reference implementation; eval5 (below) dispatches to
 * the init-time-generated lookup table when POKER_EVAL5_TABLE is on, using
 * eval5_compute both to fill the table and as the exact fallback.
 */
static inline uint32_t eval5_compute(int c0, int c1, int c2, int c3, int c4) {
    int ranks[5] = {CARD_RANK(c0), CARD_RANK(c1), CARD_RANK(c2),
                    CARD_RANK(c3), CARD_RANK(c4)};
    int suits[5] = {CARD_SUIT(c0), CARD_SUIT(c1), CARD_SUIT(c2),
                    CARD_SUIT(c3), CARD_SUIT(c4)};

    /* Sort ranks descending (insertion sort on 5 elements) */
    for (int i = 1; i < 5; i++) {
        int key = ranks[i];
        int j = i - 1;
        while (j >= 0 && ranks[j] < key) {
            ranks[j + 1] = ranks[j];
            j--;
        }
        ranks[j + 1] = key;
    }

    /* Check flush */
    int is_flush = (suits[0] == suits[1] && suits[1] == suits[2] &&
                    suits[2] == suits[3] && suits[3] == suits[4]);

    /* Check straight (sorted descending) */
    int is_straight = 0;
    int straight_high = ranks[0];
    if (ranks[0] - ranks[4] == 4 &&
        ranks[0] != ranks[1] && ranks[1] != ranks[2] &&
        ranks[2] != ranks[3] && ranks[3] != ranks[4]) {
        is_straight = 1;
    }
    /* Ace-low straight: A 5 4 3 2 → ranks = [12, 3, 2, 1, 0] */
    if (ranks[0] == 12 && ranks[1] == 3 && ranks[2] == 2 &&
        ranks[3] == 1 && ranks[4] == 0) {
        is_straight = 1;
        straight_high = 3; /* 5-high straight */
    }

    if (is_straight && is_flush)
        return (HC_STRAIGHT_FLUSH << 20) | (straight_high << 16);
    if (is_flush)
        return (HC_FLUSH << 20) | (ranks[0] << 16) | (ranks[1] << 12) |
               (ranks[2] << 8) | (ranks[3] << 4) | ranks[4];
    if (is_straight)
        return (HC_STRAIGHT << 20) | (straight_high << 16);

    /* Count rank occurrences */
    int counts[13] = {0};
    for (int i = 0; i < 5; i++) counts[ranks[i]]++;

    int quads = -1, trips = -1, pair1 = -1, pair2 = -1;
    /* Scan high to low */
    for (int r = 12; r >= 0; r--) {
        if (counts[r] == 4) quads = r;
        else if (counts[r] == 3) trips = r;
        else if (counts[r] == 2) {
            if (pair1 < 0) pair1 = r;
            else pair2 = r;
        }
    }

    if (quads >= 0) {
        int kicker = -1;
        for (int r = 12; r >= 0; r--)
            if (counts[r] > 0 && r != quads) { kicker = r; break; }
        return (HC_QUADS << 20) | (quads << 16) | (kicker << 12);
    }
    if (trips >= 0 && pair1 >= 0)
        return (HC_FULL_HOUSE << 20) | (trips << 16) | (pair1 << 12);
    if (trips >= 0) {
        int k0 = -1, k1 = -1;
        for (int r = 12; r >= 0; r--) {
            if (counts[r] > 0 && r != trips) {
                if (k0 < 0) k0 = r; else k1 = r;
            }
        }
        return (HC_TRIPS << 20) | (trips << 16) | (k0 << 12) | (k1 << 8);
    }
    if (pair1 >= 0 && pair2 >= 0) {
        int kicker = -1;
        for (int r = 12; r >= 0; r--)
            if (counts[r] > 0 && r != pair1 && r != pair2) { kicker = r; break; }
        return (HC_TWO_PAIR << 20) | (pair1 << 16) | (pair2 << 12) | (kicker << 8);
    }
    if (pair1 >= 0) {
        int k[3], ki = 0;
        for (int r = 12; r >= 0 && ki < 3; r--)
            if (counts[r] > 0 && r != pair1) k[ki++] = r;
        return (HC_PAIR << 20) | (pair1 << 16) | (k[0] << 12) |
               (k[1] << 8) | (k[2] << 4);
    }
    /* High card */
    return (HC_HIGH_CARD << 20) | (ranks[0] << 16) | (ranks[1] << 12) |
           (ranks[2] << 8) | (ranks[3] << 4) | ranks[4];
}

/* ── Table-backed eval5 (default ON; kill switch -DPOKER_EVAL5_TABLE_DISABLE) ──
 *
 * A process-global int32 table of all C(52,5) = 2,598,960 five-card ranks,
 * filled once at library load by calling eval5_compute (exact by
 * construction, nothing shipped on disk). The single definition of the table
 * and its once-guard lives in hand_eval_table.c — every link target whose TUs
 * include this header must link that file. Lookup = colex index from the 5
 * sorted cards + one array probe. Device code (__CUDACC__) cannot reach the
 * host table, so it keeps the compute path. */
#if !defined(POKER_EVAL5_TABLE_DISABLE) && !defined(__CUDACC__)
#define POKER_EVAL5_TABLE 1
#endif

void poker_eval5_table_init(void);  /* idempotent; no-op when disabled */

#ifdef POKER_EVAL5_TABLE

#define POKER_EVAL5_TABLE_SIZE 2598960u  /* C(52,5) */

/* NULL until hand_eval_table.c publishes the fully built table
 * (release-store); eval5 acquire-loads it and falls back to eval5_compute
 * while NULL, so every read is exact at every point in time. */
extern const int32_t *poker_eval5_table;

/* Binomial tables for the colex index: poker_eval5_bcK[n] = C(n, K). */
static const uint32_t poker_eval5_bc2[52] = {
    0, 0, 1, 3, 6, 10, 15, 21,
    28, 36, 45, 55, 66, 78, 91, 105,
    120, 136, 153, 171, 190, 210, 231, 253,
    276, 300, 325, 351, 378, 406, 435, 465,
    496, 528, 561, 595, 630, 666, 703, 741,
    780, 820, 861, 903, 946, 990, 1035, 1081,
    1128, 1176, 1225, 1275
};
static const uint32_t poker_eval5_bc3[52] = {
    0, 0, 0, 1, 4, 10, 20, 35,
    56, 84, 120, 165, 220, 286, 364, 455,
    560, 680, 816, 969, 1140, 1330, 1540, 1771,
    2024, 2300, 2600, 2925, 3276, 3654, 4060, 4495,
    4960, 5456, 5984, 6545, 7140, 7770, 8436, 9139,
    9880, 10660, 11480, 12341, 13244, 14190, 15180, 16215,
    17296, 18424, 19600, 20825
};
static const uint32_t poker_eval5_bc4[52] = {
    0, 0, 0, 0, 1, 5, 15, 35,
    70, 126, 210, 330, 495, 715, 1001, 1365,
    1820, 2380, 3060, 3876, 4845, 5985, 7315, 8855,
    10626, 12650, 14950, 17550, 20475, 23751, 27405, 31465,
    35960, 40920, 46376, 52360, 58905, 66045, 73815, 82251,
    91390, 101270, 111930, 123410, 135751, 148995, 163185, 178365,
    194580, 211876, 230300, 249900
};
static const uint32_t poker_eval5_bc5[52] = {
    0, 0, 0, 0, 0, 1, 6, 21,
    56, 126, 252, 462, 792, 1287, 2002, 3003,
    4368, 6188, 8568, 11628, 15504, 20349, 26334, 33649,
    42504, 53130, 65780, 80730, 98280, 118755, 142506, 169911,
    201376, 237336, 278256, 324632, 376992, 435897, 501942, 575757,
    658008, 749398, 850668, 962598, 1086008, 1221759, 1370754, 1533939,
    1712304, 1906884, 2118760, 2349060
};

/* Colex rank of the 5-card set {c0..c4} among all C(52,5) subsets: sort
 * ascending (optimal 9-comparator network), then a0 + C(a1,2) + C(a2,3) +
 * C(a3,4) + C(a4,5). Requires 5 DISTINCT cards in 0..51 (as every caller
 * already guarantees — same precondition eval5_compute has always had). */
static inline uint32_t poker_eval5_index(int c0, int c1, int c2, int c3, int c4) {
    int a0 = c0, a1 = c1, a2 = c2, a3 = c3, a4 = c4;
#define POKER_EVAL5_CSWAP(x, y) do { if ((x) > (y)) { int _t = (x); (x) = (y); (y) = _t; } } while (0)
    POKER_EVAL5_CSWAP(a0, a1); POKER_EVAL5_CSWAP(a3, a4);
    POKER_EVAL5_CSWAP(a2, a4); POKER_EVAL5_CSWAP(a2, a3);
    POKER_EVAL5_CSWAP(a1, a4); POKER_EVAL5_CSWAP(a0, a3);
    POKER_EVAL5_CSWAP(a0, a2); POKER_EVAL5_CSWAP(a1, a3);
    POKER_EVAL5_CSWAP(a1, a2);
#undef POKER_EVAL5_CSWAP
    return (uint32_t)a0 + poker_eval5_bc2[a1] + poker_eval5_bc3[a2]
           + poker_eval5_bc4[a3] + poker_eval5_bc5[a4];
}

static inline uint32_t eval5(int c0, int c1, int c2, int c3, int c4) {
    const int32_t *t = __atomic_load_n(&poker_eval5_table, __ATOMIC_ACQUIRE);
    if (t) return (uint32_t)t[poker_eval5_index(c0, c1, c2, c3, c4)];
    return eval5_compute(c0, c1, c2, c3, c4);
}

#else  /* legacy compute path */

static inline uint32_t eval5(int c0, int c1, int c2, int c3, int c4) {
    return eval5_compute(c0, c1, c2, c3, c4);
}

#endif /* POKER_EVAL5_TABLE */

/**
 * Evaluate best 5-card hand from 7 cards.
 * Tries all C(7,5) = 21 combinations.
 */
static inline uint32_t eval7(const int cards[7]) {
    uint32_t best = 0;
    /* All 21 combinations of choosing 5 from 7 */
    static const int combos[21][5] = {
        {0,1,2,3,4}, {0,1,2,3,5}, {0,1,2,3,6}, {0,1,2,4,5},
        {0,1,2,4,6}, {0,1,2,5,6}, {0,1,3,4,5}, {0,1,3,4,6},
        {0,1,3,5,6}, {0,1,4,5,6}, {0,2,3,4,5}, {0,2,3,4,6},
        {0,2,3,5,6}, {0,2,4,5,6}, {0,3,4,5,6}, {1,2,3,4,5},
        {1,2,3,4,6}, {1,2,3,5,6}, {1,2,4,5,6}, {1,3,4,5,6},
        {2,3,4,5,6}
    };
    for (int i = 0; i < 21; i++) {
        uint32_t v = eval5(cards[combos[i][0]], cards[combos[i][1]],
                           cards[combos[i][2]], cards[combos[i][3]],
                           cards[combos[i][4]]);
        if (v > best) best = v;
    }
    return best;
}

/**
 * Parse card string like "Ah" to card index 0-51.
 * Returns -1 on error.
 */
static inline int parse_card(const char *s) {
    if (!s || !s[0] || !s[1]) return -1;
    int rank = -1, suit = -1;
    switch (s[0]) {
        case '2': rank = 0; break;  case '3': rank = 1; break;
        case '4': rank = 2; break;  case '5': rank = 3; break;
        case '6': rank = 4; break;  case '7': rank = 5; break;
        case '8': rank = 6; break;  case '9': rank = 7; break;
        case 'T': rank = 8; break;  case 'J': rank = 9; break;
        case 'Q': rank = 10; break; case 'K': rank = 11; break;
        case 'A': rank = 12; break;
    }
    switch (s[1]) {
        case 'c': suit = 0; break; case 'd': suit = 1; break;
        case 'h': suit = 2; break; case 's': suit = 3; break;
    }
    if (rank < 0 || suit < 0) return -1;
    return MAKE_CARD(rank, suit);
}

/**
 * Format card index 0-51 to string like "Ah".
 * buf must have space for 3 chars (including null terminator).
 */
static inline void format_card(int card, char *buf) {
    static const char ranks[] = "23456789TJQKA";
    static const char suits[] = "cdhs";
    buf[0] = ranks[CARD_RANK(card)];
    buf[1] = suits[CARD_SUIT(card)];
    buf[2] = '\0';
}

#endif /* HAND_EVAL_H */
