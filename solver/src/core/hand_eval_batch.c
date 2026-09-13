/*
 * hand_eval_batch.c — exported batch 7-card evaluation for ctypes callers.
 *
 * The api match harness (scripts/match_aivat.py) scores offline six-max
 * matches with self-play AIVAT, which needs every seat's strength on every
 * runout of a board (up to 1,081 runouts x 6 seats per state). The Python
 * _eval7_py fallback is far too slow for that, so this wraps the shared
 * table-backed eval7 from hand_eval.h behind one flat call.
 */
#include <stdint.h>

#include "hand_eval.h"

#if defined(_WIN32)
#define HAND_EVAL_BATCH_EXPORT __declspec(dllexport)
#else
#define HAND_EVAL_BATCH_EXPORT __attribute__((visibility("default")))
#endif

/*
 * cards: n rows of 7 card indices (rank * 4 + suit, 0..51), row-major.
 * out:   n strengths; higher wins, equal strengths tie.
 * Returns 0 on success, -1 on a NULL/negative argument, -2 on a card outside
 * 0..51 (out is then partially written and must be discarded).
 */
HAND_EVAL_BATCH_EXPORT int poker_eval7_batch(const int32_t *cards, int64_t n,
                                             uint32_t *out) {
    if (!cards || !out || n < 0) return -1;
    poker_eval5_table_init();
    for (int64_t i = 0; i < n; i++) {
        int hand[7];
        for (int j = 0; j < 7; j++) {
            int32_t card = cards[i * 7 + j];
            if (card < 0 || card > 51) return -2;
            hand[j] = (int)card;
        }
        out[i] = eval7(hand);
    }
    return 0;
}
