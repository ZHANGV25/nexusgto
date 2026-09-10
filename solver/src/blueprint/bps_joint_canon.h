/**
 * bps_joint_canon.h - joint board/hand suit canonicalization for river tables.
 *
 * A river situation is five unordered board cards plus two unordered private
 * cards.  Ranks are fixed and one global permutation relabels the four suits.
 * bps_joint_canon_river() returns the lexicographically least
 * (ascending board image, ascending hand image) among all 24 relabelings.
 * Cards use the repository convention rank * 4 + suit, with ranks 0..12 and
 * suits 0..3.
 *
 * Dense-rank convention (a deliberately pinned RBK1 format choice): represent
 * each suit by a 13-trit column in increasing-rank order, where 0 means absent,
 * 1 means board, and 2 means hand.  Sort the four columns lexicographically,
 * then compare the resulting encodings row-major: rank 0 columns 0..3, rank 1
 * columns 0..3, and so on.  The rank is the zero-based lexicographic ordinal
 * among encodings containing exactly five 1 trits and two 2 trits.
 *
 * The implementation ranks this language with a suffix DP.  Its three-bit
 * tie mask records which adjacent sorted columns are still identical over the
 * rows already consumed; a tied pair must remain nondecreasing on the next
 * row.  The initial count is checked explicitly and must be exactly
 * 123,156,254.  Thus ranks form a bijection onto [0, 123156254), with neither
 * a sparse hash nor a stored key array.  This ordering, including low-rank
 * first and 0 < 1 < 2, is part of the artifact identity and must not change.
 * RBK1 algorithm_version 1 is intended to pin this exact convention; any
 * ordering or language change requires an algorithm-version bump.
 *
 * TURN is intentionally absent.  Its raw-card-seeded Monte Carlo feature path
 * is not suit-equivariant, so an orbit table cannot be bit-identical.  The
 * per-solve memo also warms roughly 57,000 turn keys per flop in under one
 * second, whereas the material cold cost is roughly 1.3 million river keys.
 * The production exhaustive OCHS river path is suit-equivariant.
 */
#ifndef BPS_JOINT_CANON_H
#define BPS_JOINT_CANON_H

#include <stdint.h>

#if defined(_WIN32)
#define BPS_JOINT_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define BPS_JOINT_EXPORT __attribute__((visibility("default")))
#else
#define BPS_JOINT_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define BPS_JOINT_RIVER_NUM_CLASSES 123156254ULL

/**
 * Visitor for dense-rank enumeration.
 *
 * board and hand are ascending and are the explicit lexicographically least
 * representative returned by bps_joint_canon_river().  They remain valid only
 * for the duration of the call.  Return zero to continue; any nonzero return
 * aborts enumeration and makes bps_joint_enumerate_river() return -1.
 */
typedef int (*BPSJointRiverVisitor)(int64_t rank,
                                    const int board[5],
                                    const int hand[2],
                                    void *ctx);

/**
 * Canonicalize a legal river board/hand pair under all 24 suit permutations.
 * Input order is immaterial and outputs may alias their matching inputs.
 * Returns 0 on success, or -1 for NULL, out-of-range, or duplicate cards.
 */
BPS_JOINT_EXPORT int bps_joint_canon_river(const int board[5],
                                           const int hand[2],
                                           int canon_board[5],
                                           int canon_hand[2]);

/**
 * Return the dense orbit rank of any legal representative, or -1 on invalid
 * cards or a failed DP invariant.  The function sorts suit-signature columns
 * directly, so callers need not canonicalize first (the production lookup
 * deliberately does so to keep the two operations explicit).
 */
BPS_JOINT_EXPORT int64_t bps_joint_rank_river(const int board[5],
                                              const int hand[2]);

/**
 * Recover the explicit lexicographically least representative of rank.
 * Returns 0 on success, or -1 when rank is outside [0, N) or DP init failed.
 */
BPS_JOINT_EXPORT int bps_joint_unrank_river(int64_t rank,
                                           int board[5],
                                           int hand[2]);

/**
 * Visit the half-open dense-rank range [begin, end) in increasing order.
 * DP subtrees wholly outside the requested range are skipped.  Returns the
 * visited count (exactly end - begin) or -1 for bad arguments, failed DP
 * initialization, representative construction failure, or visitor abort.
 */
BPS_JOINT_EXPORT int64_t bps_joint_enumerate_river(
    int64_t begin,
    int64_t end,
    BPSJointRiverVisitor visitor,
    void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* BPS_JOINT_CANON_H */
