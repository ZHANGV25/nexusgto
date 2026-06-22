/**
 * bps_reader_internal.h — cross-TU internal declarations shared between
 * bps_reader.c and the bucketizer split bps_reader_bucketize.c.
 *
 * NOT a public API. These three helpers were file-static in bps_reader.c and
 * are promoted to extern (definitions stay in bps_reader.c) solely so the
 * turn/river/OCHS/PA-EMD bucketizer block can live in a sibling translation
 * unit compiled into the SAME shared library. No behavior change.
 */
#ifndef BPS_READER_INTERNAL_H
#define BPS_READER_INTERNAL_H

#include "bps_reader.h"


/* Hidden-visibility for promoted-from-static internals (see mccfr_blueprint_internal.h).
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

/* Canonical board hash (board-isomorphism key). Definition in bps_reader.c. */
NEXUS_INTERNAL uint64_t compute_board_hash(const int *board, int num_board, int use_legacy_mixer);

/* Texture-cache flop-hash index lookup; returns texture index or -1.
 * Definition in bps_reader.c. */
NEXUS_INTERNAL int tex_index_lookup(const BPSTextureCache *tc, uint64_t flop_hash);

/* Rebuild the texture-cache open-addressing index. Definition in bps_reader.c. */
NEXUS_INTERNAL void tex_index_rebuild(BPSTextureCache *tc);

#ifdef __cplusplus
}
#endif

#endif /* BPS_READER_INTERNAL_H */
