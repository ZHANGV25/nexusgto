/**
 * mccfr_blueprint_internal.h — cross-TU internal declarations for the
 * blueprint implementation, shared between mccfr_blueprint.c and the
 * serialization split mccfr_blueprint_io.c.
 *
 * NOT a public API. These symbols were file-static in mccfr_blueprint.c and
 * are promoted to extern (with their definitions kept in mccfr_blueprint.c)
 * solely so the regret save/load IO block can live in a sibling translation
 * unit compiled into the SAME shared library. No behavior change.
 */
#ifndef MCCFR_BLUEPRINT_INTERNAL_H
#define MCCFR_BLUEPRINT_INTERNAL_H

#include "mccfr_blueprint.h"

/* These symbols are promoted-from-static so two split TUs can share them WITHIN
 * one shared library, but they are NOT a public API. Mark them hidden so the
 * dynamic linker never exports them — this preserves the old file-static
 * isolation (no cross-.so interposition if another DSO in the same process also
 * links this code). No-op on MSVC/Windows (.dll has no global interposition). */
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

/* Bump-arena allocator: returns a zeroed slice of num_ints ints, thread-local
 * fast path with a global mutex-guarded refill. Definition in mccfr_blueprint.c. */
NEXUS_INTERNAL void *arena_alloc(int num_ints);

/* Rebuild the per-solver flop-texture index after a texture cache load.
 * Definition in mccfr_blueprint.c. */
NEXUS_INTERNAL void texture_index_rebuild(BPSolver *s);

/* Hogwild-safe find-or-create of an info set slot for `key`. Returns the slot
 * index. Definition in mccfr_blueprint.c. */
NEXUS_INTERNAL int64_t info_table_find_or_create(BPInfoTable *t, BPInfoKey key, int num_actions);

/* Lazily arena-allocate the strategy_sum array for an info set on first use.
 * Definition in mccfr_blueprint.c. */
NEXUS_INTERNAL void ensure_strategy_sum(BPInfoSet *is);

#ifdef __cplusplus
}
#endif

#endif /* MCCFR_BLUEPRINT_INTERNAL_H */
