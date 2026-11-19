/* ss_cpu_stub.c — stub ss_solve_gpu for the CPU tree-build library.
 *
 * Linked into street_solve_cpu (the tree builder library). The CUDA solver
 * that once defined ss_solve_gpu is not part of this tree.
 *
 * Why it must EXIST: solver/python/engine/street_solver_gpu.py::_get_lib does
 * `lib.ss_solve_gpu.restype = ...` — a direct ctypes attribute access that
 * raises AttributeError if the symbol is absent. So the CPU .so must EXPORT
 * ss_solve_gpu (so binding succeeds for the tree-build-only path) but ABORT if
 * it is ever actually called: the library has no solve, and returning a
 * fabricated/zero result would silently hand back a made-up strategy. Fail loud.
 */
#include "tree.h"
#include <stdio.h>
#include <stdlib.h>

SS_EXPORT int ss_solve_gpu(
    SSTreeData *td, int max_iterations, SSOutput *output
) {
    (void)td; (void)max_iterations; (void)output;
    fprintf(stderr,
        "[ss_cpu_stub] FATAL: ss_solve_gpu called on the CPU-only build "
        "(street_solve_cpu.so has NO GPU solver). This build is tree-build-only "
        "for the CPU-ES path. Never fabricate a solve — aborting.\n");
    abort();
}
