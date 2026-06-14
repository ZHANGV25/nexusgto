/* ss_cpu_stub.c — CPU-only-build stub for ss_solve_gpu.
 *
 * Linked ONLY into the gcc CPU build (street_solve_cpu.so), NEVER into the nvcc
 * GPU build (which has the real ss_solve_gpu in street_solve.cu — linking this
 * there would be a duplicate-symbol error).
 *
 * Why it must EXIST: python/street_solver_gpu.py::_get_lib does
 * `lib.ss_solve_gpu.restype = ...` — a direct ctypes attribute access that
 * raises AttributeError if the symbol is absent. So the CPU .so must EXPORT
 * ss_solve_gpu (so binding succeeds for the tree-build-only path) but ABORT if
 * it is ever actually called — the CPU build has no GPU solve, and returning a
 * fabricated/zero result would silently serve made-up strategy. Fail loud.
 */
#include "cuda/street_solve.cuh"
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
