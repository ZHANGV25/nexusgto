CC = gcc
CFLAGS = -O2 -Wall -Wextra \
         -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function
LDFLAGS = -lm

SRC_DIR = src
BENCH_DIR = bench
BUILD_DIR = build

# ── Include search path ──
# Every #include uses a BARE sibling name (e.g. "card_abstraction.h",
# "mccfr_blueprint.h", "rollout.h"); the C sources now live in domain subdirs
# (core/blueprint/search/archive) so one -I per subdir resolves them all WITHOUT
# editing any #include line. -I$(SRC_DIR) MUST be kept so the relative
# "cuda/street_solve.cuh" includes still resolve (cuda/ stays at src/cuda/).
INCS = -I$(SRC_DIR) -I$(SRC_DIR)/core -I$(SRC_DIR)/blueprint -I$(SRC_DIR)/search

# ── Relocated source paths (domain subdirs; structure-only move) ──
CORE_DIR   = $(SRC_DIR)/core
BP_DIR     = $(SRC_DIR)/blueprint
SEARCH_DIR = $(SRC_DIR)/search
ARCH_DIR   = $(SRC_DIR)/archive

# Detect OS for shared library extension
ifeq ($(OS),Windows_NT)
  SHLIB_EXT = .dll
  SHLIB_FLAGS = -shared -static
else
  SHLIB_EXT = .so
  SHLIB_FLAGS = -shared -fPIC
endif

.PHONY: all clean bench dll test test_phase_1_3 test_ev_walk blueprint bps_reader bps_ev_walk rollout street_solve street_solve_joint street_solve_cpu subgame_preflop

all: blueprint

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ── Blueprint MCCFR engine (production) ──
# Requires both mccfr_blueprint.c and card_abstraction.c (k-means bucketing)
blueprint: $(BUILD_DIR)/mccfr_blueprint$(SHLIB_EXT)

$(BUILD_DIR)/mccfr_blueprint$(SHLIB_EXT): $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(BP_DIR)/mccfr_cont_ev_table.h $(CORE_DIR)/card_abstraction.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -fopenmp $(INCS) $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c -o $@ $(LDFLAGS)

# ── Street solver v2 (GPU re-solve support library) ──
$(BUILD_DIR)/solver_v2$(SHLIB_EXT): $(ARCH_DIR)/solver_v2.c $(ARCH_DIR)/solver_v2.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) $(INCS) $(ARCH_DIR)/solver_v2.c -o $@ $(LDFLAGS)

# ── Preflop subgame CFR (CPU, OpenMP-parallel over hero buckets) ──
# Pure CFR core: takes a flat tree + reaches + leaf values, returns σ̄.
# No bps_reader / blueprint dependency; orchestration lives Python-side.
# Uses -O3 -march=native to enable autovectorization of the inner bucket loops.
subgame_preflop: $(BUILD_DIR)/subgame_preflop$(SHLIB_EXT)

# OpenMP availability: gcc supports it directly; Apple clang needs
# Homebrew libomp + explicit flags. Set OMP_FLAGS / OMP_LIBS in a
# Mac-friendly way; falls back to no-OpenMP if neither is available.
# Detection: if 'gcc' isn't actually clang we can use -fopenmp; else
# try -Xpreprocessor -fopenmp -lomp via Homebrew.
ifeq ($(shell $(CC) --version 2>&1 | grep -c clang),0)
  SUBGAME_OMP_FLAGS = -fopenmp
  SUBGAME_OMP_LIBS =
else ifneq ($(wildcard /opt/homebrew/include/omp.h),)
  SUBGAME_OMP_FLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/include
  SUBGAME_OMP_LIBS = -L/opt/homebrew/lib -lomp
else
  SUBGAME_OMP_FLAGS =
  SUBGAME_OMP_LIBS =
endif

$(BUILD_DIR)/subgame_preflop$(SHLIB_EXT): $(ARCH_DIR)/subgame_preflop.c $(ARCH_DIR)/subgame_preflop.h $(BP_DIR)/bps_reader.h | $(BUILD_DIR)
	$(CC) -O3 -march=native -Wall -Wextra -Wno-unused-parameter \
	    $(SHLIB_FLAGS) $(SUBGAME_OMP_FLAGS) $(INCS) \
	    $(ARCH_DIR)/subgame_preflop.c -o $@ $(LDFLAGS) $(SUBGAME_OMP_LIBS)

# ── Street solver v1 (legacy, for Python bindings compatibility) ──
$(BUILD_DIR)/solver$(SHLIB_EXT): $(ARCH_DIR)/solver.c $(ARCH_DIR)/solver.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) $(INCS) $(ARCH_DIR)/solver.c -o $@ $(LDFLAGS)

# ── BPS reader (pure reader + turn/river bucketizer) ──
# Links card_abstraction.c for bps_hand_to_bucket_turn/river (EHS/k-means).
# No OpenMP needed — bucketizer is single-hand at a time.
bps_reader: $(BUILD_DIR)/bps_reader$(SHLIB_EXT)

$(BUILD_DIR)/bps_reader$(SHLIB_EXT): $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/bps_reader.h $(CORE_DIR)/card_abstraction.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) $(INCS) $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c -o $@ $(LDFLAGS)

# ── BPS EV walk (reader + card_abstraction + OpenMP) ──
bps_ev_walk: $(BUILD_DIR)/bps_ev_walk$(SHLIB_EXT)

$(BUILD_DIR)/bps_ev_walk$(SHLIB_EXT): $(BP_DIR)/bps_ev_walk.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/bps_reader.h $(CORE_DIR)/card_abstraction.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -fopenmp $(INCS) $(BP_DIR)/bps_ev_walk.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c -o $@ $(LDFLAGS)

# ── Rollout (C port of python/rollout.py) ──
# Depth-limited Pluribus-faithful leaf values for real-time solve.
# Links bps_reader.c (blueprint queries); eval7 is header-only.
rollout: $(BUILD_DIR)/rollout$(SHLIB_EXT)

$(BUILD_DIR)/rollout$(SHLIB_EXT): $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(SEARCH_DIR)/rollout.h $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(BP_DIR)/bps_reader.h $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(CORE_DIR)/card_abstraction.h $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c $(BP_DIR)/mccfr_cont_ev_table.h $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(BP_DIR)/mccfr_phase_r.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -fopenmp $(INCS) $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(BP_DIR)/mccfr_phase_r.c -o $@ $(LDFLAGS)

# ── CPU external-sampling ES re-solver (HYBRID 5p flop/turn path) ──
# es_resolver: lightweight (table/joint oracles only) — validation harnesses.
# es_resolver_real: links rollout + bps_reader + mccfr → adds the real-blueprint
# oracle + the BPS loader symbols. The API's CpuEsSolver loads es_resolver_real.
# On Windows, mccfr_cont_ev_table.c uses fsync() which MSYS2 ucrt64 does not
# declare → shim it to _commit (io.h). On Linux fsync is native (unistd.h),
# no shim needed. NOTE: the Linux .so build is unexercised as of 2026-06-01;
# this target is the prod-deploy build recipe (validated on Windows only).
ifeq ($(OS),Windows_NT)
  ES_SHIM = -Dfsync=_commit -include io.h
else
  ES_SHIM =
endif

.PHONY: es_resolver es_resolver_real
es_resolver: $(BUILD_DIR)/es_resolver$(SHLIB_EXT)
es_resolver_real: $(BUILD_DIR)/es_resolver_real$(SHLIB_EXT)

$(BUILD_DIR)/es_resolver$(SHLIB_EXT): $(SEARCH_DIR)/es_resolver.c $(SEARCH_DIR)/es_resolver_reach.c $(SEARCH_DIR)/es_resolver.h $(SEARCH_DIR)/es_oracles.c $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -static-libgcc -fopenmp $(INCS) \
	    $(SEARCH_DIR)/es_resolver.c $(SEARCH_DIR)/es_resolver_reach.c $(SEARCH_DIR)/es_oracles.c -o $@ $(LDFLAGS)

$(BUILD_DIR)/es_resolver_real$(SHLIB_EXT): $(SEARCH_DIR)/es_resolver.c $(SEARCH_DIR)/es_resolver_reach.c $(SEARCH_DIR)/es_oracles.c $(SEARCH_DIR)/es_real_oracle.c $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(BP_DIR)/mccfr_phase_r.c $(SEARCH_DIR)/es_resolver.h $(SEARCH_DIR)/rollout.h | $(BUILD_DIR)
	$(CC) -O2 -w $(SHLIB_FLAGS) -static-libgcc -fopenmp -D_GNU_SOURCE $(ES_SHIM) $(INCS) \
	    $(SEARCH_DIR)/es_resolver.c $(SEARCH_DIR)/es_resolver_reach.c $(SEARCH_DIR)/es_oracles.c $(SEARCH_DIR)/es_real_oracle.c \
	    $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(BP_DIR)/mccfr_phase_r.c \
	    -o $@ $(LDFLAGS)

# ── Street solve (CUDA — N-player single-street GPU solver) ──
# Builds with nvcc; requires CUDA toolkit on PATH. Targets sm_75 (T4 on
# g4dn) by default; override SS_GPU_ARCH=sm_86 for A10, sm_70 for V100.
# Skip on Mac dev machines (nvcc not present); the .so is consumed by
# python/street_solver_gpu.py at runtime via ctypes on Linux.
NVCC ?= nvcc
SS_GPU_ARCH ?= sm_75
street_solve: $(BUILD_DIR)/street_solve$(SHLIB_EXT)

$(BUILD_DIR)/street_solve$(SHLIB_EXT): $(SRC_DIR)/cuda/street_solve.cu $(SRC_DIR)/cuda/street_solve.cuh $(SEARCH_DIR)/ss_tree_builder.c $(SEARCH_DIR)/ss_tree_builder.h $(SEARCH_DIR)/ss_tree_api.c | $(BUILD_DIR)
	$(NVCC) -O2 -shared -Xcompiler -fPIC -arch=$(SS_GPU_ARCH) $(INCS) \
	    $(SRC_DIR)/cuda/street_solve.cu $(SEARCH_DIR)/ss_tree_builder.c $(SEARCH_DIR)/ss_tree_api.c -o $@

# JOINT-mode variant for the nexusgto-api Phase 5.1 diagnostic
# (docs/OP4_CONT_STRAT_MARGINAL_PLAN.md). Same sources, compiled with
# -DSS_CONT_STRAT_MODE=0 so expand_continuation_leaves_n builds the 4^N
# chain instead of the SAMPLED no-op. Diagnostic only — JOINT has the
# sequential-observation deviation documented in the plan. Memory at
# HU/3p/4p fits a T4 (0.01/0.21/5.31 GB per OP4_BENCH.md); n>=5 must
# stay on SAMPLED.
street_solve_joint: $(BUILD_DIR)/street_solve_joint$(SHLIB_EXT)

$(BUILD_DIR)/street_solve_joint$(SHLIB_EXT): $(SRC_DIR)/cuda/street_solve.cu $(SRC_DIR)/cuda/street_solve.cuh $(SEARCH_DIR)/ss_tree_builder.c $(SEARCH_DIR)/ss_tree_builder.h $(SEARCH_DIR)/ss_tree_api.c | $(BUILD_DIR)
	$(NVCC) -O2 -shared -Xcompiler -fPIC -arch=$(SS_GPU_ARCH) -DSS_CONT_STRAT_MODE=0 $(INCS) \
	    $(SRC_DIR)/cuda/street_solve.cu $(SEARCH_DIR)/ss_tree_builder.c $(SEARCH_DIR)/ss_tree_api.c -o $@

# ── Street solve (CPU-only — tree builder, NO GPU solve) ──
# gcc build of just the tree-build API (ss_tree_api.c + ss_tree_builder.c) plus
# an abort-stub for ss_solve_gpu (ss_cpu_stub.c). Produced as street_solve.so so
# python/street_solver_gpu.py loads it by its default name on a no-GPU box. The
# CPU-ES re-solver uses StreetSolverGPU ONLY as a tree builder (leaf_value_fn=
# None, then CSolver.run does the solve) — so this .so is sufficient for the
# multiway measurement + CPU-ES path. ss_solve_gpu aborts loudly if ever called
# (never fabricate a solve). Build with: make street_solve_cpu
# (Outputs build/street_solve_cpu.so; deploy by copying it to build/street_solve.so
#  on the no-GPU box so the default loader name resolves.)
CC ?= gcc
street_solve_cpu: $(BUILD_DIR)/street_solve_cpu$(SHLIB_EXT)

$(BUILD_DIR)/street_solve_cpu$(SHLIB_EXT): $(SEARCH_DIR)/ss_tree_api.c $(SEARCH_DIR)/ss_tree_builder.c $(SEARCH_DIR)/ss_cpu_stub.c $(SEARCH_DIR)/ss_tree_builder.h $(SRC_DIR)/cuda/street_solve.cuh | $(BUILD_DIR)
	$(CC) -O2 -shared -fPIC $(INCS) \
	    $(SEARCH_DIR)/ss_tree_api.c $(SEARCH_DIR)/ss_tree_builder.c $(SEARCH_DIR)/ss_cpu_stub.c -o $@

dll: blueprint $(BUILD_DIR)/solver_v2$(SHLIB_EXT) $(BUILD_DIR)/solver$(SHLIB_EXT) $(BUILD_DIR)/bps_reader$(SHLIB_EXT) $(BUILD_DIR)/bps_ev_walk$(SHLIB_EXT) $(BUILD_DIR)/rollout$(SHLIB_EXT)

# ── Benchmarks ──
$(BUILD_DIR)/bench_v2: $(BENCH_DIR)/bench_v2.c $(ARCH_DIR)/solver_v2.c $(ARCH_DIR)/solver_v2.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -static $(INCS) $< $(ARCH_DIR)/solver_v2.c -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_phase1: $(BENCH_DIR)/test_phase1.c $(ARCH_DIR)/solver_v2.c $(ARCH_DIR)/solver_v2.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -static $(INCS) $< $(ARCH_DIR)/solver_v2.c -o $@ $(LDFLAGS)

bench: $(BUILD_DIR)/bench_v2
	./$(BUILD_DIR)/bench_v2

test: $(BUILD_DIR)/test_phase1
	./$(BUILD_DIR)/test_phase1

# ── Phase 1.3 synthetic ground-truth test ──
# Statically links the blueprint solver for a self-contained test binary.
# Python-side verifier (tests/test_phase_1_3_synthetic.py) consumes the
# bundle this produces and asserts CFR identity + enumerated ground truth.
$(BUILD_DIR)/test_phase_1_3: tests/test_phase_1_3_synthetic.c $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(BP_DIR)/mccfr_cont_ev_table.h $(CORE_DIR)/card_abstraction.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_phase_1_3_synthetic.c $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c -o $@ $(LDFLAGS)

test_phase_1_3: $(BUILD_DIR)/test_phase_1_3
	./$(BUILD_DIR)/test_phase_1_3 $(BUILD_DIR)/phase_1_3_synthetic.tbn
	python3 tests/test_phase_1_3_synthetic.py $(BUILD_DIR)/phase_1_3_synthetic.tbn

# ── BPS EV walk tests ──
$(BUILD_DIR)/test_bps_ev_walk: tests/test_bps_ev_walk.c $(BP_DIR)/bps_ev_walk.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/bps_reader.h $(CORE_DIR)/card_abstraction.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_bps_ev_walk.c $(BP_DIR)/bps_ev_walk.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c -o $@ $(LDFLAGS) -static

test_ev_walk: $(BUILD_DIR)/test_bps_ev_walk
	./$(BUILD_DIR)/test_bps_ev_walk

# ── PA-EMD tests (ca_emd_1d + ca_compute_ehs_distribution) ──
$(BUILD_DIR)/test_pa_emd: tests/test_pa_emd.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(CORE_DIR)/card_abstraction.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_pa_emd.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c -o $@ $(LDFLAGS)

test_pa_emd: $(BUILD_DIR)/test_pa_emd
	./$(BUILD_DIR)/test_pa_emd

# ── BPS OCHS river bucketer regression test ──
# Pins the Phase B fix: bps_init_ochs_river + bps_hand_to_bucket_river
# routing through the OCHS path on representative river fixtures.
$(BUILD_DIR)/test_bps_ochs_river: tests/test_bps_ochs_river.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/bps_reader.h $(CORE_DIR)/card_abstraction.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_bps_ochs_river.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c -o $@ $(LDFLAGS)

test_bps_ochs_river: $(BUILD_DIR)/test_bps_ochs_river
	./$(BUILD_DIR)/test_bps_ochs_river

# ── Phase R Block A.1: dispatch helper unit tests ──
# Pure helper (mccfr_phase_r.c) tested in isolation — no card_abstraction
# / hand_eval / OMP needed. Mac-runnable, <1s wall.
$(BUILD_DIR)/test_phase_r_dispatch: tests/test_phase_r_dispatch.c $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_phase_r_dispatch.c $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c -o $@ $(LDFLAGS) -lpthread

test_phase_r_dispatch: $(BUILD_DIR)/test_phase_r_dispatch
	./$(BUILD_DIR)/test_phase_r_dispatch

# ── Phase R Block A.2: cont-ev table unit tests ──
# Tests the boundary EV slot table (mccfr_cont_ev_table.{c,h}) in
# isolation. Uses OpenMP for the multi-thread CAS test (T5) — falls
# back to a SKIP if OMP isn't available (Mac without libomp).
$(BUILD_DIR)/test_cont_ev_table: tests/test_cont_ev_table.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_cont_ev_table.c $(BP_DIR)/mccfr_cont_ev_table.c -o $@ $(LDFLAGS) -lpthread

test_cont_ev_table: $(BUILD_DIR)/test_cont_ev_table
	./$(BUILD_DIR)/test_cont_ev_table

# ── Phase R Block A.2: cont-bias reweighting helper unit tests ──
# Pure math (bp_apply_cont_bias in mccfr_phase_r.c). Mac-runnable, <1s.
$(BUILD_DIR)/test_cont_bias_strategy: tests/test_cont_bias_strategy.c $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_cont_bias_strategy.c $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c -o $@ $(LDFLAGS) -lpthread

test_cont_bias_strategy: $(BUILD_DIR)/test_cont_bias_strategy
	./$(BUILD_DIR)/test_cont_bias_strategy

# ── Phase R Block A.2: byte-identity regression smoke ──
# Drives the full trainer with depth_limit_cont_strats=0 (default) on a
# 6p postflop fixture, single-thread, twice with the same seed, and
# byte-compares the saved regret tables. Must produce identical bytes
# on both runs (determinism), which combined with the unit-level T1
# invariant in test_phase_r_dispatch.c proves byte-identity to legacy.
#
# Compiled WITHOUT -fopenmp so it runs on Mac with Apple clang. The
# trainer's omp pragmas degrade to no-ops; bp_solve runs single-threaded.
$(BUILD_DIR)/test_phase_r_regression_smoke: tests/test_phase_r_regression_smoke.c \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_phase_r_regression_smoke.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_phase_r_regression_smoke: $(BUILD_DIR)/test_phase_r_regression_smoke
	./$(BUILD_DIR)/test_phase_r_regression_smoke 10000

# ── Phase R Block A.2: HEADS-branch rollout integration ──
# Drives the trainer with depth_limit_cont_strats=1, prob=1.0 on a 3p
# postflop fixture. Verifies the HEADS-branch rollout dispatcher fires,
# allocates cont-ev table slots, and accumulates ev_count > 0 with
# finite ev_sum values. Mac-runnable, ~10-20s.
$(BUILD_DIR)/test_phase_r_heads_rollout: tests/test_phase_r_heads_rollout.c \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_phase_r_heads_rollout.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_phase_r_heads_rollout: $(BUILD_DIR)/test_phase_r_heads_rollout
	./$(BUILD_DIR)/test_phase_r_heads_rollout

# ── Phase R Block A.2: cont-bias signed-regrets correctness ──
# Drives 3p-AA-on-flop fixture with prob=1.0, asserts raise-bias EV >
# fold-bias EV (gates against bias-index swap, sign-inversion bugs).
# Plus T3: bp_compute_action_hash determinism (cont-bias-free by
# signature). Mac-runnable, ~10-15s.
$(BUILD_DIR)/test_phase_r_cont_strat_signed: tests/test_phase_r_cont_strat_signed.c \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_phase_r_cont_strat_signed.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_phase_r_cont_strat_signed: $(BUILD_DIR)/test_phase_r_cont_strat_signed
	./$(BUILD_DIR)/test_phase_r_cont_strat_signed

# ── Phase R Block B.2: bps_query_boundary_ev contract ──
# Pure-table test; no trainer dep. ~<1s wall.
$(BUILD_DIR)/test_bps_query_boundary_ev: tests/test_bps_query_boundary_ev.c \
        $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_bps_query_boundary_ev.c \
	    $(BP_DIR)/mccfr_cont_ev_table.c -o $@ $(LDFLAGS) -lpthread

test_bps_query_boundary_ev: $(BUILD_DIR)/test_bps_query_boundary_ev
	./$(BUILD_DIR)/test_bps_query_boundary_ev

# ── Phase R Block B.1: BPR5 writer/reader round-trip ──
# Drives the full trainer (5K iters, prob=1.0) on a 3p postflop fixture,
# bp_save_regrets to /tmp, bp_load_regrets into a fresh BPSolver, asserts
# byte-exact match on every (action_hash, player, bucket) → (ev_sum[4],
# ev_count[4]) tuple. Mac-runnable, ~10-20s.
$(BUILD_DIR)/test_bpr5_roundtrip: tests/test_bpr5_roundtrip.c \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_bpr5_roundtrip.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_bpr5_roundtrip: $(BUILD_DIR)/test_bpr5_roundtrip
	./$(BUILD_DIR)/test_bpr5_roundtrip

# ── Phase R Step 12: cont_ev boundary slot-key encoding pin ──
# Asserts the call-site key is the consumer's canonical full-deck pair
# index (rollout_hand_index_for_flop encoding), NOT a texture bucket and
# NOT the raw per-player hand-list index. Deterministic, <1s.
$(BUILD_DIR)/test_cont_ev_key: tests/test_cont_ev_key.c \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_cont_ev_key.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_cont_ev_key: $(BUILD_DIR)/test_cont_ev_key
	./$(BUILD_DIR)/test_cont_ev_key

# ── history→hash contract (2026-07-03): producer trainer-token emission vs
# consumer engine-assembled game_history over REAL canonical lines, plus the
# negative control pinning the pre-fix harness spot as game-impossible. The
# missing test class behind the Stage-2 hash disjunction. Deterministic, <1s.
$(BUILD_DIR)/test_cont_ev_history_hash_contract: tests/test_cont_ev_history_hash_contract.c \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_cont_ev_history_hash_contract.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_cont_ev_history_hash_contract: $(BUILD_DIR)/test_cont_ev_history_hash_contract
	./$(BUILD_DIR)/test_cont_ev_history_hash_contract

# ── Phase R Block B.1+B.2: end-to-end save → load → query parity ──
# 5K-iter trainer drive with prob=1.0; capture one populated slot's
# (ev_sum, ev_count); save+load; query via bp_query_boundary_ev; assert
# byte-exact match on stored fields and OK status with the divided ratio.
$(BUILD_DIR)/test_phase_r_save_load_e2e: tests/test_phase_r_save_load_e2e.c \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_phase_r_save_load_e2e.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_phase_r_save_load_e2e: $(BUILD_DIR)/test_phase_r_save_load_e2e
	./$(BUILD_DIR)/test_phase_r_save_load_e2e

# ── Phase R Block B.3: ss_compute_leaf_action_hashes contract ──
# CPU-only Mac-runnable test — hand-builds SSNode trees, validates the
# kernel-side root-to-leaf walk reproduces bp_compute_action_hash for
# every leaf. PASS gate per spec §9.
$(BUILD_DIR)/test_ss_action_hash: tests/test_ss_action_hash.c \
        $(ARCH_DIR)/ss_action_hash.c $(ARCH_DIR)/ss_action_hash.h \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(SRC_DIR)/cuda/street_solve.cuh \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) -I$(ARCH_DIR) tests/test_ss_action_hash.c \
	    $(ARCH_DIR)/ss_action_hash.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_ss_action_hash: $(BUILD_DIR)/test_ss_action_hash
	./$(BUILD_DIR)/test_ss_action_hash

# ── Phase R Step 5: discount-sweep gate regression ──
# Pins the gate at mccfr_blueprint.c:3915 (`if (s->depth_limit_cont_strats)`
# around bp_cont_ev_apply_discount). Without the gate, a loader-only
# flow that loads a populated BPR5 trailer into a flag-OFF solver and
# then runs bp_solve would silently scale every loaded ev_sum on each
# discount tick. Mac-runnable, ~5-10s wall.
$(BUILD_DIR)/test_phase_r_discount_gate: tests/test_phase_r_discount_gate.c \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_phase_r_discount_gate.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_phase_r_discount_gate: $(BUILD_DIR)/test_phase_r_discount_gate
	./$(BUILD_DIR)/test_phase_r_discount_gate

# ── Phase R Block B.3 (Step 6): encoding-alignment PASS gate ──
# Cross-validates kernel-built tree action emission against the
# trainer's bp_generate_actions byte-for-byte, on v3/phase3 production
# config. Mac-runnable via the ss_tree_builder.{c,h} extraction.
$(BUILD_DIR)/test_b3_encoding_alignment: tests/test_b3_encoding_alignment.c \
        $(SEARCH_DIR)/ss_tree_builder.c $(SEARCH_DIR)/ss_tree_builder.h \
        $(ARCH_DIR)/ss_action_hash.c $(ARCH_DIR)/ss_action_hash.h \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(SRC_DIR)/cuda/street_solve.cuh \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) -I$(ARCH_DIR) tests/test_b3_encoding_alignment.c \
	    $(SEARCH_DIR)/ss_tree_builder.c $(ARCH_DIR)/ss_action_hash.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_b3_encoding_alignment: $(BUILD_DIR)/test_b3_encoding_alignment
	./$(BUILD_DIR)/test_b3_encoding_alignment

# ── Phase R Block B.3 (Step 6): bucket-lookup divergence test ──
# HARD GATE before B.3 PASS. Loads /tmp/texture_cache.bin (or
# BPS_TEXTURE_CACHE override) via both bp_load_texture_cache and
# bps_load_texture_cache, walks all 1755 canonical flop boards and
# all non-blocked (c0,c1) pairs, asserts trainer-stored bucket ==
# reader-queried bucket on every covered query.
$(BUILD_DIR)/test_b3_bucket_alignment: tests/test_b3_bucket_alignment.c \
        $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(BP_DIR)/bps_reader.h \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_b3_bucket_alignment.c \
	    $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_b3_bucket_alignment: $(BUILD_DIR)/test_b3_bucket_alignment
	./$(BUILD_DIR)/test_b3_bucket_alignment

# ── Phase R Step 11: boundary-bucket fix + EV-only invariance ──
# Drives bp_init_unified (auto-loads /tmp/texture_cache.bin), then:
#   T1: query bp_compute_flop_bucket_for_test on multiple flops × hands;
#       assert every result is in [0, postflop_num_buckets).
#   T2: with BP_BOUNDARY_EV_ONLY=1 set in env, run a small bp_solve and
#       assert is->regrets stay zero + no strategy_sum allocations.
#   T3: same EV-only run, assert every cont_ev_table slot's bucket is
#       in [0, postflop_num_buckets) (the Gate C.1 acceptance criterion).
$(BUILD_DIR)/test_phase_r_boundary_bucket: tests/test_phase_r_boundary_bucket.c \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_phase_r_boundary_bucket.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS) -lpthread

test_phase_r_boundary_bucket: $(BUILD_DIR)/test_phase_r_boundary_bucket
	./$(BUILD_DIR)/test_phase_r_boundary_bucket

# ── Phase F Step 14: rollout.c cont_ev_table boundary consumer ──
# Verifies that src/rollout.c, when configured with a populated
# cont_ev_table and the boundary flag enabled, short-circuits the
# end-of-flop multiway rollout: uniform mean over 4 bias EVs becomes
# the leaf payoff; MISS / EMPTY_BIAS fails loud; flag OFF and HU flop
# leave the boundary lookup inert. Mac-runnable, ~<1s.
# Built WITHOUT -fopenmp — the public entry exercised here
# (rollout_compute_leaf_values_all_hands_c on the skip-betting path)
# never enters the OMP parallel regions.
$(BUILD_DIR)/test_rollout_cont_ev_boundary: tests/test_rollout_cont_ev_boundary.c \
        $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(SEARCH_DIR)/rollout.h \
        $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(BP_DIR)/bps_reader.h \
        $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(CORE_DIR)/card_abstraction.h \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_cont_ev_table.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_rollout_cont_ev_boundary.c \
	    $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c \
	    $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS) -lpthread

test_rollout_cont_ev_boundary: $(BUILD_DIR)/test_rollout_cont_ev_boundary
	./$(BUILD_DIR)/test_rollout_cont_ev_boundary

# ── Unit 3 (Stage-1 mean-field, 2026-07-01): bias-as-hero-action consumer ──
# Rigged slot where bias 2's EV dominates: with bias-action ON the returned
# leaf value converges toward EV_2 over repeated queries (and equals the
# uniform mean with the flag OFF, same slot); regret persistence + reset.
# Same link line as test_rollout_cont_ev_boundary. Mac-runnable, ~<1s.
# EC2 harness: add to EXTRA_C_TESTS as `test_rollout_cont_ev_bias_action`.
$(BUILD_DIR)/test_rollout_cont_ev_bias_action: tests/test_rollout_cont_ev_bias_action.c \
        $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(SEARCH_DIR)/rollout.h \
        $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(BP_DIR)/bps_reader.h \
        $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(CORE_DIR)/card_abstraction.h \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(BP_DIR)/mccfr_blueprint.h \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_cont_ev_table.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_rollout_cont_ev_bias_action.c \
	    $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c \
	    $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS) -lpthread

test_rollout_cont_ev_bias_action: $(BUILD_DIR)/test_rollout_cont_ev_bias_action
	./$(BUILD_DIR)/test_rollout_cont_ev_bias_action

# ── Dual-boundary build (2026-06-25): TURN-boundary dispatch helper ──
# Pure helper (mccfr_phase_r.c) tested in isolation. Mirrors
# test_phase_r_dispatch for bp_should_dispatch_to_turn_boundary AND re-pins
# that the FLOP helper is byte-untouched at num_board==4. Mac-runnable, <1s.
$(BUILD_DIR)/test_turn_dispatch: tests/test_turn_dispatch.c $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_turn_dispatch.c $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c -o $@ $(LDFLAGS) -lpthread

test_turn_dispatch: $(BUILD_DIR)/test_turn_dispatch
	./$(BUILD_DIR)/test_turn_dispatch

# ── Dual-boundary build: TURN-boundary KEY ALIGNMENT (anti-100%-MISS) ──
# THE key-drift gate: proves producer-write == consumer-read at the turn
# boundary using the REAL functions (not re-typed copies), plus FLOP/TURN
# key disjointness and an end-to-end table round-trip. Links the trainer
# (producer index wrapper) + rollout (consumer index hook) + table.
$(BUILD_DIR)/test_turn_boundary_key_alignment: tests/test_turn_boundary_key_alignment.c \
        $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(SEARCH_DIR)/rollout.h \
        $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(BP_DIR)/bps_reader.h \
        $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(CORE_DIR)/card_abstraction.h \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_cont_ev_table.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_turn_boundary_key_alignment.c \
	    $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c \
	    $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS) -lpthread

test_turn_boundary_key_alignment: $(BUILD_DIR)/test_turn_boundary_key_alignment
	./$(BUILD_DIR)/test_turn_boundary_key_alignment

# ── Dual-boundary build: TURN-boundary consumer contract (rollout) ──
# Mirror of test_rollout_cont_ev_boundary at the turn boundary: happy path
# (uniform mean), MISS/EMPTY_BIAS fail-loud, flag orthogonality.
$(BUILD_DIR)/test_rollout_cont_ev_turn_boundary: tests/test_rollout_cont_ev_turn_boundary.c \
        $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(SEARCH_DIR)/rollout.h \
        $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(BP_DIR)/bps_reader.h \
        $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(CORE_DIR)/card_abstraction.h \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_cont_ev_table.h $(CORE_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_rollout_cont_ev_turn_boundary.c \
	    $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c \
	    $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS) -lpthread

test_rollout_cont_ev_turn_boundary: $(BUILD_DIR)/test_rollout_cont_ev_turn_boundary
	./$(BUILD_DIR)/test_rollout_cont_ev_turn_boundary

# ── Standalone CEVS cache round-trip (Step 15) ──
# Self-contained — links only against mccfr_cont_ev_table.c.
$(BUILD_DIR)/test_cont_ev_standalone_roundtrip: \
        tests/test_cont_ev_standalone_roundtrip.c \
        $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_cont_ev_table.h \
        | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) \
	    tests/test_cont_ev_standalone_roundtrip.c \
	    $(BP_DIR)/mccfr_cont_ev_table.c \
	    -o $@ $(LDFLAGS) -lpthread

test_cont_ev_standalone_roundtrip: $(BUILD_DIR)/test_cont_ev_standalone_roundtrip
	./$(BUILD_DIR)/test_cont_ev_standalone_roundtrip

# ── Stage 1 (2026-07-01): mean-field opponent-bias regret sidecar ──
# Self-contained — links only against mccfr_bias_regrets.c + the shared
# key-hash in mccfr_cont_ev_table.c. Covers the Unit-2 item-6 contract:
# rigged-payoff sampler concentration + uniform fallback, key alignment with
# the cont_ev table, and same-seed determinism. Mac-runnable, <1s.
$(BUILD_DIR)/test_bias_regrets: tests/test_bias_regrets.c \
        $(BP_DIR)/mccfr_bias_regrets.c $(BP_DIR)/mccfr_cont_ev_table.c \
        $(BP_DIR)/mccfr_bias_regrets.h $(BP_DIR)/mccfr_cont_ev_table.h \
        | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_bias_regrets.c \
	    $(BP_DIR)/mccfr_bias_regrets.c $(BP_DIR)/mccfr_cont_ev_table.c \
	    -o $@ $(LDFLAGS) -lpthread

test_bias_regrets: $(BUILD_DIR)/test_bias_regrets
	./$(BUILD_DIR)/test_bias_regrets

# ── Stage 1: mean-field PRODUCER integration (through the trainer) ──
# Drives the 3p postflop fixture with mean-field mode OFF (sidecar never
# allocated — wiring-level byte-identity guard) and ON (sidecar allocated +
# populated, opponent regrets updated, cont_ev still populates). Mac-runnable,
# ~5-15s. Build: `make test_meanfield_producer`.
$(BUILD_DIR)/test_meanfield_producer: tests/test_meanfield_producer.c \
        $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
        $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
        $(BP_DIR)/mccfr_blueprint.h $(BP_DIR)/mccfr_bias_regrets.h $(CORE_DIR)/card_abstraction.h \
        $(CORE_DIR)/hand_eval.h $(BP_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_meanfield_producer.c \
	    $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c $(CORE_DIR)/card_abstraction_ochs.c \
	    $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c \
	    -o $@ $(LDFLAGS)

test_meanfield_producer: $(BUILD_DIR)/test_meanfield_producer
	./$(BUILD_DIR)/test_meanfield_producer

# ── Combined Phase R test target ──
test_phase_r_all: test_phase_r_dispatch test_cont_ev_table \
                  test_cont_bias_strategy test_phase_r_regression_smoke \
                  test_phase_r_heads_rollout test_phase_r_cont_strat_signed \
                  test_bps_query_boundary_ev test_bpr5_roundtrip \
                  test_phase_r_save_load_e2e test_ss_action_hash \
                  test_phase_r_discount_gate test_b3_encoding_alignment \
                  test_b3_bucket_alignment test_phase_r_boundary_bucket \
                  test_rollout_cont_ev_boundary \
                  test_rollout_cont_ev_bias_action \
                  test_cont_ev_standalone_roundtrip \
                  test_turn_all \
                  test_meanfield_all
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Phase R Block A + B.1 + B.2 + B.3 + Sess 11 + Sess 14"
	@echo "  + dual-boundary (turn) — green"
	@echo "═══════════════════════════════════════════════════════════"

# ── Dual-boundary (turn) combined target ──
# The three turn-boundary tests. test_turn_boundary_key_alignment is the
# anti-100%-MISS gate (producer-write == consumer-read).
test_turn_all: test_turn_dispatch test_turn_boundary_key_alignment \
               test_rollout_cont_ev_turn_boundary
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Dual-boundary (turn) — dispatch + key-alignment + consumer — green"
	@echo "═══════════════════════════════════════════════════════════"

# ── Stage 1 (2026-07-01) mean-field combined target ──
# The producer-side mean-field tests: the self-contained sidecar unit test
# (rigged-payoff / key-alignment / determinism) + the through-the-trainer
# integration (mode OFF sidecar-NULL byte-identity guard; mode ON populate).
# Names chosen so both can be passed individually via EXTRA_C_TESTS on EC2.
test_meanfield_all: test_bias_regrets test_meanfield_producer
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Stage 1 mean-field producer — sidecar + integration — green"
	@echo "═══════════════════════════════════════════════════════════"

clean:
	rm -rf $(BUILD_DIR)
