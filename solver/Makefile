CC = gcc
CFLAGS = -O2 -Wall -Wextra \
         -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function
LDFLAGS = -lm

SRC_DIR = src
BENCH_DIR = bench
BUILD_DIR = build

# Detect OS for shared library extension
ifeq ($(OS),Windows_NT)
  SHLIB_EXT = .dll
  SHLIB_FLAGS = -shared -static
else
  SHLIB_EXT = .so
  SHLIB_FLAGS = -shared -fPIC
endif

.PHONY: all clean bench dll test test_phase_1_3 test_ev_walk blueprint bps_reader bps_ev_walk rollout street_solve subgame_preflop

all: blueprint $(BUILD_DIR)/solver_v2$(SHLIB_EXT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ── Blueprint MCCFR engine (production) ──
# Requires both mccfr_blueprint.c and card_abstraction.c (k-means bucketing)
blueprint: $(BUILD_DIR)/mccfr_blueprint$(SHLIB_EXT)

$(BUILD_DIR)/mccfr_blueprint$(SHLIB_EXT): $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/mccfr_cont_ev_table.h $(SRC_DIR)/card_abstraction.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -fopenmp -I$(SRC_DIR) $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c -o $@ $(LDFLAGS)

# ── Street solver v2 (GPU re-solve support library) ──
$(BUILD_DIR)/solver_v2$(SHLIB_EXT): $(SRC_DIR)/solver_v2.c $(SRC_DIR)/solver_v2.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/solver_v2.c -o $@ $(LDFLAGS)

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

$(BUILD_DIR)/subgame_preflop$(SHLIB_EXT): $(SRC_DIR)/subgame_preflop.c $(SRC_DIR)/subgame_preflop.h $(SRC_DIR)/bps_reader.h | $(BUILD_DIR)
	$(CC) -O3 -march=native -Wall -Wextra -Wno-unused-parameter \
	    $(SHLIB_FLAGS) $(SUBGAME_OMP_FLAGS) -I$(SRC_DIR) \
	    $(SRC_DIR)/subgame_preflop.c -o $@ $(LDFLAGS) $(SUBGAME_OMP_LIBS)

# ── Street solver v1 (legacy, for Python bindings compatibility) ──
$(BUILD_DIR)/solver$(SHLIB_EXT): $(SRC_DIR)/solver.c $(SRC_DIR)/solver.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/solver.c -o $@ $(LDFLAGS)

# ── BPS reader (pure reader + turn/river bucketizer) ──
# Links card_abstraction.c for bps_hand_to_bucket_turn/river (EHS/k-means).
# No OpenMP needed — bucketizer is single-hand at a time.
bps_reader: $(BUILD_DIR)/bps_reader$(SHLIB_EXT)

$(BUILD_DIR)/bps_reader$(SHLIB_EXT): $(SRC_DIR)/bps_reader.c $(SRC_DIR)/card_abstraction.c $(SRC_DIR)/bps_reader.h $(SRC_DIR)/card_abstraction.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/bps_reader.c $(SRC_DIR)/card_abstraction.c -o $@ $(LDFLAGS)

# ── BPS EV walk (reader + card_abstraction + OpenMP) ──
bps_ev_walk: $(BUILD_DIR)/bps_ev_walk$(SHLIB_EXT)

$(BUILD_DIR)/bps_ev_walk$(SHLIB_EXT): $(SRC_DIR)/bps_ev_walk.c $(SRC_DIR)/bps_reader.c $(SRC_DIR)/card_abstraction.c $(SRC_DIR)/bps_reader.h $(SRC_DIR)/card_abstraction.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -fopenmp -I$(SRC_DIR) $(SRC_DIR)/bps_ev_walk.c $(SRC_DIR)/bps_reader.c $(SRC_DIR)/card_abstraction.c -o $@ $(LDFLAGS)

# ── Rollout (C port of python/rollout.py) ──
# Depth-limited Pluribus-faithful leaf values for real-time solve.
# Links bps_reader.c (blueprint queries); eval7 is header-only.
rollout: $(BUILD_DIR)/rollout$(SHLIB_EXT)

$(BUILD_DIR)/rollout$(SHLIB_EXT): $(SRC_DIR)/rollout.c $(SRC_DIR)/rollout.h $(SRC_DIR)/bps_reader.c $(SRC_DIR)/bps_reader.h $(SRC_DIR)/card_abstraction.c $(SRC_DIR)/card_abstraction.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -fopenmp -I$(SRC_DIR) $(SRC_DIR)/rollout.c $(SRC_DIR)/bps_reader.c $(SRC_DIR)/card_abstraction.c -o $@ $(LDFLAGS)

# ── Street solve (CUDA — N-player single-street GPU solver) ──
# Builds with nvcc; requires CUDA toolkit on PATH. Targets sm_75 (T4 on
# g4dn) by default; override SS_GPU_ARCH=sm_86 for A10, sm_70 for V100.
# Skip on Mac dev machines (nvcc not present); the .so is consumed by
# python/street_solver_gpu.py at runtime via ctypes on Linux.
NVCC ?= nvcc
SS_GPU_ARCH ?= sm_75
street_solve: $(BUILD_DIR)/street_solve$(SHLIB_EXT)

$(BUILD_DIR)/street_solve$(SHLIB_EXT): $(SRC_DIR)/cuda/street_solve.cu $(SRC_DIR)/cuda/street_solve.cuh $(SRC_DIR)/ss_tree_builder.c $(SRC_DIR)/ss_tree_builder.h | $(BUILD_DIR)
	$(NVCC) -O2 -shared -Xcompiler -fPIC -arch=$(SS_GPU_ARCH) -I$(SRC_DIR) \
	    $(SRC_DIR)/cuda/street_solve.cu $(SRC_DIR)/ss_tree_builder.c -o $@

dll: blueprint $(BUILD_DIR)/solver_v2$(SHLIB_EXT) $(BUILD_DIR)/solver$(SHLIB_EXT) $(BUILD_DIR)/bps_reader$(SHLIB_EXT) $(BUILD_DIR)/bps_ev_walk$(SHLIB_EXT) $(BUILD_DIR)/rollout$(SHLIB_EXT)

# ── Benchmarks ──
$(BUILD_DIR)/bench_v2: $(BENCH_DIR)/bench_v2.c $(SRC_DIR)/solver_v2.c $(SRC_DIR)/solver_v2.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -static -I$(SRC_DIR) $< $(SRC_DIR)/solver_v2.c -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_phase1: $(BENCH_DIR)/test_phase1.c $(SRC_DIR)/solver_v2.c $(SRC_DIR)/solver_v2.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -static -I$(SRC_DIR) $< $(SRC_DIR)/solver_v2.c -o $@ $(LDFLAGS)

bench: $(BUILD_DIR)/bench_v2
	./$(BUILD_DIR)/bench_v2

test: $(BUILD_DIR)/test_phase1
	./$(BUILD_DIR)/test_phase1

# ── Phase 1.3 synthetic ground-truth test ──
# Statically links the blueprint solver for a self-contained test binary.
# Python-side verifier (tests/test_phase_1_3_synthetic.py) consumes the
# bundle this produces and asserts CFR identity + enumerated ground truth.
$(BUILD_DIR)/test_phase_1_3: tests/test_phase_1_3_synthetic.c $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/mccfr_cont_ev_table.h $(SRC_DIR)/card_abstraction.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp -I$(SRC_DIR) tests/test_phase_1_3_synthetic.c $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c -o $@ $(LDFLAGS)

test_phase_1_3: $(BUILD_DIR)/test_phase_1_3
	./$(BUILD_DIR)/test_phase_1_3 $(BUILD_DIR)/phase_1_3_synthetic.tbn
	python3 tests/test_phase_1_3_synthetic.py $(BUILD_DIR)/phase_1_3_synthetic.tbn

# ── BPS EV walk tests ──
$(BUILD_DIR)/test_bps_ev_walk: tests/test_bps_ev_walk.c $(SRC_DIR)/bps_ev_walk.c $(SRC_DIR)/bps_reader.c $(SRC_DIR)/card_abstraction.c $(SRC_DIR)/bps_reader.h $(SRC_DIR)/card_abstraction.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp -I$(SRC_DIR) tests/test_bps_ev_walk.c $(SRC_DIR)/bps_ev_walk.c $(SRC_DIR)/bps_reader.c $(SRC_DIR)/card_abstraction.c -o $@ $(LDFLAGS) -static

test_ev_walk: $(BUILD_DIR)/test_bps_ev_walk
	./$(BUILD_DIR)/test_bps_ev_walk

# ── PA-EMD tests (ca_emd_1d + ca_compute_ehs_distribution) ──
$(BUILD_DIR)/test_pa_emd: tests/test_pa_emd.c $(SRC_DIR)/card_abstraction.c $(SRC_DIR)/card_abstraction.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_pa_emd.c $(SRC_DIR)/card_abstraction.c -o $@ $(LDFLAGS)

test_pa_emd: $(BUILD_DIR)/test_pa_emd
	./$(BUILD_DIR)/test_pa_emd

# ── BPS OCHS river bucketer regression test ──
# Pins the Phase B fix: bps_init_ochs_river + bps_hand_to_bucket_river
# routing through the OCHS path on representative river fixtures.
$(BUILD_DIR)/test_bps_ochs_river: tests/test_bps_ochs_river.c $(SRC_DIR)/bps_reader.c $(SRC_DIR)/card_abstraction.c $(SRC_DIR)/bps_reader.h $(SRC_DIR)/card_abstraction.h $(SRC_DIR)/hand_eval.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_bps_ochs_river.c $(SRC_DIR)/bps_reader.c $(SRC_DIR)/card_abstraction.c -o $@ $(LDFLAGS)

test_bps_ochs_river: $(BUILD_DIR)/test_bps_ochs_river
	./$(BUILD_DIR)/test_bps_ochs_river

# ── Phase R Block A.1: dispatch helper unit tests ──
# Pure helper (mccfr_phase_r.c) tested in isolation — no card_abstraction
# / hand_eval / OMP needed. Mac-runnable, <1s wall.
$(BUILD_DIR)/test_phase_r_dispatch: tests/test_phase_r_dispatch.c $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_phase_r_dispatch.c $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c -o $@ $(LDFLAGS) -lpthread

test_phase_r_dispatch: $(BUILD_DIR)/test_phase_r_dispatch
	./$(BUILD_DIR)/test_phase_r_dispatch

# ── Phase R Block A.2: cont-ev table unit tests ──
# Tests the boundary EV slot table (mccfr_cont_ev_table.{c,h}) in
# isolation. Uses OpenMP for the multi-thread CAS test (T5) — falls
# back to a SKIP if OMP isn't available (Mac without libomp).
$(BUILD_DIR)/test_cont_ev_table: tests/test_cont_ev_table.c $(SRC_DIR)/mccfr_cont_ev_table.c $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_cont_ev_table.c $(SRC_DIR)/mccfr_cont_ev_table.c -o $@ $(LDFLAGS) -lpthread

test_cont_ev_table: $(BUILD_DIR)/test_cont_ev_table
	./$(BUILD_DIR)/test_cont_ev_table

# ── Phase R Block A.2: cont-bias reweighting helper unit tests ──
# Pure math (bp_apply_cont_bias in mccfr_phase_r.c). Mac-runnable, <1s.
$(BUILD_DIR)/test_cont_bias_strategy: tests/test_cont_bias_strategy.c $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_cont_bias_strategy.c $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c -o $@ $(LDFLAGS) -lpthread

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
        $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
        $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
        $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/card_abstraction.h \
        $(SRC_DIR)/hand_eval.h $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_phase_r_regression_smoke.c \
	    $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
	    $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
	    -o $@ $(LDFLAGS)

test_phase_r_regression_smoke: $(BUILD_DIR)/test_phase_r_regression_smoke
	./$(BUILD_DIR)/test_phase_r_regression_smoke 10000

# ── Phase R Block A.2: HEADS-branch rollout integration ──
# Drives the trainer with depth_limit_cont_strats=1, prob=1.0 on a 3p
# postflop fixture. Verifies the HEADS-branch rollout dispatcher fires,
# allocates cont-ev table slots, and accumulates ev_count > 0 with
# finite ev_sum values. Mac-runnable, ~10-20s.
$(BUILD_DIR)/test_phase_r_heads_rollout: tests/test_phase_r_heads_rollout.c \
        $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
        $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
        $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/card_abstraction.h \
        $(SRC_DIR)/hand_eval.h $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_phase_r_heads_rollout.c \
	    $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
	    $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
	    -o $@ $(LDFLAGS)

test_phase_r_heads_rollout: $(BUILD_DIR)/test_phase_r_heads_rollout
	./$(BUILD_DIR)/test_phase_r_heads_rollout

# ── Phase R Block A.2: cont-bias signed-regrets correctness ──
# Drives 3p-AA-on-flop fixture with prob=1.0, asserts raise-bias EV >
# fold-bias EV (gates against bias-index swap, sign-inversion bugs).
# Plus T3: bp_compute_action_hash determinism (cont-bias-free by
# signature). Mac-runnable, ~10-15s.
$(BUILD_DIR)/test_phase_r_cont_strat_signed: tests/test_phase_r_cont_strat_signed.c \
        $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
        $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
        $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/card_abstraction.h \
        $(SRC_DIR)/hand_eval.h $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_phase_r_cont_strat_signed.c \
	    $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
	    $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
	    -o $@ $(LDFLAGS)

test_phase_r_cont_strat_signed: $(BUILD_DIR)/test_phase_r_cont_strat_signed
	./$(BUILD_DIR)/test_phase_r_cont_strat_signed

# ── Phase R Block B.2: bps_query_boundary_ev contract ──
# Pure-table test; no trainer dep. ~<1s wall.
$(BUILD_DIR)/test_bps_query_boundary_ev: tests/test_bps_query_boundary_ev.c \
        $(SRC_DIR)/mccfr_cont_ev_table.c $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_bps_query_boundary_ev.c \
	    $(SRC_DIR)/mccfr_cont_ev_table.c -o $@ $(LDFLAGS) -lpthread

test_bps_query_boundary_ev: $(BUILD_DIR)/test_bps_query_boundary_ev
	./$(BUILD_DIR)/test_bps_query_boundary_ev

# ── Phase R Block B.1: BPR5 writer/reader round-trip ──
# Drives the full trainer (5K iters, prob=1.0) on a 3p postflop fixture,
# bp_save_regrets to /tmp, bp_load_regrets into a fresh BPSolver, asserts
# byte-exact match on every (action_hash, player, bucket) → (ev_sum[4],
# ev_count[4]) tuple. Mac-runnable, ~10-20s.
$(BUILD_DIR)/test_bpr5_roundtrip: tests/test_bpr5_roundtrip.c \
        $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
        $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
        $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/card_abstraction.h \
        $(SRC_DIR)/hand_eval.h $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_bpr5_roundtrip.c \
	    $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
	    $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
	    -o $@ $(LDFLAGS)

test_bpr5_roundtrip: $(BUILD_DIR)/test_bpr5_roundtrip
	./$(BUILD_DIR)/test_bpr5_roundtrip

# ── Phase R Block B.1+B.2: end-to-end save → load → query parity ──
# 5K-iter trainer drive with prob=1.0; capture one populated slot's
# (ev_sum, ev_count); save+load; query via bp_query_boundary_ev; assert
# byte-exact match on stored fields and OK status with the divided ratio.
$(BUILD_DIR)/test_phase_r_save_load_e2e: tests/test_phase_r_save_load_e2e.c \
        $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
        $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
        $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/card_abstraction.h \
        $(SRC_DIR)/hand_eval.h $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_phase_r_save_load_e2e.c \
	    $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
	    $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
	    -o $@ $(LDFLAGS)

test_phase_r_save_load_e2e: $(BUILD_DIR)/test_phase_r_save_load_e2e
	./$(BUILD_DIR)/test_phase_r_save_load_e2e

# ── Phase R Block B.3: ss_compute_leaf_action_hashes contract ──
# CPU-only Mac-runnable test — hand-builds SSNode trees, validates the
# kernel-side root-to-leaf walk reproduces bp_compute_action_hash for
# every leaf. PASS gate per spec §9.
$(BUILD_DIR)/test_ss_action_hash: tests/test_ss_action_hash.c \
        $(SRC_DIR)/ss_action_hash.c $(SRC_DIR)/ss_action_hash.h \
        $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
        $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
        $(SRC_DIR)/cuda/street_solve.cuh \
        $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/card_abstraction.h \
        $(SRC_DIR)/hand_eval.h $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_ss_action_hash.c \
	    $(SRC_DIR)/ss_action_hash.c \
	    $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
	    $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
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
        $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
        $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
        $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/card_abstraction.h \
        $(SRC_DIR)/hand_eval.h $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_phase_r_discount_gate.c \
	    $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
	    $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
	    -o $@ $(LDFLAGS)

test_phase_r_discount_gate: $(BUILD_DIR)/test_phase_r_discount_gate
	./$(BUILD_DIR)/test_phase_r_discount_gate

# ── Phase R Block B.3 (Step 6): encoding-alignment PASS gate ──
# Cross-validates kernel-built tree action emission against the
# trainer's bp_generate_actions byte-for-byte, on v3/phase3 production
# config. Mac-runnable via the ss_tree_builder.{c,h} extraction.
$(BUILD_DIR)/test_b3_encoding_alignment: tests/test_b3_encoding_alignment.c \
        $(SRC_DIR)/ss_tree_builder.c $(SRC_DIR)/ss_tree_builder.h \
        $(SRC_DIR)/ss_action_hash.c $(SRC_DIR)/ss_action_hash.h \
        $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
        $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
        $(SRC_DIR)/cuda/street_solve.cuh \
        $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/card_abstraction.h \
        $(SRC_DIR)/hand_eval.h $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_b3_encoding_alignment.c \
	    $(SRC_DIR)/ss_tree_builder.c $(SRC_DIR)/ss_action_hash.c \
	    $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
	    $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
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
        $(SRC_DIR)/bps_reader.c $(SRC_DIR)/bps_reader.h \
        $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
        $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
        $(SRC_DIR)/mccfr_blueprint.h $(SRC_DIR)/card_abstraction.h \
        $(SRC_DIR)/hand_eval.h $(SRC_DIR)/mccfr_cont_ev_table.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) tests/test_b3_bucket_alignment.c \
	    $(SRC_DIR)/bps_reader.c \
	    $(SRC_DIR)/mccfr_blueprint.c $(SRC_DIR)/card_abstraction.c \
	    $(SRC_DIR)/mccfr_phase_r.c $(SRC_DIR)/mccfr_cont_ev_table.c \
	    -o $@ $(LDFLAGS)

test_b3_bucket_alignment: $(BUILD_DIR)/test_b3_bucket_alignment
	./$(BUILD_DIR)/test_b3_bucket_alignment

# ── Combined Phase R test target ──
test_phase_r_all: test_phase_r_dispatch test_cont_ev_table \
                  test_cont_bias_strategy test_phase_r_regression_smoke \
                  test_phase_r_heads_rollout test_phase_r_cont_strat_signed \
                  test_bps_query_boundary_ev test_bpr5_roundtrip \
                  test_phase_r_save_load_e2e test_ss_action_hash \
                  test_phase_r_discount_gate test_b3_encoding_alignment \
                  test_b3_bucket_alignment
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Phase R Block A + B.1 + B.2 + B.3 — all green"
	@echo "═══════════════════════════════════════════════════════════"

clean:
	rm -rf $(BUILD_DIR)
