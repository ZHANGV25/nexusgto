CC = gcc
CFLAGS = -O2 -Wall -Wextra \
         -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function
LDFLAGS = -lm

SRC_DIR = src
BUILD_DIR = build

# Every #include uses a bare file name; one -I per source folder resolves them.
INCS = -I$(SRC_DIR) -I$(SRC_DIR)/core -I$(SRC_DIR)/blueprint -I$(SRC_DIR)/search

CORE_DIR   = $(SRC_DIR)/core
BP_DIR     = $(SRC_DIR)/blueprint
SEARCH_DIR = $(SRC_DIR)/search

ifeq ($(OS),Windows_NT)
  SHLIB_EXT = .dll
  SHLIB_FLAGS = -shared -static
else
  SHLIB_EXT = .so
  SHLIB_FLAGS = -shared -fPIC
endif

# ── Source groups ──
EVAL_SRC     = $(CORE_DIR)/hand_eval_table.c
ABSTR_SRC    = $(CORE_DIR)/card_abstraction.c $(CORE_DIR)/card_abstraction_river_nd.c \
               $(CORE_DIR)/card_abstraction_ochs.c $(EVAL_SRC)
READER_SRC   = $(BP_DIR)/bps_reader.c $(BP_DIR)/bps_reader_bucketize.c $(BP_DIR)/bps_joint_canon.c
TRAINER_SRC  = $(BP_DIR)/mccfr_blueprint.c $(BP_DIR)/mccfr_blueprint_io.c
DEPTH_SRC    = $(BP_DIR)/mccfr_phase_r.c $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c
ES_CORE_SRC  = $(SEARCH_DIR)/es_resolver.c $(SEARCH_DIR)/es_resolver_reach.c $(SEARCH_DIR)/es_oracles.c
ROLLOUT_SRC  = $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(SEARCH_DIR)/rollout_bucket_memo.c
TREE_SRC     = $(SEARCH_DIR)/ss_tree_api.c $(SEARCH_DIR)/ss_tree_builder.c

HDRS = $(wildcard $(CORE_DIR)/*.h $(BP_DIR)/*.h $(SEARCH_DIR)/*.h)

# The seven libraries the Python code loads through ctypes.
LIBS = blueprint hand_eval_batch bps_reader rollout es_resolver es_resolver_real street_solve_cpu

# The C tests. Each builds one binary in build/ and runs it; all need only
# files in this repository.
C_TESTS = test_eval5_table test_es_allin_exact test_es_carry_freeze \
          test_es_resolver_primitives test_es_chance_sample_bitexact \
          test_es_sampled_oracle test_es_vec_sampled_oracle test_unequal_stack_abi \
          test_rollout_bias4 test_rollout_allin_leaf test_es_bias_oracle test_pa_emd \
          test_bps_ochs_river test_bps_joint_canon test_depth_limit_dispatch \
          test_cont_bias_strategy test_depth_limit_smoke \
          test_flop_bucket_alignment test_turn_dispatch

.PHONY: all clean test $(LIBS) $(C_TESTS)

all: $(LIBS)

test: $(C_TESTS)
	@echo "C tests: all $(words $(C_TESTS)) passed"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ── Libraries ──

# Blueprint trainer (MCCFR), loaded by precompute/blueprint_worker_unified.py.
blueprint: $(BUILD_DIR)/mccfr_blueprint$(SHLIB_EXT)
$(BUILD_DIR)/mccfr_blueprint$(SHLIB_EXT): $(TRAINER_SRC) $(ABSTR_SRC) $(DEPTH_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -fopenmp $(INCS) $(TRAINER_SRC) $(ABSTR_SRC) $(DEPTH_SRC) -o $@ $(LDFLAGS)

# Batch 7-card evaluator, used by match scoring (AIVAT).
hand_eval_batch: $(BUILD_DIR)/hand_eval_batch$(SHLIB_EXT)
$(BUILD_DIR)/hand_eval_batch$(SHLIB_EXT): $(CORE_DIR)/hand_eval_batch.c $(EVAL_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) $(INCS) $(CORE_DIR)/hand_eval_batch.c $(EVAL_SRC) -o $@ $(LDFLAGS)

# Blueprint reader and turn/river bucketizer (blueprint lookup).
bps_reader: $(BUILD_DIR)/bps_reader$(SHLIB_EXT)
$(BUILD_DIR)/bps_reader$(SHLIB_EXT): $(READER_SRC) $(ABSTR_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) $(INCS) $(READER_SRC) $(ABSTR_SRC) -o $@ $(LDFLAGS)

# Depth-limited leaf values by blueprint rollouts.
rollout: $(BUILD_DIR)/rollout$(SHLIB_EXT)
$(BUILD_DIR)/rollout$(SHLIB_EXT): $(ROLLOUT_SRC) $(READER_SRC) $(ABSTR_SRC) $(DEPTH_SRC) $(TRAINER_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -fopenmp $(INCS) $(ROLLOUT_SRC) $(READER_SRC) $(ABSTR_SRC) \
	    $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c $(TRAINER_SRC) $(BP_DIR)/mccfr_phase_r.c \
	    -o $@ $(LDFLAGS)

# External-sampling re-solver. es_resolver has the table oracles only (used by
# the validators); es_resolver_real adds rollouts and the blueprint reader and
# is the library the search loads. On Windows (MSYS2) fsync is shimmed.
ifeq ($(OS),Windows_NT)
  ES_SHIM = -Dfsync=_commit -include io.h
else
  ES_SHIM =
endif

es_resolver: $(BUILD_DIR)/es_resolver$(SHLIB_EXT)
$(BUILD_DIR)/es_resolver$(SHLIB_EXT): $(ES_CORE_SRC) $(SEARCH_DIR)/es_bias_oracle.c $(EVAL_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) -static-libgcc -fopenmp $(INCS) \
	    $(ES_CORE_SRC) $(SEARCH_DIR)/es_bias_oracle.c $(EVAL_SRC) -o $@ $(LDFLAGS)

ES_REAL_SRC = $(ES_CORE_SRC) $(SEARCH_DIR)/es_bias_oracle.c $(SEARCH_DIR)/es_real_oracle.c \
              $(SEARCH_DIR)/es_sampled_oracle.c $(SEARCH_DIR)/es_vec_sampled_oracle.c \
              $(ROLLOUT_SRC) $(READER_SRC) $(ABSTR_SRC) \
              $(BP_DIR)/mccfr_cont_ev_table.c $(BP_DIR)/mccfr_bias_regrets.c $(TRAINER_SRC) $(BP_DIR)/mccfr_phase_r.c

es_resolver_real: $(BUILD_DIR)/es_resolver_real$(SHLIB_EXT)
$(BUILD_DIR)/es_resolver_real$(SHLIB_EXT): $(ES_REAL_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) -O2 -w $(SHLIB_FLAGS) -static-libgcc -fopenmp -D_GNU_SOURCE $(ES_SHIM) $(INCS) \
	    $(ES_REAL_SRC) -o $@ $(LDFLAGS)

# Search tree builder (engine/street_solver_gpu.py builds trees with it; the
# solve itself runs in es_resolver_real). ss_cpu_stub.c makes the unused
# device-solve entry point abort.
street_solve_cpu: $(BUILD_DIR)/street_solve_cpu$(SHLIB_EXT)
$(BUILD_DIR)/street_solve_cpu$(SHLIB_EXT): $(TREE_SRC) $(SEARCH_DIR)/ss_cpu_stub.c $(HDRS) | $(BUILD_DIR)
	$(CC) -O2 -shared -fPIC $(INCS) $(TREE_SRC) $(SEARCH_DIR)/ss_cpu_stub.c -o $@

# ── C tests ──
# Pattern: build/<test> is the binary, `make <test>` builds and runs it.

$(BUILD_DIR)/test_eval5_table: tests/test_eval5_table.c $(EVAL_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_eval5_table.c $(EVAL_SRC) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_es_allin_exact: tests/test_es_allin_exact.c $(ES_CORE_SRC) $(EVAL_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_es_allin_exact.c $(ES_CORE_SRC) $(EVAL_SRC) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_es_carry_freeze: tests/test_es_carry_freeze.c $(ES_CORE_SRC) $(EVAL_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_es_carry_freeze.c $(ES_CORE_SRC) $(EVAL_SRC) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_es_resolver_primitives: tests/test_es_resolver_primitives.c $(ES_CORE_SRC) $(EVAL_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_es_resolver_primitives.c $(ES_CORE_SRC) $(EVAL_SRC) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_es_chance_sample_bitexact: tests/test_es_chance_sample_bitexact.c $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_es_chance_sample_bitexact.c -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_es_sampled_oracle: tests/test_es_sampled_oracle.c $(SEARCH_DIR)/es_sampled_oracle.c \
        $(SEARCH_DIR)/es_bias_oracle.c $(ES_CORE_SRC) $(SEARCH_DIR)/rollout_bucket_memo.c $(EVAL_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_es_sampled_oracle.c \
	    $(SEARCH_DIR)/es_sampled_oracle.c $(SEARCH_DIR)/es_bias_oracle.c $(ES_CORE_SRC) \
	    $(SEARCH_DIR)/rollout_bucket_memo.c $(EVAL_SRC) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_es_vec_sampled_oracle: tests/test_es_vec_sampled_oracle.c $(SEARCH_DIR)/es_vec_sampled_oracle.c \
        $(ES_CORE_SRC) $(SEARCH_DIR)/rollout_bucket_memo.c $(EVAL_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_es_vec_sampled_oracle.c \
	    $(SEARCH_DIR)/es_vec_sampled_oracle.c $(ES_CORE_SRC) \
	    $(SEARCH_DIR)/rollout_bucket_memo.c $(EVAL_SRC) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_unequal_stack_abi: tests/test_unequal_stack_abi.c $(TREE_SRC) \
        $(SEARCH_DIR)/es_resolver.c $(SEARCH_DIR)/es_resolver_reach.c $(EVAL_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_unequal_stack_abi.c $(TREE_SRC) \
	    $(SEARCH_DIR)/es_resolver.c $(SEARCH_DIR)/es_resolver_reach.c $(EVAL_SRC) -o $@ $(LDFLAGS)

# Links rollout_sampled.c only: the test stubs the rest of the rollout API.
$(BUILD_DIR)/test_rollout_bias4: tests/test_rollout_bias4.c $(SEARCH_DIR)/rollout_sampled.c $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_rollout_bias4.c $(SEARCH_DIR)/rollout_sampled.c -o $@ $(LDFLAGS)

# Uses link-only stubs for the blueprint symbols.
$(BUILD_DIR)/test_rollout_allin_leaf: tests/test_rollout_allin_leaf.c $(SEARCH_DIR)/rollout.c \
        $(SEARCH_DIR)/rollout_sampled.c $(EVAL_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_rollout_allin_leaf.c \
	    $(SEARCH_DIR)/rollout.c $(SEARCH_DIR)/rollout_sampled.c $(EVAL_SRC) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_es_bias_oracle: tests/test_es_bias_oracle.c $(SEARCH_DIR)/es_bias_oracle.c $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_es_bias_oracle.c $(SEARCH_DIR)/es_bias_oracle.c -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_pa_emd: tests/test_pa_emd.c $(ABSTR_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_pa_emd.c $(ABSTR_SRC) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_bps_ochs_river: tests/test_bps_ochs_river.c $(READER_SRC) $(ABSTR_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_bps_ochs_river.c $(READER_SRC) $(ABSTR_SRC) -o $@ $(LDFLAGS)

# The default run includes the complete river class enumerator.
$(BUILD_DIR)/test_bps_joint_canon: tests/test_bps_joint_canon.c $(READER_SRC) $(ABSTR_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fopenmp $(INCS) tests/test_bps_joint_canon.c $(READER_SRC) $(ABSTR_SRC) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_depth_limit_dispatch: tests/test_depth_limit_dispatch.c $(DEPTH_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_depth_limit_dispatch.c $(DEPTH_SRC) -o $@ $(LDFLAGS) -lpthread

$(BUILD_DIR)/test_cont_bias_strategy: tests/test_cont_bias_strategy.c $(DEPTH_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_cont_bias_strategy.c $(DEPTH_SRC) -o $@ $(LDFLAGS) -lpthread

$(BUILD_DIR)/test_turn_dispatch: tests/test_turn_dispatch.c $(DEPTH_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_turn_dispatch.c $(DEPTH_SRC) -o $@ $(LDFLAGS) -lpthread

# Trains twice on one thread with the same seed and byte-compares the saved
# regrets. Built without OpenMP.
$(BUILD_DIR)/test_depth_limit_smoke: tests/test_depth_limit_smoke.c $(TRAINER_SRC) \
        $(ABSTR_SRC) $(DEPTH_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_depth_limit_smoke.c $(TRAINER_SRC) $(ABSTR_SRC) $(DEPTH_SRC) -o $@ $(LDFLAGS)

# Trainer and reader must give the same flop bucket for every query, using
# the shipped texture cache (../data/abstraction, or BPS_TEXTURE_CACHE).
$(BUILD_DIR)/test_flop_bucket_alignment: tests/test_flop_bucket_alignment.c $(READER_SRC) $(TRAINER_SRC) \
        $(ABSTR_SRC) $(DEPTH_SRC) $(HDRS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCS) tests/test_flop_bucket_alignment.c $(READER_SRC) $(TRAINER_SRC) $(ABSTR_SRC) $(DEPTH_SRC) -o $@ $(LDFLAGS)

test_depth_limit_smoke: $(BUILD_DIR)/test_depth_limit_smoke
	./$(BUILD_DIR)/test_depth_limit_smoke 10000 $(BUILD_DIR)

$(filter-out test_depth_limit_smoke,$(C_TESTS)): %: $(BUILD_DIR)/%
	./$(BUILD_DIR)/$@

clean:
	rm -rf $(BUILD_DIR)
