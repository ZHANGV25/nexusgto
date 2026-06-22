/**
 * mccfr_blueprint_io.c — blueprint regret/texture serialization (save/load).
 *
 * Split out of mccfr_blueprint.c (no behavior change). Holds the BPR4/BPR5
 * regret file writer/loader (serial + mmap-parallel), the BPR5 cont-bias
 * trailer loader, the BPFileMap mmap shim, and the texture-cache save/load.
 *
 * Compiled into the SAME shared libraries as mccfr_blueprint.c. The four
 * parent helpers it calls (arena_alloc, texture_index_rebuild,
 * info_table_find_or_create, ensure_strategy_sum) are shared via
 * mccfr_blueprint_internal.h (promoted from file-static, defs stay in the
 * parent TU). Everything else used here is declared in mccfr_blueprint.h /
 * mccfr_cont_ev_table.h.
 */

#include "mccfr_blueprint.h"
#include "mccfr_blueprint_internal.h"
#include "card_abstraction.h"
#include "hand_eval.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sched.h>
#else
/* Windows/MSYS2 compatibility shims — mirror mccfr_blueprint.c. */
#include <windows.h>
static inline void sched_yield(void) { SwitchToThread(); }
static inline long sysconf(int name) {
    (void)name;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (long)si.dwPageSize;
}
#define _SC_PAGESIZE 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* CHK_FW: write `count` items of `size` bytes from `ptr` to `f`. If the
 * write is short (disk full, etc.), goto write_failed which closes the
 * file and returns -2. The 2026-04-29 incident: bp_save_regrets returned
 * 0 ("success") even though the C runtime's fwrite had hit ENOSPC and
 * dropped 200M entries silently. Header counter said 1.466B entries,
 * actual file contained 1.265B. Now any short write aborts the save
 * immediately so the caller knows to retry/cleanup. */
#define CHK_FW(ptr, size, count, f) do {                                   \
    size_t _n = fwrite((ptr), (size), (count), (f));                       \
    if (_n != (size_t)(count)) {                                           \
        fprintf(stderr,                                                    \
                "[BP] bp_save_regrets: short fwrite (%zu/%zu items) — "    \
                "likely ENOSPC. Aborting save.\n", _n, (size_t)(count));   \
        goto write_failed;                                                 \
    }                                                                      \
} while(0)

int bp_save_regrets(const BPSolver *s, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    const BPInfoTable *t = &s->info_table;

    /* Header: "BPR4" = bucket-in-key format v4 (int64 table_size, int64 iterations) */
    CHK_FW("BPR4", 1, 4, f);
    CHK_FW(&t->table_size, sizeof(int64_t), 1, f);
    CHK_FW(&t->num_entries, sizeof(int64_t), 1, f);
    int64_t iters64 = s->iterations_run;
    CHK_FW(&iters64, sizeof(int64_t), 1, f);

    /* Entries — each info set has bucket in key, regrets[num_actions] only */
    int64_t written = 0;
    for (int64_t i = 0; i < t->table_size; i++) {
        if (t->occupied[i] != 1) continue;
        BPInfoKey *key = &t->keys[i];
        BPInfoSet *is = &t->sets[i];

        CHK_FW(&key->player, sizeof(int), 1, f);
        CHK_FW(&key->street, sizeof(int), 1, f);
        CHK_FW(&key->bucket, sizeof(int), 1, f);
        CHK_FW(&key->board_hash, sizeof(uint64_t), 1, f);
        CHK_FW(&key->action_hash, sizeof(uint64_t), 1, f);
        CHK_FW(&is->num_actions, sizeof(int), 1, f);

        CHK_FW(is->regrets, sizeof(int), is->num_actions, f);

        int has_ss = (is->strategy_sum != NULL) ? 1 : 0;
        CHK_FW(&has_ss, sizeof(int), 1, f);
        if (has_ss) {
            CHK_FW(is->strategy_sum, sizeof(float), is->num_actions, f);
        }
        written++;
    }

    /* ── BPR5 trailer (Phase R Block B.1) ──
     *
     * Append boundary cont-bias EV slots iff Phase R was active (flag ON)
     * AND at least one slot accumulated. Either condition false → no
     * trailer, file ends at BPR4 entries — bit-identical to legacy.
     * This is the load-bearing invariant for test_phase_r_regression_smoke.
     *
     * Wire layout (spec §5):
     *   "BPR5"       (4 B)
     *   int64 n_slots
     *   per slot: int player | int bucket | uint64 action_hash
     *           | float ev_sum[4] | int64 ev_count[4]
     *
     * Field order on the wire (player, bucket, action_hash) differs from
     * the in-memory BPContEVEntry struct order — DO NOT struct-fwrite.
     * `occupied` is in-memory state, not on disk. */
    if (s->depth_limit_cont_strats == 1 &&
        bp_cont_ev_count(&s->cont_ev_table) > 0) {
        /* Step 15: delegate to the shared trailer helper so the
         * standalone CEVS cache path can't drift from this writer. The
         * helper writes "BPR5" + n_slots + slots and returns -1 on
         * fwrite error or count-mismatch (the same data-corruption
         * sentinel the inline code used to check). */
        int64_t n_slots_before = bp_cont_ev_count(&s->cont_ev_table);
        if (bp_cont_ev_write_trailer_body(f, &s->cont_ev_table) != 0) {
            fprintf(stderr,
                    "[BP] bp_save_regrets: write_trailer_body failed "
                    "(n_slots=%lld). Aborting save.\n",
                    (long long)n_slots_before);
            goto write_failed;
        }
        printf("[BP] BPR5 trailer: %lld boundary slots\n",
               (long long)n_slots_before);
    }

    /* Final flush — fclose may also report errors that prior fwrite missed
     * because of buffering. Treat fclose failure as ENOSPC too. */
    if (fflush(f) != 0) {
        fprintf(stderr, "[BP] bp_save_regrets: fflush failed\n");
        goto write_failed;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "[BP] bp_save_regrets: fclose failed\n");
        return -2;
    }
    printf("[BP] Saved %lld info sets to %s\n", (long long)written, path);
    return 0;

write_failed:
    fclose(f);
    /* Best-effort delete the partial file so callers don't accidentally use it. */
    unlink(path);
    return -2;
}

/* ── Regret loader ────────────────────────────────────────────────
 *
 * Two implementations share the same public entry point bp_load_regrets:
 *
 *   (a) load_regrets_serial: legacy fread-based loader. Always used for
 *       BPR2/BPR3 format (which we don't generate anymore but still have
 *       to be able to read). Also the fallback when BP_LEGACY_LOADER=1.
 *
 *   (b) load_regrets_mmap_parallel: BPR4-only. mmaps the whole file,
 *       walks once serially to build an offset index, then dispatches
 *       threads to process the index in parallel. ~20x faster on
 *       r6i.8xlarge for the 1.5B v2 checkpoint (60 GB).
 *
 * Set BP_LEGACY_LOADER=1 in the environment to force the serial loader
 * on BPR4 files too — useful for correctness diffing old vs new.
 *
 * Both loaders MUST produce an identical final hash table state (same
 * slot assignments, same regret sums, same strategy_sum state). A
 * round-trip test in tests/test_phase_1_3_synthetic.c validates this on
 * a toy checkpoint.
 *
 * The key correctness invariants preserved in BOTH loaders:
 *   1. Regrets are ADDED, not written. Fresh arena slots have zeroed
 *      regrets, so + is idempotent on fresh loads, and recovers split
 *      regrets on duplicate-key Hogwild collisions.
 *   2. strategy_sum is lazily arena-allocated on first encounter.
 *   3. info_table_find_or_create uses the existing Hogwild-safe CAS;
 *      its return slot is thread-safe to write to (each thread writes
 *      to its own distinct slot after the CAS). */

static int64_t load_regrets_serial(BPSolver *s, FILE *f,
                                    int is_v4, int is_v3, int is_v2,
                                    int64_t saved_entries,
                                    int64_t saved_iters,
                                    int64_t *out_post_entries_off) {
    (void)is_v3; (void)is_v2;
    BPInfoTable *t = &s->info_table;
    int64_t loaded = 0;
    int64_t merged = 0;

    /* BP_RESET_STRATEGY_SUM=1: load regrets normally, but DROP the
     * strategy_sum portion of every entry. Use case: recover from a
     * corrupted average (e.g. the iter-2.8B hash-resize info-set loss
     * documented in PHASE3_RETRAIN_STATUS / Phase B diagnostic). The
     * regret state is preserved (fresh σ matches the converged regrets);
     * only the time-averaged σ̄ is reset to zero so subsequent training
     * rebuilds the average uniformly across all slots. */
    const char *reset_env = getenv("BP_RESET_STRATEGY_SUM");
    int reset_ss = (reset_env && reset_env[0] == '1');
    if (reset_ss) {
        printf("[BP] BP_RESET_STRATEGY_SUM=1: dropping strategy_sum from all entries\n");
        fflush(stdout);
    }

    int tmp_regrets[BP_MAX_ACTIONS];
    float tmp_ss[BP_MAX_ACTIONS];

    for (int64_t e = 0; e < saved_entries; e++) {
        BPInfoKey key;
        int na;

        if (fread(&key.player, sizeof(int), 1, f) != 1) break;
        if (fread(&key.street, sizeof(int), 1, f) != 1) break;
        if (fread(&key.bucket, sizeof(int), 1, f) != 1) break;
        if (fread(&key.board_hash, sizeof(uint64_t), 1, f) != 1) break;
        if (fread(&key.action_hash, sizeof(uint64_t), 1, f) != 1) break;
        if (fread(&na, sizeof(int), 1, f) != 1) break;
        if (na > BP_MAX_ACTIONS) na = BP_MAX_ACTIONS;

        int64_t slot = info_table_find_or_create(t, key, na);
        if (slot < 0) {
            fseek(f, na * sizeof(int), SEEK_CUR);
            int has_ss;
            fread(&has_ss, sizeof(int), 1, f);
            if (has_ss) fseek(f, na * sizeof(float), SEEK_CUR);
            continue;
        }

        BPInfoSet *is = &t->sets[slot];

        /* Read regrets into a temp buffer, check if the slot already
         * has data (= duplicate key from the Hogwild race bug), then
         * ADD to the slot. For fresh slots (arena zeroed): 0 + new =
         * new. For duplicate keys: existing + new = merged sum.
         *
         * NOTE: the is_dup counter is a LOWER BOUND, not an exact
         * count. If a pair of duplicate regrets happens to sum to
         * zero in every slot, a third duplicate would see all-zero
         * regrets and be mis-reported as "not a duplicate". The
         * arithmetic is still correct (the add is idempotent on the
         * zero state); only the `[BP] WARNING: merged N` log line
         * would under-count. Rare in practice on non-trivially
         * trained checkpoints. */
        fread(tmp_regrets, sizeof(int), na, f);
        int is_dup = 0;
        for (int a = 0; a < na; a++) {
            if (is->regrets[a] != 0) { is_dup = 1; break; }
        }
        for (int a = 0; a < na; a++) {
            is->regrets[a] += tmp_regrets[a];
        }

        int has_ss;
        fread(&has_ss, sizeof(int), 1, f);
        if (has_ss) {
            if (reset_ss) {
                /* Skip the strategy_sum bytes; leave is->strategy_sum NULL
                 * so first traversal lazy-allocates it to zero. */
                fseek(f, na * sizeof(float), SEEK_CUR);
            } else {
                if (!is->strategy_sum) {
                    is->strategy_sum = (float*)arena_alloc(na);
                }
                if (is->strategy_sum) {
                    fread(tmp_ss, sizeof(float), na, f);
                    for (int a = 0; a < na; a++)
                        is->strategy_sum[a] += tmp_ss[a];
                } else {
                    fseek(f, na * sizeof(float), SEEK_CUR);
                }
            }
        }

        if (is_dup) {
            merged++;
            if (merged <= 50) {
                printf("[BP] Merged duplicate: player=%d street=%d bucket=%d "
                       "ah=%016llx\n", key.player, key.street, key.bucket,
                       (unsigned long long)key.action_hash);
            }
        }
        loaded++;
    }

    /* Capture the byte offset just past the last BPR4 entry — used by
     * bp_load_regrets to detect a BPR5 trailer (Phase R Block B.2). On
     * truncated input this points mid-entry; the trailer parser will
     * surface the corruption via magic mismatch. */
    if (out_post_entries_off) {
        long pos = ftell(f);
        *out_post_entries_off = (pos < 0) ? -1 : (int64_t)pos;
    }

    s->iterations_run = saved_iters;

    t->insertion_failures = 0;
    t->max_probe_observed = 0;

    if (merged > 0) {
        printf("[BP] WARNING: merged %lld duplicate entries (Hogwild race bug). "
               "Split regrets have been recovered.\n", (long long)merged);
    }
    printf("[BP] Loaded %lld/%lld info sets (%lld merged) [serial], table %lld/%lld\n",
           (long long)loaded, (long long)saved_entries, (long long)merged,
           (long long)t->num_entries, (long long)t->table_size);
    return loaded;
}

/* ── mmap compat shim ──────────────────────────────────────────────
 *
 * Linux: standard POSIX mmap.
 * Windows (MSYS2): CreateFileMapping + MapViewOfFile.
 *
 * We only need a read-only private mapping of the whole file. */
typedef struct {
    const unsigned char *base;
    size_t size;
#ifdef _WIN32
    HANDLE h_file;
    HANDLE h_map;
#else
    int fd;
#endif
} BPFileMap;

static int bp_file_map_open(BPFileMap *m, const char *path) {
    memset(m, 0, sizeof(*m));
#ifdef _WIN32
    m->h_file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m->h_file == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(m->h_file, &sz)) {
        CloseHandle(m->h_file); return -1;
    }
    m->size = (size_t)sz.QuadPart;
    m->h_map = CreateFileMappingA(m->h_file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m->h_map) { CloseHandle(m->h_file); return -1; }
    m->base = (const unsigned char*)MapViewOfFile(m->h_map, FILE_MAP_READ, 0, 0, 0);
    if (!m->base) { CloseHandle(m->h_map); CloseHandle(m->h_file); return -1; }
    return 0;
#else
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) return -1;
    struct stat st;
    if (fstat(m->fd, &st) < 0) { close(m->fd); return -1; }
    m->size = (size_t)st.st_size;
    m->base = (const unsigned char*)mmap(NULL, m->size, PROT_READ,
                                          MAP_PRIVATE, m->fd, 0);
    if (m->base == MAP_FAILED) { close(m->fd); return -1; }
    /* MADV_SEQUENTIAL hints the kernel to aggressively read ahead and
     * drop pages behind us. Good for a streaming one-pass walk. */
#ifdef MADV_SEQUENTIAL
    madvise((void*)m->base, m->size, MADV_SEQUENTIAL);
#endif
    return 0;
#endif
}

static void bp_file_map_close(BPFileMap *m) {
#ifdef _WIN32
    if (m->base) UnmapViewOfFile(m->base);
    if (m->h_map) CloseHandle(m->h_map);
    if (m->h_file != INVALID_HANDLE_VALUE) CloseHandle(m->h_file);
#else
    if (m->base && m->base != MAP_FAILED)
        munmap((void*)m->base, m->size);
    if (m->fd >= 0) close(m->fd);
#endif
    memset(m, 0, sizeof(*m));
}

/* BPR4 parallel loader using mmap + two-pass index.
 *
 * Pass 1 (serial, single thread):
 *   Walk the file from offset data_start (just past the 28-byte BPR4
 *   header) to eof. For each entry, read num_actions and has_ss in
 *   order to know its on-disk size, then record the entry's byte
 *   offset into an int64 array. No hash inserts, no allocations, just
 *   linear pointer arithmetic. Pages stream into cache as we touch
 *   them. Expected time: ~60-90s for 1.2B entries on r6i.8xlarge.
 *
 * Pass 2 (parallel, #OMP threads):
 *   Each thread takes a chunk of the index range. For each entry, read
 *   the full entry directly from the mmap'd pointer, call
 *   info_table_find_or_create, and merge regrets + strategy_sum into
 *   the slot. The find-or-create CAS is already lock-free; the only
 *   thing we need to preserve is the "fresh slot zeroed -> add is
 *   safe" invariant, which holds because the arena is calloc'd.
 *
 * Merge safety: two threads MAY race on the same slot if the original
 * training run had duplicate keys (Bug 11 / Hogwild race) — the serial
 * loader relies on "previous += new" being sequential, but under
 * parallelism two threads could race on the same regret[] slot. We fix
 * this with atomic fetch_add on each regret array entry. strategy_sum
 * allocation uses the same CAS pattern as ensure_strategy_sum. */
static int64_t load_regrets_mmap_parallel(BPSolver *s, const char *path,
                                           int64_t saved_entries,
                                           int64_t saved_iters,
                                           int64_t *out_post_entries_off) {
    BPFileMap mfile;
    if (bp_file_map_open(&mfile, path) != 0) {
        fprintf(stderr, "[BP] mmap failed on %s, falling back to serial\n", path);
        return -2;  /* caller will retry with serial */
    }
    printf("[BP] mmap: %zu bytes\n", mfile.size); fflush(stdout);

    /* BP_RESET_STRATEGY_SUM=1: see serial loader for full rationale.
     * Skip strategy_sum allocation/load on every entry; leave NULL for
     * lazy-allocate-to-zero on first traversal. Pass 1 byte-walk still
     * advances past the strategy_sum bytes (file layout is unchanged);
     * only Pass 2's per-thread merge skips the read+atomic-add. */
    const char *reset_env = getenv("BP_RESET_STRATEGY_SUM");
    int reset_ss = (reset_env && reset_env[0] == '1');
    if (reset_ss) {
        printf("[BP] BP_RESET_STRATEGY_SUM=1: dropping strategy_sum from all entries\n");
        fflush(stdout);
    }

    /* Data starts just past the 28-byte BPR4 header (magic 4 + table_size 8 +
     * num_entries 8 + iterations 8). */
    const size_t data_start = 28;
    const unsigned char *base = mfile.base;
    size_t pos = data_start;
    size_t eof = mfile.size;

    /* ── Pass 1: build offset index ──
     *
     * Pre-allocate enough slots for saved_entries; if the walk finds fewer
     * than that (truncated file), we shrink. On corrupt file we bail.
     * Allocating 8 bytes × 1.2B entries = 9.6 GB — not tiny but manageable
     * on r6i.8xlarge (256 GB) and on the PC's 64 GB for the 200M checkpoint
     * (which only needs ~1.6 GB of index). */
    int64_t *offsets = (int64_t*)malloc((size_t)saved_entries * sizeof(int64_t));
    if (!offsets) {
        fprintf(stderr, "[BP] OOM allocating %lld-entry offset index\n",
                (long long)saved_entries);
        bp_file_map_close(&mfile);
        return -1;
    }

    printf("[BP] Pass 1: building offset index (serial walk)...\n"); fflush(stdout);
    double t0 = 0.0;
#ifdef _OPENMP
    t0 = omp_get_wtime();
#endif

    int64_t n_indexed = 0;
    int64_t last_progress = 0;
    for (; n_indexed < saved_entries; n_indexed++) {
        if (pos + 32 > eof) {
            fprintf(stderr, "[BP] Pass 1: unexpected EOF at offset %zu, "
                    "entry %lld / %lld\n", pos, (long long)n_indexed,
                    (long long)saved_entries);
            break;
        }
        offsets[n_indexed] = (int64_t)pos;

        /* Read na without touching regrets — we need it to compute the
         * entry size. Key fields are: player(4) + street(4) + bucket(4)
         * + board_hash(8) + action_hash(8) = 28 bytes, then na at offset
         * 28 from entry start. */
        int na;
        memcpy(&na, base + pos + 28, sizeof(int));
        if (na < 0 || na > BP_MAX_ACTIONS) {
            fprintf(stderr, "[BP] Pass 1: corrupt na=%d at offset %zu\n",
                    na, pos);
            free(offsets); bp_file_map_close(&mfile); return -1;
        }

        /* Advance past key(28) + na(4) + regrets(4*na) + has_ss(4) */
        size_t next = pos + 28 + 4 + (size_t)(4 * na) + 4;
        if (next > eof) {
            fprintf(stderr, "[BP] Pass 1: truncated regrets at entry %lld\n",
                    (long long)n_indexed);
            break;
        }
        int has_ss;
        memcpy(&has_ss, base + pos + 28 + 4 + (size_t)(4 * na), sizeof(int));
        if (has_ss) {
            next += (size_t)(4 * na);
            if (next > eof) {
                fprintf(stderr, "[BP] Pass 1: truncated strategy_sum at entry %lld\n",
                        (long long)n_indexed);
                break;
            }
        }
        pos = next;

        if (n_indexed - last_progress >= 100000000) {
            last_progress = n_indexed;
#ifdef _OPENMP
            double dt = omp_get_wtime() - t0;
            printf("[BP]   indexed %lld / %lld entries (%.1fs, %.0f M/s)\n",
                   (long long)n_indexed, (long long)saved_entries, dt,
                   (double)n_indexed / (dt * 1e6));
#else
            printf("[BP]   indexed %lld / %lld entries\n",
                   (long long)n_indexed, (long long)saved_entries);
#endif
            fflush(stdout);
        }
    }

#ifdef _OPENMP
    double pass1_sec = omp_get_wtime() - t0;
    printf("[BP] Pass 1 done: %lld entries in %.1fs\n",
           (long long)n_indexed, pass1_sec);
#else
    printf("[BP] Pass 1 done: %lld entries\n", (long long)n_indexed);
#endif
    fflush(stdout);

    /* Capture the byte offset just past the last BPR4 entry — used by
     * bp_load_regrets to detect a BPR5 trailer (Phase R Block B.2). On
     * truncated input pos is at the truncation; the trailer parser will
     * either see EOF (legacy) or surface the corruption via magic check. */
    if (out_post_entries_off) {
        *out_post_entries_off = (int64_t)pos;
    }

    /* Step 15 short-circuit: BP_LOAD_TRAILER_ONLY=1 skips Pass 2's
     * BPR4 hash inserts entirely. Use this for cont_ev_table-only
     * consumers (dump_cont_ev_table → CEVS cache) where the BPInfoTable
     * load is wasted work — the hash thrash on a heavily-undersized
     * BPInfoTable (e.g. 128M slots vs 2.4B entries) takes hours.
     *
     * Skipping leaves info_table empty; the caller MUST NOT then use
     * any code path that queries it. Trailer is still parsed (the BPR5
     * load loop in bp_load_regrets's wrapper runs after this function
     * returns successfully). */
    {
        const char *trailer_only = getenv("BP_LOAD_TRAILER_ONLY");
        if (trailer_only && trailer_only[0] == '1') {
            printf("[BP] Pass 2: SKIPPED (BP_LOAD_TRAILER_ONLY=1; "
                   "info_table left empty; trailer parse continues)\n");
            fflush(stdout);
            free(offsets);
            bp_file_map_close(&mfile);
            s->iterations_run = saved_iters;
            return 0;
        }
    }

    /* ── Pass 2: parallel hash inserts ── */
    BPInfoTable *t = &s->info_table;
    int64_t loaded = 0;
    int64_t merged = 0;
    int64_t dropped = 0;

#ifdef _OPENMP
    t0 = omp_get_wtime();
#endif
    printf("[BP] Pass 2: parallel hash inserts (%d threads)...\n",
#ifdef _OPENMP
           omp_get_max_threads()
#else
           1
#endif
    ); fflush(stdout);

    int64_t progress_stride = n_indexed / 20;
    if (progress_stride < 1) progress_stride = 1;

#ifdef _OPENMP
    #pragma omp parallel reduction(+:loaded,merged,dropped)
#endif
    {
        int64_t local_loaded = 0;
        int64_t local_merged = 0;
        int64_t local_dropped = 0;

#ifdef _OPENMP
        #pragma omp for schedule(dynamic, 65536)
#endif
        for (int64_t i = 0; i < n_indexed; i++) {
            size_t ep = (size_t)offsets[i];
            /* Decode entry header. Field-by-field memcpy so no alignment
             * assumptions on the mmap region (on x86 unaligned is fine,
             * but being explicit is cheaper than debugging later). */
            BPInfoKey key;
            memcpy(&key.player,      base + ep + 0,  sizeof(int));
            memcpy(&key.street,      base + ep + 4,  sizeof(int));
            memcpy(&key.bucket,      base + ep + 8,  sizeof(int));
            memcpy(&key.board_hash,  base + ep + 12, sizeof(uint64_t));
            memcpy(&key.action_hash, base + ep + 20, sizeof(uint64_t));
            int na;
            memcpy(&na, base + ep + 28, sizeof(int));
            if (na > BP_MAX_ACTIONS) na = BP_MAX_ACTIONS;

            int64_t slot = info_table_find_or_create(t, key, na);
            if (slot < 0) {
                local_dropped++;
                continue;
            }

            BPInfoSet *is = &t->sets[slot];

            /* Merge regrets. Under parallel load, two threads may race on
             * this slot only if the source checkpoint has duplicate keys
             * (Hogwild race bug in training). We use atomic fetch_add to
             * be safe — for the 99.9999% fresh-slot case this compiles
             * to a LOCK ADD which is cheap on x86; for the rare duplicate
             * case it guarantees correctness.
             *
             * Detecting "is this a duplicate?" for reporting is racy but
             * we only use it for logging, not for correctness. We check
             * if any regret was already non-zero BEFORE our add (not
             * atomic with the add — that's fine since we only want an
             * approximate duplicate count). */
            const int *src_regrets = (const int*)(base + ep + 32);
            int is_dup = 0;
            for (int a = 0; a < na; a++) {
                int prev = __atomic_load_n(&is->regrets[a], __ATOMIC_RELAXED);
                if (prev != 0) is_dup = 1;
                int v;
                memcpy(&v, src_regrets + a, sizeof(int));
                __atomic_fetch_add(&is->regrets[a], v, __ATOMIC_RELAXED);
            }

            size_t has_ss_off = ep + 32 + (size_t)(4 * na);
            int has_ss;
            memcpy(&has_ss, base + has_ss_off, sizeof(int));
            if (has_ss && !reset_ss) {
                /* Lazy strategy_sum allocation. Same CAS pattern as
                 * ensure_strategy_sum but inlined here to avoid the extra
                 * function call in the hot path. */
                float *ss = __atomic_load_n((float**)&is->strategy_sum, __ATOMIC_ACQUIRE);
                if (!ss) {
                    float *buf = (float*)arena_alloc(na);
                    if (buf) {
                        float *expected = NULL;
                        if (__atomic_compare_exchange_n(&is->strategy_sum, &expected, buf,
                                                         0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                            ss = buf;
                        } else {
                            /* Another thread won — reload and use theirs.
                             * buf is wasted arena space. */
                            ss = __atomic_load_n((float**)&is->strategy_sum, __ATOMIC_ACQUIRE);
                        }
                    }
                }
                if (ss) {
                    const float *src_ss = (const float*)(base + has_ss_off + 4);
                    /* Atomic float add via CAS loop (no atomic_float_add in
                     * C11). Each slot is independent; contention is only
                     * on duplicate keys. */
                    for (int a = 0; a < na; a++) {
                        float add;
                        memcpy(&add, src_ss + a, sizeof(float));
                        int *slot_i = (int*)&ss[a];
                        union { float f; int i; } old_u, new_u;
                        do {
                            old_u.i = __atomic_load_n(slot_i, __ATOMIC_RELAXED);
                            new_u.f = old_u.f + add;
                        } while (!__atomic_compare_exchange_n(
                            slot_i, &old_u.i, new_u.i,
                            0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
                    }
                }
            }

            if (is_dup) local_merged++;
            local_loaded++;

            if ((i % progress_stride) == 0 && i > 0) {
#ifdef _OPENMP
                if (omp_get_thread_num() == 0) {
                    double dt = omp_get_wtime() - t0;
                    printf("[BP]   Pass 2 progress: %lld / %lld (%.1fs)\n",
                           (long long)i, (long long)n_indexed, dt);
                    fflush(stdout);
                }
#endif
            }
        }

        loaded += local_loaded;
        merged += local_merged;
        dropped += local_dropped;
    }

#ifdef _OPENMP
    double pass2_sec = omp_get_wtime() - t0;
    printf("[BP] Pass 2 done: %lld loaded, %lld merged, %lld dropped in %.1fs\n",
           (long long)loaded, (long long)merged, (long long)dropped, pass2_sec);
#else
    printf("[BP] Pass 2 done: %lld loaded, %lld merged, %lld dropped\n",
           (long long)loaded, (long long)merged, (long long)dropped);
#endif
    fflush(stdout);

    free(offsets);
    bp_file_map_close(&mfile);

    s->iterations_run = saved_iters;

    t->insertion_failures = 0;
    t->max_probe_observed = 0;

    if (merged > 0) {
        printf("[BP] WARNING: merged %lld duplicate entries (Hogwild race bug). "
               "Split regrets have been recovered.\n", (long long)merged);
    }
    printf("[BP] Loaded %lld/%lld info sets (%lld merged) [parallel], table %lld/%lld\n",
           (long long)loaded, (long long)saved_entries, (long long)merged,
           (long long)t->num_entries, (long long)t->table_size);
    return loaded;
}

/* Phase R Block B.2: parse the optional BPR5 trailer at `post_entries_off`.
 *
 * Returns 0 on success (trailer parsed, OR genuine EOF == legacy file),
 * -1 on hard error (corrupt magic, sanity-bound violation, short read,
 * find_or_create saturation, capacity-too-small for an existing table).
 *
 * Side effect on success-with-trailer: lazy-allocates s->cont_ev_table
 * if NULL (sized to next_prime(max(2*n_slots, 1024))) and populates it
 * with the trailer's slots. Does NOT flip s->depth_limit_cont_strats —
 * caller controls trainer-flag separately. */
static int bp_load_bpr5_trailer(BPSolver *s, const char *path,
                                 int64_t post_entries_off) {
    if (post_entries_off < 0) return 0;  /* loader didn't capture — skip */

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[BP] BPR5 trailer: fopen(%s) failed\n", path);
        return -1;
    }

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long file_size_l = ftell(f);
    if (file_size_l < 0) { fclose(f); return -1; }
    int64_t file_size = (int64_t)file_size_l;

    if (post_entries_off > file_size) {
        fprintf(stderr,
                "[BP] BPR5 trailer: post_entries_off=%lld > file_size=%lld\n",
                (long long)post_entries_off, (long long)file_size);
        fclose(f);
        return -1;
    }

    int64_t remaining = file_size - post_entries_off;
    if (remaining == 0) {
        /* Genuine EOF — legacy BPR4 file, no trailer. Cleanly leave the
         * cont_ev_table alone (NULL or pre-existing untouched). */
        fclose(f);
        return 0;
    }

    /* Sanity-bound the trailer header: 4 (magic) + 8 (n_slots) = 12 B. */
    if (remaining < 12) {
        fprintf(stderr,
                "[BP] BPR5 trailer corrupt: %lld bytes after BPR4 entries "
                "(need 0 for legacy or >=12 for trailer header)\n",
                (long long)remaining);
        fclose(f);
        return -1;
    }

    if (fseek(f, (long)post_entries_off, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }

    /* Step 15 refactor: delegate magic+n_slots read to the shared
     * helper (returns -1 on bad magic or short read; we still own the
     * BP_SKIP_BPR5_LOAD env-gate and sanity-bound checks because they're
     * specific to the inline-in-bin context, not the standalone CEVS
     * cache path). */
    int64_t n_slots;
    if (bp_cont_ev_read_trailer_header(f, &n_slots) != 0) {
        fclose(f); return -1;
    }

    /* Phase R Step 11: BP_SKIP_BPR5_LOAD=1 short-circuits the slot
     * load. Used by the zero-drift warm-start launcher when the on-disk
     * trailer was written with bad bucket keys (Step 10 bug); we
     * regenerate the trailer from scratch in this run by leaving the
     * cont_ev_table empty at startup so HEADS-branch dispatches all
     * insert-as-fresh. Skip is gated AFTER the magic + n_slots checks
     * so a corrupt trailer still fails loud. */
    {
        const char *skip = getenv("BP_SKIP_BPR5_LOAD");
        if (skip && skip[0] == '1') {
            fprintf(stderr,
                    "[BP] BPR5 trailer: BP_SKIP_BPR5_LOAD=1, skipping load "
                    "of %lld saved slots (cont_ev_table starts empty)\n",
                    (long long)n_slots);
            fclose(f);
            return 0;
        }
    }

    /* Per-slot wire size: 4(player) + 4(bucket) + 8(action_hash) +
     * 16(ev_sum[4]) + 32(ev_count[4]) = 64 bytes. Match the writer. */
    const int64_t SLOT_WIRE_BYTES = 64;
    int64_t bytes_after_header = remaining - 12;
    if (n_slots > bytes_after_header / SLOT_WIRE_BYTES) {
        fprintf(stderr,
                "[BP] BPR5 trailer: n_slots=%lld out of range "
                "(file has %lld bytes for slots, max %lld at %lld B/slot)\n",
                (long long)n_slots, (long long)bytes_after_header,
                (long long)(bytes_after_header / SLOT_WIRE_BYTES),
                (long long)SLOT_WIRE_BYTES);
        fclose(f); return -1;
    }

    if (n_slots == 0) {
        /* Legal-but-unusual: writer skips trailer when count=0, so this
         * shouldn't happen, but tolerate gracefully. */
        printf("[BP] BPR5 trailer: 0 slots (no-op)\n");
        fclose(f);
        return 0;
    }

    /* Slot read + table populate are shared with the standalone CEVS
     * path via the helper (Step 15 — no format drift between inline
     * and standalone). */
    int rc = bp_cont_ev_read_trailer_slots(f, &s->cont_ev_table, n_slots);
    fclose(f);
    if (rc != 0) return -1;

    printf("[BP] BPR5 trailer: loaded %lld boundary slots\n",
           (long long)n_slots);
    return 0;
}

int64_t bp_load_regrets(BPSolver *s, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Header */
    char magic[4];
    fread(magic, 1, 4, f);
    int is_v4 = (memcmp(magic, "BPR4", 4) == 0);
    int is_v3 = (memcmp(magic, "BPR3", 4) == 0);
    int is_v2 = (memcmp(magic, "BPR2", 4) == 0);
    if (!is_v4 && !is_v3 && !is_v2) {
        printf("[BP] ERROR: expected BPR2/BPR3/BPR4 format, got %.4s\n", magic);
        fclose(f);
        return -1;
    }

    int64_t saved_entries;
    int64_t saved_iters;
    if (is_v4) {
        int64_t saved_table_size64;
        fread(&saved_table_size64, sizeof(int64_t), 1, f);
        fread(&saved_entries, sizeof(int64_t), 1, f);
        fread(&saved_iters, sizeof(int64_t), 1, f);
    } else {
        int saved_table_size32, saved_entries32;
        fread(&saved_table_size32, sizeof(int), 1, f);
        fread(&saved_entries32, sizeof(int), 1, f);
        saved_entries = (int64_t)saved_entries32;
        if (is_v3) {
            fread(&saved_iters, sizeof(int64_t), 1, f);
        } else {
            int iters32; fread(&iters32, sizeof(int), 1, f);
            saved_iters = (int64_t)iters32;
        }
    }

    printf("[BP] Loading checkpoint: %lld info sets, %lld iterations\n",
           (long long)saved_entries, (long long)saved_iters);

    /* Dispatch:
     *   - BPR4 by default → parallel mmap loader
     *   - BPR4 with BP_LEGACY_LOADER=1 → serial loader (for correctness diff)
     *   - BPR2/BPR3 → serial loader (always — these formats are legacy) */
    const char *legacy = getenv("BP_LEGACY_LOADER");
    int force_serial = (legacy && legacy[0] == '1');

    int64_t loaded_count;
    int64_t post_entries_off = -1;

    if (is_v4 && !force_serial) {
        fclose(f);  /* parallel loader opens its own mmap */
        loaded_count = load_regrets_mmap_parallel(s, path,
                                                   saved_entries, saved_iters,
                                                   &post_entries_off);
        if (loaded_count == -2) {
            /* mmap failed — fall back to serial. Re-open and seek past
             * header (post-magic + 24 B). */
            f = fopen(path, "rb");
            if (!f) return -1;
            fseek(f, 28, SEEK_SET);
            loaded_count = load_regrets_serial(s, f, is_v4, is_v3, is_v2,
                                                saved_entries, saved_iters,
                                                &post_entries_off);
            fclose(f);
        }
    } else {
        loaded_count = load_regrets_serial(s, f, is_v4, is_v3, is_v2,
                                            saved_entries, saved_iters,
                                            &post_entries_off);
        fclose(f);
    }

    /* On entries-loader failure, propagate. The trailer is moot. */
    if (loaded_count < 0) return loaded_count;

    /* Phase R Block B.2: BPR5 trailer is a BPR4-only extension. For
     * legacy BPR2/BPR3, the file format is closed and there is no
     * trailer concept. */
    if (is_v4 && post_entries_off >= 0) {
        if (bp_load_bpr5_trailer(s, path, post_entries_off) != 0) {
            return -1;
        }
    }

    return loaded_count;
}

int bp_save_texture_cache(const BPSolver *s, const char *path) {
    if (!s->texture_bucket_cache || s->num_cached_textures == 0) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite("TXC1", 1, 4, f);
    fwrite(&s->num_cached_textures, sizeof(int), 1, f);
    fwrite(&s->postflop_num_buckets, sizeof(int), 1, f);
    fwrite(s->texture_hash_keys, sizeof(uint64_t), s->num_cached_textures, f);
    fwrite(s->texture_bucket_cache, sizeof(int),
           (size_t)s->num_cached_textures * BP_MAX_HANDS, f);
    fclose(f);
    printf("[BP] Saved texture cache: %d textures to %s\n",
           s->num_cached_textures, path);
    return 0;
}

int bp_load_texture_cache(BPSolver *s, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "TXC1", 4) != 0) { fclose(f); return -1; }
    int num_tex, num_buckets;
    fread(&num_tex, sizeof(int), 1, f);
    fread(&num_buckets, sizeof(int), 1, f);
    if (num_tex > BP_MAX_TEXTURES || num_buckets != s->postflop_num_buckets) {
        fclose(f); return -1;
    }
    if (!s->texture_bucket_cache)
        s->texture_bucket_cache = (int*)calloc(BP_MAX_TEXTURES * (size_t)BP_MAX_HANDS, sizeof(int));
    fread(s->texture_hash_keys, sizeof(uint64_t), num_tex, f);
    fread(s->texture_bucket_cache, sizeof(int), (size_t)num_tex * BP_MAX_HANDS, f);
    s->num_cached_textures = num_tex;
    fclose(f);
    /* Bug 7 fix: build the texture lookup hashmap */
    texture_index_rebuild(s);
    printf("[BP] Loaded texture cache: %d textures from %s\n", num_tex, path);
    return num_tex;
}
