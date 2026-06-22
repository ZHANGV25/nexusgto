/**
 * mccfr_cont_ev_table.c — Implementation of the cont-bias boundary EV
 * slot table for Phase R Block A.2.
 *
 * See mccfr_cont_ev_table.h for the contract. This file mirrors the
 * concurrency primitives of mccfr_blueprint.c:
 *   - splitmix64 hash mixer (compatible with hash_combine in trainer)
 *   - 4096-probe linear probing
 *   - 0/1/2 state machine on `occupied` with CAS publish + spin-wait
 *   - de-dup scan on insertion to guard against multi-thread races
 *
 * Self-contained — does not depend on the trainer's heavyweight
 * dependencies (card_abstraction, hand_eval, OpenMP runtime).
 */

#include "mccfr_cont_ev_table.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>

/* ── Probe / hashing primitives ──────────────────────────────────────
 *
 * Local copies of trainer's splitmix64 + hash_combine. Keeping them
 * static here (rather than linking against mccfr_blueprint.c) so that
 * the unit test can build with just this TU and not pull in the
 * trainer's full dep chain. The math is byte-identical to
 * mccfr_blueprint.c::bp_mix64 / hash_combine; if you change one,
 * change the other.
 */
#define BP_CONT_EV_PROBE_LIMIT 4096

static inline uint64_t cev_mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline uint64_t cev_hash_combine(uint64_t a, uint64_t b) {
    return cev_mix64(a ^ cev_mix64(b));
}

static inline uint64_t cev_compute_slot_hash(uint64_t action_hash,
                                              int player, int bucket) {
    uint64_t h = cev_hash_combine(action_hash, (uint64_t)player);
    h = cev_hash_combine(h, (uint64_t)bucket);
    return h;
}

/* Round n up to the next prime. Same trick as info_table_init
 * (mccfr_blueprint.c:565). Fast for the size range we care about
 * (1<<10 .. 1<<27); trial-division is fine. */
static int cev_is_prime_i64(int64_t n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if ((n & 1) == 0) return 0;
    for (int64_t d = 3; d * d <= n; d += 2)
        if (n % d == 0) return 0;
    return 1;
}

static int64_t cev_next_prime_i64(int64_t n) {
    if (n <= 2) return 2;
    if ((n & 1) == 0) n++;
    while (!cev_is_prime_i64(n)) n += 2;
    return n;
}

/* Spin-wait until a state=2 slot finishes initializing (becomes 1).
 * Mirrors mccfr_blueprint.c::spin_until_ready. */
static inline void cev_spin_until_ready(const int *flag) {
    int spins = 0;
    while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) == 2) {
        #ifdef __x86_64__
        __builtin_ia32_pause();
        #elif defined(__aarch64__)
        __asm__ volatile("yield");
        #endif
        if (++spins > 10000000) {
            spins = 0;
            sched_yield();
        }
    }
}

static inline int cev_key_eq(const BPContEVEntry *e,
                              uint64_t action_hash, int player, int bucket) {
    return e->action_hash == action_hash &&
           e->player == player &&
           e->bucket == bucket;
}

/* ── Public API ─────────────────────────────────────────────────── */

int bp_cont_ev_init(BPContEVTable *t, int64_t table_size) {
    if (t == NULL) return -1;
    memset(t, 0, sizeof(*t));
    if (table_size <= 0) {
        /* Sentinel: feature OFF. find_or_create returns NULL. */
        return 0;
    }
    int64_t prime_size = cev_next_prime_i64(table_size);
    BPContEVEntry *entries = (BPContEVEntry *)calloc(
        (size_t)prime_size, sizeof(BPContEVEntry));
    if (entries == NULL) return -1;
    t->entries = entries;
    t->table_size = prime_size;
    t->num_entries = 0;
    t->insertion_failures = 0;
    t->max_probe_observed = 0;
    return 0;
}

void bp_cont_ev_free(BPContEVTable *t) {
    if (t == NULL) return;
    if (t->entries != NULL) {
        free(t->entries);
    }
    memset(t, 0, sizeof(*t));
}

int64_t bp_cont_ev_count(const BPContEVTable *t) {
    if (t == NULL) return 0;
    return __atomic_load_n(&t->num_entries, __ATOMIC_RELAXED);
}

BPContEVEntry *bp_cont_ev_find(const BPContEVTable *t,
                                uint64_t action_hash,
                                int player, int bucket) {
    if (t == NULL || t->entries == NULL || t->table_size <= 0) return NULL;
    uint64_t h = cev_compute_slot_hash(action_hash, player, bucket);
    int64_t slot = (int64_t)(h % (uint64_t)t->table_size);
    for (int probe = 0; probe < BP_CONT_EV_PROBE_LIMIT; probe++) {
        int64_t idx = (slot + probe) % t->table_size;
        int state = __atomic_load_n(&t->entries[idx].occupied, __ATOMIC_ACQUIRE);
        if (state == 0) return NULL;
        if (state == 2) continue;
        if (cev_key_eq(&t->entries[idx], action_hash, player, bucket))
            return &t->entries[idx];
    }
    return NULL;
}

BPContEVEntry *bp_cont_ev_find_or_create(BPContEVTable *t,
                                          uint64_t action_hash,
                                          int player, int bucket) {
    if (t == NULL || t->entries == NULL || t->table_size <= 0) return NULL;

    uint64_t h = cev_compute_slot_hash(action_hash, player, bucket);
    int64_t slot = (int64_t)(h % (uint64_t)t->table_size);

    for (int probe = 0; probe < BP_CONT_EV_PROBE_LIMIT; probe++) {
        int64_t idx = (slot + probe) % t->table_size;
        int state = __atomic_load_n(&t->entries[idx].occupied, __ATOMIC_ACQUIRE);

        if (state == 1) {
            if (cev_key_eq(&t->entries[idx], action_hash, player, bucket)) {
                if (probe > 0) {
                    int64_t cur = __atomic_load_n(&t->max_probe_observed,
                                                   __ATOMIC_RELAXED);
                    while ((int64_t)probe > cur &&
                           !__atomic_compare_exchange_n(&t->max_probe_observed,
                                                         &cur, (int64_t)probe,
                                                         1, __ATOMIC_RELAXED,
                                                         __ATOMIC_RELAXED)) { }
                }
                return &t->entries[idx];
            }
            continue;
        }

        if (state == 0) {
            int expected = 0;
            if (__atomic_compare_exchange_n(&t->entries[idx].occupied,
                                             &expected, 2,
                                             0, __ATOMIC_ACQ_REL,
                                             __ATOMIC_ACQUIRE)) {
                /* We won this slot (state went 0 → 2). Before writing
                 * our key fields and publishing, scan earlier slots in
                 * the probe chain to see if another thread already
                 * published the same key. If yes — we'd be a duplicate;
                 * roll back our CAS (state 2 → 0) and return the
                 * earlier slot, ensuring all concurrent inserters
                 * converge on the same canonical slot.
                 *
                 * Pre-publish dedup matters: under high contention, two
                 * threads can each CAS-win different slots in the same
                 * probe chain and both publish their duplicates before
                 * the post-publish scan runs. A late-comer entering
                 * mid-init at the duplicate would key_eq match it and
                 * return the duplicate, never reaching the canonical.
                 * Pre-publish-rollback eliminates the window. */
                if (probe > 0) {
                    for (int p2 = 0; p2 < probe; p2++) {
                        int64_t idx2 = (slot + p2) % t->table_size;
                        int st2 = __atomic_load_n(&t->entries[idx2].occupied,
                                                   __ATOMIC_ACQUIRE);
                        if (st2 == 2) {
                            cev_spin_until_ready(&t->entries[idx2].occupied);
                            st2 = __atomic_load_n(&t->entries[idx2].occupied,
                                                   __ATOMIC_ACQUIRE);
                        }
                        if (st2 == 1 && cev_key_eq(&t->entries[idx2],
                                                    action_hash, player, bucket)) {
                            /* Earlier match exists. Roll back our CAS so
                             * this slot is reusable, and return the
                             * canonical slot. */
                            __atomic_store_n(&t->entries[idx].occupied, 0,
                                             __ATOMIC_RELEASE);
                            return &t->entries[idx2];
                        }
                    }
                }

                /* No earlier match — we ARE the canonical. Initialize
                 * key fields and publish. */
                t->entries[idx].action_hash = action_hash;
                t->entries[idx].player = player;
                t->entries[idx].bucket = bucket;
                /* ev_sum / ev_count are zeroed by calloc and stay zero
                 * until the trainer accumulates the first sample. */
                __atomic_store_n(&t->entries[idx].occupied, 1,
                                 __ATOMIC_RELEASE);

                __atomic_fetch_add(&t->num_entries, 1, __ATOMIC_RELAXED);
                if (probe > 0) {
                    int64_t cur = __atomic_load_n(&t->max_probe_observed,
                                                   __ATOMIC_RELAXED);
                    while ((int64_t)probe > cur &&
                           !__atomic_compare_exchange_n(&t->max_probe_observed,
                                                         &cur, (int64_t)probe,
                                                         1, __ATOMIC_RELAXED,
                                                         __ATOMIC_RELAXED)) { }
                }
                return &t->entries[idx];
            }
            /* CAS lost — re-read state and fall through. */
            state = __atomic_load_n(&t->entries[idx].occupied,
                                     __ATOMIC_ACQUIRE);
        }

        if (state == 2) {
            cev_spin_until_ready(&t->entries[idx].occupied);
            if (__atomic_load_n(&t->entries[idx].occupied,
                                 __ATOMIC_ACQUIRE) == 1) {
                if (cev_key_eq(&t->entries[idx], action_hash, player, bucket))
                    return &t->entries[idx];
            }
            continue;
        }
    }
    /* Probe cap exhausted — table too full or pathologically clustered. */
    __atomic_fetch_add(&t->insertion_failures, 1, __ATOMIC_RELAXED);
    {
        int64_t cur = __atomic_load_n(&t->max_probe_observed, __ATOMIC_RELAXED);
        int64_t want = (int64_t)BP_CONT_EV_PROBE_LIMIT;
        while (want > cur &&
               !__atomic_compare_exchange_n(&t->max_probe_observed, &cur, want,
                                             1, __ATOMIC_RELAXED,
                                             __ATOMIC_RELAXED)) { }
    }
    return NULL;
}

void bp_cont_ev_atomic_add_f32(float *dst, float val) {
    /* CAS loop: load current, compute new, CAS-or-retry. C11 stdatomic
     * doesn't expose float atomics; we type-pun via uint32_t. */
    _Static_assert(sizeof(float) == sizeof(uint32_t),
                   "bp_cont_ev_atomic_add_f32: float is not uint32_t-sized");
    uint32_t *as_u32 = (uint32_t *)dst;
    uint32_t old_u32 = __atomic_load_n(as_u32, __ATOMIC_RELAXED);
    for (;;) {
        float old_f;
        memcpy(&old_f, &old_u32, sizeof(float));
        float new_f = old_f + val;
        uint32_t new_u32;
        memcpy(&new_u32, &new_f, sizeof(float));
        if (__atomic_compare_exchange_n(as_u32, &old_u32, new_u32,
                                         0, __ATOMIC_RELEASE,
                                         __ATOMIC_RELAXED)) {
            return;
        }
        /* CAS failed — old_u32 has been updated to the current value.
         * Retry with the fresh load. */
    }
}

BPBoundaryEVStatus bp_query_boundary_ev(const BPContEVTable *t,
                                         uint64_t action_hash,
                                         int player, int bucket,
                                         int cont_bias,
                                         float *out_ev) {
    /* NaN sentinel for every non-OK path. Caller MUST check status. */
    if (out_ev) *out_ev = NAN;

    if (cont_bias < 0 || cont_bias >= BP_CONT_EV_NUM_BIASES) {
        return BP_BOUNDARY_EV_INVALID;
    }

    BPContEVEntry *e = bp_cont_ev_find(t, action_hash, player, bucket);
    if (e == NULL) {
        return BP_BOUNDARY_EV_MISS;
    }

    int64_t cnt = e->ev_count[cont_bias];
    if (cnt <= 0) {
        return BP_BOUNDARY_EV_EMPTY_BIAS;
    }

    if (out_ev) {
        *out_ev = e->ev_sum[cont_bias] / (float)cnt;
    }
    return BP_BOUNDARY_EV_OK;
}

BPBoundaryEVStatus bps_query_boundary_ev(const BPContEVTable *t,
                                          uint64_t action_hash,
                                          int player, int bucket,
                                          int cont_bias,
                                          float *out_ev) {
    /* Kernel/Python-facing alias. Identical semantics — both symbols
     * ship from this TU so kernel-side builds (and bps_reader.c
     * consumers) don't have to add a separate TU just to resolve the
     * bps_-prefixed name. */
    return bp_query_boundary_ev(t, action_hash, player, bucket,
                                 cont_bias, out_ev);
}

void bp_cont_ev_apply_discount(BPContEVTable *t, float discount) {
    if (t == NULL || t->entries == NULL) return;
    /* Sweeps the entire table once. Mirrors apply_discount in
     * mccfr_blueprint.c (line 2276) — the trainer wires this from the
     * same `if (do_discount)` block. We don't add an `omp for` pragma
     * here because this TU is intentionally OMP-free; the caller can
     * wrap the call in their own `#pragma omp for` if they want
     * parallel discount sweeps. */
    for (int64_t i = 0; i < t->table_size; i++) {
        if (__atomic_load_n(&t->entries[i].occupied, __ATOMIC_ACQUIRE) != 1)
            continue;
        for (int b = 0; b < BP_CONT_EV_NUM_BIASES; b++) {
            t->entries[i].ev_sum[b] *= discount;
        }
        /* ev_count is NOT discounted — it's a raw visit count. The
         * `ev_sum / ev_count` ratio at query time gives a Linear-CFR-
         * weighted EV per visit, which is what the kernel reads. */
    }
}

/* ── BPR5 trailer body helpers (Step 15) ─────────────────────────── */

int bp_cont_ev_write_trailer_body(FILE *f, const BPContEVTable *t) {
    if (!f || !t) return -1;
    int64_t n_slots = bp_cont_ev_count(t);

    if (fwrite("BPR5", 1, 4, f) != 4) return -1;
    if (fwrite(&n_slots, sizeof(int64_t), 1, f) != 1) return -1;

    int64_t cev_written = 0;
    for (int64_t i = 0; i < t->table_size; i++) {
        const BPContEVEntry *e = &t->entries[i];
        if (__atomic_load_n(&e->occupied, __ATOMIC_ACQUIRE) != 1) continue;

        if (fwrite(&e->player,      sizeof(int),      1, f) != 1) return -1;
        if (fwrite(&e->bucket,      sizeof(int),      1, f) != 1) return -1;
        if (fwrite(&e->action_hash, sizeof(uint64_t), 1, f) != 1) return -1;
        if (fwrite(e->ev_sum,       sizeof(float),
                   BP_CONT_EV_NUM_BIASES, f) != BP_CONT_EV_NUM_BIASES) return -1;
        if (fwrite(e->ev_count,     sizeof(int64_t),
                   BP_CONT_EV_NUM_BIASES, f) != BP_CONT_EV_NUM_BIASES) return -1;
        cev_written++;
    }

    /* Data-corruption sentinel: bp_cont_ev_count and an occupied-walk
     * MUST agree. Mirrors the inline check at mccfr_blueprint.c:4292. */
    if (cev_written != n_slots) {
        fprintf(stderr,
                "[cev] write_trailer_body count mismatch: header %lld, "
                "walked %lld\n",
                (long long)n_slots, (long long)cev_written);
        return -1;
    }
    return 0;
}

int bp_cont_ev_read_trailer_header(FILE *f, int64_t *out_n_slots) {
    if (!f || !out_n_slots) return -1;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4) {
        fprintf(stderr, "[cev] read_trailer_header: short read on magic\n");
        return -1;
    }
    if (memcmp(magic, "BPR5", 4) != 0) {
        fprintf(stderr,
                "[cev] read_trailer_header: unexpected magic '%c%c%c%c' "
                "(expected 'BPR5')\n",
                magic[0], magic[1], magic[2], magic[3]);
        return -1;
    }

    int64_t n_slots;
    if (fread(&n_slots, sizeof(int64_t), 1, f) != 1) {
        fprintf(stderr, "[cev] read_trailer_header: short read on n_slots\n");
        return -1;
    }
    if (n_slots < 0) {
        fprintf(stderr,
                "[cev] read_trailer_header: negative n_slots=%lld\n",
                (long long)n_slots);
        return -1;
    }
    *out_n_slots = n_slots;
    return 0;
}

int bp_cont_ev_read_trailer_slots(FILE *f, BPContEVTable *t, int64_t n_slots) {
    if (!f || !t) return -1;
    if (n_slots < 0) return -1;
    if (n_slots == 0) return 0;  /* legal no-op */

    /* Allocation policy (matches mccfr_blueprint.c lazy-alloc):
     *   - entries == NULL → lazy-alloc to next_prime(max(2*n_slots, 1024)).
     *   - already allocated → require headroom (n_slots <= table_size/2). */
    if (t->entries == NULL) {
        int64_t target = n_slots * 2;
        if (target < 1024) target = 1024;
        if (bp_cont_ev_init(t, target) != 0) {
            fprintf(stderr,
                    "[cev] read_trailer_slots: bp_cont_ev_init(%lld) failed\n",
                    (long long)target);
            return -1;
        }
    } else if (n_slots > t->table_size / 2) {
        fprintf(stderr,
                "[cev] read_trailer_slots: table too small (size=%lld) "
                "for n_slots=%lld. Free or init a fresh table.\n",
                (long long)t->table_size, (long long)n_slots);
        return -1;
    }

    for (int64_t i = 0; i < n_slots; i++) {
        int player, bucket;
        uint64_t action_hash;
        float ev_sum_buf[BP_CONT_EV_NUM_BIASES];
        int64_t ev_count_buf[BP_CONT_EV_NUM_BIASES];

        if (fread(&player,      sizeof(int),      1, f) != 1 ||
            fread(&bucket,      sizeof(int),      1, f) != 1 ||
            fread(&action_hash, sizeof(uint64_t), 1, f) != 1 ||
            fread(ev_sum_buf,   sizeof(float),    BP_CONT_EV_NUM_BIASES, f)
                != (size_t)BP_CONT_EV_NUM_BIASES ||
            fread(ev_count_buf, sizeof(int64_t),  BP_CONT_EV_NUM_BIASES, f)
                != (size_t)BP_CONT_EV_NUM_BIASES) {
            fprintf(stderr,
                    "[cev] read_trailer_slots: short read on slot %lld/%lld\n",
                    (long long)i, (long long)n_slots);
            return -1;
        }

        BPContEVEntry *e = bp_cont_ev_find_or_create(t, action_hash,
                                                      player, bucket);
        if (e == NULL) {
            fprintf(stderr,
                    "[cev] read_trailer_slots: find_or_create saturated at "
                    "slot %lld/%lld\n", (long long)i, (long long)n_slots);
            return -1;
        }
        memcpy(e->ev_sum,   ev_sum_buf,   sizeof(ev_sum_buf));
        memcpy(e->ev_count, ev_count_buf, sizeof(ev_count_buf));
    }
    return 0;
}

int bp_cont_ev_read_trailer_body(FILE *f, BPContEVTable *t) {
    int64_t n_slots;
    if (bp_cont_ev_read_trailer_header(f, &n_slots) != 0) return -1;
    return bp_cont_ev_read_trailer_slots(f, t, n_slots);
}

/* ── Standalone CEVS cache file (Step 15) ───────────────────────── */

/* Body XOR-checksum: read the file from the start of the BPR5 body to
 * EOF in chunks, XOR-accumulating 64-bit words. Used to detect corruption
 * during S3 round-trip or partial writes.
 *
 * Note: this is a tamper-detection check, not a cryptographic hash. Any
 * change to the body changes the checksum with overwhelming probability.
 * Trailing bytes that don't fit in an 8-byte word are zero-padded into
 * the final word — same on save and load, so they cancel correctly. */
static int cev_body_xor_checksum(FILE *f, long start_offset,
                                  long end_offset, uint64_t *out) {
    if (fseek(f, start_offset, SEEK_SET) != 0) return -1;
    uint64_t acc = 0;
    long remaining = end_offset - start_offset;
    while (remaining > 0) {
        uint8_t buf[4096];
        size_t want = remaining < (long)sizeof(buf) ? (size_t)remaining
                                                    : sizeof(buf);
        size_t got = fread(buf, 1, want, f);
        if (got == 0) return -1;
        /* Pad the final partial-word chunk with zeros. */
        size_t padded = (got + 7) & ~(size_t)7;
        if (padded > sizeof(buf)) padded = sizeof(buf);
        if (padded > got) memset(buf + got, 0, padded - got);
        for (size_t i = 0; i < padded; i += 8) {
            uint64_t w;
            memcpy(&w, buf + i, sizeof(w));
            acc ^= w;
        }
        remaining -= (long)got;
    }
    *out = acc;
    return 0;
}

int bp_cont_ev_save_standalone(const BPContEVTable *t, const char *path) {
    if (!t || !path) return -1;

    /* Stage to <path>.tmp, fsync, rename. Atomic per POSIX rename(2). */
    size_t tmp_len = strlen(path) + 5;
    char *tmp_path = (char*)malloc(tmp_len);
    if (!tmp_path) return -1;
    snprintf(tmp_path, tmp_len, "%s.tmp", path);

    /* "wb+" — write + read (so we can fread the body back for the
     * checksum without reopening). Truncates on open. */
    FILE *f = fopen(tmp_path, "wb+");
    if (!f) { free(tmp_path); return -1; }

    /* Header: "CEVS" + version + reserved + body_bytes (filled later) +
     * checksum (filled later). 28 bytes total. */
    if (fwrite(BP_CONT_EV_STANDALONE_MAGIC, 1, 4, f) != 4) goto bad;
    uint32_t version = BP_CONT_EV_STANDALONE_VERSION, reserved = 0;
    if (fwrite(&version,  sizeof(uint32_t), 1, f) != 1) goto bad;
    if (fwrite(&reserved, sizeof(uint32_t), 1, f) != 1) goto bad;

    /* Reserve 8 bytes for body_bytes and 8 for body_xor_checksum; we'll
     * seek back and patch after writing the body. */
    long header_size_pos = ftell(f);
    int64_t placeholder_size = 0;
    uint64_t placeholder_xor = 0;
    if (fwrite(&placeholder_size, sizeof(int64_t),  1, f) != 1) goto bad;
    if (fwrite(&placeholder_xor,  sizeof(uint64_t), 1, f) != 1) goto bad;

    long body_start_off = ftell(f);

    if (bp_cont_ev_write_trailer_body(f, t) != 0) goto bad;

    long body_end_off = ftell(f);
    int64_t body_bytes = (int64_t)(body_end_off - body_start_off);

    /* Compute checksum by re-reading the body we just wrote. fflush
     * first to ensure the kernel sees what we wrote before we fseek
     * and fread. */
    if (fflush(f) != 0) goto bad;
    uint64_t body_xor = 0;
    if (cev_body_xor_checksum(f, body_start_off, body_end_off, &body_xor) != 0)
        goto bad;

    /* Patch header in place. */
    if (fseek(f, header_size_pos, SEEK_SET) != 0) goto bad;
    if (fwrite(&body_bytes, sizeof(int64_t),  1, f) != 1) goto bad;
    if (fwrite(&body_xor,   sizeof(uint64_t), 1, f) != 1) goto bad;

    if (fflush(f) != 0) goto bad;
    int fd = fileno(f);
    if (fd >= 0) {
        if (fsync(fd) != 0) {
            /* Don't fail the save on fsync failure (some FSes return
             * EINVAL on regular files); rename will still atomicize. */
        }
    }
    if (fclose(f) != 0) {
        unlink(tmp_path);
        free(tmp_path);
        return -1;
    }

    if (rename(tmp_path, path) != 0) {
        fprintf(stderr,
                "[cev] save_standalone: rename(%s -> %s) failed: %s\n",
                tmp_path, path, strerror(errno));
        unlink(tmp_path);
        free(tmp_path);
        return -1;
    }
    fprintf(stderr,
            "[cev] save_standalone: %s (%lld body bytes, "
            "xor=0x%016llx, %lld slots)\n",
            path, (long long)body_bytes,
            (unsigned long long)body_xor,
            (long long)bp_cont_ev_count(t));
    free(tmp_path);
    return 0;

bad:
    fclose(f);
    unlink(tmp_path);
    free(tmp_path);
    return -1;
}

int bp_cont_ev_load_standalone(BPContEVTable *t, const char *path) {
    if (!t || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[cev] load_standalone: fopen(%s) failed: %s\n",
                path, strerror(errno));
        return -1;
    }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 ||
        memcmp(magic, BP_CONT_EV_STANDALONE_MAGIC, 4) != 0) {
        fprintf(stderr,
                "[cev] load_standalone: bad magic '%c%c%c%c' "
                "(expected '%s')\n",
                magic[0], magic[1], magic[2], magic[3],
                BP_CONT_EV_STANDALONE_MAGIC);
        fclose(f); return -1;
    }
    uint32_t version, reserved;
    if (fread(&version,  sizeof(uint32_t), 1, f) != 1 ||
        fread(&reserved, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "[cev] load_standalone: short read on header\n");
        fclose(f); return -1;
    }
    if (version != BP_CONT_EV_STANDALONE_VERSION) {
        fprintf(stderr,
                "[cev] load_standalone: unsupported version %u (expected %u)\n",
                (unsigned)version, BP_CONT_EV_STANDALONE_VERSION);
        fclose(f); return -1;
    }
    if (reserved != 0) {
        fprintf(stderr,
                "[cev] load_standalone: reserved field nonzero (0x%x)\n",
                (unsigned)reserved);
        fclose(f); return -1;
    }

    int64_t body_bytes;
    uint64_t body_xor_stored;
    if (fread(&body_bytes,      sizeof(int64_t),  1, f) != 1 ||
        fread(&body_xor_stored, sizeof(uint64_t), 1, f) != 1) {
        fprintf(stderr, "[cev] load_standalone: short read on header tail\n");
        fclose(f); return -1;
    }

    long body_start = ftell(f);
    if (body_start < 0) { fclose(f); return -1; }

    /* Sanity check: body extends exactly to EOF. */
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long file_end = ftell(f);
    if (file_end < 0) { fclose(f); return -1; }
    int64_t actual_body_bytes = (int64_t)(file_end - body_start);
    if (actual_body_bytes != body_bytes) {
        fprintf(stderr,
                "[cev] load_standalone: body size mismatch — header says "
                "%lld, file has %lld\n",
                (long long)body_bytes, (long long)actual_body_bytes);
        fclose(f); return -1;
    }

    /* Verify checksum BEFORE reading the body for real. */
    uint64_t body_xor_actual;
    if (cev_body_xor_checksum(f, body_start, file_end, &body_xor_actual) != 0) {
        fprintf(stderr, "[cev] load_standalone: checksum read failed\n");
        fclose(f); return -1;
    }
    if (body_xor_actual != body_xor_stored) {
        fprintf(stderr,
                "[cev] load_standalone: checksum mismatch — stored "
                "0x%016llx, computed 0x%016llx (data corruption or "
                "incomplete S3 download)\n",
                (unsigned long long)body_xor_stored,
                (unsigned long long)body_xor_actual);
        fclose(f); return -1;
    }

    /* Body is verified intact. Rewind to body start and load via the
     * shared trailer reader. */
    if (fseek(f, body_start, SEEK_SET) != 0) { fclose(f); return -1; }
    int rc = bp_cont_ev_read_trailer_body(f, t);
    fclose(f);
    if (rc != 0) {
        fprintf(stderr,
                "[cev] load_standalone: read_trailer_body failed (rc=%d)\n",
                rc);
        return -1;
    }
    fprintf(stderr,
            "[cev] load_standalone: %s loaded (%lld body bytes, "
            "xor=0x%016llx, %lld slots)\n",
            path, (long long)body_bytes,
            (unsigned long long)body_xor_stored,
            (long long)bp_cont_ev_count(t));
    return 0;
}
