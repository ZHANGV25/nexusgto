"""C-backed reader for unified .bps blueprint files.

Drop-in replacement for BlueprintV2 when loading the full unified blueprint.
Uses ~4 GB RAM instead of ~240 GB by delegating storage and lookups to a
compact C hash table (src/bps_reader.c).

Usage:
    from bps_reader_py import BPSReaderC

    bp = BPSReaderC("blueprint_data_v3/unified_blueprint_v3_1.5B.bps")

    strats = bp.get_all_bucket_strategies(
        board=[], action_history=[3, 1], player=0, street=1
    )
    # strats: np.ndarray[200, num_actions] float32, or None

    evs = bp.get_all_bucket_action_evs(
        board=[], action_history=[3, 1], player=0, street=1
    )
    # evs: np.ndarray[200, num_actions] float32, or None
"""

import ctypes
import json
import lzma
import mmap
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

# Maximum actions per decision node (must match BPS_MAX_ACTIONS in bps_reader.h)
_MAX_ACTIONS = 16

# Chunk size for streaming LZMA decompression. Large enough to amortize
# Python's per-call overhead in `LZMADecompressor.decompress(chunk)` —
# small chunks are 5-10× slower than `lzma.decompress(whole_buffer)` on
# multi-GB inputs because each call has fixed Python-side cost. 256 MB
# compressed → ~400 MB decompressed worst case. With a 32 GB host that
# leaves plenty of headroom (the compressed chunk plus the decompressor's
# own state plus a single decompressed-output bytes object); the rest
# of the decompressed blob is on disk in the tempfile, not in RAM.
_LZMA_CHUNK_BYTES = 256 * 1024 * 1024


def _stream_lzma_to_tempfile(src_file, compressed_size: int, label: str) -> str:
    """Stream-decompress `compressed_size` bytes from `src_file` (already
    positioned at the start of the LZMA stream) into a fresh tempfile.

    Returns the tempfile path. Caller is responsible for unlinking it.

    Peak Python RAM during this call: O(_LZMA_CHUNK_BYTES) — about 32 MB —
    instead of `compressed_size + decompressed_size` (40+ GB on the 3.4B
    blueprint), which is what the old `lzma.decompress(buf)` round-trip
    cost.
    """
    # IMPORTANT: the tempfile MUST be on a disk-backed filesystem. On
    # AL2023 (and many other modern Linux distros) /tmp is tmpfs (RAM),
    # which defeats the streaming refactor — the decompressed blob ends
    # up in RAM via tmpfs and OOMs around 28 GB on a 32 GB host.
    # Honor BPS_TMPDIR explicitly when set; otherwise fall back to
    # tempfile's resolution (TMPDIR / /tmp), with a loud warning if
    # the resolved path looks like tmpfs.
    tmpdir = os.environ.get("BPS_TMPDIR") or tempfile.gettempdir()
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir, exist_ok=True)
    # Loud warning if tmpdir is tmpfs (RAM-backed). Best-effort check —
    # `findmnt` is the canonical answer but isn't always available; fall
    # back to checking /proc/mounts. On Windows there's no equivalent
    # concern (tempfile resolves to %TEMP% which is on the user profile
    # disk), so skip the warning there.
    if sys.platform != "win32":
        try:
            with open("/proc/mounts", "r") as _mf:
                for _line in _mf:
                    parts = _line.split()
                    if len(parts) >= 3 and parts[2] == "tmpfs":
                        # Check whether tmpdir is mounted at this point.
                        mount_point = parts[1]
                        if (os.path.realpath(tmpdir).startswith(mount_point + "/")
                            or os.path.realpath(tmpdir) == mount_point):
                            print(
                                f"[BPSReaderC] WARNING: tempfile dir {tmpdir} is tmpfs "
                                f"(RAM-backed, mounted from {mount_point}). The "
                                f"streaming decompress will fill RAM, defeating its "
                                f"purpose. Set BPS_TMPDIR or TMPDIR to a disk-backed "
                                f"path (e.g. /var/cache/bps-tmp).",
                                flush=True,
                            )
                            break
        except Exception:
            pass
    tf = tempfile.NamedTemporaryFile(prefix="bps_decompressed_", suffix=".bin",
                                     delete=False, dir=tmpdir)
    tf_path = tf.name
    written = 0
    decompressor = lzma.LZMADecompressor()
    try:
        bytes_left = compressed_size
        last_pct = -1
        while bytes_left > 0:
            n = min(_LZMA_CHUNK_BYTES, bytes_left)
            chunk = src_file.read(n)
            if not chunk:
                break
            bytes_left -= len(chunk)
            out = decompressor.decompress(chunk)
            if out:
                tf.write(out)
                written += len(out)
            pct = int(100 * (compressed_size - bytes_left) / max(1, compressed_size))
            if pct >= last_pct + 10:
                print(
                    f"[BPSReaderC]   {label} decompress: {pct}% "
                    f"({(compressed_size - bytes_left) / (1024**2):.0f} MB in -> "
                    f"{written / (1024**2):.0f} MB out)",
                    flush=True,
                )
                last_pct = pct
        # Some lzma streams have a trailing flush; LZMADecompressor doesn't
        # expose a public flush() but `decompress(b"")` after EOF returns
        # any final buffered output.
        try:
            tail = decompressor.decompress(b"")
            if tail:
                tf.write(tail)
                written += len(tail)
        except Exception:
            pass
        tf.flush()
        try:
            os.fsync(tf.fileno())
        except OSError:
            # fsync can fail on some Windows filesystems; not fatal.
            pass
    finally:
        try:
            tf.close()
        except Exception:
            pass

    print(
        f"[BPSReaderC]   {label}: streamed {written / (1024**2):.0f} MB to "
        f"tempfile {tf_path}",
        flush=True,
    )
    return tf_path


def _find_lib() -> ctypes.CDLL:
    """Locate and load the bps_reader shared library.

    Prefers bps_ev_walk (which includes both reader + EV walk support).
    Falls back to bps_reader (read-only, no EV computation).
    """
    base = Path(__file__).resolve().parent.parent / "build"
    ext = ".dll" if sys.platform == "win32" else ".so"

    # Prefer the ev_walk build (superset of bps_reader)
    candidates = [
        base / f"bps_ev_walk{ext}",
        base / f"bps_reader{ext}",
    ]

    # Also check POKER_SOLVER_PATH env var
    env_path = os.environ.get("POKER_SOLVER_PATH")
    if env_path:
        build_dir = Path(env_path) / "build"
        candidates.append(build_dir / f"bps_ev_walk{ext}")
        candidates.append(build_dir / f"bps_reader{ext}")

    for path in candidates:
        if path.exists():
            return ctypes.CDLL(str(path))

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"bps_reader shared library not found. Searched: {searched}. "
        f"Run 'make bps_ev_walk' or 'make bps_reader' in the poker-solver directory."
    )


def _setup_ctypes(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for type safety."""
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int64_p = ctypes.POINTER(ctypes.c_int64)

    # `data` is declared c_void_p so we can hand C either a Python bytes
    # buffer (auto-converts) or a raw pointer into an mmap view (we go
    # through ctypes.c_char arrays to avoid a copy of the decompressed
    # blob — see _load_full).
    lib.bps_load_strategies.argtypes = [
        ctypes.c_void_p,         # BPSReader*
        ctypes.c_void_p,         # data (bytes or mmap-pointer)
        ctypes.c_size_t,         # len
        ctypes.c_int,            # use_legacy_mixer
        ctypes.c_int,            # postflop_buckets
        ctypes.c_int,            # preflop_buckets
    ]
    lib.bps_load_strategies.restype = ctypes.c_int

    lib.bps_load_evs.argtypes = [
        ctypes.c_void_p,         # BPSReader*
        ctypes.c_void_p,         # data (bytes or mmap-pointer)
        ctypes.c_size_t,         # len
    ]
    lib.bps_load_evs.restype = ctypes.c_int

    lib.bps_query_all_strategies.argtypes = [
        ctypes.c_void_p,         # BPSReader*
        c_int_p,                 # action_history
        ctypes.c_int,            # action_len
        ctypes.c_int,            # player
        ctypes.c_int,            # street
        c_float_p,               # out
        ctypes.c_int,            # out_size
        c_int_p,                 # num_buckets_out
    ]
    lib.bps_query_all_strategies.restype = ctypes.c_int

    lib.bps_query_all_evs.argtypes = [
        ctypes.c_void_p,         # BPSReader*
        c_int_p,                 # action_history
        ctypes.c_int,            # action_len
        ctypes.c_int,            # player
        ctypes.c_int,            # street
        c_float_p,               # out
        ctypes.c_int,            # out_size
        c_int_p,                 # num_buckets_out
    ]
    lib.bps_query_all_evs.restype = ctypes.c_int

    lib.bps_query_strategy.argtypes = [
        ctypes.c_void_p,         # BPSReader*
        c_int_p,                 # action_history
        ctypes.c_int,            # action_len
        ctypes.c_int,            # player
        ctypes.c_int,            # street
        ctypes.c_int,            # bucket
        c_float_p,               # strategy_out
    ]
    lib.bps_query_strategy.restype = ctypes.c_int

    lib.bps_has_action_evs.argtypes = [ctypes.c_void_p]
    lib.bps_has_action_evs.restype = ctypes.c_int

    lib.bps_load_texture_cache.argtypes = [
        ctypes.c_void_p,         # BPSReader*
        ctypes.c_char_p,         # path
        ctypes.c_int,            # cache_uses_legacy_mixer
    ]
    lib.bps_load_texture_cache.restype = ctypes.c_int

    lib.bps_hand_to_bucket.argtypes = [
        ctypes.c_void_p,         # BPSReader*
        c_int_p,                 # board (3 ints)
        ctypes.c_int,            # c0
        ctypes.c_int,            # c1
    ]
    lib.bps_hand_to_bucket.restype = ctypes.c_int

    # Turn/river bucketizers (added 2026-04-15 alongside the rollout port).
    # Not present in older bps_reader.dll builds — guard with hasattr.
    if hasattr(lib, "bps_load_turn_centroids"):
        lib.bps_load_turn_centroids.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p,
        ]
        lib.bps_load_turn_centroids.restype = ctypes.c_int
    if hasattr(lib, "bps_hand_to_bucket_turn"):
        lib.bps_hand_to_bucket_turn.argtypes = [
            ctypes.c_void_p, c_int_p, ctypes.c_int, ctypes.c_int,
        ]
        lib.bps_hand_to_bucket_turn.restype = ctypes.c_int
    if hasattr(lib, "bps_hand_to_bucket_river"):
        lib.bps_hand_to_bucket_river.argtypes = [
            ctypes.c_void_p, c_int_p, ctypes.c_int, ctypes.c_int,
        ]
        lib.bps_hand_to_bucket_river.restype = ctypes.c_int

    lib.bps_reader_stats.argtypes = [
        ctypes.c_void_p,
        c_int64_p, c_int64_p, c_int64_p, c_int64_p,
    ]
    lib.bps_reader_stats.restype = None

    lib.bps_save_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.bps_save_cache.restype = ctypes.c_int

    lib.bps_load_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.bps_load_cache.restype = ctypes.c_int

    # PA-EMD loaders (card_abstraction.c PA-EMD integration)
    if hasattr(lib, "bps_load_pa_emd_flop_cache"):
        lib.bps_load_pa_emd_flop_cache.argtypes = [
            ctypes.c_void_p,         # BPSReader*
            ctypes.c_char_p,         # path
        ]
        lib.bps_load_pa_emd_flop_cache.restype = ctypes.c_int

    if hasattr(lib, "bps_load_pa_emd_turn_centroids"):
        lib.bps_load_pa_emd_turn_centroids.argtypes = [
            ctypes.c_void_p,         # BPSReader*
            ctypes.c_char_p,         # path
        ]
        lib.bps_load_pa_emd_turn_centroids.restype = ctypes.c_int

    # OP3+OP1 commit 8: per-flop 500-bucket search-time bucketizer bindings.
    # Only present in bps_reader.dll builds from commit 7 onward — guard
    # with hasattr so older DLLs still load (degraded: OP3 full-depth search
    # would error out at the per-flop centroid lookup, but the rest of the
    # reader keeps working).
    if hasattr(lib, "bps_load_op1_per_flop_turn_centroids"):
        lib.bps_load_op1_per_flop_turn_centroids.argtypes = [
            ctypes.c_void_p,         # BPSReader*
            ctypes.c_char_p,         # path
        ]
        lib.bps_load_op1_per_flop_turn_centroids.restype = ctypes.c_int

    if hasattr(lib, "bps_load_op1_per_flop_river_centroids"):
        lib.bps_load_op1_per_flop_river_centroids.argtypes = [
            ctypes.c_void_p,         # BPSReader*
            ctypes.c_char_p,         # path
        ]
        lib.bps_load_op1_per_flop_river_centroids.restype = ctypes.c_int

    if hasattr(lib, "bps_compute_flop_hash"):
        lib.bps_compute_flop_hash.argtypes = [
            ctypes.c_void_p,         # BPSReader*
            c_int_p,                 # flop[3]
        ]
        lib.bps_compute_flop_hash.restype = ctypes.c_uint64

    if hasattr(lib, "bps_hand_to_bucket_turn_search"):
        lib.bps_hand_to_bucket_turn_search.argtypes = [
            ctypes.c_void_p,         # BPSReader*
            ctypes.c_uint64,         # flop_hash
            c_int_p,                 # board4[4]
            ctypes.c_int, ctypes.c_int,  # c0, c1
        ]
        lib.bps_hand_to_bucket_turn_search.restype = ctypes.c_int

    if hasattr(lib, "bps_hand_to_bucket_river_search"):
        lib.bps_hand_to_bucket_river_search.argtypes = [
            ctypes.c_void_p,         # BPSReader*
            ctypes.c_uint64,         # flop_hash
            c_int_p,                 # board5[5]
            ctypes.c_int, ctypes.c_int,  # c0, c1
        ]
        lib.bps_hand_to_bucket_river_search.restype = ctypes.c_int

    lib.bps_reader_free.argtypes = [ctypes.c_void_p]
    lib.bps_reader_free.restype = None

    # EV walk (bps_ev_walk.c) — only available in bps_ev_walk.dll/so
    if hasattr(lib, "bps_compute_evs"):
        c_float_p = ctypes.POINTER(ctypes.c_float)
        lib.bps_compute_evs.argtypes = [
            ctypes.c_void_p,         # BPSReader*
            ctypes.c_int64,          # num_iterations
            ctypes.c_int,            # num_threads
            ctypes.c_int,            # num_players
            ctypes.c_int,            # small_blind
            ctypes.c_int,            # big_blind
            ctypes.c_int,            # initial_stack
            c_float_p,              # postflop_bet_sizes
            ctypes.c_int,            # num_postflop_bet_sizes
            c_float_p,              # subsequent_bet_sizes
            ctypes.c_int,            # num_subsequent_bet_sizes
            c_float_p,              # preflop_tier0
            ctypes.c_int,            # n_tier0
            c_float_p,              # preflop_tier1
            ctypes.c_int,            # n_tier1
            c_float_p,              # preflop_tier2
            ctypes.c_int,            # n_tier2
            c_float_p,              # preflop_tier3
            ctypes.c_int,            # n_tier3
            ctypes.c_int,            # preflop_max_raises
            c_float_p,              # turn_centroids
            ctypes.c_int,            # turn_k
        ]
        lib.bps_compute_evs.restype = ctypes.c_int

    if hasattr(lib, "bps_compute_evs_checkpointed"):
        c_float_p = ctypes.POINTER(ctypes.c_float)
        lib.bps_compute_evs_checkpointed.argtypes = [
            ctypes.c_void_p,         # BPSReader*
            ctypes.c_int64,          # num_iterations
            ctypes.c_int,            # num_threads
            ctypes.c_int,            # num_players
            ctypes.c_int,            # small_blind
            ctypes.c_int,            # big_blind
            ctypes.c_int,            # initial_stack
            c_float_p,              # postflop_bet_sizes
            ctypes.c_int,            # num_postflop_bet_sizes
            c_float_p,              # subsequent_bet_sizes
            ctypes.c_int,            # num_subsequent_bet_sizes
            c_float_p,              # preflop_tier0
            ctypes.c_int,            # n_tier0
            c_float_p,              # preflop_tier1
            ctypes.c_int,            # n_tier1
            c_float_p,              # preflop_tier2
            ctypes.c_int,            # n_tier2
            c_float_p,              # preflop_tier3
            ctypes.c_int,            # n_tier3
            ctypes.c_int,            # preflop_max_raises
            c_float_p,              # turn_centroids
            ctypes.c_int,            # turn_k
            ctypes.c_int64,          # checkpoint_interval
            ctypes.c_char_p,         # checkpoint_path
        ]
        lib.bps_compute_evs_checkpointed.restype = ctypes.c_int


class BPSReaderC:
    """C-backed reader for unified .bps blueprint files.

    Drop-in replacement for BlueprintV2's query methods. Python handles
    file I/O and LZMA decompression; C handles hash table construction
    and all lookups.

    Typical memory: ~4 GB steady-state for the full 1.5B blueprint
    (~7 GB for the 3.4B). Load uses streaming LZMA decompression to a
    tempfile + mmap, so peak Python RSS during load is only the C
    hash table itself (a few extra MB for the streaming chunk
    buffer). The decompressed blob lives in page cache for the
    duration of the load and is evictable afterwards.
    """

    # Size of the BPSReader C struct. Contains inline arrays for
    # texture cache hashmap (BPS_MAX_TEXTURES * 8 + BPS_TEXTURE_INDEX_SIZE * 4).
    # Actual size is ~30KB; allocate 64KB to accommodate padding.
    _READER_SIZE = 65536

    def __init__(self, bps_path: str) -> None:
        self._lib = _find_lib()
        _setup_ctypes(self._lib)

        # Allocate the C struct (zero-initialized)
        self._reader_buf = (ctypes.c_char * self._READER_SIZE)()
        ctypes.memset(self._reader_buf, 0, self._READER_SIZE)
        self._reader = ctypes.cast(self._reader_buf, ctypes.c_void_p)

        self._metadata: dict = {}
        self._load(bps_path)

    def _load(self, bps_path: str) -> None:
        """Load the .bps file. Tries a binary cache first (~1s), falls back
        to LZMA decompression + hash table build (~21 min) and saves the
        cache for next time."""
        cache_path = bps_path + ".cache"

        # Try fast cache load first
        if os.path.exists(cache_path):
            bps_mtime = os.path.getmtime(bps_path)
            cache_mtime = os.path.getmtime(cache_path)
            if cache_mtime >= bps_mtime:
                print(
                    f"[BPSReaderC] Loading from cache: {cache_path}",
                    flush=True,
                )
                import time as _time
                t0 = _time.monotonic()
                ret = self._lib.bps_load_cache(
                    self._reader, cache_path.encode("utf-8")
                )
                if ret == 0:
                    elapsed = _time.monotonic() - t0
                    print(
                        f"[BPSReaderC] Cache loaded in {elapsed:.1f}s",
                        flush=True,
                    )
                    # Still need metadata for bucket counts etc.
                    self._load_metadata_only(bps_path)
                    self._load_texture_cache_auto(bps_path)
                    self._report_stats()
                    return
                print(
                    f"[BPSReaderC] Cache load failed, falling back to full load",
                    flush=True,
                )

        self._load_full(bps_path, cache_path)

    def _load_metadata_only(self, bps_path: str) -> None:
        """Read just the JSON metadata from the .bps without decompressing."""
        with open(bps_path, "rb") as f:
            magic = f.read(4)
            if magic != b"BPS3":
                return
            compressed_size, meta_size = struct.unpack("<QI", f.read(12))
            f.seek(4 + 12 + compressed_size)
            meta_bytes = f.read(meta_size)
            self._metadata = json.loads(meta_bytes.decode("utf-8"))

    def _load_full(self, bps_path: str, cache_path: str) -> None:
        """Full slow load: LZMA stream-decompress + hash table build + save cache.

        Memory profile:
          - Old code: peak ~43 GB on 3.4B (.bps reads 18 GB compressed
            into a Python bytes, then lzma.decompress() allocates ~25 GB
            decompressed; both live simultaneously until the C copy
            finishes).
          - New code: peak ~4 GB (just the C hash table) plus a few MB
            for the streaming chunk buffer. Compressed bytes stream
            from the source file straight into LZMADecompressor; the
            decompressed output goes to a tempfile that gets mmap'd
            for C consumption.
        """
        with open(bps_path, "rb") as f:
            magic = f.read(4)
            if magic != b"BPS3":
                raise ValueError(f"Expected BPS3 magic, got {magic!r}")

            # Outer header: compressed_size(u64) + meta_size(u32)
            compressed_size, meta_size = struct.unpack("<QI", f.read(12))

            # Read metadata FIRST by seeking past the compressed strats.
            # The mixer/bucket config lives in the JSON tail and we need
            # it before invoking bps_load_strategies. After reading meta
            # we'll seek back to the strats and stream-decompress them.
            strats_offset = f.tell()
            f.seek(strats_offset + compressed_size)
            meta_bytes = f.read(meta_size)
            self._metadata = json.loads(meta_bytes.decode("utf-8"))

            # Determine mixer mode.
            #
            # IMPORTANT: the 1.5B .bps has a mislabeled hash_mixer tag.
            # The action_hash VALUES in the file are boost-style (from v2
            # training before commit 48da71b), but the metadata says
            # "splitmix64" because the export ran after the mixer-split
            # commit. Override with BPS_FORCE_BOOST_MIXER=1 or the
            # constructor's force_boost_mixer parameter.
            mixer_tag = self._metadata.get("hash_mixer", "boost")
            force_boost = os.environ.get("BPS_FORCE_BOOST_MIXER", "0") == "1"
            if force_boost and mixer_tag != "boost":
                print(
                    f"[BPSReaderC] Overriding hash_mixer tag "
                    f"'{mixer_tag}' → 'boost' (BPS_FORCE_BOOST_MIXER=1)",
                    flush=True,
                )
                mixer_tag = "boost"
            use_legacy = 1 if mixer_tag == "boost" else 0
            nb_post = self._metadata.get("postflop_buckets", 200)
            nb_pre = self._metadata.get("preflop_buckets", 169)

            print(
                f"[BPSReaderC] Stream-decompressing strategies "
                f"({compressed_size / (1024**2):.1f} MB compressed)...",
                flush=True,
            )

            f.seek(strats_offset)
            ret = self._stream_decompress_and_load(
                f, compressed_size, "strategies",
                lambda data_ptr, data_len: self._lib.bps_load_strategies(
                    self._reader, data_ptr, data_len,
                    use_legacy, nb_post, nb_pre,
                ),
            )
            if ret != 0:
                raise RuntimeError(f"bps_load_strategies failed (ret={ret})")

            # `f` is now positioned just past the strats compressed blob.
            # Skip the metadata block (already consumed earlier in-memory)
            # and look for an optional BPR3 EV trailing section.
            f.seek(strats_offset + compressed_size + meta_size)

            # Load action EVs if present
            schema_version = self._metadata.get("schema_version", 2)
            has_evs = self._metadata.get("has_action_evs", False)

            if schema_version >= 3 and has_evs:
                ev_magic = f.read(4)
                if ev_magic == b"BPR3":
                    ev_compressed_size = struct.unpack("<Q", f.read(8))[0]
                    print(
                        f"[BPSReaderC] Stream-decompressing action EVs "
                        f"({ev_compressed_size / (1024**2):.1f} MB compressed)...",
                        flush=True,
                    )
                    ret = self._stream_decompress_and_load(
                        f, ev_compressed_size, "action EVs",
                        lambda data_ptr, data_len: self._lib.bps_load_evs(
                            self._reader, data_ptr, data_len,
                        ),
                    )
                    if ret != 0:
                        print(
                            f"[BPSReaderC] WARNING: bps_load_evs failed (ret={ret}), "
                            f"falling back to no-EV mode",
                            flush=True,
                        )
                else:
                    print(
                        f"[BPSReaderC] No BPR3 trailing section found "
                        f"(got {ev_magic!r})",
                        flush=True,
                    )

        # Save cache for fast reload next time
        if cache_path:
            print(
                f"[BPSReaderC] Saving cache to {cache_path}...",
                flush=True,
            )
            self._lib.bps_save_cache(
                self._reader, cache_path.encode("utf-8")
            )

        self._load_texture_cache_auto(bps_path)
        self._report_stats()

    def _stream_decompress_and_load(self, src_file, compressed_size: int,
                                     label: str, c_call) -> int:
        """Stream-decompress an LZMA blob and hand it to a C loader.

        Pipeline:
          1. Stream-decompress `compressed_size` bytes from `src_file`
             (currently positioned at the start of the LZMA stream) into
             a tempfile. Peak Python RAM = chunk size (~32 MB).
          2. mmap the tempfile in COPY mode. ACCESS_COPY is "private,
             copy-on-write" — ctypes treats it as a writable buffer
             (so `from_buffer` works without complaint), but the OS
             only actually copies pages we modify, and we never write
             through this pointer. So total RSS contribution is just
             whatever pages C touches during the load — typically the
             whole blob is sequential-read once, putting it briefly in
             page cache, then it's all evictable.
          3. Get the C-pointer to the mmap'd region via ctypes and call
             `c_call(data_ptr, data_len)`.
          4. Tear everything down: close mmap, unlink tempfile.

        Returns the int returned by `c_call`.
        """
        tf_path = _stream_lzma_to_tempfile(src_file, compressed_size, label)
        mm = None
        rf = None
        try:
            rf = open(tf_path, "rb")
            n = os.path.getsize(tf_path)
            if n == 0:
                print(f"[BPSReaderC]   {label}: empty after decompress",
                      flush=True)
                return -1
            # ACCESS_COPY: shared page cache, no actual copy unless we
            # write. ctypes from_buffer requires a writable buffer.
            mm = mmap.mmap(rf.fileno(), n, access=mmap.ACCESS_COPY)
            # Zero-copy view into the mmap: a c_char[n] sitting on the
            # mapped pages. addressof() is the pointer C consumes.
            view = (ctypes.c_char * n).from_buffer(mm)
            data_ptr = ctypes.addressof(view)
            print(
                f"[BPSReaderC]   {label}: handing {n / (1024**2):.0f} MB to C "
                f"(mmap COW, ptr=0x{data_ptr:x})",
                flush=True,
            )
            ret = c_call(data_ptr, n)
            # Drop the ctypes view BEFORE closing the mmap so the buffer
            # export count goes back to zero (mmap.close() raises
            # BufferError otherwise).
            del view
            return int(ret)
        finally:
            try:
                if mm is not None:
                    mm.close()
            except Exception:
                pass
            try:
                if rf is not None:
                    rf.close()
            except Exception:
                pass
            try:
                os.unlink(tf_path)
            except Exception:
                pass

    def _load_texture_cache_auto(self, bps_path: str) -> None:
        """Try to auto-load texture cache from well-known locations."""
        mixer_tag = self._metadata.get("hash_mixer", "boost")
        force_boost = os.environ.get("BPS_FORCE_BOOST_MIXER", "0") == "1"
        use_legacy = 1 if (mixer_tag == "boost" or force_boost) else 0

        bps_dir = Path(bps_path).parent
        repo_root = Path(__file__).resolve().parent.parent
        candidates = [
            bps_dir / "texture_cache.bin",
            repo_root / "texture_cache.bin",
            Path("texture_cache.bin"),
        ]
        for tc_path in candidates:
            if tc_path.exists():
                ret = self._lib.bps_load_texture_cache(
                    self._reader,
                    str(tc_path).encode("utf-8"),
                    use_legacy,
                )
                if ret > 0:
                    break

        # Auto-load turn centroids so hand_to_bucket_turn works. Required
        # for rollouts that reach a turn decision. Silently skipped if
        # the lib doesn't have the symbol (older DLL build) or no
        # centroids file exists anywhere.
        self._load_turn_centroids_auto(bps_path)

    def _load_turn_centroids_auto(self, bps_path: str) -> None:
        """Auto-load TCN1 turn centroids from standard locations."""
        if not hasattr(self._lib, "bps_load_turn_centroids"):
            return
        bps_dir = Path(bps_path).parent
        repo_root = Path(__file__).resolve().parent.parent
        candidates = [
            bps_dir / "turn_centroids.bin",
            repo_root / "turn_centroids.bin",
            Path("turn_centroids.bin"),
            Path("/tmp/turn_centroids.bin"),
            Path("C:/tmp/turn_centroids.bin"),
        ]
        for tc in candidates:
            try:
                if tc.exists():
                    ret = self._lib.bps_load_turn_centroids(
                        self._reader, str(tc).encode("utf-8"),
                    )
                    if ret > 0:
                        return
            except (OSError, ValueError):
                continue
        # Pass NULL to let C search its internal candidate list too
        # (catches /tmp paths that exist on Linux training boxes).
        self._lib.bps_load_turn_centroids(self._reader, None)

    def _report_stats(self) -> None:
        """Print loaded reader statistics."""
        n_nodes = ctypes.c_int64(0)
        n_ev = ctypes.c_int64(0)
        strat_bytes = ctypes.c_int64(0)
        ev_floats = ctypes.c_int64(0)
        self._lib.bps_reader_stats(
            self._reader,
            ctypes.byref(n_nodes),
            ctypes.byref(n_ev),
            ctypes.byref(strat_bytes),
            ctypes.byref(ev_floats),
        )
        print(
            f"[BPSReaderC] Ready: {n_nodes.value:,} strategy nodes "
            f"({strat_bytes.value / (1024**2):.1f} MB), "
            f"{n_ev.value:,} EV nodes "
            f"({ev_floats.value * 4 / (1024**2):.1f} MB)",
            flush=True,
        )

    def __del__(self) -> None:
        if hasattr(self, "_lib") and hasattr(self, "_reader"):
            self._lib.bps_reader_free(self._reader)

    # ── BlueprintV2-compatible query API ──

    def has_action_evs(self) -> bool:
        """Whether action EVs are loaded (Phase 1.3 BPR3 section)."""
        return bool(self._lib.bps_has_action_evs(self._reader))

    def get_all_bucket_strategies(
        self,
        board: List[int],
        action_history: List[int],
        player: int,
        street: int = 1,
    ) -> Optional[np.ndarray]:
        """Get strategies for ALL buckets at a node.

        Returns [num_buckets, num_actions] float32 array, or None.
        Compatible with BlueprintV2.get_all_bucket_strategies().
        """
        action_arr = (ctypes.c_int * len(action_history))(*action_history)
        nb_max = self._metadata.get(
            "postflop_buckets" if street > 0 else "preflop_buckets", 200
        )
        buf_size = nb_max * _MAX_ACTIONS
        buf = (ctypes.c_float * buf_size)()
        nb_out = ctypes.c_int(0)

        na = self._lib.bps_query_all_strategies(
            self._reader,
            action_arr,
            len(action_history),
            player,
            street,
            buf,
            buf_size,
            ctypes.byref(nb_out),
        )
        if na == 0:
            return None

        result = np.ctypeslib.as_array(buf, shape=(buf_size,))
        nb = nb_out.value
        return result[: nb * na].reshape(nb, na).copy()

    def get_all_bucket_action_evs(
        self,
        board: List[int],
        action_history: List[int],
        player: int,
        street: int = 1,
    ) -> Optional[np.ndarray]:
        """Get per-action EVs for ALL buckets at a node.

        Returns [num_buckets, num_actions] float32 array (chips), or None.
        Compatible with BlueprintV2.get_all_bucket_action_evs().
        """
        action_arr = (ctypes.c_int * len(action_history))(*action_history)
        nb_max = self._metadata.get(
            "postflop_buckets" if street > 0 else "preflop_buckets", 200
        )
        buf_size = nb_max * _MAX_ACTIONS
        buf = (ctypes.c_float * buf_size)()
        nb_out = ctypes.c_int(0)

        na = self._lib.bps_query_all_evs(
            self._reader,
            action_arr,
            len(action_history),
            player,
            street,
            buf,
            buf_size,
            ctypes.byref(nb_out),
        )
        if na == 0:
            return None

        result = np.ctypeslib.as_array(buf, shape=(buf_size,))
        nb = nb_out.value
        return result[: nb * na].reshape(nb, na).copy()

    def get_strategy(
        self,
        board: List[int],
        action_history: List[int],
        player: int,
        bucket: int,
        street: int = 1,
    ) -> Optional[np.ndarray]:
        """Get strategy at a specific (node, bucket).

        Returns float32 array of length num_actions, or None.
        Compatible with BlueprintV2.get_strategy().
        """
        action_arr = (ctypes.c_int * len(action_history))(*action_history)
        buf = (ctypes.c_float * _MAX_ACTIONS)()

        na = self._lib.bps_query_strategy(
            self._reader,
            action_arr,
            len(action_history),
            player,
            street,
            bucket,
            buf,
        )
        if na == 0:
            return None

        result = np.ctypeslib.as_array(buf, shape=(_MAX_ACTIONS,))
        return result[:na].copy()

    def load_texture_cache(self, path: str, legacy_mixer: bool = True) -> int:
        """Load a texture cache file (TXC1 format) for hand → bucket mapping.

        Args:
            path: path to texture_cache.bin
            legacy_mixer: True if the cache was built with boost mixer

        Returns number of textures loaded, or -1 on error.
        """
        return self._lib.bps_load_texture_cache(
            self._reader, path.encode("utf-8"), 1 if legacy_mixer else 0
        )

    def hand_to_bucket(self, board: List[int], c0: int, c1: int) -> int:
        """Map a hand on a flop board to its bucket index.

        Uses the precomputed texture cache (must be loaded first).

        Args:
            board: 3 flop cards (int encoding: rank*4 + suit)
            c0, c1: hero's hole cards

        Returns bucket index (0 to num_buckets-1), or -1 on error.
        """
        board_arr = (ctypes.c_int * 3)(*board[:3])
        return self._lib.bps_hand_to_bucket(self._reader, board_arr, c0, c1)

    def hand_to_bucket_turn(self, board: List[int], c0: int, c1: int) -> int:
        """Map a hand on a 4-card turn board to its bucket index.

        Uses k-means centroids (must be auto-loaded at init or via
        load_turn_centroids()). Matches bps_ev_walk.c training-time
        bucketing exactly.

        Args:
            board: 4 turn cards
            c0, c1: hero's hole cards

        Returns bucket index (0 to k-1), or -1 if centroids not loaded
        or hand conflicts with board.
        """
        if not hasattr(self._lib, "bps_hand_to_bucket_turn"):
            return -1
        board_arr = (ctypes.c_int * 4)(*board[:4])
        return self._lib.bps_hand_to_bucket_turn(self._reader, board_arr, c0, c1)

    def hand_to_bucket_river(self, board: List[int], c0: int, c1: int) -> int:
        """Map a hand on a 5-card river board to its bucket index.

        Uses deterministic 200-sample EHS percentile bucketing.
        Matches bps_ev_walk.c training-time bucketing exactly (same
        RNG seed derivation from cards + board).

        Args:
            board: 5 river cards
            c0, c1: hero's hole cards

        Returns bucket index (0 to num_postflop_buckets-1), or -1 if
        hand conflicts with board.
        """
        if not hasattr(self._lib, "bps_hand_to_bucket_river"):
            return -1
        board_arr = (ctypes.c_int * 5)(*board[:5])
        return self._lib.bps_hand_to_bucket_river(self._reader, board_arr, c0, c1)

    def load_turn_centroids(self, path: Optional[str] = None) -> int:
        """Load turn k-means centroids (TCN1 format).

        Args:
            path: explicit path, or None to search standard locations
                (turn_centroids.bin in cwd, /tmp, C:/tmp, /dev/shm).

        Returns k (centroid count) on success, -1 on miss.
        """
        if not hasattr(self._lib, "bps_load_turn_centroids"):
            return -1
        encoded = path.encode("utf-8") if path else None
        return self._lib.bps_load_turn_centroids(self._reader, encoded)

    def load_pa_emd_flop_cache(self, path: str) -> int:
        """Load PA-EMD flop bucket cache (PAEB format).

        After load, hand_to_bucket() uses PA-EMD buckets for all textures.

        Returns num_textures on success, -1 on error.
        """
        if not hasattr(self._lib, "bps_load_pa_emd_flop_cache"):
            return -1
        return self._lib.bps_load_pa_emd_flop_cache(
            self._reader, path.encode("utf-8")
        )

    def load_pa_emd_turn_centroids(self, path: str) -> int:
        """Load PA-EMD turn centroids (TCNE format).

        After load, hand_to_bucket_turn() uses EMD path instead of L2.

        Returns k on success, -1 on error.
        """
        if not hasattr(self._lib, "bps_load_pa_emd_turn_centroids"):
            return -1
        return self._lib.bps_load_pa_emd_turn_centroids(
            self._reader, path.encode("utf-8")
        )

    # ── OP3+OP1 commit 8: search-time per-flop bucketizer ────────────
    # Feeds the future-round bucket-keyed regret lookup added in
    # street_solve.cu (commit 5). Keep the 200-bucket PA-EMD turn
    # centroids loaded separately for rollout blueprint sigma lookups —
    # the two code paths coexist.

    def load_op1_per_flop_turn_centroids(self, path: str) -> int:
        """Load OP1 per-flop 500-bucket PA-EMD turn centroids (TCN3 format).

        Produced by tools/precompute_op1_per_flop_centroids.py. Required
        for CFR future-round bucket lookups under POKER_SEARCH_DEPTH=full.
        The texture cache must already be loaded (shape validation).

        Returns k (centroid count) on success, -1 on error.
        """
        if not hasattr(self._lib, "bps_load_op1_per_flop_turn_centroids"):
            return -1
        return self._lib.bps_load_op1_per_flop_turn_centroids(
            self._reader, path.encode("utf-8")
        )

    def load_op1_per_flop_river_centroids(self, path: str) -> int:
        """Load OP1 per-flop 500-bucket PA-EMD river centroids (TCN3 format)."""
        if not hasattr(self._lib, "bps_load_op1_per_flop_river_centroids"):
            return -1
        return self._lib.bps_load_op1_per_flop_river_centroids(
            self._reader, path.encode("utf-8")
        )

    def compute_flop_hash(self, flop: List[int]) -> int:
        """Compute the canonical flop hash used to index the texture cache
        and OP1 per-flop centroid slices. Mirrors the C-side compute_board_hash
        with the reader's loaded mixer mode.

        Args:
            flop: 3 cards (any order; canonicalized internally).

        Returns: uint64 flop hash, or 0 on error (uninitialized reader /
        bad flop). Real hashes are essentially never 0 in practice.
        """
        if not hasattr(self._lib, "bps_compute_flop_hash"):
            return 0
        flop_arr = (ctypes.c_int * 3)(*flop[:3])
        return int(self._lib.bps_compute_flop_hash(self._reader, flop_arr))

    def hand_to_bucket_turn_search(self, flop_hash: int, board4: List[int],
                                     c0: int, c1: int) -> int:
        """Search-time 500-bucket PA-EMD turn bucket for a hand on a
        4-card turn board. Requires load_op1_per_flop_turn_centroids
        + texture cache already loaded.

        Strict: flop_hash=0 → -1 (no flop-global fallback).
        """
        if not hasattr(self._lib, "bps_hand_to_bucket_turn_search"):
            return -1
        board_arr = (ctypes.c_int * 4)(*board4[:4])
        return int(self._lib.bps_hand_to_bucket_turn_search(
            self._reader, ctypes.c_uint64(flop_hash), board_arr, c0, c1))

    def hand_to_bucket_river_search(self, flop_hash: int, board5: List[int],
                                      c0: int, c1: int) -> int:
        """Search-time 500-bucket PA-EMD river bucket for a hand on a
        5-card river board."""
        if not hasattr(self._lib, "bps_hand_to_bucket_river_search"):
            return -1
        board_arr = (ctypes.c_int * 5)(*board5[:5])
        return int(self._lib.bps_hand_to_bucket_river_search(
            self._reader, ctypes.c_uint64(flop_hash), board_arr, c0, c1))

    def compute_evs(
        self,
        num_iterations: int = 50_000_000,
        num_threads: int = 8,
        num_players: int = 6,
        small_blind: int = 50,
        big_blind: int = 100,
        initial_stack: int = 10000,
        postflop_bet_sizes: Optional[List[float]] = None,
        subsequent_bet_sizes: Optional[List[float]] = None,
        preflop_tiers: Optional[List[List[float]]] = None,
        preflop_max_raises: int = 4,
        turn_centroids: Optional[np.ndarray] = None,
        checkpoint_interval: int = 0,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Compute per-action EVs via Monte Carlo EV walk.

        Requires bps_ev_walk library (build with 'make bps_ev_walk').

        Args:
            num_iterations: MC iterations (50M ~1hr, 500M ~overnight)
            num_threads: OpenMP threads
            num_players: 2-6
            small_blind, big_blind, initial_stack: game parameters
            postflop_bet_sizes: first-raise pot fractions (default Pluribus)
            subsequent_bet_sizes: re-raise pot fractions (default Pluribus)
            preflop_tiers: list of up to 4 lists of bet fractions per tier
            preflop_max_raises: cap on preflop raises
            turn_centroids: [k, 3] array of turn k-means centroids
            checkpoint_interval: if >0, snapshot EVs to checkpoint_path every
                N iterations. Writes are atomic (tmp + rename) so a crash
                leaves the previous checkpoint intact.
            checkpoint_path: destination for periodic snapshots. Required
                when checkpoint_interval > 0.
        """
        if not hasattr(self._lib, "bps_compute_evs"):
            raise RuntimeError(
                "bps_compute_evs not available. Build with 'make bps_ev_walk'."
            )
        if checkpoint_interval > 0 and not checkpoint_path:
            raise ValueError(
                "checkpoint_path is required when checkpoint_interval > 0"
            )
        if checkpoint_interval > 0 and not hasattr(
            self._lib, "bps_compute_evs_checkpointed"
        ):
            raise RuntimeError(
                "bps_compute_evs_checkpointed not available. Rebuild "
                "bps_ev_walk after updating src/bps_ev_walk.c."
            )

        # Defaults: Pluribus bet sizes
        if postflop_bet_sizes is None:
            postflop_bet_sizes = [0.5, 1.0]
        if subsequent_bet_sizes is None:
            subsequent_bet_sizes = [1.0]
        if preflop_tiers is None:
            # Pluribus-style tiered preflop sizing
            preflop_tiers = [
                [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # open raise
                [2.0, 3.0, 4.0],  # 3-bet
                [2.0, 3.0],       # 4-bet
                [2.0],            # 5-bet
            ]

        def _to_c_float_array(lst: List[float]) -> ctypes.Array:
            return (ctypes.c_float * len(lst))(*lst)

        pf_arr = _to_c_float_array(postflop_bet_sizes)
        sb_arr = _to_c_float_array(subsequent_bet_sizes)

        # Preflop tiers (up to 4)
        tier_arrs = []
        tier_lens = []
        for i in range(4):
            if i < len(preflop_tiers) and preflop_tiers[i]:
                arr = _to_c_float_array(preflop_tiers[i])
                tier_arrs.append(arr)
                tier_lens.append(len(preflop_tiers[i]))
            else:
                tier_arrs.append(None)
                tier_lens.append(0)

        # Turn centroids
        tc_ptr = None
        tc_k = 0
        if turn_centroids is not None and len(turn_centroids) > 0:
            tc_k = len(turn_centroids)
            tc_flat = turn_centroids.astype(np.float32).flatten()
            tc_ptr = tc_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if checkpoint_interval > 0:
            ret = self._lib.bps_compute_evs_checkpointed(
                self._reader,
                ctypes.c_int64(num_iterations),
                num_threads,
                num_players,
                small_blind,
                big_blind,
                initial_stack,
                pf_arr, len(postflop_bet_sizes),
                sb_arr, len(subsequent_bet_sizes),
                tier_arrs[0], tier_lens[0],
                tier_arrs[1], tier_lens[1],
                tier_arrs[2], tier_lens[2],
                tier_arrs[3], tier_lens[3],
                preflop_max_raises,
                tc_ptr, tc_k,
                ctypes.c_int64(checkpoint_interval),
                checkpoint_path.encode("utf-8"),
            )
        else:
            ret = self._lib.bps_compute_evs(
                self._reader,
                ctypes.c_int64(num_iterations),
                num_threads,
                num_players,
                small_blind,
                big_blind,
                initial_stack,
                pf_arr, len(postflop_bet_sizes),
                sb_arr, len(subsequent_bet_sizes),
                tier_arrs[0], tier_lens[0],
                tier_arrs[1], tier_lens[1],
                tier_arrs[2], tier_lens[2],
                tier_arrs[3], tier_lens[3],
                preflop_max_raises,
                tc_ptr, tc_k,
            )
        if ret != 0:
            raise RuntimeError(f"bps_compute_evs failed (ret={ret})")

    def save_cache(self, cache_path: str) -> None:
        """Save strategies + EVs to binary cache for fast reload."""
        ret = self._lib.bps_save_cache(
            self._reader, cache_path.encode("utf-8")
        )
        if ret != 0:
            raise RuntimeError(f"bps_save_cache failed (ret={ret})")

    def get_metadata(self, texture_key: str = "__unified__") -> Optional[dict]:
        """Get metadata from the loaded .bps file."""
        return self._metadata if self._metadata else None
