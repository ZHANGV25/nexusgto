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
import os
import struct
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Maximum actions per decision node (must match BPS_MAX_ACTIONS in bps_reader.h)
_MAX_ACTIONS = 16


def _find_lib() -> ctypes.CDLL:
    """Locate and load the bps_reader shared library."""
    # Try relative to this file's directory (poker-solver/python/ -> poker-solver/build/)
    base = Path(__file__).resolve().parent.parent / "build"

    if sys.platform == "win32":
        candidates = [base / "bps_reader.dll"]
    else:
        candidates = [base / "bps_reader.so"]

    # Also check POKER_SOLVER_PATH env var
    env_path = os.environ.get("POKER_SOLVER_PATH")
    if env_path:
        build_dir = Path(env_path) / "build"
        if sys.platform == "win32":
            candidates.append(build_dir / "bps_reader.dll")
        else:
            candidates.append(build_dir / "bps_reader.so")

    for path in candidates:
        if path.exists():
            return ctypes.CDLL(str(path))

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"bps_reader shared library not found. Searched: {searched}. "
        f"Run 'make bps_reader' in the poker-solver directory."
    )


def _setup_ctypes(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for type safety."""
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int64_p = ctypes.POINTER(ctypes.c_int64)

    lib.bps_load_strategies.argtypes = [
        ctypes.c_void_p,         # BPSReader*
        ctypes.c_char_p,         # data
        ctypes.c_size_t,         # len
        ctypes.c_int,            # use_legacy_mixer
        ctypes.c_int,            # postflop_buckets
        ctypes.c_int,            # preflop_buckets
    ]
    lib.bps_load_strategies.restype = ctypes.c_int

    lib.bps_load_evs.argtypes = [
        ctypes.c_void_p,         # BPSReader*
        ctypes.c_char_p,         # data
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

    lib.bps_reader_stats.argtypes = [
        ctypes.c_void_p,
        c_int64_p, c_int64_p, c_int64_p, c_int64_p,
    ]
    lib.bps_reader_stats.restype = None

    lib.bps_reader_free.argtypes = [ctypes.c_void_p]
    lib.bps_reader_free.restype = None


class BPSReaderC:
    """C-backed reader for unified .bps blueprint files.

    Drop-in replacement for BlueprintV2's query methods. Python handles
    file I/O and LZMA decompression; C handles hash table construction
    and all lookups.

    Typical memory: ~4 GB steady-state for the full 1.5B blueprint.
    Peak during load: +28 GB for the decompressed strategy bytes.
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
        """Load the .bps file: decompress in Python, build tables in C."""
        with open(bps_path, "rb") as f:
            magic = f.read(4)
            if magic != b"BPS3":
                raise ValueError(f"Expected BPS3 magic, got {magic!r}")

            # Outer header: compressed_size(u64) + meta_size(u32)
            compressed_size, meta_size = struct.unpack("<QI", f.read(12))
            compressed = f.read(compressed_size)
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
                f"[BPSReaderC] Decompressing strategies "
                f"({compressed_size / (1024**2):.1f} MB compressed)...",
                flush=True,
            )

            # Decompress strategies
            strat_data = lzma.decompress(compressed)
            del compressed  # free compressed buffer early

            print(
                f"[BPSReaderC] Decompressed to {len(strat_data) / (1024**2):.1f} MB. "
                f"Loading into C hash table (mixer={mixer_tag})...",
                flush=True,
            )

            # Pass raw bytes to C
            ret = self._lib.bps_load_strategies(
                self._reader,
                strat_data,
                len(strat_data),
                use_legacy,
                nb_post,
                nb_pre,
            )
            del strat_data  # free the large decompressed buffer

            if ret != 0:
                raise RuntimeError(f"bps_load_strategies failed (ret={ret})")

            # Load action EVs if present
            schema_version = self._metadata.get("schema_version", 2)
            has_evs = self._metadata.get("has_action_evs", False)

            if schema_version >= 3 and has_evs:
                ev_magic = f.read(4)
                if ev_magic == b"BPR3":
                    ev_compressed_size = struct.unpack("<Q", f.read(8))[0]
                    ev_compressed = f.read(ev_compressed_size)

                    print(
                        f"[BPSReaderC] Decompressing action EVs "
                        f"({ev_compressed_size / (1024**2):.1f} MB compressed)...",
                        flush=True,
                    )
                    ev_data = lzma.decompress(ev_compressed)
                    del ev_compressed

                    print(
                        f"[BPSReaderC] Loading {len(ev_data) / (1024**2):.1f} MB "
                        f"of action EVs...",
                        flush=True,
                    )
                    ret = self._lib.bps_load_evs(
                        self._reader, ev_data, len(ev_data)
                    )
                    del ev_data

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

        # Try to auto-load texture cache from well-known locations
        bps_dir = Path(bps_path).parent
        repo_root = Path(__file__).resolve().parent.parent
        texture_candidates = [
            bps_dir / "texture_cache.bin",
            repo_root / "texture_cache.bin",
            Path("texture_cache.bin"),
        ]
        for tc_path in texture_candidates:
            if tc_path.exists():
                ret = self._lib.bps_load_texture_cache(
                    self._reader,
                    str(tc_path).encode("utf-8"),
                    use_legacy,
                )
                if ret > 0:
                    break

        # Report stats
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

    def get_metadata(self, texture_key: str = "__unified__") -> Optional[dict]:
        """Get metadata from the loaded .bps file."""
        return self._metadata if self._metadata else None
