"""Smoke test: call the OCHS C functions via ctypes.

Exercises the same surface the blueprint worker hits at runtime: if the
symbols resolve and the structures marshal correctly, the trainer won't
blow up on the first OCHS deal.

Skipped when no DLL is available (e.g. pristine checkout without a build).
"""
from __future__ import annotations

import ctypes
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_ca_dll() -> str | None:
    candidates = [
        os.path.join(REPO, "build", "card_abstraction.so"),
        os.path.join(REPO, "build", "card_abstraction.dll"),
        os.path.join(REPO, "build", "libcard_abstraction.so"),
        # card_abstraction is typically linked into mccfr_blueprint.dll or solver_v2.dll.
        os.path.join(REPO, "build", "mccfr_blueprint.so"),
        os.path.join(REPO, "build", "mccfr_blueprint.dll"),
        os.path.join(REPO, "solver_v2.dll"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


@pytest.fixture(scope="module")
def ca_lib() -> ctypes.CDLL:
    path = _find_ca_dll()
    if path is None:
        pytest.skip("No card_abstraction DLL built; run make to compile first.")
    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        pytest.skip(f"Could not load {path}: {exc}")

    # Resolve OCHS symbols; skip if the DLL was built without OCHS.
    try:
        lib.ca_compute_ochs_clusters.restype = ctypes.c_int
        lib.ca_compute_ochs_clusters.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
        ]
        lib.ca_river_ochs_features.restype = ctypes.c_int
        lib.ca_river_ochs_features.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.ca_nearest_centroid_nd.restype = ctypes.c_int
        lib.ca_nearest_centroid_nd.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
    except AttributeError:
        pytest.skip("DLL does not export OCHS symbols; rebuild required.")
    return lib


def test_compute_ochs_clusters_169_in_range(ca_lib: ctypes.CDLL) -> None:
    cluster_map = (ctypes.c_int * 169)()
    k = ca_lib.ca_compute_ochs_clusters(8, 1000, cluster_map)
    assert 1 <= k <= 8
    assert all(0 <= cluster_map[i] < 8 for i in range(169))


def test_river_ochs_features_on_known_board(ca_lib: ctypes.CDLL) -> None:
    # Board: Ah Kh Qh 7c 2d
    board = (ctypes.c_int * 5)(
        12 * 4 + 2,  # Ah
        11 * 4 + 2,  # Kh
        10 * 4 + 2,  # Qh
        5 * 4 + 0,   # 7c
        0 * 4 + 1,   # 2d
    )
    cluster_map = (ctypes.c_int * 169)()
    ca_lib.ca_compute_ochs_clusters(8, 1000, cluster_map)

    # JhTh (royal flush on this board) vs 4c3d (air)
    royal = (ctypes.c_int * 2)(9 * 4 + 2, 8 * 4 + 2)
    air = (ctypes.c_int * 2)(2 * 4 + 0, 1 * 4 + 1)

    f_royal = (ctypes.c_float * 8)()
    f_air = (ctypes.c_float * 8)()

    hand1 = (ctypes.c_int * 2)(royal[0], royal[1])
    rc = ca_lib.ca_river_ochs_features(
        board, 5, hand1, 1, cluster_map, 8, f_royal
    )
    assert rc == 0
    hand2 = (ctypes.c_int * 2)(air[0], air[1])
    rc = ca_lib.ca_river_ochs_features(
        board, 5, hand2, 1, cluster_map, 8, f_air
    )
    assert rc == 0

    # Royal flush dominates every opponent cluster; air loses to most.
    royal_mean = sum(f_royal) / 8.0
    air_mean = sum(f_air) / 8.0
    assert royal_mean > 0.9, f"royal mean equity = {royal_mean}"
    assert air_mean < royal_mean - 0.3, (
        f"royal_mean={royal_mean} air_mean={air_mean} — too close"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
