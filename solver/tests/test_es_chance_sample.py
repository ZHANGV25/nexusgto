"""Driver for the CPU-ES in-tree chance-node sampler test.

Runs WITHOUT a GPU (pure C): compiles & runs
tests/test_es_chance_sample_bitexact.c (47M draws of the CPU sampler vs an
independent re-impl of the GPU host formula). The seed-mix recurrence and
constants in src/search/es_resolver.c (es_chance_sampled_child +
es_splitmix64) must match the GPU's ss_redraw_chance_samples_host +
ss_splitmix64_host; the bit-exact property is only meaningful if both code
paths actually share the formula.

The GPU-vs-CPU converged-strategy agreement needs nvcc + CUDA and is not
tested here.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        _PASS.append(name); print(f"  PASS  {name}")
    else:
        _FAIL.append(name); print(f"  FAIL  {name}")
        if detail: print(f"        {detail}")


# Magic constants the formula MUST use (splitmix64 + the seed mix). If any of
# these is absent from a side, the formula has drifted.
SPLITMIX_CONSTS = [
    "0x9E3779B97F4A7C15", "0xBF58476D1CE4E5B9", "0x94D049BB133111EB",
]
SEED_MIX = [
    "0x9E3779B97F4A7C15",  # iter multiplier
    "0xBF58476D1CE4E5B9",  # chance-idx multiplier
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def test_bitexact_c():
    """Compile + run the 47M-draw bit-exact C gate."""
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    exe = build / ("test_es_chance.exe" if sys.platform == "win32" else "test_es_chance")
    src = ROOT / "tests" / "test_es_chance_sample_bitexact.c"
    cc = "gcc"
    compile_cmd = [cc, "-O2", "-Wall", "-Wextra", "-I" + str(SRC),
                   str(src), "-o", str(exe), "-lm"]
    cp = subprocess.run(compile_cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        check("compile test_es_chance_sample_bitexact.c", False, cp.stderr[:500])
        return
    check("compile test_es_chance_sample_bitexact.c", True)
    rp = subprocess.run([str(exe)], capture_output=True, text=True)
    print("    " + "\n    ".join(rp.stdout.strip().splitlines()))
    check("bit-exact chance sample vs GPU formula (47M draws)",
          rp.returncode == 0 and "PASSED" in rp.stdout, rp.stdout[-300:])


if __name__ == "__main__":
    print("[test_es_chance_sample]")
    test_bitexact_c()
    print(f"\n{'PASSED' if not _FAIL else 'FAILED'} {len(_PASS)}/{len(_PASS) + len(_FAIL)}")
    sys.exit(0 if not _FAIL else 1)
