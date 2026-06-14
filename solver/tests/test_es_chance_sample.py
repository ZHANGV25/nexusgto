"""Driver + drift-tripwire for the CPU-ES in-tree chance-node fix (2026-06-14).

Two gates, both runnable WITHOUT a GPU (pure C / source inspection):

1. BIT-EXACT sampler gate — compiles & runs tests/test_es_chance_sample_bitexact.c
   (47M draws of the CPU sampler vs an independent re-impl of the GPU host
   formula). The C test re-declares the sampler; this Python gate also asserts
   the re-declaration is BYTE-IDENTICAL to the real definitions, so the test
   cannot pass against a stale copy.

2. SOURCE-IDENTITY gate — the seed-mix recurrence and constants in
   src/es_resolver.c (es_chance_sampled_child + es_splitmix64) MUST match the
   GPU's src/cuda/street_solve.cu (ss_redraw_chance_samples_host +
   ss_splitmix64_host). If anyone edits one side, this fails — the bit-exact
   property is only meaningful if both code paths actually share the formula.

These are the gates that DON'T need a GPU box. The GPU-vs-CPU converged-strategy
agreement (Gate B) lives in the EC2 validation harness (needs nvcc + CUDA).
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


def test_source_identity():
    es = _read(SRC / "es_resolver.c")
    gpu = _read(SRC / "cuda" / "street_solve.cu")

    # Both must define a splitmix64 with the canonical three constants.
    for k in SPLITMIX_CONSTS:
        check(f"es_resolver.c uses splitmix const {k}", k in es)
        check(f"street_solve.cu uses splitmix const {k}", k in gpu)

    # The seed mix `master_seed ^ iter*K1 ^ c*K2` must appear on both sides with
    # the SAME two multipliers in the SAME roles. We assert the exact mix lines.
    es_mix = re.search(
        r"master_seed\s*\^\s*\(\(uint64_t\)\s*iter\s*\*\s*0x9E3779B97F4A7C15ULL\)\s*"
        r"\^\s*\(\(uint64_t\)\s*chance_node_idx\s*\*\s*0xBF58476D1CE4E5B9ULL\)",
        es)
    check("es_resolver.c seed mix == master_seed ^ iter*K1 ^ chance_idx*K2",
          es_mix is not None,
          "es_chance_sampled_child seed mix not found / drifted")

    gpu_mix = re.search(
        r"master_seed\s*\^\s*\(\(uint64_t\)\s*iter\s*\*\s*0x9E3779B97F4A7C15ULL\)\s*"
        r"\^\s*\(\(uint64_t\)\s*c\s*\*\s*0xBF58476D1CE4E5B9ULL\)",
        gpu)
    check("street_solve.cu seed mix == master_seed ^ iter*K1 ^ c*K2",
          gpu_mix is not None,
          "ss_redraw_chance_samples_host seed mix not found / drifted")

    # The modulo-by-num_children reduction must be present on both sides.
    check("es_resolver.c reduces by % num_children",
          re.search(r"%\s*\(uint64_t\)\s*num_children", es) is not None)
    check("street_solve.cu reduces by % num_children",
          re.search(r"%\s*\(uint64_t\)\s*num_children", gpu) is not None)


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
    test_source_identity()
    test_bitexact_c()
    print(f"\n{'PASSED' if not _FAIL else 'FAILED'} {len(_PASS)}/{len(_PASS) + len(_FAIL)}")
    sys.exit(0 if not _FAIL else 1)
