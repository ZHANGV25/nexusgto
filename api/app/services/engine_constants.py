"""Per-player-count live-resolve quality configuration + measured floors.

EVERY number here is a MEASURED value with a citation — the provenance
metadata embeds these so the API never implies unmeasured quality.

Sources (poker-solver repo, measured on EC2 2026-06-10):
  - measured_logs/NR_LADDER_2026-06-10.md  — 6p self-consistent floors by
    num_rollouts (vec-Linear-FINAL, 96-192 cores): nr=3 -> 0.144, nr=10 ->
    0.089, nr=20 -> 0.095, nr=30 -> 0.073 %/pot/player; construct
    13/43/86/129 s; solve-to-floor ~23 s. Numeric bar: 0.127.
  - measured_logs/DISCRIMINATOR_2026-06-10.md — the 6p floor is leaf-sample
    noise; nr>=10 lands under the bar; nr=3 per-spot strategies are noise
    artifacts at 6p.
  - measured_logs/VEC_SPEEDUP_2026-06-10.md — vec-Linear-FINAL is the only
    validated vec config (readout='final' REQUIRED; enforced by CpuEsSolver).
  - docs/CPU_ES_RESOLVER_BENCHMARK.md — 5p ships 0.122 %/pot at ~75 s;
    2p/3p live re-solves run 1-4 s at a few hundred iterations (the
    historically validated scalar/avg analysis surface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

QUALITY_SOURCE = "poker-solver/measured_logs/NR_LADDER_2026-06-10.md"
BLUEPRINT_QUALITY_SOURCE = (
    "blueprint v4 8.05B-iteration sigma-bar (postflop 995M IS, converged "
    "L1<0.1pp)"
)
BLUEPRINT_QUALITY_NOTE = (
    "sub-second blueprint lookup — the Pluribus-style blueprint floor, no "
    "live search"
)

# Self-consistent exploitability floors, %/pot/player, keyed by
# (n_players, num_rollouts). ONLY exactly-measured combos appear here.
MEASURED_FLOORS_PCT_POT: Dict[Tuple[int, int], float] = {
    (6, 3): 0.144,
    (6, 10): 0.089,
    (6, 20): 0.095,
    (6, 30): 0.073,
    (5, 3): 0.122,  # 5p ship quality (CPU_ES_RESOLVER_BENCHMARK.md)
}

# The product's numeric quality bar (%/pot/player), anchored on 5p nr=3 ship
# quality (DISCRIMINATOR_2026-06-10.md caveat applies across nr).
NUMERIC_BAR_PCT_POT = 0.127

# Client-override bounds (the measured surface; outside = rejected).
NUM_ROLLOUTS_MIN = 3
NUM_ROLLOUTS_MAX = 30
ITERATIONS_MIN = 50
ITERATIONS_MAX = 50_000_000


@dataclass(frozen=True)
class LiveSolveConfig:
    """One live re-solve configuration (immutable)."""

    iterations: int
    num_rollouts: int
    vec_mode: bool
    readout: str  # 'avg' | 'final'; vec_mode requires 'final' (CpuEsSolver)
    note: Optional[str] = None


# Per-player-count defaults. n = players in the street-root re-solve.
#   n<=3: scalar traverser + sigma-bar readout at a few hundred iterations —
#         the validated 2p/3p analysis surface (1-4 s).
#   n>=4: vec-Linear-FINAL at 5M iterations (the Gate-1-validated playing-agent
#         path); nr=10 for 5p/6p (under-the-bar floor per NR_LADDER).
DEFAULT_LIVE_CONFIG_BY_N: Dict[int, LiveSolveConfig] = {
    2: LiveSolveConfig(
        iterations=400, num_rollouts=3, vec_mode=False, readout="avg",
        note="2p live re-solve ~1-3s (validated scalar/avg analysis surface)",
    ),
    3: LiveSolveConfig(
        iterations=400, num_rollouts=3, vec_mode=False, readout="avg",
        note="3p live re-solve ~2-4s (validated scalar/avg analysis surface)",
    ),
    4: LiveSolveConfig(
        iterations=5_000_000, num_rollouts=3, vec_mode=True, readout="final",
        note="4p vec-Linear-FINAL; no exactly-measured 4p floor",
    ),
    5: LiveSolveConfig(
        iterations=5_000_000, num_rollouts=10, vec_mode=True, readout="final",
        note=(
            "5p: measured 0.122%/pot at nr=3 (~75s); nr=10 floor not exactly "
            "measured (leaf noise shrinks with nr per DISCRIMINATOR)"
        ),
    ),
    6: LiveSolveConfig(
        iterations=5_000_000, num_rollouts=10, vec_mode=True, readout="final",
        note="6p nr=10 floor 0.089%/pot measured; ~65s cold at 96-192 cores",
    ),
}


def live_config_for(
    n_players: int,
    iterations_override: Optional[int] = None,
    num_rollouts_override: Optional[int] = None,
) -> LiveSolveConfig:
    """Resolve the live-solve config for a player count + client overrides.

    Raises:
        ValueError: unsupported player count, or an override outside the
            measured bounds (rejected, never clamped silently).
    """
    base = DEFAULT_LIVE_CONFIG_BY_N.get(n_players)
    if base is None:
        raise ValueError(
            f"unsupported player count for live re-solve: {n_players} "
            f"(supported: {sorted(DEFAULT_LIVE_CONFIG_BY_N)})"
        )
    iterations = base.iterations
    num_rollouts = base.num_rollouts
    if iterations_override is not None:
        if not (ITERATIONS_MIN <= iterations_override <= ITERATIONS_MAX):
            raise ValueError(
                f"iterations override {iterations_override} outside the "
                f"validated bounds [{ITERATIONS_MIN}, {ITERATIONS_MAX}]"
            )
        iterations = iterations_override
    if num_rollouts_override is not None:
        if not (NUM_ROLLOUTS_MIN <= num_rollouts_override <= NUM_ROLLOUTS_MAX):
            raise ValueError(
                f"num_rollouts override {num_rollouts_override} outside the "
                f"measured bounds [{NUM_ROLLOUTS_MIN}, {NUM_ROLLOUTS_MAX}]"
            )
        num_rollouts = num_rollouts_override
    return LiveSolveConfig(
        iterations=iterations,
        num_rollouts=num_rollouts,
        vec_mode=base.vec_mode,
        readout=base.readout,
        note=base.note,
    )


def measured_floor_for(n_players: int, num_rollouts: int) -> Optional[float]:
    """The exactly-measured floor for this combo, or None (never estimated)."""
    return MEASURED_FLOORS_PCT_POT.get((n_players, num_rollouts))
