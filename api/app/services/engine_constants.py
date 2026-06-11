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

# RIVER live re-solve provenance (2026-06-11). River trees terminate in exact
# showdown/fold terminals (no depth-limit leaves), so the flop/turn leaf-table
# machinery — and with it num_rollouts and the leaf-sampling noise the
# measured floors below characterize — does not apply. NO river exploitability
# floor has been measured, and citing a flop/turn floor for a river solve
# would over-claim quality, so river answers carry floor=None + this note.
RIVER_QUALITY_NOTE = (
    "river live re-solve: exact showdown terminals (side-pot settlement "
    "validated against an exact reference; poker-solver validate_sidepot.py "
    "fuzz < 1e-9), no depth-limit leaves so num_rollouts has no effect and "
    "there is no leaf-sampling noise; no measured exploitability floor "
    "exists for this configuration (the flop/turn floors do not apply)"
)
RIVER_QUALITY_SOURCE = (
    "poker-solver python/es_resolver/validate_sidepot.py (showdown "
    "settlement gate); no measured river exploitability floor"
)

# Self-consistent exploitability floors, %/pot/player, keyed by the FULL
# measured configuration (n_players, num_rollouts, vec_mode, readout).
# ONLY exactly-measured arms appear here; a floor is cited only when the
# running config matches the arm AND the iteration budget reaches the
# measured rung (review 2026-06-10: citing the 50M-rung floor for a
# 50k-iteration solve, or a scalar/avg number for a vec/final run,
# over-claims quality).
#
# 6p arms: NR_LADDER_2026-06-10.md (replicated by NR_LADDER_REWEIGHT).
# 5p vec-final arms: RAISE_CAP_2026-06-10.md (cap=2 rows, same protocol).
MEASURED_FLOORS_PCT_POT: Dict[Tuple[int, int, bool, str], float] = {
    (6, 3, True, "final"): 0.144,
    (6, 10, True, "final"): 0.089,
    (6, 20, True, "final"): 0.095,
    (6, 30, True, "final"): 0.073,
    (5, 3, True, "final"): 0.133,
    (5, 10, True, "final"): 0.076,
}
# The floors above are the best rung of a 5/20/50M-iteration ladder; below
# the 20M rung the floor is not established for that arm.
FLOOR_MIN_ITERATIONS = 20_000_000

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
            "5p vec-final nr=10 floor 0.076%/pot measured at the 50M rung "
            "(RAISE_CAP_2026-06-10.md); floors describe AGGREGATE "
            "exploitability, not per-spot mix reproducibility"
        ),
    ),
    6: LiveSolveConfig(
        iterations=5_000_000, num_rollouts=10, vec_mode=True, readout="final",
        note=(
            "6p vec-final nr=10 floor 0.089%/pot measured (~65s cold at "
            "96-192 cores). CAVEAT: the final-iterate readout serves "
            "near-pure per-spot mixes, and the per-spot mix varies with "
            "leaf-construct noise (reach-weighted |dSigma| 0.345 across "
            "noise seeds at nr=10 — NR_LADDER_REWEIGHT_2026-06-10.md); the "
            "floor describes aggregate exploitability, not per-spot "
            "reproducibility"
        ),
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


def measured_floor_for(
    n_players: int,
    num_rollouts: int,
    *,
    vec_mode: bool,
    readout: str,
    iterations: int,
    street: Optional[str] = None,
) -> Optional[float]:
    """The exactly-measured floor for this FULL config, or None (never
    estimated, never cross-config). None when the iteration budget is below
    the measured rung. None on RIVER unconditionally: every measured arm is
    a flop/turn leaf-table configuration and no river floor has been
    measured (see RIVER_QUALITY_NOTE)."""
    if street == "river":
        return None
    if iterations < FLOOR_MIN_ITERATIONS:
        return None
    return MEASURED_FLOORS_PCT_POT.get(
        (n_players, num_rollouts, vec_mode, readout)
    )
