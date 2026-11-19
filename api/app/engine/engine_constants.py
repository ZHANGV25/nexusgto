"""Per-player-count live-resolve quality configuration + measured floors.

EVERY number here is a MEASURED value. The provenance metadata embeds
these, so a result never implies quality that was not measured.

The measurements behind the numbers:
  - 6p self-consistent floors by num_rollouts (vec-Linear-FINAL, 96-192
    cores): nr=3 -> 0.144, nr=10 -> 0.089, nr=20 -> 0.095, nr=30 -> 0.073
    %/pot/player; construct 13/43/86/129 s; solve-to-floor ~23 s. Numeric
    bar: 0.127.
  - The 6p floor is leaf-sample noise: nr>=10 lands under the bar, and nr=3
    per-spot strategies are noise artifacts at 6p.
  - vec-Linear-FINAL is the only validated vec config (readout='final' is
    REQUIRED and enforced by CpuEsSolver).
  - 5p reaches 0.122 %/pot at ~75 s. The scalar/avg analysis surface
    re-solves 2p/3p spots in 1-4 s at a few hundred iterations.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

QUALITY_SOURCE = "num_rollouts ladder measurement (flop and turn exploitability floors)"

HERO_TRAVERSER_ACTUAL_FRACTION_ENV = (
    "NEXUS_HERO_TRAVERSER_ACTUAL_FRACTION"
)
HERO_TRAVERSER_ACTUAL_FRACTION_DEFAULT = 0.5
HERO_ROW_MIN_STRATEGY_SUM_MASS_ENV = (
    "NEXUS_HERO_ROW_MIN_STRATEGY_SUM_MASS"
)
HERO_ROW_MIN_STRATEGY_SUM_MASS_DEFAULT = 10.0
BLUEPRINT_QUALITY_SOURCE = (
    "blueprint v4 8.05B-iteration sigma-bar (postflop 995M IS, converged "
    "L1<0.1pp)"
)
BLUEPRINT_QUALITY_NOTE = (
    "sub-second blueprint lookup — the Pluribus-style blueprint floor, no "
    "live search"
)

# RIVER live re-solve provenance. River trees terminate in exact
# showdown/fold terminals (no depth-limit leaves), so the flop/turn leaf-table
# machinery — and with it num_rollouts and the leaf-sampling noise the
# measured floors below characterize — does not apply. NO river exploitability
# floor has been measured, and citing a flop/turn floor for a river solve
# would over-claim quality, so river answers carry floor=None + this note.
RIVER_QUALITY_NOTE = (
    "river live re-solve: exact showdown terminals (side-pot settlement "
    "validated against an exact reference; solver/python/es_resolver/validate_sidepot.py "
    "fuzz < 1e-9), no depth-limit leaves so num_rollouts has no effect and "
    "there is no leaf-sampling noise; no measured exploitability floor "
    "exists for this configuration (the flop/turn floors do not apply)"
)
RIVER_QUALITY_SOURCE = (
    "solver/python/es_resolver/validate_sidepot.py (showdown "
    "settlement gate); no measured river exploitability floor"
)

# Self-consistent exploitability floors, %/pot/player, keyed by the FULL
# measured configuration (n_players, num_rollouts, vec_mode, readout).
# ONLY exactly-measured arms appear here; a floor is cited only when the
# running config matches the arm AND the iteration budget reaches the
# measured rung (citing the 50M-rung floor for a 50k-iteration solve, or a
# scalar/avg number for a vec/final run, would over-claim quality).
#
# 6p arms: the num_rollouts ladder (replicated by a reweighted rerun).
# 5p vec-final arms: the raise-cap ladder (cap=2 rows, same protocol).
# 4p vec-final arms: the 4p num_rollouts ladder (board AhKh2c, pot_bb=20, the
#   validated %/pot regime, same methodology as 5p/6p; best rung of the
#   5/20/50M ladder):
#     nr=3:0.067  nr=10:0.072  nr=20:0.060  nr=30:0.057 %/pot/p; all << BAR 0.127.
#     Low at every nr; no cont_ev needed (the rollout already floors well).
MEASURED_FLOORS_PCT_POT: Dict[Tuple[int, int, bool, str], float] = {
    (6, 3, True, "final"): 0.144,
    (6, 10, True, "final"): 0.089,
    (6, 20, True, "final"): 0.095,
    (6, 30, True, "final"): 0.073,
    (5, 3, True, "final"): 0.133,
    (5, 10, True, "final"): 0.076,
    (4, 3, True, "final"): 0.067,
    (4, 10, True, "final"): 0.072,
}
# The 5p/6p floors above are the best rung of a 5/20/50M-iteration ladder; below
# the 20M rung the floor is not established for THAT arm (the vec-Linear-FINAL
# multiway path). This is the DEFAULT minimum for any arm without a more
# specific entry in FLOOR_MIN_ITERATIONS_BY_ARM.
FLOOR_MIN_ITERATIONS = 20_000_000

# Per-arm minimum iterations. The single 20M global is correct for the
# vec-final ladder but wrong for the 2p scalar/avg ANALYSIS SURFACE: that arm
# is a sub-second few-hundred-iteration regime (the scalar/avg 2p/3p re-solve
# in this file's header), measured at its OWN budget, not on the 20M ladder.
# An arm listed here uses its own minimum;
# any arm not listed falls back to FLOOR_MIN_ITERATIONS. Each entry's value is the
# exact iteration budget the arm's floor was MEASURED at (cite the run).
FLOOR_MIN_ITERATIONS_BY_ARM: Dict[Tuple[int, int, bool, str], int] = {
    # (2, 3, False, "avg"): NOT POPULATED. Three measurement attempts could
    # not produce a trustworthy 2p exploitability floor: exploitability_c.py is
    # validated on the pot≈stack-scale setup (20 bb pot) and structurally
    # inflates %/pot ~16x on the real HU regime (≈6 bb pot, 95 bb stack); even
    # an UNCONDITIONED control measured ~200%/pot there. So no honest 2p floor
    # exists for this arm yet. The mechanism above is ready to accept one if
    # the metric is re-derived for low-pot/high-stack trees.
    # 2p quality is instead checked behaviorally (the QJs/99 over-bet cure on
    # the real LiveResolveSource path).
    # Until then 2p reports floor=None + the "scalar/avg analysis surface"
    # note (the provenance never implies an unmeasured number).
}

# The numeric quality bar (%/pot/player), anchored on the 5p nr=3 floor
# (floors at different nr also differ by leaf-sample noise).
NUMERIC_BAR_PCT_POT = 0.127

# Client-override bounds (the measured surface; outside = rejected).
NUM_ROLLOUTS_MIN = 3
NUM_ROLLOUTS_MAX = 30
ITERATIONS_MIN = 50
ITERATIONS_MAX = 50_000_000
DEADLINE_MS_MIN = 100
DEADLINE_MS_MAX = 60_000


# ── cont_ev boundary-table policy ───────────────────────────────────────────
# Default ON for PRE-FILLED spots: cont_ev is armed for a spot that has a
# filled boundary table. Basis: the 6p mix-level ensemble check passed on both
# streets on fresh K=3 fills at the default configuration, the value-level
# EV-loss check found the 3p mix instability costs no EV (both boundaries),
# and coverage (hit rate 1.0, fail-loud) and latency (6.3/7.0 s armed at the
# 6p default configuration) were already established.
#
# WHAT "default ON for pre-filled spots" MEANS:
# arming cont_ev requires a boundary TABLE for the spot being solved. The
# engine holds ONE table per engine process (NEXUS_A3_CEV_PATH names a
# single .cev; CpuEsSolver loads it at construction iff NEXUS_A3_CONT_EV=1 and
# arms the per-solve boundary via its own predicate, see
# cpu_es_solver._a3_cont_ev_enabled and _load_cont_ev_table). Filled tables
# exist for two pinned spots only. So the POLICY this constant declares is:
# "arm cont_ev whenever a valid boundary-table artifact is configured for
# this engine process." It does NOT add multi-spot table routing or a catalog
# registry. With the one-table mechanism, a process configured with a spot's
# .cev solves that spot armed and every other spot un-armed (the boundary
# predicate never fires without a matching key; a MISS is fail-loud, with no
# silent live-rollout fallback, which keeps the armed latency win).
#
# DEFAULT OFF absent configuration: an engine with no NEXUS_A3_* env
# constructs with no cont_ev, byte-identical to the plain path. The policy
# takes effect only when an engine process is CONFIGURED with a validated
# table (NEXUS_A3_CONT_EV=1 + NEXUS_A3_CEV_PATH); it never silently arms an
# unconfigured process. FAIL-LOUD is preserved end-to-end: a
# configured-but-missing/invalid path raises at construction
# (cpu_es_solver._load_cont_ev_table), and an armed boundary MISS raises
# during the solve.
CONT_EV_ARMING_POLICY = (
    "arm cont_ev whenever a valid boundary-table artifact is "
    "configured for this engine process (NEXUS_A3_CONT_EV=1 + a readable "
    "NEXUS_A3_CEV_PATH); default OFF when unconfigured; fail-loud on a "
    "configured-but-invalid path and on an armed-boundary MISS"
)
CONT_EV_ARMING_POLICY_SOURCE = "cont-EV boundary-table experiment (off by default)"
# Env contract (read-only reflection of cpu_es_solver's construction-time
# reads — the single source of truth stays in cpu_es_solver._a3_cont_ev_enabled
# and _load_cont_ev_table; this helper only REPORTS the resolved policy so the
# provider can log/validate it at startup without duplicating the enable logic).
CONT_EV_ENABLE_ENV = "NEXUS_A3_CONT_EV"
CONT_EV_TABLE_PATH_ENV = "NEXUS_A3_CEV_PATH"
CONT_EV_TURN_ENABLE_ENV = "NEXUS_A3_CONT_EV_TURN"
# Measurement-only (the pinned-tail experiment): quietly downgrades an ARMED
# boundary MISS to a legacy live playout
# (cpu_es_solver._cont_ev_miss_fallback_enabled). Named here ONLY so a caller
# that arms a boundary (single-table or catalog) can REJECT it: a MISS during
# play must stay fail-loud.
CONT_EV_MISS_FALLBACK_ENV = "NEXUS_CONT_EV_MISS_FALLBACK"

# ── cont_ev CATALOG policy (multi-spot routing) ─────────────────────────────
# When configured, REPLACES the single-table policy above: instead of ONE table
# per process (NEXUS_A3_CEV_PATH), a MANIFEST maps many pre-filled spots to
# their .cev artifacts and a per-hand router arms the RIGHT one, or runs
# un-armed for a spot not in the catalog. Additive + OPT-IN: default OFF
# (no CONT_EV_CATALOG_MANIFEST) is byte-identical to the single-table default-OFF
# path. The catalog and the single-table NEXUS_A3_CEV_PATH are MUTUALLY EXCLUSIVE
# (two arming mechanisms cannot share the one process-global rollout pointer). The
# router and its manifest loader are not part of this module; these names are
# the authoritative env-var strings + the policy constant, alongside the
# single-table names above. NEXUS_CONT_EV_MISS_FALLBACK is DELIBERATELY not part
# of the catalog surface (measurement-only; a MISS stays fail-loud).
CONT_EV_CATALOG_POLICY = (
    "when CONT_EV_CATALOG_MANIFEST names a valid manifest, route each hand to its "
    "pre-filled cont_ev table (canonical flop texture + preflop-prefix + "
    "n_players); serve un-armed for spots not in the catalog; preserve fail-loud "
    "on an in-catalog boundary MISS; default OFF when unconfigured"
)
CONT_EV_CATALOG_POLICY_SOURCE = "cont-EV catalog routing experiment (off by default)"
CONT_EV_CATALOG_MANIFEST_ENV = "CONT_EV_CATALOG_MANIFEST"  # path to manifest JSON
CONT_EV_CATALOG_MODE_ENV = "CONT_EV_CATALOG_MODE"          # 'resident'|'lazy'
CONT_EV_CATALOG_LRU_CAP_ENV = "CONT_EV_CATALOG_LRU_CAP"    # int, lazy mode only


@dataclass(frozen=True)
class ContEvArmingDecision:
    """The resolved cont_ev arming policy for one engine process (immutable).

    A pure reflection of the environment contract — computing it has NO side
    effects and does NOT construct or arm anything. The provider logs it at
    construction and MAY assert `not error` to fail loud early (before the
    12GB blueprint load) on a misconfiguration; the authoritative fail-loud
    still lives in CpuEsSolver._load_cont_ev_table.
    """

    armed: bool               # True iff enable flag set AND a readable table path
    enable_flag: bool         # NEXUS_A3_CONT_EV truthy
    turn_enable_flag: bool    # NEXUS_A3_CONT_EV_TURN truthy (turn boundary too)
    table_path: Optional[str]  # NEXUS_A3_CEV_PATH value (may be missing/invalid)
    reason: str               # human-readable one-liner for the startup log
    error: Optional[str] = None  # set iff configured-but-invalid (fail-loud hook)


def _truthy_env(name: str) -> bool:
    """Match cpu_es_solver's construction-time predicate byte-for-byte
    (cpu_es_solver._a3_cont_ev_enabled / _a3_cont_ev_turn_enabled): first char
    in {1,t,T,y,Y}."""
    import os as _os
    return _os.environ.get(name, "")[:1] in ("1", "t", "T", "y", "Y")


def resolve_cont_ev_arming(env: Optional[Dict[str, str]] = None) -> ContEvArmingDecision:
    """Resolve CONT_EV_ARMING_POLICY from the environment.

    Injectable `env` (defaults to os.environ) so the wiring is unit-testable
    without mutating process state. Reflects — never duplicates — the enable
    logic in cpu_es_solver: enable flag + a readable table path => armed;
    enable flag + missing/unreadable path => armed-intent with `error` set (the
    provider can fail loud early); no enable flag => disarmed (default OFF).
    """
    import os as _os
    e = _os.environ if env is None else env
    enable = (e.get(CONT_EV_ENABLE_ENV, "")[:1] in ("1", "t", "T", "y", "Y"))
    turn_enable = (e.get(CONT_EV_TURN_ENABLE_ENV, "")[:1]
                   in ("1", "t", "T", "y", "Y"))
    path = e.get(CONT_EV_TABLE_PATH_ENV) or None
    if not enable:
        return ContEvArmingDecision(
            armed=False, enable_flag=False, turn_enable_flag=turn_enable,
            table_path=path,
            reason="cont_ev DISARMED (default OFF — "
                   f"{CONT_EV_ENABLE_ENV} not set)")
    # Enable flag set: a table path is now MANDATORY and must exist.
    import os.path as _osp
    if not path:
        return ContEvArmingDecision(
            armed=False, enable_flag=True, turn_enable_flag=turn_enable,
            table_path=None,
            reason=f"{CONT_EV_ENABLE_ENV} set but {CONT_EV_TABLE_PATH_ENV} "
                   "missing",
            error=f"{CONT_EV_ENABLE_ENV}=1 requires {CONT_EV_TABLE_PATH_ENV} "
                  "to name a boundary-table .cev")
    if not _osp.isfile(path):
        return ContEvArmingDecision(
            armed=False, enable_flag=True, turn_enable_flag=turn_enable,
            table_path=path,
            reason=f"{CONT_EV_TABLE_PATH_ENV}={path!r} is not a readable file",
            error=f"{CONT_EV_TABLE_PATH_ENV}={path!r} not found — cont_ev "
                  "cannot arm without its boundary table")
    boundaries = "flop+turn" if turn_enable else "flop"
    return ContEvArmingDecision(
        armed=True, enable_flag=True, turn_enable_flag=turn_enable,
        table_path=path,
        reason=f"cont_ev ARMED ({boundaries}) from table {path!r} "
               f"per CONT_EV_ARMING_POLICY ({CONT_EV_ARMING_POLICY_SOURCE})")


@dataclass(frozen=True)
class LiveSolveConfig:
    """One live re-solve configuration (immutable)."""

    iterations: int
    num_rollouts: int
    vec_mode: bool
    readout: str  # 'avg' | 'final'; vec_mode requires 'final' (CpuEsSolver)
    note: Optional[str] = None
    hero_traverser_actual_fraction: float = (
        HERO_TRAVERSER_ACTUAL_FRACTION_DEFAULT
    )
    hero_row_min_strategy_sum_mass: float = (
        HERO_ROW_MIN_STRATEGY_SUM_MASS_DEFAULT
    )
    deadline_ms: Optional[int] = None


def _finite_float_env(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    """Read one finite float from the environment, failing loud on typos."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be a finite number, got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    if value < minimum:
        raise ValueError(
            f"{name}={value:g} must be at least {minimum:g}"
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            f"{name}={value:g} outside [{minimum:g}, {maximum:g}]"
        )
    return value


# Per-player-count defaults. n = players in the street-root re-solve.
#   n<=3: vec-Linear-FINAL at 20k iterations. The scalar
#         sampled traverser at K=ALL/ISO row counts refused sparse rows and
#         left deep nodes untrained; the vec traverser updates every row each
#         iteration, so in-range rows never refuse. The 2p turn extends
#         through the river in-tree with one sampled river per iteration and
#         hero's earlier street decisions frozen (cpu_es_solver defaults), so
#         no leaf table is estimated there; the flop keeps depth-limited
#         leaves with exact per-pair all-in cells. 20k is the measured budget
#         that fits a 6 s decision window on 8 cores with the ISO grid.
#   n>=4: vec-Linear-FINAL at 5M iterations (the validated multiway
#         path); nr=10 for 5p/6p (under-the-bar floor on the nr ladder).
# With the vec-sampled leaf (NEXUS_LEAF_VEC_SAMPLED=1) the 2p flop
# and 3p flop/turn leaves are rolled out per visit, so a FIXED 20k-iteration
# solve is 16-41 s on a flop whose bucket memo is cold. The 2p/3p live config
# therefore carries a CFR wall budget (the C deadline, anytime readout) and
# the iteration count is a CEILING: a warm flop reaches ~20k in 3 s, a cold
# one returns ~100-500 iterations (still better than a frozen leaf table
# under the common referee), and the river-extended turn uses the whole
# budget (100k in 1.9 s).
# The budget applies to EVERY caller of live_config_for, so no caller can
# fall back to the unbounded fixed count; an explicit deadline_ms override
# still wins. On 8 cores, 3,000 ms gives 3.1-3.2 s of CFR and 3.4-3.5 s of
# end-to-end request time, inside a 6 s decision window.
LIVE_2P3P_ITERATIONS = 100_000
LIVE_2P3P_DEADLINE_MS = 3_000
DEFAULT_LIVE_CONFIG_BY_N: Dict[int, LiveSolveConfig] = {
    2: LiveSolveConfig(
        iterations=LIVE_2P3P_ITERATIONS, num_rollouts=3, vec_mode=True,
        readout="final", deadline_ms=LIVE_2P3P_DEADLINE_MS,
        note=(
            "2p live re-solve: vec-Linear-FINAL under the deadline; turn "
            "extends through the river in-tree (sampled river, hero line "
            "frozen), flop depth-limited with exact all-in leaves; no "
            "measured exploitability floor for this arm"
        ),
    ),
    3: LiveSolveConfig(
        iterations=LIVE_2P3P_ITERATIONS, num_rollouts=3, vec_mode=True,
        readout="final", deadline_ms=LIVE_2P3P_DEADLINE_MS,
        note=(
            "3p live re-solve: vec-Linear-FINAL under the deadline, "
            "depth-limited leaves with exact all-in cells; no measured "
            "exploitability floor for this arm"
        ),
    ),
    4: LiveSolveConfig(
        iterations=5_000_000, num_rollouts=3, vec_mode=True, readout="final",
        note=(
            "4p vec-Linear-FINAL nr=3 floor 0.067%/pot measured at the 50M rung "
            "(num_rollouts ladder, 4p); floors describe AGGREGATE "
            "exploitability, not per-spot mix reproducibility"
        ),
    ),
    5: LiveSolveConfig(
        iterations=5_000_000, num_rollouts=10, vec_mode=True, readout="final",
        note=(
            "5p vec-final nr=10 floor 0.076%/pot measured at the 50M rung "
            "(raise-cap measurement); floors describe AGGREGATE "
            "exploitability, not per-spot mix reproducibility"
        ),
    ),
    6: LiveSolveConfig(
        iterations=5_000_000, num_rollouts=10, vec_mode=True, readout="final",
        note=(
            "6p vec-final nr=10 floor 0.089%/pot measured (~65s cold at "
            "96 to 192 cores). CAVEAT: the final-iterate readout serves "
            "near-pure per-spot mixes, and the per-spot mix varies with "
            "leaf-construct noise (reach-weighted |dSigma| 0.345 across "
            "noise seeds at nr=10, reweighted ladder rerun); the "
            "floor describes aggregate exploitability, not per-spot "
            "reproducibility"
        ),
    ),
}


# ── PLAY vs DISPLAY readout split (Pluribus) ─────────────────────────────
# Pluribus PLAYS the FINAL-iteration strategy during search and reserves the
# weighted average (sigma-bar) for belief updates and analysis.
# DEFAULT_LIVE_CONFIG above is the DISPLAY contract. The 2p/3p rows are
# vec-Linear-FINAL as well (the vec traverser requires the final readout), so
# PLAY and DISPLAY coincide at every N. The split is kept because any
# consumer that ACTS on the mix (match play; an autonomous agent) must take
# this table instead: readout="final" at every N. A row that is already
# final is the same object as in DEFAULT; an "avg" row gets a final-readout
# copy. Floors are keyed on the FULL config, so such a copy cites no
# measured floor (measured_floor_for returns None on a readout mismatch).
PLAY_LIVE_CONFIG_BY_N: Dict[int, LiveSolveConfig] = {
    n: (
        cfg
        if cfg.readout == "final"
        else replace(
            cfg,
            readout="final",
            note=(
                f"{n}p PLAY config: final-iterate readout for "
                "acting consumers — Pluribus plays the final iterate; "
                "sigma-bar display floor notes do not apply"
            ),
        )
    )
    for n, cfg in DEFAULT_LIVE_CONFIG_BY_N.items()
}


def live_config_for(
    n_players: int,
    iterations_override: Optional[int] = None,
    num_rollouts_override: Optional[int] = None,
    deadline_ms_override: Optional[int] = None,
    *,
    purpose: str = "display",
) -> LiveSolveConfig:
    """Resolve the live-solve config for a player count + client overrides.

    purpose: "display" (the sigma-bar frequency contract, the default)
        or "play" (final-iterate readout, as Pluribus plays, for consumers
        that ACT on the returned mix).

    Raises:
        ValueError: unsupported player count, unknown purpose, or an
            override outside the measured bounds (rejected, never clamped
            silently).
    """
    if purpose not in ("display", "play"):
        raise ValueError(
            f"purpose must be 'display' or 'play', got {purpose!r}"
        )
    table = (
        PLAY_LIVE_CONFIG_BY_N if purpose == "play"
        else DEFAULT_LIVE_CONFIG_BY_N
    )
    base = table.get(n_players)
    if base is None:
        raise ValueError(
            f"unsupported player count for live re-solve: {n_players} "
            f"(supported: {sorted(table)})"
        )
    iterations = base.iterations
    num_rollouts = base.num_rollouts
    # Omitted client overrides must keep the default 2p/3p CFR budget.
    deadline_ms = base.deadline_ms
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
    if deadline_ms_override is not None:
        if not (DEADLINE_MS_MIN <= deadline_ms_override <= DEADLINE_MS_MAX):
            raise ValueError(
                f"deadline_ms override {deadline_ms_override} outside the "
                f"validated bounds [{DEADLINE_MS_MIN}, {DEADLINE_MS_MAX}]"
            )
        deadline_ms = deadline_ms_override
    actual_fraction = _finite_float_env(
        HERO_TRAVERSER_ACTUAL_FRACTION_ENV,
        HERO_TRAVERSER_ACTUAL_FRACTION_DEFAULT,
        minimum=0.0,
        maximum=1.0,
    )
    minimum_mass = _finite_float_env(
        HERO_ROW_MIN_STRATEGY_SUM_MASS_ENV,
        HERO_ROW_MIN_STRATEGY_SUM_MASS_DEFAULT,
        minimum=0.0,
    )
    return LiveSolveConfig(
        iterations=iterations,
        num_rollouts=num_rollouts,
        vec_mode=base.vec_mode,
        readout=base.readout,
        deadline_ms=deadline_ms,
        note=base.note,
        hero_traverser_actual_fraction=actual_fraction,
        hero_row_min_strategy_sum_mass=minimum_mass,
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
    the arm's measured rung. None on RIVER unconditionally: every measured arm is
    a flop/turn leaf-table configuration and no river floor has been
    measured (see RIVER_QUALITY_NOTE).

    The iteration gate is PER-ARM (FLOOR_MIN_ITERATIONS_BY_ARM), defaulting to the
    20M ladder minimum: the vec-final 5p/6p arms require 20M iters, but the 2p
    scalar/avg analysis-surface arm is measured at its own (few-hundred-iter)
    budget — citing the 20M global there would wrongly suppress a real floor."""
    if street == "river":
        return None
    arm = (n_players, num_rollouts, vec_mode, readout)
    min_iters = FLOOR_MIN_ITERATIONS_BY_ARM.get(arm, FLOOR_MIN_ITERATIONS)
    if iterations < min_iters:
        return None
    return MEASURED_FLOORS_PCT_POT.get(arm)
