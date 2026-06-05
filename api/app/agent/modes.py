"""Agent mode enum + per-mode config (plan §2 / §"three MODES").

The mode selects the strategy SOURCE only. All three share the game_state,
key_builder, action_translation and sampling loop.

  LOOKUP         — floor, sub-second. Blueprint sigma-bar lookup + sample.
                   MUST NOT call the GPU solver. v1 PRIMARY.
  LIVE_RESOLVE   — ceiling, 1-33s, NOT sub-second. Wraps the existing
  OFFLINE_LIFTED — sub-second, aims at the ceiling. Serves a precomputed
                   cache built by running LIVE_RESOLVE offline. v1 scaffolds
                   the source + cache format only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # py3.8+: Literal is in typing
    from typing import Literal

    AgentMode = Literal["LOOKUP", "LIVE_RESOLVE", "OFFLINE_LIFTED"]
except ImportError:  # pragma: no cover - all supported runtimes have Literal
    AgentMode = str  # type: ignore[misc,assignment]


# The valid mode strings, for runtime validation (Literal isn't enforced
# at runtime). Kept in sync with AgentMode above.
VALID_MODES = ("LOOKUP", "LIVE_RESOLVE", "OFFLINE_LIFTED")


@dataclass(frozen=True)
class ModeConfig:
    """Immutable per-agent configuration.

    Attributes:
        mode: which strategy source to use (see module docstring).
        seed: RNG seed for deterministic action sampling. Required for
            reproducible play / tests.
        blueprint_path: path to the unified .bps blueprint (LOOKUP /
            OFFLINE_LIFTED source). None for LIVE_RESOLVE-only use.
        cache_path: OFFLINE_LIFTED lifted-strategy cache location.
        uniform_on_miss: when the blueprint node is absent, fall back to
            uniform-over-legal-actions (Pluribus Algorithm 1 substitution)
            instead of raising. Default True (play must not crash).
    """

    mode: str = "LOOKUP"
    seed: int = 0
    blueprint_path: Optional[str] = None
    cache_path: Optional[str] = None
    uniform_on_miss: bool = True

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Unknown agent mode {self.mode!r}; expected one of {VALID_MODES}"
            )
