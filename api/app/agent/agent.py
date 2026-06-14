"""PokerAgent — the play loop (plan §2, §3.5).

PokerAgent holds a StrategySource + a ModeConfig and turns a GameState into a
sampled action. It is the thin loop the plan describes: get the mix for the
node, sample one action from it with a seeded RNG. All three modes share this
loop; only the source differs.

For LOOKUP this is the sub-second blueprint floor: one σ̄ lookup + one sample,
no GPU. (Honest framing — plan §0: this is Pluribus's blueprint floor, an
equilibrium-frequency ESTIMATE, not its superhuman live-search ceiling.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.agent.game_state import GameState
from app.agent.modes import ModeConfig
from app.agent.strategy_source import ActionMix, HeroCards, StrategySource


@dataclass(frozen=True)
class SampledAction:
    """The agent's chosen action at a decision node.

    action_idx: index into the node's legal action set (feed to
        GameState.advance).
    mix: the full ActionMix the action was sampled from (for logging /
        honesty — exposes is_uniform_fallback when the blueprint missed).
    seat: the full seat (0-5) that acted.
    """

    action_idx: int
    mix: ActionMix
    seat: int


class PokerAgent:
    """A blueprint-floor playing agent.

    Args:
        source: the per-decision strategy source (LookupSource for v1).
        config: mode + seed. The seed drives a single RNG stream so a hand is
            reproducible; each act() advances the stream.
    """

    def __init__(self, source: StrategySource, config: ModeConfig) -> None:
        self._source = source
        self._config = config
        self._rng = np.random.default_rng(config.seed)

    @property
    def mode(self) -> str:
        return self._config.mode

    def act(self, state: GameState, hero_cards: HeroCards) -> SampledAction:
        """Choose an action for the player to act in `state`.

        Raises:
            ValueError: if `state` is terminal (no decision to make).
        """
        if state.terminal:
            raise ValueError("act() called on a terminal GameState")
        seat = state.to_act_seat()
        assert seat is not None  # guaranteed by the terminal check above
        mix = self._source.get_mix(state, hero_cards)
        action_idx = mix.sample(self._rng)
        return SampledAction(action_idx=action_idx, mix=mix, seat=seat)
