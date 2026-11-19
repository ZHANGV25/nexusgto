"""Agent package.

A game loop around a pluggable strategy SOURCE. The three modes (LOOKUP /
LIVE_RESOLVE / OFFLINE_LIFTED) differ ONLY in the strategy source; everything
else (game_state, key_builder, action_translation, the sampling loop) is
shared.

LOOKUP mode is a sub-second blueprint sigma-bar lookup + action sampling.
This is Pluribus's blueprint FLOOR (strong, bucketed), NOT its superhuman
live-search ceiling; LIVE_RESOLVE adds the per-decision search.

Public surface:
    AgentMode, ModeConfig                          (modes)
    GameState, Seat, advance, ...                  (game_state)
    NodeKey, build_key                             (key_builder)
    build_preflop_prefix                           (preflop_constructor)
    translate_action                               (action_translation)
    StrategySource, LookupSource, ...              (strategy_source)
    PokerAgent, SampledAction                      (agent)
"""

from __future__ import annotations

from app.engine.agent.action_translation import TranslatedAction, translate_action
from app.engine.agent.agent import PokerAgent, SampledAction
from app.engine.agent.game_state import GameState, new_postflop_state
from app.engine.agent.key_builder import NodeKey, build_key
from app.engine.agent.modes import AgentMode, ModeConfig
from app.engine.agent.preflop_constructor import (
    PreflopAction,
    PreflopConstructionUnsupported,
    build_preflop_prefix,
)
from app.engine.agent.strategy_source import (
    ActionMix,
    LiveResolveSource,
    LookupSource,
    OfflineLiftedSource,
    StrategySource,
)

__all__ = [
    "AgentMode",
    "ModeConfig",
    "GameState",
    "new_postflop_state",
    "NodeKey",
    "build_key",
    "build_preflop_prefix",
    "PreflopAction",
    "PreflopConstructionUnsupported",
    "translate_action",
    "TranslatedAction",
    "StrategySource",
    "LookupSource",
    "OfflineLiftedSource",
    "LiveResolveSource",
    "ActionMix",
    "PokerAgent",
    "SampledAction",
]
