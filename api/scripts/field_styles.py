"""Opponent-style fields for run_match: blueprint seats with systematic leaks.

A strategy that is harder to exploit should win more chips than the blueprint
against opponents who make recognizable mistakes, which a blueprint field never
does. Each style moves probability mass between fold, check/call and raises of
the wrapped source's mix. Preflop and postflop states share the action order
[fold (only when facing a bet)][check/call][raises ascending], so one structural
rule applies on every street. The perturbed mix is what the seat samples from,
so AIVAT prices it exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import numpy as np

from app.engine.agent.strategy_source import ActionMix


@dataclass(frozen=True)
class Style:
    """Shares of mass moved between action groups (each in [0, 1])."""

    fold_to_call: float = 0.0       # facing a bet: fold mass moved to call
    raise_to_call: float = 0.0      # raise mass moved to check/call
    call_to_raise: float = 0.0      # check/call mass moved to raises
    continue_to_fold: float = 0.0   # facing a bet: call and raise mass moved to fold
    raise_to_check: float = 0.0     # not facing a bet: raise mass moved to check

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


STYLES: Mapping[str, Style] = {
    "station": Style(fold_to_call=0.7, raise_to_call=0.5),
    "maniac": Style(call_to_raise=0.5, fold_to_call=0.3),
    "nit": Style(continue_to_fold=0.5, raise_to_check=0.5),
}


def _move(probs: np.ndarray, src: np.ndarray, dst: np.ndarray, share: float) -> np.ndarray:
    """Move ``share`` of the mass on ``src`` to ``dst``, keeping dst's proportions."""
    moved = share * probs[src].sum()
    if moved <= 0.0 or len(dst) == 0:
        return probs
    dst_mass = probs[dst]
    weights = dst_mass / dst_mass.sum() if dst_mass.sum() > 0.0 else np.full(len(dst), 1.0 / len(dst))
    out = probs.copy()
    out[src] = probs[src] * (1.0 - share)
    out[dst] = out[dst] + moved * weights
    return out


def perturb(probs: np.ndarray, facing_bet: bool, style: Style) -> np.ndarray:
    """Apply ``style`` to one decision's action distribution."""
    base = np.asarray(probs, dtype=np.float64)
    out = base / base.sum()
    n_nonraise = 2 if facing_bet else 1
    if len(out) < n_nonraise:
        return out
    fold = np.array([0]) if facing_bet else np.array([], dtype=int)
    call = np.array([n_nonraise - 1])
    raises = np.arange(n_nonraise, len(out))
    if facing_bet:
        out = _move(out, fold, call, style.fold_to_call)
    if len(raises):
        out = _move(out, raises, call, style.raise_to_call)
        out = _move(out, call, raises, style.call_to_raise)
    if facing_bet:
        out = _move(out, np.concatenate([call, raises]), fold, style.continue_to_fold)
    elif len(raises):
        out = _move(out, raises, call, style.raise_to_check)
    return out / out.sum()


class StyleSource:
    """Wrap a strategy source so its seat plays one style's leaks."""

    def __init__(self, inner: Any, style_name: str) -> None:
        if style_name not in STYLES:
            raise ValueError(f"unknown field style {style_name!r}; choose from {sorted(STYLES)}")
        self._inner = inner
        self.name = style_name
        self.style = STYLES[style_name]

    def get_mix(self, state: Any, hero_cards: Any) -> ActionMix:
        mix = self._inner.get_mix(state, hero_cards)
        n_legal = int(state.num_legal_actions())
        probs = np.zeros(n_legal, dtype=np.float64)
        for idx, freq in mix.freqs.items():
            probs[int(idx)] += float(freq)
        styled = perturb(probs, bool(state.facing_bet), self.style)
        return ActionMix(
            {idx: float(styled[idx]) for idx in range(n_legal)},
            is_uniform_fallback=bool(getattr(mix, "is_uniform_fallback", False)),
        )
