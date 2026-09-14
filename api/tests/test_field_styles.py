"""Pure tests for run_match opponent-style fields (no blueprint)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import field_styles as fs
from scripts import run_match
from scripts import run_match_cli as cli


def _state(facing_bet, n_legal):
    return SimpleNamespace(facing_bet=facing_bet, num_legal_actions=lambda: n_legal)


class _Source:
    def __init__(self, freqs):
        self.freqs = freqs

    def get_mix(self, state, hero_cards):
        return SimpleNamespace(freqs=dict(self.freqs), is_uniform_fallback=False)


def test_station_calls_more_and_raises_less_facing_a_bet():
    out = fs.perturb(np.array([0.5, 0.3, 0.2]), True, fs.STYLES["station"])
    np.testing.assert_allclose(out, [0.15, 0.3 + 0.35 + 0.1, 0.1])


def test_maniac_moves_call_mass_to_raises_in_proportion():
    out = fs.perturb(np.array([0.2, 0.4, 0.3, 0.1]), True, fs.STYLES["maniac"])
    fold = 0.2 * 0.7
    call = (0.4 + 0.06) * 0.5
    moved = (0.4 + 0.06) * 0.5
    np.testing.assert_allclose(out, [fold, call, 0.3 + moved * 0.75, 0.1 + moved * 0.25])


def test_maniac_spreads_over_raises_when_none_had_mass():
    out = fs.perturb(np.array([1.0, 0.0, 0.0]), False, fs.STYLES["maniac"])
    np.testing.assert_allclose(out, [0.5, 0.25, 0.25])


def test_nit_folds_facing_bets_and_checks_instead_of_betting():
    facing = fs.perturb(np.array([0.2, 0.5, 0.3]), True, fs.STYLES["nit"])
    np.testing.assert_allclose(facing, [0.6, 0.25, 0.15])
    unopened = fs.perturb(np.array([0.4, 0.6]), False, fs.STYLES["nit"])
    np.testing.assert_allclose(unopened, [0.7, 0.3])


@pytest.mark.parametrize("name", sorted(fs.STYLES))
@pytest.mark.parametrize("facing,n", [(True, 2), (True, 4), (False, 1), (False, 3)])
def test_every_style_returns_a_distribution(name, facing, n):
    rng = np.random.default_rng(3)
    out = fs.perturb(rng.dirichlet(np.ones(n)), facing, fs.STYLES[name])
    assert out.shape == (n,) and np.all(out >= 0.0)
    assert out.sum() == pytest.approx(1.0)


def test_style_source_returns_full_sampling_law():
    source = fs.StyleSource(_Source({0: 0.5, 1: 0.5}), "station")
    mix = source.get_mix(_state(True, 3), (0, 1))
    assert set(mix.freqs) == {0, 1, 2}
    assert run_match.sampling_probs_for(SimpleNamespace(mix=mix), 3) == pytest.approx((0.15, 0.85, 0.0))


def test_style_source_rejects_unknown_style():
    with pytest.raises(ValueError):
        fs.StyleSource(_Source({0: 1.0}), "whale")


def test_parse_args_field_style_requires_ab(tmp_path):
    args = cli.parse_args(
        run_match, ["--mode", "ab", "--out", str(tmp_path), "--field-style", "nit"]
    )
    assert args.field_style == "nit"
    with pytest.raises(SystemExit):
        cli.parse_args(run_match, ["--mode", "aa", "--out", str(tmp_path), "--field-style", "nit"])
    with pytest.raises(SystemExit):
        cli.parse_args(run_match, ["--mode", "ab", "--out", str(tmp_path), "--field-style", "whale"])
