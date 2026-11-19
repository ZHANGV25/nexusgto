"""C-side bias-as-hero-action binding tests — drives the REAL rollout .so.

Bias-as-hero-action mode makes the querying player's 4 stored per-bias EVs
at an armed cont-EV boundary a CFR-selectable action: the boundary
regret-matches over them and returns the mixture value instead of their
uniform mean. This pytest exercises the ctypes bindings in
python/engine/rollout_c.py (value convergence itself is not asserted here):
  - set_cont_ev_bias_action / get_cont_ev_bias_action round-trip + default OFF
  - get_cont_ev_bias_action_stats + reset_cont_ev_bias_regrets lifecycle
  - fail-loud wrapper shape (RuntimeError only when the symbol is absent)
  - orthogonality: toggling bias-action does NOT touch the flop/turn enable
    flags or the cont_ev table pointer.

REQUIRES build/rollout.so (or rollout.dll on Windows) — built by
`make rollout`. When the library is absent the whole module SKIPS.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

from engine import rollout_c  # noqa: E402  (import is .so-free; load is lazy)

# Module-level skip when the shared library is absent (mirrors
# test_rollout_vocab_ctypes.py). A RuntimeError from the struct-parity assert
# is a REAL failure and must propagate, not skip.
try:
    _LIB = rollout_c._lib()
except (FileNotFoundError, OSError) as _e:  # pragma: no cover
    pytest.skip(
        f"rollout shared library not available ({_e}); build it with "
        f"'make rollout' to "
        f"produce build/rollout.so",
        allow_module_level=True,
    )

# The bias-action symbols only exist in a .so built with bias-action support.
# If the loaded library lacks them, skip the whole module (an older .so is a
# legitimate build to run against; the fail-loud wrapper is asserted directly in
# test_wrapper_is_fail_loud_when_symbol_absent by simulation below).
if not hasattr(_LIB, "rollout_set_cont_ev_bias_action"):  # pragma: no cover
    pytest.skip(
        "rollout.so lacks bias-action support (no rollout_set_cont_ev_bias_action) — "
        "rebuild it with 'make rollout'",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _reset_bias_action_state():
    """Leave the process-global bias-action mode OFF and the regret store
    cleared between tests so ordering can't poison a later assertion."""
    yield
    rollout_c.set_cont_ev_bias_action(False)
    rollout_c.reset_cont_ev_bias_regrets()


def test_default_off_and_roundtrip():
    """Default mode is OFF; set(True)/set(False) round-trips through get."""
    rollout_c.set_cont_ev_bias_action(False)
    assert rollout_c.get_cont_ev_bias_action() is False

    rollout_c.set_cont_ev_bias_action(True)
    assert rollout_c.get_cont_ev_bias_action() is True

    rollout_c.set_cont_ev_bias_action(False)
    assert rollout_c.get_cont_ev_bias_action() is False


def test_reset_zeros_select_stat():
    """The select stat starts at 0 and reset re-zeros it. With no boundary
    driven here, the stat stays 0 — this asserts the lifecycle wiring, not
    convergence (that is the C test's job)."""
    rollout_c.reset_cont_ev_bias_regrets()
    assert rollout_c.get_cont_ev_bias_action_stats() == 0
    # Re-reset is idempotent and keeps it at 0.
    rollout_c.reset_cont_ev_bias_regrets()
    assert rollout_c.get_cont_ev_bias_action_stats() == 0


def test_bias_action_orthogonal_to_boundary_flags():
    """Toggling bias-action must not touch the flop/turn enable flags or the
    cont_ev table pointer — it only changes HOW an armed boundary combines its
    4 stored EVs."""
    # Establish a known boundary-flag + table state.
    rollout_c.set_use_cont_ev_at_flop_boundary(False)
    rollout_c.set_use_cont_ev_at_turn_boundary(False)
    rollout_c.set_rollout_cont_ev_table(None)

    rollout_c.set_cont_ev_bias_action(True)
    assert rollout_c.get_use_cont_ev_at_flop_boundary() is False
    assert rollout_c.get_use_cont_ev_at_turn_boundary() is False
    assert rollout_c.get_rollout_cont_ev_table() is None

    rollout_c.set_cont_ev_bias_action(False)
    assert rollout_c.get_use_cont_ev_at_flop_boundary() is False
    assert rollout_c.get_use_cont_ev_at_turn_boundary() is False


def test_wrapper_fail_loud_when_symbol_absent(monkeypatch):
    """set_cont_ev_bias_action must RAISE (never silently no-op) when the .so
    lacks the symbol — the caller asked for bias-action and must not fall back
    to legacy uniform-mean unnoticed. Simulate an old .so by hiding the symbol
    from hasattr via a stand-in lib object."""

    class _NoSymLib:
        pass

    monkeypatch.setattr(rollout_c, "_lib", lambda: _NoSymLib())
    with pytest.raises(RuntimeError, match="rollout_set_cont_ev_bias_action"):
        rollout_c.set_cont_ev_bias_action(True)
    # The read-side wrappers degrade gracefully (return False/0) on an old .so.
    assert rollout_c.get_cont_ev_bias_action() is False
    assert rollout_c.get_cont_ev_bias_action_stats() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
