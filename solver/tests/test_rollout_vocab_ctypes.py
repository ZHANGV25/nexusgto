"""C-side runtime bet-vocab tests — drives the REAL rollout .so via ctypes.

Companion to the pure-python encoder tests (test_bet_vocab_contract.py /
test_per_leaf_flop_history.py): those cover only the Python encoder; this
file covers the C consumer's runtime-vocab behavior
(src/search/rollout.c: rollout_set_train_vocab / rollout_get_train_vocab /
rollout_classify_postflop_actions_c / rollout_action_amount_chips_c)
through the shipping ctypes bindings (python/engine/rollout_c.py).

REQUIRES build/rollout.so (or rollout.dll on Windows) — built by
`make rollout`. NOTE: the link gate (scripts/verify_cpu_links.sh) does NOT
produce it — it links into build_verify/ and deletes it. The EC2 harness
must run `make rollout` before this file. When the library is absent the
whole module SKIPS with a clear reason (local Windows dev box, zero-build
rule).

ORDERING CONSTRAINT: rollout_set_train_vocab is process-global and
set-once-per-value; the byte-identity-defaults assertion must observe the
state BEFORE any set in this process. It therefore lives at the TOP of the
first test function (pytest runs file order by default), and this file must
run in its own pytest invocation (or before any test that calls
set_rollout_train_vocab). The EC2 harness's per-file invocation satisfies
this. A module-teardown restores the compiled defaults so a combined
invocation's later byte-identity tests are not poisoned.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

from engine import rollout_c  # noqa: E402  (import is .so-free; load is lazy)

# Module-level skip when the shared library is absent. Only FileNotFoundError
# (rollout_c._find_lib: no build/rollout.{so,dll}) and OSError (CDLL load
# failure, e.g. missing runtime deps) skip; a RuntimeError from the struct-
# parity assert is a REAL failure and must propagate, not skip.
try:
    rollout_c._lib()
except (FileNotFoundError, OSError) as _e:
    pytest.skip(
        f"rollout shared library not available ({_e}); build it with "
        f"'make rollout' — the link gate (verify_cpu_links.sh) does NOT "
        f"produce build/rollout.so",
        allow_module_level=True,
    )


# The compiled byte-identity defaults (rollout.h ROLLOUT_FIRST_RAISE_DEFAULT_N
# / ROLLOUT_SUBSEQUENT_DEFAULT_N; rollout.c g_rollout_*_fracs initializers).
_DEFAULT_FIRST = [0.5, 1.0]
_DEFAULT_SUBSEQ = [1.0]
# The served 8.05B manifest vocab (config_hash bc8ac105).
_SERVED_FIRST = [0.75]
_SERVED_SUBSEQ = [1.0]


@pytest.fixture(scope="module", autouse=True)
def _restore_defaults_after_module():
    """Teardown-only: put the process-global vocab back to the compiled
    defaults so later tests in a combined invocation see byte-identity
    state. Setup does NOTHING (the defaults-before-set assertion below must
    observe the untouched state)."""
    yield
    lib = rollout_c._lib()
    if hasattr(lib, "rollout_set_train_vocab"):
        rollout_c.set_rollout_train_vocab(_DEFAULT_FIRST, _DEFAULT_SUBSEQ)


def _allin_gate() -> int:
    """Derive the explicit-allin gate (BPS_ROLLOUT_ADD_EXPLICIT_ALLIN; read
    lazily by the C side and cached per process) from an observed count:
    facing-first count = 2 + n_first + gate. Returns 0 or 1."""
    n_first = len(rollout_c.get_rollout_train_vocab()[0])
    n = len(rollout_c.classify_postflop_actions_c(True, 0))
    gate = n - 2 - n_first
    assert gate in (0, 1), (
        f"facing-first action count {n} inconsistent with n_first={n_first} "
        f"(gate must be 0 or 1, got {gate})"
    )
    return gate


def test_defaults_then_set_roundtrip():
    """(a) BEFORE any set: the C runtime vocab is the compiled byte-identity
    defaults [0.5,1.0]/[1.0]. (b) set([0.75],[1.0]) round-trips through get.
    Sequenced in ONE function because (a) is unobservable after any set."""
    # (a) — must be the first vocab interaction in this process.
    first, subseq = rollout_c.get_rollout_train_vocab()
    assert first == _DEFAULT_FIRST, (
        f"pre-set vocab is not the compiled default: {first} "
        f"(byte-identity default must be {_DEFAULT_FIRST})"
    )
    assert subseq == _DEFAULT_SUBSEQ

    # (b) — set the served manifest vocab, read it back.
    rollout_c.set_rollout_train_vocab(_SERVED_FIRST, _SERVED_SUBSEQ)
    first, subseq = rollout_c.get_rollout_train_vocab()
    assert first == _SERVED_FIRST   # 0.75/1.0 are exact in float32
    assert subseq == _SERVED_SUBSEQ


def test_classify_counts_served_vocab():
    """(c) After set([0.75],[1.0]), the C classify emits the right action
    COUNTS: facing-first 3+gate ([fold,call,raise0.75(,allin)]),
    facing-subsequent 3+gate ([fold,call,raise1.0(,allin)]),
    non-facing 2+gate ([check,bet0.75(,allin)]); tree-closed counts are
    gate- and vocab-independent."""
    rollout_c.set_rollout_train_vocab(_SERVED_FIRST, _SERVED_SUBSEQ)
    gate = _allin_gate()

    # Categories: 0=FOLD, 1=CALL(covers check), 2=RAISE (rollout.h CAT_*).
    facing_first = rollout_c.classify_postflop_actions_c(True, 0)
    assert facing_first == [0, 1, 2] + [2] * gate

    facing_subseq = rollout_c.classify_postflop_actions_c(True, 1)
    assert facing_subseq == [0, 1, 2] + [2] * gate

    non_facing = rollout_c.classify_postflop_actions_c(False, 0)
    assert non_facing == [1, 2] + [2] * gate

    # Tree closed (raises_so_far >= 3): [fold, call] / [check].
    assert rollout_c.classify_postflop_actions_c(True, 3) == [0, 1]
    assert rollout_c.classify_postflop_actions_c(False, 3) == [1]


def test_action_amounts_served_vocab():
    """(c) After set([0.75],[1.0]), the C amount translation uses the
    RUNTIME fractions. Documented formulas (rollout.c
    rollout_action_amount_chips_c):
      facing:     raise_to = to_call + int(frac * (pot + to_call))
      non-facing: bet      = int(frac * pot)
    The sized slots sit at idx 2 (facing) / idx 1 (non-facing) regardless of
    the allin gate (allin is appended LAST)."""
    rollout_c.set_rollout_train_vocab(_SERVED_FIRST, _SERVED_SUBSEQ)
    pot, to_call, stack = 200, 100, 10000

    # Facing, first raise, sized slot idx 2 → frac 0.75 (first list).
    kind, amt = rollout_c.action_amount_chips_c(2, True, 0, pot, to_call, stack)
    assert (kind, amt) == ("raise", to_call + int(0.75 * (pot + to_call)))  # 325

    # Facing, subsequent raise, sized slot idx 2 → frac 1.0 (subseq list).
    kind, amt = rollout_c.action_amount_chips_c(2, True, 1, pot, to_call, stack)
    assert (kind, amt) == ("raise", to_call + int(1.0 * (pot + to_call)))  # 400

    # Non-facing, sized slot idx 1 → frac 0.75 (first list).
    kind, amt = rollout_c.action_amount_chips_c(1, False, 0, pot, 0, stack)
    assert (kind, amt) == ("bet", int(0.75 * pot))  # 150

    # Fixed slots are vocab-independent: fold / call.
    assert rollout_c.action_amount_chips_c(0, True, 0, pot, to_call, stack) \
        == ("fold", 0)
    assert rollout_c.action_amount_chips_c(1, True, 0, pot, to_call, stack) \
        == ("call", to_call)
    assert rollout_c.action_amount_chips_c(0, False, 0, pot, 0, stack) \
        == ("check", 0)


def test_classify_and_amounts_legacy_vocab():
    """Cross-check: setting the legacy default [0.5,1.0]/[1.0] back changes
    the counts (facing-first grows a slot) and the sized amounts (idx 2
    facing-first becomes the 0.5 raise). Proves the vocab is live runtime
    state, not compiled in."""
    rollout_c.set_rollout_train_vocab(_DEFAULT_FIRST, _DEFAULT_SUBSEQ)
    gate = _allin_gate()
    pot, to_call, stack = 200, 100, 10000

    assert rollout_c.classify_postflop_actions_c(True, 0) \
        == [0, 1, 2, 2] + [2] * gate            # fold,call,0.5,1.0(,allin)
    assert rollout_c.classify_postflop_actions_c(True, 1) \
        == [0, 1, 2] + [2] * gate               # fold,call,1.0(,allin)
    assert rollout_c.classify_postflop_actions_c(False, 0) \
        == [1, 2, 2] + [2] * gate               # check,0.5,1.0(,allin)

    # Facing-first sized slots: idx 2 → 0.5, idx 3 → 1.0.
    assert rollout_c.action_amount_chips_c(2, True, 0, pot, to_call, stack) \
        == ("raise", to_call + int(0.5 * (pot + to_call)))   # 250
    assert rollout_c.action_amount_chips_c(3, True, 0, pot, to_call, stack) \
        == ("raise", to_call + int(1.0 * (pot + to_call)))   # 400
    # Non-facing sized slots: idx 1 → 0.5, idx 2 → 1.0.
    assert rollout_c.action_amount_chips_c(1, False, 0, pot, 0, stack) \
        == ("bet", int(0.5 * pot))                           # 100
    assert rollout_c.action_amount_chips_c(2, False, 0, pot, 0, stack) \
        == ("bet", int(1.0 * pot))                           # 200


def test_setter_rejects_seven_sizes():
    """(d) The node-capacity bound: 7 sizes pass the Python-side list-length
    check (<= ROLLOUT_MAX_RAISE_SIZES = 8) but must be REJECTED by the C
    setter's n > ROLLOUT_MAX_ACTIONS_PER_NODE-3 (= 6) bound — a facing node
    would need 7+3 = 10 > 9 action slots. The failed set must not modify the
    configured vocab."""
    rollout_c.set_rollout_train_vocab(_SERVED_FIRST, _SERVED_SUBSEQ)
    seven = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
    assert len(seven) == 7

    with pytest.raises(RuntimeError, match="rejected"):
        rollout_c.set_rollout_train_vocab(seven, _SERVED_SUBSEQ)
    msg, _, _ = rollout_c._last_error()
    assert "too many raise sizes" in msg, (
        f"expected the distinct node-capacity error message, got: {msg!r}"
    )

    with pytest.raises(RuntimeError, match="rejected"):
        rollout_c.set_rollout_train_vocab(_SERVED_FIRST, seven)
    msg, _, _ = rollout_c._last_error()
    assert "too many raise sizes" in msg

    # Rejection is all-or-nothing: the configured vocab is unchanged
    # (the C setter validates BEFORE writing the frac arrays).
    first, subseq = rollout_c.get_rollout_train_vocab()
    assert first == _SERVED_FIRST
    assert subseq == _SERVED_SUBSEQ


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
