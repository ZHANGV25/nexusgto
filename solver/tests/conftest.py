"""Make the script-style check() tests fail under pytest.

Several files here were written as scripts: a module-level check() helper
records each failed condition (in ``_FAIL``, ``FAILED`` or a ``FAIL`` counter)
and ``__main__`` turns the tally into the exit code. Under pytest nothing
asserted, so a test whose checks failed still passed. This fixture fails the
test when its checks recorded a new failure.
"""

from __future__ import annotations

from typing import Dict

import pytest

_TALLY_NAMES = ("_FAIL", "FAILED", "FAIL")


def _tally(module: object) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name in _TALLY_NAMES:
        value = getattr(module, name, None)
        if isinstance(value, list):
            out[name] = len(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
    return out


@pytest.fixture(autouse=True)
def _checks_fail_the_test(request: pytest.FixtureRequest):
    module = request.module
    before = _tally(module)
    yield
    after = _tally(module)
    grew = {k: v - before.get(k, 0) for k, v in after.items() if v > before.get(k, 0)}
    if grew:
        pytest.fail(f"check() recorded failures: {grew}", pytrace=False)
