# Top-level build. `make` builds the native libraries; `make check` builds them,
# runs the C tests, then runs the Python tests that need only files in git
# (the fixture blueprint stands in for the full one).
#
# Python: install the package first, e.g. `python -m pip install -e ".[test]"`.

PYTHON ?= python3

.PHONY: all check test-c test-python clean

all:
	$(MAKE) -C solver all

test-c: all
	$(MAKE) -C solver test

test-python: all
	$(PYTHON) -m pytest -q tests solver/tests api/tests

check: test-c test-python

clean:
	$(MAKE) -C solver clean
