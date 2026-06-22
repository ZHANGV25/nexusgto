"""Per-query leaf-lookup coverage metric.

B2 (2026-04-14) surfaced that our equity-only leaf fallback is degenerate,
which makes coverage of the blueprint-backed leaf path (BPR3 per-action
EVs + rollout σ̄ lookups) the load-bearing quality signal. This module
is the measurement hook: every blueprint lookup inside the leaf-value
path increments a hit or miss counter, and the enclosing query can
read the totals to log a "coverage: H/(H+M) = X%" line.

Scope:
- One counter per thread. A query running on one thread's GPUPostflopProvider
  will see its own lookups only. Threaded workers each see their own view.
- The counter is activated explicitly via `CoverageScope` (context manager)
  so test harnesses and one-off scripts don't accidentally pay the
  bookkeeping cost.
- Hits and misses are aggregated by `category` so we can tell "missed
  BPR3 action-EV at flop root" apart from "missed σ̄ at a turn rollout
  interior node." Category strings are free-form; callers pick them.

Typical shape:

    from engine.leaf_coverage import CoverageScope, record

    with CoverageScope() as cov:
        result = provider.solve(query)
    hit_rate = cov.hit_rate()  # 0..1
    logger.info("leaf coverage: %d hit / %d miss = %.1f%%",
                cov.hits, cov.misses, 100 * hit_rate)

Inside the lookup code:

    from engine.leaf_coverage import record
    ...
    if strat is None:
        record("sigma_lookup", hit=False, detail=f"street={street}")
        return None
    record("sigma_lookup", hit=True)
"""

from __future__ import annotations

import threading
from collections import Counter
from contextlib import contextmanager
from typing import Dict, Generator, Optional


_thread_local = threading.local()


class CoverageScope:
    """Context manager that records leaf-lookup hits and misses.

    Nested scopes are supported: inner scopes get their own counts, but
    they ALSO count toward the outer scope (for end-of-query aggregation).

    Thread safety: each thread has its own active scope stack; cross-thread
    counts are not merged. Callers solving in parallel should aggregate
    explicitly.
    """

    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0
        self.hit_categories: Counter = Counter()
        self.miss_categories: Counter = Counter()
        self.miss_details: Counter = Counter()

    def __enter__(self) -> "CoverageScope":
        stack = getattr(_thread_local, "scopes", None)
        if stack is None:
            stack = []
            _thread_local.scopes = stack
        stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        stack = getattr(_thread_local, "scopes", None)
        if stack and stack[-1] is self:
            stack.pop()

    def _record(self, category: str, hit: bool, detail: Optional[str] = None) -> None:
        if hit:
            self.hits += 1
            self.hit_categories[category] += 1
        else:
            self.misses += 1
            self.miss_categories[category] += 1
            if detail:
                self.miss_details[f"{category}:{detail}"] += 1

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 1.0  # vacuous-truth: no lookups means no misses
        return self.hits / total

    def summary(self) -> Dict[str, object]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": self.hits + self.misses,
            "hit_rate": self.hit_rate(),
            "hit_categories": dict(self.hit_categories),
            "miss_categories": dict(self.miss_categories),
            "miss_details": dict(self.miss_details.most_common(8)),
        }


def record(category: str, hit: bool, detail: Optional[str] = None) -> None:
    """Record a lookup outcome. No-op if no CoverageScope is active.

    Call sites:
        - leaf_values.compute_flop_leaf_values_bpr3: root per-player EV lookup
        - rollout.py / rollout_c.py: per-decision-node σ̄ lookup during the
          rollout walk (used for C1 leaf values).
    """
    stack = getattr(_thread_local, "scopes", None)
    if not stack:
        return
    # Walk the stack: an inner scope is definitely counting, but outer
    # scopes should also see the event (so a per-query scope wrapping
    # a per-leaf scope can still report totals).
    for scope in stack:
        scope._record(category, hit, detail)


@contextmanager
def disabled() -> Generator[None, None, None]:
    """Temporarily disable recording without popping the active scope.

    Useful for instrumentation-of-instrumentation test paths. The active
    scopes are saved and restored around the `with` block.
    """
    saved = getattr(_thread_local, "scopes", None)
    _thread_local.scopes = []
    try:
        yield
    finally:
        _thread_local.scopes = saved if saved is not None else []
