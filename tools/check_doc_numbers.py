#!/usr/bin/env python3
"""Check that README.md and docs/*.md use only numbers from results/numbers.yaml.

Every number in the prose of those files must be a value in numbers.yaml (an
estimate, its SE, a 95% bound, a deal count, or a number quoted in a row's
`fact`), an integer from 0 to 20, or a number in ALLOWED below, which lists the
numbers that are not results and why each is allowed. Signs are ignored, so
"-2.88" in the table and "cost 2.88" in prose are the same number.

Skipped: fenced code blocks, inline code, HTML comments, link targets, URLs,
doi: identifiers, hexadecimal literals, and numbers joined to letters (BPS3,
u64, x86-64, v1.0.0, 8.05B).

    python tools/check_doc_numbers.py         # exit 1 and a list if any number fails
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent
NUMBERS = REPO / "results" / "numbers.yaml"
SMALL_INT_MAX = 20

# Numbers that are not results: (value, reason).
ALLOWED: Tuple[Tuple[float, str], ...] = (
    (0.7, "preflop raise size in the bet menu, pot fraction (blueprint manifest)"),
    (0.75, "postflop first-bet size in the bet menu, pot fraction (blueprint manifest)"),
    (8.05, "blueprint iterations in billions (blueprint manifest, data/training/)"),
    (12.2, "size of the 8.05B blueprint file in GB"),
    (49.3, "size of the results release asset in MB (results/README.md)"),
    (95, "the level of the 95% intervals"),
    (100, "stack depth in big blinds, and the 100 in bb/100"),
    (75, "bb/100 lost by folding every hand heads-up in the blinds: (0.5 + 1) / 2 bb per hand"),
    (21, "bytes in a blueprint record before its strategy bytes (docs/bps-format.md)"),
    (50, "small blind in chips (blueprint manifest)"),
    (10000, "starting stack in chips (blueprint manifest)"),
    (168, "index of the last preflop hand class (docs/bps-format.md)"),
    (169, "number of preflop hand classes"),
    (200, "number of postflop buckets per street"),
    (255, "largest quantized strategy byte (docs/bps-format.md)"),
    (2598960, "number of five-card hands, 52 choose 5"),
    (2017, "year of a cited paper"),
    (2018, "year of a cited paper"),
    (2019, "year of a cited paper"),
    (13.3, "gcc version"),
    (24.04, "Ubuntu version"),
    (3.9, "lowest supported Python version"),
    (3.12, "tested Python version"),
)

# Row keys whose values are not published numbers.
SKIP_KEYS = {"id", "data", "sha256", "seeds", "arm", "field", "baseline", "compute", "stat",
             "metric", "unit", "reason", "note", "reproducible"}

NUMBER = re.compile(r"(?<![\w.])(?<!\w-)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!\.?\w)")
STRIP = (
    re.compile(r"```.*?```", re.S),           # fenced code blocks
    re.compile(r"<!--.*?-->", re.S),          # HTML comments
    re.compile(r"`[^`]*`"),                   # inline code, which may wrap a line
    re.compile(r"\]\([^)]*\)"),               # link targets
    re.compile(r"https?://\S+"),              # bare URLs
    re.compile(r"doi:\S+"),                   # DOI identifiers
    re.compile(r"\b0x[0-9A-Fa-f]+\b"),        # hexadecimal literals
)


def _load_rows() -> list:
    spec = importlib.util.spec_from_file_location("reproduce", REPO / "results" / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_numbers(NUMBERS)


def numbers_in(text: str) -> Iterable[Tuple[int, str, float]]:
    """(line, token, value) for every number in `text` outside the skipped spans."""
    for pattern in STRIP:
        # Keep newlines so line numbers stay right.
        text = pattern.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in NUMBER.finditer(line):
            token = m.group(0)
            yield lineno, token, float(token.replace(",", ""))


def published_values(rows: list) -> Set[float]:
    """Every number numbers.yaml publishes, as absolute values."""
    values: Set[float] = set()
    for row in rows:
        for key, value in row.items():
            if key in SKIP_KEYS:
                continue
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, bool) or item is None:
                    continue
                if isinstance(item, (int, float)):
                    values.add(abs(float(item)))
                elif isinstance(item, str):
                    values.update(abs(v) for _, _, v in numbers_in(item))
    return values


def _known(value: float, known: Set[float]) -> bool:
    if value.is_integer() and value <= SMALL_INT_MAX:
        return True
    return any(abs(value - k) < 1e-9 for k in known)


def doc_paths() -> List[Path]:
    return [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.name


def check(paths: Iterable[Path]) -> List[str]:
    """One "path:line: token" string per number that is not allowed."""
    known = published_values(_load_rows()) | {v for v, _ in ALLOWED}
    failures: List[str] = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, token, value in numbers_in(text):
            if not _known(value, known):
                failures.append(f"{_display(path)}:{lineno}: {token}")
    return failures


def main(argv: List[str]) -> int:
    paths = [Path(a).resolve() for a in argv] or doc_paths()
    failures = check(paths)
    for line in failures:
        print(line)
    if failures:
        print(f"{len(failures)} number(s) not in results/numbers.yaml or the allow-list")
        return 1
    print("doc numbers: all in results/numbers.yaml or the allow-list")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
