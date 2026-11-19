"""README.md and docs/*.md use only numbers from results/numbers.yaml (tools/check_doc_numbers.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_doc_numbers", REPO / "tools" / "check_doc_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_checker()


def _tokens(text: str) -> list:
    return [token for _, token, _ in C.numbers_in(text)]


def test_docs_use_only_published_numbers():
    assert (REPO / "README.md").is_file()
    assert C.check(C.doc_paths()) == []


def test_unknown_number_fails(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("The search won 5.73 bb/100, or 43.21 on a rerun.\n", encoding="utf-8")
    assert C.check([doc]) == ["doc.md:1: 43.21"]


def test_skipped_spans():
    text = ("`x >> 30` and\n```\n123.45\n```\n<!-- 77 -->\n[a](https://e.org/987) "
            "doi:10.1126/x 0xFEED BPS3 u64 x86-64 v1.0.0 8.05B\n")
    assert _tokens(text) == []


def test_number_forms():
    text = "14,250 deals, −6.05 to 0.30, 2.36% of 38,451, bb/100, a 3-second budget.\n"
    assert _tokens(text) == ["14,250", "6.05", "0.30", "2.36", "38,451", "100", "3"]


def test_signs_ignored_and_small_integers_allowed():
    known = C.published_values(C._load_rows())
    assert 2.88 in known  # published as -2.88
    assert C._known(20.0, set()) and not C._known(21.0, set())


def test_every_allowed_number_has_a_reason():
    values = [v for v, _ in C.ALLOWED]
    assert len(values) == len(set(values))
    assert all(reason.strip() for _, reason in C.ALLOWED)
