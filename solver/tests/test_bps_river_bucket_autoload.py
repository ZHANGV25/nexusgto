"""Mocked RBK1 sidecar loading tests; no blueprint or native build required.

Run from solver/ with:
    pytest -q tests/test_bps_river_bucket_autoload.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from engine import bps_reader_py


RIVER_CLASSES = 123156254


@pytest.fixture
def autoload_case(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    module_path = repo_root / "python" / "engine" / "bps_reader_py.py"
    module_path.parent.mkdir(parents=True)
    monkeypatch.setattr(bps_reader_py, "__file__", str(module_path))
    monkeypatch.delenv("BPS_FORCE_BOOST_MIXER", raising=False)

    blueprint = tmp_path / "artifacts" / "live.bps"
    blueprint.parent.mkdir()
    filename = blueprint.name + ".river_buckets.rbk1"
    loader = Mock(return_value=RIVER_CLASSES)
    reader = object.__new__(bps_reader_py.BPSReaderC)
    # __del__ must never pass our fake handle to a native free function.
    reader._borrowed = True
    reader._reader = object()
    reader._lib = SimpleNamespace(bps_load_river_bucket_table=loader)
    reader._metadata = {}
    return SimpleNamespace(
        reader=reader,
        loader=loader,
        blueprint=blueprint,
        adjacent=blueprint.parent / filename,
        fallback=repo_root / filename,
    )


def test_old_library_is_silent_even_when_sidecars_exist(autoload_case, capsys):
    case = autoload_case
    case.adjacent.touch()
    case.fallback.touch()
    case.reader._lib = SimpleNamespace()

    case.reader._load_river_bucket_table_auto(str(case.blueprint))

    case.loader.assert_not_called()
    assert capsys.readouterr() == ("", "")


def test_missing_sidecars_are_silent(autoload_case, capsys):
    case = autoload_case

    case.reader._load_river_bucket_table_auto(str(case.blueprint))

    case.loader.assert_not_called()
    assert capsys.readouterr() == ("", "")


def test_valid_adjacent_sidecar_wins_and_logs_class_count(autoload_case, capsys):
    case = autoload_case
    case.adjacent.touch()
    case.fallback.touch()

    case.reader._load_river_bucket_table_auto(str(case.blueprint))

    case.loader.assert_called_once_with(
        case.reader._reader, str(case.adjacent).encode("utf-8")
    )
    output = capsys.readouterr().out
    assert str(case.adjacent) in output
    assert str(case.fallback) not in output
    assert "loaded" in output.lower()
    assert "123156254 classes" in output


@pytest.mark.parametrize("adjacent_present", [False, True])
def test_missing_or_refused_adjacent_uses_repo_fallback(
    autoload_case, adjacent_present, capsys
):
    case = autoload_case
    case.fallback.touch()
    if adjacent_present:
        case.adjacent.touch()
        case.loader.side_effect = [-1, RIVER_CLASSES]

    case.reader._load_river_bucket_table_auto(str(case.blueprint))

    expected = []
    if adjacent_present:
        expected.append(call(case.reader._reader, str(case.adjacent).encode("utf-8")))
    expected.append(call(case.reader._reader, str(case.fallback).encode("utf-8")))
    assert case.loader.call_args_list == expected
    output = capsys.readouterr().out
    assert str(case.fallback) in output
    assert "123156254 classes" in output
    if adjacent_present:
        assert "refused" in output.lower()


@pytest.mark.parametrize("outcome", [-1, 0, OSError("unreadable"), ValueError("bad path")])
def test_failed_load_is_harmless_and_logged(autoload_case, outcome, capsys):
    case = autoload_case
    case.adjacent.touch()
    if isinstance(outcome, Exception):
        case.loader.side_effect = outcome
    else:
        case.loader.return_value = outcome

    case.reader._load_river_bucket_table_auto(str(case.blueprint))

    case.loader.assert_called_once_with(
        case.reader._reader, str(case.adjacent).encode("utf-8")
    )
    output = capsys.readouterr().out
    assert str(case.adjacent) in output
    assert "skipped" in output.lower()
    assert "live river bucketing" in output.lower()
    assert "123156254 classes" not in output


@pytest.mark.parametrize("exception", [OSError("unreadable"), ValueError("bad path")])
def test_load_exception_still_allows_validated_fallback(
    autoload_case, exception, capsys
):
    case = autoload_case
    case.adjacent.touch()
    case.fallback.touch()
    case.loader.side_effect = [exception, RIVER_CLASSES]

    case.reader._load_river_bucket_table_auto(str(case.blueprint))

    assert case.loader.call_args_list == [
        call(case.reader._reader, str(case.adjacent).encode("utf-8")),
        call(case.reader._reader, str(case.fallback).encode("utf-8")),
    ]
    output = capsys.readouterr().out
    assert str(exception) in output
    assert str(case.fallback) in output
    assert "123156254 classes" in output


def test_stat_error_still_allows_validated_fallback(autoload_case, monkeypatch, capsys):
    case = autoload_case
    case.fallback.touch()
    original_is_file = Path.is_file

    def is_file(path):
        if path == case.adjacent:
            raise OSError("cannot inspect adjacent sidecar")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", is_file)

    case.reader._load_river_bucket_table_auto(str(case.blueprint))

    case.loader.assert_called_once_with(
        case.reader._reader, str(case.fallback).encode("utf-8")
    )
    output = capsys.readouterr().out
    assert "cannot inspect adjacent sidecar" in output
    assert "123156254 classes" in output


def test_river_table_load_follows_texture_turn_and_ochs(autoload_case, monkeypatch):
    case = autoload_case
    (case.blueprint.parent / "texture_cache.bin").touch()
    case.adjacent.touch()
    events = []

    def load_texture(*args):
        events.append("texture")
        return 1

    def load_river(*args):
        events.append("river")
        return RIVER_CLASSES

    case.reader._lib.bps_load_texture_cache = Mock(side_effect=load_texture)
    case.loader.side_effect = load_river
    monkeypatch.setattr(
        case.reader, "_load_turn_centroids_auto", lambda path: events.append("turn")
    )
    monkeypatch.setattr(
        case.reader, "_init_ochs_river_auto", lambda: events.append("ochs")
    )

    case.reader._load_texture_cache_auto(str(case.blueprint))

    assert events == ["texture", "turn", "ochs", "river"]
