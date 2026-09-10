"""Importing a script must not run it: pytest imports every test_*.py it finds."""

import importlib
import sys

import pytest


def test_importing_test_queries_sends_no_requests(monkeypatch, tmp_path):
    import requests

    def no_network(*args, **kwargs):
        raise AssertionError("scripts/test_queries.py sent a request on import")
    monkeypatch.setattr(requests, "post", no_network)
    monkeypatch.chdir(tmp_path)                       # it would write experiments/ here
    sys.modules.pop("scripts.test_queries", None)
    module = importlib.import_module("scripts.test_queries")
    assert callable(module.main) and len(module.queries) == 24
    assert not (tmp_path / "experiments").exists()


def test_pytest_only_collects_src_tests():
    from pathlib import Path
    ini = Path(__file__).resolve().parents[2] / "pytest.ini"
    assert "testpaths = src/tests" in ini.read_text()
