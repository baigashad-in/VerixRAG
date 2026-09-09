"""trace_report --since: validated and compared as times, not strings."""

import json
import sys

import pytest

from scripts import trace_report


def write_traces(tmp_path):
    path = tmp_path / "traces.jsonl"
    rows = [{"trace_id": "old", "started_at": "2026-11-12T20:48:36.900000+00:00"},
            {"trace_id": "new", "started_at": "2026-11-12T20:48:37.600000+00:00"},
            {"trace_id": "bad", "started_at": "not a time"}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


@pytest.mark.parametrize("since", ["2026-11-12T20:48:37", "2026-11-12T20:48:37Z",
                                   "2026-11-12T20:48:37.557532+00:00",
                                   "2026-11-12T22:48:37+02:00"])
def test_since_keeps_only_newer_traces_in_any_iso_form(tmp_path, since):
    traces = trace_report.load([write_traces(tmp_path)], trace_report.since_arg(since))
    assert [t["trace_id"] for t in traces] == ["new"]


def test_without_since_everything_is_loaded(tmp_path):
    assert len(trace_report.load([write_traces(tmp_path)])) == 3


@pytest.mark.parametrize("value", ["20-48-37", "PASTE_THE_TIME_HERE", ""])
def test_invalid_since_is_rejected(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["trace_report", "--since", value])
    with pytest.raises(SystemExit):
        trace_report.main()
