"""Quality gate: baseline comparison for BEIR and retrieval_eval reports."""

import json

import pytest

from scripts import quality_gate as qg

BEIR = {"queries": 300, "configs": {"hybrid": {"ndcg@10": 0.71, "mrr@10": 0.68}}}
OWN = {"label_quality": {"questions": 290},
       "configs": {"full": {"metrics": {"mrr": 0.946, "hit@1": 0.90}}}}


def test_lookup_handles_both_report_shapes():
    assert qg.lookup(BEIR, "hybrid.ndcg@10") == 0.71
    assert qg.lookup(OWN, "full.mrr") == 0.946
    assert qg.lookup(OWN, "label_quality.questions") == 290
    assert qg.lookup(OWN, "full.missing") is None and qg.lookup(OWN, "nope.x") is None


def test_check_passes_within_tolerance_and_fails_below():
    base = {"tolerance": 0.01, "metrics": {"full.mrr": 0.95, "full.hit@1": 0.92}}
    rows, problems = qg.check(OWN, base)
    assert problems == []
    assert [r["ok"] for r in rows] == [True, False]             # 0.946 >= 0.94; 0.90 < 0.91


def test_missing_metric_and_mismatch_fail():
    base = {"match": {"queries": 100}, "metrics": {"hybrid.recall@10": 0.8}}
    rows, problems = qg.check(BEIR, base)
    assert rows[0]["ok"] is False and rows[0]["delta"] is None
    assert problems and "queries" in problems[0]


def _files(tmp_path, report, baseline, name="beir_scifact_x.json"):
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "scifact.json").write_text(json.dumps(baseline))
    (tmp_path / name).write_text(json.dumps(report))
    return ["--suite", "scifact", "--reports-dir", str(tmp_path), "--baselines-dir", str(tmp_path / "base")]


def test_main_exit_codes_and_update(tmp_path):
    base = {"tolerance": 0.005, "match": {"queries": 300}, "metrics": {"hybrid.ndcg@10": 0.72}}
    args = _files(tmp_path, BEIR, base)
    assert qg.main(args) == 1                                     # 0.71 < 0.715
    assert qg.main(args + ["--update"]) == 0
    saved = json.loads((tmp_path / "base" / "scifact.json").read_text())
    assert saved["metrics"] == {"hybrid.ndcg@10": 0.71} and saved["tolerance"] == 0.005
    assert qg.main(args) == 0


def test_update_refused_when_not_comparable(tmp_path):
    base = {"match": {"queries": 50}, "metrics": {"hybrid.ndcg@10": 0.5}}
    assert qg.main(_files(tmp_path, BEIR, base) + ["--update"]) == 2


def test_no_report_is_an_error(tmp_path):
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "own.json").write_text(json.dumps({"metrics": {}}))
    with pytest.raises(FileNotFoundError):
        qg.main(["--suite", "own", "--reports-dir", str(tmp_path), "--baselines-dir", str(tmp_path / "base")])


def test_committed_baselines_are_valid():
    for suite in qg.SUITES:
        base = json.loads(open(f"config/baselines/{suite}.json").read())
        assert base["metrics"] and all(isinstance(v, float) for v in base["metrics"].values())
        assert 0 < base["tolerance"] <= 0.02
