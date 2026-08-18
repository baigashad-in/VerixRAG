"""Tests for the retrieval metrics (no LLM calls)."""

import json
import math
import shutil
import sys

import pytest

from src.evaluation.retrieval_metrics import (
    Label, QuestionResult, aggregate, chunk_sections, grade, parse_labels, question_metrics,
    section_matches,
)
from src.tests.conftest import FIXTURE_KB

KS = (1, 3, 5, 8)


# ── labels and heading matching ──

def test_parse_labels_single_and_cross_document():
    assert parse_labels(["Password Reset"], "account_guide.md") == [
        Label("account_guide.md", "Password Reset")]
    assert parse_labels(["refund_policy.md: Late Returns", "shipping_policy.md"], "cross_document") == [
        Label("refund_policy.md", "Late Returns"), Label("shipping_policy.md", "")]
    assert parse_labels(["Late Returns"], "cross_document") == [Label(None, "Late Returns")]
    assert parse_labels(["", None], "a.md") == []


@pytest.mark.parametrize("label, path, expected", [
    ("Password Reset", "Account Management Guide › Password Reset", True),
    ("password reset", "Account Management Guide › Password Reset", True),
    ("2. Password Reset", "Account Management Guide › Password Reset", True),   # numbering
    ("Section 3: Late Returns", "Refund Policy › Late Returns", True),
    ("Resetting your Password", "Account Management Guide › Password Reset", False),
    ("Account Deletion", "Account Management Guide › Password Reset", False),
    ("Account Management Guide", "Account Management Guide › Password Reset", True),  # H1 = whole doc
    ("Express Shipping Costs", "Shipping Policy › Express Shipping", True),          # containment
    ("", "Anything › At All", True),                                               # whole document
])
def test_section_matches(label, path, expected):
    assert section_matches(label, path) is expected


def test_grade_levels():
    labels = [Label("account_guide.md", "Password Reset")]
    src = {"account_guide.md"}
    assert grade("account_guide.md", "Guide › Password Reset", labels, src) == (2, {0})
    assert grade("account_guide.md", "Guide › Account Deletion", labels, src) == (1, set())
    assert grade("refund_policy.md", "Refund Policy › Password Reset", labels, src) == (0, set())


def test_chunk_covers_headings_inside_its_text():
    # A chunk starting at the title holds two "##" sections
    content = "# Shipping Policy\n\n## Express Shipping\n$15.\n\n## Lost or Delayed Packages\nCall us."
    covered = chunk_sections("Shipping Policy", content)
    labels = [Label("shipping_policy.md", "Lost or Delayed Packages")]
    assert grade("shipping_policy.md", "Shipping Policy", labels, {"shipping_policy.md"})[0] == 1
    assert grade("shipping_policy.md", covered, labels, {"shipping_policy.md"}) == (2, {0})
    assert chunk_sections("Guide › Reset", None) == ["Guide › Reset"]


# ── metric maths on hand-built rankings ──

def _result(grades, satisfied=None, matchable=1, ideal=None):
    return QuestionResult(
        question="q", category="c", sources=["a.md"], labels=["a.md: s"],
        matchable_labels=matchable, grades=grades,
        satisfied=satisfied or [({0} if g == 2 else set()) for g in grades],
        ideal=ideal if ideal is not None else sorted(grades, reverse=True))


def test_perfect_ranking():
    m = question_metrics(_result([2, 1, 0, 0, 0, 0, 0, 0]), KS)
    assert m["hit@1"] == m["recall@1"] == m["mrr"] == m["ndcg@8"] == 1.0
    assert m["precision@1"] == 1.0 and m["precision@5"] == 0.2


def test_relevant_at_rank_three():
    m = question_metrics(_result([0, 1, 2, 0, 0, 0, 0, 0], ideal=[2, 1, 0, 0, 0, 0, 0, 0]), KS)
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["mrr"] == pytest.approx(1 / 3)
    assert m["doc_hit@1"] == 0.0 and m["doc_hit@3"] == 1.0
    dcg = (2 ** 1 - 1) / math.log2(3) + (2 ** 2 - 1) / math.log2(4)
    idcg = (2 ** 2 - 1) / math.log2(2) + (2 ** 1 - 1) / math.log2(3)
    assert m["ndcg@3"] == pytest.approx(dcg / idcg)


def test_multi_section_recall_needs_every_section():
    # Two labelled sections; only label 0 is retrieved in the top 5, label 1 at rank 7
    sat = [{0}, set(), set(), set(), set(), set(), {1}, set()]
    grades = [2, 0, 0, 0, 0, 0, 2, 0]
    m = question_metrics(_result(grades, satisfied=sat, matchable=2), KS)
    assert m["recall@5"] == 0.5 and m["recall@8"] == 1.0


def test_miss_and_unlabelled_question():
    miss = question_metrics(_result([1, 0, 0, 0, 0, 0, 0, 0], ideal=[2, 1]), KS)
    assert miss["hit@8"] == 0.0 and miss["mrr"] == 0.0 and miss["doc_hit@1"] == 1.0
    unlabelled = question_metrics(_result([1, 0], matchable=0), KS)
    assert "hit@1" not in unlabelled and "mrr" not in unlabelled       # excluded, not failed
    assert unlabelled["doc_hit@1"] == 1.0


def test_aggregate_skips_missing_keys():
    agg = aggregate([{"mrr": 1.0, "doc_hit@1": 1.0}, {"doc_hit@1": 0.0}])
    assert agg["mrr"] == 1.0 and agg["n_mrr"] == 1
    assert agg["doc_hit@1"] == 0.5 and agg["n_doc_hit@1"] == 2


# ── end to end on the fixture knowledge base (pgvector + fake models) ──

@pytest.mark.db
def test_retrieval_eval_end_to_end(clean_db, fake_models, tmp_path, monkeypatch):
    from src.config import reset_settings
    from src.ingestion.ingest import IngestionPipeline

    kb = tmp_path / "documents"
    shutil.copytree(FIXTURE_KB, kb)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", clean_db)
    monkeypatch.setenv("DOMAINS_CONFIG", str(kb / "_domains.json"))
    reset_settings()
    IngestionPipeline(db_connection=clean_db, documents_root=str(kb)).ingest_directory(str(kb))

    dataset = tmp_path / "eval.json"
    dataset.write_text(json.dumps({"questions": [
        {"question": "How long is the password reset link valid?", "category": "factual",
         "source_file": "account_guide.md", "sections": ["Password Reset"]},
        {"question": "How much does express shipping cost?", "category": "factual",
         "source_file": "shipping_policy.md", "sections": ["Express Shipping"]},
        {"question": "Late refunds and lost packages?", "category": "hard",
         "source_file": "cross_document",
         "sections": ["refund_policy.md: Late Returns", "shipping_policy.md: Lost or Delayed Packages"]},
        {"question": "Tell me about the moon", "category": "out_of_scope",
         "source_file": "none", "sections": []},
        {"question": "Where is the onboarding checklist?", "category": "factual",
         "source_file": "hr_handbook.md", "sections": ["Onboarding Checklist"]},   # no such heading
    ]}))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["retrieval_eval", "--dataset", str(dataset),
                                      "--label", "t", "--output-dir", str(out)])
    from scripts import retrieval_eval
    retrieval_eval.main()

    report = json.loads(next(out.glob("retrieval_eval_t_*.json")).read_text())
    assert set(report["configs"]) == {"dense", "bm25", "hybrid", "rerank", "full"}
    lq = report["label_quality"]
    assert lq["questions"] == 4                       # out_of_scope excluded
    assert lq["section_labelled"] == 3                # bad label reported, not scored
    assert "Onboarding Checklist" in json.dumps(lq["unmatched_label_examples"])
    for data in report["configs"].values():
        m = data["metrics"]
        assert m["n_mrr"] == 3 and m["n_doc_hit@8"] == 4
        for key in ("hit@5", "recall@8", "precision@5", "mrr", "ndcg@8"):
            assert 0.0 <= m[key] <= 1.0