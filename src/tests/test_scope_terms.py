"""Tests for keyword scope seeding."""

import json
import sys

from src.guardrails.query_classifier import KeywordClassifier, keyword_classifier_from_settings
from src.guardrails.scope_terms import GENERIC, alias_terms, build_scope_terms, stem
from src.retrieval.scoping import load_domain_config
from src.tests.conftest import FIXTURE_KB
from src.config import reset_settings
from scripts import scope_eval


ALIASES = load_domain_config(f"{FIXTURE_KB}/_domains.json").aliases


def test_stem_folds_plurals_only():
    assert stem("refunds") == "refund" and stem("policies") == "policy"
    assert stem("express") == "express" and stem("bus") == "bus"


def test_terms_come_from_documents_and_skip_generic_words():
    terms = build_scope_terms(FIXTURE_KB, ALIASES, top_n=0)
    assert {"refund", "shipping", "password", "clouddeck", "nexora"} <= terms
    assert not terms & GENERIC
    assert not {"monday", "click", "policy", "guide"} & terms


def test_alias_terms_are_distinctive_names():
    assert {"clouddeck", "nexora", "verixstore"} <= alias_terms(ALIASES)


def test_min_matches_and_strong_terms():
    clf = KeywordClassifier(in_scope_terms=["refund", "shipping", "password", "reset"],
                            strong_terms=["clouddeck"], min_matches=2)
    assert clf.classify("Can I get a refund on shipping costs?")["in_scope"] is True
    assert clf.classify("What's the DHL shipping time?")["in_scope"] is None      # one term: ask the LLM
    assert clf.classify("Does CloudDeck work offline?")["in_scope"] is True       # product name alone
    assert clf.classify("tell me a joke about refunds and shipping")["in_scope"] is False  # patterns first


def test_classifier_from_settings_respects_switch(monkeypatch):
    monkeypatch.setenv("DOMAINS_CONFIG", f"{FIXTURE_KB}/_domains.json")
    reset_settings()
    seeded = keyword_classifier_from_settings(FIXTURE_KB)
    assert seeded.in_scope_terms and seeded.min_matches == 2 and "clouddeck" in seeded.strong_terms
    monkeypatch.setenv("KEYWORD_SCOPE_ENABLED", "false")
    reset_settings()
    assert not keyword_classifier_from_settings(FIXTURE_KB).in_scope_terms
    assert not keyword_classifier_from_settings("/no/such/dir").in_scope_terms


def test_scope_eval_script(tmp_path, monkeypatch):
    dataset = tmp_path / "eval.json"
    dataset.write_text(json.dumps({"questions": [
        {"question": "Can I get a refund after 30 days?", "category": "easy"},
        {"question": "How long is the password reset link valid?", "category": "easy"},
        {"question": "What is the weather in Paris?", "category": "out_of_scope"},
    ]}))
    monkeypatch.setattr(sys, "argv", [
        "scope_eval", "--dataset", str(dataset), "--documents", FIXTURE_KB,
        "--domains", f"{FIXTURE_KB}/_domains.json", "--output-dir", str(tmp_path)])
    scope_eval.main()
    report = json.loads(next(tmp_path.glob("scope_eval_*.json")).read_text())
    assert set(report["variants"]) == {"body0_min1", "body10_min1", "body0_min2", "body10_min2"}
    for v in report["variants"].values():
        assert 0 <= v["coverage"] <= 1 and 0 <= v["leakage"] <= 1