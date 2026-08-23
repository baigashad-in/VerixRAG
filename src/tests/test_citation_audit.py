"""Tests for the citation audit script."""

import json
import sys

from src.generation.hallucination_check import has_fact_signal


def test_fact_signal():
    for text in ("Costs $15.", "Within 30 days.", "Email support@verixstore.com.",
                 "Rate is 5%.", "See https://example.com."):
        assert has_fact_signal(text), text
    for text in ("Hope this helps!", "Yes, it does.", "Store credit is offered."):
        assert not has_fact_signal(text), text


def test_audit_compares_old_and_new_rules(tmp_path, monkeypatch, capsys):
    report = tmp_path / "adv.json"
    report.write_text(json.dumps({"results": [
        {"question": "q1", "answer": "Express shipping costs $15. Delivery takes 2 days."},
        {"question": "q2", "answer": "Refunds take 30 days [Source 1]."},
        {"question": "q3", "answer": "Refunds take 30 days [Source 1]."},          # duplicate
    ]}))
    monkeypatch.setattr(sys, "argv", ["citation_audit", str(report)])
    from scripts import citation_audit
    citation_audit.main()
    out = capsys.readouterr().out
    assert "2 unique saved answers" in out
    assert "Sentences newly counted as uncited claims: 2" in out
    assert "Express shipping costs $15." in out