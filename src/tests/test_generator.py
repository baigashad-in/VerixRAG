"""Tests for the generation and hallucination detection components."""

import pytest
from src.generation.hallucination_check import HallucinationDetector

class TestCitationCoverage:

    @pytest.fixture
    def detector(self):
        return HallucinationDetector()
    
    def test_fully_cited_response(self, detector):
        answer = "Returns are accepted within 30 days [Source 1]. After that, store credit is offered [Source 2]."
        result = detector.check_citation_coverage(answer, num_sources=2)
        assert result["citation_coverage"] >= 0.9
        assert not result["is_suspicious"]

    def test_uncited_response(self, detector):
        answer = "Returns are accepted within 30 days. Store credit is offered after that. Free shipping is available on all orders."
        result = detector.check_citation_coverage(answer, num_sources=2)
        assert result["is_suspicious"]
        assert len(result["uncited_claims"]) > 0

    def test_short_factual_claims_are_checked(self, detector):
        result = detector.check_citation_coverage("Express shipping costs $15.", num_sources=1)
        assert result["uncited_claims"] == ["Express shipping costs $15."]
        result = detector.check_citation_coverage("Refunds take 30 days [Source 1].", num_sources=1)
        assert result["cited_claims"] == 1 and not result["uncited_claims"]

    def test_short_filler_is_ignored(self, detector):
        result = detector.check_citation_coverage(
            "Hope this helps! Express shipping costs $15 [Source 1].", num_sources=1)
        assert not result["uncited_claims"] and result["skipped_transitional"] == 1

    def test_old_rule_still_available_for_comparison(self, detector):
        answer = "Returns are accepted within 30 days."
        old = detector.check_citation_coverage(answer, 1, min_claim_words=8, short_facts=False)
        assert old["uncited_claims"] == [] and old["citation_coverage"] == 1.0

    def test_disclaimer_not_flagged(self, detector):
        answer = "I don't have enough information in the available documents to answer this question."
        result = detector.check_citation_coverage(answer, num_sources=0)
        assert not result["is_suspicious"]

    def test_empty_answer(self, detector):
        answer = "I don't have enough information in the available documents to answer this question."
        result = detector.check_citation_coverage(answer, num_sources=0)
        assert not result["is_suspicious"]

    def test_empty_answer(self, detector):
        result = detector.check_citation_coverage("", num_sources=0)
        assert result["citation_coverage"] == 1.0

