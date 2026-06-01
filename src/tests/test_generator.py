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

