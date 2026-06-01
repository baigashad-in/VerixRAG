"""Tests for guardrails components."""

import pytest
from src.guardrails.pii_detector import PIIDetector
from src.guardrails.safety_filter import SafetyFilter
from src.guardrails.query_classifier import KeywordClassifier


class TestPIIDetector:
    
    @pytest.fixture
    def detector(self):
        return PIIDetector()
    
    def test_detects_email(self, detector):
        matches = detector.detect("Contact me at john@example.com please")
        assert any(m.pii_type == "email" for m in matches)

    def test_detects_ssn(self, detector):
        matches = detector.detect("My SSN is 123-45-6789")
        assert any(m.pii_type == "ssn" for m in matches)

    def test_detects_phone(self, detector):
        matches = detector.detect("Call me at (555) 123-4567")
        assert any(m.pii_type == "phone_us" for m in matches)

    def test_redacts_correctly(self, detector):
        text = "Email me at john@test.com about the refund"
        redacted, matches = detector.redact(text)
        assert "john@test.com" not in redacted
        assert "[EMAIL_REDACTED]" in redacted
        assert len(matches) == 1
    
    def test_no_false_positives(self, detector):
        text = "What is the refund policy for orders over 30 days?"
        matches = detector.detect(text)
        assert len(matches) == 1

    def test_no_false_positives(self, detector):
        text = "What is the refund policy for orders over 30 days?"
        matches = detector.detect(text)
        assert len(matches) == 0

    def test_multiple_pii(self, detector):
        text = "My email is a@b.com and SSN is 123-45-6789"
        matches = detector.detect(text)
        assert len(matches) == 2
    
    def test_has_pii_shortcut(self, detector):
        assert detector.has_pii("email: test@test.com")
        assert not detector.has_pii("What is the refund policy?")

class TestSafetyFilter:

    @pytest.fixture
    def safety(self):
        return SafetyFilter()
    
    def test_safe_response_passes(self, safety):
        result = safety.check("Returns are accepted within 30 days.")
        assert result.is_safe
        assert len(result.flags) == 0

    def test_medical_advice_blocked(self, safety):
        result = safety.check("You should stop taking your medication immediately.")
        assert not result.is_safe
        assert "medical_advice" in result.flags

    def test_financial_advice_blocked(self, safety):
        result = safety.check("You should buy this stock for guaranteed returns.")
        assert not result.is_safe
        assert "financial_advice" in result.flags

    def test_filtered_response_provided(self, safety):
        result = safety.check("You should stop taking your medication.")
        assert "healthcare professional" in result.filtered_response

class TestKeywordClassifier:

    @pytest.fixture
    def classifier(self):
        return KeywordClassifier()
    
    def test_in_scope_query(self, classifier):
        result = classifier.classify("What is the refund policy?")
        assert result["in_scope"] is True

    def test_out_of_scope_query(self, classifier):
        result = classifier.classify("Tell me a joke about cats")
        assert result["in_scope"] is False

    def test_uncertain_query(self, classifier):
        result = classifier.classify("How does this work exactly?")
        assert result["confidence"] == "low"
