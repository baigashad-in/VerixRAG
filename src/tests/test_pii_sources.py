"""Output PII redaction must not remove contact details quoted from the documents."""

from src.guardrails.pii_detector import PIIDetector


def test_published_contact_details_are_kept():
    pii = PIIDetector()
    sources = ["Contact support@verixstore.com or call 1-800-555-0199 during business hours."]
    allowed = pii.public_values(sources)
    answer = "Email support@verixstore.com or call (800) 555-0199 [Source 1]."
    clean, redacted = pii.redact(answer, allowed)
    assert clean == answer and redacted == []


def test_user_pii_still_redacted_alongside_published_values():
    pii = PIIDetector()
    allowed = pii.public_values(["Call 1-800-555-0199."])
    clean, redacted = pii.redact("Call 1-800-555-0199. Your SSN 123-45-6789 is on file.", allowed)
    assert "1-800-555-0199" in clean and "[SSN_REDACTED]" in clean
    assert [m.pii_type for m in redacted] == ["ssn"]


def test_without_allow_list_behaviour_is_unchanged():
    clean, redacted = PIIDetector().redact("call 1-800-555-0199")
    assert "[PHONE_US_REDACTED]" in clean and len(redacted) == 1
