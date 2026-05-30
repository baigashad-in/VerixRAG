"""
Detect and redact Personally Identifiable Information.

WHY THIS MATTERS:
1. User types: "My SSN is 123-45-6789, can I get a refund?"
   That SSN gets embedded, stored in your vector DB, and
   potentially sent to an external LLM API. That's a
   compliance violation.

2. Your LLM response might accidentally include PII from
   retrieved chunks: "John Smith at john@company.com filed
   a refund on..." - now you've leaked someone's info.

PII detection runs on BOTH input and output.
"""

import re
from dataclasses import dataclass

@dataclass
class PIIMatch:
    """A detected PII instance."""
    pii_type: str # "email", "phone", "ssn", etc.
    value: str    # the matched text
    start: int    # position in string
    end: int      # position in string

class PIIDetector:
    """Regex-based PII detection and redaction.
    
    WHY REGEX AND NOT AN LLM?
    - Speed: regex runs in microseconds, LLM takes seconds.
    - Reliability: regex either matches or doesn't - no hallucination
    - Cost: zero
    - Privacy: the text never leaves the machine

    LIMITATIONS:
    - Won't catch names (too many false positives with regex)
    - Won't catch addresses without zip codes.
    - Won't understand context ("my birthday" vs "the company's founding date")

    For production, you'd add a Named Entity Recognition (NER) model
    like spaCy's en_core_web_sm for names and addresses. We keep it
    regex-only for simplicity and zero dependencies.
    """

    # Patterns ordered from most specific to least
    PATTERNS = {
        "ssn": {
            "regex": r'\b\d{3}-\d{2}-\d{4}\b',
            "description": "Social Security Number",
        },
        "credit_card": {
            "regex": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "description": "Credit card number",
        },
        "email": {
            "regex": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b',
            "description": "Email address",
        },
        "phone_us": {
            "regex": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "description": "US phone number",
        },
        "ip_address": {
            "regex": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "description": "IP address",
        },
        "date_of_birth": {
            "regex": r'\b(?:DOB|born|birthday|birth date)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
            "description": "Date of birth",
        },
    }

    def __init__(self, additional_patterns: dict = None):
        self.patterns = {**self.PATTERNS}
        if additional_patterns:
            self.patterns.update(additional_patterns)

        # Pre-compile all patterns
        self._compiled = {
            name: re.compile(config["regex"], re.IGNORECASE)
            for name, config in self.patterns.items()
        }

    def detect(self, text: str) -> list[PIIMatch]:
        """Find all PII in the text."""
        matches = []

        for pii_type, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                ))

        # Sort by position
        matches.sort(key = lambda m: m.start)
        return matches
    
    def redact(self, text: str) -> tuple[str, list[PIIMatch]]:
        """Replace PII with redaction markers.
        
        "My email is john@test.com"
        -> "My email is [EMAIL_REDACTED]"

        Returns the redacted text AND the list of detections,
        so you can log what was removed without logging the
        actual values.
        """
        matches = self.detect(text)

        if not matches:
            return text, []
        
        # Replace from end to start so positions stay valid
        redacted = text
        for match in reversed(matches):
            placeholder = f"[{match.pii_type.upper()}_REDACTED]"
            redacted = (redacted[:match.start] +
                        placeholder + 
                        redacted[match.end:])
            
        return redacted, matches
    
    def has_pii(self, text: str) -> bool:
        """Quick check - does this text contain any PII?"""
        return len(self.detect(text)) > 0
    