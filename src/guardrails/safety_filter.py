"""
Response safety filtering.

Checks the LLM's generated response before returning it
to the user. Catches cases where the LLM produces harmful,
inappropriate, or policy-violating content despite the
system prompt instructions.

This is the last line of defense.
"""

import re
from dataclasses import dataclass

@dataclass
class SafetyResult:
    """Result of a safety check."""
    is_safe: bool
    flags: list[str]        # what was flagged
    filtered_response: str # cleaned version if needed

class SafetyFilter:
    """Rule-based response safety filter.
    
    WHY NOT USE AN LLM FOR THIS?
    An LLM safety check adds latency and cost to every response.
    Rule-based filters catch the known-bad patterns instantly.
    You'd add an LLM safety check only in high-risk domains
    (medical, legal, financial advice).
    """

    def __init__(self):
        # Patterns that should never appear in a knowledge base response
        self.blocked_patterns = {
            "medical_advice": [
                r'you should (take|stop taking) [\w\s]+ medication',
                r'(increase|decrease) your dosage',
                r'this (is|could be) (a sign of|symptom of)',
            ],
            "legal_advice": [
                r'you (should|must) (sue|file a lawsuit)',
                r'this (is|constitutes) (fraud|illegal)',
                r'you have (legal|grounds|a case)',
            ],
            "financial_advice": [
                r'you should (buy|sell|invest in)',
                r'this stock (will|is going to)',
                r'guaranteed (returns|profit)',
            ],
            "harmful_content": [
                r'(how to|instructions for) (hack|break into)',
                r'(how to|ways to) (harm|hurt|injure)',
            ],
        }

        # Compile all patterns
        self._compiled = {}
        for category, patterns in self.blocked_patterns.items():
            self._compiled[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def check(self, response: str) -> SafetyResult:
        """Check a response for safety issues.
        
        Returns SafetyResult with is_safe = False if any
        blocked pattern matches.
        """

        flags = []

        for category, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(response):
                    flags.append(category)
                    break # one match per category is enough

        if flags:
            return SafetyResult(
                is_safe = False,
                flags = flags,
                filtered_response = self._build_safe_response(flags),
            )
        
        return SafetyResult(
            is_safe = True,
            flags = [],
            filtered_response = response,
        )
    
    def _build_safe_response(self, flags: list[str]) -> str:
        """Generate a safe fallback response when content is flagged."""
        flag_messages = {
            "medical_advice": "I can't provide medical advice. Please consult a healthcare professional.",
            "legal_advice": "I can't provide legal advice. Please consult a qualified attorney.",
            "harmful_content": "I can't help with that request.",
            "financial_advice": "I can't provide financial advice. Please consult a financial advisor.",
        }

        messages = [flag_messages.get(f, "") for f in flags if f in flag_messages]

        return " ".join(messages) + " I can help you find information from our documentation instead."
        