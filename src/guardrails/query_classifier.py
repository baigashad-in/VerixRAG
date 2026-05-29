"""
Detect out-of-scope queries BEFORE they hit the pipeline.

WHY THIS MATTERS:
Without this, a user asks "What's the weather in Tokyo?" and
your system retrieves the least-irrelevant chunks, feeds them
to the LLM, and generates a nonsense answer with fake citations.
That destroys trust instantly.

With this, you catch the question early and say: "That's
outside what I can help with. I can answer questions about
our refund policy, shipping, and account management."

TWO APPROACHES:
1. sklearn classifier - fast, works offline, needs trainging data
2. LLM classifier - no training data needed, slower, costs a call
We implement both.
"""

import re
import json
from litellm import completion

class KeywordClassifier:
    """Fast, zero-cost, zero-latency scope checker.
    
    This runs BEFORE any LLM call or retrieval. It checks
    whether the query contains any terms related to your
    domain. If it's clearly off-topic, reject immediately.
    
    This catches obvious cases like "tell me a joke" or
    "what's 2+2" without spending any resources.
    """

    def __init__(self, in_scope_terms: list[str] = None,
                 out_of_scope_patterns: list[str] = None):
        
        """
        Args:
            in_scope_terms: words that suggest an in-scope query.
                Build this list from your actual documents.
            out_of_scope_patterns: regex patterns for clearly
                off-topic queries.
        """
        self.in_scope_terms = set(t.lower() for t in (in_scope_terms or [
            "refund", "return", "shipping", "delivery", "account", "password", "billing", "invoice", "cancel", "subscription",
            "payment", "credit", "order", "track", "policy",
            "pricing", "plan", "upgrade", "support", "help",
        ]))

        self.out_of_scope_patterns = out_of_scope_patterns or [
            r"tell me a joke",
            r"what.s the weather",
            r"write me a (poem|story|essay|song)",
            r"who is the president",
            r"what.s? \d+\s*[\+\-\*\/]\s*\d+",   # math like "what's 2+2"
        ]
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.out_of_scope_patterns
        ]
    
    def classify(self, query: str) -> dict:
        """Quick scope check - no LLM needed.
        
        Returns:
            {"in_scope": bool, "confidence": str, "reason": str}

        confidence is "high" or "low":
        - high: we're sure it's out of scope (matched a pattern)
        - low: no keywords matched, but might still be in scope
                (fall through to LLM classifier)
        """

        query_lower = query.lower()

        # Check explicit out-of-scope patterns
        for pattern in self._compiled_patterns:
            if pattern.search(query_lower):
                return {
                    "in_scope": False,
                    "confidence": "high",
                    "reason": "Matched out-of-scope pattern",
                }
            
        # Check for in-scope keyword matches
        query_words = set(re.findall(r'[a-z]+', query_lower))
        matches = query_words & self.in_scope_terms

        if matches:
            return {
                "in_scope": True,
                "confidence": "high",
                "reason": f"Matched terms: {', '.join(matches)}",
            }
        
        # Uncertain = no keywords matched but not clearly off-topic
        return {
            "in_scope": None, # uncertain
            "confidence": "low",
            "reason": "No keyword matches - needs LLM classification",
        }
    
class LLMClassifier:
    """LLM-based scope classifier for ambiguous queries.
    
    Only called when KeywordClassifier is uncertain, This handles
    paraphrased queries like "I want my money back" (no keyword
    "refund" but clearly in scope) or edge cases the keyword
    list doesn't cover.
    """

    def __init__(self, model: str = "groq/llama-3.3-70b-versatile",
                 scope_description: str = None):
        self.model = model
        self.scope_description = scope_description or (
            "Customer support for an e-commerce platform. Topics include:"
            "refunds, returns, shipping, account management, billing,"
            "subscriptions, order tracking, and product policies."
        )

    def classify(self, query: str) -> dict:
        response = completion(
            model = self.model,
            messages = [{
                "role": "user",
                "content": f"""Determine if this query is within scope.
Scope: {self.scope_description}
Query: {query}
Return ONLY JSON: {{"in_scope": true/false, "reason": "brief explanation"}}"""
            }],
            temperature = 0.0,
        )

        raw = response.choices[0].messages.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

        try:
            parsed = json.loads(raw)
            return {
                "in_scope": bool(parsed.get("in_scope", False)),
                "confidence": "high",
                "reason": str(parsed.get("reason", ""))[:200],
            }
        except json.JSONDecodeError:
            # If LLM response is unparseable, default to allowing
            return {
                "in_scope": True,
                "confidence": "low",
                "reason": "LLM classification failed - defaulting to in-scope",
            }
        
class QueryClassifierPipeline:
    """Two-stage classifier: fast keyword check -> LLM fallback.
    
    ARCHITECTURE:
    Stage 1 (KeywordClassifier): <1ms, free, catches obvious cases
    Stage 2 (LLMClassifier): ~500ms, costs an API call, handles ambiguity

    90% of queries get  resolved at Stage 1. The LLM only
    fires for genuinely ambiguous queries.
    """

    def __init__(self, keyword_classifier: KeywordClassifier = None,
                 llm_classifier: LLMClassifier = None):
        self.keyword = keyword_classifier or KeywordClassifier()
        self.llm = llm_classifier or LLMClassifier()

    def classify(self, query: str) -> dict:
        # Stage 1: fast keyword check
        result = self.keyword.classify(query)

        if result["confidence"] == "high":
            result["stage"] = "keyword"
            return result
        
        # Stage 2: LLM fallback for uncertain queries
        result = self.llm.classify(query)
        result["stage"] = "llm"
        return result
