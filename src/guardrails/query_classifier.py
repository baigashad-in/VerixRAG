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

import os
import re
import json
from litellm import completion

from src.config import get_settings
from src.guardrails.scope_terms import alias_terms, build_scope_terms, stem
from src.retrieval.scoping import load_domain_config

class KeywordClassifier:
    """Rule-based scope check that runs before the LLM classifier."""

    def __init__(self, in_scope_terms: list[str] = None,
                 out_of_scope_patterns: list[str] = None, min_matches: int = 1,
                 strong_terms: list[str] = None):
        self.in_scope_terms = {stem(t.lower()) for t in (in_scope_terms or [])}
        self.min_matches = max(1, min_matches)
        # Product names etc.: one match is enough
        self.strong_terms = {stem(t.lower()) for t in (strong_terms or [])}

        self.out_of_scope_patterns = out_of_scope_patterns or [
            r"\btell me a joke",
            r"\bwhat(?:'s|s| is) the weather\b",
            r"\bwrite me an? (?:poem|story|essay|song)",
            r"\bwho is the (?:current )?president",
            r"\bwhat(?:'s|s| is)\s+\d+\s*[-+*/x]\s*\d+",   # math like "what's 2+2"
        ]
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.out_of_scope_patterns
        ]

    def classify(self, query: str) -> dict:
        """Return in_scope True/False with high confidence, or None to defer to the LLM."""

        query_lower = query.lower().replace("\u2019", "'")   # curly apostrophe

        # Check explicit out-of-scope patterns
        for pattern in self._compiled_patterns:
            if pattern.search(query_lower):
                return {
                    "in_scope": False,
                    "confidence": "high",
                    "reason": "Matched out-of-scope pattern",
                }

        # Check for in-scope keyword matches
        query_words = {stem(w) for w in re.findall(r'[a-z][a-z0-9]+', query_lower)}
        matches = sorted(query_words & self.in_scope_terms)

        strong = query_words & self.strong_terms
        if strong or len(matches) >= self.min_matches:
            matches = sorted(set(matches) | strong)
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


def keyword_classifier_from_settings(documents_dir: str) -> KeywordClassifier:
    """Build the keyword stage from the documents; empty when disabled or missing."""
    settings = get_settings()
    if not settings.keyword_scope_enabled or not os.path.isdir(documents_dir):
        return KeywordClassifier()
    aliases = load_domain_config(settings.domains_config).aliases
    return KeywordClassifier(
        in_scope_terms=list(build_scope_terms(documents_dir, aliases,
                                              top_n=settings.keyword_scope_body_terms)),
        strong_terms=list(alias_terms(aliases)),
        min_matches=settings.keyword_scope_min_matches,
    )
    
class LLMClassifier:
    """LLM-based scope classifier for ambiguous queries.
    
    Only called when KeywordClassifier is uncertain, This handles
    paraphrased queries like "I want my money back" (no keyword
    "refund" but clearly in scope) or edge cases the keyword
    list doesn't cover.
    """

    def __init__(self, model: str = "gemini/gemini-3.1-flash-lite",
                 scope_description: str = None,
                 timeout: float | None = None, num_retries: int | None = None):
        self.model = model
        # Bound the time a slow or failing provider can add to a request.
        # Without these, a Gemini 503 took 15-70 s (or crashed the request).
        self.timeout = timeout if timeout is not None else float(
            os.getenv("CLASSIFIER_TIMEOUT", "5"))
        self.num_retries = num_retries if num_retries is not None else int(
            os.getenv("CLASSIFIER_RETRIES", "1"))
        self.scope_description = scope_description or (
            "A knowledge base containing internal company documents. "
            "A query is in-scope if it relates to any topic that an organization "
            "might document internally, such as policies, procedures, products, "
            "technical documentation, or employee resources. "
            "Only reject queries that are clearly unrelated, like jokes, math, "
            "weather, or celebrity gossip."
        )

    def classify(self, query: str) -> dict:
        """Ask the LLM whether the query is in scope.

        FAILS OPEN: if the provider errors or times out (e.g. Gemini
        503 "model overloaded"), the query is treated as in scope, the
        same as an unparseable reply. The scope check is a convenience
        filter, not a safety control: an off-topic question that slips
        through still meets the confidence gate and a generator that
        only answers from retrieved sources. Failing closed would turn
        every provider outage into an outage of the whole API.
        """
        try:
            response = self._complete(query)
        except Exception as e:
            print(f"Scope classifier unavailable ({type(e).__name__}), defaulting to in-scope")
            return {
                "in_scope": True,
                "confidence": "low",
                "reason": "LLM classifier unavailable - defaulting to in-scope",
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }

        raw = (response.choices[0].message.content or "").strip()
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

    def _complete(self, query: str):
        return completion(
            model = self.model,
            messages = [{
                "role": "user",
                "content": f"""Determine if this query is within scope.
                Scope: {self.scope_description}
                Query: {query}
                Return ONLY JSON: {{"in_scope": true/false, "reason": "brief explanation"}}"""
                        }],
                            temperature = 0.0, 
                            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                            custom_llm_provider = "gemini",
                            timeout = self.timeout,
                            num_retries = self.num_retries,
                        )
        
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
