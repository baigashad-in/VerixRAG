"""
Query transformation strategies to improve retrieval quality.

1. MULTI-QUERY: break a complex question into sub-questions,
   retrieve for each, merge the candidates.
2. HyDE (Hypothetical Document Embeddings): embed an LLM-written
   hypothetical ANSWER instead of the question, because an answer
   looks more like the document chunks than the question does.
3. QUERY EXPANSION: add related terms to improve keyword matching
   ("refund" -> "refund return money back reimburse").

The model, timeout and retries come from QUERY_REWRITE_MODEL,
QUERY_REWRITE_TIMEOUT and QUERY_REWRITE_RETRIES.
"""

import json
import os
import threading
import time
from collections import Counter

from litellm import completion


class QueryTransformer:
    """LLM-based query rewriting with bounded cost and call statistics."""

    MAX_QUERY_LENGTH = 2000

    def __init__(self, model: str | None = None, timeout: float | None = None,
                 num_retries: int | None = None):
        self.model = model or os.getenv("QUERY_REWRITE_MODEL", "gemini/gemini-3.1-flash-lite")
        self.timeout = timeout if timeout is not None else float(
            os.getenv("QUERY_REWRITE_TIMEOUT", "5"))
        self.num_retries = num_retries if num_retries is not None else int(
            os.getenv("QUERY_REWRITE_RETRIES", "0"))
        self.stats = Counter()          # calls, errors, tokens, ms
        self._lock = threading.Lock()

    def _complete(self, prompt: str) -> str:
        """Call the rewrite model and return its text; records calls, tokens and time."""
        kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                  "timeout": self.timeout, "num_retries": self.num_retries}
        if self.model.startswith("gemini/"):             # no temperature: deprecated for Gemini 3
            kwargs["api_key"] = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            kwargs["custom_llm_provider"] = "gemini"
        else:
            kwargs["temperature"] = 0.0
        t0 = time.perf_counter()
        try:
            response = completion(**kwargs)
        except Exception:
            with self._lock:
                self.stats["calls"] += 1
                self.stats["errors"] += 1
                self.stats["ms"] += round((time.perf_counter() - t0) * 1000)
            raise
        usage = getattr(response, "usage", None)
        with self._lock:
            self.stats["calls"] += 1
            self.stats["tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
            self.stats["ms"] += round((time.perf_counter() - t0) * 1000)
        return (response.choices[0].message.content or "").strip()

    def _check(self, query: str):
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValueError("Query too long")

    @staticmethod
    def _safe_parse_json_list(raw: str, max_items: int = 5) -> list[str]:
        """Parse LLM output as a JSON list of short strings; [] if malformed."""
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [item.strip() for item in parsed[:max_items]
                if isinstance(item, str) and item.strip() and len(item) <= 500]

    def multi_query(self, query: str, n: int = 3) -> list[str]:
        """Return the original query plus up to n focused sub-queries."""
        self._check(query)
        raw = self._complete(
            f"Break this question into {n} focused sub-questions for searching a "
            "knowledge base. Return ONLY a JSON array of strings, no other text.\n"
            f"Question: {query}\n"
            'Example output: ["sub-question 1", "sub-question 2", "sub-question 3"]')
        sub_queries = self._safe_parse_json_list(raw, max_items=n)
        return [query] + [q for q in sub_queries if q != query]

    def hyde(self, query: str) -> str:
        """Return a short hypothetical documentation passage that answers the query."""
        self._check(query)
        text = self._complete(
            "Write a short paragraph that would answer this question, as if it came "
            "from a documentation page. Do not say you don't know; give your best "
            f"guess at what the answer would look like. Keep it to 2-3 sentences.\n\nQuestion: {query}")
        return text[:2000]

    def expand_query(self, query: str) -> str:
        """Return the query with 3-5 related search terms appended, on one line."""
        self._check(query)
        text = self._complete(
            "Add 3-5 related search terms to this query to help find relevant documents. "
            "Return ONLY the original query followed by the added terms, on one line.\n"
            f"Query: {query}\n"
            'Example: "refund policy" -> "refund policy return money back reimburse exchange"')
        line = text.splitlines()[0].strip().strip('"') if text else ""
        return line[:500] or query