"""
Query transformation startegies to improve retrieval quality.

THREE STRATEGIES:

1. MULTI-QUERY: Break a complex question into sub-questions,
retrieve for each, merge results. Solves the "3 questions
in one" problem.

2. HyDE (Hypothetical Document Embeddings): Instead of embedding
the QUESTION, ask the LLM to write a hypothetical ANSWER,
then embed THAT. Why? Because a hypothetical answer looks more like the actual document chunks than the question does.

Query: "how do I undo a payment?"
HyDE answer: "To reverse a payment, navigate to Transaction History, select the payment, and click Initiate Chargeback..."
-> This embedding is much closer to your actual docs.

3. QUERY EXPANSION: Add related terms to the query to improve
keyword matching. "refund" -> "refund return money back reimburse"
"""

import json
from litellm import completion

class QueryTransformer:
    """Transforms user queries to improve retrieval.
    
    WHY LITELLM?
    It's unified interface to OpenAI, Anthropic, local models,
    etc. One function call, swap providers with a string change.
    """
    MAX_QUERY_LENGTH = 2000

    def __init__(self, model: str = "gemini/gemini-2.0-flash"):
        self.model = model

    def _safe_parse_json_list(self, raw: str, max_items: int = 5) -> list[str]:
        """Safely parse LLM output as a JSON list of strings."""
        # Strip markdown code if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            return[]
        
        if not isinstance(parsed, list):
            return []

        # Validate: must be strings, limit count and length
        result = []
        for item in parsed[:max_items]:
            if isinstance(item, str) and len(item) <= 500:
                result.append(item.strip())

        return result          


    def multi_query(self, query: str, n: int = 3) -> list[str]:
        """Break a complex query into focused sub-queries.
        
        WHEN THIS HELPS:
        "Tell me about pricing, setup, and integrations"
        -> ["What are the pricing plans and costs?",
            "How do I set up and install the product?",
            "What third-party integrations are available?"]

        Each su-query retrieves its own chunks, then we merge.
        This dramatically improves recall on multi-topic questions.
        """

        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValueError("Query too long")

        response = completion(
            model = self.model,
            messages = [{
                "role": "user",
                "content": f"""Break this question into {n} focused
                 sub-questions for searching a knowledge base. Return ONLY a JSON array of strings, no other text.
                 Question: {query}
Example output: ["sub-question 1", "sub-question 2", "sub-question 3"]
"""
            }],
            temperature = 0.0,
        )

        raw = response.choices[0].message.content.strip()
        sub_queries = self._safe_parse_json_list(raw, max_items=n + 2)

        return [query] + sub_queries if sub_queries else [query]
    
    def hyde(self, query: str) -> str:
        """Generate a Hypothetical Document Embedding.
        
        THE KEY INSIGHT:
        Questions and answers live in different parts of embedding space. "What is the refund policy?" and "The refund policy
        allows returns within 30 days" are semantically related
        but structurally very different.

        By generating a hypothetical answer, we create text that
        LOOKS like the chunks in our database, so vector similarity
        works much better.

        The hypothetical answer doesn't need to be CORRECT - it
        just needs to be in the right semantic neighborhood.
        """

        response = completion(
            model = self.model,
            messages = [{
                "role": "user",
                "content": f"""Write a short paragraph that would
                answer this question. Write it as if it's from a documentation page.
                Don't say "I don't know" - make your best guess at what the answer
                would look like. Keep it to 2-3 sentences.

                Question: {query}"""
            }],
            temperature = 0.0,
        )

        hypothetical = response.choices[0].message.content.strip()

        # Limit output length to avoid tokenization issues downstream
        return hypothetical[:2000]
    
    def expand_query(self, query: str) -> str:
        """"Add related terms to improve keyword matching.
        
        Simple but effective for BM25. If the user says "refund"
        but your docs say "return," expansion bridges the gap.
        """

        # Length check to prevent abuse - we don't want to feed a huge query into the LLM
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValueError("Query too long")

        response = completion(
            model = self.model,
            messages = [{
                "role": "user",
                "content": f"""Add 3-5 related search terms to this query to help find relevant documents. Return the original query
                with the added terms, all on one line.
                
                Query: {query}
Example: "refund policy" -> "refund policy return money back
reimburse exchange"  
"""
            }],
            temperature = 0.0,
        )

        expanded = response.choices[0].message.content.strip()
        return expanded