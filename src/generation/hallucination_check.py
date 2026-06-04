# hallucination_check.py
"""
Basic hallucination detection: does the answer stay grounded 
in the provided sources?

THIS IS NOT FOOLPROOF — it's a heuristic that catches obvious 
cases.
"""
import json
from litellm import completion


class HallucinationDetector:
    """Checks whether a generated response is grounded in sources.
    
    TWO APPROACHES:
    1. LLM-as-judge: ask another LLM call to verify (more accurate,
       adds latency and cost)
    2. Heuristic: check citation coverage (fast, less accurate)
    
    We implement both. Use heuristic in real-time, LLM-as-judge 
    in evaluation pipeline.
    """
    
    def __init__(self, model: str = "gemini/gemini-2.0-flash"):
        self.model = model
    
    def check_citation_coverage(self, answer: str, 
                                 num_sources: int) -> dict:
        """Fast heuristic: are claims actually cited?
        
        Checks:
        - Does the response contain citation markers?
        - What percentage of sentences have citations?
        - Are there claims without any citation? (red flag)
        """
        import re
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer) 
                     if s.strip()]
        
        cited = 0
        uncited_claims = []
        
        for sentence in sentences:
            # Skip short filler sentences
            if len(sentence.split()) < 4:
                continue
            
            has_citation = bool(
                re.search(r'\[Source \d+\]', sentence)
            )
            
            if has_citation:
                cited += 1
            else:
                # Check if it's a disclaimer (allowed without citation)
                disclaimers = [
                    "don't have enough information",
                    "not covered in the sources",
                    "no information available",
                ]
                is_disclaimer = any(d in sentence.lower() 
                                   for d in disclaimers)
                if not is_disclaimer:
                    uncited_claims.append(sentence)
        
        total = cited + len(uncited_claims)
        coverage = cited / total if total > 0 else 1.0
        
        return {
            "citation_coverage": coverage,
            "cited_claims": cited,
            "uncited_claims": uncited_claims,
            "is_suspicious": coverage < 0.5 or len(uncited_claims) > 2,
        }
    
    def llm_judge(self, query: str, answer: str, 
                  sources: list[str]) -> dict:
        """Ask an LLM to verify grounding. Slower but more accurate.
        
        Use this in your evaluation pipeline, not in real-time.
        """
        # Limit input sizes
        if len(answer) > 10_000:
            return {"is_grounded": None, "error": "answer_too_long"}
    

        sources_text = "\n\n".join(
            f"Source {i+1}: {s[:2000]}" # truncate each source
            for i, s in enumerate(sources[:10]) # max 10 sources
        )
        
        response = completion(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""Evaluate whether this answer is 
fully supported by the provided sources.

Sources:
{sources_text}

Answer being evaluated:
{answer}

Respond in JSON:
{{
    "is_grounded": true/false,
    "unsupported_claims": ["list of claims not in sources"],
    "confidence": 0.0-1.0
}}"""
            }],
            temperature=0.0,
        )
        
        try:
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

            parsed = json.loads(raw)

            # Validate expected structure
            if not isinstance(parsed.get("is_grounded"), bool):
                return {"is_grounded": None, "error": "invalid_format"}
            
            return {
                "is_grounded": parsed["is_grounded"],
                "unsupported_claims": parsed.get("unsupported_claims", [])[:20],
                "confidence": min(max(float(parsed.get("confidence", 0.0)), 0.0), 1.0),
                
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return {"is_grounded": None, "error": "parse_failed"}