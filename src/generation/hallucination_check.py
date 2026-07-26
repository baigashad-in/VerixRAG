# hallucination_check.py
"""
Basic hallucination detection: does the answer stay grounded 
in the provided sources?

THIS IS NOT FOOLPROOF — it's a heuristic that catches obvious 
cases.
"""
import os
import json
import re
from litellm import completion

# ─── SKIP PATTERNS ───
# Sentences matching these are structural/transitional — they 
# don't make factual claims, so they shouldn't count as 
# "uncited claims" even if they lack a [Source N] marker.

SKIP_PATTERNS = [
    # Transitional openers
    r"^in summary\b",
    r"^to summarize\b",
    r"^overall\b",
    r"^in conclusion\b",
    r"^to conclude\b",
    r"^in short\b",
    
    # Source-referencing phrases
    r"^based on (the |these )?(provided |available )?(sources|documents|information|context)\b",
    r"^according to (the |these )?(provided |available )?(sources|documents|information)\b",
    
    # Structural phrases
    r"^(here is|here are|the following|below is|below are)\b",
    r"^(this (includes|covers|involves|describes|explains))\b",
    
    # Additive transitions (when they're short filler)
    r"^(additionally|furthermore|moreover|however|nevertheless|"
    r"nonetheless|meanwhile|consequently|therefore|thus|hence)\s*$",
    
    # Meta-commentary about the sources
    r"^(it is |it's )?(important|worth|notable|interesting) to note\b",
    r"^(as (mentioned|noted|stated|described|shown|indicated|"
    r"outlined|detailed|explained|discussed|highlighted))\b",
    r"^(the (sources?|documents?) (show|indicate|suggest|mention|"
    r"state|describe|provide|contain|cover|discuss|explain|note|"
    r"outline|specify|include|highlight|address))\b",
    r"^(please note|note that|keep in mind)\b",
    
    # Self-referential
    r"^(let me|i can|i will|i'll|i would)\b",
    r"^(for more (information|details|context))\b",
    
    # Question echoes
    r"^(regarding|concerning|with respect to|as for|about)\b",
    
    # Extended disclaimers
    r"(don't|do not|doesn't|does not) (have|contain|include|provide|cover) "
    r"(enough |sufficient |specific |detailed )?(information|details|data)",
    r"(not (covered|mentioned|addressed|discussed|included|found|available|specified|provided))"
    r" in the (sources|documents|available information)",
    r"no (specific |relevant |direct )?(information|details|data|mention) "
    r"(is |was )?(available|found|provided|given)",
    r"(cannot|can't) (find|locate|determine|confirm|verify) "
    r"(this|that|the answer|specific)",
    r"(the |these )?(sources?|documents?) (do not|don't|doesn't) "
    r"(mention|address|cover|discuss|include|provide|contain|specify)",
]

# Pre-compile for performance
_COMPILED_SKIP = [re.compile(p, re.IGNORECASE) for p in SKIP_PATTERNS]


def _is_transitional(sentence: str) -> bool:
    """Check if a sentence is a transitional/structural phrase."""
    s = sentence.strip().lower()
    for pattern in _COMPILED_SKIP:
        if pattern.search(s):
            return True
    return False

class HallucinationDetector:
    """Checks whether a generated response is grounded in sources.
    
    TWO APPROACHES:
    1. LLM-as-judge: ask another LLM call to verify (more accurate,
       adds latency and cost)
    2. Heuristic: check citation coverage (fast, less accurate)
    
    We implement both. Use heuristic in real-time, LLM-as-judge 
    in evaluation pipeline.
    """
    
    def __init__(self, model: str = "gemini/gemini-3.1-flash-lite"):
        self.model = model
    
    def check_citation_coverage(self, answer: str, num_sources: int) -> dict:
        
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?!\[Source)', answer) 
                     if s.strip()]
        
        cited = 0
        uncited_claims = []
        skipped = 0
        
        for sentence in sentences:
            # ── STEP 1: Skip very short filler ──
            if len(sentence.split()) < 8:
                skipped += 1
                continue
            
            # ── STEP 2: Check for citation marker ──
            has_citation = bool(
                re.search(r'\[Source[\s\d,]+\]', sentence)
            )
            
            if has_citation:
                cited += 1
                continue
            
            # ── STEP 3: Skip disclaimers ──
            disclaimers = [
                "don't have enough information",
                "not covered in the sources",
                "no information available",
                "cannot be determined from",
                "not mentioned in the",
                "not addressed in the",
                "sources do not",
                "documents do not",
            ]
            is_disclaimer = any(d in sentence.lower() 
                               for d in disclaimers)
            if is_disclaimer:
                skipped += 1
                continue
            
            # ── STEP 4: Skip transitional/structural phrases ──
            if _is_transitional(sentence):
                skipped += 1
                continue
            
            # ── If we get here, it's a real uncited claim ──
            uncited_claims.append(sentence)
        
        total = cited + len(uncited_claims)
        coverage = cited / total if total > 0 else 1.0
        
        return {
            "citation_coverage": coverage,
            "cited_claims": cited,
            "uncited_claims": uncited_claims,
            "skipped_transitional": skipped,
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
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            custom_llm_provider="gemini",
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