"""
RAG evaluation metrics.

We implement these from scratch first so you understand the math,
then show the RAGAS library shortcut. In interviews, knowing
HOW faithfulness is calculated matters more than knowing which
library to pip install.
"""
import os
from litellm import completion
import json
import time

class JudgeError(Exception):
    """Judge reply missing or not parseable."""


class RAGMetrics:
    """Calculate retrieval and generation quality metrics.
    
    All metrics return a score between 0.0 and 1.0.
    Higher is better for all metrics.
    """

    def __init__(self, model: str = "groq/openai/gpt-oss-120b"):
        self.model = model
        self.request_delay = 2 # seconds between LLM calls (rate limits)

    def _llm_call(self, prompt: str) -> str:
        keys = [v for k, v in sorted(os.environ.items())
                if k.startswith("GROQ_API_KEY") and v]
        if not keys:
            raise RuntimeError("No GROQ_API_KEY* found in environment")
        if not hasattr(self, "_key_idx"):
            self._key_idx = 0

        time.sleep(4)  # light throttle; rotation handles the rest
        last_err = None
        for attempt in range(len(keys) * 3):
            key = keys[self._key_idx % len(keys)]
            try:
                response = completion(
                    model=os.environ.get("JUDGE_MODEL", "groq/openai/gpt-oss-20b"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    reasoning_effort="low",
                    max_tokens=2048,
                    api_key=key,
                )
                text = (response.choices[0].message.content or "").strip()
                if not hasattr(self, "trace"):
                    self.trace = []
                self.trace.append(text[:1500])
                if text:
                    return text
                print("Judge returned empty content, retrying", flush=True)
                continue
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if not any(t in msg for t in ["429", "rate", "quota", "timeout", "limit"]):
                    raise
                print(f"Judge key #{self._key_idx % len(keys) + 1} failed: "
                      f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                self._key_idx += 1
                # After every full pass over all keys, back off
                if (attempt + 1) % len(keys) == 0:
                    wait = 15 * ((attempt + 1) // len(keys))
                    print(f"All {len(keys)} keys limited, waiting {wait}s", flush=True)
                    time.sleep(wait)
        raise RuntimeError(f"LLM judge failed on all keys: {last_err}")

    def _safe_parse_json(self, raw: str) -> dict:
        """Parse LLM JSON output safely."""
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return {}
        
    # -- METRIC 1: FAITHFULNESS --

    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Does the answer ONLY conatin information from the sources?
        
        HOW IT WORKS:
        1. Extract individual claims from the answer
        2. For each claim, check if ANY source supports it
        3. Score = supported claims / total claims

        EXAMPLE:
        Answer: "Refunds take 30 days. We also offer free shipping."
        Sources: ["Refunds are processed within 30 days..."]
        
        Claim 1: "Refunds take 30 days" → supported ✓
        Claim 2: "We offer free shipping" → NOT in sources ✗
        Faithfulness = 1/2 = 0.5
        
        A score below 0.8 means your system is hallucinating.
        """

        # Step 1: Extract claims
        claims_raw = self._llm_call(f"""Extract every factual claim
                                    from this answer as a JSON array of strings.
                    Each claim should be a single, atomic statement.
                    
                    Answer: {answer}

                    Retrun ONLY a JSON array, example: ["Claim 1", "Claim 2"]""")
        
        parsed = self._safe_parse_json(claims_raw)
        if isinstance(parsed, list) and not parsed:
            return None  # no claims (refusal) - counted separately
        if not isinstance(parsed, list):
            raise JudgeError("claims extraction not a JSON list")
        
        claims = [str(c) for c in parsed[:20]]

        # Step 2: Check each claim against sources
        context_text = "\n\n".join(
            f"Source {i+1}: {c[:2000]}"
            for i, c in enumerate(contexts[:10])
        )

        verification_raw = self._llm_call(f"""For each claim below,
        determine if it is supported by the sources. Return a JSON object
        with claim as key and true/false as value.
                                          
        Sources:
        {context_text}
        Claims to verify:
        {json.dumps(claims)}
        Return ONLY JSON, example: {{"Claim 1": true, "Claim 2": false}}""")

        verdicts = self._safe_parse_json(verification_raw)
        if not verdicts:
            raise JudgeError("claim verification not parseable")
        
        supported = sum(1 for v in verdicts.values() if v is True)
        total = len(verdicts)
        self.last_claims = (supported, total)

        return supported / total if total > 0 else 0.0
    
    # -- METRIC 2: ANSWER RELEVANCY --

    def answer_relevancy(self, question: str, answer: str) -> float:
        """Does the answer actually address the question?
        
        HOW IT WORKS:
        1. Given the answer, generate N questions it WOULD answer
        2. Compare those generated questions to the ORIGINAL question
        3. High similaarity = the asnwer is on-topic
        
        WHY THIS APPOACH?
        Direct "is this relevant?" is too subjective. By generating
        reverse-questions, we get a more objective measurement.

        EXAMPLE:
        Question: "What is the refund policy?"
        Answer: "Standard shipping takes 5-7 business days."

        Generated questions from answer:
        ["How long does shipping take?", "What is standard shipping?"]

        Noen of these match "What is the refund policy?" -> low score.
        """

        # Generate questions the answer would address
        generated_raw = self._llm_call(f"""Given this answerm generate
        3 questions that this answer would be a good response to. Reurn
        ONLY a JSON array.
                                       
        Answer: {answer}

        Return: ["question 1", "question 2", "question 3"]""")

        generated = self._safe_parse_json(generated_raw)
        if not isinstance(generated, list) or not generated:
            raise JudgeError("reverse questions not parseable")
        
        # Score similarity between generated and original question

        score_raw = self._llm_call(f"""Score how similar each generated
        question is to the original question. Use a scale of 0.0 to 1.0
        where 1.0 means they ask the same thing.
                                   
        Original question: {question}

        Generated questions: {json.dumps(generated[:3])}
        Return ONLY a JSON array of scores, example: [0.8, 0.3, 0.9]""")

        scores = self._safe_parse_json(score_raw)
        if not isinstance(scores, list) or not scores:
            raise JudgeError("similarity scores not parseable")
        
        valid_scores = []
        for s in scores:
            try:
                val = float(s)
                valid_scores.append(min(max(val, 0.0), 1.0))
            except (TypeError, ValueError):
                continue

        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    # -- METRICS 3: CONTEXT PRECISION --

    def context_precision(self, question: str,
                          retrieved_contexts: list[str],
                          ground_truth: str) -> float:
         """Are relevant chunks ranked at the TOP of results?
         
         HOW IT WORKS:
         1. For each retrieved chunk, check if it's relevant to
         answering the question (givien ground truth)
         2. Calculate precision at each rank position
         3. Average the precision values (only at relevant positions)

         This is Mean Average Precision - the standard IR metric.

         EXAMPLE:
         Retrieved: [irrelevant, RELEVANT, irrelevant, RELEVANT, irrelevant]
         Relevant: position 2 and 4

         Precision@2 = 1/2 = 0.5 (1 relevant in top 2)
         Precision@4 = 2/4 = 0.5 (2 relevant in top 4)

         Context Precision = (0.5 + 0.5) / 2 = 0.5

         If the relevant chunks were at positions 1 and 2:
         Precision@1 = 1/1 = 1.0
         Precision@2 = 2/2 = 1.0
         Context Precision = 1.0 <- perfect ranking
         """

         if not retrieved_contexts:
            return 0.0
         
         # Judge each chunk's relevance
         relevance = []
         for i, ctx in enumerate(retrieved_contexts[:10]):
             judgement_raw = self._llm_call(f"""Is this context relevant
            to answering the question? Consider the fround truth answer.
                                            
            Question: {question}
            Ground truth answer: {ground_truth}
            Context to judge: {ctx[:2000]}

            Return ONLLY JSON: {{"relevant": true}} or {{"relevant": false}}""")
             
             judgement = self._safe_parse_json(judgement_raw)
             if not isinstance(judgement, dict) or "relevant" not in judgement:
                 raise JudgeError("relevance judgement not parseable")
             is_relevant = judgement["relevant"] is True
             relevance.append(is_relevant)

        # Calculate Mean Average Precision
         relevant_count = 0
         precision_sum = 0.0
    
         for i, is_rel in enumerate(relevance):
            if is_rel:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i

         if relevant_count == 0:
            return 0.0
         return precision_sum / relevant_count

    # -- METRIC 4: CONTEXT RECALL --

    def context_recall(self, ground_truth: str,
                       retrieved_contexts: list[str]) -> float:
        """Did retrieval find ALL the information needed?
        
        HOW IT WORKS:
        1. Break the ground truth answer into individual statements
        2. For each statement, check if ANY retrieved chunk covers it
        3. Score = covered statements / total statements

        EXAMPLE:
        Ground truth: "Refunds within 30 days. After that, store credit."
        Statements: ["refunds within 30 days", "store credit after 30 days"]

        Retrieved chunks mention refunds but NOT store credit.
        Recall = 1/2 = 0.5 - retrieval missed half the answer.
        """

        # Extract statements from ground truth
        statements_raw = self._llm_call(f"""Break this answer into individual factual statements. Return ONLY a JSON array.
                                        
        Answer: {ground_truth}
        Return: ["statement 1", "statement 2"]""")

        statements = self._safe_parse_json(statements_raw)
        if not isinstance(statements, list) or not statements:
            raise JudgeError("ground-truth statements not parseable")

        context_text = "\n\n".join(
            f"Context {i+1}: {c[:2000]}"
            for i, c in enumerate(retrieved_contexts[:10])
        )

        # Check coverage
        coverage_raw = self._llm_call(f"""For each statement, determine
        if it is supported by any of the provided contexts. Return a JSON
        object with statement as key and true/false as value.
                                      
        Contexts:
        {context_text}

        Statements:
        {json.dumps(statements[:20])}

        Return ONLY JSON: {{"statement 1": true, "statement 2": false}}""")        

        coverage = self._safe_parse_json(coverage_raw)
        if not coverage:
            raise JudgeError("coverage verdicts not parseable")
        
        covered = sum(1 for v in coverage.values() if v is True)
        total = len(coverage)

        return covered / total if total > 0 else 0.0
    