"""Run diverse test queries and check for expected results. These are integration tests that cover the entire pipeline, so they may be slower to run."""

import requests
import json
import time
import os
from dotenv import load_dotenv

API = "http://localhost:8000"

load_dotenv()
API_KEY = os.getenv("API_KEY")

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
}

queries = [
    # Easy - answer clearly in one document
    "What is the refund policy?",
    "How much does express shipping cost?",
    "How do I delete my account?",
    "What is the password requirement?",
    "How do I enable two-factor authentication?",

    # Medium - requires specific details
    "Can I get a refund on a digital product after 20 days?",
    "What items are non-refundable?",
    "How long does international shipping take?",
    "How do I track my order?",
    "How many shipping addresses can I save?",
    
    # Hard — multi-hop, requires connecting info across documents
    "What happens to my store credit if I delete my account?",
    "If I bought something 45 days ago and it arrived damaged, can I get a refund?",
    "What's the difference between standard and express shipping?",
    "How do I request a refund and how long does it take to process?",
    
    # Keyword specific — tests BM25
    "error code E-4012",
    "What does error E-4012 mean?",
    
    # Edge cases — vague or tricky
    "Can I get my money back?",
    "What are my options if I'm unhappy with a purchase?",
    "Tell me everything about shipping",
    "What notifications can I turn off?",
    
    # Out of scope — should be rejected
    "What is the weather in Tokyo?",
    "Write me a poem about cats",
    "Who won the World Cup?",
    "What is quantum computing?",
]

results = []
passed = 0
failed = 0
total_coverage = 0
cited_count = 0
uncited_count = 0

print(f"Running {len(queries)} test queries...\n")
print("=" * 70)

for i, query in enumerate(queries):
    print(f"\n[{i+1}/{len(queries)}] {query}")

    try:
        res = requests.post(
            f"{API}/api/chat",
            headers = headers,
            json = {"query": query},
            timeout = 30,
        )
        data = res.json()
    except Exception as e:
        print(f" ERROR: {e}")
        failed += 1
        continue

    answer = data.get("answer", "")
    guardrails = data.get("guardrails", {})
    hallucination = data.get("hallucination_check")
    citations = data.get("citations", [])
    classified = guardrails.get("classified_as", "unknown")

    # Check results
    coverage = hallucination.get("citation_coverage", 0) if hallucination else None
    is_suspicious = hallucination.get("is_suspicious", False) if hallucination else None
    cited = hallucination.get("cited_claims", 0) if hallucination else 0
    uncited = len(hallucination.get("uncited_claims", [])) if hallucination else 0

    print(f" Scope: {classified}")
    print(f" Answer: {answer[:100]}...")
    print(f" Citations: {len(citations)}")

    if coverage is not None:
        print(f" Coverage: {coverage:.0%}({cited} cited, {uncited} uncited)")
        print(f" Suspicious: {is_suspicious}")
        total_coverage += coverage
        cited_count += cited
        uncited_count += uncited

    if guardrails.get("input_pii_detected", 0) > 0:
        print(f"PII: {guardrails['input_pii_detected']} detected")

    results.append({
        "query": query,
        "classified_as": classified,
        "citation_coverage": coverage,
        "is_suspicious": is_suspicious,
        "cited_claims": cited,
        "uncited_claims": uncited,
        "num_citations": len(citations),
        "answer_length": len(answer),
    })

    passed += 1
    time.sleep(2) # respect rate limits

# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

in_scope = [r for r in results if r["classified_as"] == "in_scope"]
out_scope = [r for r in results if r["classified_as"] == "out_of_scope"]

print(f"\nTotal Queries: {len(queries)}")
print(f"Successful: {passed}")
print(f"Failed: {failed}")
print(f"In-scope: {len(in_scope)}")
print(f"Out-of-scope: {len(out_scope)}")

if in_scope:
    coverages = [r["citation_coverage"] for r in in_scope if r["citation_coverage"] is not None]
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0
    suspicious_count = sum(1 for r in in_scope if r["is_suspicious"])
    total_cited = sum(r["cited_claims"] for r in in_scope)
    total_uncited = sum(r["uncited_claims"] for r in in_scope)

    print(f"\n--- In-Scope Metrics ---")
    print(f"Avg citation coverage: {avg_coverage:.1%}")
    print(f"Total cited claims:    {total_cited}")
    print(f"Total uncited claims:  {total_uncited}")
    print(f"Hallucination rate:    {total_uncited}/{total_cited + total_uncited} = {total_uncited/(total_cited + total_uncited):.1%}" if (total_cited + total_uncited) > 0 else "N/A")
    print(f"Suspicious answers:    {suspicious_count}/{len(in_scope)}")

    # Per-query breakdown
    print(f"\n--- Per Query ---")
    for r in in_scope:
        status = "✓" if not r["is_suspicious"] else "⚠"
        cov = f"{r['citation_coverage']:.0%}" if r["citation_coverage"] is not None else "N/A"
        print(f"{status} [{cov}] {r['query'][:55]}")

print(f"\n--- Out-of-Scope ---")
for r in out_scope:
    print(f"✓ Rejected: {r['query'][:55]}")

# Save full results
with open("experiments/test_queries_results.json", "w") as f:
    json.dump(results, f, indent = 2)

print(f"\nFull results saved to experiments/test_queries_results.json")