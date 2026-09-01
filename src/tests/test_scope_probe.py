"""Off-topic probe: attributing each answer to the layer that stopped it."""

from scripts.scope_probe import classifier_path, stopped_by, summarize


def test_stopped_by_layers():
    oos = {"guardrails": {"classified_as": "out_of_scope", "stage": "keyword"}}
    assert stopped_by(oos) == "keyword"
    assert stopped_by({"guardrails": {"classified_as": "out_of_scope", "stage": "llm"}}) == "llm"
    assert stopped_by({"abstained": {"passed": False}, "answer": "x"}) == "gate"
    refusal = "I don't have enough information in the available documents to answer this question."
    assert stopped_by({"answer": refusal}) == "generator"
    assert stopped_by({"answer": "The documents do not cover the weather."}) == "generator"
    partial = "Returns take 30 days [Source 1]. The documents do not cover Amazon orders."
    assert stopped_by({"answer": partial}) == "partial"
    assert stopped_by({"answer": "Paris is the capital of France."}) == "answered"


def test_classifier_path():
    assert classifier_path({"stage": "keyword"}) == "keyword"
    assert classifier_path({"stage": "llm"}) == "llm"
    assert classifier_path({"stage": "llm", "fallback": "Timeout: x"}) == "llm_failed_open"
    assert classifier_path({"stage": "llm_disabled"}) == "llm_disabled"
    assert classifier_path({}) == "unknown"


def test_summarize_counts_leaks_and_errors():
    rows = [{"source": "probe", "stopped_by": "keyword", "classifier": "keyword", "ms": 10.0},
            {"source": "probe", "stopped_by": "answered", "classifier": "llm_disabled", "ms": 900.0},
            {"source": "eval", "question": "q", "error": "HTTP 429"}]
    s = summarize(rows)
    assert s["questions"] == 3 and s["errors"] == 1 and s["leaks"] == 1
    assert s["stopped_by"]["keyword"] == 1 and s["stopped_by"]["gate"] == 0
    assert s["by_source"] == {"probe": {"keyword": 1, "answered": 1}}