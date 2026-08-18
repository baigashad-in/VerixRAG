"""
Classic information-retrieval metrics for the retriever, with no LLM calls.

The eval dataset labels each question with the document (`source_file`)
and the section headings (`sections`) that contain the answer. Since
ingestion stores every chunk's source file and heading path, a chunk's
relevance can be graded without a judge:

    2 = right document AND right section   ("relevant")
    1 = right document, other section      (partial credit, NDCG only)
    0 = anything else

Chunk ids are NOT used as labels: they change every time the database
is rebuilt, while (document, section) does not.

METRICS (k = cut-off, e.g. 5 or 8; production retrieves 8):
- Hit@k:       any relevant chunk in the top k
- DocHit@k:    any chunk from the right document in the top k
- Recall@k:    fraction of the question's labelled sections that appear
               in the top k (multi-section questions need all of them)
- Precision@k: relevant chunks / k. Bounded by how many chunks a section
               has: a one-chunk section caps P@8 at 0.125, so compare
               precision between configurations, not against 1.0.
- MRR:         mean of 1 / rank of the first relevant chunk (0 if none
               in the top k_max)
- NDCG@k:      graded ranking quality; the ideal ranking is built from
               every chunk in the corpus, so it is the true optimum.

WHICH SECTIONS A CHUNK COVERS:
A chunk's stored `section` is the heading in effect where the chunk
STARTS. A chunk that crosses a heading (or begins at the document title
and holds several "##" sections) also covers every heading inside its
text, so those count too.

LABEL QUALITY:
Section labels are free text written, so some won't match any heading.
Those labels are counted and reported, and a question with no matchable 
label is left out of the section-level metrics (it still counts for 
DocHit), so a bad label is never reported as a retrieval failure.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)
_NUMBERING = re.compile(r"^\s*(section|part|chapter)?\s*[\d.]+[:.)-]?\s+", re.I)
_NON_WORD = re.compile(r"[^a-z0-9]+")
PATH_SEP = "›"


def _norm(text: str) -> str:
    text = _NUMBERING.sub("", text or "")
    return _NON_WORD.sub(" ", text.lower()).strip()


def _tokens(text: str) -> set[str]:
    return set(_norm(text).split())


@dataclass(frozen=True)
class Label:
    source: str | None     # file the section is in (None = the question's source_file)
    section: str           # heading text as written in the dataset


def parse_labels(sections: list[str] | None, source_file: str | None) -> list[Label]:
    """Turn dataset `sections` into labels.

    "Password Reset"                      -> (source_file, "Password Reset")
    "refund_policy.md: Digital Products"  -> ("refund_policy.md", "Digital Products")
    "refund_policy.md"                    -> ("refund_policy.md", "")  (whole document)
    """
    labels = []
    default = source_file if source_file and source_file != "cross_document" else None
    for raw in sections or []:
        raw = (raw or "").strip()
        if not raw:
            continue
        match = re.match(r"^([\w\-. ]+\.(?:md|txt|pdf|docx))\s*[:›>-]?\s*(.*)$", raw, re.I)
        if match:
            labels.append(Label(match.group(1).strip(), match.group(2).strip()))
        else:
            labels.append(Label(default, raw))
    return labels


def section_matches(label_section: str, chunk_section: str) -> bool:
    """Does a dataset heading refer to this chunk's heading path?

    Matches when the label equals, contains or is contained in any
    component of the path, or shares at least 60% of its words with the
    deepest heading. An empty label means the whole document.
    """
    if not label_section.strip():
        return True
    want = _norm(label_section)
    if not want:
        return True
    parts = [_norm(p) for p in (chunk_section or "").split(PATH_SEP) if _norm(p)]
    for part in parts:
        if want == part or (len(want) >= 4 and want in part) or (len(part) >= 4 and part in want):
            return True
    if parts:
        a, b = _tokens(label_section), set(parts[-1].split())
        if a and b and len(a & b) / len(a | b) >= 0.6:
            return True
    return False


def chunk_sections(section_path: str | None, content: str | None = None) -> list[str]:
    """Every heading path a chunk covers: where it starts, plus headings inside it."""
    path = section_path or ""
    covered = [path]
    root = path.split(PATH_SEP)[0].strip()
    for heading in _HEADING.findall(content or ""):
        heading = heading.strip()
        covered.append(f"{root} {PATH_SEP} {heading}" if root and heading != root else heading)
    return covered


def grade(chunk_source: str | None, chunk_section: str | list[str] | None,
          labels: list[Label], question_sources: set[str]) -> tuple[int, set[int]]:
    """(grade, indexes of the labels this chunk satisfies).

    chunk_section: one heading path, or every path the chunk covers
    (see chunk_sections).
    """
    paths = chunk_section if isinstance(chunk_section, list) else [chunk_section or ""]
    satisfied = {i for i, lab in enumerate(labels)
                 if lab.source in (None, chunk_source)
                 and (lab.source is not None or chunk_source in question_sources)
                 and any(section_matches(lab.section, p) for p in paths)}
    if satisfied:
        return 2, satisfied
    if chunk_source in question_sources:
        return 1, set()
    return 0, set()


@dataclass
class QuestionResult:
    question: str
    category: str
    sources: list[str]
    labels: list[str]
    matchable_labels: int
    grades: list[int]                     # grade of each retrieved chunk, by rank
    satisfied: list[list[int]]            # label indexes each retrieved chunk satisfies
    ideal: list[int] = field(default_factory=list)   # best possible grades, sorted
    retrieved: list[dict] = field(default_factory=list)

    @property
    def section_labelled(self) -> bool:
        return self.matchable_labels > 0


def _dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def question_metrics(r: QuestionResult, ks: tuple[int, ...]) -> dict:
    m: dict = {}
    for k in ks:
        top = r.grades[:k]
        m[f"doc_hit@{k}"] = float(any(g >= 1 for g in top))
        if r.section_labelled:
            m[f"hit@{k}"] = float(any(g == 2 for g in top))
            covered = set().union(*r.satisfied[:k]) if r.satisfied[:k] else set()
            m[f"recall@{k}"] = len(covered) / r.matchable_labels
            m[f"precision@{k}"] = sum(g == 2 for g in top) / k
            idcg = _dcg(r.ideal[:k])
            m[f"ndcg@{k}"] = _dcg(top) / idcg if idcg > 0 else 0.0
    if r.section_labelled:
        first = next((i for i, g in enumerate(r.grades) if g == 2), None)
        m["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    return m


def aggregate(per_question: list[dict]) -> dict:
    keys = sorted({k for m in per_question for k in m})
    out = {}
    for k in keys:
        values = [m[k] for m in per_question if k in m]
        out[k] = round(sum(values) / len(values), 4) if values else None
        out[f"n_{k}"] = len(values)
    return out