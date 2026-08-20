"""Build in-scope keyword terms from the knowledge base for the keyword classifier."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)
_WORD = re.compile(r"[a-z][a-z0-9]+")

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before
being below between both but by can cannot could did do does doing down during each few for
from further had has have having he her here hers him his how i if in into is it its itself
just let me more most my no nor not now of off on once only or other our ours out over own
same she should so some such than that the their them then there these they this those
through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself
""".split())

# Words that appear in document headings but say nothing about the domain.
GENERIC = frozenset("""
about access account accounts action actions add additional after all also appendix apply
available basic before best can change changes check common contact contents create
creating current day days default details does example examples faq faqs following
general get getting guide guidelines help how important info information introduction
issue issues item items key level levels limit limits list make manage management manual
may mode new note notes number one only option options other overview page part per plan
plans please policies policy practices process provide questions quick reference request
requests required requirements section see set settings setup start started step steps
summary support system table terms time times troubleshooting two type types update
updates use used user users using via way what when within work year years
monday tuesday wednesday thursday friday saturday sunday january february march april june
july august september october november december click select navigate open enter based first
last next recently expected indicate original standard equal est inc minute minutes hour
hours character characters call returned accessed received receive send sent within
""".split())


def stem(word: str) -> str:
    """Light plural folding so 'refunds' matches 'refund'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "ches", "shes", "xes")):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokens(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower())
            if w not in STOPWORDS and len(w) >= 3]


def _keep(term: str) -> bool:
    return term not in GENERIC and stem(term) not in {stem(g) for g in GENERIC}


def alias_terms(aliases: dict[str, list[str]] | None) -> set[str]:
    """Return distinctive single-word product names from domain aliases."""
    return {t for names in (aliases or {}).values() for alias in names
            for t in tokens(alias) if _keep(t) and len(t) >= 5}


def build_scope_terms(documents_dir: str, aliases: dict[str, list[str]] | None = None,
                      top_n: int = 10, min_count: int = 2) -> set[str]:
    """Return stemmed terms from headings, titles, file names, aliases and distinctive words."""
    docs = {p.name: p.read_text(encoding="utf-8", errors="ignore")
            for p in sorted(Path(documents_dir).rglob("*"))
            if p.suffix in (".md", ".txt") and not p.name.startswith((".", "_"))}
    terms: set[str] = set()

    for name, text in docs.items():
        terms.update(tokens(Path(name).stem.replace("_", " ").replace("-", " ")))
        for heading in _HEADING.findall(text):
            terms.update(tokens(heading))

    for names in (aliases or {}).values():
        for alias in names:
            terms.update(tokens(alias))

    # Distinctive body words: high TF-IDF within one document
    counts = {name: Counter(tokens(text)) for name, text in docs.items()}
    doc_freq = Counter(t for c in counts.values() for t in c)
    n_docs = len(docs) or 1
    for c in counts.values():
        frequent = [t for t in c if c[t] >= min_count]
        scored = sorted(frequent, key=lambda t: c[t] * math.log((1 + n_docs) / doc_freq[t]),
                        reverse=True)
        terms.update(scored[:top_n])

    return {t for t in terms if _keep(t)}