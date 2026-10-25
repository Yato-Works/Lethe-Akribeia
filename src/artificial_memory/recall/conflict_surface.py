"""Deterministic conflict surfacing (V4 4.24 / M14).

A generic detector: two stored documents asserting DIFFERENT values for
the same (entity, property) pair.  The surfaced annotation is a NOTE line
naming both values with their day provenance and the plan-21 semantics
("both on record, neither confirmed; report both rather than choosing
one").  Neither value is merged, overwritten or dropped -- supersession
stays exactly what it was on the write path.

Measured at 0 LLM on the controlled arena before being wired anywhere
(scripts/benchmarks/price_conflict_detector.py): the planted
contradiction pairs are captured 20/20 and the query-gated annotation
fires on 0/180 questions of the other nine categories, so the added
context text can only reach questions whose correct answer is "both, and
they conflict".

Single source of truth: the benchmark player (scripts/benchmarks/
am_current_player.py) and the shipped runtime (EnhancedContextBuilder,
opt-in via ``RuntimeConfig.conflict_surfacing``) both consume the
functions here, so benchmark and product stay in parity by construction.

Boundary: scorer-side evaluation vocabulary is referenced nowhere.
"""

from __future__ import annotations

import re
from collections import defaultdict

# Generic relational extractor: "<property phrase> <predicate> <prep> <value>".
_PRED = r"(runs|was|is|uses|pinned|set)"
_RE_ASSERT = re.compile(
    r"\b([A-Za-z][A-Za-z -]{2,30}?)\s+"
    + _PRED
    + r"\s+(?:actually\s+|really\s+|currently\s+)?"
    + r"(?:on|with|to|for)"
    + r"\s+([A-Za-z0-9][A-Za-z0-9_-]*)\b",
    re.I,
)
_RE_PROJECT = re.compile(r"\bfor project ([A-Za-z0-9_-]+)\b", re.I)

# Generic noise trim so "the user reported the cache layer" and "the cache
# layer actually" land on one key.  Fixed English stop-sets only.
_LEAD_NOISE = {
    "the", "a", "an", "my", "our", "their", "user", "teammate", "coworker",
    "colleague", "they", "i", "we", "he", "she", "reported", "mentions",
    "mention", "mentioned", "said", "says", "noted", "let", "us",
    "know", "knows", "that", "but", "and", "actual", "who", "which",
}
_TRAIL_NOISE = {
    "actually", "really", "currently", "still", "now", "allegedly",
    "reportedly", "regrettably", "seemingly",
}


def _norm_prop(prop: str) -> str:
    tokens = prop.lower().split()
    while tokens and tokens[0] in _LEAD_NOISE:
        tokens.pop(0)
    while tokens and tokens[-1] in _TRAIL_NOISE:
        tokens.pop()
    return " ".join(tokens)


def index_assertions(documents) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Index assertions over ``(content, provenance_label)`` documents.

    Returns ``{(entity, property): [(value, provenance), ...]}`` in
    document order.  Documents without an extracted entity are keyed
    under ``("(none)", property)``.
    """
    asserts: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for content, provenance in documents:
        projects = {m.group(1).lower() for m in _RE_PROJECT.finditer(content)}
        for m in _RE_ASSERT.finditer(content):
            prop = _norm_prop(m.group(1))
            if not prop:
                continue
            value = m.group(3)
            for proj in projects or ("(none)",):
                asserts[(proj, prop)].append((value, str(provenance)))
    return asserts


def detect_conflicts(documents) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Keys asserted with more than one distinct value, in document order."""
    asserts = index_assertions(documents)
    return {
        key: vals
        for key, vals in asserts.items()
        if len({v for v, _ in vals}) > 1
    }


def note_line(entity_display: str, prop: str, entries) -> str:
    """The conflict-surfacing NOTE: both values, their provenance kept.

    ``entries`` is ``[(value, provenance_label), ...]``; the label comes
    pre-formatted in the caller's own unit ("day 45", "2023-05-30", ...).
    """
    distinct: list[tuple[str, str]] = []
    for value, label in entries:
        if value not in {v for v, _ in distinct}:
            distinct.append((value, str(label)))
    chain = " vs ".join(f"{v} ({label})" for v, label in distinct)
    return (
        f"NOTE: conflicting reports for {entity_display} ({prop}): {chain}."
        " Both are on record and neither is confirmed; report both values"
        " rather than choosing one."
    )


def query_gated_keys(query: str,
                     conflicts: dict[tuple[str, str], list]) -> list[tuple[str, str]]:
    """Only the conflict keys whose entity AND property the query names.

    This is the measured 0/180 false-fire guard (V4 4.24 pricing).
    """
    qlow = query.lower()
    return [
        (proj, prop)
        for (proj, prop) in sorted(conflicts)
        if proj != "(none)" and proj in qlow and prop in qlow
    ]
