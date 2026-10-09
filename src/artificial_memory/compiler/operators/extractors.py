"""Generic extractors — topic-agnostic value extraction from records.

These patterns know UNITS and RELATIONS, never proper nouns.  A festival, a
restaurant, and a venue are all "events"; the extractor does not care which.

Every extractor returns a list of (value, source_text, date) so callers can
build auditable proofs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


@dataclass(frozen=True)
class Observation:
    value: str
    source: str
    date: str = ""


# --------------------------------------------------------------------------- #
#  record accessors — the engine is record-shape agnostic.
#
#  Synthetic corpus turns expose .text/.date; production StructuredIR exposes
#  .raw_content/.value/.time_scope.  Both are accepted so the SAME executor
#  code runs the synthetic suite and the real pipeline.  This is deliberate:
#  it prevents the production path from ever depending on a shape the tests
#  do not exercise.
# --------------------------------------------------------------------------- #

def _text_of(r) -> str:
    for attr in ("raw_content", "text", "value"):
        v = getattr(r, attr, None)
        if isinstance(v, str) and v:
            return v
    return ""


def _date_of(r) -> str:
    for attr in ("time_scope", "date"):
        v = getattr(r, attr, None)
        if isinstance(v, str) and v:
            return v
    return ""


# --------------------------------------------------------------------------- #
#  numeric + unit  (SUM, RATIO, GAP, FILTER-by-unit)
# --------------------------------------------------------------------------- #

_NUM_UNIT = re.compile(
    r"\b(\d{1,6}(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?|"
    r"points?|pts?|books?|coins?|records?|miles?|km|dollars?|\$|%)",
    re.I,
)

_UNIT_ALIASES = {
    "hr": "hours", "hrs": "hours", "hour": "hours",
    "min": "minutes", "mins": "minutes", "minute": "minutes",
    "day": "days", "week": "weeks", "month": "months", "year": "years",
    "pt": "points", "pts": "points", "point": "points",
    "book": "books", "coin": "coins", "record": "records",
    "$": "dollars", "dollar": "dollars",
    "km": "km", "mile": "miles",
}


def _norm_unit(raw: str) -> str:
    u = raw.lower().strip()
    return _UNIT_ALIASES.get(u, u)


def extract_numeric(records: Sequence, unit: str | None = None) -> list[Observation]:
    """Extract (number, unit) observations, optionally filtered to one unit."""
    out: list[Observation] = []
    for r in records:
        text = _text_of(r)
        date = _date_of(r)
        for m in _NUM_UNIT.finditer(text):
            raw_num, raw_unit = m.group(1), m.group(2)
            u = _norm_unit(raw_unit)
            if unit and u != unit.lower():
                continue
            num = raw_num.replace(",", "")
            out.append(Observation(value=f"{num} {u}", source=text.strip(), date=date))
    return out


# --------------------------------------------------------------------------- #
#  events  (UNION, COUNT over attended/visited/went-to things)
# --------------------------------------------------------------------------- #

_EVENT_VERBS = (
    r"went to|attended|visited|participated in|took part in|joined|signed up for",
)
_EVENT_PATTERNS = [
    re.compile(
        r"\b(?:went to|attended|visited|participated in|took part in|joined|signed up for)\s+"
        r"(?:the\s+)?([A-Z0-9][\w'&.\-]*(?:\s+[A-Z0-9][\w'&.\-]*)*)",
        re.I,
    ),
]


def extract_events(records: Sequence, concept: str | None = None) -> list[Observation]:
    """Extract event mentions.  ``concept`` optionally scopes by keyword."""
    out: list[Observation] = []
    for r in records:
        text = _text_of(r)
        date = _date_of(r)
        low = text.lower()
        if concept and concept.lower() not in low:
            continue
        for pat in _EVENT_PATTERNS:
            for m in pat.finditer(text):
                name = m.group(1).strip().rstrip(".,;:")
                if len(name) < 3:
                    continue
                out.append(Observation(value=name, source=text.strip(), date=date))
    return out


# --------------------------------------------------------------------------- #
#  key-value slots  (SLOT_LOOKUP: "X is Y", "X: Y")
# --------------------------------------------------------------------------- #

_SLOT_PATTERNS = [
    re.compile(r"\b([A-Za-z][A-Za-z ]{2,30}?)\s*(?:is|:)\s*([^\n.,;]{1,60})"),
]


def extract_slots(records: Sequence, slot_hint: str | None = None) -> list[Observation]:
    """Extract 'attribute is value' style slots.

    ``slot_hint`` scopes to a single attribute (e.g. "city", "email").
    """
    out: list[Observation] = []
    for r in records:
        text = _text_of(r)
        date = _date_of(r)
        for pat in _SLOT_PATTERNS:
            for m in pat.finditer(text):
                key, val = m.group(1).strip().lower(), m.group(2).strip()
                if slot_hint:
                    hint = slot_hint.lower()
                    # match the hint against the attribute key (last word wins
                    # for multiword keys like "home city" -> "city")
                    key_tokens = key.split()
                    if hint not in key and hint not in key_tokens and hint not in val.lower():
                        continue
                out.append(Observation(value=f"{key}={val}", source=text.strip(), date=date))
    return out


# --------------------------------------------------------------------------- #
#  ordered lists  (INDEX: "1. foo\n2. bar")
# --------------------------------------------------------------------------- #

_LIST_ITEM = re.compile(r"^\s*(\d+)[.)]\s+(.+?)\s*$", re.M)


def extract_list(records: Sequence, scope: str | None = None) -> list[str]:
    """Extract an ordered numbered list from the record that contains one.

    ``scope`` optionally restricts to the record whose text contains that
    token (e.g. a city name), so a multi-list corpus returns the RIGHT list
    instead of the first one encountered.  Among records containing the scope
    token, the one with the most scope occurrences wins, so a stray proper-noun
    collision elsewhere does not hijack the selection.
    """
    if scope:
        sco = scope.lower()
        # Prefer a record where the scope appears as "in <scope>" / "for
        # <scope>" (a genuine contextual anchor) over a stray name collision.
        anchor = re.compile(rf"\b(?:in|for|at|of)\s+{re.escape(sco)}\b", re.I)
        best: tuple[int, list[str]] = (0, [])
        for r in records:
            text = _text_of(r)
            if sco not in text.lower():
                continue
            items = [m.group(2).strip() for m in _LIST_ITEM.finditer(text)]
            if len(items) >= 2:
                # anchored occurrence scores higher than a bare mention
                score = (2 if anchor.search(text) else 0) + text.lower().count(sco)
                if score > best[0]:
                    best = (score, items)
        if best[1]:
            return best[1]
    # Pass 2: first record with a list
    for r in records:
        text = _text_of(r)
        items = [m.group(2).strip() for m in _LIST_ITEM.finditer(text)]
        if len(items) >= 2:
            return items
    return []


# --------------------------------------------------------------------------- #
#  persons / relations  (COUNT born, FILTER by relation)
# --------------------------------------------------------------------------- #

_PERSON = re.compile(r"\b([A-Z][a-z]+)\b")


def extract_persons(records: Sequence) -> list[Observation]:
    """Extract capitalised person-like tokens (crude NER, topic-agnostic)."""
    seen: set[str] = set()
    out: list[Observation] = []
    for r in records:
        text = _text_of(r)
        date = _date_of(r)
        for m in _PERSON.finditer(text):
            name = m.group(1)
            if name in seen:
                continue
            seen.add(name)
            out.append(Observation(value=name, source=text.strip(), date=date))
    return out


# --------------------------------------------------------------------------- #
#  dates (SUPERSEDES ordering)
# --------------------------------------------------------------------------- #

def parse_date(text: str) -> datetime | None:
    if not text:
        return None
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None
