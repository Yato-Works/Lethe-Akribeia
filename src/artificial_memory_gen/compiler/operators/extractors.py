"""Generic extractors — topic-agnostic value extraction from records.

These patterns know UNITS and RELATIONS, never proper nouns.  A festival, a
restaurant, and a venue are all "events"; the extractor does not care which.

Every extractor returns a list of (value, source_text, date) so callers can
build auditable proofs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


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

# A countable noun that can carry a bare quantity: "<number> <noun>".
# Allows one intervening adjective ("five short stories", "three red books")
# so compound nouns match, but not arbitrary words.
_COUNT_NOUN = (
    r"short stories|stories|poems|poem|books?|coins?|records?|pieces?|"
    r"essays?|articles?|novels?|chapters?|games?|movies?|films?|songs?|"
    r"classes?|sessions?|meetings?|trips?|visits?|hours?|days?|weeks?|"
    r"months?|years?|points?|miles?|dollars?",
)

# Bare integer or decimal, followed by a measurement unit OR a countable
# noun.  This captures "17 poems", "1 piece", "5 short stories" as well as
# "70 hours", "$200,000", "10%".
_NUM_UNIT = re.compile(
    r"\b(\d{1,6}(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?|"
    r"points?|pts?|books?|coins?|records?|miles?|km|dollars?|\$|%|"
    rf"{_COUNT_NOUN})",
    re.I,
)

# Number words (English) used as quantities, e.g. "five short stories".
# NOTE: vague quantifiers ("a", "an", "couple", "few", "several") are
# deliberately EXCLUDED — "a book" is not reliably a count of 1, and
# guessing would inject noise into every SUM.  Only unambiguous number
# words map to integers.
_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

# "<number word> [optional single adjective] <countable noun>"
_NUMWORD_COUNT = re.compile(
    rf"\b({'|'.join(_NUM_WORDS)})\s+(?:\w+\s+)??({_COUNT_NOUN})\b",
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


def extract_numeric(records: Sequence, unit: str | None = None,
                    completion_only: bool = False) -> list[Observation]:
    """Extract (number, unit) observations, optionally filtered to one unit.

    Recognises both digit quantities ("17 poems") and number-word quantities
    ("five short stories").  Unit normalisation maps synonyms (hr->hours).

    When ``completion_only`` is True, only quantities that sit in a
    completion context are returned — i.e. the number is governed by a verb
    like "have written / completed / finished five short stories".  This keeps
    a "how many have you completed" SUM from counting incidental mentions
    ("a connection between the two stories", "one piece of clothing to pack"),
    which reference an object rather than assert a completed count.
    """
    out: list[Observation] = []
    for r in records:
        text = _text_of(r)
        date = _date_of(r)

        # Pass 1: digit + unit
        for m in _NUM_UNIT.finditer(text):
            raw_num, raw_unit = m.group(1), m.group(2)
            u = _norm_unit(raw_unit)
            if unit and u != unit.lower():
                continue
            if completion_only and not _in_completion_context(text, m.start()):
                continue
            num = raw_num.replace(",", "")
            out.append(Observation(value=f"{num} {u}", source=text.strip(), date=date))

        # Pass 2: number-word + countable noun
        for m in _NUMWORD_COUNT.finditer(text):
            word, noun = m.group(1).lower(), m.group(2).lower()
            n = _NUM_WORDS[word]
            u = _norm_unit(noun)
            if unit and u != unit.lower() and u.rstrip("s") != unit.lower():
                continue
            if completion_only and not _in_completion_context(text, m.start()):
                continue
            out.append(Observation(value=f"{n} {u}", source=text.strip(), date=date))
    return out


# Verbs that assert a quantity has been produced/completed (vs. merely
# mentioned).  Dataset-agnostic: these are generic English completion verbs.
_COMPLETION_VERBS = (
    "written", "wrote", "completed", "complete", "finished", "finish",
    "published", "publish", "read", "watched", "attended", "earned", "saved",
    "bought", "purchased", "made", "built", "collected", "achieved", "done",
    "produced", "recorded",
)


def _in_completion_context(text: str, pos: int) -> bool:
    """True if the quantity at ``pos`` is governed by a completion verb.

    Looks in a short window immediately BEFORE the number for a completion
    verb (optionally preceded by have/has/had/'ve).  Falls back to False.
    """
    window = text[max(0, pos - 60):pos].lower()
    for v in _COMPLETION_VERBS:
        if re.search(rf"\b{re.escape(v)}\b", window):
            return True
    return False


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
