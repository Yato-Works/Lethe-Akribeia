"""Query Parser — the ONLY layer that reads the raw question string.

NL -> Operator AST.  The parser recognises *operations and units*, never
benchmark topics.  "How many hours in total" -> SUM(EXTRACT(numeric, hours)).
It cannot emit a topic-specific branch because its output vocabulary is the
AST node set, not the benchmark vocabulary.
"""

from __future__ import annotations

import re

from artificial_memory.compiler.operators import ast_nodes as A


# Generic operation cues (surface forms of the six cognitive operators).
_SUM_CUES = (
    "how many", "how much", "total", "in total", "altogether", "combined",
    "sum", "overall",
)
_COUNT_CUES = ("how many", "number of", "count")
_LAST_CUES = ("last", "final", "most recent")
_NTH_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.I)
_PREVIOUS_CUES = ("previous", "previously", "earlier", "before", "former",
                  "initially", "originally", "used to")
_RATIO_CUES = ("percentage", "percent", "what fraction", "ratio", "proportion")
_GAP_CUES = ("need to earn", "still need", "remaining", "left to", "to reach",
             "how many more", "short of")


_UNIT_WORDS = {
    "hours": "hours", "hour": "hours", "hrs": "hours", "hr": "hours",
    "minutes": "minutes", "minute": "minutes", "mins": "minutes",
    "days": "days", "day": "days",
    "weeks": "weeks", "week": "weeks",
    "months": "months", "month": "months",
    "years": "years", "year": "years",
    "points": "points", "point": "points", "pts": "points",
    "books": "books", "book": "books",
    "coins": "coins", "coin": "coins",
    "records": "records", "record": "records",
}


class QueryParser:
    """Compiles a natural-language question into an Operator AST.

    Returns ``None`` when the question does not map to a known deterministic
    operation (the caller then falls back to the LLM reader).
    """

    def parse(self, question: str) -> A.Node | None:
        ql = question.lower()

        # Confidence gate: never fire a deterministic operator on a question
        # that is NOT a precise numeric/extraction ask.  Yes/No questions,
        # advice/preference requests, and open-ended prompts belong to the LLM.
        if _is_non_computational(ql):
            return None

        # 1. GOAL_GAP — target minus current
        if any(c in ql for c in _GAP_CUES) and ("point" in ql or "step" in ql or "mile" in ql):
            return A.Gap(
                target=A.Extract(concept="numeric", unit="points"),
                current=A.Extract(concept="numeric", unit="points"),
                topic_terms=tuple(_topic_terms(ql)),
            )

        # 2. RATIO — a percentage / fraction of two quantities
        if any(c in ql for c in _RATIO_CUES):
            return A.Ratio(
                numerator=A.Extract(concept="numeric"),
                denominator=A.Extract(concept="numeric"),
                as_percent=True,
                topic_terms=tuple(_topic_terms(ql)),
            )

        # 3. INDEX — nth item, or last item
        nth = _NTH_RE.search(question)
        scope = _detect_scope(question)
        if nth and ("list" in ql or "parameter" in ql or "item" in ql
                    or "step" in ql or "rule" in ql or "ingredient" in ql):
            return A.Index(operand=A.Extract(concept="list"), ordinal=int(nth.group(1)),
                           selector="nth", scope=scope)
        if any(c in ql for c in _LAST_CUES) and ("list" in ql or "venue" in ql
                                                 or "item" in ql or "place" in ql
                                                 or "store" in ql or "option" in ql):
            return A.Index(operand=A.Extract(concept="list"), ordinal=-1,
                           selector="last", scope=scope)

        # 4. SUPERSEDES — a state that changed over time
        if _is_state_question(ql):
            return A.Supersedes(
                topic=A.Extract(concept="state"),
                want="earliest" if any(c in ql for c in _PREVIOUS_CUES) else "latest",
                topic_terms=tuple(_topic_terms(ql)),
            )

        # 4b. ORDER_FIRST — which of several events happened earliest
        if ("first" in ql or "earliest" in ql or "before the others" in ql) and (
            "event" in ql or "happened" in ql or "occurred" in ql or "which" in ql
            or "these" in ql
        ):
            return A.OrderFirst(operand=A.Extract(concept="event"))

        # 4c. SLOT_LOOKUP — "what is X's <attribute>?"
        slot_attr = _detect_slot_attribute(ql)
        if slot_attr and re.search(r"\bwhat(?:'s| is| was)\b", ql):
            return A.SlotLookup(operand=A.Extract(concept="slot"), attribute=slot_attr)

        # 5. SUM — a quantity distributed across the corpus
        unit = _detect_unit(ql)
        if unit and any(c in ql for c in _SUM_CUES):
            return A.Sum(operand=A.Extract(concept="numeric", unit=unit),
                         topic_terms=tuple(_topic_terms(ql)),
                         time_scope=_detect_time_scope(ql))

        # 6. COUNT / UNION over events or persons
        concept = _detect_concept(ql)
        if concept and any(c in ql for c in _COUNT_CUES):
            return A.Count(operand=A.Extract(concept=concept))

        return None


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _detect_unit(ql: str) -> str | None:
    """Detect the unit of the quantity being asked about.

    Prefers the noun immediately after 'how many/much <noun>' — that is the
    quantity being counted — over incidental time scopes like 'three weeks
    ago' which describe the window, not the counted unit.
    """
    # 1. The counted unit: "how many <noun>" / "how much <noun>"
    m = re.search(r"\bhow (?:many|much)\s+([a-z]+)", ql)
    if m:
        head = m.group(1)
        if head in _UNIT_WORDS:
            return _UNIT_WORDS[head]
        # a counted noun right after how many: "how many pieces", "how many books"
        if head.rstrip("s") in _COUNTABLE_HEADS:
            return head
        # "how many total pieces" / "how many pieces of X" -> skip fillers
        m2 = re.search(r"\bhow many\s+(?:total\s+|more\s+|additional\s+)?([a-z]+)", ql)
        if m2 and m2.group(1).rstrip("s") in _COUNTABLE_HEADS:
            return m2.group(1)
    # 2. Fall back to any explicit unit word in the question
    words = re.findall(r"[a-z]+", ql)
    for w in words:
        if w in _UNIT_WORDS:
            return _UNIT_WORDS[w]
    return None


# Countable heads that act as a unit when they directly follow "how many".
_COUNTABLE_HEADS = {
    "piece", "book", "coin", "record", "story", "poem", "essay", "article",
    "novel", "chapter", "game", "movie", "film", "song", "class", "session",
    "meeting", "trip", "visit", "hour", "day", "week", "month", "year",
    "point", "mile", "dollar", "item", "device", "place", "venue", "person",
    "baby", "event", "project",
}


_EVENT_HINTS = ("festival", "event", "concert", "class", "trip", "show",
                "meeting", "party", "workshop", "restaurant", "venue")
_PERSON_HINTS = ("who", "person", "people", "friend", "colleague", "baby",
                 "child", "born")


def _detect_concept(ql: str) -> str | None:
    if any(h in ql for h in _EVENT_HINTS):
        return "event"
    if any(h in ql for h in _PERSON_HINTS):
        return "person"
    return None


_STATE_VERBS = ("keep", "store", "live", "move", "work", "own", "have", "use")


def _detect_slot_attribute(ql: str) -> str | None:
    """Detect the attribute a slot-lookup question is asking for.

    Generic attribute vocabulary only — no benchmark entities.
    """
    attrs = (
        "phone number", "email", "home city", "address", "name",
        "birthday", "handle", "brand", "color", "favourite book",
    )
    for a in attrs:
        if a in ql:
            return a.split()[-1] if " " in a else a
    return None


def _detect_scope(question: str) -> str | None:
    """Detect a scoping proper noun, e.g. 'venues in Northgate' -> 'Northgate'.

    Used by INDEX to pick the right list when several exist.  Purely
    positional: whatever capitalised token follows in/for/at/of.
    """
    m = re.search(r"\b(?:in|for|at|of)\s+([A-Z][a-zA-Z]+)\b", question)
    return m.group(1) if m else None


_TOPIC_STOP = frozenset({
    "where", "what", "when", "which", "currently", "keep", "kept", "store",
    "stored", "have", "has", "had", "the", "my", "your", "are", "were",
    "with", "that", "this", "into", "from", "does", "did", "done", "now",
    "these", "days", "there", "their", "them", "then", "than", "also",
    # generic question/quantity fillers — never a topic entity
    "many", "much", "total", "spend", "spent", "since", "started", "again",
    "including", "combined", "across", "sessions", "session", "number",
    "count", "long", "time", "times", "went", "gone", "been", "being",
})


def _is_non_computational(ql: str) -> bool:
    """True if the question is NOT a precise numeric/extraction ask.

    Yes/No questions, advice/preference requests, and open-ended prompts must
    be handled by the LLM, never overridden by a deterministic operator.  A
    'Did I receive a higher discount?' or 'Do you have any ideas?' question
    has no single computable answer, so firing RATIO/GAP on it is a false
    positive.
    """
    # Yes/No openers
    if re.match(r"^\s*(did|do|does|is|are|was|were|have|has|had|can|could|should|would|will)\b", ql):
        # but allow genuine extraction phrasings that merely start with an aux
        if not re.search(r"\bhow (many|much|old|long|far)\b", ql):
            return True
    # advice / preference / recommendation REQUESTS (explicit ask forms only —
    # a bare "recommend"/"suggest" may be a past-fact reference like "the last
    # venue you recommended", which IS a computable extraction).
    advice = (
        "do you have any", "any ideas", "any suggestions", "any tips",
        "what should i", "do you think", "would it be", "help me",
        "feeling", "stuck", "advice on", "what do you", "your opinion",
        "can you recommend", "could you suggest", "would you recommend",
    )
    if any(a in ql for a in advice):
        return True
    return False


def _detect_time_scope(ql: str) -> str | None:
    """Detect a relative time window in the question.

    Returns a canonical token ('last_week', 'last_month', 'last_year',
    'past_months', 'past_week') when the question restricts the quantity to a
    relative window ("last month", "past few months", "previous year").  SUM
    uses this to drop quantities outside the window instead of summing the
    entire history.  Returns None when no relative window is expressed.
    """
    if re.search(r"\blast (?:week|7 days)\b", ql):
        return "last_week"
    if re.search(r"\blast month\b", ql):
        return "last_month"
    if re.search(r"\blast year\b", ql):
        return "last_year"
    if re.search(r"\bpast (?:few |several |couple of )?months?\b", ql):
        return "past_months"
    if re.search(r"\bpast (?:few |several )?weeks?\b", ql):
        return "past_weeks"
    if re.search(r"\bpast year\b", ql):
        return "last_year"
    return None


def _topic_terms(ql: str) -> list[str]:
    """Content words that identify the topic entity of a state question.

    e.g. 'where do i keep my brass compasses' -> ['compasses'].  Used by
    SUPERSEDES to pick the value phrase from a dated observation instead of
    returning the whole raw turn text.
    """
    words = re.findall(r"[a-z]+", ql)
    terms = [w for w in words if len(w) >= 4 and w not in _TOPIC_STOP]
    out: list[str] = []
    seen: set[str] = set()
    for w in terms:
        base = w[:-1] if w.endswith("s") else w
        if base in seen:
            continue
        seen.add(base)
        out.append(w)
    return out


def _is_state_question(ql: str) -> bool:
    # "where are my X" / "where is my X" is inherently a state question even
    # without an explicit state verb in the phrasing.
    if re.search(r"\bwhere\s+(are|is)\s+(my|the)\b", ql):
        return True
    if re.search(r"\bwhere\s+do\b.*\b(my|the)\b", ql) and any(
        v in ql for v in _STATE_VERBS
    ):
        return True
    if any(c in ql for c in _PREVIOUS_CUES):
        return any(v in ql for v in _STATE_VERBS)
    if "now" in ql or "currently" in ql or "these days" in ql:
        return any(v in ql for v in _STATE_VERBS)
    return False
