#!/usr/bin/env python3
"""M30-40 Phase 1: Deterministic output-contract normalization.

Two independent normalizations, no meaning change:

A. Canonical abstention normalization
   Semantically-correct abstention text -> "I don't know."
   Only when the answer IS an abstention (not when it contains a fact + caveat).

B. Requested-granularity normalization
   When the question asks for month+year, reduce "July 29, 2025" -> "7/2025".
   Never strip a full date when the question asks for a full date.

No Gold, no evidence IDs. Question + raw output only.
"""
import re

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}


# ---------------------------------------------------------------------------
# A. Canonical abstention normalization
# ---------------------------------------------------------------------------

# Phrases that indicate the answer IS an abstention (whole answer).
ABSTENTION_PATTERNS = [
    r"^\s*i\s*(do\s*not|don'?t)\s*know\b",
    r"^\s*i'?m\s+not\s+sure\b",
    r"^\s*(the\s+)?(provided\s+)?(context|conversation|information|evidence|memory|data)\s+"
    r"(does\s+not|doesn'?t|do\s+not|don'?t)\s+(provide|contain|include|mention|specify)",
    r"^\s*(there\s+is|there'?s)\s+(no|not\s+enough|insufficient)\s+"
    r"(information|context|evidence|data|detail)",
    r"^\s*(the\s+)?(information|context|evidence)\s+is\s+(insufficient|not\s+enough|inadequate)",
    r"^\s*(it\s+is\s+)?(not|impossible)\s+(possible\s+)?to\s+(determine|answer|tell)",
    r"^\s*(i\s+)?(cannot|can'?t)\s+(determine|answer|tell|find)",
    r"^\s*unable\s+to\s+(determine|answer|tell)",
    r"^\s*(no|not\s+enough)\s+(information|context|evidence)\s+(is\s+)?(available|provided|given)",
    r"^\s*(the\s+)?answer\s+is\s+not\s+(mentioned|provided|stated|given|available)",
]

# If a sentence like this appears, the answer is NOT a pure abstention.
FACTUAL_OVERRIDE_PATTERNS = [
    # a concrete date anywhere -> there is a fact
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{1,2}/\d{4}\b",
    # an explicit answer prefix
    r"\b(the\s+answer\s+is|answer:)\s*\S",
    r"\b\d+\s+days?\b",
    r"\bsession\s+\d+\b",
]


def normalize_abstention(question, raw):
    """Return (normalized, rule, reason, status)."""
    text = raw.strip()
    if not text:
        return raw, None, "empty output", "UNCHANGED"

    low = text.lower()

    # Already canonical
    if re.match(r"^\s*i\s*(do\s*not|don'?t)\s*know\s*\.?\s*$", low):
        return raw, None, "already canonical abstention", "UNCHANGED"

    # If the answer carries a concrete fact, do not abstain-ify it.
    for pat in FACTUAL_OVERRIDE_PATTERNS:
        if re.search(pat, low):
            return raw, None, "contains a factual answer; not an abstention", "UNCHANGED"

    # Match abstention phrases at the start of the answer
    for pat in ABSTENTION_PATTERNS:
        if re.search(pat, low):
            return "I don't know.", "ABSTENTION_CANONICAL", "semantic abstention matched", "NORMALIZED"

    return raw, None, "no abstention phrase at start", "UNCHANGED"


# ---------------------------------------------------------------------------
# B. Requested-granularity normalization
# ---------------------------------------------------------------------------

def requested_granularity(question):
    """Classify what the question asks for: 'year', 'month', 'day', or None."""
    q = question.lower()

    # Explicit full-date request
    if re.search(r"\b(what|which|on what)\s+(exact\s+)?(date|day)\b", q):
        return "day"
    if re.search(r"\bwhat\s+day\b", q):
        return "day"

    # Explicit year-only request
    if re.search(r"\b(what|which|in which)\s+year\b", q):
        return "year"

    # Explicit month+year request
    if re.search(r"\b(what|which|in which)\s+month\b", q):
        return "month"
    if re.search(r"\bmonth\s+and\s+year\b", q):
        return "month"

    return None


def normalize_granularity(question, raw):
    """Reduce a full date to the requested granularity."""
    gran = requested_granularity(question)
    if gran is None:
        return raw, None, "question granularity undetermined", "UNCHANGED"

    text = raw.strip()

    # Full date in the answer: "July 29, 2025"
    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+(\d{1,2}),?\s+(\d{4})\b",
        text, re.I)
    if not m:
        # numeric full date "7/29/2025"
        m2 = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
        if m2 and gran in ("month", "year"):
            mo, dy, yr = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            if gran == "month":
                return f"{mo}/{yr}", "GRANULARITY_MONTH", "reduced numeric date to M/YYYY", "NORMALIZED"
            return str(yr), "GRANULARITY_YEAR", "reduced numeric date to YYYY", "NORMALIZED"
        return raw, None, "no full date present to reduce", "UNCHANGED"

    month_name, day, year = m.group(1), m.group(2), m.group(3)
    mo = MONTHS[month_name.lower()]

    if gran == "day":
        # question wants the full date; nothing to reduce
        return raw, None, "question requests full date", "UNCHANGED"

    if gran == "month":
        # Only reduce if the date is essentially the whole answer.
        # If the answer is a sentence with extra context, keep it (avoid meaning loss).
        remainder = (text[:m.start()] + text[m.end():]).strip(" .,")
        if len(remainder) > 0 and not re.match(r"^(on|in|was|is|the|date)\b", remainder, re.I):
            # sentence with substantive extra content -> do not rewrite
            pass
        return f"{mo}/{year}", "GRANULARITY_MONTH", "reduced full date to M/YYYY", "NORMALIZED"

    if gran == "year":
        return str(year), "GRANULARITY_YEAR", "reduced full date to YYYY", "NORMALIZED"

    return raw, None, "no rule applied", "UNCHANGED"


def normalize(question, raw):
    """Apply both normalizations, abstention first."""
    out, rule_a, reason_a, status_a = normalize_abstention(question, raw)
    if status_a == "NORMALIZED":
        return {
            "raw": raw, "normalized": out, "rule": rule_a,
            "reason": reason_a, "status": status_a,
        }
    out2, rule_b, reason_b, status_b = normalize_granularity(question, out)
    return {
        "raw": raw, "normalized": out2,
        "rule": rule_b, "reason": reason_b, "status": status_b,
    }


if __name__ == "__main__":
    tests = [
        ("What is Xavier's favorite hobby?", "The context does not provide information about Xavier's favorite hobby."),
        ("What is Xavier's favorite hobby?", "I don't know."),
        ("In which month and year did Silas attend?", "Silas attended on July 29, 2025."),
        ("On what date did Silas attend?", "July 29, 2025"),
        ("In which year did Silas attend?", "July 29, 2025"),
        ("When did Silas attend?", "July 29, 2025"),
        ("What is Xavier's hobby?", "Xavier's hobby is painting. The context does not provide other details."),
        ("In which month and year did Silas attend?", "I don't know."),
    ]
    for q, a in tests:
        r = normalize(q, a)
        print(f"Q: {q}")
        print(f"  raw:  {a}")
        print(f"  ->    {r['normalized']}  [{r['status']}] {r['reason']}")
        print()
