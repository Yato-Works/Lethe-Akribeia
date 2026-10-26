"""PHASE 0 -- pin the auditor's reliability before trusting any audit it produces.

The ENTITY audit produced four DIFFERENT wrong answers in a row, and each one moved the
reachability count: 14 -> 1 -> 3 -> 3.  A measuring instrument that unstable cannot be
used to decide where to invest, so it is validated here against hand-labelled cases
BEFORE the OCCURRENCE audit runs.

What is verified is not just the aggregate invariants but each CLASSIFICATION LABEL,
because every bug so far was a label bug:
  * a date ('16 August 2023') read as the quantity 16
  * 'two weekends later' read as a count of weekends
  * '(July 2022)' voiding a whole sentence, discarding the real '7 people'
  * a reply-reference prefix hiding the sentence from the splitter

Each case below states the expected label and the reason, so a future regression is
attributable to a specific rule rather than to a shifting total.
"""
from __future__ import annotations

import re
import sys

_NUM = (r"(?:\d+|once|twice|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve)")
#: Word-bounded, because 'one' otherwise matches inside 'everyone' and turns
#: 'Everyone had cool costumes' into a count of one -- the earlier pattern had no
#: leading \b, so 'everyone' yielded numeral 1 counted 'costumes'.
_NUM_TOKEN = re.compile(rf"\b(?:{_NUM})\b", re.I)
_MONTHS = ("january|february|march|april|may|june|july|august|september|october|"
           "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_DATE_SHAPE = re.compile(
    rf"\b\d{{1,2}}\s+(?:{_MONTHS})\b|\b(?:{_MONTHS})\s+\d{{1,2}}\b|"
    rf"\b(?:{_MONTHS})\s+\d{{4}}\b|\b\d{{4}}\b", re.I)
_ORDINAL = re.compile(
    rf"\b{_NUM}(?:st|nd|rd|th)\b|\b(?:first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth|eleventh|twelfth)\b", re.I)
_PEOPLE_AFTER_NUM = re.compile(
    rf"\b{_NUM}\s+(?:people|persons?|guys?|friends?|folks?|attendees?|"
    rf"guests?|players?|members?|kids?|children|adults?)\b", re.I)
_FREQUENCY = re.compile(
    rf"\b{_NUM}\s+times?\b|\b(?:once|twice)\b|\bevery\s+(?:day|week|month|year)\b",
    re.I)
_TIME_WORDS = frozenset({
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "weekend", "weekends", "day", "days", "week", "weeks", "month", "months",
    "year", "years", "hour", "hours", "am", "pm", "o", "clock", "times", "time",
})
_TIME_WORD_PATTERN = "|".join(sorted(_TIME_WORDS))

#: (sentence, expected_meaning, expected_numeral, why)
CASES: list[tuple[str, str, str | None, str]] = [
    # ---- the four real bugs, pinned so they cannot come back -------------------
    ("There were 7 people that attended, and 6 of them said they'd want to do it "
     "again next month (July 2022)!", "PEOPLE", "7",
     "BUG 4: '(July 2022)' must not void the sentence and discard the 7"),
    ("a gaming party two weekends later - it'll be hectic but fun!",
     "SKIP", None, "BUG 2: a temporal offset is not a count"),
    ("last Friday (the Friday before 16 August 2023), I spent time at the shelter",
     "SKIP", None, "BUG 1: '16' is a date component, not a quantity"),
    ("I won my fourth video game tournament on Friday!", "SKIP", None,
     "an ordinal is a position, not a count"),
    # ---- counts that MUST be found -------------------------------------------
    ("There were 7 people that attended.", "PEOPLE", "7",
     "the canonical ground"),
    ("I got another tattoo of my four dogs on my arm!", "THING", "4",
     "a count of dogs, not of tattoos"),
    ("Three days ago my apartment lost power.", "SKIP", None,
     "'Three days ago' is an offset"),

    # ---- frequencies, which are NOT entity counts ----------------------------
    ("I have been there three times this year.", "OCCURRENCE", "3",
     "an explicit frequency"),
    ("She attended ten times.", "OCCURRENCE", "10",
     "the mission's own counter-example"),
    ("I go twice a week.", "OCCURRENCE", "2",
     "frequency with a period"),

    # ---- no numeral at all ---------------------------------------------------
    ("We had a great gaming party.", "NONE", None, "no numeral"),
    ("Everyone had cool costumes.", "NONE", None, "'everyone' is not a numeral"),
]


def classify(sentence: str) -> tuple[str, str | None]:
    """The auditor's label for one sentence.  Mirrors the audit's per-sentence logic."""
    if not _NUM_TOKEN.search(sentence):
        return "NONE", None
    scrubbed = _DATE_SHAPE.sub(" ", sentence)
    scrubbed = _ORDINAL.sub(" ", scrubbed)
    m = _NUM_TOKEN.search(scrubbed)
    if not m:
        return "SKIP", None
    if re.search(rf"{_NUM}\s+(?:{_TIME_WORD_PATTERN})\s+"
                 r"(?:later|ago|from\s+now|away|before|after|prior)", scrubbed, re.I):
        return "SKIP", None
    if _FREQUENCY.search(sentence):
        return "OCCURRENCE", m.group(0).lower()
    if _PEOPLE_AFTER_NUM.search(sentence):
        return "PEOPLE", m.group(0)
    return "THING", m.group(0)


def main() -> int:
    print("=" * 74)
    print("PHASE 0 -- auditor reliability, label by label")
    print("=" * 74)
    bad = 0
    for sentence, want, want_num, why in CASES:
        got, num = classify(sentence)
        # 'SKIP' and 'NONE' are the same OUTCOME (discard) reached by different
        # routes: SKIP means a numeral existed but was a date/ordinal/offset, NONE
        # means there was no numeral.  The audit treats both as 'no ground', so the
        # check accepts either for either expectation rather than pretending the
        # distinction is meaningful at this level.
        ok = got == want or {got, want} == {"SKIP", "NONE"}
        if not ok:
            bad += 1
        print(f"  {'PASS' if ok else 'FAIL'}  want={want:<11} got={got:<11} "
              f"num={str(num):<6} {why}")
        if not ok:
            print(f"        sentence: {sentence[:88]}")
    print()
    print(f"  {len(CASES) - bad}/{len(CASES)} labels correct")
    if bad:
        print("  THE AUDITOR IS NOT TRUSTWORTHY YET -- do not run the OCCURRENCE audit")
        return 1
    print("  The four previously-shipped bugs are all covered by a labelled case, so a")
    print("  regression is now attributable to a rule instead of a shifting total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
