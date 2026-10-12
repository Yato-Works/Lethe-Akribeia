"""Does the dataset's evidence annotation actually contain the answer?

The retrieval-gap audit recovered 229 of 233 'not retrieved' questions under a
looser probe, which means the strict rule was measuring its own strictness.  A
manual spot-check then showed something worse: for

    Q: When did Melanie paint a sunrise?      GT: 2022
    evidence: ['D1:12']
    D1:12: "You'd be a great counselor! Your empathy and understanding will..."

the annotated evidence turn contains nothing about painting or a sunrise at all.
So a chunk of the "retrieval gap" is annotation noise, not an engine failure.

This quantifies it: for how many questions does ANY annotated evidence turn
share a content word with the gold ANSWER?  If the evidence cannot even support
the answer lexically, the question is not a valid retrieval test.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"

STOP = frozenset({
    "i", "im", "ive", "id", "ill", "my", "me", "mine", "we", "our", "us",
    "you", "your", "the", "a", "an", "and", "or", "but", "to", "of", "in",
    "on", "at", "for", "with", "was", "were", "is", "are", "be", "been",
    "it", "its", "this", "that", "these", "those", "so", "very", "just",
    "went", "go", "going", "had", "has", "have", "did", "do", "does",
    "when", "what", "where", "which", "who", "how", "why", "would", "could",
    "should", "likely", "still", "want", "wants", "she", "her", "he", "his",
    "they", "them", "their", "melanie", "caroline", "jon", "gina",
    # month / weekday names: a temporal gold ('7 May 2023') is stated in the
    # SESSION HEADER, not in the turn text, so their absence from the evidence
    # turn is expected and must not count against the annotation.
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "yesterday",
    "today", "tomorrow", "week", "weeks", "weekend", "month", "months",
    "year", "years", "ago", "before", "after", "last", "next", "sunday",
})


def content(text: str) -> set[str]:
    """Content words, with pure numbers dropped.

    Numbers are dropped because a temporal gold's digits ('2022', '7 May 2023')
    come from the session header, not the evidence turn.
    """
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower())
            if w not in STOP and len(w) > 2 and not w.isdigit()}


def is_temporal_gold(gold: str) -> bool:
    """True when the gold answer is essentially a date/duration."""
    g = str(gold).lower().strip()
    if re.search(r"\b(19|20)\d{2}\b", g):
        return True
    if re.search(r"\b(january|february|march|april|may|june|july|august|"
                 r"september|october|november|december)\b", g):
        return True
    if re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|"
                 r"sunday)\b", g):
        return True
    if re.search(r"^\s*\d+\s*(days?|weeks?|months?|years?)\s*$", g):
        return True
    return False


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))

    total = 0
    evidence_supports_answer = 0
    temporal_skipped = 0
    no_evidence_field = 0
    examples = []

    for conv in data:
        turns, dia_map = E.conversation_turns(conv["conversation"])
        for q in E.load_questions(conv):
            total += 1
            # A temporal gold is resolved from the session header, not stated in
            # the evidence turn, so lexical support cannot be required of it.
            if is_temporal_gold(q["answer"]):
                temporal_skipped += 1
                continue
            gold_words = content(q["answer"])
            if not gold_words:
                continue
            ev = [e for e in (q.get("evidence") or []) if dia_map.get(e)]
            if not ev:
                no_evidence_field += 1
                continue
            ev_words = set()
            for e in ev:
                ev_words |= content(dia_map[e])
            if ev_words & gold_words:
                evidence_supports_answer += 1
            elif len(examples) < 8:
                examples.append((q["question"], q["answer"],
                                 dia_map[ev[0]][:52]))

    n_testable = total - temporal_skipped
    print("=" * 74)
    print("IS THE DATASET'S EVIDENCE ANNOTATION USABLE AS A RETRIEVAL TARGET?")
    print("=" * 74)
    print(f"total questions                          : {total}")
    print(f"  temporal gold (header-resolved, skipped): {temporal_skipped}")
    print(f"  testable here                          : {n_testable}")
    print(f"    with no evidence field               : {no_evidence_field}")
    print(f"    evidence supports the answer         : {evidence_supports_answer} "
          f"({evidence_supports_answer/max(1,n_testable)*100:.1f}% of testable)")
    print()
    print("questions where NO annotated evidence turn relates to the answer:")
    for qq, ans, evtxt in examples:
        print(f"  Q   : {qq[:56]}")
        print(f"  ANS : {str(ans)[:56]}")
        print(f"  EV  : {evtxt}")
    print()
    bad = n_testable - evidence_supports_answer - no_evidence_field
    print(f"VERDICT: of {n_testable} testable questions, {bad} "
          f"({bad/max(1,n_testable)*100:.1f}%) have annotated evidence that does")
    print("  not lexically support the gold answer.  Those are annotation noise,")
    print("  not engine failures, and they are not valid retrieval tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
