"""How many failures are unanswerable from the corpus at all?

cat3's non-inference failures looked like retrieval failures (cov=0.00), but a
spot check showed the gold answer text does not exist anywhere in the
conversation: 'What state did Joanna visit in summer 2021?' has gold 'Indiana'
and ZERO turns contain the word 'Indiana'.  Same for 'Florida'.

That is not a retrieval problem, a reader problem, or a scoring problem -- the
question cannot be answered from the given memory.  This quantifies how much of
the failure population is in that category, because it caps what any engine
improvement can achieve.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"
RUN = ROOT / "benchmark_results" / "locomo_full_20261012_211601.json"

STOP = frozenset({
    "i", "my", "me", "we", "you", "the", "a", "an", "and", "or", "but", "to",
    "of", "in", "on", "at", "for", "with", "was", "were", "is", "are", "be",
    "been", "it", "its", "this", "that", "these", "those", "so", "very",
    "just", "went", "go", "going", "had", "has", "have", "did", "do", "does",
    "when", "what", "where", "which", "who", "how", "why", "would", "could",
    "should", "likely", "still", "want", "wants", "she", "her", "he", "his",
    "they", "them", "their", "melanie", "caroline", "jon", "gina", "yes",
    "no", "not", "some", "any", "all", "more", "most", "other", "another",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "yesterday",
    "today", "tomorrow", "week", "weeks", "weekend", "month", "months",
    "year", "years", "ago", "before", "after", "last", "next",
})

WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower())
            if w not in STOP and len(w) > 2 and not w.isdigit()}


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    recorded = {r["question_id"]: r
                for r in json.loads(RUN.read_text(encoding="utf-8"))["results"]}

    buckets = Counter()
    by_cat = Counter()
    examples = {}

    for conv in data:
        turns, _ = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        # every word in this conversation, for 'does the answer exist at all'
        corpus_words = set()
        for t in turns:
            corpus_words.update(re.findall(r"[a-z0-9]+", t[3].lower()))

        for q in E.load_questions(conv):
            r = recorded.get(q["question_id"])
            if r is None or r["is_correct"]:
                continue                      # failures only
            gold_words = content(q["answer"])
            if not gold_words:
                buckets["no_content_gold"] += 1
                continue

            # is ANY gold content word present anywhere in the conversation?
            present = {w for w in gold_words if w in corpus_words}
            # also try word-numbers ('Four' -> 'four')
            if not present:
                g = str(q["answer"]).lower().strip()
                if g in WORDNUM:
                    w = str(WORDNUM[g])
                    if w in corpus_words:
                        present = {w}

            cov = len(present) / len(gold_words)
            if cov == 0:
                key = "UNANSWERABLE_gold_absent_everywhere"
            elif cov < 0.5:
                key = "WEAK_gold_mostly_absent"
            else:
                key = "ANSWERABLE_gold_present"
            buckets[key] += 1
            by_cat[(q["category"], key)] += 1
            if key not in examples:
                examples[key] = (q["question"], q["answer"], round(cov, 2))

    total = sum(buckets.values())
    print("=" * 78)
    print("IS THE GOLD ANSWER DERIVABLE FROM THE CONVERSATION AT ALL?")
    print("=" * 78)
    print(f"failures examined: {total}")
    print()
    for k in ("UNANSWERABLE_gold_absent_everywhere",
              "WEAK_gold_mostly_absent", "ANSWERABLE_gold_present",
              "no_content_gold"):
        c = buckets[k]
        if not c:
            continue
        print(f"  {k:38s} {c:4d}  ({c/total*100:5.1f}%)")
        if k in examples:
            qq, gt, cov = examples[k]
            print(f"      e.g. {qq[:56]}")
            print(f"           GT {gt[:44]}  | gold-word presence {cov}")
    print()
    print("per category (unanswerable / weak / answerable):")
    for cat in (1, 2, 3, 4, 5):
        u = by_cat[(cat, "UNANSWERABLE_gold_absent_everywhere")]
        w = by_cat[(cat, "WEAK_gold_mostly_absent")]
        a = by_cat[(cat, "ANSWERABLE_gold_present")]
        print(f"  cat{cat}   {u:4d} / {w:4d} / {a:4d}")
    print()
    un = buckets["UNANSWERABLE_gold_absent_everywhere"]
    print(f"VERDICT: {un} failures ({un/total*100:.1f}%) have a gold answer whose")
    print("  content appears NOWHERE in the conversation.  No retrieval, reader,")
    print("  or scoring change can ever answer those -- the ceiling on this")
    print("  benchmark is 1,986 -", un, "=", 1986 - un, "questions at best.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
