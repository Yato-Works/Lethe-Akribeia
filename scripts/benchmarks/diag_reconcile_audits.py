"""Reconcile the two audit figures before Oracle Search runs.

Eli's two arithmetic questions, answered by measurement rather than by hand:

  1. annotation audit: 1455 supported + 247 unrelated = 1702, but 'testable' was
     reported as 1709.  What are the missing 7?
  2. retrieval-gap audit: 229 of 233 recovered under the loosest probe implies
     4/233 unrecovered, but the report said 0.2%.  Those have different
     denominators -- reconcile them onto one population.

Both audits must describe the SAME question population, with the SAME rule for
'is the evidence present', before an Oracle experiment is designed on top of
them.
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
from eval_v4_pipeline_dev import V4CandidateRetriever  # noqa: E402

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
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "yesterday",
    "today", "tomorrow", "week", "weeks", "weekend", "month", "months",
    "year", "years", "ago", "before", "after", "last", "next",
})


def content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower())
            if w not in STOP and len(w) > 2 and not w.isdigit()}


def is_temporal_gold(gold: str) -> bool:
    g = str(gold).lower().strip()
    if re.search(r"\b(19|20)\d{2}\b", g):
        return True
    if re.search(r"\b(january|february|march|april|may|june|july|august|"
                 r"september|october|november|december)\b", g):
        return True
    if re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|"
                 r"sunday)\b", g):
        return True
    return bool(re.search(r"^\s*\d+\s*(days?|weeks?|months?|years?)\s*$", g))


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    retriever = V4CandidateRetriever()

    # ---- one population, classified once -------------------------------
    n_total = 0
    n_temporal = 0
    n_no_evidence_field = 0
    n_gold_no_content = 0          # <-- the missing 7
    n_unrelated = 0
    n_related = 0
    related_not_in_pool = 0
    unrelated_not_in_pool = 0

    for conv in data:
        turns, dia_map = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        # cache the pool per question -- retrieval depends only on the question
        for q in E.load_questions(conv):
            n_total += 1
            if is_temporal_gold(q["answer"]):
                n_temporal += 1
                continue

            gold_words = content(q["answer"])
            ev = [e for e in (q.get("evidence") or []) if dia_map.get(e)]
            if not ev:
                n_no_evidence_field += 1
                continue
            if not gold_words:
                # answer is all stopwords/digits ('yes', 'no', '2022'...)
                n_gold_no_content += 1
                continue

            ev_words = set()
            for e in ev:
                ev_words |= content(dia_map[e])
            related = bool(ev_words & gold_words)

            # evidence present in the retrieved pool?  loosest rule (ANY word)
            probes = [" ".join(sorted(content(dia_map[e]))[:5]) for e in ev]
            top, _ = retriever.rank_turns(q["question"], turns, top_k=45)
            pool_words = set()
            for t in top:
                pool_words.update(re.findall(r"[a-z0-9]+", t[3].lower()))
            in_pool = any(w in pool_words for p in probes for w in p.split())

            if related:
                n_related += 1
                if not in_pool:
                    related_not_in_pool += 1
            else:
                n_unrelated += 1
                if not in_pool:
                    unrelated_not_in_pool += 1

    print("=" * 76)
    print("RECONCILED POPULATION -- both audits on one denominator")
    print("=" * 76)
    print(f"total questions                          : {n_total}")
    print(f"  temporal gold (header-resolved)        : {n_temporal}")
    print(f"  no evidence field                      : {n_no_evidence_field}")
    print(f"  gold has no content words (yes/no/...) : {n_gold_no_content}")
    print()
    print("Eli's question 1 -- the missing 7:")
    print(f"  related + unrelated = {n_related} + {n_unrelated} = "
          f"{n_related + n_unrelated}")
    print(f"  testable (total - temporal)            = "
          f"{n_total - n_temporal}")
    print(f"  difference                             = "
          f"{n_total - n_temporal - n_related - n_unrelated}")
    print(f"  accounted for by no-evidence-field ({n_no_evidence_field}) + "
          f"no-content-gold ({n_gold_no_content})")
    print()
    print("Eli's question 2 -- the 0.2% denominator:")
    print(f"  retrieval audit's own population       : 1955 (>=40 chars evidence)")
    print(f"  this run's population                  : {n_related + n_unrelated}")
    print()
    print("EVIDENCE-IN-POOL under the LOOSEST rule (ANY content word):")
    print(f"  related-annotation questions           : {n_related}")
    print(f"    evidence NOT in pool                 : {related_not_in_pool} "
          f"({related_not_in_pool/max(1,n_related)*100:.1f}%)")
    print(f"  unrelated-annotation questions         : {n_unrelated}")
    print(f"    evidence NOT in pool                 : {unrelated_not_in_pool} "
          f"({unrelated_not_in_pool/max(1,n_unrelated)*100:.1f}%)")
    print()
    both = n_related + n_unrelated
    tot_missing = related_not_in_pool + unrelated_not_in_pool
    print(f"  combined                               : {tot_missing}/{both} "
          f"({tot_missing/max(1,both)*100:.1f}%)")
    print()
    print("=" * 76)
    print("WHAT ORACLE SEARCH SHOULD BE RUN ON")
    print("=" * 76)
    print(f"Valid retrieval-test population = questions whose annotation actually")
    print(f"supports the answer = {n_related} questions.")
    print(f"  of those, evidence missing from the pool = {related_not_in_pool} "
          f"({related_not_in_pool/max(1,n_related)*100:.1f}%)")
    print(f"The {n_unrelated} unrelated-annotation questions are excluded: forcing")
    print("their 'gold evidence' into the pool would test the reader's response to")
    print("irrelevant text, not retrieval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
