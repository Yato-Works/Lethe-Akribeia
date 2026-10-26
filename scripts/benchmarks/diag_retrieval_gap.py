"""Audit the '21.1% not retrieved' figure for false negatives before trusting it.

Eli's question, exactly: was that number measured, or inferred from failed
search experiments?  It was MEASURED -- 100% minus the 78.9% of questions whose
gold content words appear in the top-45 retrieved pool.  But a measured number
can still be wrong if the probe is too strict, and 21.1% is about to drive a
whole research direction, so it gets audited.

Method: for the questions currently counted as NOT retrieved, re-check with
progressively LOOSER probes:
  L0  current rule   >= max(2, len(probe)-1) of 5 content words
  L1  >= 2 of 5
  L2  >= 1 of 5
  L3  ANY gold content word present anywhere in the pool

If loosening recovers a large share, 21.1% is overstated and the real
not-retrieved figure is smaller.
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
})


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower())
            if w not in STOP and len(w) > 2]


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    retriever = V4CandidateRetriever()

    n_q = 0
    n_not_retrieved = 0
    recovered = {"L1": 0, "L2": 0, "L3": 0}
    examples = []

    for conv in data:
        turns, dia_map = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        for q in E.load_questions(conv):
            ev = [e for e in (q.get("evidence") or []) if dia_map.get(e)]
            # gold content words, capped at 5 per the existing probe
            probes = [" ".join(content_words(dia_map[e])[:5]) for e in ev
                      if len(dia_map[e]) >= 40 and content_words(dia_map[e])]
            if not probes:
                continue
            n_q += 1

            top, _ = retriever.rank_turns(q["question"], turns, top_k=45)
            pool_words = set()
            for t in top:
                pool_words.update(re.findall(r"[a-z0-9]+", t[3].lower()))

            def hits_needed(p: str, k: int) -> bool:
                return sum(1 for w in p.split() if w in pool_words) >= k

            # L0: the rule that produced the 78.9% figure
            l0 = any(hits_needed(p, max(2, len(p.split()) - 1)) for p in probes)
            if l0:
                continue
            n_not_retrieved += 1

            # progressively looser
            if any(hits_needed(p, 2) for p in probes):
                recovered["L1"] += 1
            if any(hits_needed(p, 1) for p in probes):
                recovered["L2"] += 1
                if len(examples) < 5:
                    examples.append((q["question"], probes[0][:44],
                                     sum(1 for w in probes[0].split() if w in pool_words),
                                     len(probes[0].split())))
            if any(w in pool_words for p in probes for w in p.split()):
                recovered["L3"] += 1

    print("=" * 74)
    print("AUDIT: is the 21.1% 'not retrieved' figure overstated?")
    print("=" * 74)
    print(f"questions with identifiable gold evidence : {n_q}")
    print(f"counted as NOT retrieved (L0 rule)        : {n_not_retrieved} "
          f"({n_not_retrieved/n_q*100:.1f}%)")
    print()
    print("re-checking those same questions with looser probes:")
    print(f"  L1  >=2 of 5 content words in pool : {recovered['L1']:4d} recovered")
    print(f"  L2  >=1 of 5 content words in pool : {recovered['L2']:4d} recovered")
    print(f"  L3  ANY content word in pool       : {recovered['L3']:4d} recovered")
    print()
    strict_gap = n_not_retrieved / n_q * 100
    loose_gap = (n_not_retrieved - recovered["L3"]) / n_q * 100
    print(f"not-retrieved under the strict rule : {strict_gap:.1f}pt")
    print(f"not-retrieved under the loosest rule: {loose_gap:.1f}pt")
    print(f"  -> the honest range is {loose_gap:.1f}pt to {strict_gap:.1f}pt")
    print()
    if examples:
        print("cases recovered only by L2 (partial overlap, e.g. 1 of 5 words):")
        for qq, p, nh, tot in examples:
            print(f"  Q   : {qq[:56]}")
            print(f"  probe({nh}/{tot} words found): {p}")
    print()
    if recovered["L2"] > n_not_retrieved * 0.2:
        print("VERDICT: the strict rule OVERSTATES the retrieval gap. More than a")
        print("  fifth of 'not retrieved' questions have partial evidence overlap,")
        print("  so this is partly a ranking/coverage problem, not pure absence.")
    else:
        print("VERDICT: the figure is robust. Loosening the probe recovers little,")
        print("  so the evidence really is absent from the candidate pool for")
        print("  those questions, not merely ranked low.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
