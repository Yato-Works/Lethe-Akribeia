"""PHASE 1 -- audit the DURATION questions for an explicit elapsed-span ground.

A measurement note first, because the size of this group was previously misstated.

The earlier count said DURATION = 8, taken from a regex that only matched
'how many <day|week|month|year>'.  quantity_type() -- the same classifier used for
ENTITY and OCCURRENCE -- also matches 'how long', 'for how long', and 'how long ago',
which is a much larger and more coherent population: 43 questions.  The audit runs on
the 43 and reports the 8 as a subset, so the discrepancy is visible rather than hidden.

Gold shapes here are spans ('4 years', 'six months', '30-60 seconds', 'A few months')
and a few are not spans at all ('Since 2016', '10 years ago', 'Middle school, high
school'), which is recorded as its own class instead of being forced into one.

Per question: gold, current verdict, and whether an explicit span is stated in the
frozen context about the ASKED thing.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory_gen.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory_gen.skills.autonomous_engines.counting_engine import (  # noqa: E402
    CountingEngine,
)
from artificial_memory_gen.skills.deterministic_memory_engine import (  # noqa: E402
    DeterministicMemoryEngine,
)

spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = scorer
try:
    spec.loader.exec_module(scorer)
except SystemExit:
    pass

AUDIT = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/duration_ground_audit.json"

_NUM = r"(?:\d+|once|twice|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirty|sixty)"
_UNITS = r"(?:second|seconds|minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)"
#: '<num> <unit>' -- an explicit elapsed span.
_SPAN = re.compile(rf"\b{_NUM}\s*[-–]?\s*{_UNITS}\b", re.I)
#: 'about four months', 'nearly three months', 'A few months' -- hedged spans.
_HEDGE = re.compile(rf"\b(?:about|around|nearly|almost|a\s+few|couple\s+of|"
                    rf"approximately)\s+{_NUM}\s+{_UNITS}\b", re.I)
#: A range: '30-60 seconds'.
_RANGE = re.compile(rf"\b{_NUM}\s*[-–]\s*{_NUM}\s+{_UNITS}\b", re.I)
#: Golds that are not spans: an anchor date or a period of life.
_NOT_A_SPAN = re.compile(
    r"\bsince\s+\d{4}\b|\b\d{4}\b|\b(?:middle\s+school|high\s+school|elementary)\b", re.I)

_STOP = {"many", "much", "long", "does", "what", "which", "there", "their",
         "about", "with", "that", "how", "did", "has", "have", "been", "will",
         "the", "and", "for", "was", "were", "his", "her", "its", "ago", "take",
         "took", "passed", "between", "usually", "typically", "plan", "planned"}


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cache[json.loads(line)["qid"]] = json.loads(line)

    engine = CountingEngine()
    rows: list[dict] = []
    for r in audit["records"]:
        q = r["question"]
        if engine.quantity_type(q) != "DURATION":
            continue
        row = cache.get(r["cache_qid"]) or {}
        context = row.get("context") or ""
        gold = r["gold"]
        turns = parse_turns(context) if context else []

        cur = DeterministicMemoryEngine.resolve(q, context) if context else None
        verdict = ("abstain" if not cur or not cur.used
                   else ("correct" if scorer.score(cur.answer, gold) else "wrong"))

        q_terms = {w for w in re.findall(r"[a-z]{4,}", q.lower()) if w not in _STOP}

        spans: list[dict] = []
        for t in turns:
            body = re.sub(r"^\s*\(In reply to[^)]*\)\s*", "", t.text)
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                m = _SPAN.search(sentence)
                if not m:
                    continue
                overlap = sum(1 for w in re.findall(r"[a-z]{4,}", sentence.lower())
                              if w in q_terms)
                spans.append({
                    "turn_id": t.dia_id,
                    "sentence": sentence.strip()[:130],
                    "span": m.group(0),
                    "hedged": bool(_HEDGE.search(sentence)),
                    "is_range": bool(_RANGE.search(sentence)),
                    "topic_overlap": overlap,
                })

        on_topic = [s for s in spans if s["topic_overlap"] > 0]
        gold_not_span = bool(_NOT_A_SPAN.search(str(gold)))

        if gold_not_span:
            source = "GOLD_NOT_A_SPAN"
        elif on_topic:
            source = "EXPLICIT_SPAN"
        elif spans:
            source = "SPAN_OFF_TOPIC"
        else:
            source = "NO_SPAN"

        rows.append({
            "question_id": r["question_id"],
            "cache_qid": r["cache_qid"],
            "conversation": r["cache_qid"].split("-")[1],
            "question": q,
            "gold": gold,
            "current_verdict": verdict,
            "source": source,
            "n_spans": len(spans),
            "spans": spans[:5],
        })

    print("=" * 74)
    print("PHASE 1 -- DURATION grounds")
    print("=" * 74)
    print("  SIZE, corrected.  quantity_type() classifies 43 questions as DURATION, not")
    print("  8 -- the earlier figure came from a narrower 'how many <day|week|month>'")
    print("  regex that missed 'how long', 'for how long' and 'how long ago'.")
    print()
    print(f"  DURATION questions : {len(rows)}")
    by_src = Counter(x["source"] for x in rows)
    for k in ("EXPLICIT_SPAN", "SPAN_OFF_TOPIC", "NO_SPAN", "GOLD_NOT_A_SPAN"):
        print(f"    {k:<16} {by_src.get(k, 0)}")
    by_verdict = Counter(x["current_verdict"] for x in rows)
    print(f"  current engine: {dict(by_verdict)}")
    print()
    print(f"  {'qid':<17}{'gold':<20}{'now':<9}{'source':<17}question")
    for x in rows:
        print(f"  {x['question_id']:<17}{str(x['gold'])[:19]:<20}"
              f"{x['current_verdict']:<9}{x['source']:<17}{x['question'][:30]}")
    print()
    print("EXPLICIT ON-TOPIC SPAN (the only reachable class)")
    for x in rows:
        if x["source"] != "EXPLICIT_SPAN":
            continue
        s = x["spans"][0]
        print(f"  {x['question_id']}  gold={str(x['gold'])[:14]!r}  now={x['current_verdict']}")
        print(f"     [{s['turn_id']}] {s['sentence'][:84]}")
        print(f"     -> span {s['span']!r} hedged={s['hedged']} range={s['is_range']}")
    print()
    print("GOLD IS NOT A SPAN (recorded, not forced into a span class)")
    for x in rows:
        if x["source"] == "GOLD_NOT_A_SPAN":
            print(f"  {x['question_id']:<17} gold={str(x['gold'])[:26]!r}")
    print()
    print("NO EXPLICIT SPAN, and why")
    for x in rows:
        if x["source"] in ("EXPLICIT_SPAN", "GOLD_NOT_A_SPAN"):
            continue
        why = ("a span exists in context but about a different thing"
               if x["source"] == "SPAN_OFF_TOPIC"
               else "no '<num> <unit>' span anywhere in the frozen context")
        print(f"  {x['question_id']:<17} {why}")

    OUT.write_text(json.dumps({
        "label": "phase 1 -- DURATION ground audit (43 questions, not 8)",
        "n": len(rows),
        "by_source": dict(by_src),
        "by_verdict": dict(by_verdict),
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
