"""Is the abstain caused by the event gate, and how many of the 33 does that explain?

The map's 33 actionable abstentions all report 'fallback to LLM reader'.  Tracing one
(lc_conv-48_66) showed the real mechanism:

    Q keywords : ['home','janeiro','jolene','partner','return','rio']
    turn words : [... 'back', 'got', ...]
    event match: False
    -> resolve_date returns before any date branch runs

_date_event_matches requires EVERY non-person keyword to appear in the turn window.
'return home' is written as 'got back', a paraphrase, so the gate rejects a turn that
states the answer explicitly in parentheses.  This is a GATE defect, not a
date-resolution defect -- which is why the rejected branch's whole framing was wrong:
it tried to change date precedence when the engine never reached a date branch.

This measures, across all 33 actionable abstentions:
  * how many fail the gate,
  * of those, how many have an explicit date in the gated turn,
  * and the specific missing keywords, so a fix can be sized rather than guessed.

No fix is proposed here.  This decides whether a fix is worth proposing, and on how
many cases it could possibly land.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory_gen.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory_gen.skills.autonomous_engines.temporal_algebra_engine import (  # noqa: E402
    TemporalAlgebraEngine,
    _date_event_matches,
    LEMMA_MAP,
    _EVENT_TERM_ALIASES,
    _NON_EVENT_DATE_TERMS,
)

MAP = ROOT / "benchmark_results/when_date_map/when_date_failure_map.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"

EXPLICIT_DATE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+20\d\d\b")


def main() -> int:
    data = json.loads(MAP.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    actionable = [r for r in data["records"]
                  if r["engine_class"] == "ENGINE_ABSTAIN"
                  and r["gold_in_context"]]
    print(f"actionable abstentions (gold in context) : {len(actionable)}")
    print()

    gate_failed = 0
    gate_failed_with_explicit_date = 0
    missing_kw = Counter()
    details: list[str] = []

    for r in actionable:
        ctx = (cache.get(r["cache_qid"]) or {}).get("context") or ""
        if not ctx:
            continue
        turns = parse_turns(ctx)
        if not turns:
            continue
        person = TemporalAlgebraEngine._extract_person(r["question"], turns)
        q_kws = TemporalAlgebraEngine._extract_keywords(r["question"])
        event_terms = q_kws - _NON_EVENT_DATE_TERMS - (
            {person.lower()} if person else set())

        # Does ANY turn pass the gate?
        any_pass = any(_date_event_matches(r["question"], turns, i, person)
                       for i in range(len(turns)))
        if any_pass:
            continue
        gate_failed += 1

        # Which keywords are missing everywhere?
        all_words: set[str] = set()
        for t in turns:
            all_words |= set(re.findall(r"\b[a-zA-Z]{2,}\b", t.text.lower()))
        for term in event_terms:
            variants = LEMMA_MAP.get(term, {term}) | _EVENT_TERM_ALIASES.get(term, set())
            if not (variants & all_words):
                missing_kw[term] += 1

        # Does a turn carry an explicit full date at all?
        if any(EXPLICIT_DATE.search(t.text) for t in turns):
            gate_failed_with_explicit_date += 1
            if len(details) < 8:
                missing = sorted(
                    term for term in event_terms
                    if not ((LEMMA_MAP.get(term, {term})
                             | _EVENT_TERM_ALIASES.get(term, set())) & all_words))
                details.append(
                    f"  {r['question_id']:<18} gold={str(r['gold'])[:20]!r:<22} "
                    f"missing={missing}")

    print(f"of those, the event gate rejects EVERY turn : {gate_failed}")
    print(f"  and a turn carries an explicit full date  : "
          f"{gate_failed_with_explicit_date}")
    print()
    print("MOST-MISSING KEYWORDS (a lemma/alias gap in the gate)")
    for kw, n in missing_kw.most_common(12):
        print(f"  {n:>3}  {kw}")
    print()
    print("CASES: gate fails AND the date is written in a turn")
    for line in details:
        print(line)
    print()
    print("HOW TO READ THIS")
    print(f"  A high {gate_failed} means the abstentions are a GATE problem, not a")
    print("  date problem: the engine is rejecting turns that state the answer.  The")
    print("  fix is to widen the gate's paraphrase coverage, which is a small, local")
    print("  change -- and unlike the rejected branch, it does not touch date")
    print("  precedence at all.")
    print()
    print("  The 'missing' keywords are the exact vocabulary a fix would have to cover.")
    print("  If that vocabulary is small and regular, the fix is tractable.  If it is")
    print("  open-ended paraphrase, it is the same model dependency found for")
    print("  extraction, and no regex will close it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
