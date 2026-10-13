"""Where does the +1 day come from in the 26 off-by-one cases?

The '% 7 or 7' weekday expression is a real latent defect, but its signature is a
7-day jump while the observed signature is +1 day in 26 of 30 cases -- so that
line is exonerated for these and the cause is elsewhere.

Three candidates, each testable from the frozen context:

  A  gold+1 IS stated in the context  -> the engine picked a real date, but from
     the wrong turn.  That is a binding error wearing an arithmetic costume.
  B  gold IS stated verbatim          -> the evidence is right there and the
     engine normalised it wrong.  That is a parsing/formatting error.
  C  neither is stated                -> the engine derived the date by arithmetic
     from some other anchor, and the arithmetic is off by one.

The distinction decides the fix: A needs turn selection, B needs date parsing,
C needs calendar arithmetic.  Building the wrong one is the failure mode this
sprint has been guarding against.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def parse_full(text: str) -> datetime | None:
    t = str(text).lower()
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+),?\s+(\d{4})\b", t)
    if m and m.group(2) in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    m = re.search(r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", t)
    if m and m.group(1) in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
    return None


def stated(context: str, day: int, month: int) -> bool:
    """Is 'D Month' written in the context, ignoring the year position?"""
    name = [k for k, v in MONTHS.items() if v == month][0]
    return bool(re.search(rf"\b{day}\s+{name}\b", context.lower())) or \
        bool(re.search(rf"\b{name}\s+{day}(?:st|nd|rd|th)?\b", context.lower()))


def main() -> int:
    atlas = [json.loads(l) for l in ATLAS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    cases = []
    for r in atlas:
        if r.get("shape") != "when_date":
            continue
        g, p = parse_full(r["gold"]), parse_full(r.get("predicted_answer") or "")
        if g and p and (g.year, g.month) == (p.year, p.month) and \
                (p - g).days == 1:
            cases.append((r, g, p))

    verdict = Counter()
    detail: list[str] = []
    for r, g, p in cases:
        ctx = (cache.get(r.get("cache_qid")) or {}).get("context") or ""
        if not ctx:
            verdict["no_context"] += 1
            continue
        gold_stated = stated(ctx, g.day, g.month)
        pred_stated = stated(ctx, p.day, p.month)
        if pred_stated and not gold_stated:
            verdict["A_wrong_turn_real_date"] += 1
        elif gold_stated:
            verdict["B_parsing_error"] += 1
        else:
            verdict["C_arithmetic_derived"] += 1
        if len(detail) < 10:
            detail.append(
                f"  {r['question_id']:<16} gold {g:%d %b} stated={gold_stated!s:<5} "
                f"pred {p:%d %b} stated={pred_stated!s:<5}")

    print(f"+1-day cases                          : {len(cases)}")
    print()
    print("CAUSE OF THE +1")
    for k, n in verdict.most_common():
        print(f"  {k:<26} {n}")
    print()
    for line in detail:
        print(line)
    print()
    print("WHICH FIX EACH CAUSE NEEDS")
    print("  A wrong_turn_real_date -> turn SELECTION (the date is real, from")
    print("     another event).  This is the binding disease after all, and it is")
    print("     hiding inside the 'arithmetic' class.")
    print("  B parsing_error        -> date PARSING/normalisation.  The evidence is")
    print("     verbatim in the context and the engine misreads it.")
    print("  C arithmetic_derived   -> calendar ARITHMETIC from another anchor.")
    print()
    print("The largest class names the fix.  Nothing is built until that is known.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
