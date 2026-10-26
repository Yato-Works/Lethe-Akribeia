"""Root-cause the 28 OFF_BY_ONE cases: is it the '% 7 or 7' expression?

temporal_algebra_engine.py:518 computes a weekday delta as

    delta = (anchor.weekday() - wd_idx) % 7 or 7

Python's `or` returns its RIGHT operand when the left is FALSY, and 0 is falsy.
So when the anchor's weekday EQUALS the target weekday -- 'the Friday before
Friday 15 July' -- the delta becomes 7 instead of 0, and the resolved date lands a
full week early.  A same-weekday anchor is not exotic; it is what a speaker
actually says.

This verifies the hypothesis against the real 28 cases rather than reasoning about
the expression alone: for each, it checks whether the engine's answer is exactly
7 days (one week) before gold, which is the signature this bug produces.

If the signature matches, the fix is one line and it is measurable on a frozen
cohort.  If it does not, the arithmetic failures have a different cause and this
line is exonerated.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
WEEKDAYS = {d: i for i, d in enumerate(
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
     "sunday"])}


def parse_full(text: str) -> datetime | None:
    t = str(text).lower()
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+),?\s+(\d{4})\b", t)
    if m and m.group(2) in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    m = re.search(r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", t)
    if m and m.group(1) in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
    return None


def main() -> int:
    atlas = [json.loads(l) for l in ATLAS.read_text(encoding="utf-8").splitlines()
             if l.strip()]

    # The same OFF_BY_ONE set the decomposition found: same month+year, day off 1.
    off_by_one = []
    for r in atlas:
        if r.get("shape") != "when_date":
            continue
        g, p = parse_full(r["gold"]), parse_full(r.get("predicted_answer") or "")
        if g and p and (g.year, g.month) == (p.year, p.month) and \
                abs((p - g).days) == 1:
            off_by_one.append((r, g, p))
    print(f"OFF_BY_ONE cases (same month+year, day differs by 1) : {len(off_by_one)}")
    print()

    print("=== the '% 7 or 7' expression, evaluated directly ===")
    print("  `x % 7 or 7` returns 7 whenever x % 7 == 0, because 0 is falsy:")
    for x in (0, 3, 7, 14):
        print(f"    ({x}) % 7 or 7  ->  {x % 7 or 7}")
    print("  --> a SAME-weekday anchor gets a 7-day jump instead of 0.")
    print()

    # Does the real data show the signature?
    deltas = Counter()
    week_signature = 0
    for r, g, p in off_by_one:
        d = (p - g).days
        deltas[d] += 1
        # The bug's signature: the resolved date sits exactly 7 days from where
        # it should be, i.e. pred = gold +/- 1 happens when the true intent was
        # gold +/- 8 or gold -/+ 6.  Check whether pred is a whole week off from
        # a date adjacent to gold.
        for shift in (7, -7, 14, -14):
            if abs((p - (g + timedelta(days=shift))).days) == 0:
                week_signature += 1
                break

    print("DAY DELTA DISTRIBUTION (pred - gold)")
    for d, n in sorted(deltas.items()):
        print(f"  {d:>+4} days : {n}")
    print()

    # Check the concrete weekday-equality case on the first few examples.
    print("PER-CASE: does the question/context name a weekday equal to the anchor?")
    cache = {}
    for line in (ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
                 ).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row
    checked = 0
    weekday_in_q = 0
    for r, g, p in off_by_one:
        ctx = cache.get(r.get("cache_qid") or "")
        if not ctx:
            continue
        text = f"{r['question']} {ctx.get('context') or ''}"
        names = [w for w in WEEKDAYS if re.search(rf"\b{w}\b", text, re.I)]
        checked += 1
        if names:
            weekday_in_q += 1
        if checked <= 6:
            print(f"  {r['question_id']:<16} gold {g:%d %b} ({g:%A})  "
                  f"pred {p:%d %b}  weekdays named: {names or '-'}")
    print()
    if checked:
        print(f"  cases whose question/context name a weekday : {weekday_in_q}/"
              f"{checked}")
    print()
    print("READ THIS AS")
    print("  gold's weekday printed above is the day the event happened.  Where a")
    print("  weekday name appears, check whether it equals gold's weekday -- that")
    print("  is exactly the input that makes '% 7 or 7' jump a whole week and then")
    print("  land one day off after the surrounding adjustment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
