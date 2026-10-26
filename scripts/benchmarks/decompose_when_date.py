"""Decompose the when_date cluster by DISEASE, not by question count.

The roadmap described this cluster as 'when_date binds to the wrong turn' -- an
event-binding failure.  Two of the first five examples contradict that outright:
gold '7 May 2023' vs engine '8 May, 2023' is the SAME event with the day wrong by
one, which is a date-arithmetic failure, not a binding failure.

Treating both as one cluster would mean one fix designed for the wrong disease.
So each failure is classified by what actually went wrong:

  OFF_BY_ONE        same month and year, day differs by 1 -> arithmetic error
  RIGHT_MONTH_WRONG same month, day differs by more        -> arithmetic error
  WRONG_EVENT       a real date from a different event     -> binding error
  RELATIVE_UNRESOLVED gold is a relative expression the engine did not resolve
  ABSTAIN           no engine committed
  OTHER             none of the above

Reported with n per class, so the largest ACTUAL disease is visible.  The fix goes
at whatever the numbers say is largest, not at the label the cluster carries.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def parse_date(text: str) -> tuple[int, int, int] | None:
    """(year, month, day) from the common 'D Month YYYY' / 'Month D, YYYY' forms."""
    t = str(text).lower()
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+),?\s+(\d{4})\b", t)
    if m and m.group(2) in MONTHS:
        return int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))
    m = re.search(r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", t)
    if m and m.group(1) in MONTHS:
        return int(m.group(3)), MONTHS[m.group(1)], int(m.group(2))
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\b", t)
    if m and m.group(2) in MONTHS:
        return 0, MONTHS[m.group(2)], int(m.group(1))
    return None


RELATIVE = re.compile(
    r"\b(the\s+)?(week|month|year|day)s?\s+(before|after)|"
    r"\b(tuesday|monday|wednesday|thursday|friday|saturday|sunday)\b|"
    r"\b(last|next)\s+(week|month|year)\b|\bago\b", re.I)


def classify(row: dict) -> str:
    gold, pred = str(row.get("gold")), str(row.get("predicted_answer") or "")
    if not row.get("engine_claims"):
        return "ABSTAIN"
    if RELATIVE.search(gold) and not parse_date(gold):
        return "RELATIVE_UNRESOLVED"
    gd, pd = parse_date(gold), parse_date(pred)
    if gd and pd:
        gy, gm, gday = gd
        py, pm, pday = pd
        if (gy, gm) == (py, pm):
            return "OFF_BY_ONE" if abs(pday - gday) == 1 else "RIGHT_MONTH_WRONG"
        if gm == pm and gy == 0 and py == 0:
            return "OFF_BY_ONE" if abs(pday - gday) == 1 else "RIGHT_MONTH_WRONG"
        return "WRONG_EVENT"
    return "OTHER"


def main() -> int:
    atlas = [json.loads(l) for l in ATLAS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    wd = [r for r in atlas if r.get("shape") == "when_date"]

    counts = Counter(classify(r) for r in wd)
    total = len(wd)

    print(f"when_date cluster (ALL)          : {total}")
    print()
    print("CLASSIFIED BY WHAT ACTUALLY WENT WRONG")
    print(f"  {'class':<22} {'n':>5} {'%':>7}")
    order = ["OFF_BY_ONE", "RIGHT_MONTH_WRONG", "WRONG_EVENT",
             "RELATIVE_UNRESOLVED", "ABSTAIN", "OTHER"]
    for cls in order:
        n = counts.get(cls, 0)
        if n:
            print(f"  {cls:<22} {n:>5} {n / total * 100:>6.1f}%")
    print()

    arithmetic = counts.get("OFF_BY_ONE", 0) + counts.get("RIGHT_MONTH_WRONG", 0)
    binding = counts.get("WRONG_EVENT", 0)
    print("TWO CANDIDATE DISEASES")
    print(f"  date ARITHMETIC (right event, wrong day) : {arithmetic} "
          f"({arithmetic / total * 100:.1f}%)")
    print(f"  event BINDING   (date from another event): {binding} "
          f"({binding / total * 100:.1f}%)")
    print()

    if arithmetic > binding:
        print("=> The cluster is mostly an ARITHMETIC problem, not the binding")
        print("   problem its label claims.  A fix aimed at binding would be built")
        print("   for the minority disease.")
    elif binding > arithmetic:
        print("=> The cluster is mostly a BINDING problem, as the roadmap claimed.")
    else:
        print("=> Both diseases are present in comparable size.")

    print()
    print("EXAMPLES PER CLASS (up to 3)")
    for cls in order:
        rows = [r for r in wd if classify(r) == cls][:3]
        if not rows:
            continue
        print(f"  [{cls}]")
        for r in rows:
            print(f"    {r['question_id']}")
            print(f"      Q    : {r['question'][:64]}")
            print(f"      gold : {str(r['gold'])[:52]!r}")
            print(f"      eng  : {str(r.get('predicted_answer'))[:52]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
