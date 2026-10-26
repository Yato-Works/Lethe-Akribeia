"""The third disease: gold answers that name only a YEAR.

The when_date decomposition found arithmetic (40) and binding (18), but a large
'OTHER/ABSTAIN' remainder (89) that the first pass did not explain.  Reading the
examples shows why: gold is frequently just '2022' or 'January, 2023' -- a bare
year or month with no day at all -- while the engine answers a full specific date.

That is not an arithmetic failure and not a binding failure.  It is a PRECISION
failure: the evidence supports only '2022', and the engine over-commits to a day
the conversation never stated.  Guessing a day inside the right year is exactly
the shape that earns a wrong answer where abstention would have been honest.

This measures how much of the cluster is that shape, because a fix for it is the
opposite of a fix for arithmetic: arithmetic needs BETTER day resolution, this
needs COARSER -- answering at the granularity the evidence actually supports.
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


def granularity(text: str) -> str:
    """The precision a date string actually commits to."""
    t = str(text).lower().strip()
    if not t:
        return "empty"
    if re.fullmatch(r"\s*(1[89]\d{2}|20\d{2})\s*", t):
        return "year_only"
    if re.fullmatch(r"[a-z]+,?\s*(1[89]\d{2}|20\d{2})", t):
        return "month_year"
    if re.search(r"\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+,?\s+\d{4}", t) or \
       re.search(r"[a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}", t):
        return "full_day"
    if re.search(r"\d{4}", t):
        return "has_year_plus_other"
    return "relative_or_none"


def main() -> int:
    atlas = [json.loads(l) for l in ATLAS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    wd = [r for r in atlas if r.get("shape") == "when_date"]

    print(f"when_date cluster : {len(wd)}")
    print()
    print("GOLD GRANULARITY -- what precision the evidence actually supports")
    gc = Counter(granularity(r["gold"]) for r in wd)
    print(f"  {'gold granularity':<22} {'n':>5}")
    for k, n in gc.most_common():
        print(f"  {k:<22} {n:>5}")
    print()

    coarse = gc.get("year_only", 0) + gc.get("month_year", 0)
    print(f"gold names ONLY a year or month-year : {coarse} "
          f"({coarse / len(wd) * 100:.1f}%)")
    print()

    print("THE OVER-COMMIT: coarse gold vs a full-day engine answer")
    over = [
        r for r in wd
        if granularity(r["gold"]) in ("year_only", "month_year")
        and granularity(r.get("predicted_answer") or "") == "full_day"
    ]
    print(f"  engine gave a specific day where gold states a year/month only : "
          f"{len(over)}")
    print(f"  of those, engine claimed (did not abstain)                      : "
          f"{sum(1 for r in over if r.get('engine_claims'))}")
    print()
    print("EXAMPLES")
    for r in over[:8]:
        print(f"  {r['question_id']}")
        print(f"    Q    : {r['question'][:62]}")
        print(f"    gold : {str(r['gold'])[:40]!r}   <- only this precision exists")
        print(f"    eng  : {str(r.get('predicted_answer'))[:40]!r}")
    print()
    print("WHY THIS IS A SEPARATE DISEASE")
    print("  Arithmetic failures need BETTER day resolution.  These need COARSER:")
    print("  answer at the granularity the evidence supports, and abstain when it")
    print("  supports none.  A day invented inside the correct year is still a")
    print("  wrong answer, and it is worse than abstaining because it looks")
    print("  confident.  One mechanism cannot serve both diseases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
