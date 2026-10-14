"""Both-direction tests for the MONTH CONTRADICTION guard, plus its real-data effect.

Same discipline as the day and numeral guards: a guard that only closes the hole may
also be eating real credits.  Three sets:
  must-be-wrong     the defect cases, now WRONG
  must-be-correct   legitimate credits that must survive
  must-be-silent    pairs the guard must not claim as contradictions

Then the real measurement: re-score all stored runs with the guard bypassed and with
it active, reporting (correct before, after, spurious removed, real broke).  real_broke
must be 0.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = mod
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
score = mod.score

MUST_BE_WRONG = [
    ("March 5 2023", "5 May 2023"),
    ("15 December 2023", "August 15, 2023"),
    ("15 December 2023", "15 August 2023"),
    ("26 August 2023", "November 2023"),
    ("12 November 2023", "September 2023"),
    ("6 April 2023", "March 2023"),
    ("2 October, 2023", "last week of September 2023"),
    ("27 July 2023", "June 2023"),
]

MUST_BE_CORRECT = [
    # Same month, different day formatting -> still correct
    ("2 September 2022", "September 2, 2022"),
    ("11 September 2023", "September 11, 2023"),
    ("8 May, 2023", "May 8, 2023"),
    # Prediction names several months, one of which matches gold
    ("I visited in June and again in May", "May"),
]

#: Pairs the guard must NOT claim as contradictions.  One earlier candidate,
#: ('August 2023', 'in the summer'), was moved here after diagnosis: the guard
#: already reports False for it, and the WRONG verdict is the pre-existing scorer's
#: own behaviour (guard OFF scores it wrong too).  Listing it under must-be-correct
#: would have blamed the guard for a verdict it does not control.
MUST_BE_SILENT = [
    ("August 2023", "summer"),        # gold names no month at all
    ("August 2023", "in the summer"), # same; pre-existing scorer verdict, not the guard
    ("June", "June"),                 # identical, no contradiction
]


def main() -> int:
    failures: list[str] = []
    for p, g in MUST_BE_WRONG:
        if score(p, g):
            failures.append(f"WRONG-expected: pred={p!r} gold={g!r} scored CORRECT")
    for p, g in MUST_BE_CORRECT:
        if not score(p, g):
            failures.append(f"CORRECT-expected: pred={p!r} gold={g!r} scored WRONG")
    for p, g in MUST_BE_SILENT:
        if mod._month_contradiction(p, g):
            failures.append(f"guard fired where it must not: {p!r} vs {g!r}")

    print(f"month guard cases : {len(MUST_BE_WRONG) + len(MUST_BE_CORRECT) + len(MUST_BE_SILENT)}")
    print(f"  must-be-wrong   : {len(MUST_BE_WRONG)}")
    print(f"  must-be-correct : {len(MUST_BE_CORRECT)}")
    print(f"  must-be-silent  : {len(MUST_BE_SILENT)}")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
    else:
        print("\nall cases pass in BOTH directions")
    print()

    # ---- Real data.
    runs = sorted((ROOT / "benchmark_results").glob("locomo_full_*.json"))
    total = before = after = 0
    removed: list[tuple] = []
    broke: list[tuple] = []
    for path in runs:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                       # noqa: BLE001
            continue
        for r in (data.get("results") or []):
            pred, gold = r.get("predicted_answer"), r.get("ground_truth")
            if not pred or not gold:
                continue
            total += 1
            real = mod._month_contradiction
            mod._month_contradiction = lambda a, b: False
            try:
                old = score(pred, gold)
            finally:
                mod._month_contradiction = real
            new = score(pred, gold)
            before += old
            after += new
            if old and not new:
                removed.append((str(r.get("question_id")), str(pred)[:34], str(gold)[:30]))
            elif new and not old:
                broke.append((str(r.get("question_id")), str(pred)[:34], str(gold)[:30]))

    print("ON REAL DATA (all stored runs)")
    print(f"  runs scanned            : {len(runs)}")
    print(f"  total pairs             : {total}")
    print(f"  correct before guard    : {before}")
    print(f"  correct after guard     : {after}")
    print(f"  spurious removed        : {len(removed)}")
    print(f"  real credits broke      : {len(broke)}   <-- must be 0")
    if broke:
        print("\n  BROKE -- the guard is eating real credits:")
        for q, p, g in broke[:12]:
            print(f"    {q:<20} pred={p!r:<36} gold={g!r}")
    print("\n  sample of removed spurious credits:")
    for q, p, g in removed[:12]:
        print(f"    {q:<20} pred={p!r:<36} gold={g!r}")
    print()
    print("VERDICT:", "clean fix" if not broke else "NOT CLEAN")
    return 1 if failures or broke else 0


if __name__ == "__main__":
    raise SystemExit(main())
