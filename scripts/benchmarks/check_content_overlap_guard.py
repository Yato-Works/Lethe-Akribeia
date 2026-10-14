"""Both-direction verification for the CONTENT-OVERLAP requirement.

This guard is riskier than its three siblings: it gates the two MOST-USED credit
rules in the scorer, so a mistake here silently changes thousands of verdicts.  It is
therefore checked harder.

  must-be-wrong     the article-only credits, now WRONG
  must-be-correct   legitimate short-label credits that must survive -- this is the
                    set most at risk, because LoCoMo golds are often two or three
                    words and a careless stopword list would veto real answers
  on real data      every stored run re-scored with the requirement bypassed and
                    active, reporting (spurious removed, real broke).  real broke
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
    ("the", "The Alchemist"),
    ("the book", "The Alchemist"),
    ("The Name of the Wind", "The Alchemist"),
    ("the same way with my team", "The Alchemist"),
    ("The headliner", "The Fireworks"),
]

#: Legitimate credits.  The first group is the real risk: short gold labels where a
#: content word must still be found.
MUST_BE_CORRECT = [
    # A content word IS shared even though the surface forms differ.
    ("She is lactose intolerant", "lactose intolerance"),
    ("a letter", "a handwritten letter"),
    ("Chicken dish with garlic, lemon, and herbs", "Roasted Chicken"),
    ("Sweden", "Sweden"),
    ("She moved from Sweden", "Sweden"),
    ("the painting", "the painting of the sunset"),
    # Gold whose only tokens are content words.
    ("Oliver, Luna and Bailey", "Oliver, Luna, Bailey"),
]

MUST_BE_SILENT = [
    # Gold with no content words at all must not be vetoed by this guard.
    ("the", "the"),
    ("and", "and"),
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
        if not mod._has_content_overlap(p, g):
            failures.append(f"guard vetoed a content-free gold: {p!r} vs {g!r}")

    n = len(MUST_BE_WRONG) + len(MUST_BE_CORRECT) + len(MUST_BE_SILENT)
    print(f"content-overlap cases : {n}")
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

    # ---- Real data.  Bypass by making the requirement always true.
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
            real = mod._has_content_overlap
            mod._has_content_overlap = lambda a, b: True
            try:
                old = score(pred, gold)
            finally:
                mod._has_content_overlap = real
            new = score(pred, gold)
            before += old
            after += new
            if old and not new:
                removed.append((str(r.get("question_id")), str(pred)[:34], str(gold)[:30]))
            elif new and not old:
                broke.append((str(r.get("question_id")), str(pred)[:34], str(gold)[:30]))

    print("ON REAL DATA (all stored runs)")
    print(f"  runs scanned          : {len(runs)}")
    print(f"  total pairs           : {total}")
    print(f"  correct before        : {before}")
    print(f"  correct after         : {after}")
    print(f"  spurious removed      : {len(removed)}")
    print(f"  real credits broke    : {len(broke)}   <-- must be 0")
    if broke:
        print("\n  BROKE -- the guard is eating real credits:")
        for q, p, g in broke[:15]:
            print(f"    {q:<20} pred={p!r:<36} gold={g!r}")
    if removed:
        print("\n  removed spurious credits:")
        for q, p, g in removed[:15]:
            print(f"    {q:<20} pred={p!r:<36} gold={g!r}")
    print()
    print("VERDICT:", "clean fix" if not broke else "NOT CLEAN")
    return 1 if failures or broke else 0


if __name__ == "__main__":
    raise SystemExit(main())
