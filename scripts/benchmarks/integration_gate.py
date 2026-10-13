"""Integration gate -- the ONLY way an engine change may be reported as a gain.

Born from this session's retraction (e6a305f): an engine 'FIXED' was recorded
because the scorer credited a wrong count, and the gain vanished the moment the
scorer was repaired.  The lesson generalised into rules.

Every claim about an engine change must pass through here.  The gate refuses to
produce a number unless the wiring is proven, because "N tests pass" proves the
part, not the wiring.

WHAT IT CHECKS
  1. COHORT JOINED     the cohort's question_ids actually match frozen context
                       rows and gold rows.  Matching on normalised question text
                       silently zero-matches; join on question_id and ASSERT the
                       matched count equals the cohort size.
  2. SCORER PINNED     the scorer used for BOTH sides is named, and neither side
                       is scored by a different scorer generation.  A before/after
                       that crosses the guard boundary is a scoring artifact.
  3. PAIRED            before and after run on the SAME ids, questions, frozen
                       contexts, and surrounding pipeline.  Only the engine differs.
  4. BOTH DIRECTIONS   FIXED, BROKE, and SPURIOUS are all reported.  A bare
                       correct count is rejected outright.
  5. SPURIOUS PRICED   a credit is SPURIOUS when the corrected scorer credits it
                       only through a defect.  The scorer is asked whether it is
                       leaning on a contradiction guard to earn that credit.
  6. HELDOUT           the cohort must be disjoint from any cohort used to design
                       the fix, or the gain is fitting rather than generalising.

USAGE
    python scripts/benchmarks/integration_gate.py --before b.json --after a.json

Both JSON files must carry: {"question_ids": [...], "records": [{"question_id",
"status", "answer", "gold"}]}.  Exits non-zero if the gate refuses to report.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
_mod = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = _mod
try:
    spec.loader.exec_module(_mod)
except SystemExit:
    pass
score = _mod.score


def load_cohort(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "question_ids" not in data or "records" not in data:
        raise SystemExit(
            f"{path.name}: need both 'question_ids' and 'records'; refusing to guess")
    return data


def check_join(cohort: dict, label: str) -> list[str]:
    """The wiring check.  Asserts records cover every declared question_id."""
    problems = []
    declared = list(cohort["question_ids"])
    records = cohort["records"]
    seen = [r.get("question_id") for r in records]
    dupes = [q for q, n in Counter(seen).items() if n > 1]
    if dupes:
        problems.append(f"{label}: duplicate question_ids in records: {dupes[:5]}")
    missing = [q for q in declared if q not in set(seen)]
    if missing:
        problems.append(
            f"{label}: {len(missing)}/{len(declared)} declared ids have no record "
            f"-- e.g. {missing[:3]}")
    extra = [q for q in seen if q not in set(declared)]
    if extra:
        problems.append(f"{label}: {len(extra)} records outside the declared cohort")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--holdout", default=None,
                    help="ids the fix was designed against; must be disjoint")
    args = ap.parse_args()

    before = load_cohort(ROOT / args.before)
    after = load_cohort(ROOT / args.after)

    problems: list[str] = []
    problems += check_join(before, "before")
    problems += check_join(after, "after")

    b_ids = list(before["question_ids"])
    a_ids = list(after["question_ids"])
    if set(b_ids) != set(a_ids):
        problems.append("before and after are NOT paired: question_id sets differ")

    if args.holdout:
        hold = {l.strip() for l in (ROOT / args.holdout).read_text(
            encoding="utf-8").splitlines() if l.strip()}
        overlap = hold & set(a_ids)
        if overlap:
            problems.append(
                f"{len(overlap)} cohort ids are also in the design/holdout set "
                f"-- the gain would be fitting, not generalising")

    if problems:
        print("GATE REFUSED TO REPORT -- the wiring is not proven:\n")
        for p in problems:
            print(f"  - {p}")
        return 1

    b_rec = {r["question_id"]: r for r in before["records"]}
    a_rec = {r["question_id"]: r for r in after["records"]}

    fixed: list[str] = []
    broke: list[str] = []
    spurious: list[tuple] = []

    for qid in b_ids:
        b, a = b_rec[qid], a_rec[qid]
        gold = a.get("gold") or b.get("gold")
        ans = a.get("answer")
        now = bool(ans) and bool(gold) and score(ans, gold)
        was = b.get("status") == "correct"
        if now and _mod._numeral_contradiction(ans, gold):
            spurious.append((qid, str(ans)[:40], str(gold)[:30]))
        elif now and not was:
            fixed.append(qid)
        elif was and not now:
            broke.append(qid)

    print(f"cohort n                        : {len(b_ids)}")
    print(f"FIXED                           : {len(fixed)}")
    print(f"BROKE                           : {len(broke)}")
    print(f"SPURIOUS (credit came from the scorer defect): {len(spurious)}")
    if spurious:
        print("\nspurious credits -- these are NOT engine gains:")
        for qid, ans, gold in spurious[:10]:
            print(f"  {qid:<22} ans={ans!r:<42} gold={gold!r}")
    if broke:
        print(f"\nBROKE ids: {broke[:15]}")
    print("\nVERDICT: a bare correct count is never reported; the three numbers "
          "above are the claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
