#!/usr/bin/env python3
"""M30-44 Phase 1: Derive the 15 production failures from the REAL run.

Source: results/m30_43/production_bucketA.json (locomo_adapter path).
For each failure, classify by stage using the actual predicted text and
the compile() context — NOT a hand-rolled copy.

Classification (only what the text supports):
  ABSTAINED          - reader refused ("no mention", "does not contain")
  WRONG_DATE         - reader gave a date, different from gold
  WRONG_OTHER        - reader gave a non-date answer
  GOLD_SUSPECT       - gold itself may be unsupported (flag for review)
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))

ABSTAIN_PAT = re.compile(r"(no mention|does not contain|does not provide|no information|"
                         r"not specified|no specific|cannot|does not specify|not clear)", re.I)
DATE_PAT = re.compile(r"(\d{1,2}\s+)?(january|february|march|april|may|june|july|august|"
                      r"september|october|november|december)(\s+\d{1,2})?,?\s+\d{4}", re.I)

print("=" * 78)
print("M30-44 Phase 1: Production failures (from locomo_adapter run)")
print("=" * 78)
print()

fails = [r for r in prod["rows"] if not r["correct"]]
print(f"Total: {prod['total']}, correct: {prod['correct']}, failures: {len(fails)}")
print()

buckets = {"ABSTAINED": [], "WRONG_DATE": [], "WRONG_OTHER": []}
for r in fails:
    p = r["predicted"]
    if ABSTAIN_PAT.search(p):
        cls = "ABSTAINED"
    elif DATE_PAT.search(p):
        cls = "WRONG_DATE"
    else:
        cls = "WRONG_OTHER"
    buckets[cls].append(r)

for cls, rows in buckets.items():
    print(f"### {cls}: {len(rows)}")
    for r in rows:
        print(f"  {r['qid']:16s} gold={r['gold']!r}")
        print(f"      pred={r['predicted'][:90]!r}")
    print()

# Cross-check: did the reader see the gold turn? Use the context from the run
# if available; otherwise flag as needing a re-run trace.
print("=" * 78)
print("SUMMARY")
print("=" * 78)
for cls, rows in buckets.items():
    print(f"  {cls}: {len(rows)}")
print(f"  TOTAL: {sum(len(v) for v in buckets.values())}")
print()

out = PROJECT_ROOT / "results/m30_44"
out.mkdir(parents=True, exist_ok=True)
(out / "production_failures.json").write_text(json.dumps({
    "correct": prod["correct"], "total": prod["total"],
    "buckets": {k: [r["qid"] for r in v] for k, v in buckets.items()},
    "rows": fails,
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved: {out / 'production_failures.json'}")
