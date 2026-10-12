import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

r73_path = Path("benchmark_results/pipeline_v4/lme_dev250_v4_R7.3.json")
with open(r73_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", [])
fails = [r for r in results if not r.get("is_correct", False)]

print(f"Total Residual Failures: {len(fails)} / {len(results)}")

by_cat = {}
for f in fails:
    cat = f.get("question_type", "unknown")
    by_cat.setdefault(cat, []).append(f)

for cat, items in by_cat.items():
    print(f"\n{'='*75}")
    print(f"Category: {cat} ({len(items)} items)")
    print(f"{'='*75}")
    for item in items:
        qid = item["question_id"]
        q = item.get("question", "")
        gt = str(item.get("ground_truth", ""))
        pred = str(item.get("predicted_answer", ""))
        print(f"\n[{qid}] Q: {q}")
        print(f"   GT:   {gt}")
        print(f"   PRED: {pred}")
