import json

with open("benchmark_results/pipeline_v4/lme_dev250_v4_R7.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results", [])
fails = [d for d in results if not d.get("is_correct", False)]

print(f"Total Fails: {len(fails)}")

by_cat = {}
for f in fails:
    cat = f.get("question_type", "unknown")
    by_cat.setdefault(cat, []).append(f)

for cat, items in by_cat.items():
    print(f"\n=== Category: {cat} ({len(items)} items) ===")
    for item in items:
        qid = item["question_id"]
        q = item.get("question", "")[:80]
        gt = str(item.get("ground_truth", ""))[:50]
        pred = str(item.get("predicted_answer", ""))[:50]
        print(f"[{qid}] Q: {q}")
        print(f"   GT:   {gt}")
        print(f"   PRED: {pred}")

