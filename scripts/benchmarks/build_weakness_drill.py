import json
from collections import defaultdict, Counter

items = json.load(open('benchmark_results/_official_scoring/f1_by_question_mid7b_20261002.json', encoding='utf-8'))

# Filter all non-perfect questions (F1 < 0.999)
weaknesses = [x for x in items if x['f1'] < 0.999]
print(f"Total Questions Evaluated : {len(items)}")
print(f"Total Perfect (F1 == 1.0) : {len(items) - len(weaknesses)} ({(len(items)-len(weaknesses))/len(items)*100:.2f}%)")
print(f"Total Weaknesses (F1 < 1.0): {len(weaknesses)} ({len(weaknesses)/len(items)*100:.2f}%)")

cat_counts = Counter()
zero_counts = Counter()
for x in weaknesses:
    cat_counts[x['cat_name']] += 1
    if x['f1'] == 0.0:
        zero_counts[x['cat_name']] += 1

print("\nWeakness Breakdown by Category:")
for cat, count in cat_counts.most_common():
    zero = zero_counts[cat]
    print(f"  {cat:<15}: {count:3d} questions (Zero F1: {zero:2d})")

# Save as weakness dataset for targeted drilling
drill_data = []
for x in weaknesses:
    drill_data.append({
        "question_id": x["qid"],
        "conv_id": x["qid"].split("-qa-")[0],
        "category": x["category"],
        "cat_name": x["cat_name"],
        "question": x["question"],
        "ground_truth": x["gt"],
        "pred": x["pred"],
        "f1": x["f1"],
        "oracle": x.get("oracle", False)
    })

out_path = 'benchmark_results/weakness_drill_mid7b.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(drill_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(drill_data)} weakness drill questions to: {out_path}")
