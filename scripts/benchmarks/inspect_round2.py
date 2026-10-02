import json
from collections import defaultdict

with open('benchmark_results/_official_scoring/f1_by_question_hard_smoke_phase5_round2.json', 'r', encoding='utf-8') as f:
    items = json.load(f)

print(f'Total questions scored: {len(items)}')

cat_items = defaultdict(list)
for it in items:
    cat_items[it.get('category', 0)].append(it)

for cat in sorted(cat_items.keys()):
    c_list = cat_items[cat]
    avg_f1 = sum(x['f1'] for x in c_list) / len(c_list)
    print(f'\n=== Category {cat} ({c_list[0].get("cat_name")}): Count={len(c_list)}, Avg F1={avg_f1:.4f} ===')
    for x in c_list:
        if x['f1'] < 0.8:
            print(f"  [{x['qid']}] F1={x['f1']:.2f}")
            print(f"    Q: {x['question']}")
            print(f"    Gold: {x['gt']}")
            print(f"    Pred: {x['pred']}")
