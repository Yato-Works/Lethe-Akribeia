import json
from collections import defaultdict

items = json.load(open('benchmark_results/_official_scoring/f1_by_question_mid7b_20261002.json', encoding='utf-8'))

zero_items = [x for x in items if x['f1'] == 0.0]
print(f"Total zero items: {len(zero_items)}")

by_cat = defaultdict(list)
for x in zero_items:
    by_cat[x['cat_name']].append(x)

for cat in ['multi-hop', 'single-hop', 'temporal', 'open-domain']:
    c_list = by_cat[cat]
    print(f"\n{'='*70}")
    print(f"=== CATEGORY: {cat} (Zero count: {len(c_list)}) ===")
    print('='*70)
    for x in c_list[:8]:  # Show first 8 per category
        print(f"[{x['qid']}] Oracle={x.get('oracle', '?')}")
        print(f"  Q:    {x['question']}")
        print(f"  GT:   {x['gt']}")
        print(f"  PRED: {x['pred']}")
        print("-" * 50)
