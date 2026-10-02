import json
from collections import defaultdict

items = json.load(open('benchmark_results/_official_scoring/f1_by_question_mid7b_20261002.json', encoding='utf-8'))

partial_items = [x for x in items if 0.0 < x['f1'] < 0.5]
print(f"Total partial items (0 < F1 < 0.5): {len(partial_items)}")

by_cat = defaultdict(list)
for x in partial_items:
    by_cat[x['cat_name']].append(x)

for cat in ['multi-hop', 'single-hop', 'temporal', 'open-domain']:
    c_list = by_cat[cat]
    print(f"\n{'='*70}")
    print(f"=== CATEGORY: {cat} (Partial count: {len(c_list)}) ===")
    print('='*70)
    for x in c_list[:6]:
        print(f"[{x['qid']}] F1={x['f1']:.4f} | Oracle={x.get('oracle', '?')}")
        print(f"  Q:    {x['question']}")
        print(f"  GT:   {x['gt']}")
        print(f"  PRED: {x['pred']}")
        print("-" * 50)
