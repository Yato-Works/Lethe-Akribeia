import json
from collections import defaultdict

items = json.load(open('benchmark_results/_official_scoring/f1_by_question_mid7b_20261002.json', encoding='utf-8'))

# Group by category and examine lowest F1 questions
by_cat = defaultdict(list)
for x in items:
    by_cat[x['cat_name']].append(x)

for cat in ['multi-hop', 'single-hop', 'temporal', 'open-domain']:
    c_list = sorted(by_cat[cat], key=lambda x: x['f1'])
    low_f1 = [x for x in c_list if x['f1'] < 0.6]
    print(f"\n{'='*75}")
    print(f"CATEGORY: {cat} | Total: {len(c_list)} | F1 < 0.6: {len(low_f1)}")
    print('='*75)
    for x in low_f1[:15]:
        print(f"[{x['qid']}] F1={x['f1']:.4f} | Orc={x.get('oracle', '?')}")
        print(f"  Q:    {x['question']}")
        print(f"  GT:   {x['gt']}")
        print(f"  PRED: {x['pred']}")
        print("-" * 50)
