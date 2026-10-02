import json

items = json.load(open('benchmark_results/_official_scoring/f1_by_question_mid7b_20261002.json', encoding='utf-8'))
sh = [x for x in items if x['cat_name'] == 'single-hop' and x['f1'] < 0.6]
sh.sort(key=lambda x: x['f1'])
print(f"Total SH < 0.6: {len(sh)}")
for x in sh[:12]:
    print(f"[{x['qid']}] F1={x['f1']:.4f} | Orc={x.get('oracle', '?')}")
    print(f"  Q:    {x['question']}")
    print(f"  GT:   {x['gt']}")
    print(f"  PRED: {x['pred']}")
    print("-" * 50)
