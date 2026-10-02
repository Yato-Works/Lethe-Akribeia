import json

items = json.load(open('benchmark_results/_official_scoring/f1_by_question_mid7b_20261002.json', encoding='utf-8'))
mh = [x for x in items if x['cat_name'] == 'multi-hop' and x['f1'] < 0.6]
mh.sort(key=lambda x: x['f1'])
print(f"Total MH < 0.6: {len(mh)}")
for x in mh[:15]:
    print(f"[{x['qid']}] F1={x['f1']:.4f} | Orc={x.get('oracle', '?')}")
    print(f"  Q:    {x['question']}")
    print(f"  GT:   {x['gt']}")
    print(f"  PRED: {x['pred']}")
    print("-" * 50)
