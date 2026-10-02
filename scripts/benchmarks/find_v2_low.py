import sys
sys.path.insert(0, 'scripts')
import json
import re
from benchmarks.simulate_mid_scale_v2 import clean_and_normalize, official

items = json.load(open('benchmark_results/_official_scoring/f1_by_question_mid7b_20261002.json', encoding='utf-8'))

still_low = []
for it in items:
    q = it['question']
    gt = it['gt']
    cat = it['category']
    p_opt = clean_and_normalize(q, it['pred'], cat)
    
    item_o = {'category': cat, 'answer': gt, 'am_prediction': p_opt}
    ems_o, _, _ = official.eval_question_answering([item_o], 'am_prediction', metric='f1')
    f1 = float(ems_o[0])
    if f1 < 0.5:
        still_low.append({
            'qid': it['qid'],
            'cat': it['cat_name'],
            'q': q,
            'gt': gt,
            'pred': p_opt,
            'f1': f1,
            'orc': it.get('oracle', '?')
        })

print(f"Still low F1 (< 0.5): {len(still_low)}")
for x in still_low[:20]:
    print(f"[{x['qid']} - {x['cat']}] F1={x['f1']:.4f} | Orc={x['orc']}")
    print(f"  Q:    {x['q']}")
    print(f"  GT:   {x['gt']}")
    print(f"  PRED: {x['pred']}")
    print("-" * 50)
