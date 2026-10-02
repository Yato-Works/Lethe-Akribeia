import json
import re

items = json.load(open('benchmark_results/_official_scoring/f1_by_question_mid7b_20261002.json', encoding='utf-8'))

# Look at all questions where GT and PRED are related but F1 < 0.5
for it in items:
    f1 = it['f1']
    if 0.0 <= f1 < 0.5:
        q = it['question']
        gt = it['gt']
        p = it['pred']
        cat = it['cat_name']
        print(f"[{it['qid']} - {cat}] F1={f1:.4f}")
        print(f"  Q:    {q}")
        print(f"  GT:   {gt}")
        print(f"  PRED: {p}")
        print("-" * 60)
