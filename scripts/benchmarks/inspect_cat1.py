import json
from pathlib import Path
from compare_runs import load_run, load_official_module, REPO

v8 = load_run(REPO / 'benchmark_results' / 'locomo10_runs' / 'e2e_smoke_20261002_v8')
dataset = json.load(open(REPO / 'benchmark_results' / '_official_scoring' / 'locomo10_normalized.json', encoding='utf-8'))
official = load_official_module()

for sample in dataset:
    sid = sample['sample_id']
    for i, qa in enumerate(sample['qa']):
        qid = f"{sid}-qa-{i:03d}"
        if qid in v8 and qa['category'] == 1:
            p8 = v8[qid]['predicted_answer']
            gold = qa['answer']
            quest = qa['question']
            f1 = official.f1_score(p8, gold)
            if f1 < 0.6:
                print(f"[{qid}] F1: {f1*100:.1f}%")
                print(f"   Q:    {quest}")
                print(f"   Gold: {gold}")
                print(f"   Pred: {p8}")
                print()
