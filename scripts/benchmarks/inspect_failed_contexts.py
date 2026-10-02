import json
import glob

with open('benchmark_results/_official_scoring/f1_by_question_hard_smoke_phase5_round2.json', 'r', encoding='utf-8') as f:
    items = {x['qid']: x for x in json.load(f)}

conv_files = glob.glob('benchmark_results/locomo10_runs/hard_smoke_phase5_round2/conv_*_results.json')
preds = {}
for cf in conv_files:
    with open(cf, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for r in data.get('results', []):
            preds[r['question_id']] = r

print(f"Loaded {len(preds)} prediction results from {len(conv_files)} files")

target_qids = [
    'conv-26-qa-026', 'conv-41-qa-007', 'conv-41-qa-018', 
    'conv-41-qa-040', 'conv-26-qa-140', 'conv-30-qa-031',
    'conv-42-qa-004', 'conv-41-qa-024', 'conv-41-qa-009',
    'conv-42-qa-073', 'conv-42-qa-085', 'conv-26-qa-111'
]

for qid in target_qids:
    if qid not in preds:
        print(f"QID {qid} not found in preds")
        continue
    p = preds[qid]
    it = items.get(qid, {})
    gold = it.get('gt', '')
    pred = it.get('pred', '')
    q = it.get('question', '')
    context = p.get('context', '')
    print(f"\n==========================================")
    print(f"[{qid}]")
    print(f"Q: {q}")
    print(f"Gold: {gold}")
    print(f"Pred: {pred}")
    print(f"Context length: {len(context)} chars")
    gold_words = [w.lower() for w in gold.replace(',', ' ').replace('.', ' ').replace(';', ' ').replace('"', '').split() if len(w) > 2]
    hit_words = [w for w in gold_words if w in context.lower()]
    print(f"Gold words: {gold_words} -> Found in context: {hit_words}")
    if hit_words:
        first_w = hit_words[0]
        idx = context.lower().find(first_w)
        if idx != -1:
            start = max(0, idx - 100)
            end = min(len(context), idx + 200)
            print(f"Snippet:\n... {context[start:end]} ...")
    else:
        print("Gold words NOT found in context! First 300 chars of context:")
        print(context[:300])
