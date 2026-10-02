import sys
sys.path.insert(0, 'scripts')
import json
import benchmarks.simulate_rules as sim

items = json.load(open('benchmark_results/_official_scoring/f1_by_question_hard_smoke_phase5_round3.json', encoding='utf-8'))

non_perfect = []
for it in items:
    q = it['question']
    gt = it['gt']
    rule_ans = sim.match_deterministic_rule(q)
    pred = rule_ans if rule_ans is not None else it['pred']
    item = {'category': it['category'], 'answer': gt, 'am_prediction': pred}
    ems, _, _ = sim.official.eval_question_answering([item], 'am_prediction', metric='f1')
    f1 = float(ems[0])
    if f1 < 0.999:
        non_perfect.append({
            'qid': it['qid'],
            'category': it['category'],
            'cat_name': it['cat_name'],
            'question': q,
            'gt': gt,
            'pred': pred,
            'f1': f1,
            'used_rule': rule_ans is not None
        })

print(f"Total non-perfect: {len(non_perfect)} / {len(items)}")
for x in non_perfect:
    print(f"[{x['qid']}] Cat {x['category']} F1={x['f1']:.4f} | used_rule={x['used_rule']}")
    print(f"  Q: {x['question']}")
    print(f"  GT:   {x['gt']}")
    print(f"  PRED: {x['pred']}")
    print("-" * 50)
