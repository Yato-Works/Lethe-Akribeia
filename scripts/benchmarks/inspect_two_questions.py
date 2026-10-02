import sys
sys.path.insert(0, 'scripts')
import json
import benchmarks.simulate_rules as sim

items = json.load(open('benchmark_results/_official_scoring/f1_by_question_hard_smoke_phase5_round3.json', encoding='utf-8'))
for it in items:
    if it['qid'] in ['conv-42-qa-066', 'conv-50-qa-013']:
        gt = it['gt']
        pred = sim.match_deterministic_rule(it['question'])
        print(f"=== {it['qid']} ===")
        print(f"GT repr:   {repr(gt)}")
        print(f"PRED repr: {repr(pred)}")
        item = {'category': it['category'], 'answer': gt, 'am_prediction': pred}
        ems, _, _ = sim.official.eval_question_answering([item], 'am_prediction', metric='f1')
        print(f"F1: {ems[0]}")
