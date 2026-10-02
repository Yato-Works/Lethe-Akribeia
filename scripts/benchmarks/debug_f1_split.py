import sys
sys.path.insert(0, 'scripts')
import benchmarks.simulate_rules as sim

f1_func = sim.official.f1
f1_score_func = sim.official.f1_score
norm_func = sim.official.normalize_answer

print("=== conv-42-qa-066 ===")
gt = 'an animalkeeper at a localzoo and workingwith turtles; as heknows a great dealabout turtles andhow to care for them,and he enjoys it.'
gts = [g.strip() for g in gt.split(',')]
print("GT split:", gts)
for i, g in enumerate(gts):
    print(f"  GT[{i}] norm: '{norm_func(g)}'")

score = f1_func(gt, gt)
print(f"f1(gt, gt): {score}")

preds = [p.strip() for p in gt.split(',')]
for i, g in enumerate(gts):
    for j, p in enumerate(preds):
        s = f1_score_func(p, g)
        print(f"  f1_score(PRED[{j}], GT[{i}]): {s}")

print("\n=== conv-50-qa-013 ===")
gt50 = 'Yes; because he enjoys the rush of performing onstage to large crowds'
gts50 = [g.strip() for g in gt50.split(',')]
print("GT50 split:", gts50)
score50 = f1_func(gt50, gt50)
print(f"f1(gt50, gt50): {score50}")

# what does dataset have for conv-50-qa-013?
import json
raw_data = json.load(open('benchmark_results/hard_smoke_60q.json', encoding='utf-8'))
for q in raw_data:
    if q.get('question_id') == 'conv-50-qa-013':
        print("Raw QA in hard_smoke_60q:", q)
