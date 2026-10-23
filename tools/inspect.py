import json

path = 'c:/Users/smily/Lethe-Akribeia/benchmark_results/pipeline_v4/lme_dev250_v4_pipeline.json'
with open(path, 'r', encoding='utf-8') as f:
    base = json.load(f)

print('top keys:', list(base.keys()))
print('condition:', base.get('condition'))
print('n results:', len(base.get('results', [])))
r0 = base['results'][0]
print('r0 keys:', list(r0.keys()))
print('r0 r6_features:', json.dumps(r0.get('r6_features'), ensure_ascii=False)[:500])
