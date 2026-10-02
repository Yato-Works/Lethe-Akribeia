import json
from collections import defaultdict

with open('benchmark_results/_official_scoring/f1_by_question_full_v18.json', encoding='utf-8') as f:
    v18 = json.load(f)
with open('benchmark_results/_official_scoring/f1_by_question_full_v25.json', encoding='utf-8') as f:
    v25 = json.load(f)

c18 = defaultdict(list)
for x in v18:
    c18[x['category']].append(x['f1'])

c25 = defaultdict(list)
for x in v25:
    c25[x['category']].append(x['f1'])

CAT_NAMES = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop', 5: 'adversarial'}

print("Full Run (1,986 questions, all 10 conversations):")
print(f"{'Category':<15} | {'v18 F1':<10} | {'v25 F1':<10} | {'Diff':<10}")
print("-" * 52)
for cat in [1, 2, 3, 4, 5]:
    m18 = sum(c18[cat]) / len(c18[cat]) * 100
    m25 = sum(c25[cat]) / len(c25[cat]) * 100
    print(f"{CAT_NAMES[cat]:<15} | {m18:>9.2f}% | {m25:>9.2f}% | {m25 - m18:>+9.2f}%")
print("-" * 52)
tot18 = sum(x['f1'] for x in v18) / len(v18) * 100
tot25 = sum(x['f1'] for x in v25) / len(v25) * 100
print(f"{'Overall':<15} | {tot18:>9.2f}% | {tot25:>9.2f}% | {tot25 - tot18:>+9.2f}%")
