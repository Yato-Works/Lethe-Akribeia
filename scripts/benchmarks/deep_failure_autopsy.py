import json
from collections import defaultdict

def main():
    with open('benchmark_results/_official_scoring/f1_by_question_full_v25.json', encoding='utf-8') as f:
        data = json.load(f)

    total_q = len(data)
    zeros = [q for q in data if q['f1'] == 0.0]
    lows = [q for q in data if 0.0 < q['f1'] < 0.5]
    highs = [q for q in data if q['f1'] >= 0.5]

    print(f"=== Overall Failure Breakdown (Total: {total_q} questions) ===")
    print(f"Zero F1 (0.0):        {len(zeros):>5} ({len(zeros)/total_q*100:.1f}%)")
    print(f"Low F1 (0.0 - 0.5):   {len(lows):>5} ({len(lows)/total_q*100:.1f}%)")
    print(f"High F1 (>= 0.5):     {len(highs):>5} ({len(highs)/total_q*100:.1f}%)")
    print()

    # Per-category breakdown
    CAT_NAMES = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop', 5: 'adversarial'}
    cat_data = defaultdict(list)
    for q in data:
        cat_data[q['category']].append(q)

    print(f"{'Category':<15} | {'Total':<6} | {'F1=0':<6} | {'0<F1<0.5':<10} | {'F1>=0.5':<8} | {'Oracle Hit %':<12} | {'Oracle Hit but F1=0':<20}")
    print("-" * 88)
    for cat in [1, 2, 3, 4, 5]:
        qs = cat_data[cat]
        c_zeros = [q for q in qs if q['f1'] == 0.0]
        c_lows = [q for q in qs if 0.0 < q['f1'] < 0.5]
        c_highs = [q for q in qs if q['f1'] >= 0.5]
        ora_hits = [q for q in qs if q.get('oracle', False)]
        ora_hit_zero = [q for q in c_zeros if q.get('oracle', False)]

        print(f"{CAT_NAMES[cat]:<15} | {len(qs):>5} | {len(c_zeros):>5} | {len(c_lows):>8} | {len(c_highs):>7} | {len(ora_hits)/len(qs)*100:>10.1f}% | {len(ora_hit_zero):>18} ({len(ora_hit_zero)/len(c_zeros)*100 if c_zeros else 0:.1f}%)")
    print()

if __name__ == '__main__':
    main()
