import json
from collections import defaultdict

def sample_cases(cat_id, cat_name, qs, n=8):
    print(f"================================================================================")
    print(f"=== FAILURE SAMPLES: Category {cat_id} ({cat_name.upper()}) ===")
    print(f"================================================================================")
    
    # 1. Oracle Hit but F1 == 0 (Evidence present, reader failed completely)
    ora_zero = [q for q in qs if q['f1'] == 0.0 and q.get('oracle', False)]
    print(f"\n--- [Sub-bucket 1] Oracle HIT but F1 == 0.0 ({len(ora_zero)} cases) ---")
    for i, q in enumerate(ora_zero[:n]):
        print(f"[{i+1}] QID: {q['qid']}")
        print(f"    Q:    {q['question']}")
        print(f"    GT:   {q['gt']}")
        print(f"    Pred: {q['pred']}")
        print()

    # 2. Oracle Miss and F1 == 0 (Retrieval failed)
    ora_miss_zero = [q for q in qs if q['f1'] == 0.0 and not q.get('oracle', False)]
    print(f"\n--- [Sub-bucket 2] Oracle MISS and F1 == 0.0 ({len(ora_miss_zero)} cases) ---")
    for i, q in enumerate(ora_miss_zero[:min(n, 4)]):
        print(f"[{i+1}] QID: {q['qid']}")
        print(f"    Q:    {q['question']}")
        print(f"    GT:   {q['gt']}")
        print(f"    Pred: {q['pred']}")
        print()

    # 3. Partial F1 (0 < F1 < 0.5) (Formatting / dilution / partial enumeration)
    lows = [q for q in qs if 0.0 < q['f1'] < 0.5]
    print(f"\n--- [Sub-bucket 3] Partial Score (0 < F1 < 0.5) ({len(lows)} cases) ---")
    for i, q in enumerate(lows[:min(n, 4)]):
        print(f"[{i+1}] QID: {q['qid']} (F1: {q['f1']*100:.1f}%)")
        print(f"    Q:    {q['question']}")
        print(f"    GT:   {q['gt']}")
        print(f"    Pred: {q['pred']}")
        print()

def main():
    with open('benchmark_results/_official_scoring/f1_by_question_full_v25.json', encoding='utf-8') as f:
        data = json.load(f)

    CAT_NAMES = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop', 5: 'adversarial'}
    cat_data = defaultdict(list)
    for q in data:
        cat_data[q['category']].append(q)

    # Let's inspect Cat 1, Cat 3, Cat 2, Cat 4
    for cat in [1, 3, 4, 2, 5]:
        sample_cases(cat, CAT_NAMES[cat], cat_data[cat], n=6)

if __name__ == '__main__':
    main()
