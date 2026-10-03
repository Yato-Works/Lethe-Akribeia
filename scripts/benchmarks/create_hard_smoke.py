import json
from collections import defaultdict
from pathlib import Path

def create_hard_smoke():
    repo = Path(".")
    v25_path = repo / "benchmark_results" / "_official_scoring" / "f1_by_question_full_v25.json"
    with open(v25_path, encoding="utf-8") as f:
        data = json.load(f)

    # Group by category
    cat_items = defaultdict(list)
    for q in data:
        cat_items[q["category"]].append(q)

    hard_selected = []
    # 12 questions per category = 60 questions total
    TARGET_PER_CAT = 12

    CAT_NAMES = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop', 5: 'adversarial'}

    print("=== Selecting Worst Offenders for Hard Smoke (12 per category) ===")

    for cat in [1, 2, 3, 4, 5]:
        items = cat_items[cat]
        # Prioritize: Oracle Hit True & F1 == 0.0 (Reader-side failures where evidence IS present!)
        ora_hit_zeros = [q for q in items if q["f1"] == 0.0 and q.get("oracle", False)]
        ora_miss_zeros = [q for q in items if q["f1"] == 0.0 and not q.get("oracle", False)]
        low_partials = [q for q in items if 0.0 < q["f1"] < 0.3 and q.get("oracle", False)]

        selected = []
        # First take up to 10 oracle-hit zeros
        selected.extend(ora_hit_zeros[:10])
        # Then fill with low partials or oracle miss zeros
        remaining = TARGET_PER_CAT - len(selected)
        if remaining > 0:
            selected.extend(low_partials[:remaining])
        remaining = TARGET_PER_CAT - len(selected)
        if remaining > 0:
            selected.extend(ora_miss_zeros[:remaining])
        remaining = TARGET_PER_CAT - len(selected)
        if remaining > 0:
            # If still remaining, just take lowest F1
            sorted_items = sorted(items, key=lambda x: x["f1"])
            for q in sorted_items:
                if q not in selected:
                    selected.append(q)
                if len(selected) == TARGET_PER_CAT:
                    break

        selected = selected[:TARGET_PER_CAT]
        hard_selected.extend(selected)

        mean_f1 = sum(q["f1"] for q in selected) / len(selected) * 100
        ora_rate = sum(1 for q in selected if q.get("oracle", False)) / len(selected) * 100
        print(f"Cat {cat} ({CAT_NAMES[cat]}): {len(selected)} Qs | Baseline F1: {mean_f1:.2f}% | Oracle Hit: {ora_rate:.1f}%")

    out_qids = [q["qid"] for q in hard_selected]
    out_path = repo / "benchmark_results" / "hard_smoke_60q.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_qids, f, indent=2)

    total_mean_f1 = sum(q["f1"] for q in hard_selected) / len(hard_selected) * 100
    print("-" * 65)
    print(f"Total Hard Smoke: {len(hard_selected)} questions saved to {out_path}")
    print(f"Initial Hard Smoke Baseline F1: {total_mean_f1:.2f}% (A true stress-test!)")

if __name__ == "__main__":
    create_hard_smoke()
