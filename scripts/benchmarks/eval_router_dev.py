"""
Evaluate QueryAnalyzer routing accuracy on LongMemEval Dev 250.

Measures classification precision, recall, and confusion matrix against
the dataset's true question_type.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"
SPLIT_PATH = REPO / "benchmark_results" / "splits" / "lme_dev_qids.json"

sys.path.insert(0, str(REPO / "src"))

from artificial_memory.compiler.query_analyzer import QueryAnalyzer, QueryType


def main():
    with open(SPLIT_PATH, encoding="utf-8") as f:
        dev_qids = set(json.load(f))

    with open(DATASET_PATH, encoding="utf-8") as f:
        all_items = json.load(f)

    dev_items = [it for it in all_items if it["question_id"] in dev_qids]

    analyzer = QueryAnalyzer()
    
    correct = 0
    total = len(dev_items)

    confusion = defaultdict(Counter)
    by_true_type = defaultdict(list)
    errors = []

    for item in dev_items:
        qid = item["question_id"]
        true_type = item["question_type"]
        q_text = item["question"]

        predicted_type = analyzer.route(q_text).value

        is_match = (predicted_type == true_type)
        if is_match:
            correct += 1
        else:
            errors.append({
                "qid": qid,
                "question": q_text,
                "true": true_type,
                "pred": predicted_type,
            })

        confusion[true_type][predicted_type] += 1
        by_true_type[true_type].append(is_match)

    acc = correct / total * 100

    print("=" * 75)
    print(f"  QUERY ANALYZER ROUTING ACCURACY ON DEV 250 (n={total})")
    print(f"  Overall Routing Accuracy: {acc:.2f}% ({correct}/{total})")
    print("=" * 75)

    print("\n--- Per-Class Metrics ---")
    print(f"{'Question Type':<28} | {'Total':<6} | {'Correct':<8} | {'Recall':<8}")
    print("-" * 58)
    for qt, matches in sorted(by_true_type.items()):
        c = sum(matches)
        tot = len(matches)
        print(f"{qt:<28} | {tot:<6} | {c:<8} | {c/tot*100:6.1f}%")

    print("\n--- Confusion Matrix (Row=True, Col=Predicted) ---")
    types = sorted(by_true_type.keys())
    header = f"{'True / Pred':<26} " + "".join(f"{t[:8]:>10}" for t in types)
    print(header)
    print("-" * len(header))
    for t_row in types:
        row_str = f"{t_row:<26} "
        for t_col in types:
            cnt = confusion[t_row][t_col]
            row_str += f"{cnt:>10}"
        print(row_str)

    if errors:
        print(f"\n--- Misclassified Sample Errors (Total {len(errors)}) ---")
        for err in errors[:10]:
            print(f"Q:    {err['question']}")
            print(f"True: {err['true']:<26} | Pred: {err['pred']}")
            print("-" * 50)


if __name__ == "__main__":
    main()
