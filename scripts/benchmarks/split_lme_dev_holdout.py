"""
Stratified deterministic split of LongMemEval (500 items) into dev (250) and holdout (250).

Uses sha256(qid) within each question_type to ensure exact 50/50 balance
across all 6 memory capabilities without any randomness or human bias.
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"
OUT_DIR = REPO / "benchmark_results" / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha_score(qid: str) -> str:
    return hashlib.sha256(qid.encode("utf-8")).hexdigest()


def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        items = json.load(f)

    print(f"Total items loaded: {len(items)}")

    # Group by question_type
    by_type = defaultdict(list)
    for it in items:
        by_type[it["question_type"]].append(it)

    dev_items = []
    holdout_items = []

    dev_qids = []
    holdout_qids = []

    print("\n--- Stratified Split Summary ---")
    print(f"{'question_type':<28} | {'Total':<6} | {'Dev':<6} | {'Holdout':<8}")
    print("-" * 55)

    # To balance exactly 250 / 250 when two categories have odd counts (133 and 133):
    # We assign 67 to dev for one, and 66 to dev for the other.
    odd_toggle = True
    for qtype, qlist in sorted(by_type.items()):
        qlist_sorted = sorted(qlist, key=lambda x: sha_score(x["question_id"]))
        if len(qlist_sorted) % 2 != 0:
            n_dev = (len(qlist_sorted) // 2) + (1 if odd_toggle else 0)
            odd_toggle = not odd_toggle
        else:
            n_dev = len(qlist_sorted) // 2

        dev_part = qlist_sorted[:n_dev]
        holdout_part = qlist_sorted[n_dev:]

        dev_items.extend(dev_part)
        holdout_items.extend(holdout_part)

        dev_qids.extend([x["question_id"] for x in dev_part])
        holdout_qids.extend([x["question_id"] for x in holdout_part])

        print(f"{qtype:<28} | {len(qlist):<6} | {len(dev_part):<6} | {len(holdout_part):<8}")

    print("-" * 55)
    print(f"{'TOTAL':<28} | {len(items):<6} | {len(dev_items):<6} | {len(holdout_items):<8}\n")

    # Save qids and full items
    (OUT_DIR / "lme_dev_qids.json").write_text(json.dumps(dev_qids, indent=2), encoding="utf-8")
    (OUT_DIR / "lme_holdout_qids.json").write_text(json.dumps(holdout_qids, indent=2), encoding="utf-8")
    (OUT_DIR / "lme_dev_500_split.json").write_text(
        json.dumps({
            "split_method": "stratified_sha256_by_question_type",
            "dev_count": len(dev_items),
            "holdout_count": len(holdout_items),
            "dev_qids": dev_qids,
            "holdout_qids": holdout_qids,
        }, indent=2),
        encoding="utf-8"
    )

    print(f"✅ Saved splits to {OUT_DIR}")


if __name__ == "__main__":
    main()
