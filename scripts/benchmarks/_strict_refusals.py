"""Strict count of spurious abstentions.

An abstention is only called *spurious* when the ground-truth answer text is
verbatim present in the very context the reader was shown.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
NAMES = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop"}
AB = "no information available"
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def main() -> int:
    run = sys.argv[1] if len(sys.argv) > 1 else "cap24_coder7b"
    cache = sys.argv[2] if len(sys.argv) > 2 else "benchmark_results/locomo_context_cache.jsonl"
    res = json.load(open(REPO / "benchmark_results" / "locomo1540" / f"{run}.json",
                         encoding="utf-8"))["results"]
    ctx = {}
    for line in open(REPO / cache, encoding="utf-8"):
        row = json.loads(line)
        ctx[row["qid"]] = row["context"]

    loose = strict = ora = 0
    by_cat: Counter = Counter()
    examples: list[dict] = []
    for r in res:
        if r["is_correct"]:
            continue
        if AB not in str(r["prediction"]).lower():
            continue
        loose += 1
        if r["oracle_recall"]:
            ora += 1
        if norm(r["ground_truth"]) in norm(ctx.get(r["question_id"], "")):
            strict += 1
            by_cat[NAMES[r["category"]]] += 1
            examples.append(r)

    print(f"wrong answers that abstain          : {loose}")
    print(f"  ...and oracle_recall is True      : {ora}")
    print(f"  STRICT spurious (GT verbatim there): {strict}")
    print(f"  by category: {dict(by_cat)}")
    for r in examples[:12]:
        print(f"   [{r['question_id']}] {NAMES[r['category']]:<11} "
              f"Q={r['question'][:64]}")
        print(f"        GT={str(r['ground_truth'])[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
