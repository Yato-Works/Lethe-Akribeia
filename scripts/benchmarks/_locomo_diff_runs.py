"""Scratch comparison of two LoCoMo result JSONs (read-only).

Reports per-category gain/loss counts and the exact questions that flipped, so a
prompt change can be judged as a mechanism (category-wide) rather than by a single
aggregate percentage.

Usage:
    uv run python scripts/benchmarks/_locomo_diff_runs.py A.json B.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["question_id"]: r for r in data["results"]}


def main() -> None:
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = load(a_path), load(b_path)
    shared = [q for q in a if q in b]

    print(f"A = {a_path}  n={len(a)}")
    print(f"B = {b_path}  n={len(b)}")
    print(f"shared questions: {len(shared)}")

    acc_a = sum(1 for q in shared if a[q]["is_correct"]) / len(shared)
    acc_b = sum(1 for q in shared if b[q]["is_correct"]) / len(shared)
    print(f"\nA accuracy {acc_a*100:.1f}%  ->  B accuracy {acc_b*100:.1f}%  ({(acc_b-acc_a)*100:+.1f}pp)")

    per_cat = defaultdict(lambda: {"n": 0, "a": 0, "b": 0})
    gained, lost = [], []
    for q in shared:
        cat = CAT.get(a[q]["category"], str(a[q]["category"]))
        per_cat[cat]["n"] += 1
        per_cat[cat]["a"] += int(bool(a[q]["is_correct"]))
        per_cat[cat]["b"] += int(bool(b[q]["is_correct"]))
        if b[q]["is_correct"] and not a[q]["is_correct"]:
            gained.append(q)
        elif a[q]["is_correct"] and not b[q]["is_correct"]:
            lost.append(q)

    print("\ncategory                 n      A      B     delta")
    for cat, s in sorted(per_cat.items()):
        print(f"{cat:<22}{s['n']:>4}  {s['a']/s['n']*100:6.1f}% {s['b']/s['n']*100:6.1f}%"
              f"  {(s['b']-s['a'])/s['n']*100:+6.1f}pp")
    print(f"\ngained: {len(gained)}  lost: {len(lost)}  net: {len(gained)-len(lost):+d}")

    label = "lost" if lost else "gained"
    for q in (lost or gained)[:5]:
        print(f"  {label} {q} [{CAT.get(a[q]['category'])}]")
        print(f"    q   : {str(a[q]['question'])[:80]}")
        print(f"    gt  : {str(a[q]['ground_truth'])[:60]}")
        print(f"    A   : {str(a[q]['prediction'])[:80]}")
        print(f"    B   : {str(b[q]['prediction'])[:80]}")


if __name__ == "__main__":
    main()
