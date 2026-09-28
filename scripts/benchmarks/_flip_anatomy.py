"""Why did the Option-2 reader run lose questions?  Classify the flips."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
NAMES = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop"}
ISO = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")


def load(tag: str) -> dict[str, dict]:
    p = REPO / "benchmark_results" / "locomo1540" / f"{tag}.json"
    return {r["question_id"]: r for r in json.load(open(p, encoding="utf-8"))["results"]}


def refusal(text) -> bool:
    return "no information available" in str(text).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="cap24_coder7b")
    ap.add_argument("--b", default="opt2b_coder7b")
    args = ap.parse_args()
    A = load(args.a)
    B = load(args.b)
    lost = [k for k in A if A[k]["is_correct"] and not B[k]["is_correct"]]
    gained = [k for k in A if not A[k]["is_correct"] and B[k]["is_correct"]]

    print(f"LOST {len(lost)} | B prediction has ISO date: "
          f"{sum(1 for k in lost if ISO.search(str(B[k]['prediction'])))}")
    print(f"GAINED {len(gained)} | B prediction has ISO date: "
          f"{sum(1 for k in gained if ISO.search(str(B[k]['prediction'])))}")
    print("lost by category:", dict(Counter(NAMES[A[k]['category']] for k in lost)))
    print("gained by category:", dict(Counter(NAMES[A[k]['category']] for k in gained)))
    print()

    print("--- LOST with an ISO date in the new prediction")
    for k in lost:
        pred = str(B[k]["prediction"])
        if not ISO.search(pred):
            continue
        print(f"[{k}] {NAMES[A[k]['category']]}")
        print(f"   Q    : {A[k]['question']}")
        print(f"   GT   : {A[k]['ground_truth']}")
        print(f"   old  : {A[k]['prediction']}")
        print(f"   new  : {pred}")
    print()

    new_ref = [k for k in lost if refusal(B[k]["prediction"]) and not refusal(A[k]["prediction"])]
    old_ref = [k for k in gained if refusal(A[k]["prediction"]) and not refusal(B[k]["prediction"])]
    print(f"new refusals (old was right): {len(new_ref)}")
    for k in new_ref[:8]:
        print(f"   [{k}] {NAMES[A[k]['category']]} Q={A[k]['question'][:70]} "
              f"GT={str(A[k]['ground_truth'])[:40]}")
    print(f"refusals recovered (new is right): {len(old_ref)}")
    for k in old_ref[:8]:
        print(f"   [{k}] {NAMES[A[k]['category']]} Q={A[k]['question'][:70]} "
              f"GT={str(A[k]['ground_truth'])[:40]}")

    # Non-refusal, non-ISO losses: is the answer still in the context?
    rest = [k for k in lost if k not in new_ref and not ISO.search(str(B[k]['prediction']))]
    print(f"\nother losses: {len(rest)}  (by category "
          f"{dict(Counter(NAMES[A[k]['category']] for k in rest))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
