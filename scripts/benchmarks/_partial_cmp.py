#!/usr/bin/env python3
"""Compare a *running* benchmark arm against a baseline on the same questions.

A partial run's headline accuracy is not comparable with a completed run's
headline: the runner evaluates in cache order, so "120/321 so far" says nothing
about the final number.  This tool restricts a baseline artefact to exactly the
questions the running arm has already answered, so the two are always paired.

Usage:
  uv run python scripts/benchmarks/_partial_cmp.py --n 120 \
      --baseline temporal321_rules --cache benchmark_results/locomo_context_cache_rules.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description="Subset-restricted baseline comparison")
    ap.add_argument("--cache", default="benchmark_results/locomo_context_cache.jsonl")
    ap.add_argument("--baseline", default="locomo_instruct_full")
    ap.add_argument("--category", type=int, default=2,
                    help="restrict the cache prefix to this LoCoMo category")
    ap.add_argument("--n", type=int, default=120,
                    help="how many questions the running arm has answered so far")
    args = ap.parse_args()

    order: list[str] = []
    with open(REPO / args.cache, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if args.category and row.get("category") != args.category:
                continue
            order.append(row["qid"])
            if len(order) >= args.n:
                break
    wanted = set(order)

    path = REPO / "benchmark_results" / "locomo1540" / f"{args.baseline}.json"
    if not path.exists():
        print(f"!! baseline artefact {path.name} not found")
        return 2
    with open(path, encoding="utf-8") as fh:
        results = json.load(fh)["results"]
    subset = [r for r in results if r["question_id"] in wanted]
    correct = sum(1 for r in subset if r["is_correct"])
    print(f"baseline {args.baseline} on the SAME first {len(order)} "
          f"category-{args.category} qids: {correct}/{len(subset)} = "
          f"{100 * correct / max(1, len(subset)):.1f}%")
    print("compare this with the running arm's own accuracy at the same n - "
          "not with the baseline's headline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
