"""Pull real failing questions per category with their retrieved context.

Reads a runner artefact (predictions) + the matching context cache and prints,
for every failure in a category, the question, ground truth, reader prediction,
oracle flag, and the context lines most lexically related to the question/answer.

Usage:
  uv run python scripts/benchmarks/_failure_forensics.py --run cap24_coder7b --per-cat 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
NAMES = {1: "MULTI-HOP", 2: "TEMPORAL", 3: "OPEN-DOMAIN", 4: "SINGLE-HOP"}
STOP = {
    "what", "when", "where", "which", "who", "whom", "how", "why", "did", "does",
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "are",
    "his", "her", "they", "them", "she", "you", "your", "have", "has", "had",
    "not", "any", "all", "one", "two", "into", "out", "about", "after", "before",
    "before", "during", "both", "there", "then", "them", "been", "would", "could",
    "should", "will", "its", "it's", "over", "than", "then", "also", "ever",
}


def toks(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9']+", text.lower())
        if len(w) > 2 and w not in STOP
    }


def relevant_lines(context: str, seed: str, k: int = 4) -> list[str]:
    want = toks(seed)
    scored: list[tuple[int, int, str]] = []
    for i, line in enumerate(context.split("\n")):
        have = toks(line)
        score = len(want & have)
        if score:
            scored.append((-score, i, line))
    scored.sort()
    return [ln for _, _, ln in scored[:k]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cap24_coder7b")
    ap.add_argument("--cache", default="benchmark_results/locomo_context_cache.jsonl")
    ap.add_argument("--per-cat", type=int, default=5)
    args = ap.parse_args()

    run_path = REPO / "benchmark_results" / "locomo1540" / f"{args.run}.json"
    results = json.load(open(run_path, encoding="utf-8"))["results"]

    contexts: dict[str, str] = {}
    for line in open(REPO / args.cache, encoding="utf-8"):
        d = json.loads(line)
        if d.get("category") == 5:
            continue
        contexts[d["qid"]] = d["context"]

    by_cat: dict[int, list[dict]] = {}
    for r in results:
        if r["is_correct"]:
            continue
        by_cat.setdefault(r["category"], []).append(r)

    for cat in sorted(by_cat):
        rows = by_cat[cat]
        print("=" * 100)
        print(f"### {NAMES.get(cat, cat)}  -- {len(rows)} failures of "
              f"{sum(1 for r in results if r['category'] == cat)} "
              f"({100 * len(rows) / max(1, sum(1 for r in results if r['category'] == cat)):.1f}% wrong)")
        print("=" * 100)
        for r in rows[: args.per_cat]:
            ctx = contexts.get(r["question_id"], "")
            seed = f"{r['question']} {r['ground_truth']}"
            print(f"\n[{r['question_id']}] oracle_recall={r['oracle_recall']} "
                  f"words={len(ctx.split())}")
            print(f"  Q  : {r['question']}")
            print(f"  GT : {r['ground_truth']}")
            print(f"  PRED: {r['prediction']}")
            print("  CONTEXT (most relevant lines):")
            for ln in relevant_lines(ctx, seed):
                print(f"    | {ln[:300]}")
            if not r["oracle_recall"]:
                print("    !! EVIDENCE NOT IN CONTEXT (Layer-1 miss)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
