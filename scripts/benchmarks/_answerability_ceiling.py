#!/usr/bin/env python3
"""Answerability ceiling of a context cache: is the answer even in there?

Two different questions get conflated in post-mortems:

* *retrieval* misses - the ground-truth turn is not in the context (oracle recall);
* *reader* failures - the answer is right there and the reader still misses it.

This tool separates them per category, which is what decides whether the next
effort belongs in retrieval or in the answer path:

    oracle_miss       gold turn absent  -> retrieval's problem, and its hard cap
    GT_verbatim       the ground-truth string appears verbatim -> a perfect
                      "copy the value" answerer would get these for free
    GT_words_in_ctx   every content word of the truth appears -> reachable by a
                      paraphrasing/extractive answerer
    unreachable       neither -> for summary-shaped truths ("Running, pottery")
                      this over-counts, because the facts are present even when
                      the exact wording is not; read it as a bound, not a verdict

Usage:
  uv run python scripts/benchmarks/_answerability_ceiling.py \
      --cache benchmark_results/locomo_context_cache_rules.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if len(w) > 2}


def main() -> int:
    ap = argparse.ArgumentParser(description="Answerability ceiling of a context cache")
    ap.add_argument("--cache", default="benchmark_results/locomo_context_cache.jsonl")
    ap.add_argument("--results", default="locomo_instruct_full",
                    help="reader artefact supplying oracle flags and per-question results")
    args = ap.parse_args()

    contexts: dict[str, str] = {}
    with open(REPO / args.cache, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("category") in CATEGORY_NAMES:
                contexts[row["qid"]] = row.get("context", "")

    results_path = REPO / "benchmark_results" / "locomo1540" / f"{args.results}.json"
    if not results_path.exists():
        print(f"!! results artefact {results_path.name} not found")
        return 2
    with open(results_path, encoding="utf-8") as fh:
        results = json.load(fh)["results"]

    total: Counter = Counter()
    oracle_miss: Counter = Counter()
    verbatim: Counter = Counter()
    words_in: Counter = Counter()
    unreachable: Counter = Counter()
    for r in results:
        cat = r["category"]
        if cat not in CATEGORY_NAMES:
            continue
        total[cat] += 1
        if not r["oracle_recall"]:
            oracle_miss[cat] += 1
        context_norm = normalize(contexts.get(r["question_id"], ""))
        context_words = content_words(contexts.get(r["question_id"], ""))
        truth = str(r["ground_truth"])
        truth_norm = normalize(truth)
        if truth_norm and truth_norm in context_norm:
            verbatim[cat] += 1
        else:
            words = content_words(truth)
            if words and words <= context_words:
                words_in[cat] += 1
            else:
                unreachable[cat] += 1

    print(f"{'category':<12}{'n':>6}{'oracle_miss':>12}{'GT_verbatim':>13}"
          f"{'GT_words_in_ctx':>17}{'unreachable':>12}")
    for cat in (1, 2, 3, 4):
        print(f"{CATEGORY_NAMES[cat]:<12}{total[cat]:>6}{oracle_miss[cat]:>12}"
              f"{verbatim[cat]:>13}{words_in[cat]:>17}{unreachable[cat]:>12}")
    print(f"{'TOTAL':<12}{sum(total.values()):>6}{sum(oracle_miss.values()):>12}"
          f"{sum(verbatim.values()):>13}{sum(words_in.values()):>17}"
          f"{sum(unreachable.values()):>12}")
    ceiling = 100 * (1 - sum(oracle_miss.values()) / max(1, sum(total.values())))
    print(f"\nretrieval ceiling (1 - oracle_miss) = {ceiling:.1f}%  "
          f"<- no answerer can exceed this on this cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
