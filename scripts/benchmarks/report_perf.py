"""Aggregate prompt-token + latency stats from a pipeline result JSON.

Usage:
  python scripts/benchmarks/report_perf.py --input benchmark_results/pipeline_v4/lme_dev250_v4_pipeline.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def pct(data: list[float], q: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, list):
        results = d
        d = {"results": results, "total_questions": len(results),
             "correct_answers": sum(1 for r in results if r.get("is_correct"))}
    else:
        results = d.get("results", [])

    print(f"File: {args.input}")
    print(f"Accuracy: {d.get('correct_answers')}/{d.get('total_questions')} "
          f"({d.get('accuracy', 0):.1%})" if "accuracy" in d else "")
    print(f"Elapsed: {d.get('elapsed_seconds', 0):.1f}s  "
          f"Model: {d.get('model', '?')}")

    toks = [r.get("prompt_tokens", 0) for r in results]
    lats = [r.get("latency_ms", 0) for r in results]
    if toks:
        print(f"\n[TOKENS] mean={statistics.mean(toks):.0f} "
              f"median={statistics.median(toks):.0f} "
              f"p50={pct(toks, .5):.0f} p90={pct(toks, .9):.0f} "
              f"p95={pct(toks, .95):.0f} max={max(toks)} "
              f"total={sum(toks)}")
    if lats:
        print(f"[LAT ms] mean={statistics.mean(lats):.0f} "
              f"median={statistics.median(lats):.0f} "
              f"p50={pct(lats, .5):.0f} p90={pct(lats, .9):.0f} "
              f"p95={pct(lats, .95):.0f} max={max(lats):.0f} "
              f"total={sum(lats)/1000:.1f}s")

    zero = sum(1 for r in results if r.get("deterministic_override"))
    print(f"Deterministic overrides: {zero}/{len(results)}")

    by_type: dict[str, dict] = defaultdict(lambda: {"t": [], "l": [], "n": 0, "c": 0})
    for r in results:
        t = r.get("question_type", "?")
        by_type[t]["t"].append(r.get("prompt_tokens", 0))
        by_type[t]["l"].append(r.get("latency_ms", 0))
        by_type[t]["n"] += 1
        by_type[t]["c"] += 1 if r.get("is_correct") else 0
    print(f"\n{'Type':<28} {'Acc':>7} {'n':>4} {'tok mean':>9} {'tok p90':>8} {'lat mean':>9} {'lat p90':>8}")
    for t, v in sorted(by_type.items()):
        print(f"{t:<28} {v['c']/v['n']:>6.1%} {v['n']:>4} "
              f"{statistics.mean(v['t']):>9.0f} {pct(v['t'], .9):>8.0f} "
              f"{statistics.mean(v['l']):>9.0f} {pct(v['l'], .9):>8.0f}")

    # Token x latency heavy hitters
    print("\n[Top-10 slowest]")
    for r in sorted(results, key=lambda x: x.get("latency_ms", 0), reverse=True)[:10]:
        print(f"  {r['question_id']} ({r.get('question_type', '?')[:14]}): "
              f"{r.get('latency_ms', 0):.0f}ms tok={r.get('prompt_tokens', 0)} "
              f"{'OK' if r.get('is_correct') else 'FAIL'} "
              f"{str(r.get('predicted_answer', ''))[:40]}")


if __name__ == "__main__":
    main()
