#!/usr/bin/env python3
"""Layer 1 (ORACLE RECALL) evaluation harness.

Runs Lethe's context construction ONLY -- no reader, no LLM calls -- and
measures whether the ground-truth evidence ends up in the built context.

This is the cheapest possible validation layer for Lethe: it isolates
retrieval/selection from reasoning, costs $0, and is the layer Lethe is
actually judged on.  Reader accuracy must never be mixed into this number.

Usage:
  uv run python scripts/benchmarks/oracle_recall_eval.py                 # all convs
  uv run python scripts/benchmarks/oracle_recall_eval.py --limit-conv 3
  uv run python scripts/benchmarks/oracle_recall_eval.py --out benchmark_results/retrieval/base.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler  # noqa: E402
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Layer 1 oracle-recall evaluation (no reader)")
    ap.add_argument("--limit-conv", type=int, default=None, help="only evaluate the first N conversations")
    ap.add_argument("--conv", type=int, action="append", default=None, help="specific conversation index")
    ap.add_argument("--exclude-adversarial", action="store_true", default=True)
    ap.add_argument("--include-adversarial", action="store_true")
    ap.add_argument("--window-cap", type=int, default=24,
                    help="MinimumSufficientContextCompiler.selection_window_cap")
    ap.add_argument("--rescue-order", choices=["relevance", "pool"], default="pool",
                    help="candidate ordering used by _apply_evidence_widening")
    ap.add_argument("--condense", action="store_true",
                    help="enable the high-density content condenser (filler/sign-off strip)")
    ap.add_argument("--quote-mode", choices=["keep", "drop", "provenance", "redundant"],
                    default="keep",
                    help="what to do with the duplicated (In reply to ...) quote; "
                         "'redundant' drops it only when the payload appears elsewhere")
    ap.add_argument("--compact-header", action="store_true",
                    help="rewrite [D7:22 on 8:56 pm on 20 July, 2023] -> [D7:22 20 July, 2023 20:56]")
    ap.add_argument("--out", default="benchmark_results/retrieval/oracle_recall_eval.json")
    args = ap.parse_args()

    adapter = LoCoMoAdapter()
    compiler = MinimumSufficientContextCompiler(
        selection_window_cap=args.window_cap,
        rescue_order=args.rescue_order,
        condense=args.condense,
        quote_mode=args.quote_mode,
        compact_header=args.compact_header,
    )
    print(f"compiler: selection_window_cap={args.window_cap} rescue_order={args.rescue_order} "
          f"condense={args.condense} quote_mode={args.quote_mode} "
          f"compact_header={args.compact_header}")

    conv_range = args.conv if args.conv else list(range(10))
    if args.limit_conv:
        conv_range = conv_range[: args.limit_conv]

    rows: list[dict] = []
    t0 = time.perf_counter()
    for c_idx in conv_range:
        _, questions, ir = adapter.load_conversation(c_idx)
        for q in questions:
            if not args.include_adversarial and q.category == 5:
                continue
            t_q = time.perf_counter()
            pcc = compiler.compile(q.question, ir)
            ms = (time.perf_counter() - t_q) * 1000
            if q.evidence_ids:
                oracle = any(ev in pcc.context_text for ev in q.evidence_ids)
            else:
                oracle = True  # nothing to find: not a retrieval failure
            rows.append({
                "question_id": q.question_id,
                "category": q.category,
                "oracle_recall": bool(oracle),
                "coverage_sufficient": bool(pcc.certificate.is_sufficient),
                "evidence_ids": list(q.evidence_ids),
                "evidence_hits": [ev for ev in (q.evidence_ids or []) if ev in pcc.context_text],
                "token_cost": pcc.token_cost,
                "is_abstention": bool(pcc.is_abstention),
                "compile_ms": round(ms, 1),
            })
        print(f"  conv {c_idx}: cumulative n={len(rows)} "
              f"oracle={100 * sum(r['oracle_recall'] for r in rows) / len(rows):.2f}%", flush=True)

    elapsed = time.perf_counter() - t0
    n = len(rows)
    hits = [r for r in rows if r["oracle_recall"]]
    by_cat: dict[int, dict[str, float]] = defaultdict(lambda: {"n": 0, "hit": 0})
    for r in rows:
        by_cat[r["category"]]["n"] += 1
        by_cat[r["category"]]["hit"] += int(r["oracle_recall"])

    summary = {
        "config": {"selection_window_cap": args.window_cap,
                   "rescue_order": args.rescue_order,
                   "condense": args.condense,
                   "quote_mode": args.quote_mode,
                   "compact_header": args.compact_header,
                   "evidence_widening": compiler.evidence_widening},
        "n": n,
        "oracle_recall": round(100 * len(hits) / n, 2),
        "misses": n - len(hits),
        "mean_tokens": round(statistics.mean(r["token_cost"] for r in rows), 1),
        "median_tokens": round(statistics.median(r["token_cost"] for r in rows), 1),
        "max_tokens": max(r["token_cost"] for r in rows),
        "abstentions": sum(r["is_abstention"] for r in rows),
        "mean_compile_ms": round(statistics.mean(r["compile_ms"] for r in rows), 1),
        "elapsed_s": round(elapsed, 1),
        "per_category": {
            CATEGORY_NAMES.get(c, str(c)): {
                "n": int(v["n"]),
                "oracle_recall": round(100 * v["hit"] / v["n"], 2),
            }
            for c, v in sorted(by_cat.items())
        },
        # Partial credit: how many of the annotated evidence ids were found.
        "per_evidence_id_recall": round(
            100 * sum(len(r["evidence_hits"]) for r in rows)
            / max(1, sum(len(r["evidence_ids"]) for r in rows)), 2),
    }

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    print("\n=== LAYER 1: ORACLE RECALL (no reader involved) ===")
    # Does the compiler's own coverage verdict predict an oracle miss?  If it does,
    # the selection window can be widened only for the questions that need it.
    n_sf = sum(1 for r in rows if r["coverage_sufficient"])
    n_ins = n - n_sf
    miss_when_sf = sum(1 for r in rows if r["coverage_sufficient"] and not r["oracle_recall"])
    miss_when_ins = sum(1 for r in rows if not r["coverage_sufficient"] and not r["oracle_recall"])
    if n_ins:
        print("  coverage verdict vs oracle miss:")
        print(f"    sufficient  n={n_sf:5d}  miss={miss_when_sf:4d} ({100 * miss_when_sf / max(1, n_sf):5.1f}%)")
        print(f"    insufficient n={n_ins:5d}  miss={miss_when_ins:4d} "
              f"({100 * miss_when_ins / max(1, n_ins):5.1f}%)")
        if miss_when_ins + miss_when_sf:
            print(f"    miss captured by 'insufficient': "
                  f"{100 * miss_when_ins / (miss_when_ins + miss_when_sf):.1f}% "
                  f"of {miss_when_ins + miss_when_sf} misses, "
                  f"flagging {100 * n_ins / n:.1f}% of questions")
    print(f"  questions              : {summary['n']}")
    print(f"  oracle recall           : {summary['oracle_recall']}%  (misses {summary['misses']})")
    print(f"  per-evidence-id recall  : {summary['per_evidence_id_recall']}%")
    print(f"  tokens/question         : mean {summary['mean_tokens']} / p50 {summary['median_tokens']} / max {summary['max_tokens']}")
    print(f"  abstention contexts     : {summary['abstentions']}")
    print(f"  compile latency mean    : {summary['mean_compile_ms']} ms   total {summary['elapsed_s']} s")
    for k, v in summary["per_category"].items():
        print(f"    {k:12s} n={v['n']:5d}  oracle={v['oracle_recall']:6.2f}%")
    print(f"\nwritten -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
