#!/usr/bin/env python3
"""Build a LoCoMo MSC context cache for a given compiler configuration.

The frozen ``benchmark_results/locomo_context_cache.jsonl`` records the exact
context Lethe produced for each question under the baseline configuration.
This tool rebuilds that file under a different configuration so the *reader*
stays frozen (same prompts, same model, same scorer) and only Lethe's context
construction varies -- which is what makes a Layer-1 change measurable at the
end-to-end layer without confounding the Reader.

Usage:
  uv run python scripts/benchmarks/build_context_cache.py \
      --window-cap 96 --out benchmark_results/locomo_context_cache_cap96.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler  # noqa: E402
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a LoCoMo MSC context cache")
    ap.add_argument("--window-cap", type=int, default=24)
    ap.add_argument("--rescue-order", choices=["relevance", "pool"], default="pool")
    ap.add_argument("--condense", action="store_true",
                    help="enable the high-density content condenser")
    ap.add_argument("--quote-mode", choices=["keep", "drop", "provenance", "redundant"],
                    default="keep",
                    help="what to do with the duplicated (In reply to ...) quote")
    ap.add_argument("--compact-header", action="store_true",
                    help="rewrite provenance headers to the compact ISO form")
    ap.add_argument("--out", default="benchmark_results/locomo_context_cache_variant.jsonl")
    ap.add_argument("--limit-conv", type=int, default=None)
    args = ap.parse_args()

    adapter = LoCoMoAdapter()
    compiler = MinimumSufficientContextCompiler(
        selection_window_cap=args.window_cap,
        rescue_order=args.rescue_order,
        condense=args.condense,
        quote_mode=args.quote_mode,
        compact_header=args.compact_header,
    )
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    t0 = time.perf_counter()
    with open(out, "w", encoding="utf-8") as fh:
        for conv_idx in range(10):
            _, questions, ir = adapter.load_conversation(conv_idx)
            for q in questions:
                pcc = compiler.compile(q.question, ir)
                fh.write(json.dumps({
                    "qid": q.question_id,
                    "question": q.question,
                    "category": q.category,
                    "context": pcc.context_text,
                    "token_cost": pcc.token_cost,
                    "is_abstention": bool(pcc.is_abstention),
                }, ensure_ascii=False) + "\n")
                n += 1
            print(f"  conv {conv_idx}: {n} rows  ({time.perf_counter() - t0:.0f}s)", flush=True)
            if args.limit_conv is not None and conv_idx + 1 >= args.limit_conv:
                break

    print(f"\nwrote {n} rows -> {args.out}  ({time.perf_counter() - t0:.0f}s)")
    print(f"config: selection_window_cap={args.window_cap} rescue_order={args.rescue_order} "
          f"condense={args.condense} quote_mode={args.quote_mode} "
          f"compact_header={args.compact_header}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
