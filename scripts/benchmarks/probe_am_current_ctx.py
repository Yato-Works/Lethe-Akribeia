"""Probe: gold-reachability of the "AM current" player's compiled contexts.

0 LLM calls.  For every arena question, compile the R7 context exactly as
AMCurrentPlayer does and report, per category, whether the ground-truth terms
are reachable.  Baseline comparison: the runtime's lexical channel measured
75% gold-in-context over all 200 questions (V4_PROGRAM.md 4.13).

Run:
    python scripts/benchmarks/probe_am_current_ctx.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from artificial_memory.research.benchmarks.arena import ARENA_VERSION, build_dataset  # noqa: E402
from am_current_player import build_pipeline, compile_r7_context  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="AM current gold-reachability probe")
    parser.add_argument("--output", default="benchmark_results/am_current/probe_ctx.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    dataset = build_dataset(version=ARENA_VERSION, created_at=datetime.now().isoformat())
    all_turns = []
    for scenario in dataset.scenarios:
        for turn in scenario.turns:
            all_turns.append((scenario.scenario_id, str(scenario.day),
                              turn.speaker, turn.content))
    pipeline = build_pipeline()
    pipeline[5].build(all_turns)
    fact_index = pipeline[5]

    rows = []
    per_cat = defaultdict(Counter)
    t0 = time.perf_counter()
    for question in dataset.questions:
        ctx, meta = compile_r7_context(
            pipeline, all_turns, question.question, fact_index=fact_index)
        ctx_lower = ctx.lower()
        groups_hit = [any(term.lower() in ctx_lower for term in group)
                      for group in question.ground_truth]
        reachable = bool(groups_hit) and all(groups_hit)
        cat = question.category.value
        if not question.expected_abstention:
            per_cat[cat]["answerable"] += 1
            per_cat[cat]["reachable"] += 1 if reachable else 0
        else:
            per_cat[cat]["abstention"] += 1
            if reachable:
                per_cat[cat]["abstention_with_evidence"] += 1
        rows.append({
            "question_id": question.question_id,
            "category": cat,
            "expected_abstention": question.expected_abstention,
            "reachable": reachable,
            **meta,
        })
    elapsed = time.perf_counter() - t0

    print(f"\ncompiled {len(rows)} contexts in {elapsed:.1f}s "
          f"({elapsed / max(1, len(rows)) * 1000:.0f} ms per question)")
    print(f"{'category':<28} {'answerable':>10} {'reachable':>10} {'recall':>8}")
    total = Counter()
    for cat in sorted(per_cat):
        c = per_cat[cat]
        total += c
        if c["answerable"]:
            print(f"{cat:<28} {c['answerable']:>10} {c['reachable']:>10} "
                  f"{100 * c['reachable'] / c['answerable']:>7.1f}%")
    print("-" * 58)
    if total["answerable"]:
        print(f"{'ALL (answerable)':<28} {total['answerable']:>10} "
              f"{total['reachable']:>10} "
              f"{100 * total['reachable'] / total['answerable']:>7.1f}%")
    mean_budget = sum(r["token_budget"] for r in rows) / len(rows)
    mean_ctx = sum(r["web_ctx_tokens"] for r in rows) / len(rows)
    over = sum(1 for r in rows if r["web_ctx_tokens"] > 2000)
    rec = sum(r["recompiles"] for r in rows)
    print(f"mean token budget: {mean_budget:.0f}   mean ctx tokens (arena est): {mean_ctx:.0f}")
    print(f"contexts over 2,000 arena tokens: {over}   recompiles: {rec}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rows": rows,
        "per_category": {k: dict(v) for k, v in per_cat.items()},
        "elapsed_s": elapsed,
    }, indent=2), encoding="utf-8")
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
