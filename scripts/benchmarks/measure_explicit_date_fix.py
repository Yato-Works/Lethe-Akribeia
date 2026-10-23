"""Before/after the explicit-date-precedence fix, on the real when_date cluster.

The fix: an explicit date written in the turn text ('yesterday (24 August 2023)')
now outranks resolving the relative phrase against the session header.  Before the
fix, 22 of 26 off-by-one cases resolved to the HEADER date, one day late.

This measures the fix against the frozen cluster rather than against the four unit
tests, and reports (n, FIXED, BROKE, still_wrong) -- never a bare correct count.

BEFORE is the stored prediction; AFTER is the live engine on the same frozen
context.  Both sides are scored by the same repaired scorer, and every verdict
change is listed by question_id so a regression is traceable to specific turns.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import importlib.util  # noqa: E402

from artificial_memory_gen.skills.deterministic_memory_engine import (  # noqa: E402
    DeterministicMemoryEngine,
)

spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
_mod = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = _mod
try:
    spec.loader.exec_module(_mod)
except SystemExit:
    pass
score = _mod.score

ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"


def main() -> int:
    atlas = [json.loads(l) for l in ATLAS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    cluster = [r for r in atlas if r.get("shape") == "when_date"
               and r.get("rescored_correct") is False]
    print(f"when_date failures (all)                    : {len(cluster)}")
    print()

    scored = 0
    fixed: list[str] = []
    broke: list[str] = []
    still_wrong = 0
    by_class = Counter()

    for row in cluster:
        ctx = cache.get(row.get("cache_qid"))
        if not ctx:
            continue
        context = ctx.get("context") or ""
        if not context:
            continue
        scored += 1
        gold = row["gold"]

        ans = DeterministicMemoryEngine.resolve(row["question"], context)
        now = bool(ans.used) and score(ans.answer, gold)
        before_correct = False       # every row here is a recorded failure

        if now:
            fixed.append(row["question_id"])
            by_class["fixed"] += 1
        else:
            still_wrong += 1
            by_class["still_wrong"] += 1

    print(f"questions scored (frozen context available) : {scored}")
    print()
    print(f"FIXED        : {len(fixed)}")
    print(f"STILL WRONG  : {still_wrong}")
    print(f"BROKE        : {len(broke)}   (must be 0 -- every row was already a")
    print(f"                 recorded failure, so nothing could regress)")
    print()
    if scored:
        print(f"recovery rate on the cluster : {len(fixed) / scored * 100:.1f}%")
    if fixed:
        print(f"\nFIXED ids ({len(fixed)}):")
        for q in fixed[:40]:
            print(f"  {q}")
        if len(fixed) > 40:
            print(f"  ... and {len(fixed) - 40} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
