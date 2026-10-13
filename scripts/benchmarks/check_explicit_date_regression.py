"""Regression check: did the explicit-date fix break anything it should not touch?

measure_explicit_date_fix.py reported FIXED 37 / BROKE 0, but BROKE 0 was
structurally guaranteed -- every row in that cohort was ALREADY a recorded
failure, so nothing there could regress.  That number proves nothing about
collateral damage.

The real risk is the OTHER 1,882 questions, especially the currently-correct ones.
A new branch placed FIRST in the date path can steal a turn that another engine was
answering correctly, or re-classify a question that previously abstained honestly.

So this re-runs the currently-correct questions plus a broad sample of the rest and
reports (n, regressed, improved, unchanged).  'regressed' must be 0.  A fix that
adds 37 and breaks 5 is not a gain of 32, it is a trade, and it gets reported as
one.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import importlib.util  # noqa: E402

from artificial_memory.skills.deterministic_memory_engine import (  # noqa: E402
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

CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
RUN = ROOT / "benchmark_results/locomo_full_20261012_211601.json"


def main() -> int:
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row
    run = json.loads(RUN.read_text(encoding="utf-8"))
    results = run["results"]

    # Every currently-correct question must stay correct.  That is the hard bar.
    correct = [r for r in results if r.get("is_correct")]
    # Plus a random sample of the currently-wrong ones, to see whether the new
    # branch accidentally fixed or broke anything outside the target cluster.
    wrong = [r for r in results if not r.get("is_correct")]
    rng = random.Random(20261013)
    sample = rng.sample(wrong, min(400, len(wrong)))

    print(f"run: {RUN.name}  n={len(results)}")
    print(f"  currently-correct (must NOT regress) : {len(correct)}")
    print(f"  random wrong sample (watch either way): {len(sample)}")
    print()

    def evaluate(rows: list[dict], label: str) -> tuple[int, int, list]:
        scored = regressed = improved = 0
        examples: list[tuple] = []
        for row in rows:
            qid = row.get("question_id")
            ctx = cache.get(qid)
            if not ctx:
                continue
            context = ctx.get("context") or ""
            if not context:
                continue
            scored += 1
            gold = row.get("ground_truth") or ""
            if not gold:
                continue
            ans = DeterministicMemoryEngine.resolve(row["question"], context)
            now = bool(ans.used) and score(ans.answer, gold)
            was = bool(row.get("is_correct"))
            if was and not now:
                regressed += 1
                if len(examples) < 10:
                    examples.append((qid, gold, ans.answer))
            elif now and not was:
                improved += 1
        print(f"[{label}] scored={scored}  REGRESSED={regressed}  improved={improved}")
        return scored, regressed, examples

    _, reg_correct, ex_correct = evaluate(correct, "currently-correct")
    _, _, ex_wrong = evaluate(sample, "wrong sample")

    print()
    if reg_correct:
        print("REGRESSIONS AMONG CURRENTLY-CORRECT -- the fix is not clean:")
        for qid, gold, got in ex_correct:
            print(f"  {qid:<20} gold={str(gold)[:34]!r:<36} got={str(got)[:34]!r}")
    else:
        print("No currently-correct question regressed.  The fix is additive on")
        print("the questions it touches.")
    return 1 if reg_correct else 0


if __name__ == "__main__":
    raise SystemExit(main())
