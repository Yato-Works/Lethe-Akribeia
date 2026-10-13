"""Before/after the NUMERAL CONTRADICTION guard, on every stored run.

Scores every stored pred/gold pair with the OLD scorer (guard bypassed) and the
NEW scorer (guard active), then reports:

  (TOTAL, previously-correct, still-correct, SPURIOUS REMOVED, regressions)

'regressions' is the count of pairs that were correct before the guard and are
wrong after it.  That number must be zero, or the guard is eating real credits.
Anything above zero is reported, never hidden -- a guard that removes 32
spurious credits while breaking 3 real ones is not a clean fix.
"""
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = mod
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
score_new = mod.score


def score_old(pred: str, gold: str) -> bool:
    """The pre-guard scorer: temporarily neutralise the numeral guard."""
    real = mod._numeral_contradiction
    mod._numeral_contradiction = lambda p, g: False
    try:
        return score_new(pred, gold)
    finally:
        mod._numeral_contradiction = real


def main() -> int:
    runs = sorted((ROOT / "benchmark_results").glob("locomo_full_*.json"))
    total = 0
    old_correct = new_correct = 0
    removed: list[tuple] = []
    regressed: list[tuple] = []

    for path in runs:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("results") or data.get("predictions") or []
        for row in rows:
            pred = row.get("predicted_answer")
            gold = row.get("ground_truth") or row.get("gold")
            if not pred or not gold:
                continue
            total += 1
            before, after = score_old(pred, gold), score_new(pred, gold)
            old_correct += before
            new_correct += after
            if before and not after:
                removed.append((str(row.get("question_id")), str(pred)[:44], str(gold)[:44]))
            elif after and not before:
                regressed.append((str(row.get("question_id")), str(pred)[:44], str(gold)[:44]))

    print(f"runs scanned                        : {len(runs)}")
    print(f"total pred/gold pairs               : {total}")
    print(f"correct BEFORE the guard            : {old_correct}")
    print(f"correct AFTER  the guard            : {new_correct}")
    print(f"spurious credits REMOVED            : {len(removed)}")
    print(f"real credits BROKE (must be 0)      : {len(regressed)}")

    if regressed:
        print("\nREGRESSIONS -- guard ate real credits, investigate:")
        for qid, pred, gold in regressed[:20]:
            print(f"  {qid:<24} pred={pred!r:<46} gold={gold!r}")
    print("\nsample of the spurious credits now removed:")
    for qid, pred, gold in removed[:20]:
        print(f"  {qid:<24} pred={pred!r:<46} gold={gold!r}")

    print("\nVERDICT:", "clean fix" if not regressed else "NOT clean -- regressions present")
    return 0 if not regressed else 1


if __name__ == "__main__":
    raise SystemExit(main())
