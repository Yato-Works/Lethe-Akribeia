"""Why does the re-score flip BOTH directions in near-equal measure?

rescore_all_runs reported, across 46 runs:
    correct -> wrong : 505
    wrong -> correct : 480

A guard that only removes spurious credits flips ONE direction.  A near-even
split means something else in the scorer changed too.  This script isolates the
cause by re-scoring a single newest full run under four scorer configurations:

  A  stored flags                    (what the run reported)
  B  current score() as-is
  C  current score() with BOTH guards disabled
  D  current score() with only the DATE guard disabled
  E  current score() with only the NUMERAL guard disabled

C vs B isolates the guards' total effect.  D and E isolate each guard.
If C still disagrees with A, the difference is NOT the guards -- it is some
other change in score() or in is_list_gold, and that is what must be found
before any baseline number is trusted.
"""
import importlib.util
import json
import pathlib
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

TARGET = ROOT / "benchmark_results/locomo_full_20261012_211601.json"


def run_variant(data: list[dict], date_guard: bool, numeral_guard: bool) -> dict:
    real_date, real_num = mod._date_contradiction, mod._numeral_contradiction
    mod._date_contradiction = real_date if date_guard else (lambda p, g: False)
    mod._numeral_contradiction = real_num if numeral_guard else (lambda p, g: False)
    try:
        correct = 0
        flips = {"cw": 0, "wc": 0}
        for r in data:
            pred, gold = r.get("predicted_answer"), r.get("ground_truth")
            now = bool(pred) and bool(gold) and mod.score(pred, gold)
            correct += now
            was = bool(r.get("is_correct"))
            if was and not now:
                flips["cw"] += 1
            elif now and not was:
                flips["wc"] += 1
        return {"correct": correct, **flips}
    finally:
        mod._date_contradiction, mod._numeral_contradiction = real_date, real_num


def main() -> int:
    data = json.loads(TARGET.read_text(encoding="utf-8"))["results"]
    stored = sum(1 for r in data if r.get("is_correct"))

    variants = {
        "stored flags (run's own report)": stored,
        "B both guards ON  (current)": run_variant(data, True, True),
        "C both guards OFF": run_variant(data, False, False),
        "D date ON, numeral OFF": run_variant(data, True, False),
        "E date OFF, numeral ON": run_variant(data, False, True),
    }

    print(f"target: {TARGET.name}  n={len(data)}")
    print()
    print(f"{'variant':<34} {'correct':>8} {'vs stored':>10} {'c->w':>6} {'w->c':>6}")
    print("-" * 68)
    for name, v in variants.items():
        if isinstance(v, int):
            print(f"{name:<34} {v:>8} {'--':>10} {'--':>6} {'--':>6}")
            continue
        print(f"{name:<34} {v['correct']:>8} {v['correct'] - stored:>+10} "
              f"{v['cw']:>6} {v['wc']:>6}")

    print()
    c = run_variant(data, False, False)
    print("INTERPRETATION")
    print(f"  guards OFF still flips {c['cw']} correct->wrong and {c['wc']} wrong->correct")
    print(f"  -> if these are large, the run's stored flags disagree with today's")
    print(f"     score() for reasons OTHER than the two guards.")
    if c["cw"] or c["wc"]:
        print("  -> the baseline is NOT safe to state until this is explained.")
    else:
        print("  -> guards account for the whole difference; baseline is explainable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
