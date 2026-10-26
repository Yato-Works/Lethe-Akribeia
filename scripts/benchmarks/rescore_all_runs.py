"""Re-score every stored full run with the current scorer.

Establishes the REAL baseline now that both contradiction guards are in place.
Every stored run carries its own is_correct, produced by whatever scorer was live
at the time -- those flags are historical artifacts, not measurements.  This
script re-scores all of them with today's score() and reports, per run:

  run, code_revision, n, stored_correct, rescored_correct, delta, and the
  question_ids where the verdict changed.

A run whose stored flags disagree with the re-score is reported, never silently
replaced.  'delta' is the size of the historical inflation or deflation.

Guards priced: the date guard (c449aa0) and the numeral guard (e6a305f) are both
active in the scorer used here.  Runs predating either guard will show a negative
delta -- that is the point, not an error.
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
score = mod.score


def main() -> int:
    runs = sorted(
        (ROOT / "benchmark_results").glob("locomo_full_*.json"),
        key=lambda p: p.name,
    )
    rows_out = []
    changed_all: list[dict] = []

    for path in runs:
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results") or []
        if not results:
            continue
        stored_correct = sum(1 for r in results if r.get("is_correct"))
        rescored_correct = 0
        changed: list[dict] = []
        for r in results:
            pred = r.get("predicted_answer")
            gold = r.get("ground_truth") or r.get("gold")
            now = bool(pred) and bool(gold) and score(pred, gold)
            rescored_correct += now
            was = bool(r.get("is_correct"))
            if now != was:
                changed.append({
                    "question_id": r.get("question_id"),
                    "category": r.get("category"),
                    "question": r.get("question"),
                    "gold": gold,
                    "pred": pred,
                    "stored_verdict": "correct" if was else "wrong",
                    "rescored_verdict": "correct" if now else "wrong",
                })
        changed_all.extend(changed)
        rows_out.append({
            "run": path.name,
            "code_revision": data.get("code_revision"),
            "config": data.get("config"),
            "n": len(results),
            "stored_correct": stored_correct,
            "rescored_correct": rescored_correct,
            "delta": rescored_correct - stored_correct,
            "verdict_changes": len(changed),
        })

    print(f"{'run':<44} {'rev':<9} {'n':>5} {'stored':>7} {'rescored':>9} {'delta':>6} {'chg':>4}")
    print("-" * 92)
    for row in rows_out:
        print(f"{row['run']:<44} {str(row['code_revision']):<9} {row['n']:>5} "
              f"{row['stored_correct']:>7} {row['rescored_correct']:>9} "
              f"{row['delta']:>6} {row['verdict_changes']:>4}")

    print()
    print(f"runs re-scored                 : {len(rows_out)}")
    print(f"total verdict changes          : {len(changed_all)}")

    # The headline run: the newest full 1,986-question run.
    full = [r for r in rows_out if r["n"] >= 1900]
    if full:
        latest = full[-1]
        print(f"\nnewest full run                : {latest['run']} ({latest['code_revision']})")
        print(f"  stored accuracy              : {latest['stored_correct']}/{latest['n']} "
              f"= {latest['stored_correct'] / latest['n'] * 100:.1f}%")
        print(f"  RESCORED accuracy            : {latest['rescored_correct']}/{latest['n']} "
              f"= {latest['rescored_correct'] / latest['n'] * 100:.1f}%")
        print(f"  delta                        : {latest['delta']:+d}")

    out = {
        "label": "rescore-all-stored-runs-with-current-scorer",
        "scorer": "eval_locomo_full.score with date guard (c449aa0) and numeral guard (e6a305f)",
        "runs": rows_out,
        "total_verdict_changes": len(changed_all),
        "verdict_changes": changed_all,
    }
    target = ROOT / "benchmark_results/scorer_audit/rescored_all_runs.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
