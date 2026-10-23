#!/usr/bin/env python3
"""M30-40 Phase 4: Paired evaluation of the normalizer.

Same raw output, two paths:
  Baseline  = raw output scored as-is
  Treatment = normalized output scored

Scoring is exact match against the case's target (the expected answer).
This isolates the normalizer from LLM/retrieval variance.
"""
import json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from m30_40_normalize import normalize


def exact_match(predicted, target):
    return predicted.strip().rstrip(".") == target.strip().rstrip(".")


def run(cases, label):
    baseline_correct = 0
    treatment_correct = 0
    recovered = 0
    regressed = 0
    unchanged_correct = 0
    unchanged_wrong = 0

    abst_applied = 0
    gran_applied = 0
    false_positive = 0     # negative case wrongly normalized
    meaning_changed = 0    # normalized but no longer matches either target
    rows = []

    for c in cases:
        q = c["question"]
        raw = c["raw_output"]
        target = c["target"]
        should = c["should_normalize"]

        result = normalize(q, raw)
        norm = result["normalized"]
        status = result["status"]

        b_ok = exact_match(raw, target)
        t_ok = exact_match(norm, target)

        if b_ok:
            baseline_correct += 1
        if t_ok:
            treatment_correct += 1

        if not b_ok and t_ok:
            recovered += 1
        elif b_ok and not t_ok:
            regressed += 1
        elif b_ok and t_ok:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1

        if status == "NORMALIZED":
            if result["rule"] == "ABSTENTION_CANONICAL":
                abst_applied += 1
            elif result["rule"] and result["rule"].startswith("GRANULARITY"):
                gran_applied += 1

        # false positive: negative case that got normalized
        if not should and status == "NORMALIZED":
            false_positive += 1

        # meaning change: was correct before, wrong after (or vice versa on negatives)
        if b_ok and not t_ok:
            meaning_changed += 1

        rows.append({
            "question_id": c["question_id"],
            "group": c["group"],
            "question": q,
            "raw": raw,
            "normalized": norm,
            "target": target,
            "rule": result["rule"],
            "status": status,
            "should_normalize": should,
            "baseline_correct": b_ok,
            "treatment_correct": t_ok,
        })

    print("=" * 78)
    print(f"{label}")
    print("=" * 78)
    print(f"  Baseline correct:  {baseline_correct}/{len(cases)}")
    print(f"  Treatment correct: {treatment_correct}/{len(cases)}")
    print(f"  Recovered:         {recovered}")
    print(f"  Regressed:         {regressed}")
    print(f"  Unchanged correct: {unchanged_correct}")
    print(f"  Unchanged wrong:   {unchanged_wrong}")
    print(f"  Abstention normalized: {abst_applied}")
    print(f"  Granularity normalized: {gran_applied}")
    print(f"  FALSE POSITIVES (negative wrongly normalized): {false_positive}")
    print(f"  Meaning changed:   {meaning_changed}")
    print()

    # by group
    from collections import defaultdict
    g = defaultdict(lambda: {"n": 0, "b": 0, "t": 0})
    for r in rows:
        g[r["group"]]["n"] += 1
        g[r["group"]]["b"] += int(r["baseline_correct"])
        g[r["group"]]["t"] += int(r["treatment_correct"])
    print("  By group:")
    for grp, s in sorted(g.items()):
        print(f"    {grp}: baseline {s['b']}/{s['n']}, treatment {s['t']}/{s['n']}")
    print()

    if false_positive:
        print("  FALSE POSITIVE detail:")
        for r in rows:
            if not r["should_normalize"] and r["status"] == "NORMALIZED":
                print(f"    {r['question_id']}: {r['raw'][:60]}")
                print(f"      -> {r['normalized'][:60]}")
        print()

    return {
        "label": label, "n": len(cases),
        "baseline_correct": baseline_correct, "treatment_correct": treatment_correct,
        "recovered": recovered, "regressed": regressed,
        "unchanged_correct": unchanged_correct, "unchanged_wrong": unchanged_wrong,
        "abstention_normalized": abst_applied, "granularity_normalized": gran_applied,
        "false_positives": false_positive, "meaning_changed": meaning_changed,
        "rows": rows,
    }


def main():
    dev = json.loads((PROJECT_ROOT / "results/m30_40/dev_cases.json").read_text(encoding="utf-8"))["cases"]
    frozen = json.loads((PROJECT_ROOT / "results/m30_40/frozen_cases.json").read_text(encoding="utf-8"))["cases"]

    dev_res = run(dev, "DEVELOPMENT (40)")
    frozen_res = run(frozen, "FROZEN EVALUATION (120)")

    out_dir = PROJECT_ROOT / "results/m30_40"
    (out_dir / "evaluation_results.json").write_text(
        json.dumps({"development": dev_res, "frozen": frozen_res}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"Saved: {out_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
