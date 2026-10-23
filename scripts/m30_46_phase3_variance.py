#!/usr/bin/env python3
"""M30-46 Phase 3-0b: Measure baseline variance — is 13 vs 14 nondeterminism?

Run the production path TWICE with identical inputs and compare per-question
correctness. If they differ, the 13/28 vs 14/28 gap is model noise, and any
A/B must account for it (paired, same-run comparison only).
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer


def to_conv(q):
    m = re.match(r"lc_conv-(\d+)_(\d+)", q)
    return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}"


def run_once():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")
    tmap = {to_conv(q): q for q in targets}

    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
    answerer = OllamaAnswerer(model="qwen2.5:7b-instruct")
    rows = {}
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            if q.question_id not in tmap:
                continue
            res = adapter.evaluate_question(q, turns, ir_records, answerer)
            rows[tmap[q.question_id]] = {"correct": bool(res.is_correct),
                                         "pred": str(res.predicted_answer)[:60]}
    return rows


def main():
    print("=" * 78)
    print("M30-46 Phase 3-0b: baseline variance (two identical runs)")
    print("=" * 78)
    print()
    r1 = run_once()
    n1 = sum(1 for v in r1.values() if v["correct"])
    print(f"run 1: {n1}/28")
    r2 = run_once()
    n2 = sum(1 for v in r2.values() if v["correct"])
    print(f"run 2: {n2}/28")
    print()
    flips = [k for k in r1 if r1[k]["correct"] != r2[k]["correct"]]
    print(f"per-question flips between identical runs: {len(flips)}")
    for k in flips:
        print(f"  {k}: run1={'OK' if r1[k]['correct'] else '--'} run2={'OK' if r2[k]['correct'] else '--'}")
    print()
    print("=" * 78)
    if flips:
        print("VERDICT: model is NONDETERMINISTIC at temperature=0.1.")
        print("Any A/B must be PAIRED within one run, and small deltas (<=2) are noise.")
    else:
        print("VERDICT: deterministic; the 13 vs 14 gap has another cause.")
    print("=" * 78)

    out = PROJECT_ROOT / "results/m30_46"
    out.mkdir(parents=True, exist_ok=True)
    (out / "variance.json").write_text(json.dumps({
        "run1": n1, "run2": n2, "flips": flips,
        "run1_rows": r1, "run2_rows": r2,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'variance.json'}")


if __name__ == "__main__":
    main()
