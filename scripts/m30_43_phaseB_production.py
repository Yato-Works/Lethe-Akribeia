#!/usr/bin/env python3
"""M30-43 Phase B: The FIRST honest measurement — production pipeline on Bucket A.

Every prior M30 experiment used a hand-rolled record builder (make_records)
instead of the production LoCoMoAdapter path. This runs the REAL pipeline:

  LoCoMoAdapter.load_conversation()  -> structured IR via UniversalIRExtractor
  adapter.evaluate_question()        -> compile() + reader + verifier
  OfficialLongMemEvalGrader          -> score

Scope: Bucket A 28 questions (dev data). 553 holdout untouched.
No production edit.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer
from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader


def main():
    # Bucket A 28 target QIDs (lc_conv-XX_NN)
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucket["cases"]}
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")

    # Map lc_conv-XX_NN -> conv-XX-qa-NNN
    def to_conv_qid(qid):
        m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
        return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}" if m else qid

    target_conv_qids = {to_conv_qid(q): q for q in targets}
    print("=" * 78)
    print("M30-43 Phase B: PRODUCTION pipeline on Bucket A 28")
    print("=" * 78)
    print(f"Targets: {len(targets)}")
    print(f"Conv-qids: {list(target_conv_qids)[:3]} ...")
    print()

    # Which conversations do we need?
    convs_needed = sorted(set(q.split("-qa-")[0] for q in target_conv_qids))
    print(f"Conversations needed: {convs_needed}")
    print()

    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    print(f"Available sample_ids: {sample_ids}")
    print()

    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
    answerer = OllamaAnswerer(model="qwen2.5:7b-instruct")
    grader = OfficialLongMemEvalGrader()

    results = []
    for idx, sid in enumerate(sample_ids):
        if sid not in convs_needed:
            continue
        turns, questions, ir_records = adapter.load_conversation(idx)
        print(f"[{sid}] turns={len(turns)} questions={len(questions)} ir_records={len(ir_records)}")
        for q in questions:
            if q.question_id not in target_conv_qids:
                continue
            orig_qid = target_conv_qids[q.question_id]
            try:
                res = adapter.evaluate_question(q, turns, ir_records, answerer)
                pred = res.predicted_answer
                ok = res.is_correct
            except Exception as e:
                pred = f"ERROR: {e}"
                ok = False
            results.append({
                "qid": orig_qid, "conv_qid": q.question_id,
                "question": q.question, "gold": q.ground_truth,
                "predicted": pred, "correct": ok,
                "category": q.category,
            })
            print(f"  {orig_qid}: {'OK' if ok else '--'}  pred={str(pred)[:50]}")

    n_ok = sum(1 for r in results if r["correct"])
    print()
    print("=" * 78)
    print(f"PRODUCTION PIPELINE RESULT: {n_ok}/{len(results)}")
    print("=" * 78)
    print()

    out = PROJECT_ROOT / "results/m30_43"
    out.mkdir(parents=True, exist_ok=True)
    (out / "production_bucketA.json").write_text(json.dumps({
        "correct": n_ok, "total": len(results), "rows": results,
        "note": "PRODUCTION path (locomo_adapter), first honest measurement",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'production_bucketA.json'}")


if __name__ == "__main__":
    main()
