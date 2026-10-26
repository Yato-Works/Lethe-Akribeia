#!/usr/bin/env python3
"""M30-46 Phase 3-0c: Which question differs under context_override?

Production = 13/28 (deterministic). Override = 14/28. Find the 1 question
that flips, and inspect why (e.g. token_cost affecting a guard, or the
is_abstention flag).

This matters: if context_override is not behaviour-preserving, it cannot be
used for the order experiment.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter, CachedContext
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

PROD = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))


def to_conv(q):
    m = re.match(r"lc_conv-(\d+)_(\d+)", q)
    return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}"


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")
    tmap = {to_conv(q): q for q in targets}
    prod_map = {r["qid"]: r for r in PROD["rows"]}

    answerer = OllamaAnswerer(model="qwen2.5:7b-instruct")

    # override path with exact compiled contexts
    a0 = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
    override = {}
    tok = {}
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = a0.load_conversation(idx)
        for q in questions:
            if q.question_id in tmap:
                pcc = a0.compiler.compile(q.question, ir_records)
                override[q.question_id] = CachedContext(context_text=pcc.context_text)
                tok[tmap[q.question_id]] = (len(pcc.context_text), pcc.token_cost)

    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
    adapter.context_override = override

    print("=" * 78)
    print("M30-46 Phase 3-0c: context_override vs production, per question")
    print("=" * 78)
    print()

    diffs = []
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            if q.question_id not in tmap:
                continue
            qid = tmap[q.question_id]
            res = adapter.evaluate_question(q, turns, ir_records, answerer)
            ov = bool(res.is_correct)
            pr = prod_map[qid]["correct"]
            if ov != pr:
                diffs.append((qid, pr, ov))
                print(f"  DIFF {qid}: production={'OK' if pr else '--'} override={'OK' if ov else '--'}")
                print(f"       prod pred: {prod_map[qid]['predicted'][:70]}")
                print(f"       ovr  pred: {str(res.predicted_answer)[:70]}")
                print(f"       ctx len/token_cost: {tok[qid]}")

    print()
    print(f"total differing questions: {len(diffs)}")
    print()
    print("=" * 78)
    print("NOTE: if the difference is a token_cost / is_abstention default in")
    print("CachedContext vs ProofCarryingContext, then context_override is NOT")
    print("behaviour-preserving and the order experiment must use a different hook.")
    print("=" * 78)

    out = PROJECT_ROOT / "results/m30_46"
    out.mkdir(parents=True, exist_ok=True)
    (out / "override_diff.json").write_text(json.dumps({
        "diffs": diffs, "token_info": tok,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'override_diff.json'}")


if __name__ == "__main__":
    main()
