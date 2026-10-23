#!/usr/bin/env python3
"""M30-42 Phase 3: E2E impact of the context changes.

Phase 2b proved the ablations DO change the compiled context (26/28, 24/28).
So Reader output and score CAN differ. This measures the actual E2E effect.

Baseline vs A/B/E/F: run the reader on each ablation's context and compare
answers + correctness.

Uses the same frozen reader (qwen2.5:7b-instruct) and the existing grader.
No production edit. No holdout.
"""
import sys, json, re, hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory_gen.context.msc_compiler import MinimumSufficientContextCompiler
from artificial_memory.research.benchmarks.llm import OllamaAnswerer
from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader
from artificial_memory.recall import evidence_scorer as es
from artificial_memory.recall import state_reconstructor as sr
from artificial_memory.recall import query_planner as qp


def load_gold():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    by_q = {}
    for conv in locomo:
        sid = conv.get("sample_id", "")
        for qa in conv.get("qa", []):
            by_q[(sid, qa.get("question", "").strip().lower())] = qa.get("evidence", []) or []
    return by_q


def make_records(turns):
    return [StructuredIR(
        entity=t.get("speaker", ""), property="statement", value=t.get("text", ""),
        source=t.get("speaker", ""),
        raw_content=f"[{t.get('dia_id','')}] {t.get('speaker','')}: {t.get('text','')}",
    ) for t in turns]


def main():
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    convs = bucket["conversations"]
    cases = {c["question_id"]: c for c in bucket["cases"]}
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")

    compiler = MinimumSufficientContextCompiler()
    reader = OllamaAnswerer(model="qwen2.5:7b-instruct")
    grader = OfficialLongMemEvalGrader()
    qtype_map = {1:"multi-session",2:"temporal-reasoning",3:"single-session-user",
                 4:"single-session-assistant",5:"abstention"}

    ORIG_KNOWN = list(es.UniversalEvidenceScorer.KNOWN_ACTORS)
    ORIG_EXP = es.UniversalEvidenceScorer.SEMANTIC_EXPANSIONS
    ORIG_NAMED = set(qp.QueryPlanner.NAMED_ENTITIES)
    ORIG_ES = es.UniversalEvidenceScorer.extract_target_actor
    ORIG_SR = sr.StateReconstructor._extract_target_actor

    def reset():
        es.UniversalEvidenceScorer.extract_target_actor = ORIG_ES
        sr.StateReconstructor._extract_target_actor = ORIG_SR
        es.UniversalEvidenceScorer.KNOWN_ACTORS = ORIG_KNOWN
        es.UniversalEvidenceScorer.SEMANTIC_EXPANSIONS = ORIG_EXP
        qp.QueryPlanner.NAMED_ENTITIES = ORIG_NAMED

    def apply(dis):
        reset()
        if "A" in dis:
            es.UniversalEvidenceScorer.KNOWN_ACTORS = []
        if "B" in dis:
            sr.StateReconstructor._extract_target_actor = lambda self, q: None
        if "C" in dis:
            qp.QueryPlanner.NAMED_ENTITIES = set()
        if "E" in dis:
            es.UniversalEvidenceScorer.SEMANTIC_EXPANSIONS = {}

    configs = [("Baseline", ""), ("A", "A"), ("B", "B"), ("E", "E"), ("F", "ABCE")]
    results = {}

    print("=" * 78)
    print("M30-42 Phase 3: E2E impact of context changes")
    print("=" * 78)
    print()

    for label, dis in configs:
        apply(dis)
        rows = []
        for qid in targets:
            case = cases.get(qid)
            if not case:
                continue
            turns = convs.get(case.get("source_conversation", ""), [])
            if not isinstance(turns, list):
                continue
            q = case.get("question", "")
            gold = str(case.get("gold_answer", ""))
            try:
                pcc = compiler.compile(q, make_records(turns))
                ans = reader.answer(q, pcc.context_text).text.strip()
            except Exception as e:
                ans = f"ERROR: {e}"
            qt = qtype_map.get(case.get("category", 3), "single-session-user")
            try:
                ok = grader.grade(qid, qt, ans, gold).is_correct
            except Exception:
                ok = False
            rows.append({"qid": qid, "answer": ans, "correct": ok,
                         "ctx_hash": hashlib.sha256(pcc.context_text.encode()).hexdigest()[:12]
                                     if 'pcc' in dir() else ""})
        n_ok = sum(1 for r in rows if r["correct"])
        results[label] = {"rows": rows, "correct": n_ok, "n": len(rows)}
        print(f"{label:10s} correct={n_ok}/{len(rows)}")
        # incremental save
        out = PROJECT_ROOT / "results/m30_42"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"e2e_{label}.json").write_text(json.dumps(results[label], indent=2, ensure_ascii=False), encoding="utf-8")

    reset()
    print()

    base = {r["qid"]: r for r in results["Baseline"]["rows"]}
    print("=" * 78)
    print("DELTA vs Baseline")
    print("=" * 78)
    for label in ["A", "B", "E", "F"]:
        rec = reg = unch = 0
        for r in results[label]["rows"]:
            b = base.get(r["qid"])
            if not b:
                continue
            if r["correct"] and not b["correct"]:
                rec += 1
            elif b["correct"] and not r["correct"]:
                reg += 1
            else:
                unch += 1
        print(f"  {label:3s}: correct {results[label]['correct']}/28  recovered={rec} regressed={reg} unchanged={unch}  (sum={rec+reg+unch})")
    print()

    out = PROJECT_ROOT / "results/m30_42"
    (out / "e2e_ablation.json").write_text(json.dumps(
        {k: {"correct": v["correct"], "n": v["n"]} for k, v in results.items()},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'e2e_ablation.json'}")


if __name__ == "__main__":
    main()
