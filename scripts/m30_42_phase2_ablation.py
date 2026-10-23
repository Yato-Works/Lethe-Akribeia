#!/usr/bin/env python3
"""M30-42 Phase 2: Ablation of benchmark-specific mechanisms (isolated, runtime).

Production files are NOT edited. Each mechanism is disabled at runtime via
monkeypatch, then the resulting ranking on Bucket A 28 is compared to Baseline.

Mechanisms:
  A. evidence_scorer.KNOWN_ACTORS        (benchmark names)
  B. state_reconstructor name list        (benchmark names)
  C. query_planner.NAMED_ENTITIES         (benchmark names)
  D. aggregation +35 topic anchor         (domain anchor)
  E. DOMAIN_SYNONYMS / SEMANTIC_EXPANSIONS (domain anchor, corpus-specific terms)
  F. all of the above

Metric: gold-evidence rank in the reconstruct_world output order.
No LLM. No production edit.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory_gen.recall.state_reconstructor import StateReconstructor
from artificial_memory.recall import evidence_scorer as es
from artificial_memory.recall import query_planner as qp

LOCOMO = PROJECT_ROOT / "datasets/external/locomo10.json"


def load_gold():
    locomo = json.loads(LOCOMO.read_text(encoding="utf-8"))
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
    gold_by_q = load_gold()

    recon = StateReconstructor()
    ORIG_ACTOR = StateReconstructor._extract_target_actor
    ORIG_KNOWN = list(es.UniversalEvidenceScorer.KNOWN_ACTORS)
    ORIG_NAMED = set(qp.QueryPlanner.NAMED_ENTITIES)

    print("=" * 78)
    print("M30-42 Phase 2: Mechanism ablation (Bucket A 28)")
    print("=" * 78)
    print()

    def run(label, disable):
        # reset
        StateReconstructor._extract_target_actor = ORIG_ACTOR
        es.UniversalEvidenceScorer.KNOWN_ACTORS = ORIG_KNOWN
        qp.QueryPlanner.NAMED_ENTITIES = ORIG_NAMED
        if "A" in disable:
            es.UniversalEvidenceScorer.KNOWN_ACTORS = []
        if "B" in disable:
            StateReconstructor._extract_target_actor = lambda self, q: None
        if "C" in disable:
            qp.QueryPlanner.NAMED_ENTITIES = set()
        # D/E require source patching; approximate by clearing the expansions
        if "E" in disable:
            es.UniversalEvidenceScorer.SEMANTIC_EXPANSIONS = {}

        rows = []
        for qid in targets:
            case = cases.get(qid)
            if not case:
                continue
            conv_key = case.get("source_conversation", "")
            turns = convs.get(conv_key, [])
            if not isinstance(turns, list):
                continue
            gold = set(g for g in gold_by_q.get((conv_key, case.get("question","").strip().lower()), []) if isinstance(g, str))
            try:
                intent, units = recon.reconstruct_world(case.get("question", ""), make_records(turns))
            except Exception:
                rows.append({"qid": qid, "rank": None})
                continue
            rank = None
            for i, u in enumerate(units, 1):
                m = re.search(r"\[(D\d+:\d+)\]", u.ir.raw_content)
                if m and m.group(1) in gold:
                    rank = i
                    break
            rows.append({"qid": qid, "rank": rank})
        return rows

    def summ(rows):
        r1 = sum(1 for r in rows if r["rank"] == 1)
        r5 = sum(1 for r in rows if r["rank"] and r["rank"] <= 5)
        r10 = sum(1 for r in rows if r["rank"] and r["rank"] <= 10)
        found = sum(1 for r in rows if r["rank"] is not None)
        return {"top1": r1, "top5": r5, "top10": r10, "found": found, "n": len(rows)}

    configs = [
        ("Baseline (all ON)", ""),
        ("A: KNOWN_ACTORS off", "A"),
        ("B: name-list off", "B"),
        ("C: NAMED_ENTITIES off", "C"),
        ("E: SEMANTIC_EXPANSIONS off", "E"),
        ("F: A+B+C+E off", "ABCE"),
    ]

    results = {}
    for label, dis in configs:
        rows = run(label, dis)
        s = summ(rows)
        results[label] = s
        print(f"{label:28s} top1={s['top1']:2d}  top5={s['top5']:2d}  top10={s['top10']:2d}  found={s['found']:2d}")
        results[label + "_rows"] = rows
    print()

    # restore
    StateReconstructor._extract_target_actor = ORIG_ACTOR
    es.UniversalEvidenceScorer.KNOWN_ACTORS = ORIG_KNOWN
    qp.QueryPlanner.NAMED_ENTITIES = ORIG_NAMED

    base = results["Baseline (all ON)"]
    print("DELTA vs Baseline:")
    for label, _ in configs[1:]:
        s = results[label]
        print(f"  {label:28s} top1={s['top1']-base['top1']:+d}  top5={s['top5']-base['top5']:+d}  top10={s['top10']-base['top10']:+d}")
    print()

    out = PROJECT_ROOT / "results/m30_42"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
