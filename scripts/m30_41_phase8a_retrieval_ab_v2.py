#!/usr/bin/env python3
"""M30-41 Phase 8a-v2: Retrieval A/B with gold evidence from locomo10.json.

Fixes v1: v4_bucketA_all.json has no evidence_ids; load them from the
official locomo10.json (matched by question text), as done in earlier M30 work.

Read-only wrt production: runtime monkeypatch of the two name lists only.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.core.ir.memory_types import ApexMemoryUnit
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.recall import evidence_scorer as es

LOCOMO = PROJECT_ROOT / "datasets/external/locomo10.json"


def load_gold_evidence():
    """question text -> evidence list, from official LoCoMo."""
    locomo = json.loads(LOCOMO.read_text(encoding="utf-8"))
    by_q = {}
    for conv in locomo:
        sid = conv.get("sample_id", "")
        for qa in conv.get("qa", []):
            key = (sid, qa.get("question", "").strip().lower())
            by_q[key] = qa.get("evidence", []) or []
    return by_q


def make_units(conv_turns):
    units = []
    for t in conv_turns:
        ir = StructuredIR(
            entity=t.get("speaker", ""), property="statement", value=t.get("text", ""),
            source=t.get("speaker", ""),
            raw_content=f"[{t.get('dia_id','')}] {t.get('speaker','')}: {t.get('text','')}",
        )
        units.append((t.get("dia_id", ""), ApexMemoryUnit(ir=ir)))
    return units


def score_all(query, units, scorer):
    out = []
    for dia_id, u in units:
        try:
            b = scorer.compute_score(query, u)
            tot = getattr(b, "total_score", None)
            if tot is None:
                tot = sum(v for k, v in vars(b).items() if isinstance(v, (int, float)))
            out.append((tot, dia_id))
        except Exception:
            out.append((0.0, dia_id))
    out.sort(key=lambda x: -x[0])
    return out


def main():
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucket["cases"]}
    convs = bucket["conversations"]
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")

    gold_by_q = load_gold_evidence()
    scorer = es.UniversalEvidenceScorer()
    ORIG = list(es.UniversalEvidenceScorer.KNOWN_ACTORS)

    print("=" * 78)
    print("M30-41 Phase 8a-v2: Retrieval A/B (actor boost ON vs OFF)")
    print("=" * 78)
    print(f"TARGET: {len(targets)}   KNOWN_ACTORS: {len(ORIG)}")
    print()

    def run(disable):
        es.UniversalEvidenceScorer.KNOWN_ACTORS = [] if disable else ORIG
        rows = []
        for qid in targets:
            case = cases.get(qid)
            if not case:
                continue
            conv_key = case.get("source_conversation", "")
            turns = convs.get(conv_key, [])
            if not isinstance(turns, list):
                continue
            gold = gold_by_q.get((conv_key, case.get("question", "").strip().lower()), [])
            gold_ids = set(g for g in gold if isinstance(g, str))
            units = make_units(turns)
            scored = score_all(case.get("question", ""), units, scorer)
            rank = None
            for i, (tot, dia) in enumerate(scored, 1):
                if dia in gold_ids:
                    rank = i
                    break
            rows.append({"qid": qid, "gold": sorted(gold_ids), "rank": rank})
        return rows

    base = run(False)
    off = run(True)

    def summ(rows, label):
        n = len(rows)
        r1 = sum(1 for r in rows if r["rank"] == 1)
        r5 = sum(1 for r in rows if r["rank"] and r["rank"] <= 5)
        r10 = sum(1 for r in rows if r["rank"] and r["rank"] <= 10)
        found = sum(1 for r in rows if r["rank"] is not None)
        nogold = sum(1 for r in rows if not r["gold"])
        print(f"{label}: n={n}  no_gold_label={nogold}")
        print(f"  top-1={r1}  top-5={r5}  top-10={r10}  retrieved={found}")
        return {"n": n, "top1": r1, "top5": r5, "top10": r10, "found": found, "no_gold": nogold}

    b = summ(base, "Baseline (actor ON) ")
    o = summ(off,  "Treatment (actor OFF)")
    print()
    print(f"DELTA top-1: {o['top1']-b['top1']:+d}   top-5: {o['top5']-b['top5']:+d}   top-10: {o['top10']-b['top10']:+d}")
    print()

    # Show rank changes
    print("Per-question rank change (only where it differs):")
    bmap = {r["qid"]: r["rank"] for r in base}
    omap = {r["qid"]: r["rank"] for r in off}
    diffs = 0
    for qid in targets:
        if bmap.get(qid) != omap.get(qid):
            diffs += 1
            print(f"  {qid}: ON={bmap.get(qid)} -> OFF={omap.get(qid)}  gold={bmap.get(qid) and ''}")
    if diffs == 0:
        print("  (none)")
    print()

    out = PROJECT_ROOT / "results/m30_41"
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_ab_actor_boost_v2.json").write_text(json.dumps({
        "baseline_actor_on": b, "treatment_actor_off": o,
        "base_rows": base, "off_rows": off,
        "note": "runtime monkeypatch; gold from locomo10.json; no LLM; no production edit",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'retrieval_ab_actor_boost_v2.json'}")


if __name__ == "__main__":
    main()
