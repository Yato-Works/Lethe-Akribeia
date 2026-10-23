#!/usr/bin/env python3
"""M30-41 Phase 8a: Deterministic retrieval A/B — actor boost ON vs OFF.

Read-only wrt production: we monkeypatch the two hardcoded name lists at
RUNTIME (in this process only) and compare the resulting retrieval ranking
on Bucket A's 28 questions.

No LLM. No production file edit. No holdout access.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.core.ir.memory_types import ApexMemoryUnit
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.recall import evidence_scorer as es
from artificial_memory.recall import state_reconstructor as sr_mod

MONTHS = {m.lower(): i for i, m in enumerate([
    "january","february","march","april","may","june","july","august",
    "september","october","november","december"], start=1)}


def load_bucket_a():
    """Load the 28 TARGET questions + their conversations."""
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucket["cases"]}
    convs = bucket["conversations"]
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")
    return cases, convs, targets


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
    cases, convs, targets = load_bucket_a()
    print("=" * 78)
    print("M30-41 Phase 8a: Retrieval A/B (actor boost ON vs OFF)")
    print("=" * 78)
    print()
    print(f"TARGET questions: {len(targets)}")
    print()

    scorer = es.UniversalEvidenceScorer()

    # Baseline = production (KNOWN_ACTORS active)
    def run_ab(label, disable_actor):
        if disable_actor:
            es.UniversalEvidenceScorer.KNOWN_ACTORS = []
        else:
            # restore the original list captured at import
            es.UniversalEvidenceScorer.KNOWN_ACTORS = ORIG_KNOWN_ACTORS

        ranks = []
        for qid in targets:
            case = cases.get(qid)
            if not case:
                continue
            conv_key = case.get("source_conversation", "")
            turns = convs.get(conv_key, [])
            if not isinstance(turns, list):
                continue
            units = make_units(turns)
            scored = score_all(case.get("question", ""), units, scorer)
            # find the gold evidence turn id
            ev = case.get("evidence_ids") or case.get("evidence") or []
            gold_ids = set()
            for e in ev:
                if isinstance(e, str):
                    gold_ids.add(e)
            rank = None
            for i, (tot, dia) in enumerate(scored, 1):
                if dia in gold_ids:
                    rank = i
                    break
            ranks.append(rank)
        return ranks

    ORIG_KNOWN_ACTORS = list(es.UniversalEvidenceScorer.KNOWN_ACTORS)
    print(f"Production KNOWN_ACTORS: {len(ORIG_KNOWN_ACTORS)} names")
    print()

    base_ranks = run_ab("baseline", disable_actor=False)
    off_ranks = run_ab("no_actor", disable_actor=True)

    def summarize(ranks, label):
        r1 = sum(1 for r in ranks if r == 1)
        r5 = sum(1 for r in ranks if r and r <= 5)
        r10 = sum(1 for r in ranks if r and r <= 10)
        found = sum(1 for r in ranks if r is not None)
        print(f"{label}:")
        print(f"  gold in top-1:  {r1}/{len(ranks)}")
        print(f"  gold in top-5:  {r5}/{len(ranks)}")
        print(f"  gold in top-10: {r10}/{len(ranks)}")
        print(f"  gold retrieved: {found}/{len(ranks)}")
        return {"top1": r1, "top5": r5, "top10": r10, "found": found, "n": len(ranks)}

    print("--- Actor boost ON (production) ---")
    b = summarize(base_ranks, "Baseline")
    print()
    print("--- Actor boost OFF (KNOWN_ACTORS=[]) ---")
    o = summarize(off_ranks, "Treatment")
    print()

    print("=" * 78)
    print("DELTA")
    print("=" * 78)
    print(f"  top-1:  {b['top1']} -> {o['top1']}  ({o['top1']-b['top1']:+d})")
    print(f"  top-5:  {b['top5']} -> {o['top5']}  ({o['top5']-b['top5']:+d})")
    print(f"  top-10: {b['top10']} -> {o['top10']}  ({o['top10']-b['top10']:+d})")
    print()

    out = PROJECT_ROOT / "results/m30_41"
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_ab_actor_boost.json").write_text(json.dumps({
        "baseline_actor_on": b, "treatment_actor_off": o,
        "base_ranks": base_ranks, "off_ranks": off_ranks,
        "note": "runtime monkeypatch, no production file change, no LLM",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'retrieval_ab_actor_boost.json'}")


if __name__ == "__main__":
    main()
