#!/usr/bin/env python3
"""M30-41 Phase 8c: query_planner.NAMED_ENTITIES A/B.

NAMED_ENTITIES feeds QueryPlan.target_entities (for non-possessive names).
Possessives ("Caroline's") are extracted separately and dynamically.

Measure: does removing NAMED_ENTITIES change the plan's target_entities on
Bucket A 28? And does it change the downstream retrieval rank?

Read-only wrt production (runtime monkeypatch). No LLM.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory_gen.recall.query_planner import QueryPlanner

PROD = {"caroline", "melanie", "eli", "dave", "michael", "rachel", "juan", "kadence"}


def main():
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucket["cases"]}
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")

    planner = QueryPlanner()
    ORIG = set(QueryPlanner.NAMED_ENTITIES)

    print("=" * 78)
    print("M30-41 Phase 8c: query_planner.NAMED_ENTITIES A/B")
    print("=" * 78)
    print(f"NAMED_ENTITIES: {sorted(ORIG)}")
    print()

    def plan_entities(disable):
        QueryPlanner.NAMED_ENTITIES = set() if disable else ORIG
        out = {}
        for qid in targets:
            case = cases.get(qid)
            if not case:
                continue
            p = planner.plan(case.get("question", ""))
            out[qid] = list(p.target_entities)
        return out

    base = plan_entities(False)
    off = plan_entities(True)

    # Compare
    changed = [(q, base[q], off[q]) for q in targets if base.get(q) != off.get(q)]
    print(f"Questions where target_entities changed: {len(changed)}/{len(targets)}")
    for q, b, o in changed[:15]:
        qtext = cases[q].get("question", "")[:60]
        print(f"  {q}: {b} -> {o}")
        print(f"     Q: {qtext}")
    print()

    base_has = sum(1 for q in targets if base.get(q))
    off_has = sum(1 for q in targets if off.get(q))
    print(f"Questions with non-empty target_entities:")
    print(f"  Baseline (names ON):  {base_has}/{len(targets)}")
    print(f"  Treatment (names OFF): {off_has}/{len(targets)}")
    print()

    # Do synthetic names get recognized? (they never would)
    print("Cross-check: synthetic names are NEVER in NAMED_ENTITIES")
    for nm in ["olivia", "aiden", "silas", "victor"]:
        p = planner.plan(f"When did {nm.capitalize()} go to the event?")
        print(f"  plan('... {nm.capitalize()} ...').target_entities = {p.target_entities}")
    print()

    out = PROJECT_ROOT / "results/m30_41"
    out.mkdir(parents=True, exist_ok=True)
    (out / "query_planner_ab.json").write_text(json.dumps({
        "base_entities": base, "off_entities": off,
        "changed": [q for q, _, _ in changed],
        "base_nonempty": base_has, "off_nonempty": off_has,
        "note": "runtime monkeypatch; no LLM; no production edit",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'query_planner_ab.json'}")


if __name__ == "__main__":
    main()
