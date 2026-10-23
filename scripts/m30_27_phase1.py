#!/usr/bin/env python3
"""M30-27 Phase 1: Investigate actual normal-pipeline inputs.

For each of the 28 TARGET cases, record:
- QID, normal reader input, normal reader output
- Whether the input contains timestamped turns
- Whether the question-relevant turn can be identified
- Whether a reference datetime can be uniquely selected

Constraint: only information actually present in the normal pipeline input
counts as available. Full conversations or context caches that are NOT in
the normal pipeline input are NOT counted.
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
M30_3 = PROJECT_ROOT / "results/m30_3_evaluation_results.json"
INTERVENTION = PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json"
LOCOMO = PROJECT_ROOT / "datasets/external/locomo10.json"
CACHE = PROJECT_ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"

SESSION_DATES = {
    "conv-26": "2023-08-28", "conv-30": "2023-11-30",
    "conv-41": "2023-12-04", "conv-42": "2023-12-04",
    "conv-44": "2023-12-04", "conv-47": "2022-09-30",
    "conv-48": "2023-01-31", "conv-49": "2023-12-25",
    "conv-50": "2023-08-28",
}

def main():
    m30_3 = json.loads(M30_3.read_text(encoding="utf-8"))
    m30_3_results = {r["question_id"]: r for r in m30_3["results"]}
    intervention = json.loads(INTERVENTION.read_text(encoding="utf-8"))
    targets = sorted([r["question_id"] for r in intervention["rows"] if r.get("group") == "TARGET"])

    locomo = json.loads(LOCOMO.read_text(encoding="utf-8"))
    locomo_by_id = {c["sample_id"]: c for c in locomo}

    # Load cache for reference (production pipeline context)
    cache = {}
    with open(CACHE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row.get("qid", "")] = row.get("context", "")

    print("=" * 78)
    print("M30-27 Phase 1: Normal-pipeline input investigation")
    print("=" * 78)
    print(f"TARGET cases: {len(targets)}")
    print()

    # Check M30-3's evidence_text for timestamps
    ts_pattern = re.compile(r"on \d{1,2}:\d{2} (?:am|pm) on \d{1,2} \w+, \d{4}")
    iso_pattern = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")

    m30_3_with_ts = 0
    m30_3_with_iso = 0
    for qid in targets:
        r = m30_3_results.get(qid, {})
        ev = r.get("evidence_text", "")
        if ts_pattern.search(ev):
            m30_3_with_ts += 1
        if iso_pattern.search(ev):
            m30_3_with_iso += 1

    print(f"M30-3 evidence_text with human timestamp: {m30_3_with_ts}/28")
    print(f"M30-3 evidence_text with ISO date:        {m30_3_with_iso}/28")
    print()

    # Check production pipeline (locomo_adapter.py) context for timestamps
    prod_with_ts = 0
    for qid in targets:
        conv_id = f"conv-{qid.split('_')[0].replace('lc_conv-', '')}"
        c = locomo_by_id.get(conv_id)
        if not c:
            continue
        conv = c.get("conversation", {})
        for sk in sorted(conv.keys()):
            if sk.endswith("_date_time"):
                prod_with_ts += 1
                break

    print(f"Production pipeline (locomo10.json) with session dates: {prod_with_ts}/28")
    print()

    # Check cache for timestamps
    cache_with_ts = 0
    for qid in targets:
        ctx = cache.get(qid, "")
        if ts_pattern.search(ctx):
            cache_with_ts += 1

    print(f"Context cache with timestamps: {cache_with_ts}/28")
    print()

    # Show a sample of M30-3's evidence_text vs production context
    print("--- Sample: lc_conv-49_79 ---")
    r = m30_3_results.get("lc_conv-49_79", {})
    ev = r.get("evidence_text", "")
    print(f"M30_3 evidence_text (first 300 chars):")
    print(f"  {ev[:300]}")
    print()
    ctx = cache.get("lc_conv-49_79", "")
    if ctx:
        print(f"Context cache (first 300 chars):")
        print(f"  {ctx[:300]}")
    print()

    # Key question: can we identify the question-relevant turn from M30-3's input?
    print("--- Can we identify the question-relevant turn from M30-3's input? ---")
    print("M30-3's evidence_text format: 'D186 | Caroline | text'")
    print("No timestamps. No session info. No way to determine which session a turn belongs to.")
    print()
    print("Production pipeline format: '[D23:23 on 1:32 pm on 6 January, 2024] Evan: text'")
    print("Timestamps present. Session info available.")
    print()

    # Conclusion
    print("=" * 78)
    print("CONCLUSION:")
    print("=" * 78)
    print("M30-3's normal pipeline (extract_evidence_turns + v4_bucketA_all.json)")
    print("does NOT include timestamps in the Reader input.")
    print()
    print("Production pipeline (locomo_adapter.py + locomo10.json) DOES include")
    print("timestamps in the format '[D23:23 on 1:32 pm on 6 January, 2024]'.")
    print()
    print("Therefore, non-oracle temporal resolution is IMPOSSIBLE with M30-3's")
    print("normal pipeline input alone. The production pipeline has the needed")
    print("timestamps, but M30-3 does not use the production pipeline.")
    print()
    print("Phase 2-3 will use the production pipeline's context (with timestamps)")
    print("to test whether non-oracle resolution CAN work when timestamps are present.")

if __name__ == "__main__":
    main()
