#!/usr/bin/env python3
"""M30-41 Phase 2: Does the compiled context preserve the ORIGINAL turn ID?

The adapter's own comment says: "the context compiler numbers the turns it
emits in its own group order, so D1:3 in a compiled context is not D1:3 in
locomo10.json."

Phase 2 requires: a structured fact must be traceable back to the source span.

Test: build ir_records with the ORIGINAL dia_id embedded in raw_content, run
compile(), and check whether the [D..] headers in context_text match the
original dia_ids.

Read-only. No LLM.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory_gen.context.msc_compiler import MinimumSufficientContextCompiler


def main():
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    convs = bucket["conversations"]
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")
    cases = {c["question_id"]: c for c in bucket["cases"]}

    qid = targets[0]
    case = cases[qid]
    conv_key = case["source_conversation"]
    turns = convs[conv_key]

    print("=" * 78)
    print("M30-41 Phase 2: Turn-ID provenance preservation")
    print("=" * 78)
    print(f"QID: {qid}")
    print(f"Question: {case['question']}")
    print()

    # Build ir_records carrying the ORIGINAL dia_id in raw_content
    records = []
    for t in turns[:60]:
        records.append(StructuredIR(
            entity=t.get("speaker", ""), property="statement", value=t.get("text", ""),
            source=t.get("speaker", ""),
            raw_content=f"[{t.get('dia_id','')}] {t.get('speaker','')}: {t.get('text','')}",
        ))

    original_ids = set(t.get("dia_id", "") for t in turns[:60])
    print(f"Original dia_ids (sample): {sorted(original_ids)[:8]}")
    print()

    compiler = MinimumSufficientContextCompiler()
    try:
        pcc = compiler.compile(case["question"], records)
    except Exception as e:
        print(f"compile() error: {e}")
        import traceback; traceback.print_exc()
        return

    ctx = pcc.context_text
    emitted = re.findall(r"\[(D\d+:\d+)", ctx)
    emitted_set = set(emitted)

    print(f"Compiled context: {len(ctx)} chars")
    print(f"Emitted [D..] headers: {len(emitted)} (unique {len(emitted_set)})")
    print(f"  sample: {sorted(emitted_set)[:8]}")
    print()

    # Provenance check: are emitted ids a subset of original ids?
    invented = emitted_set - original_ids
    preserved = emitted_set & original_ids
    print("=" * 78)
    print("PROVENANCE CHECK")
    print("=" * 78)
    print(f"  emitted ids that match an ORIGINAL dia_id: {len(preserved)}")
    print(f"  emitted ids NOT in the original set (invented): {len(invented)}")
    if invented:
        print(f"    invented sample: {sorted(invented)[:10]}")
    print()

    if invented:
        print("RESULT: PROVENANCE BREAK CONFIRMED")
        print("  The compiler emits turn ids that do not exist in the source data.")
        print("  A fact citing [D1:3] cannot be traced back to locomo10.json's D1:3.")
    else:
        print("RESULT: provenance preserved (emitted ids are original ids)")
    print()

    # Show a concrete mismatch if any
    if invented:
        bad = sorted(invented)[0]
        m = re.search(rf"\[{re.escape(bad)}[^\]]*\][^\n]*", ctx)
        if m:
            print(f"Example emitted line for {bad}:")
            print(f"  {m.group(0)[:160]}")
            # what does the original D:3 say?
            orig = next((t for t in turns if t.get("dia_id") == bad), None)
            if orig:
                print(f"Original {bad} in locomo10.json:")
                print(f"  {orig.get('speaker')}: {orig.get('text','')[:120]}")
    print()

    out = PROJECT_ROOT / "results/m30_41"
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance_check.json").write_text(json.dumps({
        "qid": qid,
        "emitted_unique": len(emitted_set),
        "preserved": len(preserved),
        "invented": len(invented),
        "invented_sample": sorted(invented)[:20],
        "verdict": "PROVENANCE_BREAK" if invented else "PRESERVED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'provenance_check.json'}")


if __name__ == "__main__":
    main()
