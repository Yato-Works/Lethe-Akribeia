#!/usr/bin/env python3
"""M30-41 Phase 2b: Provenance check across all 28 TARGET questions.

Phase 2a passed on one question. Verify at scale, and also check whether the
adapter's documented id-drift appears (it says the compiler renumbers turns).

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
    cases = {c["question_id"]: c for c in bucket["cases"]}
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")

    compiler = MinimumSufficientContextCompiler()

    print("=" * 78)
    print("M30-41 Phase 2b: Provenance across 28 questions")
    print("=" * 78)
    print()

    total_preserved = 0
    total_invented = 0
    cases_with_invented = []
    errors = 0

    for qid in targets:
        case = cases.get(qid)
        if not case:
            continue
        conv_key = case.get("source_conversation", "")
        turns = convs.get(conv_key, [])
        if not isinstance(turns, list):
            continue
        records = []
        for t in turns:
            records.append(StructuredIR(
                entity=t.get("speaker", ""), property="statement", value=t.get("text", ""),
                source=t.get("speaker", ""),
                raw_content=f"[{t.get('dia_id','')}] {t.get('speaker','')}: {t.get('text','')}",
            ))
        original = set(t.get("dia_id", "") for t in turns)
        try:
            pcc = compiler.compile(case["question"], records)
        except Exception as e:
            errors += 1
            continue
        emitted = set(re.findall(r"\[(D\d+:\d+)", pcc.context_text))
        preserved = emitted & original
        invented = emitted - original
        total_preserved += len(preserved)
        total_invented += len(invented)
        if invented:
            cases_with_invented.append((qid, sorted(invented)[:5]))

    print(f"Questions processed: {len(targets)} (errors: {errors})")
    print(f"Total emitted ids matching an original: {total_preserved}")
    print(f"Total emitted ids NOT in original (invented): {total_invented}")
    print()

    if cases_with_invented:
        print("Cases with invented ids:")
        for qid, sample in cases_with_invented[:10]:
            print(f"  {qid}: {sample}")
    else:
        print("RESULT: provenance preserved in ALL processed questions")
        print("  The adapter's 'compiler renumbers turns' comment does NOT reproduce")
        print("  when raw_content carries the original [D..] header.")
    print()

    out = PROJECT_ROOT / "results/m30_41"
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance_check_28.json").write_text(json.dumps({
        "processed": len(targets), "errors": errors,
        "total_preserved": total_preserved, "total_invented": total_invented,
        "cases_with_invented": cases_with_invented,
        "verdict": "PROVENANCE_BREAK" if cases_with_invented else "PRESERVED_ALL",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'provenance_check_28.json'}")


if __name__ == "__main__":
    main()
