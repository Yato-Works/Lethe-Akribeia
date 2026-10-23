#!/usr/bin/env python3
"""M30-46 Phase 3-0: Sanity check — does context_override reproduce 13/28?

Before running the order experiment, verify that injecting the SAME compiled
context via context_override yields the production 13/28. If not, the harness
is still wrong and no comparison is valid.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter, CachedContext
from artificial_memory.research.benchmarks.llm import OllamaAnswerer


def to_conv(q):
    m = re.match(r"lc_conv-(\d+)_(\d+)", q)
    return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}"


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")
    tmap = {to_conv(q): q for q in targets}

    answerer = OllamaAnswerer(model="qwen2.5:7b-instruct")

    # Build overrides with the EXACT compiled context
    a0 = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
    override = {}
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = a0.load_conversation(idx)
        for q in questions:
            if q.question_id in tmap:
                pcc = a0.compiler.compile(q.question, ir_records)
                override[q.question_id] = CachedContext(context_text=pcc.context_text)

    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
    adapter.context_override = override

    n_ok = 0
    total = 0
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            if q.question_id not in tmap:
                continue
            res = adapter.evaluate_question(q, turns, ir_records, answerer)
            total += 1
            if res.is_correct:
                n_ok += 1

    print("=" * 78)
    print("M30-46 Phase 3-0: context_override sanity check")
    print("=" * 78)
    print(f"  context_override baseline: {n_ok}/{total}")
    print(f"  production baseline (M30-44): 13/28")
    print()
    if n_ok == 13:
        print("MATCH -> the override harness reproduces production. Safe to compare.")
    else:
        print(f"MISMATCH ({n_ok} vs 13) -> harness still differs; do NOT compare.")

    out = PROJECT_ROOT / "results/m30_46"
    out.mkdir(parents=True, exist_ok=True)
    (out / "override_sanity.json").write_text(json.dumps({
        "override_baseline": n_ok, "total": total, "production_baseline": 13,
        "match": n_ok == 13,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'override_sanity.json'}")


if __name__ == "__main__":
    main()
