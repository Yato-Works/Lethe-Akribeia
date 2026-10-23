#!/usr/bin/env python3
"""M30-47 Phase 2: Why does only lc_conv-26_63 emit Entity Grounding tags?

The distribution (1/28) rules out a global noise-removal fix. So trace THIS
case: what makes adaptive_search emit 31 grounding tags here?

Read-only. Inspect the query, the cue, and the loop that appends tags.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

QID = "conv-26-qa-063"


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]

    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
    q = ctx = None
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        for cand in questions:
            if cand.question_id == QID:
                q = cand
                pcc = adapter.compiler.compile(cand.question, ir_records)
                ctx = pcc.context_text
                break
        if q:
            break

    print("=" * 78)
    print("M30-47 Phase 2: lc_conv-26_63 Entity Grounding trace")
    print("=" * 78)
    print(f"Question: {q.question}")
    print(f"Gold:     {q.ground_truth}")
    print()

    eg = re.findall(r"\[Entity Grounding:([^\]]*)\]", ctx)
    print(f"EG tags: {len(eg)}")
    from collections import Counter
    for tag, n in Counter(x.strip() for x in eg).most_common():
        print(f"  {n:3d}x  '{tag}'")
    print()

    # Where do they sit?
    lines = ctx.split("\n")
    eg_idx = [i for i, ln in enumerate(lines) if "[Entity Grounding:" in ln]
    print(f"Total lines: {len(lines)}")
    print(f"EG lines at indices: {eg_idx[:10]}{'...' if len(eg_idx)>10 else ''}")
    print()

    # The question's content words -> the "missing_cue" that drives the loop
    qk = [w for w in re.findall(r"[a-z]{4,}", q.question.lower())]
    print(f"Question content words (len>=4): {qk}")
    print()

    # Show the first non-EG line after the EG block
    print("--- context around the EG block (lines 25-35) ---")
    for i in range(24, min(36, len(lines))):
        print(f"  [{i:3d}] {lines[i][:110]}")
    print()

    out = PROJECT_ROOT / "results/m30_47"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eg_trace_26_63.json").write_text(json.dumps({
        "qid": QID, "question": q.question, "gold": q.ground_truth,
        "eg_count": len(eg),
        "eg_tags": dict(Counter(x.strip() for x in eg)),
        "eg_line_indices": eg_idx[:20],
        "total_lines": len(lines),
        "question_words": qk,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'eg_trace_26_63.json'}")


if __name__ == "__main__":
    main()
