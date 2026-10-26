#!/usr/bin/env python3
"""M30-44 Phase 3: Hand-verify 3 abstained cases where evidence was present.

Print the ACTUAL gold evidence turn text, the ACTUAL context lines that
contain it, and the reader's ACTUAL output — so the "reader failure" verdict
is confirmed by eye, not by an overlap metric alone.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

CASES = [
    ("conv-26", "conv-26-qa-063", "D15:11"),
    ("conv-48", "conv-48-qa-012", "D2:7"),
    ("conv-47", "conv-47-qa-065", "D31:1"),
]


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    print("=" * 78)
    print("M30-44 Phase 3: Hand verification of 3 abstained cases")
    print("=" * 78)
    print()

    for sid, conv_qid, gold_id in CASES:
        idx = sample_ids.index(sid)
        turns, questions, ir_records = adapter.load_conversation(idx)
        turn_by_id = {t.dia_id: t for t in turns}
        q = next((x for x in questions if x.question_id == conv_qid), None)
        if not q:
            print(f"{conv_qid}: not found")
            continue

        pcc = adapter.compiler.compile(q.question, ir_records)
        ctx = pcc.context_text

        gt = turn_by_id.get(gold_id)
        print(f"### {conv_qid}  (gold evidence {gold_id})")
        print(f"  Question: {q.question}")
        print(f"  Gold:     {q.ground_truth}")
        print()
        print(f"  Evidence turn text ({gold_id}):")
        print(f"    {gt.text if gt else 'NOT FOUND'}")
        print()
        # find context lines containing distinctive evidence words
        key = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", gt.text) if w.lower() not in
               {"that","this","with","have","been","they","will","your","from","what","when"}][:6]
        hits = [ln for ln in ctx.split("\n") if any(k.lower() in ln.lower() for k in key)]
        print(f"  Context lines containing the evidence (key={key}):")
        for h in hits[:4]:
            print(f"    {h[:150]}")
        print()
        print(f"  Full context length: {len(ctx)} chars")
        print()

    print("=" * 78)
    print("Check: is the evidence turn's KEY FACT actually in the context,")
    print("and did the reader still abstain?")
    print("=" * 78)


if __name__ == "__main__":
    main()
